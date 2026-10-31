"""唯一训练入口：四组预设 + 两阶段骨架 + 一次性 test 评估（阶段 A）。

用法示例：
    python3 scripts/train_full.py --mode all
    python3 scripts/train_full.py --mode frozen --weights none \
        --out runs/diag/frozen-no-pretrain --tag no-pretrain
    python3 scripts/train_full.py --mode scratch --smoke --out /tmp/opencode/smoke-scratch

约定：
  * 四组共用同一份 train_epoch / evaluate / write_run_artifacts；
  * resolve() 负责全部派生字段，本文件不自行推导；
  * test 只在 best 权重上评估一次，由计数器断言。
"""

import argparse
import base64
import copy
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import warnings
import zlib
from datetime import datetime, timezone
from pathlib import Path

# 必须在 import torch 之前设置 CUBLAS workspace（确定性矩阵乘）。
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from schema import CANONICAL_MODES, CSV_HEADERS, METRICS_FIELDS, META_SHA256, NUM_CLASSES, SCHEMA_VERSION, SPLIT_FINGERPRINT, SPLIT_SEED, TEST_N, TRAIN_N, VAL_N, VAL_PER_CLASS, source_sha256
from presets import PRESETS, resolve
from data_split import build_split, split_fingerprint

# @@KERNEL_EMBED@@

ROOT = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
_INJECTED = globals().get("_KERNEL_SOURCE_SHA256")
source_sha = _INJECTED if _INJECTED else source_sha256(ROOT)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
ARTIFACTS = (
    "history.csv",
    "metrics.json",
    "per_class.csv",
    "confusion_matrix.csv",
    "misclassified.csv",
    "top_confusions.csv",
)
EVAL_RESIZE = 256
EVAL_CROP = 224


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_worker_init(base_seed):
    def worker_init(worker_id):
        seed = base_seed + worker_id
        random.seed(seed)
        torch.manual_seed(seed)

    return worker_init


class Flowers(Dataset):
    def __init__(self, img_dir, rows, tf):
        self.img_dir = Path(img_dir)
        self.rows = list(rows)
        self.tf = tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image_id, label = self.rows[index]
        image = Image.open(self.img_dir / image_id).convert("RGB")
        return self.tf(image), label - 1, image_id


_AUG_FACTORY = {
    "RandomResizedCrop(224,scale=(0.6,1.0))": lambda: transforms.RandomResizedCrop(
        224, scale=(0.6, 1.0)
    ),
    "RandomHorizontalFlip()": lambda: transforms.RandomHorizontalFlip(),
    "RandomResizedCrop(224,scale=(0.35,1.0))": lambda: transforms.RandomResizedCrop(
        224, scale=(0.35, 1.0)
    ),
    "RandomRotation(25)": lambda: transforms.RandomRotation(25),
    "ColorJitter(0.3,0.3,0.3,0.05)": lambda: transforms.ColorJitter(
        0.3, 0.3, 0.3, 0.05
    ),
}


def build_train_tf(cfg):
    tfms = []
    erase = False
    for spec in cfg["aug_ops"]:
        if spec.startswith("RandomErasing"):
            erase = True
            continue
        factory = _AUG_FACTORY.get(spec)
        if factory is None:
            raise RuntimeError(f"未知增强算子: {spec}")
        tfms.append(factory())
    tfms.append(transforms.ToTensor())
    tfms.append(transforms.Normalize(MEAN, STD))
    if erase:
        tfms.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)))
    return transforms.Compose(tfms)


def build_eval_tf():
    return transforms.Compose(
        [
            transforms.Resize(EVAL_RESIZE),
            transforms.CenterCrop(EVAL_CROP),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def build_model(weights_init):
    if weights_init == "imagenet1k_v1":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif weights_init == "none":
        model = models.resnet18(weights=None)
    else:
        raise RuntimeError(f"未知 weights_init: {weights_init}")
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def set_backbone_trainable(model, trainable):
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            param.requires_grad = True
        else:
            param.requires_grad = bool(trainable)


def backbone_bns(model):
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.BatchNorm2d) and not name.startswith("fc")
    ]


def set_backbone_bn(model, bn_phase):
    flag = bn_phase != "frozen"
    for _, module in backbone_bns(model):
        module.train(flag)


def bn_counts(model):
    counts = {}
    for name, module in backbone_bns(model):
        tracked = module.num_batches_tracked
        counts[name] = int(tracked.item()) if tracked is not None else 0
    return counts


def assert_bn_behavior(phase, before, after):
    if not before:
        return
    deltas = {name: after[name] - before[name] for name in before}
    if phase == "frozen":
        changed = [name for name, delta in deltas.items() if delta != 0]
        if changed:
            raise RuntimeError(f"phase=frozen 时骨干 BN 统计发生变化: {changed[:3]}")
    else:
        if not any(delta > 0 for delta in deltas.values()):
            raise RuntimeError(f"phase={phase} 时没有任何骨干 BN 更新")


def make_optimizer(model, lr_head, lr_backbone, weight_decay):
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(param)
        else:
            backbone_params.append(param)

    groups = []
    for group_name, params, lr in (
        ("head", head_params, lr_head),
        ("backbone", backbone_params, lr_backbone),
    ):
        if not params:
            continue
        if group_name == "backbone" and lr == 0.0:
            continue
        decay = [param for param in params if param.ndim >= 2]
        no_decay = [param for param in params if param.ndim < 2]
        if decay:
            groups.append(
                {
                    "params": decay,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "group_name": group_name,
                }
            )
        if no_decay:
            groups.append(
                {
                    "params": no_decay,
                    "lr": lr,
                    "weight_decay": 0.0,
                    "group_name": group_name,
                }
            )
    if not groups:
        raise RuntimeError("没有可训练参数，无法构建优化器")
    return torch.optim.AdamW(groups)


def current_lrs(optimizer):
    lr_head = 0.0
    lr_backbone = 0.0
    for group in optimizer.param_groups:
        if group.get("group_name") == "head":
            lr_head = float(group["lr"])
        elif group.get("group_name") == "backbone":
            lr_backbone = float(group["lr"])
    return lr_head, lr_backbone


def phase_for(cfg, epoch):
    if cfg["stage1_epochs"] > 0:
        return "unfrozen" if epoch >= cfg["unfreeze_epoch"] else "frozen"
    return "scratch" if cfg["lr_backbone"] > 0 else "frozen"


def train_epoch(model, loader, criterion, optimizer, device, limit_batches=None):
    """唯一的训练循环实现（四组共用）。调用方负责 model.train()/BN 模式。"""
    total = 0
    correct = 0
    loss_sum = 0.0
    for index, (images, labels, _names) in enumerate(loader):
        if limit_batches is not None and index >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch = labels.numel()
        loss_sum += float(loss.detach()) * batch
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += batch
    if total == 0:
        return 0.0, 0.0
    return loss_sum / total, correct / total


def _f1_from_cm(cm):
    size = len(cm)
    f1s = []
    supports = []
    for index in range(size):
        tp = cm[index][index]
        row = sum(cm[index])
        col = sum(cm[row_index][index] for row_index in range(size))
        precision = tp / col if col else 0.0
        recall = tp / row if row else 0.0
        f1s.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        supports.append(row)
    support_total = sum(supports)
    macro = sum(f1s) / size if size else 0.0
    weighted = (
        sum(f1 * support for f1, support in zip(f1s, supports)) / support_total
        if support_total
        else 0.0
    )
    return macro, weighted


@torch.no_grad()
def evaluate(model, loader, device, criterion, num_classes, collect=False, limit_batches=None):
    """唯一的评估实现；val/test 共用同一口径（label_smoothing=0 的 CE）。"""
    model.eval()
    total = 0
    loss_sum = 0.0
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    truep_sum = torch.zeros(num_classes, dtype=torch.float64)
    records = []

    for index, (images, labels, names) in enumerate(loader):
        if limit_batches is not None and index >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, dim=1)
        top = probs.topk(5, dim=1).indices
        preds = top[:, 0]
        batch = labels.numel()

        loss_sum += float(loss) * batch
        total += batch
        correct_top1 += int((preds == labels).sum().item())
        correct_top3 += int((top[:, :3] == labels.unsqueeze(1)).any(dim=1).sum().item())
        correct_top5 += int((top[:, :5] == labels.unsqueeze(1)).any(dim=1).sum().item())

        flat_index = labels * num_classes + preds
        cm.view(-1).index_add_(
            0, flat_index.cpu(), torch.ones(batch, dtype=torch.long)
        )
        true_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        truep_sum += true_probs.double().cpu()
        if collect:
            pred_probs = probs.gather(1, preds.unsqueeze(1)).squeeze(1)
            for row in range(batch):
                records.append(
                    (
                        names[row],
                        int(labels[row]),
                        int(preds[row]),
                        float(true_probs[row]),
                        float(pred_probs[row]),
                    )
                )

    if total == 0:
        raise RuntimeError("评估集为空")

    cm_list = cm.tolist()
    row_sums = [sum(row) for row in cm_list]
    per_class_acc = [
        (float(cm_list[i][i]) / float(row_sums[i])) if row_sums[i] else 0.0
        for i in range(num_classes)
    ]
    mean_true_prob = [
        float(truep_sum[i]) / float(row_sums[i]) if row_sums[i] else 0.0
        for i in range(num_classes)
    ]
    macro, weighted = _f1_from_cm(cm_list)
    return {
        "loss": loss_sum / total,
        "top1": correct_top1 / total,
        "top3": correct_top3 / total,
        "top5": correct_top5 / total,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "cm": cm_list,
        "per_class_acc": per_class_acc,
        "mean_true_prob": mean_true_prob,
        "records": records,
    }


def _iter_metric_leaves(obj, prefix=""):
    for key, value in obj.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _iter_metric_leaves(value, path + ".")
        else:
            yield path, value


def assert_metrics_shape(metrics):
    paths = {path for path, _ in _iter_metric_leaves(metrics)}
    expected = set(METRICS_FIELDS)
    if paths != expected:
        raise RuntimeError(
            "metrics 字段集与 schema 不一致: "
            f"缺 {sorted(expected - paths)} 多 {sorted(paths - expected)}"
        )


def write_run_artifacts(out, history, metrics, test_out):
    """唯一的产物写出函数（四组共用）。"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADERS["history"])
        for row in history:
            writer.writerow(
                [
                    int(row["epoch"]),
                    f"{row['train_loss']:.6f}",
                    f"{row['train_acc']:.6f}",
                    f"{row['val_loss']:.6f}",
                    f"{row['val_acc']:.6f}",
                    row["phase"],
                    f"{row['lr_head']:.6f}",
                    f"{row['lr_backbone']:.6f}",
                    f"{row['sec']:.6f}",
                ]
            )

    cm = test_out["cm"]
    row_sums = [sum(row) for row in cm]
    with (out / "per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADERS["per_class"])
        for index in range(NUM_CLASSES):
            writer.writerow(
                [
                    index,
                    index + 1,
                    row_sums[index],
                    cm[index][index],
                    f"{test_out['per_class_acc'][index]:.6f}",
                    f"{test_out['mean_true_prob'][index]:.6f}",
                ]
            )

    with (out / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADERS["confusion_matrix"])
        for true_index in range(NUM_CLASSES):
            writer.writerow([true_index] + list(cm[true_index]))

    misclassified = [
        record for record in test_out["records"] if record[1] != record[2]
    ]
    misclassified.sort(key=lambda record: (round(record[3], 6), record[0]))
    with (out / "misclassified.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADERS["misclassified"])
        for image_id, true_idx, pred_idx, true_prob, pred_prob in misclassified:
            writer.writerow(
                [
                    image_id,
                    true_idx,
                    pred_idx,
                    f"{true_prob:.6f}",
                    f"{pred_prob:.6f}",
                ]
            )

    confusions = []
    for true_index in range(NUM_CLASSES):
        for pred_index in range(NUM_CLASSES):
            if true_index != pred_index and cm[true_index][pred_index] > 0:
                confusions.append(
                    (
                        true_index,
                        pred_index,
                        cm[true_index][pred_index],
                        row_sums[true_index],
                    )
                )
    confusions.sort(key=lambda item: (-item[2], item[0], item[1]))
    with (out / "top_confusions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADERS["top_confusions"])
        for row in confusions[:20]:
            writer.writerow(list(row))

    with (out / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def collect_env():
    if torch.cuda.is_available():
        device = "cuda"
        cuda_version = torch.version.cuda or ""
        gpu_name = torch.cuda.get_device_name(0)
    else:
        device = "cpu"
        cuda_version = ""
        gpu_name = ""
    return {
        "device": device,
        "cuda": cuda_version,
        "gpu": gpu_name,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
    }


def collect_provenance():
    injected = globals().get("_KERNEL_PROVENANCE")
    if injected:
        return {
            "git_commit": injected.get("git_commit"),
            "git_dirty": injected.get("git_dirty"),
            "provenance": "embedded",
        }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        commit_text = commit.stdout.strip()
        return {
            "git_commit": commit_text or None,
            "git_dirty": bool(status.stdout.strip()),
            "provenance": "git",
        }
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None, "provenance": "git"}


def ensure_meta():
    base = _HERE if _INJECTED else ROOT
    meta = base / "data" / "meta.csv"
    if not meta.exists():
        payload = globals().get("_KERNEL_META_B85")
        if not payload:
            raise SystemExit(f"缺少数据文件: {meta}")
        raw = zlib.decompress(base64.b85decode(payload.encode("ascii")))
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_bytes(raw)
    digest = hashlib.sha256(meta.read_bytes()).hexdigest()
    if digest != META_SHA256:
        raise SystemExit(f"meta.csv sha256 不符: {digest} != {META_SHA256}")
    return meta


def resolve_images_dir(data_root):
    if data_root:
        root = Path(data_root)
    elif _INJECTED:
        root = Path("/kaggle/input/oxford-102-flowers")
    else:
        return ROOT / "data" / "raw" / "jpg"
    candidates = [
        root,
        root / "jpg",
        root / "data" / "raw" / "jpg",
        root / "data" / "raw",
    ]
    for candidate in candidates:
        if (candidate / "image_00001.jpg").is_file():
            return candidate
    for found in root.rglob("image_00001.jpg"):
        return found.parent
    raise SystemExit(f"找不到 image_00001.jpg（数据根 {root}）")


def load_split(meta):
    split = build_split(meta)
    fingerprint = split_fingerprint(split)
    if fingerprint != SPLIT_FINGERPRINT:
        raise SystemExit(
            f"划分指纹不符: {fingerprint} != {SPLIT_FINGERPRINT}（任何训练开始前退出）"
        )
    counts = (len(split["train"]), len(split["val"]), len(split["test"]))
    if counts != (TRAIN_N, VAL_N, TEST_N):
        raise SystemExit(f"划分数量不符: {counts} != {(TRAIN_N, VAL_N, TEST_N)}")
    return split


def run_one(cfg, args, img_dir, split, env):
    set_seed(args.seed)
    configure_determinism()
    started_at = now_iso()
    started = time.time()
    device = torch.device(env["device"])

    train_rows = list(split["train"])
    val_rows = list(split["val"])
    test_rows = list(split["test"])
    if cfg["smoke"]:
        val_rows = val_rows[:64]
        test_rows = test_rows[:64]

    train_tf = build_train_tf(cfg)
    eval_tf = build_eval_tf()
    train_dataset = Flowers(img_dir, train_rows, train_tf)
    val_dataset = Flowers(img_dir, val_rows, eval_tf)
    test_dataset = Flowers(img_dir, test_rows, eval_tf)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    pin_memory = env["device"] == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch"],
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        worker_init_fn=make_worker_init(args.seed),
        generator=generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch"],
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch"],
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    model = build_model(cfg["weights_init"]).to(device)
    if cfg["stage1_epochs"] > 0:
        set_backbone_trainable(model, False)
    else:
        set_backbone_trainable(
            model, cfg["params_trainable_stage1_rule"] == "all"
        )

    optimizer = make_optimizer(
        model, cfg["lr_head"], cfg["lr_backbone"], cfg["weight_decay"]
    )
    scheduler = None
    if cfg["stage1_epochs"] == 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs_planned"]
        )

    train_criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    eval_criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    limit_batches = 3 if cfg["smoke"] else None
    epochs_run = 1 if cfg["smoke"] else cfg["epochs_planned"]

    history = []
    best_state = None
    best_epoch = 0
    best_val_top1 = -1.0
    best_val = None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for epoch in range(1, epochs_run + 1):
            if (
                cfg["unfreeze_epoch"] is not None
                and epoch == cfg["unfreeze_epoch"]
            ):
                set_backbone_trainable(model, True)
                optimizer = make_optimizer(
                    model, cfg["lr_head"], cfg["lr_backbone"], cfg["weight_decay"]
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cfg["stage2_epochs"]
                )

            phase = phase_for(cfg, epoch)
            if (
                cfg["unfreeze_epoch"] is not None
                and epoch >= cfg["unfreeze_epoch"]
            ):
                bn_phase = cfg["bn_stage2"]
            else:
                bn_phase = cfg["bn_stage1"]

            model.train()
            set_backbone_bn(model, bn_phase)
            lr_head_now, lr_backbone_now = current_lrs(optimizer)
            bn_before = bn_counts(model)

            epoch_started = time.time()
            train_loss, train_acc = train_epoch(
                model,
                train_loader,
                train_criterion,
                optimizer,
                device,
                limit_batches=limit_batches,
            )
            epoch_sec = time.time() - epoch_started

            assert_bn_behavior(phase, bn_before, bn_counts(model))
            val_out = evaluate(
                model, val_loader, device, eval_criterion, NUM_CLASSES
            )
            if scheduler is not None:
                scheduler.step()

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_out["loss"],
                    "val_acc": val_out["top1"],
                    "phase": phase,
                    "lr_head": lr_head_now,
                    "lr_backbone": lr_backbone_now,
                    "sec": epoch_sec,
                }
            )

            if val_out["top1"] > best_val_top1:
                best_val_top1 = val_out["top1"]
                best_epoch = epoch
                best_val = {
                    key: val_out[key]
                    for key in ("loss", "top1", "top3", "top5", "macro_f1")
                }
                best_state = copy.deepcopy(model.state_dict())

    if best_state is None or best_val is None:
        raise RuntimeError("训练未产生可用的 best 权重")

    det_warnings = sorted(
        {
            str(item.message)
            for item in caught
            if "nondeterministic" in str(item.message).lower()
        }
    )

    model.load_state_dict(copy.deepcopy(best_state))

    test_eval_counter = 0
    test_eval_counter += 1
    test_out = evaluate(
        model, test_loader, device, eval_criterion, NUM_CLASSES, collect=True
    )
    if test_eval_counter != 1:
        raise RuntimeError(f"test 评估次数必须恰为 1，实际 {test_eval_counter}")

    params_total = sum(param.numel() for param in model.parameters())
    if cfg["params_trainable_stage1_rule"] == "all":
        params_stage1 = params_total
    else:
        params_stage1 = sum(
            param.numel()
            for name, param in model.named_parameters()
            if name.startswith("fc.")
        )
    if cfg["unfreeze_epoch"] is not None:
        params_peak = params_total
    else:
        params_peak = params_stage1

    provenance = collect_provenance()
    elapsed = time.time() - started

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "mode": cfg["mode"],
            "tag": cfg["tag"],
            "is_diag": cfg["is_diag"],
            "base": cfg["base"],
            "seed": args.seed,
            "split_seed": SPLIT_SEED,
            "val_per_class": VAL_PER_CLASS,
            "weights_init": cfg["weights_init"],
            "epochs_planned": cfg["epochs_planned"],
            "epochs_run": len(history),
            "batch": cfg["batch"],
            "optimizer": "adamw",
            "weight_decay": cfg["weight_decay"],
            "wd_no_decay": "bn_and_bias",
            "scheduler": "cosine",
            "lr_head": cfg["lr_head"],
            "lr_backbone": cfg["lr_backbone"],
            "label_smoothing": cfg["label_smoothing"],
            "aug": cfg["aug"],
            "aug_ops": list(cfg["aug_ops"]),
            "bn_stage1": cfg["bn_stage1"],
            "bn_stage2": cfg["bn_stage2"],
            "bn_mode": cfg["bn_mode"],
            "unfreeze_epoch": cfg["unfreeze_epoch"],
            "params_total": params_total,
            "params_trainable_stage1": params_stage1,
            "params_trainable_peak": params_peak,
            "train_n": len(train_rows),
            "val_n": len(val_rows),
            "test_n": len(test_rows),
            "num_classes": NUM_CLASSES,
            "split_fingerprint": SPLIT_FINGERPRINT,
            "meta_sha256": META_SHA256,
            "source_sha256": source_sha,
            "git_commit": provenance["git_commit"],
            "git_dirty": provenance["git_dirty"],
            "provenance": provenance["provenance"],
            "python": env["python"],
            "torch": env["torch"],
            "torchvision": env["torchvision"],
            "cuda": env["cuda"],
            "gpu": env["gpu"],
            "device": env["device"],
            "reproducibility": "best_effort" if det_warnings else "strict",
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "deterministic_algorithms": "warn_only",
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "determinism_warnings": det_warnings,
            "smoke": cfg["smoke"],
            "started_at": started_at,
            "elapsed_sec": elapsed,
            "test_eval_count": test_eval_counter,
        },
        "selection": {
            "criterion": "val_top1",
            "best_epoch": best_epoch,
            "best_val_top1": best_val["top1"],
        },
        "val": {
            "loss": best_val["loss"],
            "top1": best_val["top1"],
            "top3": best_val["top3"],
            "top5": best_val["top5"],
            "macro_f1": best_val["macro_f1"],
        },
        "test": {
            "loss": test_out["loss"],
            "top1": test_out["top1"],
            "top3": test_out["top3"],
            "top5": test_out["top5"],
            "macro_f1": test_out["macro_f1"],
            "weighted_f1": test_out["weighted_f1"],
            "checkpoint_epoch": best_epoch,
            "per_class_acc": list(test_out["per_class_acc"]),
        },
    }
    assert_metrics_shape(metrics)
    write_run_artifacts(cfg["out"], history, metrics, test_out)
    if args.save_pt:
        torch.save(best_state, Path(cfg["out"]) / "best.pt")
    return metrics


def _write_summary(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_all(modes, args, img_dir, split, env):
    """kernel run_all：逐 mode 更新 summary.json，failed 即非零退出。"""
    summary_path = Path("runs") / "summary.json"
    selected = list(modes)
    statuses = {mode: "skipped" for mode in CANONICAL_MODES}
    elapsed_by_mode = {mode: 0.0 for mode in CANONICAL_MODES}
    errors = {mode: None for mode in CANONICAL_MODES}
    started_at = now_iso()

    def write_summary():
        payload = {
            "kernel_ref": globals().get("_KERNEL_REF", "feisama/flower-102-transfer"),
            "built_source_sha256": source_sha,
            "started_at": started_at,
            "finished_at": now_iso(),
            "gpu": env["gpu"],
            "modes": [
                {
                    "mode": mode,
                    "status": statuses[mode],
                    "elapsed_sec": float(elapsed_by_mode[mode]),
                    "error": errors[mode],
                }
                for mode in CANONICAL_MODES
            ],
        }
        _write_summary(summary_path, payload)

    write_summary()
    for mode in selected:
        statuses[mode] = "running"
        write_summary()
        started = time.time()
        try:
            cfg = resolve(
                mode,
                out=args.out,
                weights=args.weights,
                tag=args.tag,
                smoke=args.smoke,
            )
            out_dir = Path(cfg["out"])
            if all((out_dir / name).is_file() for name in ARTIFACTS):
                statuses[mode] = "skipped"
            else:
                run_one(cfg, args, img_dir, split, env)
                statuses[mode] = "ok"
        except Exception as exc:  # noqa: BLE001 - 记录后继续其它 mode
            statuses[mode] = "failed"
            errors[mode] = str(exc)
            print(f"MODE_FAILED {mode}: {exc}", file=sys.stderr)
        elapsed_by_mode[mode] = time.time() - started
        write_summary()

    failed = any(statuses[mode] == "failed" for mode in selected)
    succeeded = any(statuses[mode] == "ok" for mode in selected)
    if failed or not succeeded:
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="flower-102 四组训练入口")
    parser.add_argument("--mode", default="all")
    parser.add_argument("--out")
    parser.add_argument("--weights", choices=("imagenet1k_v1", "none"))
    parser.add_argument("--tag")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-pt", action="store_true")
    parser.add_argument("--data-root")
    args = parser.parse_args(argv)

    if _INJECTED:
        os.chdir(_HERE)

    meta = ensure_meta()
    split = load_split(meta)
    img_dir = resolve_images_dir(args.data_root)

    kernel_modes = globals().get("_KERNEL_MODES")
    if kernel_modes:
        modes = list(kernel_modes)
        args.smoke = bool(globals().get("_KERNEL_SMOKE"))
        is_kernel = True
    else:
        is_kernel = False
        if args.mode == "all":
            modes = list(CANONICAL_MODES)
        else:
            modes = [args.mode]

    set_seed(args.seed)
    configure_determinism()
    env = collect_env()

    if is_kernel:
        sys.exit(run_all(modes, args, img_dir, split, env))

    for mode in modes:
        cfg = resolve(
            mode,
            out=args.out,
            weights=args.weights,
            tag=args.tag,
            smoke=args.smoke,
        )
        run_one(cfg, args, img_dir, split, env)
    sys.exit(0)


if __name__ == "__main__":
    main()
