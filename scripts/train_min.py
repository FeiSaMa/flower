#!/usr/bin/env python3
"""最小可跑版：冻结骨干，只训分类头。

目的不是刷分，是让整条链路（数据 → 前向 → 损失 → 反向 → 评估）先跑通。
跑通后再解锁骨干微调、数据增强、BN 策略。

用法（Kaggle）:
    python train_min.py --data-root /kaggle/input/oxford-102-flowers

用法（本机，需先装 torch）:
    python train_min.py --data-root data/raw
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

MEAN = (0.485, 0.456, 0.406)  # ImageNet 统计量，因为骨干是 ImageNet 预训练
STD = (0.229, 0.224, 0.225)


class Flowers(Dataset):
    """从 meta.csv 读 (文件名, 标签)。不用 torchvision.datasets.Flowers102 是因为
    它在找不到数据时会自己去官网下载，Kaggle 上没网会卡很久。"""

    def __init__(self, root: Path, rows: list[tuple[str, int]], tf):
        self.root, self.rows, self.tf = root, rows, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        name, label = self.rows[i]
        img = Image.open(self.root / name).convert("RGB")  # 统一成 3 通道，防灰度图
        return self.tf(img), label - 1  # 官方标签 1..102 → 训练用 0..101


def build_rows(meta_csv: Path, val_per_class: int = 3, seed: int = 0):
    """train+val 合并后分层抽验证集。

    官方 train 和 val 各 10 张/类，合并成 20 张/类；抽 3 张做验证，
    剩 17 张训练。test 完全不动。
    """
    by_class: dict[int, list[str]] = {}
    with meta_csv.open() as f:
        for r in csv.DictReader(f):
            if r["split"] in ("train", "val"):
                by_class.setdefault(int(r["label"]), []).append(r["image_id"])

    rng = random.Random(seed)
    train_rows, val_rows = [], []
    for label, names in by_class.items():
        rng.shuffle(names)
        val_rows += [(n, label) for n in names[:val_per_class]]
        train_rows += [(n, label) for n in names[val_per_class:]]
    return train_rows, val_rows


def make_loaders(root: Path, meta: Path, batch: int, workers: int):
    train_rows, val_rows = build_rows(meta)
    # 增强只在训练集上做；验证集只缩放+中心裁剪，保证评估可复现
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_dl = DataLoader(Flowers(root, train_rows, train_tf), batch_size=batch,
                          shuffle=True, num_workers=workers, pin_memory=True)
    val_dl = DataLoader(Flowers(root, val_rows, val_tf), batch_size=batch,
                        shuffle=False, num_workers=workers, pin_memory=True)
    print(f"训练 {len(train_rows)} 张 / 验证 {len(val_rows)} 张 / 每类验证 {3}")
    return train_dl, val_dl


def build_model(freeze: bool, device):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 102)  # 新的分类头，唯一随机初始化的部分
    if freeze:
        for p in m.parameters():
            p.requires_grad = False
        m.fc.requires_grad_(True)
    return m.to(device)


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()  # 切 eval：关闭 Dropout、BN 用 running stats
    correct = total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss(reduction="sum")
    for x, y in dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        z = model(x)
        loss_sum += crit(z, y).item()
        correct += (z.argmax(1) == y).sum().item()
        total += y.numel()
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("runs/min"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    train_dl, val_dl = make_loaders(args.data_root, Path("data/meta.csv"),
                                    args.batch, args.workers)
    model = build_model(freeze=True, device=device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    args.out.mkdir(parents=True, exist_ok=True)
    hist = []
    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()  # 切 train：BN 用 batch 统计量（骨干冻结时影响有限）
        run_loss = seen = correct = 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()          # 梯度是累加的，必须清零
            z = model(x)
            loss = crit(z, y)
            loss.backward()          # 反向传播：autograd 自动算每个参数的梯度
            opt.step()               # 沿负梯度方向挪一步
            run_loss += loss.item() * y.numel()
            correct += (z.argmax(1) == y).sum().item()
            seen += y.numel()

        tr_loss, tr_acc = run_loss / seen, correct / seen
        va_loss, va_acc = evaluate(model, val_dl, device)
        sched.step()
        hist.append(dict(epoch=ep, train_loss=tr_loss, train_acc=tr_acc,
                         val_loss=va_loss, val_acc=va_acc))
        print(f"ep{ep:2d}  train {tr_loss:.4f}/{tr_acc:.3f}   val {va_loss:.4f}/{va_acc:.3f}")
        if va_acc > best:
            best = va_acc
            torch.save(model.state_dict(), args.out / "best.pt")

    with (args.out / "history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0]))
        w.writeheader()
        w.writerows(hist)
    print(f"最佳验证准确率 {best:.3f}，曲线已写入 {args.out/'history.csv'}")


if __name__ == "__main__":
    main()
