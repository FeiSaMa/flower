"""本机 CPU 分批训练演示：frozen ResNet-18 只训 fc，跑 2 个 epoch。

输出：
  demo/batch_log.csv
  demo/pred_ep{0,1,2}.json
  demo/sample_ids.json
  demo/shapes.json
  report/figures/{batch_flow,batch_loss,pred_evolution}.png

运行：
  uv run --no-project --python 3.13 --with torch --with torchvision \
    --with pillow --with matplotlib python3 scripts/demo_batches.py
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_split import build_split  # noqa: E402

FONT = "Source Han Sans CN"
NAVY = "#1F3864"
ORANGE = "#C55A11"
GRAY = "#595959"
LIGHT = "#F2F2F2"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 64
DEFAULT_EPOCHS = 2
LR = 1e-3
DEMO = ROOT / "demo"
FIGURES = ROOT / "report" / "figures"


class Flowers(Dataset):
    def __init__(self, image_dir, rows, tf):
        self.image_dir = Path(image_dir)
        self.rows = list(rows)
        self.tf = tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image_id, label = self.rows[index]
        image = Image.open(self.image_dir / image_id).convert("RGB")
        return self.tf(image), label - 1


def make_transforms():
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_tf, eval_tf


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 102)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc.requires_grad_(True)
    return model


def set_font():
    plt.rcParams["font.sans-serif"] = [FONT, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def top5_records(model, loader, device, sample_ids):
    model.eval()
    records = {}
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probabilities = torch.softmax(model(images), dim=1)
            values, indices = torch.topk(probabilities, 5, dim=1)
            for row in range(images.shape[0]):
                true_idx = int(labels[row])
                records[sample_ids[row]] = {
                    "true_idx": true_idx,
                    "true_label": true_idx + 1,
                    "top5": [
                        {
                            "class_idx": int(indices[row, position]),
                            "prob": float(values[row, position]),
                        }
                        for position in range(5)
                    ],
                }
    return records


def load_class_names():
    path = ROOT / "data" / "class_names.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {int(row["class_idx"]): row["name"] for row in csv.DictReader(handle)}


def draw_box(axis, x, y, width, height, title, subtitle="", color=NAVY):
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor="#DCE6F1",
            edgecolor=color,
            linewidth=1.6,
        )
    )
    axis.text(
        x + width / 2,
        y + height / 2 + (0.16 if subtitle else 0),
        title,
        ha="center",
        va="center",
        fontsize=11,
        color=color,
        weight="bold",
    )
    if subtitle:
        axis.text(
            x + width / 2,
            y + height / 2 - 0.2,
            subtitle,
            ha="center",
            va="center",
            fontsize=9,
            color=GRAY,
        )


def plot_batch_flow(shapes, output):
    set_font()
    fig, axis = plt.subplots(figsize=(16, 5.4))
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 6)
    axis.axis("off")
    stages = [
        ("输入 batch", f"({shapes['input'][0]},3,224,224)"),
        ("ResNet18\n骨干", f"({shapes['input'][0]},512)"),
        ("fc", "512 → 102"),
        ("分数", f"({shapes['input'][0]},102)"),
        ("softmax", f"({shapes['input'][0]},102)"),
        ("交叉熵", "标量 loss"),
        ("梯度", "∂L/∂θ"),
        ("参数更新", "AdamW"),
    ]
    width = 1.55
    gap = 0.34
    start = 0.35
    for index, (title, subtitle) in enumerate(stages):
        x = start + index * (width + gap)
        draw_box(axis, x, 2.2, width, 1.55, title, subtitle)
        if index < len(stages) - 1:
            axis.add_patch(
                FancyArrowPatch(
                    (x + width + 0.04, 2.98),
                    (x + width + gap - 0.04, 2.98),
                    arrowstyle="->",
                    color=ORANGE,
                    lw=2,
                )
            )
    axis.text(
        8,
        0.95,
        "28 个 batch = 1 epoch；每个 batch 前向 → 反向 → 更新一次",
        ha="center",
        fontsize=15,
        color=ORANGE,
    )
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_batch_loss(rows, output):
    set_font()
    fig, axis = plt.subplots(figsize=(10, 5.4))
    colors = {1: NAVY, 2: ORANGE}
    for epoch in (1, 2):
        selected = [row for row in rows if row["epoch"] == epoch]
        x = [row["batch_idx"] for row in selected]
        y = [row["loss"] for row in selected]
        axis.plot(x, y, marker="o", markersize=4, linewidth=1.8, color=colors[epoch], label=f"epoch {epoch}")
        for index, (batch, loss) in enumerate(zip(x, y), start=1):
            if batch == 1 or batch % 5 == 0 or batch == len(x):
                axis.annotate(
                    f"{loss:.3f}",
                    (batch, loss),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=7,
                    color=colors[epoch],
                )
    axis.set_xlabel("batch 序号")
    axis.set_ylabel("batch loss")
    axis.set_title("一个 batch 一次更新：loss 随步数下降")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_pred_evolution(sample_ids, predictions, output):
    set_font()
    names = load_class_names()
    fig, axes = plt.subplots(3, 16, figsize=(18, 5.8))
    for row_index, epoch in enumerate((0, 1, 2)):
        for column, image_id in enumerate(sample_ids):
            axis = axes[row_index, column]
            image = Image.open(ROOT / "data" / "raw" / "jpg" / image_id).convert("RGB")
            axis.imshow(image)
            axis.axis("off")
            top = predictions[epoch][image_id]["top5"][0]
            name = names.get(top["class_idx"], f"class {top['class_idx']}")
            if len(name) > 13:
                name = name[:12] + "…"
            axis.text(
                0.5,
                -0.08,
                f"{name}\n{top['prob'] * 100:.1f}%",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=6,
                color=NAVY,
            )
        axes[row_index, 0].text(
            -0.3,
            0.5,
            f"ep{epoch}",
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=14,
            color=NAVY,
            weight="bold",
        )
    fig.suptitle("训练前 vs 训练后：同一批图的预测变化", fontsize=17, color=NAVY)
    fig.subplots_adjust(left=0.045, right=0.995, top=0.9, bottom=0.12, wspace=0.08, hspace=0.5)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _short_name(names, class_idx, limit=14):
    name = names.get(class_idx, f"class {class_idx}")
    return name if len(name) <= limit else name[: limit - 1] + "…"


def plot_pred_evolution5(sample_ids, predictions, output, epochs):
    set_font()
    names = load_class_names()
    selected = [epoch for epoch in (0, 1, 2, 3, 5) if epoch in epochs]
    fig, axes = plt.subplots(len(selected), 16, figsize=(18, 8.6), squeeze=False)
    for row_index, epoch in enumerate(selected):
        for column, image_id in enumerate(sample_ids):
            axis = axes[row_index, column]
            image = Image.open(ROOT / "data" / "raw" / "jpg" / image_id).convert("RGB")
            axis.imshow(image)
            axis.axis("off")
            entry = predictions[epoch][image_id]
            top = entry["top5"][0]
            correct = top["class_idx"] == entry["true_idx"]
            pred_name = _short_name(names, top["class_idx"], 13)
            true_name = _short_name(names, entry["true_idx"], 13)
            axis.text(
                0.5,
                -0.08,
                f"{pred_name} {top['prob'] * 100:.1f}%",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=5.5,
                color="#2E7D32" if correct else "#C62828",
                weight="bold",
            )
            axis.text(
                0.5,
                -0.2,
                f"真实 {true_name}",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=5,
                color=GRAY,
            )
        axes[row_index, 0].text(
            -0.25,
            0.5,
            f"ep{epoch}",
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=13,
            color=NAVY,
            weight="bold",
        )
    fig.suptitle("同一批图：预测如何从随机变成认对", fontsize=17, color=NAVY)
    fig.subplots_adjust(
        left=0.06, right=0.995, top=0.91, bottom=0.07, wspace=0.06, hspace=0.55
    )
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_sample_acc_curve(sample_ids, predictions, output):
    set_font()
    epochs = sorted(predictions)
    accuracies = []
    for epoch in epochs:
        hits = sum(
            1
            for image_id in sample_ids
            if predictions[epoch][image_id]["top5"][0]["class_idx"]
            == predictions[epoch][image_id]["true_idx"]
        )
        accuracies.append(hits / len(sample_ids))
    fig, axis = plt.subplots(figsize=(9, 5.2))
    axis.plot(epochs, accuracies, marker="o", color=NAVY, linewidth=2)
    for epoch, accuracy in zip(epochs, accuracies):
        axis.annotate(
            f"{accuracy * 100:.1f}%",
            (epoch, accuracy),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            color=NAVY,
        )
    axis.axhline(
        1 / 102,
        color=ORANGE,
        linestyle="--",
        linewidth=1.5,
        label="随机基线 1/102 ≈ 0.98%",
    )
    axis.set_xticks(epochs)
    axis.set_xlabel("epoch")
    axis.set_ylabel("16 张固定测试图的 top-1 准确率")
    axis.set_ylim(0, 1.02)
    axis.set_title("固定 16 张测试图：准确率随 epoch 变化")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="CPU 分批训练演示")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()
    epochs = args.epochs
    if epochs < 1:
        raise SystemExit("--epochs 必须 >= 1")
    random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    device = torch.device("cpu")
    DEMO.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    split = build_split(ROOT / "data" / "meta.csv")
    train_rows = split["train"]
    test_rows = split["test"]
    # 固定 16 张 test 图，但按 meta 行序取 16 个不同类别，
    # 否则前 16 张可能全属同一类，图中看不出类别多样性。
    sample_rows = []
    seen_labels = set()
    for image_id, label in test_rows:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        sample_rows.append((image_id, label))
        if len(sample_rows) == 16:
            break
    sample_ids = [image_id for image_id, _label in sample_rows]
    assert len(train_rows) == 1734, len(train_rows)
    assert len(sample_ids) == 16, len(sample_ids)

    train_tf, eval_tf = make_transforms()
    train_dataset = Flowers(ROOT / "data" / "raw" / "jpg", train_rows, train_tf)
    eval_dataset = Flowers(ROOT / "data" / "raw" / "jpg", sample_rows, eval_tf)
    generator = torch.Generator()
    generator.manual_seed(0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = build_model().to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=LR)

    predictions = {0: top5_records(model, eval_loader, device, sample_ids)}
    rows = []
    shapes = None

    for epoch in range(1, epochs + 1):
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            if epoch == 1 and batch_idx == 1:
                with torch.no_grad():
                    backbone = nn.Sequential(*list(model.children())[:-1])
                    features = torch.flatten(backbone(images), 1)
                    logits = model.fc(features)
                    probabilities = torch.softmax(logits, dim=1)
                    loss = criterion(logits, labels)
                shapes = {
                    "chain": "x -> backbone -> fc -> softmax -> loss",
                    "input": list(images.shape),
                    "backbone": list(features.shape),
                    "fc": list(logits.shape),
                    "softmax": list(probabilities.shape),
                    "loss": [],
                }

            optimizer.zero_grad(set_to_none=True)
            started = time.time()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            elapsed = time.time() - started
            accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
            rows.append(
                {
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "loss": float(loss.detach()),
                    "batch_acc": accuracy,
                    "lr": LR,
                    "sec": elapsed,
                }
            )
        predictions[epoch] = top5_records(model, eval_loader, device, sample_ids)

    with (DEMO / "batch_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "batch_idx", "loss", "batch_acc", "lr", "sec"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "batch_idx": row["batch_idx"],
                    "loss": f"{row['loss']:.6f}",
                    "batch_acc": f"{row['batch_acc']:.6f}",
                    "lr": f"{row['lr']:.6f}",
                    "sec": f"{row['sec']:.6f}",
                }
            )

    (DEMO / "sample_ids.json").write_text(
        json.dumps(sample_ids, indent=2) + "\n", encoding="utf-8"
    )
    for epoch in range(0, epochs + 1):
        payload = predictions[epoch]
        (DEMO / f"pred_ep{epoch}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    (DEMO / "shapes.json").write_text(
        json.dumps(shapes, indent=2) + "\n", encoding="utf-8"
    )

    plot_batch_flow(shapes, FIGURES / "batch_flow.png")
    plot_batch_loss(rows, FIGURES / "batch_loss.png")
    plot_pred_evolution(sample_ids, predictions, FIGURES / "pred_evolution.png")
    plot_pred_evolution5(
        sample_ids, predictions, FIGURES / "pred_evolution5.png", list(range(epochs + 1))
    )
    plot_sample_acc_curve(sample_ids, predictions, FIGURES / "sample_acc_curve.png")

    print(f"batch rows: {len(rows)}")
    print(f"first loss: {rows[0]['loss']:.6f}")
    print(f"last loss: {rows[-1]['loss']:.6f}")
    for epoch in sorted(predictions):
        hits = sum(
            1
            for image_id in sample_ids
            if predictions[epoch][image_id]["top5"][0]["class_idx"]
            == predictions[epoch][image_id]["true_idx"]
        )
        mean_prob = np.mean(
            [
                predictions[epoch][image_id]["top5"][0]["prob"]
                for image_id in sample_ids
            ]
        )
        print(
            f"ep{epoch} sample top1 acc: {hits}/{len(sample_ids)} "
            f"({hits / len(sample_ids) * 100:.1f}%) "
            f"mean prob {mean_prob * 100:.2f}%"
        )
    names = load_class_names()
    if 5 in predictions:
        print("ep5 predictions:")
        for image_id in sample_ids:
            entry = predictions[5][image_id]
            top = entry["top5"][0]
            print(
                f"  {image_id}: {names.get(top['class_idx'], 'class ' + str(top['class_idx']))} "
                f"({top['prob'] * 100:.1f}%) true={names.get(entry['true_idx'], 'class ' + str(entry['true_idx']))}"
            )
    print(f"shapes: {shapes}")
    for name in (
        "batch_flow.png",
        "batch_loss.png",
        "pred_evolution.png",
        "pred_evolution5.png",
        "sample_acc_curve.png",
    ):
        path = FIGURES / name
        print(f"{path} {path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
