"""生成阶段 C 的六张图（runs/ + report/results.csv → report/figures/）。

所有数值从 runs/ 与 report/results.csv 读取，脚本内不写任何指标字面量。
matplotlib 使用 Agg 后端；中文字体缺失时自动退化为英文标签。
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

MODES = ("scratch", "frozen", "two-stage", "strong")
PREFERRED_FONTS = ("Source Han Sans CN", "Noto Sans CJK SC", "Noto Sans CJK JP")
N_MISCLASSIFIED = 9
N_WORST_CLASSES = 15
CM_TICK_STEP = 10

ZH = {
    "fig1_title": "四组训练 / 验证准确率曲线",
    "train_acc": "训练准确率",
    "val_acc": "验证准确率",
    "epoch": "epoch",
    "accuracy": "准确率",
    "best": "best",
    "fig2_title": "two-stage 混淆矩阵（行归一化，按对角准确率升序）",
    "true_label": "真实类别（官方 1-based）",
    "pred_label": "预测类别（官方 1-based）",
    "normalized": "行归一化比例",
    "fig3_caption": "two-stage 错例 top-9\n按 true_prob 升序\n图片来自 data/raw/jpg",
    "fig4_title": "two-stage 最差 15 类（按 accuracy 升序）",
    "class_label": "类别（官方 1-based）",
    "n_total": "n",
    "fig5_title": "四组指标对比",
    "top1": "top-1",
    "top3": "top-3",
    "macro_f1": "macro-F1",
    "score": "数值",
    "fig6_title": "meta.csv 各 split 每类图片数",
    "images_per_class": "每类图片数",
}
EN = {
    "fig1_title": "Train / val accuracy curves (four modes)",
    "train_acc": "Train accuracy",
    "val_acc": "Val accuracy",
    "epoch": "epoch",
    "accuracy": "accuracy",
    "best": "best",
    "fig2_title": "two-stage confusion matrix (row-normalized, sorted by diagonal accuracy)",
    "true_label": "True class (official 1-based)",
    "pred_label": "Predicted class (official 1-based)",
    "normalized": "Row-normalized fraction",
    "fig3_caption": "two-stage misclassified top-9\nsorted by true_prob\nimages from data/raw/jpg",
    "fig4_title": "two-stage worst 15 classes (accuracy ascending)",
    "class_label": "Class (official 1-based)",
    "n_total": "n",
    "fig5_title": "Metric comparison across four modes",
    "top1": "top-1",
    "top3": "top-3",
    "macro_f1": "macro-F1",
    "score": "score",
    "fig6_title": "meta.csv images per class by split",
    "images_per_class": "images per class",
}


def setup_fonts():
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in PREFERRED_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return None


CJK_FONT = setup_fonts()
TEXT = ZH if CJK_FONT else EN


def load_metrics(runs_dir):
    runs_dir = Path(runs_dir)
    metrics = {}
    histories = {}
    for mode in MODES:
        metrics_path = runs_dir / mode / "metrics.json"
        history_path = runs_dir / mode / "history.csv"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"缺少 {metrics_path}")
        if not history_path.is_file():
            raise FileNotFoundError(f"缺少 {history_path}")
        with metrics_path.open(encoding="utf-8") as handle:
            metrics[mode] = json.load(handle)
        histories[mode] = pd.read_csv(history_path)
    return metrics, histories


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_curves(metrics, histories, output):
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for index, mode in enumerate(MODES):
        history = histories[mode]
        color = colors[index]
        axes[0].plot(history["epoch"], history["train_acc"], color=color, label=mode)
        axes[1].plot(history["epoch"], history["val_acc"], color=color, label=mode)
        best_epoch = int(metrics[mode]["selection"]["best_epoch"])
        row = history.loc[history["epoch"] == best_epoch]
        if row.empty:
            row = history.iloc[[best_epoch - 1]]
        axes[0].scatter(
            row["epoch"], row["train_acc"], color=color, marker="*", s=90, zorder=5
        )
        axes[1].scatter(
            row["epoch"], row["val_acc"], color=color, marker="*", s=90, zorder=5
        )
        axes[1].annotate(
            f"{TEXT['best']} {best_epoch}",
            (float(row["epoch"].iloc[0]), float(row["val_acc"].iloc[0])),
            textcoords="offset points",
            xytext=(3, 6),
            fontsize=7,
            color=color,
        )
    for axis, key in ((axes[0], "train_acc"), (axes[1], "val_acc")):
        axis.set_xlabel(TEXT["epoch"])
        axis.set_ylabel(TEXT[key])
        axis.grid(alpha=0.25)
    axes[0].set_title(TEXT["train_acc"])
    axes[1].set_title(TEXT["val_acc"])
    axes[0].legend(fontsize=8)
    fig.suptitle(TEXT["fig1_title"])
    save(fig, output / "fig1_curves.png")


def fig_cm(runs_dir, output):
    frame = pd.read_csv(Path(runs_dir) / "two-stage" / "confusion_matrix.csv")
    class_ids = frame["true_label"].to_numpy()
    matrix = frame.drop(columns=["true_label"]).to_numpy(dtype=float)
    row_sums = matrix.sum(axis=1)
    normalized = np.divide(
        matrix,
        row_sums[:, None],
        out=np.zeros_like(matrix),
        where=row_sums[:, None] != 0,
    )
    if not np.allclose(normalized.sum(axis=1), 1.0, atol=1e-9):
        raise AssertionError("混淆矩阵行归一化后行和不为 1")
    diagonal = np.diag(normalized)
    order = np.argsort(diagonal, kind="stable")
    normalized = normalized[np.ix_(order, order)]
    display_labels = class_ids[order] + 1

    fig, axis = plt.subplots(figsize=(9.5, 8.2))
    image = axis.imshow(
        normalized, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto"
    )
    ticks = np.arange(0, len(order), CM_TICK_STEP)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_xticklabels([str(int(display_labels[i])) for i in ticks], fontsize=7)
    axis.set_yticklabels([str(int(display_labels[i])) for i in ticks], fontsize=7)
    axis.set_xlabel(TEXT["pred_label"])
    axis.set_ylabel(TEXT["true_label"])
    axis.set_title(TEXT["fig2_title"])
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label=TEXT["normalized"])
    save(fig, output / "fig2_cm.png")


def fig_misclassified(root, output):
    frame = pd.read_csv(root / "runs" / "two-stage" / "misclassified.csv")
    frame = frame.sort_values(["true_prob", "image_id"], kind="stable").head(
        N_MISCLASSIFIED
    )
    image_ids = frame["image_id"].tolist()
    print(f"fig3 image_ids: {image_ids}")
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.8))
    arrow = "→" if CJK_FONT else "->"
    for axis, row in zip(axes.flat, frame.itertuples(index=False)):
        image_path = root / "data" / "raw" / "jpg" / row.image_id
        axis.imshow(plt.imread(image_path))
        axis.axis("off")
        axis.set_title(
            f"{int(row.true_idx)}{arrow}{int(row.pred_idx)} "
            f"(p_true={row.true_prob:.3f}, p_pred={row.pred_prob:.3f})",
            fontsize=7,
        )
    caption_axis = axes.flat[-1]
    caption_axis.axis("off")
    caption_axis.text(0.0, 0.5, TEXT["fig3_caption"], fontsize=8, va="center")
    fig.suptitle(TEXT["fig3_caption"].splitlines()[0])
    save(fig, output / "fig3_misclassified.png")


def fig_worst_classes(runs_dir, output):
    frame = pd.read_csv(Path(runs_dir) / "two-stage" / "per_class.csv")
    worst = frame.sort_values(["accuracy", "class_idx"], kind="stable").head(
        N_WORST_CLASSES
    )
    fig, axis = plt.subplots(figsize=(9, 6))
    positions = np.arange(len(worst))
    axis.barh(positions, worst["accuracy"], color="#c44e52", alpha=0.85)
    axis.set_yticks(positions)
    axis.set_yticklabels([f"class {int(label)}" for label in worst["label"]], fontsize=8)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.12)
    axis.set_xlabel(TEXT["accuracy"])
    axis.set_ylabel(TEXT["class_label"])
    for position, accuracy, total in zip(
        positions, worst["accuracy"], worst["n_total"]
    ):
        axis.text(
            float(accuracy) + 0.01,
            position,
            f"{TEXT['n_total']}={int(total)}",
            va="center",
            fontsize=7,
        )
    axis.set_title(TEXT["fig4_title"])
    axis.grid(axis="x", alpha=0.25)
    save(fig, output / "fig4_worst_classes.png")


def fig_bars(results, output):
    fig, axis = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(results))
    width = 0.26
    series = (
        ("test_top1", TEXT["top1"], "#4c72b0"),
        ("test_top3", TEXT["top3"], "#55a868"),
        ("test_macro_f1", TEXT["macro_f1"], "#c44e52"),
    )
    for index, (column, label, color) in enumerate(series):
        values = results[column].to_numpy(dtype=float)
        bars = axis.bar(
            positions + (index - 1) * width, values, width, label=label, color=color
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    axis.set_xticks(positions)
    axis.set_xticklabels(results["mode"].tolist())
    axis.set_ylim(0.0, 1.08)
    axis.set_ylabel(TEXT["score"])
    axis.set_title(TEXT["fig5_title"])
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    save(fig, output / "fig5_bars.png")


def fig_split(root, output):
    meta = pd.read_csv(root / "data" / "meta.csv")
    counts = meta.groupby(["label", "split"]).size().unstack(fill_value=0)
    splits = ["train", "val", "test"]
    data = [counts[split].to_numpy() for split in splits]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.boxplot(data, tick_labels=splits)
    axis.set_ylabel(TEXT["images_per_class"])
    axis.set_title(TEXT["fig6_title"])
    axis.grid(axis="y", alpha=0.25)
    save(fig, output / "fig6_split.png")


def main():
    root = Path(__file__).resolve().parents[1]
    results_path = root / "report" / "results.csv"
    if not results_path.is_file():
        print(f"make_figures FAIL: 缺少 {results_path}", file=sys.stderr)
        return 1
    output = root / "report" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    try:
        metrics, histories = load_metrics(root / "runs")
        results = pd.read_csv(results_path)
        fig_curves(metrics, histories, output)
        fig_cm(root / "runs", output)
        fig_misclassified(root, output)
        fig_worst_classes(root / "runs", output)
        fig_bars(results, output)
        fig_split(root, output)
    except FileNotFoundError as exc:
        print(f"make_figures FAIL: {exc}", file=sys.stderr)
        return 1

    if CJK_FONT:
        print(f"font: {CJK_FONT}")
    else:
        print("font: CJK not found, fallback to DejaVu Sans (English labels)")
    for path in sorted(output.glob("*.png")):
        print(f"{path.relative_to(root)} {path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
