"""生成 26 页 16:9 中文演讲幻灯（阶段 E 过程性展示版）。

实验数值从 report/results.csv、runs/*/metrics.json、history.csv、
per_class.csv、misclassified.csv、data/meta.csv 与 data/class_names.csv 读取；
代码页片段从 scripts/ 源文件按锚点提取，再交给 Pygments 渲染；
公开对比数字来自外部事实，单独置于 PUBLIC_BENCHMARKS。
公式由 matplotlib mathtext 渲染为透明 PNG；fig7 为卷积示意图。
"""

import csv
import json
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

MONO_FONT = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSansMono.ttf"

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image
from pptx import Presentation
from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import PythonLexer
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

MODES = ("scratch", "frozen", "two-stage", "strong")
FONT = "Source Han Sans CN"
SLIDE_W = 13.333
SLIDE_H = 7.5
TOTAL_SLIDES = 26
IMAGE_SIZE = 224
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 3

NAVY = RGBColor(0x1F, 0x38, 0x64)
ORANGE = RGBColor(0xC5, 0x5A, 0x11)
GRAY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0xF2, 0xF2, 0xF2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x26, 0x26, 0x26)

# (LaTeX, 白话解释)；顺序对应 f1..f8
FORMULAS = [
    (
        r"x \in \mathbb{R}^{3\times224\times224}",
        "一张图就是 3 个通道、224×224 个数字。",
    ),
    (
        r"p_c = \frac{\exp(z_c)}{\sum_k \exp(z_k)}",
        "分数变成概率：正确类分数越高，概率越大。",
    ),
    (
        r"y_{i,j} = f\left(\sum_{c,k,l} w_{c,k,l}\,x_{c,i+k,j+l}+b\right)",
        "3×3 小窗口扫过全图，同一组权重反复使用。",
    ),
    (
        r"L = -\sum_c y_c \log p_c",
        "正确类概率越低，交叉熵罚得越狠。",
    ),
    (
        r"\theta \leftarrow \theta - \eta \nabla_\theta L",
        "沿让损失变小的方向，挪一小步。",
    ),
    (
        r"\theta_{\mathrm{backbone}} \leftarrow \theta_{\mathrm{ImageNet}}^{*},\;\theta_{\mathrm{head}} \leftarrow \mathrm{random}",
        "骨干借 ImageNet，分类头重新随机初始化。",
    ),
    (
        r"52{,}326 = 512\times102+102",
        "只训练分类头：512 维到 102 类。",
    ),
    (
        r"\eta_{\mathrm{head}}=10^{-3},\;\eta_{\mathrm{backbone}}=10^{-4}",
        "头用大学习率，骨干用小学习率。",
    ),
]

# 公开结果：非本实验数据，仅用于量级参照；协议不同，不可直接比较。
PUBLIC_BENCHMARKS = (
    ("Nilsback 2008", 72.8, "103 类旧库 · 手工特征+SVM"),
    ("CCT-14/7×2", 99.76, "from scratch · Transformer"),
    ("ViT-L/16 (2026)", 99.37, "303.4M · 官方划分"),
)

# 代码片段必须从这些源文件提取；锚点失配即报错退出。
SNIPPETS = {
    "fc": ("scripts/train_full.py", "model.fc =", 4),
    "freeze": ("scripts/train_full.py", "def set_backbone_trainable", 8),
    "dataset": ("scripts/train_full.py", "class Flowers(", 10),
    "aug": ("scripts/presets.py", "AUG_OPS =", 13),
    "split": ("scripts/data_split.py", "def _hash_key", 8),
    "eval": ("scripts/train_full.py", "top = probs.topk", 16),
}


def pct(value):
    return f"{float(value) * 100:.2f}%"


def load_results(root):
    rows = {}
    with (root / "report" / "results.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["mode"]] = {
                key: float(value) if key != "mode" else value
                for key, value in row.items()
            }
    return rows


def load_metrics(root):
    result = {}
    for mode in MODES:
        with (root / "runs" / mode / "metrics.json").open(encoding="utf-8") as handle:
            result[mode] = json.load(handle)
    return result


def load_history(root, mode):
    rows = []
    with (root / "runs" / mode / "history.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            converted = {}
            for key, value in row.items():
                try:
                    converted[key] = float(value)
                except ValueError:
                    converted[key] = value
            rows.append(converted)
    return rows


def load_dataset(root):
    with (root / "data" / "meta.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = Counter(int(row["label"]) for row in rows)
    splits = Counter(row["split"] for row in rows)
    test_counts = Counter(
        int(row["label"]) for row in rows if row["split"] == "test"
    )
    return {
        "total": len(rows),
        "classes": len(labels),
        "min_class": min(labels.values()),
        "max_class": max(labels.values()),
        "official_train": splits["train"],
        "official_val": splits["val"],
        "official_test": splits["test"],
        "test_min": min(test_counts.values()),
        "test_max": max(test_counts.values()),
    }


def load_class_data(root):
    with (root / "data" / "class_names.csv").open(encoding="utf-8") as handle:
        names = {int(row["class_idx"]): row["name"] for row in csv.DictReader(handle)}
    with (root / "runs" / "two-stage" / "per_class.csv").open(
        encoding="utf-8"
    ) as handle:
        per_class = [
            {
                "class_idx": int(row["class_idx"]),
                "label": int(row["label"]),
                "n_total": int(row["n_total"]),
                "accuracy": float(row["accuracy"]),
                "name": names[int(row["class_idx"])],
            }
            for row in csv.DictReader(handle)
        ]
    with (root / "runs" / "two-stage" / "misclassified.csv").open(
        encoding="utf-8"
    ) as handle:
        misclassified = [
            {
                "true_idx": int(row["true_idx"]),
                "pred_idx": int(row["pred_idx"]),
                "true_prob": float(row["true_prob"]),
            }
            for row in csv.DictReader(handle)
        ]
    misclassified.sort(key=lambda item: (item["true_prob"],))
    worst = min(per_class, key=lambda item: (item["accuracy"], item["class_idx"]))
    largest = max(per_class, key=lambda item: item["n_total"])
    name_to_idx = {name: index for index, name in names.items()}
    camellia_idx = name_to_idx["camellia"]
    mallow_idx = name_to_idx["mallow"]
    camellia_mallow = sum(
        1
        for item in misclassified[:9]
        if item["true_idx"] == camellia_idx and item["pred_idx"] == mallow_idx
    )
    return {
        "names": names,
        "worst": worst,
        "largest": largest,
        "camellia_mallow": camellia_mallow,
        "top9": misclassified[:9],
    }


def style_run(run, size, color, bold=False, font=FONT):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        element = rpr.find(qn(tag))
        if element is None:
            element = rpr.makeelement(qn(tag), {})
            rpr.append(element)
        element.set("typeface", font)


def add_text(
    slide,
    x,
    y,
    width,
    height,
    text,
    size=18,
    color=INK,
    bold=False,
    align=PP_ALIGN.LEFT,
    anchor=MSO_ANCHOR.TOP,
    font=FONT,
):
    box = slide.shapes.add_textbox(
        Inches(x), Inches(y), Inches(width), Inches(height)
    )
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    frame.vertical_anchor = anchor
    for index, line in enumerate(text.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = align
        paragraph.space_after = Pt(4)
        run = paragraph.add_run()
        run.text = line
        style_run(run, size, color, bold, font)
    return box


def add_bullets(slide, x, y, width, height, items, size=18, color=INK):
    box = slide.shapes.add_textbox(
        Inches(x), Inches(y), Inches(width), Inches(height)
    )
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    for index, item in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.space_after = Pt(8)
        run = paragraph.add_run()
        run.text = f"• {item}"
        style_run(run, size, color)
    return box


def add_rect(slide, x, y, width, height, color):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(x),
        Inches(y),
        Inches(width),
        Inches(height),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def add_header(slide, title, page):
    add_rect(slide, 0, 0, SLIDE_W, 0.95, NAVY)
    add_rect(slide, 0, 0.95, SLIDE_W, 0.06, ORANGE)
    add_text(slide, 0.65, 0.2, 11.8, 0.6, title, 26, WHITE, True)
    add_text(
        slide,
        12.15,
        7.12,
        0.9,
        0.3,
        f"{page:02d}/{TOTAL_SLIDES}",
        11,
        GRAY,
        align=PP_ALIGN.RIGHT,
    )


def add_image_fit(slide, path, x, y, width, height):
    with Image.open(path) as image:
        image_w, image_h = image.size
    scale = min(width / image_w, height / image_h)
    draw_w = image_w * scale
    draw_h = image_h * scale
    left = x + (width - draw_w) / 2
    top = y + (height - draw_h) / 2
    slide.shapes.add_picture(
        str(path), Inches(left), Inches(top), Inches(draw_w), Inches(draw_h)
    )
    return left, top, draw_w, draw_h


def add_caption(slide, text, y=6.95):
    add_text(slide, 0.65, y, 12.0, 0.35, text, 14, GRAY)


def add_table(slide, headers, rows, x, y, width, height, col_widths):
    shape = slide.shapes.add_table(
        len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(width), Inches(height)
    )
    table = shape.table
    for index, col_width in enumerate(col_widths):
        table.columns[index].width = Inches(col_width)
    for row_index, values in enumerate([headers] + rows):
        for col_index, value in enumerate(values):
            cell = table.cell(row_index, col_index)
            cell.text = str(value)
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.04)
            cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if row_index == 0 else (
                LIGHT if row_index % 2 == 0 else WHITE
            )
            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = (
                    PP_ALIGN.LEFT if col_index == 0 else PP_ALIGN.CENTER
                )
                for run in paragraph.runs:
                    style_run(
                        run,
                        15 if row_index == 0 else 14,
                        WHITE if row_index == 0 else INK,
                        bold=row_index == 0,
                    )


def render_formula(tex, path):
    plt.rcParams["mathtext.fontset"] = "stix"
    fig = plt.figure(figsize=(10, 1.8))
    fig.patch.set_alpha(0)
    fig.text(
        0.5,
        0.5,
        f"${tex}$",
        fontsize=40 if len(tex) < 65 else 32,
        ha="center",
        va="center",
        color="#1F3864",
    )
    fig.savefig(
        path, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.08
    )
    plt.close(fig)


def make_fig7(path):
    plt.rcParams["font.sans-serif"] = [FONT, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.set_xlim(0, 11)
    axis.set_ylim(0, 5.2)
    axis.axis("off")
    cell = 0.55
    gap = 0.05
    for row in range(5):
        for col in range(5):
            axis.add_patch(
                Rectangle(
                    (0.5 + col * (cell + gap), 1.2 + row * (cell + gap)),
                    cell,
                    cell,
                    facecolor="#DCE6F1",
                    edgecolor="#1F3864",
                    linewidth=1.2,
                )
            )
    for row in range(3):
        for col in range(3):
            axis.add_patch(
                Rectangle(
                    (4.4 + col * (cell + gap), 1.9 + row * (cell + gap)),
                    cell,
                    cell,
                    facecolor="#F4B183",
                    edgecolor="#C55A11",
                    linewidth=1.4,
                )
            )
    for row in range(3):
        for col in range(3):
            axis.add_patch(
                Rectangle(
                    (7.7 + col * (cell + gap), 1.9 + row * (cell + gap)),
                    cell,
                    cell,
                    facecolor="#E2F0D9",
                    edgecolor="#1F3864",
                    linewidth=1.2,
                )
            )
    axis.add_patch(
        FancyArrowPatch(
            (3.75, 2.55), (4.35, 2.55), arrowstyle="->", color="#C55A11", lw=2.5
        )
    )
    axis.add_patch(
        FancyArrowPatch(
            (6.55, 2.55), (7.65, 2.55), arrowstyle="->", color="#C55A11", lw=2.5
        )
    )
    axis.text(1.6, 4.65, "输入 5×5", ha="center", fontsize=15, color="#1F3864")
    axis.text(5.2, 4.65, "3×3 卷积核", ha="center", fontsize=15, color="#C55A11")
    axis.text(8.8, 4.65, "输出 3×3", ha="center", fontsize=15, color="#1F3864")
    axis.text(
        5.5,
        0.35,
        "滑动窗口，共享权重",
        ha="center",
        fontsize=15,
        color="#C55A11",
    )
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def wrap_code(lines, width=64):
    output = []
    for line in lines:
        body = line.strip()
        if not body:
            output.append("")
            continue
        if len(line) <= width:
            output.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip())]
        chunks = textwrap.wrap(
            body,
            width=max(20, width - len(indent) - 4),
            break_long_words=True,
            break_on_hyphens=False,
        )
        output.append(indent + chunks[0])
        output.extend(indent + "    " + chunk for chunk in chunks[1:])
    return output


def extract_snippet(root, key):
    rel_path, anchor, count = SNIPPETS[key]
    path = root / rel_path
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    for index, line in enumerate(lines):
        if anchor in line:
            start = index
            break
    if start is None:
        raise SystemExit(f"SNIPPET {key}: 锚点未找到 {rel_path}::{anchor}")
    chunk = lines[start : start + count]
    if len(chunk) < count:
        raise SystemExit(f"SNIPPET {key}: 源文件行数不足（{rel_path}）")
    indents = [len(line) - len(line.lstrip()) for line in chunk if line.strip()]
    cut = min(indents) if indents else 0
    stripped = [line[cut:] if len(line) >= cut else line.lstrip() for line in chunk]
    while stripped and not stripped[-1].strip():
        stripped.pop()
    if not stripped:
        raise SystemExit(f"SNIPPET {key}: 提取结果为空")
    return {
        "path": rel_path,
        "start": start + 1,
        "end": start + len(chunk),
        "lines": wrap_code(stripped),
        "first": stripped[0],
    }


def render_code_images(root, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for index, key in enumerate(SNIPPETS, start=1):
        info = extract_snippet(root, key)
        code = "\n".join(info["lines"])
        # DejaVu Sans Mono 不含 CJK；split 片段含中文 docstring，单独用 Noto CJK。
        font_path = MONO_FONT
        if key == "split":
            font_path = Path("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")
        formatter = ImageFormatter(
            font_name=str(font_path),
            font_size=20,
            line_numbers=True,
            style="monokai",
            background_color="#1E1E1E",
            line_pad=6,
            image_pad=12,
        )
        image_path = output_dir / f"c{index}.png"
        image_path.write_bytes(highlight(code, PythonLexer(), formatter))
        info["image"] = image_path
        result[key] = info
    return result


def make_pipeline(path):
    plt.rcParams["font.sans-serif"] = [FONT, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axis = plt.subplots(figsize=(12.6, 2.9))
    axis.set_xlim(0, 13)
    axis.set_ylim(0, 3)
    axis.axis("off")
    labels = [
        "meta.csv",
        "sha256\n划分",
        "Dataset /\nDataLoader",
        "ResNet18\nfc=102",
        "两阶段\n训练",
        "评估\nCM / top-k",
        "runs/\n*.csv,json",
    ]
    centers = [0.95, 2.85, 4.75, 6.65, 8.55, 10.45, 12.35]
    width = 1.55
    height = 1.35
    for center, label in zip(centers, labels):
        axis.add_patch(
            FancyBboxPatch(
                (center - width / 2, 1.0),
                width,
                height,
                boxstyle="round,pad=0.05,rounding_size=0.12",
                facecolor="#DCE6F1",
                edgecolor="#1F3864",
                linewidth=1.5,
            )
        )
        axis.text(
            center,
            1.68,
            label,
            ha="center",
            va="center",
            fontsize=10.5,
            color="#1F3864",
        )
    for left, right in zip(centers[:-1], centers[1:]):
        axis.add_patch(
            FancyArrowPatch(
                (left + width / 2 + 0.02, 1.68),
                (right - width / 2 - 0.02, 1.68),
                arrowstyle="->",
                color="#C55A11",
                lw=2,
            )
        )
    axis.text(
        6.5,
        0.35,
        "数据 → 划分 → 模型 → 训练 → 评估 → 产物",
        ha="center",
        fontsize=11,
        color="#C55A11",
    )
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_aug_examples(root, path):
    images = sorted((root / "data" / "raw" / "jpg").glob("*.jpg"))
    if not images:
        raise SystemExit("缺少 data/raw/jpg/*.jpg，无法生成数据增强示例")
    base = Image.open(images[0]).convert("RGB")
    width, height = base.size
    variants = (
        ("原图", base),
        ("裁剪", base.crop((width * 0.1, height * 0.1, width * 0.9, height * 0.9))),
        ("翻转", base.transpose(Image.FLIP_LEFT_RIGHT)),
        ("旋转", base.rotate(25, expand=True, fillcolor="white")),
    )
    plt.rcParams["font.sans-serif"] = [FONT, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.2))
    for axis, (label, image) in zip(axes.flat, variants):
        axis.imshow(image)
        axis.set_title(label, fontsize=13, color="#1F3864")
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def add_code_panel(slide, path, x, y, width, height):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(x),
        Inches(y),
        Inches(width),
        Inches(height),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0x1E, 0x1E, 0x1E)
    shape.line.color.rgb = RGBColor(0x7F, 0x7F, 0x7F)
    shape.line.width = Pt(1)
    add_image_fit(slide, path, x + 0.08, y + 0.08, width - 0.16, height - 0.16)


def build(root, output):
    results = load_results(root)
    metrics = load_metrics(root)
    dataset = load_dataset(root)
    class_data = load_class_data(root)
    histories = {mode: load_history(root, mode) for mode in MODES}
    epoch12 = {
        mode: next(row["val_acc"] for row in histories[mode] if row["epoch"] == 12)
        for mode in MODES
    }
    scratch_top1 = results["scratch"]["test_top1"]
    frozen_top1 = results["frozen"]["test_top1"]
    two_top1 = results["two-stage"]["test_top1"]
    strong_top1 = results["strong"]["test_top1"]
    total_params = metrics["scratch"]["run"]["params_total"]
    trainable = metrics["frozen"]["run"]["params_trainable_stage1"]
    trainable_pct = trainable / total_params
    two_stage = (
        metrics["two-stage"]["run"]["unfreeze_epoch"] - 1,
        metrics["two-stage"]["run"]["epochs_planned"]
        - (metrics["two-stage"]["run"]["unfreeze_epoch"] - 1),
    )
    strong_stage = (
        metrics["strong"]["run"]["unfreeze_epoch"] - 1,
        metrics["strong"]["run"]["epochs_planned"]
        - (metrics["strong"]["run"]["unfreeze_epoch"] - 1),
    )
    frozen_epochs = metrics["frozen"]["run"]["epochs_run"]
    strong_ls = metrics["strong"]["run"]["label_smoothing"]

    figures = root / "report" / "figures"
    formula_dir = figures / "formulas"
    formula_dir.mkdir(parents=True, exist_ok=True)
    formula_paths = []
    for index, (tex, _plain) in enumerate(FORMULAS, start=1):
        path = formula_dir / f"f{index}.png"
        render_formula(tex, path)
        formula_paths.append(path)
    fig7 = figures / "fig7_conv.png"
    make_fig7(fig7)
    code_data = render_code_images(root, figures / "code")
    print("SNIPPETS:")
    for key in SNIPPETS:
        info = code_data[key]
        print(
            f"  {key}: {info['path']}:{info['start']}-{info['end']} | {info['first']}"
        )
    pipeline = figures / "pipeline.png"
    make_pipeline(pipeline)
    aug_example = figures / "aug_examples.png"
    make_aug_examples(root, aug_example)

    with (root / "runs" / "two-stage" / "history.csv").open(
        encoding="utf-8"
    ) as handle:
        raw_history = list(csv.reader(handle))
    log_columns = ("epoch", "train_loss", "val_loss", "val_acc", "phase")
    log_index = [raw_history[0].index(column) for column in log_columns]
    log_lines = [",".join(raw_history[0][index] for index in log_index)]
    log_lines += [
        ",".join(raw_history[row][index] for index in log_index)
        for row in (1, 2, 3, 6)
    ]
    log_text = "\n".join(log_lines)

    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    blank = prs.slide_layouts[6]

    # 1 封面
    slide = prs.slides.add_slide(blank)
    add_rect(slide, 0, 0, SLIDE_W, SLIDE_H, NAVY)
    add_rect(slide, 0.85, 2.15, 0.12, 2.25, ORANGE)
    add_text(slide, 1.25, 1.85, 11.2, 0.9, "从零？不，借用", 54, WHITE, True)
    add_text(
        slide,
        1.27,
        2.95,
        11.2,
        0.7,
        "牛津 102 花卉细粒度分类的迁移学习实验",
        26,
        WHITE,
    )
    add_text(
        slide,
        1.27,
        5.25,
        8.0,
        0.7,
        f"{pct(scratch_top1)} → {pct(strong_top1)}",
        32,
        ORANGE,
        True,
    )
    add_text(
        slide,
        1.27,
        6.55,
        9.0,
        0.35,
        "ResNet-18 · Oxford-102 · 2026",
        14,
        RGBColor(0xBF, 0xBF, 0xBF),
    )

    # 2 计算机眼里的图片
    slide = prs.slides.add_slide(blank)
    add_header(slide, "计算机眼里的图片", 2)
    add_image_fit(slide, formula_paths[0], 0.7, 1.2, 12.0, 2.4)
    add_text(
        slide,
        0.8,
        3.95,
        11.8,
        0.6,
        f"一张 224×224 的彩色图 = {IMAGE_PIXELS:,} 个 0–255 的数字",
        23,
        INK,
        True,
    )
    add_bullets(
        slide,
        0.8,
        4.85,
        11.8,
        1.7,
        ["3 个颜色通道：红、绿、蓝", "模型看到的不是花，而是数字矩阵"],
        18,
    )

    # 3 神经网络在做什么
    slide = prs.slides.add_slide(blank)
    add_header(slide, "神经网络在做什么", 3)
    add_image_fit(slide, formula_paths[1], 1.1, 1.15, 11.2, 2.3)
    add_text(
        slide,
        0.8,
        3.95,
        11.8,
        0.6,
        "102 个分数经过 softmax，变成 102 个概率",
        23,
        INK,
        True,
    )
    add_bullets(
        slide,
        0.8,
        4.85,
        11.8,
        1.7,
        ["z_c 越大，第 c 类概率越高", "所有类别概率之和等于 1"],
        18,
    )

    # 4 卷积
    slide = prs.slides.add_slide(blank)
    add_header(slide, "卷积：一个小窗口扫过图片", 4)
    add_image_fit(slide, formula_paths[2], 0.7, 1.1, 12.0, 1.65)
    add_image_fit(slide, fig7, 0.4, 2.95, 7.7, 3.9)
    add_bullets(
        slide,
        8.35,
        3.05,
        4.4,
        3.4,
        ["3×3 窗口在图上滑动", "同一组权重反复使用", "每次算加权和再激活"],
        17,
    )

    # 5 交叉熵
    slide = prs.slides.add_slide(blank)
    add_header(slide, "怎么知道模型错了", 5)
    add_image_fit(slide, formula_paths[3], 0.8, 1.2, 11.8, 2.4)
    add_text(
        slide,
        0.8,
        4.0,
        11.8,
        0.6,
        "正确类概率越低，罚得越狠",
        24,
        INK,
        True,
    )
    add_bullets(
        slide,
        0.8,
        4.9,
        11.8,
        1.5,
        ["训练目标：让正确类概率接近 1", "交叉熵越小，模型越自信且正确"],
        18,
    )

    # 6 梯度下降
    slide = prs.slides.add_slide(blank)
    add_header(slide, "怎么改：梯度下降", 6)
    add_image_fit(slide, formula_paths[4], 1.2, 1.2, 10.8, 2.4)
    add_text(
        slide,
        0.8,
        4.0,
        11.8,
        0.6,
        "沿让损失变小的方向，挪一小步",
        24,
        INK,
        True,
    )
    add_bullets(
        slide,
        0.8,
        4.9,
        11.8,
        1.5,
        ["η 是步长：太大震荡，太小太慢", "重复：算损失 → 求梯度 → 更新"],
        18,
    )

    # 7 为什么不能从零学
    slide = prs.slides.add_slide(blank)
    add_header(slide, "为什么不能从零学", 7)
    for x, label, value, note, color in (
        (
            0.75,
            "scratch",
            pct(scratch_top1),
            f"{metrics['scratch']['run']['epochs_run']} epoch 从零训练",
            ORANGE,
        ),
        (
            6.95,
            "frozen",
            pct(frozen_top1),
            f"{frozen_epochs} epoch 只训分类头",
            NAVY,
        ),
    ):
        add_rect(slide, x, 1.55, 5.6, 3.4, LIGHT)
        add_rect(slide, x, 1.55, 0.12, 3.4, color)
        add_text(slide, x + 0.35, 1.85, 4.9, 0.45, label, 22, GRAY, True)
        add_text(slide, x + 0.35, 2.4, 4.9, 1.1, value, 58, color, True)
        add_text(slide, x + 0.35, 3.75, 4.9, 0.4, note, 17, INK)
    add_text(
        slide,
        0.8,
        5.55,
        11.8,
        0.6,
        "预训练让模型从“记住训练集”转向可迁移特征",
        24,
        INK,
        True,
    )

    # 8 迁移学习
    slide = prs.slides.add_slide(blank)
    add_header(slide, "迁移学习：借别人学过的「看」的能力", 8)
    add_image_fit(slide, formula_paths[5], 0.7, 1.15, 12.0, 2.1)
    add_text(
        slide,
        0.8,
        3.7,
        11.8,
        0.7,
        "预训练：ImageNet 120 万张先学会看边缘、纹理、形状",
        21,
        INK,
        True,
    )
    add_bullets(
        slide,
        0.8,
        4.65,
        11.8,
        1.5,
        ["骨干借过来；分类头重新随机初始化", "先在通用视觉上“学会看”，再学认花"],
        18,
    )

    # 9 只换最后一层
    slide = prs.slides.add_slide(blank)
    add_header(slide, "只换最后一层", 9)
    add_image_fit(slide, formula_paths[6], 1.2, 1.05, 10.8, 1.55)
    cards = (
        (0.75, "总参数", f"{total_params:,}", "ResNet-18"),
        (4.85, "首阶段可训", f"{trainable:,}", "frozen / two-stage / strong"),
        (8.95, "可训占比", f"{trainable_pct * 100:.2f}%", "只换 512→102 分类头"),
    )
    for x, label, value, note in cards:
        add_rect(slide, x, 2.9, 3.65, 2.5, LIGHT)
        add_text(slide, x + 0.3, 3.1, 3.05, 0.4, label, 17, GRAY)
        add_text(slide, x + 0.3, 3.55, 3.05, 0.7, value, 28, NAVY, True)
        add_text(slide, x + 0.3, 4.55, 3.05, 0.6, note, 14, INK)
    add_text(
        slide,
        0.8,
        5.85,
        11.8,
        0.5,
        "骨干保留 ImageNet 权重；分类头把 512 维映射到 102 类",
        19,
        INK,
    )

    # 10 代码：换掉最后一层
    slide = prs.slides.add_slide(blank)
    add_header(slide, "代码：换掉最后一层", 10)
    add_code_panel(slide, code_data["fc"]["image"], 0.45, 1.1, 12.4, 2.55)
    add_bullets(
        slide,
        0.7,
        3.9,
        12.0,
        0.95,
        ["`model.fc` 只替换分类头", "512 维 → 102 类：骨干权重保持不变"],
        17,
        ORANGE,
    )
    add_text(
        slide,
        0.7,
        5.05,
        12.0,
        0.45,
        "输出层是唯一和 102 类绑定的部分；骨干不需要动",
        19,
        INK,
    )
    add_text(
        slide,
        0.7,
        5.75,
        12.0,
        0.35,
        f"{trainable:,} = 512×102 + 102（回扣第 9 页）",
        14,
        GRAY,
    )

    # 11 两阶段微调
    slide = prs.slides.add_slide(blank)
    add_header(slide, "两阶段微调", 11)
    add_image_fit(slide, formula_paths[7], 1.0, 1.05, 11.3, 1.55)
    for y, mode, stages, color in (
        (2.9, "two-stage", two_stage, NAVY),
        (4.0, "strong", strong_stage, ORANGE),
    ):
        total = stages[0] + stages[1]
        freeze_w = 10.0 * stages[0] / total
        add_text(slide, 0.7, y, 1.6, 0.4, mode, 16, color, True)
        add_rect(slide, 2.25, y + 0.05, freeze_w, 0.5, GRAY)
        add_rect(slide, 2.25 + freeze_w, y + 0.05, 10.0 - freeze_w, 0.5, color)
        add_text(
            slide,
            2.4,
            y + 0.1,
            max(freeze_w - 0.2, 1.0),
            0.35,
            f"冻结 {stages[0]}",
            13,
            WHITE,
            True,
        )
        add_text(
            slide,
            2.25 + freeze_w + 0.2,
            y + 0.1,
            4.5,
            0.35,
            f"解冻 {stages[1]}",
            13,
            WHITE,
            True,
        )
    add_bullets(
        slide,
        0.7,
        5.2,
        12.0,
        1.7,
        ["解冻时重建优化器与 scheduler", "解冻处允许 loss 抖动，不改语义"],
        17,
    )

    # 12 代码：冻结与解冻
    slide = prs.slides.add_slide(blank)
    add_header(slide, "代码：冻结与解冻", 12)
    add_code_panel(slide, code_data["freeze"]["image"], 0.45, 1.1, 12.4, 4.85)
    add_bullets(
        slide,
        0.7,
        6.15,
        12.0,
        1.0,
        ["fc 层始终 requires_grad=True", "解冻后重建 optimizer，动量重置"],
        17,
        ORANGE,
    )

    # 13 数据增强 + 代码
    slide = prs.slides.add_slide(blank)
    add_header(slide, "数据增强 + 代码", 13)
    add_image_fit(slide, aug_example, 0.4, 1.1, 5.8, 4.4)
    add_code_panel(slide, code_data["aug"]["image"], 6.45, 1.1, 6.45, 4.4)
    add_bullets(
        slide,
        0.7,
        5.8,
        12.0,
        1.2,
        ["裁剪/翻转/旋转/颜色抖动：一张图变多样本", "`AUG_OPS`：baseline 与 strong 两套配置"],
        16,
        ORANGE,
    )

    # 14 流水线 + dataset 代码
    slide = prs.slides.add_slide(blank)
    add_header(slide, "流水线：从数据到产物", 14)
    add_image_fit(slide, pipeline, 0.4, 1.05, 12.5, 2.8)
    add_code_panel(slide, code_data["dataset"]["image"], 0.45, 4.05, 12.4, 2.9)
    add_text(
        slide,
        0.7,
        7.05,
        12.0,
        0.3,
        "Dataset 负责把 image_id 和标签读成模型输入",
        12,
        GRAY,
    )

    # 15 数据集与划分
    slide = prs.slides.add_slide(blank)
    add_header(slide, "数据集与划分", 15)
    add_bullets(
        slide,
        0.7,
        1.45,
        5.9,
        4.3,
        [
            f"{dataset['total']} 张 / {dataset['classes']} 类 / 每类 {dataset['min_class']}–{dataset['max_class']}",
            f"官方 split：{dataset['official_train']} / {dataset['official_val']} / {dataset['official_test']}",
            f"我们的 train/val：{metrics['scratch']['run']['train_n']} / {metrics['scratch']['run']['val_n']}",
            f"test {metrics['scratch']['run']['test_n']} 只做最终评估",
        ],
        19,
    )
    add_image_fit(slide, figures / "fig6_split.png", 6.7, 1.2, 6.0, 4.7)
    add_text(
        slide,
        6.7,
        6.1,
        6.0,
        0.3,
        f"test 每类 {dataset['test_min']}–{dataset['test_max']}，分布不均",
        13,
        GRAY,
    )

    # 16 代码：sha256 分桶
    slide = prs.slides.add_slide(blank)
    add_header(slide, "代码：sha256 分桶", 16)
    add_code_panel(slide, code_data["split"]["image"], 0.45, 1.1, 12.4, 4.85)
    add_bullets(
        slide,
        0.7,
        6.15,
        12.0,
        1.0,
        ["每个 image_id 算 sha256 键", "按 (k, image_id) 排序：前 3 张进 val"],
        17,
        ORANGE,
    )

    # 17 四组对照
    slide = prs.slides.add_slide(blank)
    add_header(slide, "四组对照实验", 17)
    cards = (
        (0.7, 1.25, "scratch", "不借：随机初始化，全部参数可训", ORANGE),
        (6.8, 1.25, "frozen", f"借骨干：只训 {trainable:,} 个参数", NAVY),
        (
            0.7,
            3.85,
            "two-stage",
            f"先借后放开：{two_stage[0]} 轮冻结 + {two_stage[1]} 轮解冻",
            NAVY,
        ),
        (
            6.8,
            3.85,
            "strong",
            f"两阶段 + 强增强 + LS {strong_ls:.1f} + {strong_stage[1]} 轮",
            ORANGE,
        ),
    )
    for x, y, mode, note, color in cards:
        add_rect(slide, x, y, 5.85, 2.2, LIGHT)
        add_rect(slide, x, y, 0.12, 2.2, color)
        add_text(slide, x + 0.3, y + 0.22, 5.3, 0.45, mode, 24, color, True)
        add_text(slide, x + 0.3, y + 1.0, 5.3, 0.8, note, 17, INK)
    add_text(
        slide,
        0.7,
        6.35,
        12.0,
        0.4,
        "四组只差在“借多少”：从全随机到全解冻",
        15,
        GRAY,
    )

    # 18 结果表
    slide = prs.slides.add_slide(blank)
    add_header(slide, "结果总表", 18)
    rows = [
        [
            mode,
            pct(results[mode]["test_top1"]),
            pct(results[mode]["test_top3"]),
            pct(results[mode]["test_macro_f1"]),
        ]
        for mode in MODES
    ]
    add_table(
        slide,
        ["组别", "top-1", "top-3", "macro-F1"],
        rows,
        0.9,
        1.45,
        11.5,
        3.9,
        [3.1, 2.8, 2.8, 2.8],
    )
    add_text(
        slide,
        0.9,
        5.75,
        11.5,
        0.4,
        "全部为 test 集；数据源：report/results.csv",
        14,
        GRAY,
    )

    # 19 曲线 + 真实日志
    slide = prs.slides.add_slide(blank)
    add_header(slide, "训练曲线与日志", 19)
    add_image_fit(slide, figures / "fig1_curves.png", 0.5, 1.0, 12.3, 3.7)
    add_text(
        slide,
        0.7,
        4.72,
        6.0,
        0.25,
        "关键列：epoch / train_loss / val_loss / val_acc / phase",
        11,
        GRAY,
    )
    add_text(
        slide,
        9.1,
        4.7,
        3.5,
        0.3,
        "第 6 轮 = 解冻",
        13,
        ORANGE,
        True,
        align=PP_ALIGN.RIGHT,
    )
    add_rect(slide, 0.55, 5.0, 12.2, 1.6, LIGHT)
    add_text(
        slide,
        0.75,
        5.08,
        11.9,
        1.45,
        log_text,
        14,
        RGBColor(0x1E, 0x1E, 0x1E),
        font="DejaVu Sans Mono",
    )
    add_text(
        slide,
        0.7,
        6.72,
        12.0,
        0.3,
        "runs/two-stage/history.csv：第 1–3、6 轮",
        12,
        GRAY,
    )

    # 20 柱状
    slide = prs.slides.add_slide(blank)
    add_header(slide, "指标对比", 20)
    add_image_fit(slide, figures / "fig5_bars.png", 0.45, 1.05, 12.45, 5.9)
    add_caption(
        slide,
        f"strong {pct(strong_top1)}，two-stage {pct(two_top1)}，相差 {(strong_top1 - two_top1) * 100:.2f} 个百分点",
    )

    # 21 混淆矩阵
    slide = prs.slides.add_slide(blank)
    add_rect(slide, 0, 0, 4.0, SLIDE_H, NAVY)
    add_text(slide, 0.4, 0.55, 3.2, 0.8, "怎么看混淆矩阵", 26, WHITE, True)
    add_text(
        slide,
        0.4,
        1.55,
        3.2,
        1.7,
        "two-stage\n102×102\n行 = 真实\n列 = 预测\n对角线越亮越好",
        15,
        RGBColor(0xD9, 0xD9, 0xD9),
    )
    worst_class = class_data["worst"]
    add_text(
        slide,
        0.4,
        5.55,
        3.2,
        1.1,
        f"左上角最差：\n{worst_class['name']} {pct(worst_class['accuracy'])}",
        18,
        ORANGE,
        True,
    )
    add_text(
        slide,
        3.0,
        7.12,
        0.8,
        0.3,
        "21/26",
        11,
        RGBColor(0xBF, 0xBF, 0xBF),
        align=PP_ALIGN.RIGHT,
    )
    add_image_fit(slide, figures / "fig2_cm.png", 4.1, 0.1, 9.1, 7.3)

    # 22 代码：评估与混淆矩阵
    slide = prs.slides.add_slide(blank)
    add_header(slide, "代码：评估与混淆矩阵", 22)
    add_code_panel(slide, code_data["eval"]["image"], 0.45, 1.1, 12.4, 4.85)
    add_bullets(
        slide,
        0.7,
        6.15,
        12.0,
        1.0,
        ["topk(5) 一次算 top-1/3/5", "CM 用 index_add_ 逐 batch 累加"],
        17,
        ORANGE,
    )

    # 23 错例
    slide = prs.slides.add_slide(blank)
    add_header(slide, "典型错例", 23)
    add_image_fit(slide, figures / "fig3_misclassified.png", 0.35, 1.05, 12.6, 5.65)
    add_caption(
        slide,
        f"true_prob 最小的 {len(class_data['top9'])} 条中，camellia→mallow 有 {class_data['camellia_mallow']} 条",
    )

    # 24 最差类别
    slide = prs.slides.add_slide(blank)
    add_rect(slide, 0, 0, 2.4, SLIDE_H, NAVY)
    add_text(slide, 0.3, 0.55, 1.9, 0.75, "最差类别", 24, WHITE, True)
    add_text(
        slide,
        0.3,
        5.25,
        1.9,
        1.5,
        f"{class_data['largest']['name']}\nn={class_data['largest']['n_total']}\n拖累最大",
        16,
        ORANGE,
        True,
    )
    add_text(
        slide,
        0.3,
        7.12,
        1.8,
        0.3,
        "24/26",
        11,
        RGBColor(0xBF, 0xBF, 0xBF),
    )
    add_image_fit(slide, figures / "fig4_worst_classes.png", 2.45, 0.1, 10.8, 7.3)

    # 25 公开对比
    slide = prs.slides.add_slide(blank)
    add_header(slide, "公开结果量级", 25)
    public_rows = [
        [name, f"{value:.2f}%", note] for name, value, note in PUBLIC_BENCHMARKS
    ]
    public_rows.append(
        [
            "本实验 strong",
            pct(strong_top1),
            f"ResNet-18 · {total_params / 1e6:.1f}M",
        ]
    )
    add_table(
        slide,
        ["方法", "top-1", "说明"],
        public_rows,
        0.8,
        1.35,
        11.7,
        3.7,
        [3.7, 2.0, 6.0],
    )
    add_text(
        slide,
        0.8,
        5.55,
        11.8,
        0.5,
        "协议不同，不可直接比较；这里只做量级参照",
        18,
        ORANGE,
        True,
    )

    # 26 结论、边界与踩坑
    slide = prs.slides.add_slide(blank)
    add_header(slide, "结论、边界与踩坑", 26)
    add_text(slide, 0.7, 1.15, 5.6, 0.35, "结论", 18, NAVY, True)
    add_bullets(
        slide,
        0.7,
        1.55,
        5.6,
        2.4,
        [
            f"预训练是主力：frozen {frozen_epochs} 轮 {pct(frozen_top1)}",
            f"解冻跃迁：strong {pct(strong_top1)}",
            f"scratch 仅 {pct(scratch_top1)}：从零不划算",
        ],
        16,
    )
    add_text(slide, 7.1, 1.15, 5.5, 0.35, "边界", 18, ORANGE, True)
    add_bullets(
        slide,
        7.1,
        1.55,
        5.5,
        2.4,
        [
            "单 seed，无种子方差",
            f"验证 {metrics['scratch']['run']['val_n']} 张，每类 {metrics['scratch']['run']['val_per_class']} 张",
            "strong 是三因素 bundle，不做单因素归因",
        ],
        16,
    )
    add_text(slide, 0.7, 4.2, 5.6, 0.35, "踩坑记录", 16, ORANGE, True)
    add_bullets(
        slide,
        0.7,
        4.6,
        12.0,
        1.5,
        [
            "`/kaggle/src` 只读 → 写 `/kaggle/working`",
            "`truep_sum` 102 vs 64 → 用 index_add_",
            "GPU 长队列 → CPU 冒烟先跑通",
        ],
        15,
    )
    add_text(
        slide,
        0.7,
        6.55,
        12.0,
        0.55,
        "Q & A",
        28,
        ORANGE,
        True,
        align=PP_ALIGN.CENTER,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(output)
    print(f"wrote {output}")
    print(f"slides={len(prs.slides)}")
    return prs

def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "report" / "slides.pptx"
    prs = build(root, output)
    pictures = sum(
        1 for slide in prs.slides for shape in slide.shapes if shape.shape_type == 13
    )
    print(f"pictures={pictures}")
    print("FORMULAS:")
    for index, (tex, plain) in enumerate(FORMULAS, start=1):
        print(f"  f{index}: {tex}  |  {plain}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
