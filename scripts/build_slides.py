"""生成 14 页 16:9 中文演讲幻灯（阶段 E）。

所有实验数值从 report/results.csv、runs/*/metrics.json、history.csv、
per_class.csv、misclassified.csv、data/meta.csv 与 data/class_names.csv 读取；
公开对比数字来自外部事实，单独置于 PUBLIC_BENCHMARKS。
"""

import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

MODES = ("scratch", "frozen", "two-stage", "strong")
FONT = "Source Han Sans CN"
SLIDE_W = 13.333
SLIDE_H = 7.5
TOTAL_SLIDES = 14

NAVY = RGBColor(0x1F, 0x38, 0x64)
ORANGE = RGBColor(0xC5, 0x5A, 0x11)
GRAY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0xF2, 0xF2, 0xF2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x26, 0x26, 0x26)

# 公开结果：非本实验数据，仅用于量级参照；协议不同，不可直接比较。
PUBLIC_BENCHMARKS = (
    ("Nilsback 2008", 72.8, "103 类旧库 · 手工特征+SVM"),
    ("CCT-14/7×2", 99.76, "from scratch · Transformer"),
    ("ViT-L/16 (2026)", 99.37, "303.4M · 官方划分"),
)


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
    camellia_row = next(item for item in per_class if item["class_idx"] == camellia_idx)
    return {
        "names": names,
        "worst": worst,
        "largest": largest,
        "camellia": camellia_row,
        "camellia_mallow": camellia_mallow,
        "top9": misclassified[:9],
    }


def style_run(run, size, color, bold=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        element = rpr.find(qn(tag))
        if element is None:
            element = rpr.makeelement(qn(tag), {})
            rpr.append(element)
        element.set("typeface", FONT)


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
        style_run(run, size, color, bold)
    return box


def add_bullets(slide, x, y, width, height, items, size=18):
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
        style_run(run, size, INK)
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

    # 2 问题
    slide = prs.slides.add_slide(blank)
    add_header(slide, "从零学认花为什么难", 2)
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

    # 3 数据集
    slide = prs.slides.add_slide(blank)
    add_header(slide, "数据集与划分", 3)
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
    add_image_fit(
        slide, root / "report" / "figures" / "fig6_split.png", 6.7, 1.2, 6.0, 4.7
    )
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

    # 4 四组对照
    slide = prs.slides.add_slide(blank)
    add_header(slide, "四组对照实验", 4)
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
        "四组共用同一训练循环、评估口径与产物写出函数",
        15,
        GRAY,
    )

    # 5 参数
    slide = prs.slides.add_slide(blank)
    add_header(slide, "只换最后一层", 5)
    cards = (
        (0.75, "总参数", f"{total_params:,}", "ResNet-18"),
        (4.85, "首阶段可训", f"{trainable:,}", "frozen / two-stage / strong"),
        (8.95, "可训占比", f"{trainable_pct * 100:.2f}%", "只换 512→102 分类头"),
    )
    for x, label, value, note in cards:
        add_rect(slide, x, 1.6, 3.65, 3.6, LIGHT)
        add_text(slide, x + 0.3, 1.9, 3.05, 0.4, label, 18, GRAY)
        add_text(slide, x + 0.3, 2.45, 3.05, 0.9, value, 34, NAVY, True)
        add_text(slide, x + 0.3, 3.8, 3.05, 0.9, note, 15, INK)
    add_text(
        slide,
        0.8,
        5.7,
        11.8,
        0.5,
        "骨干保留 ImageNet 权重；分类头把 512 维映射到 102 类",
        20,
        INK,
    )

    # 6 时间线
    slide = prs.slides.add_slide(blank)
    add_header(slide, "两阶段时间线", 6)
    for y, mode, stages, color in (
        (1.55, "two-stage", two_stage, NAVY),
        (2.85, "strong", strong_stage, ORANGE),
    ):
        total = stages[0] + stages[1]
        freeze_w = 10.6 * stages[0] / total
        add_text(slide, 0.7, y, 1.5, 0.4, mode, 16, color, True)
        add_rect(slide, 2.15, y + 0.05, freeze_w, 0.5, GRAY)
        add_rect(slide, 2.15 + freeze_w, y + 0.05, 10.6 - freeze_w, 0.5, color)
        add_text(
            slide,
            2.3,
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
            2.15 + freeze_w + 0.2,
            y + 0.1,
            5.0,
            0.35,
            f"解冻 {stages[1]}",
            13,
            WHITE,
            True,
        )
    add_bullets(
        slide,
        0.7,
        4.35,
        12.0,
        2.2,
        [
            "解冻时重建优化器与 scheduler，分类头动量重置",
            "解冻处允许 loss 抖动，不改预设语义",
            f"strong 加标签平滑 {strong_ls:.1f} 与更强数据增强",
        ],
        18,
    )
    add_text(
        slide,
        0.7,
        6.45,
        12.0,
        0.35,
        "BN：首阶段冻结，解冻后恢复训练",
        14,
        GRAY,
    )

    # 7 结果表
    slide = prs.slides.add_slide(blank)
    add_header(slide, "四组结果总表", 7)
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

    # 8 曲线
    slide = prs.slides.add_slide(blank)
    add_header(slide, "训练曲线", 8)
    add_image_fit(
        slide, root / "report" / "figures" / "fig1_curves.png", 0.45, 1.05, 12.45, 5.9
    )
    add_caption(
        slide,
        f"scratch 长期落后：第 12 轮 {pct(epoch12['scratch'])}，frozen {pct(epoch12['frozen'])}",
    )

    # 9 柱状
    slide = prs.slides.add_slide(blank)
    add_header(slide, "指标对比", 9)
    add_image_fit(
        slide, root / "report" / "figures" / "fig5_bars.png", 0.45, 1.05, 12.45, 5.9
    )
    add_caption(
        slide,
        f"strong {pct(strong_top1)}，two-stage {pct(two_top1)}，相差 {(strong_top1 - two_top1) * 100:.2f} 个百分点",
    )

    # 10 混淆矩阵（左侧色块，图片占大面积）
    slide = prs.slides.add_slide(blank)
    add_rect(slide, 0, 0, 4.0, SLIDE_H, NAVY)
    add_text(slide, 0.4, 0.55, 3.2, 0.8, "混淆矩阵", 28, WHITE, True)
    add_text(
        slide,
        0.4,
        1.55,
        3.2,
        1.4,
        "two-stage\n102×102\n行归一化\n按对角准确率升序",
        16,
        RGBColor(0xD9, 0xD9, 0xD9),
    )
    worst_class = class_data["worst"]
    add_text(
        slide,
        0.4,
        5.4,
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
        "10/14",
        11,
        RGBColor(0xBF, 0xBF, 0xBF),
        align=PP_ALIGN.RIGHT,
    )
    add_image_fit(
        slide, root / "report" / "figures" / "fig2_cm.png", 4.1, 0.1, 9.1, 7.3
    )

    # 11 错例
    slide = prs.slides.add_slide(blank)
    add_header(slide, "典型错例", 11)
    add_image_fit(
        slide, root / "report" / "figures" / "fig3_misclassified.png", 0.35, 1.05, 12.6, 5.65
    )
    add_caption(
        slide,
        f"true_prob 最小的 {len(class_data['top9'])} 条中，camellia→mallow 有 {class_data['camellia_mallow']} 条",
    )

    # 12 最差类别（左侧窄色块，图片占大面积）
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
        "12/14",
        11,
        RGBColor(0xBF, 0xBF, 0xBF),
    )
    add_image_fit(
        slide, root / "report" / "figures" / "fig4_worst_classes.png", 2.45, 0.1, 10.8, 7.3
    )

    # 13 公开对比
    slide = prs.slides.add_slide(blank)
    add_header(slide, "公开结果量级", 13)
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

    # 14 结论与边界
    slide = prs.slides.add_slide(blank)
    add_header(slide, "结论与边界", 14)
    add_text(slide, 0.7, 1.25, 5.6, 0.4, "结论", 20, NAVY, True)
    add_bullets(
        slide,
        0.7,
        1.75,
        5.6,
        3.8,
        [
            f"预训练是主力：frozen {frozen_epochs} 轮 {pct(frozen_top1)}",
            f"解冻跃迁：strong {pct(strong_top1)}",
            f"scratch 仅 {pct(scratch_top1)}：从零不划算",
        ],
        18,
    )
    add_text(slide, 7.1, 1.25, 5.5, 0.4, "边界", 20, ORANGE, True)
    add_bullets(
        slide,
        7.1,
        1.75,
        5.5,
        3.8,
        [
            "单 seed，无种子方差",
            f"验证 {metrics['scratch']['run']['val_n']} 张，每类 {metrics['scratch']['run']['val_per_class']} 张",
            "strong 是三因素 bundle，不做单因素归因",
        ],
        18,
    )
    add_text(
        slide,
        0.7,
        6.35,
        12.0,
        0.6,
        "Q & A",
        32,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
