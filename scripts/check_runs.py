"""产物契约校验：逐 run 断言与四组齐门禁（阶段 A）。

只遍历 CANONICAL_MODES；runs/diag 与 summary.json 默认跳过
（summary.json 仅在 --assert-complete 时校验）。

`--relaxed` 只跳过 5 条数量约束：test_n==6149、train_n+val_n==2040、
num_classes==102、per_class 行数、CM 102×102；其余一致性断言不放松。
"""

import argparse
import csv
import json
import re
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

from presets import AUG_OPS, PRESETS, resolve
from schema import (
    CANONICAL_MODES,
    CSV_HEADERS,
    METRICS_FIELDS,
    META_SHA256,
    NUM_CLASSES,
    SCHEMA_VERSION,
    SPLIT_FINGERPRINT,
    SUMMARY_FIELDS,
    TEST_N,
    TOL_CSV,
    TOL_JSON,
    TRAIN_N,
    VAL_N,
)

REQUIRED_ARTIFACTS = (
    "history.csv",
    "metrics.json",
    "per_class.csv",
    "confusion_matrix.csv",
    "misclassified.csv",
    "top_confusions.csv",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")

SUMMARY_MODE_STATUSES = ("running", "ok", "failed", "skipped")


def _int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _bool(value):
    return isinstance(value, bool)


def _num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _str(value):
    return isinstance(value, str)


def _close(left, right, tol=TOL_JSON):
    return _num(left) and _num(right) and abs(float(left) - float(right)) <= tol


def _iso(value):
    if not _str(value):
        return False
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


def iter_leaves(obj, prefix=""):
    for key, value in obj.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from iter_leaves(value, path + ".")
        else:
            yield path, value


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _per_class_expected(num_classes):
    if num_classes == NUM_CLASSES:
        return CSV_HEADERS["per_class"]
    return ("class_idx", "label", "n_total", "n_correct", "accuracy", "mean_true_prob")


def _expected_cm_header(num_classes):
    if num_classes == NUM_CLASSES:
        return CSV_HEADERS["confusion_matrix"]
    return ("true_label",) + tuple(str(i) for i in range(num_classes))


def _f1_from_cm(cm):
    size = len(cm)
    f1s = []
    supports = []
    for idx in range(size):
        tp = cm[idx][idx]
        row = sum(cm[idx])
        col = sum(cm[row_idx][idx] for row_idx in range(size))
        precision = tp / col if col else 0.0
        recall = tp / row if row else 0.0
        if precision + recall > 0:
            f1s.append(2 * precision * recall / (precision + recall))
        else:
            f1s.append(0.0)
        supports.append(row)
    total_support = sum(supports)
    macro = sum(f1s) / size if size else 0.0
    weighted = (
        sum(f1 * support for f1, support in zip(f1s, supports)) / total_support
        if total_support
        else 0.0
    )
    return macro, weighted


def _expected_top_confusions(cm, true_totals):
    rows = []
    size = len(cm)
    for true_idx in range(size):
        for pred_idx in range(size):
            if true_idx != pred_idx and cm[true_idx][pred_idx] > 0:
                rows.append((true_idx, pred_idx, cm[true_idx][pred_idx], true_totals[true_idx]))
    rows.sort(key=lambda row: (-row[2], row[0], row[1]))
    return rows[:20]


def check_presets_alignment(run_dir, metrics):
    """run 的关键字段必须与 PRESETS + derive 规则逐字段一致（D2 ⑤）。"""
    errors = []
    run = metrics.get("run", {})
    mode = run.get("mode")
    if mode not in CANONICAL_MODES:
        return [f"{run_dir}: run.mode 非 canonical: {mode!r}"]
    expected = resolve(mode, out=f"runs/{mode}", tag=None, smoke=False)
    fields = (
        "mode",
        "tag",
        "is_diag",
        "base",
        "weights_init",
        "epochs_planned",
        "unfreeze_epoch",
        "bn_stage1",
        "bn_stage2",
        "bn_mode",
        "lr_head",
        "lr_backbone",
        "label_smoothing",
        "aug",
    )
    for field in fields:
        got = run.get(field)
        want = expected[field]
        if isinstance(want, float):
            if not _close(got, want, 1e-12):
                errors.append(f"{run_dir}: {field}={got!r} != 预设 {want!r}")
        elif got != want:
            errors.append(f"{run_dir}: {field}={got!r} != 预设 {want!r}")
    return errors


def check_env(run_dir, metrics):
    errors = []
    run = metrics.get("run", {})
    device = run.get("device")
    cuda = run.get("cuda")
    gpu = run.get("gpu")
    if device not in ("cuda", "cpu"):
        errors.append(f"{run_dir}: device 非法: {device!r}")
    elif device == "cpu":
        if cuda != "":
            errors.append(f"{run_dir}: CPU 下 cuda 必须为空串，得到 {cuda!r}")
        if gpu != "":
            errors.append(f"{run_dir}: CPU 下 gpu 必须为空串，得到 {gpu!r}")
    else:
        if not _str(cuda) or not cuda:
            errors.append(f"{run_dir}: CUDA 下 cuda 必须非空")
        if not _str(gpu) or not gpu:
            errors.append(f"{run_dir}: CUDA 下 gpu 必须非空")
    for field in ("python", "torch", "torchvision"):
        if not _str(run.get(field)) or not run.get(field):
            errors.append(f"{run_dir}: {field} 必须为非空字符串")
    warnings = run.get("determinism_warnings")
    if not isinstance(warnings, list) or not all(_str(item) for item in warnings):
        errors.append(f"{run_dir}: determinism_warnings 必须是 list[str]")
    else:
        expected = "best_effort" if warnings else "strict"
        if run.get("reproducibility") != expected:
            errors.append(
                f"{run_dir}: reproducibility={run.get('reproducibility')!r} "
                f"与 warnings 数量不一致（期望 {expected}）"
            )
    return errors


def check_run(run_dir, relaxed=False, check_presets=False, check_env_fields=False):
    """对单个 run 目录做全部逐 run 断言，返回错误列表（空=通过）。"""
    run_dir = Path(run_dir)
    errors = []

    for name in REQUIRED_ARTIFACTS:
        if not (run_dir / name).is_file():
            errors.append(f"{run_dir}: 缺少产物 {name}")
    if errors:
        return errors

    try:
        metrics = _read_json(run_dir / "metrics.json")
    except (json.JSONDecodeError, OSError) as exc:
        return [f"{run_dir}: metrics.json 无法解析: {exc}"]
    if not isinstance(metrics, dict):
        return [f"{run_dir}: metrics.json 顶层必须是对象"]

    flat = dict(iter_leaves(metrics))
    missing = sorted(set(METRICS_FIELDS) - set(flat))
    extra = sorted(set(flat) - set(METRICS_FIELDS))
    if missing:
        errors.append(f"{run_dir}: metrics 缺字段 {missing}")
    if extra:
        errors.append(f"{run_dir}: metrics 多字段 {extra}")
    if errors:
        return errors

    run = metrics["run"]
    selection = metrics["selection"]
    val = metrics["val"]
    test = metrics["test"]

    def err(message):
        errors.append(f"{run_dir}: {message}")

    # ---- 基础类型 / 枚举 / 常量 ----
    if not _int(metrics["schema_version"]) or metrics["schema_version"] != SCHEMA_VERSION:
        err(f"schema_version != {SCHEMA_VERSION}")
    if not _str(run["mode"]) or run["mode"] not in CANONICAL_MODES:
        err(f"run.mode 非法: {run['mode']!r}")
    if run["mode"] != run_dir.name:
        err(f"run.mode={run['mode']!r} 与目录名 {run_dir.name!r} 不一致")
    if not _str(run["tag"]) or not run["tag"]:
        err("run.tag 必须为非空字符串")
    if not _bool(run["is_diag"]):
        err("run.is_diag 必须是 bool")
    if run["is_diag"]:
        if run["tag"] == run["mode"]:
            err("诊断 run 的 tag 不得等于 mode")
    elif run["tag"] != run["mode"]:
        err("预设 run 的 tag 必须等于 mode")
    if run["base"] not in (None, "two-stage"):
        err(f"run.base 非法: {run['base']!r}")
    if not _int(run["seed"]):
        err("run.seed 必须是 int")
    if run["split_seed"] != 0:
        err("run.split_seed 必须为 0")
    if run["val_per_class"] != 3:
        err("run.val_per_class 必须为 3")
    if run["weights_init"] not in ("imagenet1k_v1", "none"):
        err(f"run.weights_init 非法: {run['weights_init']!r}")
    if not _int(run["epochs_planned"]) or run["epochs_planned"] <= 0:
        err("run.epochs_planned 必须是正整数")
    if not _int(run["epochs_run"]) or run["epochs_run"] <= 0:
        err("run.epochs_run 必须是正整数")
    elif run["epochs_run"] > run["epochs_planned"]:
        err("run.epochs_run 不得超过 epochs_planned")
    if run["batch"] != 64:
        err("run.batch 必须为 64")
    if run["optimizer"] != "adamw":
        err("run.optimizer 必须为 adamw")
    if not _close(run["weight_decay"], 0.01):
        err("run.weight_decay 必须为 0.01")
    if run["wd_no_decay"] != "bn_and_bias":
        err("run.wd_no_decay 必须为 bn_and_bias")
    if run["scheduler"] != "cosine":
        err("run.scheduler 必须为 cosine")
    if not _num(run["lr_head"]) or run["lr_head"] <= 0:
        err("run.lr_head 必须为正数")
    if not _num(run["lr_backbone"]) or run["lr_backbone"] < 0:
        err("run.lr_backbone 必须非负")
    if not _num(run["label_smoothing"]):
        err("run.label_smoothing 必须是数值")
    if not _str(run["aug"]) or run["aug"] not in AUG_OPS:
        err(f"run.aug 非法: {run['aug']!r}")
    elif run["aug_ops"] != list(AUG_OPS[run["aug"]]):
        err("run.aug_ops 与 AUG_OPS[aug] 不一致")
    if run["bn_stage1"] not in ("frozen", "train"):
        err(f"run.bn_stage1 非法: {run['bn_stage1']!r}")
    if run["bn_stage2"] not in (None, "frozen", "train"):
        err(f"run.bn_stage2 非法: {run['bn_stage2']!r}")
    if run["bn_mode"] not in ("frozen", "train", "frozen_to_train"):
        err(f"run.bn_mode 非法: {run['bn_mode']!r}")
    for field in ("params_total", "params_trainable_stage1", "params_trainable_peak"):
        if not _int(run[field]) or run[field] <= 0:
            err(f"run.{field} 必须是正整数")
    if (
        _int(run["params_total"])
        and _int(run["params_trainable_stage1"])
        and _int(run["params_trainable_peak"])
        and not (run["params_trainable_stage1"] <= run["params_trainable_peak"] <= run["params_total"])
    ):
        err("params 关系不满足 stage1 <= peak <= total")
    for field in ("train_n", "val_n", "test_n", "num_classes"):
        if not _int(run[field]) or run[field] <= 0:
            err(f"run.{field} 必须是正整数")
    if (
        not _int(run["test_n"])
        or run["test_n"] <= 0
        or not _int(run["num_classes"])
        or run["num_classes"] <= 0
    ):
        # 关键数量字段不可用时，后续行/列推导无法安全进行。
        return errors
    if run["split_fingerprint"] != SPLIT_FINGERPRINT:
        err("run.split_fingerprint 与契约常量不一致")
    if run["meta_sha256"] != META_SHA256:
        err("run.meta_sha256 与契约常量不一致")
    if not _str(run["source_sha256"]) or not _HEX64.match(run["source_sha256"]):
        err("run.source_sha256 必须是 64 位小写 hex")
    if run["git_commit"] is not None and not _str(run["git_commit"]):
        err("run.git_commit 必须是 str 或 null")
    if run["git_dirty"] is not None and not _bool(run["git_dirty"]):
        err("run.git_dirty 必须是 bool 或 null")
    if run["provenance"] not in ("git", "embedded"):
        err(f"run.provenance 非法: {run['provenance']!r}")
    for field in ("python", "torch", "torchvision", "cuda", "gpu"):
        if not _str(run[field]):
            err(f"run.{field} 必须是字符串")
    if run["device"] not in ("cuda", "cpu"):
        err(f"run.device 非法: {run['device']!r}")
    if run["reproducibility"] not in ("strict", "best_effort"):
        err(f"run.reproducibility 非法: {run['reproducibility']!r}")
    if run["cudnn_deterministic"] is not True:
        err("run.cudnn_deterministic 必须为 True")
    if run["cudnn_benchmark"] is not False:
        err("run.cudnn_benchmark 必须为 False")
    if run["deterministic_algorithms"] != "warn_only":
        err("run.deterministic_algorithms 必须为 warn_only")
    if run["cublas_workspace_config"] != ":4096:8":
        err("run.cublas_workspace_config 必须为 ':4096:8'")
    if not isinstance(run["determinism_warnings"], list) or not all(
        _str(item) for item in run["determinism_warnings"]
    ):
        err("run.determinism_warnings 必须是 list[str]")
    if not _bool(run["smoke"]):
        err("run.smoke 必须是 bool")
    if not _iso(run["started_at"]):
        err("run.started_at 必须是 ISO8601 字符串")
    if not _num(run["elapsed_sec"]) or run["elapsed_sec"] < 0:
        err("run.elapsed_sec 必须非负")
    if run["test_eval_count"] != 1:
        err("run.test_eval_count 必须为 1")

    # ---- derive 关系 ----
    preset = PRESETS.get(run["mode"]) if _str(run["mode"]) else None
    if preset is not None:
        stage1 = preset["stage1_epochs"]
        stage2 = preset["stage2_epochs"]
        if run["epochs_planned"] != stage1 + stage2:
            err("run.epochs_planned != stage1_epochs + stage2_epochs")
        expected_unfreeze = stage1 + 1 if stage1 > 0 else None
        if run["unfreeze_epoch"] != expected_unfreeze:
            err("run.unfreeze_epoch 与派生公式不一致")
        expected_bn = (
            "frozen_to_train"
            if preset["bn_stage2"] is not None
            else preset["bn_stage1"]
        )
        if run["bn_mode"] != expected_bn:
            err("run.bn_mode 与派生公式不一致")

    # ---- 数量约束（--relaxed 只跳过这 5 条） ----
    num_classes = run["num_classes"]
    if not relaxed:
        if run["test_n"] != TEST_N:
            err(f"test_n={run['test_n']} != {TEST_N}")
        if run["train_n"] + run["val_n"] != TRAIN_N + VAL_N:
            err(f"train_n+val_n={run['train_n'] + run['val_n']} != {TRAIN_N + VAL_N}")
        if num_classes != NUM_CLASSES:
            err(f"num_classes={num_classes} != {NUM_CLASSES}")

    # ---- selection / val 关系 ----
    if selection["criterion"] != "val_top1":
        err("selection.criterion 必须为 val_top1")
    if not _int(selection["best_epoch"]) or selection["best_epoch"] < 1:
        err("selection.best_epoch 必须是 1-based 正整数")
    for field in ("loss", "top1", "top3", "top5", "macro_f1"):
        if not _num(val[field]):
            err(f"val.{field} 必须是数值")
        if not _num(test[field]):
            err(f"test.{field} 必须是数值")
    if not _num(test["weighted_f1"]):
        err("test.weighted_f1 必须是数值")
    if not _int(test["checkpoint_epoch"]):
        err("test.checkpoint_epoch 必须是 int")
    if not isinstance(test["per_class_acc"], list) or not all(
        _num(item) for item in test["per_class_acc"]
    ):
        err("test.per_class_acc 必须是 list[float]")

    # ---- history.csv ----
    header, rows = _read_csv(run_dir / "history.csv")
    if tuple(header) != CSV_HEADERS["history"]:
        err(f"history.csv 表头不符: {header}")
        hist = []
    else:
        hist = []
        for row in rows:
            if len(row) != len(CSV_HEADERS["history"]):
                err(f"history.csv 行列数错误: {row}")
                continue
            epoch = _parse_int(row[0])
            values = {
                "epoch": epoch,
                "train_loss": _parse_float(row[1]),
                "train_acc": _parse_float(row[2]),
                "val_loss": _parse_float(row[3]),
                "val_acc": _parse_float(row[4]),
                "phase": row[5],
                "lr_head": _parse_float(row[6]),
                "lr_backbone": _parse_float(row[7]),
                "sec": _parse_float(row[8]),
            }
            if None in (
                values["epoch"],
                values["train_loss"],
                values["train_acc"],
                values["val_loss"],
                values["val_acc"],
                values["lr_head"],
                values["lr_backbone"],
                values["sec"],
            ):
                err(f"history.csv 存在无法解析的数值行: {row}")
                continue
            hist.append(values)
        if len(hist) != run["epochs_run"]:
            err(f"history.csv 行数 {len(hist)} != epochs_run {run['epochs_run']}")
        for index, item in enumerate(hist, start=1):
            if item["epoch"] != index:
                err(f"history.csv epoch 不连续: 第 {index} 行为 {item['epoch']}")
                break
        for item in hist:
            if item["phase"] not in ("scratch", "frozen", "unfrozen"):
                err(f"history.csv phase 非法: {item['phase']!r}")
                break
        if hist:
            val_accs = [item["val_acc"] for item in hist]
            max_acc = max(val_accs)
            best_index = val_accs.index(max_acc) + 1
            if selection["best_epoch"] != best_index:
                err(
                    f"selection.best_epoch={selection['best_epoch']} "
                    f"!= argmax(history.val_acc)={best_index}"
                )
            if not _close(val["top1"], max_acc, TOL_CSV):
                err(f"val.top1={val['top1']} 与 max(history.val_acc)={max_acc} 不一致")
            if not _close(selection["best_val_top1"], val["top1"], TOL_JSON):
                err("selection.best_val_top1 必须等于 val.top1")
            best_row = hist[best_index - 1] if best_index - 1 < len(hist) else None
            if best_row is not None:
                if not _close(val["loss"], best_row["val_loss"], TOL_CSV):
                    err("val.loss 与 best epoch 的 history.val_loss 不一致")
                if not _close(val["top1"], best_row["val_acc"], TOL_CSV):
                    err("val.top1 与 best epoch 的 history.val_acc 不一致")
            preset = PRESETS.get(run["mode"]) if _str(run["mode"]) else None
            if preset is not None:
                if preset["stage1_epochs"] > 0:
                    pattern = ["frozen"] * preset["stage1_epochs"] + [
                        "unfrozen"
                    ] * preset["stage2_epochs"]
                else:
                    phase = "scratch" if preset["lr_backbone"] > 0 else "frozen"
                    pattern = [phase] * (
                        preset["stage1_epochs"] + preset["stage2_epochs"]
                    )
                actual = [item["phase"] for item in hist]
                if actual != pattern[: len(actual)]:
                    err("history.csv phase 序列与 PRESETS 阶段划分不一致")
            for item in hist:
                if item["lr_head"] < 0 or item["lr_backbone"] < 0 or item["sec"] < 0:
                    err("history.csv 存在负的 lr/sec")
                    break

    if test["checkpoint_epoch"] != selection["best_epoch"]:
        err("test.checkpoint_epoch != selection.best_epoch")

    # ---- per_class.csv ----
    header, rows = _read_csv(run_dir / "per_class.csv")
    if tuple(header) != _per_class_expected(num_classes):
        err(f"per_class.csv 表头不符: {header}")
        per_class = []
    else:
        per_class = []
        for row in rows:
            if len(row) != 6:
                err(f"per_class.csv 行列数错误: {row}")
                continue
            class_idx = _parse_int(row[0])
            label = _parse_int(row[1])
            n_total = _parse_int(row[2])
            n_correct = _parse_int(row[3])
            accuracy = _parse_float(row[4])
            mean_true_prob = _parse_float(row[5])
            if None in (class_idx, label, n_total, n_correct, accuracy, mean_true_prob):
                err(f"per_class.csv 存在无法解析的行: {row}")
                continue
            per_class.append(
                {
                    "class_idx": class_idx,
                    "label": label,
                    "n_total": n_total,
                    "n_correct": n_correct,
                    "accuracy": accuracy,
                    "mean_true_prob": mean_true_prob,
                }
            )
        if not relaxed and len(per_class) != NUM_CLASSES:
            err(f"per_class.csv 行数 {len(per_class)} != {NUM_CLASSES}")
        if len(per_class) != num_classes:
            err(f"per_class.csv 行数 {len(per_class)} != num_classes {num_classes}")
        for index, item in enumerate(per_class):
            if item["class_idx"] != index:
                err(f"per_class.csv class_idx 不连续: 第 {index} 行为 {item['class_idx']}")
                break
            if item["label"] != item["class_idx"] + 1:
                err(f"per_class.csv label != class_idx+1（class_idx={item['class_idx']}）")
            if item["n_total"] < 0 or item["n_correct"] < 0:
                err("per_class.csv 存在负计数")
            expected_acc = (
                item["n_correct"] / item["n_total"] if item["n_total"] > 0 else 0.0
            )
            if not _close(item["accuracy"], expected_acc, TOL_CSV):
                err(
                    f"per_class.csv accuracy={item['accuracy']} "
                    f"!= n_correct/n_total={expected_acc}"
                )
            if not (0.0 <= item["mean_true_prob"] <= 1.0):
                err("per_class.csv mean_true_prob 越界")

    # ---- confusion_matrix.csv ----
    header, rows = _read_csv(run_dir / "confusion_matrix.csv")
    if tuple(header) != _expected_cm_header(num_classes):
        err(f"confusion_matrix.csv 表头不符（期望 {num_classes + 1} 列）")
        cm = []
    else:
        cm = []
        for row in rows:
            if len(row) != num_classes + 1:
                err(f"confusion_matrix.csv 行列数错误: {len(row)}")
                continue
            values = [_parse_int(cell) for cell in row]
            if any(value is None for value in values):
                err("confusion_matrix.csv 存在非整数单元格")
                continue
            if values[0] != len(cm):
                err("confusion_matrix.csv true_label 必须按 0..N-1 排列")
            if any(value < 0 for value in values[1:]):
                err("confusion_matrix.csv 存在负计数")
            cm.append(values[1:])
        if len(cm) != num_classes:
            err(f"confusion_matrix.csv 行数 {len(cm)} != num_classes {num_classes}")
        if cm:
            if sum(sum(row) for row in cm) != run["test_n"]:
                err("sum(CM) != test_n")
            if per_class:
                for index, row in enumerate(cm):
                    if index < len(per_class) and sum(row) != per_class[index]["n_total"]:
                        err(f"CM 行和 != per_class.n_total（class {index}）")
                        break
            total_correct = sum(cm[i][i] for i in range(len(cm)))
            if not _close(total_correct / run["test_n"], test["top1"], TOL_JSON):
                err("sum(n_correct)/test_n != test.top1")

    # ---- misclassified.csv ----
    header, rows = _read_csv(run_dir / "misclassified.csv")
    misclassified = []
    if tuple(header) != CSV_HEADERS["misclassified"]:
        err(f"misclassified.csv 表头不符: {header}")
    else:
        for row in rows:
            if len(row) != 5:
                err(f"misclassified.csv 行列数错误: {row}")
                continue
            true_idx = _parse_int(row[1])
            pred_idx = _parse_int(row[2])
            true_prob = _parse_float(row[3])
            pred_prob = _parse_float(row[4])
            if None in (true_idx, pred_idx, true_prob, pred_prob):
                err(f"misclassified.csv 存在无法解析的行: {row}")
                continue
            if not (0 <= true_idx < num_classes and 0 <= pred_idx < num_classes):
                err("misclassified.csv 索引越界")
            if true_idx == pred_idx:
                err("misclassified.csv 不得包含正确分类样本")
            misclassified.append(
                {
                    "image_id": row[0],
                    "true_idx": true_idx,
                    "pred_idx": pred_idx,
                    "true_prob": true_prob,
                    "pred_prob": pred_prob,
                }
            )
        if not _close(1.0 - test["top1"], len(misclassified) / run["test_n"], TOL_JSON):
            err("1 - test.top1 != len(misclassified)/test_n")
        keys = [(item["true_prob"], item["image_id"]) for item in misclassified]
        if keys != sorted(keys):
            err("misclassified.csv 必须按 (true_prob, image_id) 升序")
        if cm:
            expected_confusions = Counter()
            for true_idx, row in enumerate(cm):
                for pred_idx, count in enumerate(row):
                    if true_idx != pred_idx and count > 0:
                        expected_confusions[(true_idx, pred_idx)] += count
            actual_confusions = Counter(
                (item["true_idx"], item["pred_idx"]) for item in misclassified
            )
            if actual_confusions != expected_confusions:
                err("misclassified.csv 的 (true,pred) 计数与 CM 非对角元素不一致")

    # ---- top_confusions.csv ----
    header, rows = _read_csv(run_dir / "top_confusions.csv")
    if tuple(header) != CSV_HEADERS["top_confusions"]:
        err(f"top_confusions.csv 表头不符: {header}")
    else:
        parsed = []
        for row in rows:
            if len(row) != 4:
                err(f"top_confusions.csv 行列数错误: {row}")
                continue
            values = [_parse_int(cell) for cell in row]
            if any(value is None for value in values):
                err("top_confusions.csv 存在非整数单元格")
                continue
            parsed.append(tuple(values))
        if len(parsed) > 20:
            err("top_confusions.csv 超过 20 行")
        if parsed != sorted(parsed, key=lambda item: (-item[2], item[0], item[1])):
            err("top_confusions.csv 排序键不是 (-count, true_idx, pred_idx)")
        if any(item[0] == item[1] for item in parsed):
            err("top_confusions.csv 不得包含对角元素")
        if per_class:
            for true_idx, pred_idx, count, true_total in parsed:
                if not (0 <= true_idx < len(per_class) and 0 <= pred_idx < len(per_class)):
                    err("top_confusions.csv 索引越界")
                    continue
                if count <= 0:
                    err("top_confusions.csv count 必须为正")
                if true_total != per_class[true_idx]["n_total"]:
                    err("top_confusions.csv true_total != per_class.n_total")
            if cm and len(per_class) == len(cm):
                expected_top = _expected_top_confusions(
                    cm, [item["n_total"] for item in per_class]
                )
                if parsed != expected_top:
                    err("top_confusions.csv 与 CM 推导的前 20 项不一致")
        elif cm:
            err("无法校验 top_confusions.csv（per_class 不可用）")

    # ---- 逐类准确率 ----
    if per_class and isinstance(test["per_class_acc"], list):
        if len(test["per_class_acc"]) != num_classes:
            err("test.per_class_acc 长度 != num_classes")
        else:
            for index, accuracy in enumerate(test["per_class_acc"]):
                if index < len(per_class) and not _close(
                    accuracy, per_class[index]["accuracy"], TOL_CSV
                ):
                    err(f"test.per_class_acc[{index}] 与 per_class.accuracy 不一致")
                    break

    if check_presets:
        errors.extend(check_presets_alignment(run_dir, metrics))
    if check_env_fields:
        errors.extend(check_env(run_dir, metrics))
    return errors


def _check_summary(root):
    errors = []
    path = Path(root) / "summary.json"
    if not path.is_file():
        return [f"{path}: 缺少 summary.json"]
    try:
        summary = _read_json(path)
    except (json.JSONDecodeError, OSError) as exc:
        return [f"{path}: 无法解析: {exc}"]
    if not isinstance(summary, dict):
        return [f"{path}: 顶层必须是对象"]
    if set(summary) != set(SUMMARY_FIELDS):
        errors.append(
            f"{path}: 字段集不符，缺 {sorted(set(SUMMARY_FIELDS) - set(summary))}，"
            f"多 {sorted(set(summary) - set(SUMMARY_FIELDS))}"
        )
    if not _str(summary.get("kernel_ref")) or not summary.get("kernel_ref"):
        errors.append(f"{path}: kernel_ref 必须为非空字符串")
    if not _str(summary.get("built_source_sha256")) or not _HEX64.match(
        summary.get("built_source_sha256", "")
    ):
        errors.append(f"{path}: built_source_sha256 必须是 64 位小写 hex")
    for field in ("started_at", "finished_at"):
        if not _iso(summary.get(field)):
            errors.append(f"{path}: {field} 必须是 ISO8601 字符串")
    if not _str(summary.get("gpu")):
        errors.append(f"{path}: gpu 必须是字符串")
    modes = summary.get("modes")
    if not isinstance(modes, list) or len(modes) != 4:
        errors.append(f"{path}: modes 必须是长度为 4 的 list")
        return errors
    if [item.get("mode") if isinstance(item, dict) else None for item in modes] != list(
        CANONICAL_MODES
    ):
        errors.append(f"{path}: modes 顺序必须等于 CANONICAL_MODES")
    for index, item in enumerate(modes):
        if not isinstance(item, dict):
            errors.append(f"{path}: modes[{index}] 必须是对象")
            continue
        if set(item) != {"mode", "status", "elapsed_sec", "error"}:
            errors.append(f"{path}: modes[{index}] 键集不符")
            continue
        if item["mode"] not in CANONICAL_MODES:
            errors.append(f"{path}: modes[{index}].mode 非 canonical")
        if item["status"] not in SUMMARY_MODE_STATUSES:
            errors.append(f"{path}: modes[{index}].status 非法: {item['status']!r}")
        elif item["status"] in ("failed", "running"):
            errors.append(f"{path}: modes[{index}].status={item['status']} 阻止交付")
        if not _num(item["elapsed_sec"]) or item["elapsed_sec"] < 0:
            errors.append(f"{path}: modes[{index}].elapsed_sec 必须非负")
        if item["error"] is not None and not _str(item["error"]):
            errors.append(f"{path}: modes[{index}].error 必须是 str 或 null")
    return errors


def assert_complete(root, relaxed=False):
    """D2 --assert-complete 五条件。"""
    root = Path(root)
    errors = []
    if not root.is_dir():
        return [f"{root}: runs 根目录不存在"]

    dirs = sorted(
        path.name for path in root.iterdir() if path.is_dir() and path.name != "diag"
    )
    if dirs != sorted(CANONICAL_MODES):
        errors.append(
            f"{root}: 必须恰好包含 4 个 canonical 目录 {sorted(CANONICAL_MODES)}，"
            f"实际 {dirs}"
        )

    for mode in CANONICAL_MODES:
        run_dir = root / mode
        if not run_dir.is_dir():
            errors.append(f"{run_dir}: 目录缺失")
            continue
        errors.extend(
            check_run(run_dir, relaxed=relaxed, check_presets=True, check_env_fields=True)
        )

    errors.extend(_check_summary(root))

    fingerprints = set()
    meta_digests = set()
    for mode in CANONICAL_MODES:
        path = root / mode / "metrics.json"
        if not path.is_file():
            continue
        try:
            metrics = _read_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        run = metrics.get("run", {})
        fingerprints.add(run.get("split_fingerprint"))
        meta_digests.add(run.get("meta_sha256"))
    if fingerprints != {SPLIT_FINGERPRINT}:
        errors.append(f"{root}: 四份 split_fingerprint 不唯一/不等于契约常量: {fingerprints}")
    if meta_digests != {META_SHA256}:
        errors.append(f"{root}: 四份 meta_sha256 不唯一/不等于契约常量: {meta_digests}")

    return errors


# --------------------------------------------------------------------------
# selftest：在 /tmp/opencode 下生成满足契约的合成 run，再验证三类破坏被拒。
# --------------------------------------------------------------------------


def _write_csv(path, header, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _fixture_dirs():
    base = Path("/tmp/opencode")
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="check_runs_selftest_", dir=str(base)))


def _fixture_root():
    root = _fixture_dirs()
    nclasses = 3
    test_n = 12
    cm = [
        [3, 1, 0],
        [1, 1, 2],
        [1, 3, 0],
    ]
    n_total = [sum(row) for row in cm]
    n_correct = [cm[i][i] for i in range(nclasses)]
    accuracies = [
        n_correct[i] / n_total[i] if n_total[i] else 0.0 for i in range(nclasses)
    ]
    mean_true_prob = [0.6, 0.5, 0.4]
    top1 = sum(n_correct) / test_n
    best_epoch = 3
    best_val_acc = 0.5

    for mode in CANONICAL_MODES:
        cfg = resolve(mode, out=f"runs/{mode}", tag=None, smoke=False)
        out = root / mode
        out.mkdir(parents=True, exist_ok=True)
        epochs = cfg["epochs_planned"]

        history_rows = []
        for epoch in range(1, epochs + 1):
            if cfg["stage1_epochs"] > 0:
                phase = "frozen" if epoch <= cfg["stage1_epochs"] else "unfrozen"
            else:
                phase = "scratch" if cfg["lr_backbone"] > 0 else "frozen"
            val_acc = best_val_acc if epoch == best_epoch else 0.25
            history_rows.append(
                [
                    epoch,
                    f"{1.0 - 0.01 * epoch:.6f}",
                    f"{0.2 + 0.005 * epoch:.6f}",
                    f"{2.0 - 0.05 * epoch:.6f}",
                    f"{val_acc:.6f}",
                    phase,
                    f"{cfg['lr_head']:.6f}",
                    f"{cfg['lr_backbone']:.6f}",
                    f"{0.5:.6f}",
                ]
            )
        _write_csv(out / "history.csv", CSV_HEADERS["history"], history_rows)

        per_class_rows = []
        for index in range(nclasses):
            per_class_rows.append(
                [
                    index,
                    index + 1,
                    n_total[index],
                    n_correct[index],
                    f"{accuracies[index]:.6f}",
                    f"{mean_true_prob[index]:.6f}",
                ]
            )
        _write_csv(out / "per_class.csv", _per_class_expected(nclasses), per_class_rows)

        cm_rows = []
        for index, row in enumerate(cm):
            cm_rows.append([index] + list(row))
        _write_csv(
            out / "confusion_matrix.csv", _expected_cm_header(nclasses), cm_rows
        )

        confusion_rows = []
        for true_idx in range(nclasses):
            for pred_idx in range(nclasses):
                if true_idx != pred_idx:
                    for _ in range(cm[true_idx][pred_idx]):
                        confusion_rows.append((true_idx, pred_idx))
        misclassified_rows = []
        for offset, (true_idx, pred_idx) in enumerate(confusion_rows):
            true_prob = 0.05 + 0.01 * offset
            misclassified_rows.append(
                [
                    f"image_{offset + 1:05d}.jpg",
                    true_idx,
                    pred_idx,
                    f"{true_prob:.6f}",
                    f"{1.0 - true_prob:.6f}",
                ]
            )
        misclassified_rows.sort(key=lambda row: (float(row[3]), row[0]))
        _write_csv(
            out / "misclassified.csv",
            CSV_HEADERS["misclassified"],
            misclassified_rows,
        )

        top_rows = _expected_top_confusions(cm, n_total)
        _write_csv(
            out / "top_confusions.csv",
            CSV_HEADERS["top_confusions"],
            [list(row) for row in top_rows],
        )

        macro, weighted = _f1_from_cm(cm)
        metrics = {
            "schema_version": SCHEMA_VERSION,
            "run": {
                "mode": mode,
                "tag": cfg["tag"],
                "is_diag": cfg["is_diag"],
                "base": cfg["base"],
                "seed": 0,
                "split_seed": 0,
                "val_per_class": 3,
                "weights_init": cfg["weights_init"],
                "epochs_planned": cfg["epochs_planned"],
                "epochs_run": epochs,
                "batch": 64,
                "optimizer": "adamw",
                "weight_decay": 0.01,
                "wd_no_decay": "bn_and_bias",
                "scheduler": "cosine",
                "lr_head": cfg["lr_head"],
                "lr_backbone": cfg["lr_backbone"],
                "label_smoothing": cfg["label_smoothing"],
                "aug": cfg["aug"],
                "aug_ops": cfg["aug_ops"],
                "bn_stage1": cfg["bn_stage1"],
                "bn_stage2": cfg["bn_stage2"],
                "bn_mode": cfg["bn_mode"],
                "unfreeze_epoch": cfg["unfreeze_epoch"],
                "params_total": 1000,
                "params_trainable_stage1": 100,
                "params_trainable_peak": 1000
                if cfg["unfreeze_epoch"] is not None
                else 100,
                "train_n": 30,
                "val_n": 6,
                "test_n": test_n,
                "num_classes": nclasses,
                "split_fingerprint": SPLIT_FINGERPRINT,
                "meta_sha256": META_SHA256,
                "source_sha256": "a" * 64,
                "git_commit": "b" * 40,
                "git_dirty": True,
                "provenance": "git",
                "python": "3.14.8",
                "torch": "2.6.0",
                "torchvision": "0.21.0",
                "cuda": "",
                "gpu": "",
                "device": "cpu",
                "reproducibility": "strict",
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "deterministic_algorithms": "warn_only",
                "cublas_workspace_config": ":4096:8",
                "determinism_warnings": [],
                "smoke": False,
                "started_at": "2026-10-31T12:00:00+00:00",
                "elapsed_sec": 1.0,
                "test_eval_count": 1,
            },
            "selection": {
                "criterion": "val_top1",
                "best_epoch": best_epoch,
                "best_val_top1": best_val_acc,
            },
            "val": {
                "loss": 1.85,
                "top1": best_val_acc,
                "top3": 0.75,
                "top5": 1.0,
                "macro_f1": 0.4,
            },
            "test": {
                "loss": 1.2,
                "top1": top1,
                "top3": 0.75,
                "top5": 1.0,
                "macro_f1": macro,
                "weighted_f1": weighted,
                "checkpoint_epoch": best_epoch,
                "per_class_acc": accuracies,
            },
        }
        with (out / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
            handle.write("\n")

    summary = {
        "kernel_ref": "feisama/flower-102-transfer",
        "built_source_sha256": "c" * 64,
        "started_at": "2026-10-31T12:00:00+00:00",
        "finished_at": "2026-10-31T12:30:00+00:00",
        "gpu": "Tesla T4",
        "modes": [
            {"mode": mode, "status": "ok", "elapsed_sec": 1.0, "error": None}
            for mode in CANONICAL_MODES
        ],
    }
    with (root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return root


def _selftest():
    base = _fixture_root()
    problems = []
    try:
        errors = assert_complete(base, relaxed=True)
        if errors:
            problems.append(f"合格 fixture 未被接受: {errors[:3]}")

        broken = _fixture_root()
        try:
            shutil.rmtree(broken / "scratch")
            rejected = assert_complete(broken, relaxed=True)
            if not rejected:
                problems.append("删除 mode 目录未被拒绝")
            print("reject deleted-mode-dir -> ok")
        finally:
            shutil.rmtree(broken, ignore_errors=True)

        broken = _fixture_root()
        try:
            summary_path = broken / "summary.json"
            summary = _read_json(summary_path)
            summary["modes"][1]["status"] = "failed"
            with summary_path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
            rejected = assert_complete(broken, relaxed=True)
            if not rejected:
                problems.append("summary status=failed 未被拒绝")
            print("reject summary-failed -> ok")
        finally:
            shutil.rmtree(broken, ignore_errors=True)

        broken = _fixture_root()
        try:
            metrics_path = broken / "frozen" / "metrics.json"
            metrics = _read_json(metrics_path)
            metrics["run"]["split_fingerprint"] = "d" * 64
            with metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
            rejected = assert_complete(broken, relaxed=True)
            if not rejected:
                problems.append("篡改 fingerprint 未被拒绝")
            print("reject bad-fingerprint -> ok")
        finally:
            shutil.rmtree(broken, ignore_errors=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)

    if problems:
        for item in problems:
            print(f"selftest FAIL: {item}", file=sys.stderr)
        return 1
    print("selftest ok")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="run 产物契约校验")
    parser.add_argument("--root", default="runs")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--assert-complete", action="store_true")
    parser.add_argument("--assert-presets", action="store_true")
    parser.add_argument("--assert-env", action="store_true")
    parser.add_argument("--relaxed", action="store_true")
    parser.add_argument("--dump-schema", action="store_true")
    args = parser.parse_args(argv)

    if args.dump_schema:
        print(f"{len(METRICS_FIELDS)} {len(SUMMARY_FIELDS)} {SCHEMA_VERSION}")
        return 0
    if args.selftest:
        return _selftest()

    root = Path(args.root)
    if args.assert_complete:
        errors = assert_complete(root, relaxed=args.relaxed)
    else:
        errors = []
        present = [mode for mode in CANONICAL_MODES if (root / mode).is_dir()]
        if not present:
            errors.append(f"{root}: 下没有 canonical run 目录")
        for mode in present:
            errors.extend(
                check_run(
                    root / mode,
                    relaxed=args.relaxed,
                    check_presets=args.assert_presets,
                    check_env_fields=args.assert_env,
                )
            )
    if errors:
        for item in errors:
            print(f"check_runs FAIL: {item}", file=sys.stderr)
        return 1
    print("check_runs ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
