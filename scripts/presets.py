"""四组超参预设与 resolve() 校验规则（阶段 A）。

`PRESETS` 是模式差异的唯一来源；`resolve()` 返回全部生效配置（含派生字段），
`train_full.py` 只消费返回字典，不得自行推导任何派生字段。

`--check` 与模块内另写的一份冻结期望表逐字段比对（浮点容差 1e-12），
不是自洽检查。
"""

import argparse
import json
import sys
from pathlib import Path

from schema import CANONICAL_MODES, DIAG_PREFIX, SMOKE_PREFIX

# 与 D4 表逐字一致；派生字段不在此表内。
PRESETS = {
    "scratch": {
        "base": None,
        "weights": "none",
        "stage1_epochs": 0,
        "stage2_epochs": 120,
        "lr_head": 1e-3,
        "lr_backbone": 1e-3,
        "weight_decay": 0.01,
        "batch": 64,
        "aug": "baseline",
        "label_smoothing": 0.0,
        "bn_stage1": "train",
        "bn_stage2": None,
    },
    "frozen": {
        "base": None,
        "weights": "imagenet1k_v1",
        "stage1_epochs": 0,
        "stage2_epochs": 12,
        "lr_head": 1e-3,
        "lr_backbone": 0.0,
        "weight_decay": 0.01,
        "batch": 64,
        "aug": "baseline",
        "label_smoothing": 0.0,
        "bn_stage1": "frozen",
        "bn_stage2": None,
    },
    "two-stage": {
        "base": None,
        "weights": "imagenet1k_v1",
        "stage1_epochs": 5,
        "stage2_epochs": 20,
        "lr_head": 1e-3,
        "lr_backbone": 1e-4,
        "weight_decay": 0.01,
        "batch": 64,
        "aug": "baseline",
        "label_smoothing": 0.0,
        "bn_stage1": "frozen",
        "bn_stage2": "train",
    },
    "strong": {
        "base": "two-stage",
        "weights": "imagenet1k_v1",
        "stage1_epochs": 5,
        "stage2_epochs": 35,
        "lr_head": 1e-3,
        "lr_backbone": 1e-4,
        "weight_decay": 0.01,
        "batch": 64,
        "aug": "strong",
        "label_smoothing": 0.1,
        "bn_stage1": "frozen",
        "bn_stage2": "train",
    },
}

AUG_OPS = {
    "baseline": [
        "RandomResizedCrop(224,scale=(0.6,1.0))",
        "RandomHorizontalFlip()",
    ],
    "strong": [
        "RandomResizedCrop(224,scale=(0.35,1.0))",
        "RandomHorizontalFlip()",
        "RandomRotation(25)",
        "ColorJitter(0.3,0.3,0.3,0.05)",
        "RandomErasing(p=0.25,scale=(0.02,0.15))",
    ],
}

# 冻结期望表：独立手写，用于 --check 的逐字段比对（不引用 PRESETS）。
_EXPECTED = {
    "scratch": {
        "base": None,
        "weights": "none",
        "stage1_epochs": 0,
        "stage2_epochs": 120,
        "lr_head": 1e-3,
        "lr_backbone": 1e-3,
        "weight_decay": 0.01,
        "batch": 64,
        "aug": "baseline",
        "label_smoothing": 0.0,
        "bn_stage1": "train",
        "bn_stage2": None,
    },
    "frozen": {
        "base": None,
        "weights": "imagenet1k_v1",
        "stage1_epochs": 0,
        "stage2_epochs": 12,
        "lr_head": 1e-3,
        "lr_backbone": 0.0,
        "weight_decay": 0.01,
        "batch": 64,
        "aug": "baseline",
        "label_smoothing": 0.0,
        "bn_stage1": "frozen",
        "bn_stage2": None,
    },
    "two-stage": {
        "base": None,
        "weights": "imagenet1k_v1",
        "stage1_epochs": 5,
        "stage2_epochs": 20,
        "lr_head": 1e-3,
        "lr_backbone": 1e-4,
        "weight_decay": 0.01,
        "batch": 64,
        "aug": "baseline",
        "label_smoothing": 0.0,
        "bn_stage1": "frozen",
        "bn_stage2": "train",
    },
    "strong": {
        "base": "two-stage",
        "weights": "imagenet1k_v1",
        "stage1_epochs": 5,
        "stage2_epochs": 35,
        "lr_head": 1e-3,
        "lr_backbone": 1e-4,
        "weight_decay": 0.01,
        "batch": 64,
        "aug": "strong",
        "label_smoothing": 0.1,
        "bn_stage1": "frozen",
        "bn_stage2": "train",
    },
}

_EXPECTED_KEYS = (
    "base",
    "weights",
    "stage1_epochs",
    "stage2_epochs",
    "lr_head",
    "lr_backbone",
    "weight_decay",
    "batch",
    "aug",
    "label_smoothing",
    "bn_stage1",
    "bn_stage2",
)


def _fail(message):
    print(f"presets: {message}", file=sys.stderr)
    raise SystemExit(2)


def _default_out(mode, smoke):
    if smoke:
        return f"{SMOKE_PREFIX}opencode/flower-smoke/{mode}"
    return f"runs/{mode}"


def resolve(mode, out=None, weights=None, tag=None, smoke=False):
    """把 (mode, CLI 覆盖项) 解析为全部生效配置。

    返回字典恰含 D1 规定的 22 个键。强制规则不合一律 SystemExit(2)。
    """
    if mode == "all":
        _fail("resolve() 不接受 mode='all'，请由 main() 循环四个 canonical mode")
    if mode not in CANONICAL_MODES:
        _fail(f"未知 mode: {mode!r}（canonical: {', '.join(CANONICAL_MODES)}）")

    preset = PRESETS[mode]
    if out is None:
        out = _default_out(mode, smoke)
    out_str = str(out)
    if not out_str:
        _fail("out 不能为空")

    weights_init = preset["weights"] if weights is None else weights
    if weights_init not in ("imagenet1k_v1", "none"):
        _fail(f"weights 必须为 imagenet1k_v1 或 none，收到 {weights_init!r}")
    weights_overridden = weights is not None and weights != preset["weights"]

    if smoke:
        resolved = Path(out_str).resolve()
        if resolved == Path("/tmp") or not resolved.is_relative_to(Path("/tmp")):
            _fail(f"smoke run 的 --out 必须是 /tmp 下的子目录（不能恰为 /tmp）: {out_str}")
        if not tag:
            tag = "smoke"
        if out_str.startswith("runs/"):
            _fail(f"smoke run 不得写入 runs/: {out_str}")
    else:
        if weights_overridden:
            if not out_str.startswith(DIAG_PREFIX):
                _fail(
                    "覆盖 --weights 且与预设不同时，--out 必须以 "
                    f"{DIAG_PREFIX} 开头: {out_str}"
                )
            if tag is None:
                tag = Path(out_str).name
            if not tag or tag == mode:
                _fail(f"诊断 run 的 --tag 必须非空且不等于 mode: {tag!r}")
        else:
            if out_str != f"runs/{mode}":
                _fail(f"预设 run 的 --out 必须恰为 runs/{mode}: {out_str}")
            if tag is None:
                tag = mode
            if tag != mode:
                _fail(f"预设 run 的 --tag 必须等于 mode: {tag!r}")

    stage1 = int(preset["stage1_epochs"])
    stage2 = int(preset["stage2_epochs"])
    epochs_planned = stage1 + stage2
    unfreeze_epoch = stage1 + 1 if stage1 > 0 else None
    if preset["bn_stage2"] is not None:
        bn_mode = "frozen_to_train"
    else:
        bn_mode = preset["bn_stage1"]
    if stage1 == 0 and preset["lr_backbone"] > 0:
        params_rule = "all"
    else:
        params_rule = "fc_only"

    return {
        "mode": mode,
        "tag": str(tag),
        "is_diag": (not smoke) and out_str.startswith(DIAG_PREFIX),
        "smoke": bool(smoke),
        "base": preset["base"],
        "weights_init": weights_init,
        "epochs_planned": epochs_planned,
        "stage1_epochs": stage1,
        "stage2_epochs": stage2,
        "unfreeze_epoch": unfreeze_epoch,
        "bn_stage1": preset["bn_stage1"],
        "bn_stage2": preset["bn_stage2"],
        "bn_mode": bn_mode,
        "params_trainable_stage1_rule": params_rule,
        "lr_head": preset["lr_head"],
        "lr_backbone": preset["lr_backbone"],
        "weight_decay": preset["weight_decay"],
        "batch": preset["batch"],
        "aug": preset["aug"],
        "aug_ops": list(AUG_OPS[preset["aug"]]),
        "label_smoothing": preset["label_smoothing"],
        "out": out_str,
    }


def _check():
    problems = []
    if set(_EXPECTED) != set(PRESETS):
        problems.append(f"mode 集合不一致: {sorted(_EXPECTED)} vs {sorted(PRESETS)}")
    for mode in CANONICAL_MODES:
        if mode not in PRESETS or mode not in _EXPECTED:
            continue
        actual = PRESETS[mode]
        expected = _EXPECTED[mode]
        if set(actual) != set(expected):
            problems.append(
                f"{mode}: 键集合不一致 {sorted(actual)} vs {sorted(expected)}"
            )
            continue
        for key in _EXPECTED_KEYS:
            got, want = actual[key], expected[key]
            if isinstance(want, float) or isinstance(got, float):
                if not isinstance(got, (int, float)) or isinstance(got, bool):
                    problems.append(f"{mode}.{key}: 非数值 {got!r}")
                elif abs(float(got) - float(want)) > 1e-12:
                    problems.append(f"{mode}.{key}: {got!r} != {want!r}")
            elif got != want:
                problems.append(f"{mode}.{key}: {got!r} != {want!r}")
    if problems:
        for item in problems:
            print(f"presets check FAIL: {item}", file=sys.stderr)
        return 1
    print("presets check ok")
    return 0


_RESOLVE_FROM_MODE = object()


def _main(argv=None):
    parser = argparse.ArgumentParser(description="四组预设与 resolve()")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--json", action="store_true", help="输出四组 resolve 结果")
    action.add_argument("--check", action="store_true", help="与冻结期望表逐字段比对")
    action.add_argument(
        "--resolve",
        nargs="?",
        const=_RESOLVE_FROM_MODE,
        metavar="MODE",
        help="解析单个 mode（可写 --resolve MODE 或 --resolve --mode MODE）",
    )
    parser.add_argument("--mode", help="与 --resolve 搭配使用的 canonical mode")
    parser.add_argument("--out")
    parser.add_argument("--weights")
    parser.add_argument("--tag")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        return _check()
    if args.json:
        result = {
            mode: resolve(mode, out=args.out, weights=args.weights, tag=args.tag)
            for mode in CANONICAL_MODES
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.resolve is not None:
        if args.resolve is _RESOLVE_FROM_MODE:
            mode = args.mode
        else:
            mode = args.resolve
            if args.mode is not None and args.mode != mode:
                print(
                    f"presets: --resolve {mode} 与 --mode {args.mode} 冲突",
                    file=sys.stderr,
                )
                return 2
        if not mode:
            print(
                "presets: --resolve 需要 mode，请用 --resolve --mode <mode> 或 --resolve <mode>",
                file=sys.stderr,
            )
            return 2
        result = resolve(
            mode,
            out=args.out,
            weights=args.weights,
            tag=args.tag,
            smoke=args.smoke,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
