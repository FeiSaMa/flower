"""汇总四组 metrics.json 为 report/results.csv（阶段 C）。

列固定、mode 顺序固定、浮点 6 位小数；四份 split_fingerprint 必须一致。
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schema import META_SHA256, SPLIT_FINGERPRINT

MODES = ("scratch", "frozen", "two-stage", "strong")
COLUMNS = (
    "mode",
    "test_top1",
    "test_top3",
    "test_top5",
    "test_macro_f1",
    "test_weighted_f1",
    "val_top1",
    "best_epoch",
    "epochs_run",
    "elapsed_sec",
    "params_total",
    "params_trainable_stage1",
)
FLOAT_COLUMNS = {
    "test_top1",
    "test_top3",
    "test_top5",
    "test_macro_f1",
    "test_weighted_f1",
    "val_top1",
    "elapsed_sec",
}


def load_all(runs_dir):
    metrics = {}
    problems = []
    for mode in MODES:
        path = Path(runs_dir) / mode / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"缺少 {path}")
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        metrics[mode] = data
        fingerprint = data["run"]["split_fingerprint"]
        meta_sha256 = data["run"]["meta_sha256"]
        if fingerprint != SPLIT_FINGERPRINT:
            problems.append(
                f"{mode}.split_fingerprint={fingerprint} != {SPLIT_FINGERPRINT}"
            )
        if meta_sha256 != META_SHA256:
            problems.append(f"{mode}.meta_sha256={meta_sha256} != {META_SHA256}")
    if problems:
        raise ValueError("; ".join(problems))
    return metrics


def build_rows(metrics):
    rows = []
    for mode in MODES:
        data = metrics[mode]
        run = data["run"]
        test = data["test"]
        val = data["val"]
        selection = data["selection"]
        rows.append(
            {
                "mode": mode,
                "test_top1": test["top1"],
                "test_top3": test["top3"],
                "test_top5": test["top5"],
                "test_macro_f1": test["macro_f1"],
                "test_weighted_f1": test["weighted_f1"],
                "val_top1": val["top1"],
                "best_epoch": int(selection["best_epoch"]),
                "epochs_run": int(run["epochs_run"]),
                "elapsed_sec": run["elapsed_sec"],
                "params_total": int(run["params_total"]),
                "params_trainable_stage1": int(run["params_trainable_stage1"]),
            }
        )
    return rows


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    row[column]
                    if column not in FLOAT_COLUMNS
                    else f"{row[column]:.6f}"
                    for column in COLUMNS
                ]
            )


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="汇总四组 metrics.json")
    parser.add_argument("--root", default=None, help="runs 目录（默认仓库 runs/）")
    args = parser.parse_args(argv)
    runs_dir = Path(args.root) if args.root else root / "runs"
    try:
        metrics = load_all(runs_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"collect_results FAIL: {exc}", file=sys.stderr)
        return 1
    rows = build_rows(metrics)
    output = root / "report" / "results.csv"
    write_csv(rows, output)
    print(f"wrote {output.relative_to(root)} ({len(rows)} rows)")
    print(f"split_fingerprint {metrics[MODES[0]]['run']['split_fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
