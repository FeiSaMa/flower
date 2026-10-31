#!/usr/bin/env python3
"""把官方 .mat 元数据导出为 CSV，方便 Kaggle notebook 直接用（免装 scipy）。

输出:
    data/meta.csv   image_id,label,split
    split 取值: train / val / test

用法:
    python3 scripts/export_meta.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matread import load_mat  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def main() -> None:
    labels = load_mat(RAW / "imagelabels.mat")["labels"]
    ids = load_mat(RAW / "setid.mat")
    split_of = {}
    for name in ("trnid", "valid", "tstid"):
        for i in ids[name]:
            split_of[int(i)] = "train" if name == "trnid" else ("val" if name == "valid" else "test")

    out = ROOT / "data" / "meta.csv"
    with out.open("w", encoding="utf-8") as f:
        f.write("image_id,label,split\n")
        for i, lab in enumerate(labels, start=1):
            f.write(f"image_{i:05d}.jpg,{lab},{split_of[i]}\n")

    print(f"写出 {out}  ({len(labels)} 行, {out.stat().st_size/1024:.0f} KiB)")


if __name__ == "__main__":
    main()
