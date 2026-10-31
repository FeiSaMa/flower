#!/usr/bin/env python3
"""校验 Oxford 102 Flowers 原始数据：图片数、标签数、官方划分是否自洽。

用法:
    python3 scripts/check_data.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matread import load_mat  # noqa: E402  (零依赖 .mat 解析)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
JPG = RAW / "jpg"


def main() -> int:
    labels = load_mat(RAW / "imagelabels.mat")["labels"]
    ids = load_mat(RAW / "setid.mat")
    splits = {k: ids[k] for k in ("trnid", "valid", "tstid")}

    print(f"标签总数      : {len(labels)}")
    print(f"类别数        : {min(labels)} .. {max(labels)} (期望 1..102)")
    for name, ids in splits.items():
        print(f"{name:<14}: {len(ids):>5} 张   id 范围 {min(ids)}..{max(ids)}")

    total = sum(len(v) for v in splits.values())
    print(f"三个划分合计  : {total} (期望 {len(labels)})")

    # 划分之间是否有重叠 / 是否覆盖全部图片
    all_ids = [i for v in splits.values() for i in v]
    dup = [i for i, c in Counter(all_ids).items() if c > 1]
    missing = sorted(set(range(1, len(labels) + 1)) - set(all_ids))
    print(f"划分间重叠 id : {len(dup)}")
    print(f"未被划分覆盖  : {len(missing)}")

    per_class = Counter(labels)
    print(f"每类图片数    : {min(per_class.values())} .. {max(per_class.values())}")
    trn_per_class = Counter(labels[i - 1] for i in splits["trnid"])
    print(f"每类训练图数  : {min(trn_per_class.values())} .. {max(trn_per_class.values())}")

    if JPG.is_dir():
        n_jpg = len(list(JPG.glob("image_*.jpg")))
        print(f"磁盘 jpg 数量 : {n_jpg} (期望 {len(labels)})")
        problem = (n_jpg != len(labels)) or dup or missing or total != len(labels)
    else:
        print(f"磁盘 jpg 目录 : 不存在 ({JPG}) —— 还没解压 102flowers.tgz")
        problem = bool(dup or missing or total != len(labels))

    print("\n结论: " + ("发现问题，需要核查" if problem else "数据自洽，可以进入下一步"))
    return 1 if problem else 0


if __name__ == "__main__":
    sys.exit(main())
