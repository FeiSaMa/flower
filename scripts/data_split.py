"""确定性 train/val/test 划分与指纹（阶段 A）。

划分算法不得使用 random / numpy 随机：
  1. 对类 L（1..102 升序），取 meta.csv 中 split ∈ {train,val} 且 label==L 的 image_id；
  2. 每个 image_id 计算 k = sha256(f"{split_seed}:{image_id}").hexdigest()；
  3. 按 (k, image_id) 升序排序；
  4. 前 val_per_class 个 → val，其余 → train，均保持该排序；
  5. 类号升序展平 train/val；test 按 meta.csv 行序。

理由：random.shuffle 不在跨解释器版本稳定契约内；hash 排序给契约保证。
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

from schema import META_SHA256, NUM_CLASSES, SPLIT_FINGERPRINT, SPLIT_SEED, TEST_N, TRAIN_N, VAL_N, VAL_PER_CLASS


def _read_meta(meta_csv):
    """返回 (train_val_by_class, test_rows)。test_rows 保持 meta.csv 行序。"""
    by_class = {label: [] for label in range(1, NUM_CLASSES + 1)}
    test_rows = []
    with Path(meta_csv).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = row["image_id"]
            label = int(row["label"])
            split = row["split"]
            if split == "test":
                test_rows.append((image_id, label))
            elif split in ("train", "val"):
                by_class.setdefault(label, []).append(image_id)
    return by_class, test_rows


def _hash_key(split_seed, image_id):
    return hashlib.sha256(f"{split_seed}:{image_id}".encode("ascii")).hexdigest()


def build_split(meta_csv, split_seed=SPLIT_SEED, val_per_class=VAL_PER_CLASS):
    """纯函数：按 hash 分桶生成 {'train': [...], 'val': [...], 'test': [...]}。"""
    by_class, test_rows = _read_meta(meta_csv)
    train_rows = []
    val_rows = []
    for label in range(1, NUM_CLASSES + 1):
        names = by_class.get(label, [])
        ordered = sorted(names, key=lambda image_id: (_hash_key(split_seed, image_id), image_id))
        val_rows.extend((image_id, label) for image_id in ordered[:val_per_class])
        train_rows.extend((image_id, label) for image_id in ordered[val_per_class:])
    return {"train": train_rows, "val": val_rows, "test": test_rows}


def split_fingerprint(split):
    """train/val/test 依次拼接 "{split}:{image_id},{label}\\n" 的 SHA-256。"""
    digest = hashlib.sha256()
    for part in ("train", "val", "test"):
        for image_id, label in split[part]:
            digest.update(f"{part}:{image_id},{label}\n".encode("utf-8"))
    return digest.hexdigest()


def _meta_sha256(meta_csv):
    return hashlib.sha256(Path(meta_csv).read_bytes()).hexdigest()


def _report(meta_csv, split_seed, val_per_class):
    split = build_split(meta_csv, split_seed=split_seed, val_per_class=val_per_class)
    fingerprint = split_fingerprint(split)
    print(f"meta_sha256 {_meta_sha256(meta_csv)}")
    print(
        "counts "
        f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}"
    )
    print(f"split_fingerprint {fingerprint}")
    return split, fingerprint


def _verify(meta_csv, split_seed, val_per_class):
    problems = []
    meta_digest = _meta_sha256(meta_csv)
    if meta_digest != META_SHA256:
        problems.append(f"meta.csv sha256 不匹配: {meta_digest} != {META_SHA256}")

    split = build_split(meta_csv, split_seed=split_seed, val_per_class=val_per_class)
    fingerprint = split_fingerprint(split)
    expected = {
        "train": TRAIN_N,
        "val": VAL_N,
        "test": TEST_N,
    }
    for part, want in expected.items():
        got = len(split[part])
        if got != want:
            problems.append(f"{part} 数量不匹配: {got} != {want}")
    if fingerprint != SPLIT_FINGERPRINT:
        problems.append(f"划分指纹不匹配: {fingerprint} != {SPLIT_FINGERPRINT}")

    # 自检：seed 0 与 seed 1 必须给出不同划分（防止 hash 键退化为常量）。
    if split_seed == SPLIT_SEED and val_per_class == VAL_PER_CLASS:
        alt = split_fingerprint(
            build_split(meta_csv, split_seed=SPLIT_SEED + 1, val_per_class=val_per_class)
        )
        if alt == fingerprint:
            problems.append("split_seed=0 与 split_seed=1 的划分指纹相同，hash 分桶失效")

    print(f"meta_sha256 {meta_digest}")
    print(
        "counts "
        f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}"
    )
    print(f"split_fingerprint {fingerprint}")
    if problems:
        for item in problems:
            print(f"data_split FAIL: {item}", file=sys.stderr)
        return 3
    print("OK")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="确定性划分与指纹校验")
    parser.add_argument("--verify", action="store_true", help="按契约常量校验并退出 0/3")
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--val-per-class", type=int, default=VAL_PER_CLASS)
    parser.add_argument(
        "--meta",
        default=str(Path(__file__).resolve().parents[1] / "data" / "meta.csv"),
    )
    args = parser.parse_args(argv)

    if args.verify:
        return _verify(Path(args.meta), args.split_seed, args.val_per_class)
    _report(Path(args.meta), args.split_seed, args.val_per_class)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
