"""产物契约与仓库常量的唯一事实源（阶段 A）。

任何脚本都不得另行复制这些常量：`METRICS_FIELDS` / `SUMMARY_FIELDS` /
`CSV_HEADERS` / 指纹常量 / `source_sha256` 均以本模块为准。

字段表用 (path, type, constraint, source) 四元组描述，`METRICS_FIELDS`
与 `SUMMARY_FIELDS` 均由表派生，禁止手写字段字典。
"""

import argparse
import hashlib
from pathlib import Path

SCHEMA_VERSION = 1

META_SHA256 = "314da2e0d03d3e2483266d0b213ee1b30488e5cd41b0b2b6f4e448f604273188"
SPLIT_FINGERPRINT = "ce25c1c1b75c47e5625dc5b0ff76905109f37756172500a98d5d61b82a15e81d"
SPLIT_SEED = 0
VAL_PER_CLASS = 3
NUM_CLASSES = 102
TEST_N = 6149
TRAIN_N = 1734
VAL_N = 306

CANONICAL_MODES = ("scratch", "frozen", "two-stage", "strong")
DIAG_PREFIX = "runs/diag/"
SMOKE_PREFIX = "/tmp/"

TOL_JSON = 1e-9
TOL_CSV = 1e-6

# (path, type, constraint, source)
_METRICS_TABLE = (
    ("schema_version", "int", "==1", "const"),
    # ---- run (53) ----
    ("run.mode", "str", "canonical", "preset"),
    ("run.tag", "str", "nonempty;==mode unless diag", "resolve"),
    ("run.is_diag", "bool", "False for canonical runs", "resolve"),
    ("run.base", "str|null", "preset (strong->two-stage)", "preset"),
    ("run.seed", "int", "cli --seed", "cli"),
    ("run.split_seed", "int", "==0", "const"),
    ("run.val_per_class", "int", "==3", "const"),
    ("run.weights_init", "str", "in {imagenet1k_v1,none}", "preset"),
    ("run.epochs_planned", "int", "stage1_epochs+stage2_epochs", "resolve"),
    ("run.epochs_run", "int", "<=epochs_planned", "measured"),
    ("run.batch", "int", "==64", "const"),
    ("run.optimizer", "str", "==adamw", "const"),
    ("run.weight_decay", "float", "==0.01", "const"),
    ("run.wd_no_decay", "str", "==bn_and_bias", "const"),
    ("run.scheduler", "str", "==cosine", "const"),
    ("run.lr_head", "float", ">0", "preset"),
    ("run.lr_backbone", "float", ">=0", "preset"),
    ("run.label_smoothing", "float", "in {0.0,0.1}", "preset"),
    ("run.aug", "str", "in {baseline,strong}", "preset"),
    ("run.aug_ops", "list[str]", "==AUG_OPS[aug]", "resolve"),
    ("run.bn_stage1", "str", "in {frozen,train}", "preset"),
    ("run.bn_stage2", "str|null", "in {frozen,train}|null", "preset"),
    ("run.bn_mode", "str", "in {frozen,train,frozen_to_train}", "resolve"),
    ("run.unfreeze_epoch", "int|null", "stage1_epochs>0 ? stage1+1 : null", "resolve"),
    ("run.params_total", "int", ">0", "measured"),
    ("run.params_trainable_stage1", "int", ">0", "measured"),
    ("run.params_trainable_peak", "int", ">=stage1", "measured"),
    ("run.train_n", "int", "==1734", "const"),
    ("run.val_n", "int", "==306", "const"),
    ("run.test_n", "int", "==6149", "const"),
    ("run.num_classes", "int", "==102", "const"),
    ("run.split_fingerprint", "str", "==SPLIT_FINGERPRINT", "const"),
    ("run.meta_sha256", "str", "==META_SHA256", "const"),
    ("run.source_sha256", "str", "64 hex", "measured"),
    ("run.git_commit", "str|null", "git rev-parse HEAD", "measured"),
    ("run.git_dirty", "bool|null", "git status --porcelain", "measured"),
    ("run.provenance", "str", "in {git,embedded}", "measured"),
    ("run.python", "str", "runtime", "measured"),
    ("run.torch", "str", "runtime", "measured"),
    ("run.torchvision", "str", "runtime", "measured"),
    ("run.cuda", "str", "runtime; '' on cpu", "measured"),
    ("run.gpu", "str", "runtime; '' on cpu", "measured"),
    ("run.device", "str", "in {cuda,cpu}", "measured"),
    ("run.reproducibility", "str", "in {strict,best_effort}", "measured"),
    ("run.cudnn_deterministic", "bool", "==True", "const"),
    ("run.cudnn_benchmark", "bool", "==False", "const"),
    ("run.deterministic_algorithms", "str", "==warn_only", "const"),
    ("run.cublas_workspace_config", "str|null", "==:4096:8", "const"),
    ("run.determinism_warnings", "list[str]", "nondeterministic warnings", "measured"),
    ("run.smoke", "bool", "resolve smoke", "resolve"),
    ("run.started_at", "str", "ISO8601", "measured"),
    ("run.elapsed_sec", "float", ">=0", "measured"),
    ("run.test_eval_count", "int", "==1", "measured"),
    # ---- selection (3) ----
    ("selection.criterion", "str", "==val_top1", "const"),
    ("selection.best_epoch", "int", "1-based", "measured"),
    ("selection.best_val_top1", "float", "==val.top1", "measured"),
    # ---- val (5) ----
    ("val.loss", "float", "best epoch", "measured"),
    ("val.top1", "float", "best epoch", "measured"),
    ("val.top3", "float", "best epoch", "measured"),
    ("val.top5", "float", "best epoch", "measured"),
    ("val.macro_f1", "float", "best epoch", "measured"),
    # ---- test (8) ----
    ("test.loss", "float", "one-shot", "measured"),
    ("test.top1", "float", "one-shot", "measured"),
    ("test.top3", "float", "one-shot", "measured"),
    ("test.top5", "float", "one-shot", "measured"),
    ("test.macro_f1", "float", "one-shot", "measured"),
    ("test.weighted_f1", "float", "one-shot", "measured"),
    ("test.checkpoint_epoch", "int", "==selection.best_epoch", "measured"),
    ("test.per_class_acc", "list[float]", "len==102", "measured"),
)

_SUMMARY_TABLE = (
    ("kernel_ref", "str", "nonempty", "const"),
    ("built_source_sha256", "str", "64 hex", "measured"),
    ("started_at", "str", "ISO8601", "measured"),
    ("finished_at", "str", "ISO8601", "measured"),
    ("gpu", "str", "runtime; '' on cpu", "measured"),
    ("modes", "list", "len 4, CANONICAL_MODES order", "measured"),
)

METRICS_FIELDS = tuple(row[0] for row in _METRICS_TABLE)
SUMMARY_FIELDS = tuple(row[0] for row in _SUMMARY_TABLE)

CSV_HEADERS = {
    "history": (
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "phase",
        "lr_head",
        "lr_backbone",
        "sec",
    ),
    "per_class": (
        "class_idx",
        "label",
        "n_total",
        "n_correct",
        "accuracy",
        "mean_true_prob",
    ),
    "confusion_matrix": ("true_label",) + tuple(str(i) for i in range(NUM_CLASSES)),
    "misclassified": (
        "image_id",
        "true_idx",
        "pred_idx",
        "true_prob",
        "pred_prob",
    ),
    "top_confusions": ("true_idx", "pred_idx", "count", "true_total"),
}

SOURCE_FILES = (
    "scripts/schema.py",
    "scripts/presets.py",
    "scripts/data_split.py",
    "scripts/train_full.py",
)


def source_sha256(root):
    """四份源文件按固定顺序入库字节的 SHA-256（不得在掩码/替换后计算）。"""
    digest = hashlib.sha256()
    for rel in SOURCE_FILES:
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update((Path(root) / rel).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _main(argv=None):
    parser = argparse.ArgumentParser(description="schema 常量与源文件指纹")
    parser.add_argument("--print-source-sha", action="store_true")
    parser.add_argument("--dump-schema", action="store_true")
    args = parser.parse_args(argv)

    if args.print_source_sha:
        print(source_sha256(Path(__file__).resolve().parents[1]))
    if args.dump_schema:
        print(f"{len(METRICS_FIELDS)} {len(SUMMARY_FIELDS)} {SCHEMA_VERSION}")
    if not args.print_source_sha and not args.dump_schema:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
