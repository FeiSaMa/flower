# flower — 牛津 102 花卉细粒度分类：从零？不，借用

用 ImageNet 预训练 ResNet-18 做 102 类花卉分类，回答「迁移学习借来了什么、代价多少、在哪里失效」。一句话结论：**预训练带来 55.46% → 81.27%，两阶段微调到 92.50%，strong 92.96%**。

## 结果摘要

数据源：`report/results.csv`（test 集；耗时含训练与评估）。

| 组别 | test top-1 | top-3 | macro-F1 | 耗时 |
|---|---:|---:|---:|---:|
| scratch | 55.46% | 72.47% | 54.80% | 1350.2 s |
| frozen | 81.27% | 92.11% | 80.45% | 122.6 s |
| two-stage | 92.50% | 97.69% | 92.14% | 241.7 s |
| strong | 92.96% | 97.76% | 92.52% | 552.1 s |

公开结果（Nilsback 2008 72.8%、CCT-14/7×2 99.76%、ViT-L/16 99.37%）的划分、骨干与训练协议均不同，**不可直接比较**；只做量级参照，详见 `report/report.md` 第四章。

## 目录

```
flower/
├── data/
│   ├── raw/              jpg/ + imagelabels.mat + setid.mat
│   ├── meta.csv          8189 行：image_id,label,split
│   └── class_names.csv   102 类英文花名（社区整理，非官方）
├── scripts/
│   ├── matread.py / check_data.py / export_meta.py / train_min.py   基线
│   ├── schema.py / presets.py / data_split.py / check_runs.py / train_full.py   四组实验
│   └── build_kernel.py / collect_results.py / make_figures.py / build_slides.py
├── kaggle/flower/        kernel.py（生成）+ kernel-metadata.json
├── runs/                 scratch / frozen / two-stage / strong（csv/json，无权重）
├── report/
│   ├── report.md         六章结课报告
│   ├── results.csv       四组汇总
│   ├── figures/          六张图
│   └── slides.pptx       14 页演讲 PPT
└── README.md
```

Kaggle 训练使用 **script kernel**；`kaggle/flower/` 下为可直接 push 的生成物。

`pyproject.toml` 只声明项目元信息与可选依赖（`plot` / `slides`）；训练依赖由 Kaggle 镜像提供，本机不安装 torch。

## 数据

官方站点 <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>。

| 项 | 值 |
|---|---|
| 图片总数 | 8189 |
| 类别 | 1..102（1-based） |
| 每类图片数 | 40 .. 258（极不均衡） |
| train / val / test | 1020 / 1020 / 6149 |
| 每类训练张数 | 10（固定） |
| 划分重叠 / 未覆盖 | 0 / 0 |

训练时把 train+val 合并成 2040 张；test 只做最终评估。归一化使用 ImageNet 的 mean/std（骨干为 ImageNet 预训练），不要用本数据集统计量。

## 复现步骤

### 1. 补数据

```bash
# 从 Kaggle 镜像 trnglbotrn/oxford-102-flowers 取 jpg/ → data/raw/jpg/
# 从牛津站取 imagelabels.mat、setid.mat → data/raw/
python3 scripts/check_data.py        # 期望输出「数据自洽」
```

### 2. 校验划分

```bash
python3 scripts/data_split.py --verify
# 期望：split_fingerprint ce25c1c1b75c47e5625dc5b0ff76905109f37756172500a98d5d61b82a15e81d + OK
```

### 3. 生成并推送 kernel

```bash
python3 scripts/build_kernel.py
uvx --python 3.13 --from kaggle kaggle kernels push -p kaggle/flower
```

### 4. 取回与硬门禁

```bash
uvx --python 3.13 --from kaggle kaggle kernels output feisama/flower-102-transfer -p /tmp/opencode/flower-out
rsync -a /tmp/opencode/flower-out/runs/ runs/
python3 scripts/check_runs.py runs --assert-complete --assert-presets --assert-env
```

### 5. 结果表与图

```bash
python3 scripts/collect_results.py
uv run --no-project --python 3.13 --with matplotlib --with pandas python3 scripts/make_figures.py
```

### 6. 报告与 PPT

```bash
uvx --python 3.13 --with python-pptx python3 scripts/build_slides.py
# 若 uvx 把 python3 解析为版本请求，改用：
# uv run --no-project --python 3.13 --with python-pptx python3 scripts/build_slides.py
```

## 已知坑

1. **标签是 1-based**：`labels[i]` 对应 `image_{i+1:05d}.jpg`，读成 0-based 会整体错一位。
2. **train 每类只有 10 张**：必须合并 val、做数据增强，不能从零训练。
3. 归一化用 ImageNet 的 mean/std（骨干是 ImageNet 预训练），不要用本数据集统计量。
4. **不要 `uv run --with scipy`**：本机 Python 3.14，uv 缓存只有 cp313 的 scipy wheel，会卡在联网找 cp314 轮子。读 mat 一律用 `scripts/matread.py`；普通元素按 8 字节对齐，miCOMPRESSED 块紧排不对齐。
5. Kaggle 镜像 `trnglbotrn/oxford-102-flowers` 只含 `jpg/`，没有 `.mat`；元数据从牛津站取。
6. 本机直连牛津站约 30 KB/s，Kaggle API 多连接能到 5 MB/s，优先走 Kaggle。
7. **Kaggle script kernel 的 `__file__` 在只读 `/kaggle/src`**；工作区必须写 `/kaggle/working`，否则 `meta.csv` 或 `runs/` 会报 read-only filesystem。
8. **GPU 队列可能长时间 QUEUED**；冒烟验证改用 CPU（`enable_gpu: false`）更快，正式四组训练再上 GPU。
9. `evaluate()` 的按类累加器必须用 `index_add_`，不能直接加 batch 张量（102 vs 64 会 shape mismatch）。
10. 训练损失与评估损失解耦：训练按预设可用 label smoothing，**评估固定 `label_smoothing=0`**。
11. `random.shuffle` 不在跨版本稳定契约内；划分用 **sha256 分桶**，test 按 meta 行序。

## 报告与 PPT

`report/report.md` 已完成，六章为：为什么不能从零、迁移学习原理、方法与设置、结果与公开对比、错例分析、讨论与边界。报告中的结果表来自 `report/results.csv`，图来自 `report/figures/`；`report/slides.pptx` 为 14 页演讲版。
