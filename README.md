# flower — 通识选修课项目：从零？不，借用

用预训练 ResNet 做牛津 102 花卉细粒度分类，回答一个问题：**迁移学习到底借来了什么，代价多少，在哪里失效。**

## 目录

```
flower/
├── data/
│   ├── raw/            原始数据（zip 331 MB + jpg 347 MB + 两个 .mat）
│   └── meta.csv        8189 行，image_id,label,split（192 KiB）
├── scripts/
│   ├── matread.py      零依赖 MATLAB v5 解析器
│   ├── check_data.py   数据自洽性校验
│   └── export_meta.py  .mat → CSV
├── notebooks/          Kaggle 训练 notebook（待建）
└── report/             结课报告（待写）
```

## 数据

官方站点 <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>。

| 项 | 值 |
|---|---|
| 图片总数 | 8189 |
| 类别 | 1..102 |
| 每类图片数 | 40 .. 258（极不均衡） |
| train / val / test | 1020 / 1020 / 6149 |
| 每类训练张数 | 10（固定） |
| 划分重叠 / 未覆盖 | 0 / 0 |

训练时把 train+val 合并成 2040 张，test 只做最终评估。

## 快速校验

```bash
python3 scripts/check_data.py     # 期望输出「数据自洽」
```

## 已知坑

1. **标签是 1-based**，`labels[i]` 对应 `image_{i+1:05d}.jpg`，读成 0-based 会整体错一位。
2. **train 每类只有 10 张**，所以必须合并 val、必须做数据增强，且不能从零训练。
3. 归一化用 ImageNet 的 mean/std（骨干是 ImageNet 预训练），不要用本数据集统计量。
4. **不要 `uv run --with scipy`**：本机 Python 3.14，uv 缓存里只有 cp313 的 scipy wheel，会卡在联网找 cp314 轮子。读 mat 一律用 `scripts/matread.py`。该解析器两个格式细节：普通元素按 8 字节对齐填充，miCOMPRESSED 块紧排不对齐。
5. Kaggle 镜像 `trnglbotrn/oxford-102-flowers` 只含 `jpg/`，没有 `.mat`；元数据从牛津站取（小文件，慢速也不影响）。
6. 本机直连牛津站约 30 KB/s，Kaggle API 多连接能到 5 MB/s，优先走 Kaggle。

## 报告骨架

```
一、为什么不能让计算机从零开始学认花    ← 含 weights=None 对照实验
二、迁移学习的原理：借用的是什么
三、方法：两阶段微调与关键设置
四、结果：准确率、混淆矩阵、与公开结果对比
五、错例分析：模型在哪里栽跟头
六、讨论：这套方法的边界
```

数据集介绍不单独成章，塞进第三节开头，作为「为什么要做增强、为什么要合并 train+val」的依据。
