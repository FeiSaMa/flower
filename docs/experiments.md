# 四组实验详解

> 分工：`report/report.md` 是正式报告（六章，结论导向）；`docs/walkthrough.md` 是整体讲解（口语顺序）；本文只讲四组实验：配置、训练、结果、机制与可比性。数字取自 `report/results.csv` 与 `runs/<mode>/{metrics.json,history.csv}`。

## 0. 先看一张总表

| 组别 | 骨干初始化 | 训练什么 | epoch | test top-1 | macro-F1 | 耗时 |
|---|---|---:|---:|---:|---:|---:|
| scratch | 随机 | 全部 1122.9 万参数 | 120 | 55.46% | 54.80% | 22.5 min |
| frozen | ImageNet | 仅分类头 5.2 万 | 12 | 81.27% | 80.45% | 2.0 min |
| two-stage | ImageNet | 5 轮头 + 20 轮全网络 | 25 | 92.50% | 92.14% | 4.0 min |
| strong | ImageNet | 5 轮头 + 35 轮全网络 | 40 | 92.96% | 92.52% | 9.2 min |

曲线 `../report/figures/fig1_curves.png`，柱状图 `../report/figures/fig5_bars.png`。

## 1. 共同的底座

- **数据**：`data/meta.csv` 8189 张、102 类；官方 train/val 各 1020 张（每类 10 张），test 6149 张（每类 20–238 张）。合并 train+val 得 2040 张，按 `sha256("0:"+image_id)` 每类抽 3 张验证，得 1734 训 / 306 验；test 只评一次。
- **模型**：ResNet-18，`fc` 换 512→102；全模型 11,228,838 参数，分类头 52,326 参数（0.47%）。
- **优化**：AdamW，weight decay 0.01（BN/bias 不衰减），batch 64（约 28 batch/epoch），余弦退火，seed/split_seed=0，Tesla T4；归一化用 ImageNet mean/std。
- **评估**：val top-1 选 best epoch，test 只评一次；评估损失固定 `label_smoothing=0`。

## 2. 组一：scratch —— 不借，从零开始

**改了什么**：去掉预训练权重；两个 lr 同为 1e-3，轮数 120。

| 项 | 值 |
|---|---|
| weights | none（随机初始化） |
| stage1 / stage2 | 0 / 120 |
| lr_head / lr_backbone | 1e-3 / 1e-3 |
| batch | 64 |
| 增强 | baseline：RRC(224, 0.6–1.0) + 翻转 |
| 标签平滑 | 0.0 |
| BN | train |
| 可训参数 | 11,228,838 |

| 节点 | epoch | train loss | train acc | val loss | val acc |
|---|---:|---:|---:|---:|---:|
| 第 1 轮 | 1 | 4.2271 | 6.98% | 5.8943 | 3.92% |
| 最好一轮 | 103 | 0.0619 | 99.37% | 1.8211 | 65.03% |
| 最后一轮 | 120 | 0.0557 | 99.54% | 1.7874 | 62.09% |

train loss 4.2271 → 0.0557，val 5.8943 → 1.7874。第 12 轮 val 仅 30.07%，第 103 轮见顶后回落，训练准确率却稳在 99.5%——已进入背题区间。

**结果**：top-1/3/5 = 55.46 / 72.47 / 78.97%，macro-F1 54.80%，weighted-F1 55.68%，1350.16 s（22.5 min）。

**怎么解释**：1734 张图要训 1122 万参数，容量远大于数据能约束的自由度；最省力的解是记住背景与构图，而非花瓣结构。末轮 train 99.54% / val 62.09% 差 37.4 点，val 65.03% / test 55.46% 又差 9.6 点：验证每类 3 张且平衡，test 按自然分布、难类更多。

**一句话结论**：不借预训练时，模型做的是记忆而非归纳；55.46% 是这个配置的下界，不是从零训练的上限。

## 3. 组二：frozen —— 只借特征，不动骨干

**改了什么**：权重换 `imagenet1k_v1`；骨干与 BN 冻结（`lr_backbone=0`）；12 轮。

| 项 | 值 |
|---|---|
| weights | imagenet1k_v1 |
| stage1 / stage2 | 0 / 12 |
| lr_head / lr_backbone | 1e-3 / 0 |
| batch | 64 |
| 增强 | baseline |
| 标签平滑 | 0.0 |
| BN | frozen |
| 可训参数 | 52,326 |

| 节点 | epoch | train loss | train acc | val loss | val acc |
|---|---:|---:|---:|---:|---:|
| 第 1 轮 | 1 | 4.3543 | 9.05% | 3.6813 | 26.47% |
| 最好一轮 | 8 | 1.0982 | 92.79% | 1.3083 | 83.66% |
| 最后一轮 | 12 | 0.9549 | 94.23% | 1.2305 | 81.70% |

train loss 4.3543 → 0.9549，val 3.6813 → 1.2305。第 8 轮见顶 83.66%，之后 val 回落而 train 升到 94.23%——5.2 万参数同样会过拟合，只是幅度小。

**结果**：top-1/3/5 = 81.27 / 92.11 / 94.67%，macro-F1 80.45%，weighted-F1 81.18%，122.60 s（2.0 min）。

**怎么解释**：只训 0.47% 的参数、2 分钟，比 scratch 高 25.81 点：骨干的 512 维特征对 102 种花已线性可分到八成，分类头只负责重新加权。第 8 轮后 val 不再涨，说明冻结特征表达已用尽，剩下的 11 点必须动骨干。

**一句话结论**：预训练的价值在最便宜的「只训线性头」里兑现了大部分，2 分钟换 25.81 点。

## 4. 组三：two-stage —— 先借特征，再微调

**改了什么**：解冻骨干。前 5 轮与 frozen 同起点（第 1 轮 loss/acc 逐位相同），第 2 轮起分叉：frozen 已降 lr，stage1 恒为 1e-3。第 6 轮起 `requires_grad=True`、BN 转 train，以 `lr_backbone=1e-4`（头的 1/10）重建优化器与 scheduler，再跑 20 轮。

| 项 | 值 |
|---|---|
| weights | imagenet1k_v1 |
| stage1 / stage2 | 5 / 20 |
| lr_head / lr_backbone | 1e-3 / 1e-4 |
| batch | 64 |
| 增强 | baseline |
| 标签平滑 | 0.0 |
| BN | frozen → train |
| 可训参数 | 52,326 → 11,228,838 |

| 节点 | epoch | train loss | train acc | val loss | val acc |
|---|---:|---:|---:|---:|---:|
| 第 1 轮 | 1 | 4.3543 | 9.05% | 3.6813 | 26.47% |
| 最好一轮 | 24 | 0.0101 | 100.00% | 0.2346 | 94.44% |
| 最后一轮 | 25 | 0.0102 | 100.00% | 0.2287 | 94.44% |

train loss 4.3543 → 0.0102，val 3.6813 → 0.2287。解冻当轮抖动明显：第 5 轮 train loss / acc 1.4801 / 86.04%、val 1.5350 / 77.78%；第 6 轮 train 0.9921 / 81.20%（−4.84）、val 0.6687 / 83.99%（+6.21）。第 7 轮 train acc 回到 96.31%。抖掉的是只对旧头有利的预测，换来骨干向花卉域移动。

**结果**：top-1/3/5 = 92.50 / 97.69 / 98.75%，macro-F1 92.14%，weighted-F1 92.50%，241.68 s（4.0 min）。

**怎么解释**：frozen 的特征是为 ImageNet 的 1000 类训练的，与花卉线索（花蕊、花萼、花瓣排布）不完全重合；1e-4 微调是在保留通用能力的前提下做域适配。先冻 5 轮让新头到 77.78%，梯度不至于冲垮骨干。代价是 train acc 100%、val 94.44%，val→test 只差 1.94 点。

**一句话结论**：光借特征到 81%，把骨干也交出来微调 2 分钟，再换 11.24 点。

## 5. 组四：strong —— 在 two-stage 上加强

**改了什么**：一次改三件事（bundle，不能拆开归因）：增强 baseline→strong（裁剪 0.35、旋转、颜色抖动、RandomErasing）；标签平滑 0→0.1；stage2 从 20 轮拉到 35 轮。

| 项 | 值 |
|---|---|
| weights | imagenet1k_v1 |
| stage1 / stage2 | 5 / 35 |
| lr_head / lr_backbone | 1e-3 / 1e-4 |
| batch | 64 |
| 增强 | strong（裁剪 0.35 / 翻转 / 旋转 25 / 颜色抖动 / Erasing） |
| 标签平滑 | 0.1 |
| BN | frozen → train |
| 可训参数 | 52,326 → 11,228,838 |

| 节点 | epoch | train loss | train acc | val loss | val acc |
|---|---:|---:|---:|---:|---:|
| 第 1 轮 | 1 | 4.5576 | 4.50% | 3.9688 | 15.03% |
| 最好一轮 | 17 | 0.9787 | 99.31% | 0.4262 | 94.77% |
| 最后一轮 | 40 | 0.8970 | 100.00% | 0.4638 | 94.77% |

train loss 4.5576 → 0.8970，val 3.9688 → 0.4638。强增强让起步变慢：第 1 轮 4.50% / 15.03%（two-stage 9.05% / 26.47%），第 5 轮 train acc 64.13%（同期 86.04%）。第 5→6 轮 train loss 2.6053 → 2.1364、acc 64.13% → 69.09%，val loss 2.0911 → 0.9952、acc 65.36% → 77.12%（+11.76）：train acc 也升，没有 two-stage 的掉点。之后 val 在 88%–94.8% 波动，第 17 与 40 轮同为 94.77%，取 17。

**结果**：top-1/3/5 = 92.96 / 97.76 / 98.52%，macro-F1 92.52%，weighted-F1 92.95%，552.05 s（9.2 min）。

**怎么解释**：强增强与标签平滑是正则，长训练给模型时间收敛；train acc 到第 40 轮才 100%，val 稳定在 94% 以上，但 test 只多 0.46 点。瓶颈已不是过拟合：验证集仅 306 张，best epoch 有 ±1 点噪声；剩余错误集中在 camellia、petunia 等相近类。

**一句话结论**：三因素捆绑只换来 +0.46 点、+5 分钟，这批数据上方法已接近饱和。

## 6. 横着比：同预算、同参数、同时间

**（1）同 epoch 对照。** 四份 `history.csv` 第 12 轮的 val top-1：

| 组别 | 第 12 轮 val top-1 | 总 epoch |
|---|---:|---:|
| scratch | 30.07% | 120 |
| frozen | 81.70% | 12 |
| two-stage | 92.81% | 25 |
| strong | 88.56% | 40 |

同样 12 轮，scratch 只有 30.07%，frozen 已 81.70%；scratch 最终追到 65.03% 靠后面 108 轮，55.46% 是 11 倍时间换的。

**（2）参数效率。** 首阶段可训 52,326 对 11,228,838，差 214.6 倍：前者用 0.47% 参数拿到 81.27% 与 92.50%，后者用 100% 参数拿到 55.46%——多 214 倍参数，反少 25.81 点。

**（3）时间成本。** 22.5 / 2.0 / 4.0 / 9.2 分钟，对应每分钟 2.46 / 39.77 / 22.97 / 10.10 个 top-1 点。边际值更关键：frozen→two-stage 多花 1.98 分钟换 11.24 点（5.66 点/分）；two-stage→strong 多花 5.17 分钟换 0.46 点（0.088 点/分）。

**（4）收益递减。** 25.81 点（scratch→frozen，总时间反而少 20.5 分钟）→ 11.24 点（+2.0 分钟）→ 0.46 点（+5.2 分钟）。边际收益两步掉两个数量级：5.66 → 0.088，约 64 倍。

**（5）测试集分布。** test 每类 20–238 张，train/val 每类固定 10 张。macro-F1 与 top-1 的差：0.66 / 0.82 / 0.37 / 0.43 点，two-stage 后收窄，说明微调救回一些弱类；但 test 不均衡使 top-1 被大类别主导，macro-F1 才是平权读法。strong 的 top-5（98.52%）反低于 two-stage（98.75%），0.46 点并非各指标齐升。

## 7. 这四组能说明什么、不能说明什么

**能说明**：

1. 预训练的价值量级：同模型、同数据、同增强，只换初始化并冻结骨干，top-1 从 55.46% 到 81.27%（+25.81），耗时反从 22.5 分钟降到 2.0 分钟。
2. 微调的必要性：frozen 第 8 轮见顶后 val 回落，说明冻结特征用尽；解冻后 +11.24 点。
3. 增强/平滑/更长训练的边际收益很小：三因素一起上只有 +0.46 点，时间却翻倍多。

**不能说明**：

1. 0.46 点不是 strong 的真实优势。单 seed、无方差；验证集 306 张，换个划分或 seed 可能翻转。
2. 不能把 strong 的增益归因到单一因素。增强、平滑、训练长度一起改，没做消融。
3. 不能把 scratch 的 55.46% 当成从零训练的上限。它拿了最多的 120 轮，这个数字是下界。
4. loss 不可跨组比较。strong 的训练目标带 label smoothing 0.1，`train_loss` 尺度不同；val loss 虽在评估时固定 `label_smoothing=0`，但 best epoch 按 top-1 选，loss 只能组内看趋势。
5. 四组不是等预算设计，方向对 scratch 有利：22.5 分钟是 frozen 的 11 倍、two-stage 的 5.6 倍。它拿这么多预算仍落后 25.81 点；但不是等算力对照。

## 8. 复现

```bash
python3 scripts/data_split.py --verify     # 指纹 ce25c1c1…e81d
python3 scripts/build_kernel.py            # 生成 kernel
K="uvx --python 3.13 --from kaggle kaggle"
$K kernels push -p kaggle/flower
$K kernels output feisama/flower-102-transfer -p /tmp/opencode/flower-out
rsync -a /tmp/opencode/flower-out/runs/ runs/
python3 scripts/check_runs.py runs --assert-complete --assert-presets --assert-env
python3 scripts/collect_results.py
```

单组重跑：`python3 scripts/train_full.py --mode two-stage`（需 torch）。四组合计约 39 分钟。

## 9. 常见疑问

**为什么第 6 轮才解冻？** 第 5 轮头已把 val 做到 77.78%；一开始全解冻，随机头的梯度会冲进骨干。

**解冻时 train acc 为什么掉？** 重建优化器 + 骨干更新，一部分靠「冻结特征 + 旧头」凑对的样本被打散；同轮 val 反而 +6.21 点。

**为什么 test 比 val 低？** val 每类 3 张且平衡，best epoch 又在它上面选；test 按自然分布，petunia 就有 238 张。

---

数据源：`report/results.csv`、`runs/*/metrics.json`、`runs/*/history.csv`、`data/meta.csv`。
