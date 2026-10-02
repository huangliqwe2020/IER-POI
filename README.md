# IER-POI

**Interest-aware Evidence-robust Recommendation for Points of Interest**  
面向 UGC 多模态证据的兴趣感知鲁棒推荐。

本仓库包含研究方案、数据审计与预处理、开源基线修复、MMPOI-NYC
复现、逐评论证据适配器，以及 Clean / Duplicate / Noise / Missing /
Sparse 鲁棒性评测的完整实现。

## 当前可复现结果

MMPOI-NYC，随机种子 618：

| 方法 | HR@5 | HR@20 | mAP@20 | NDCG@20 | MRR |
|---|---:|---:|---:|---:|---:|
| 修复后的 MMPOI checkpoint | 0.500940 | 0.700256 | 0.317036 | 0.405568 | 0.319469 |
| IER 评论证据适配器 | 0.503422 | 0.700256 | 0.317370 | 0.405832 | 0.319802 |

IER 第一阶段是“冻结复现基线，只训练证据适配器”的安全接入实验。目前属于
无损、小幅改善，不应表述为显著提升。详细结论见
[`docs/MMPOI_IER_STAGE1.md`](docs/MMPOI_IER_STAGE1.md)。

## 快速复现

### 1. 创建环境

```bash
conda env create -f environment.yml
conda activate ier-poi
python scripts/check_environment.py
```

已验证核心环境：Python 3.10.19、PyTorch 2.5.1+cu118、torchvision
0.20.1+cu118、NumPy 2.1.2、pandas 2.3.3、SciPy 1.15.3、
scikit-learn 1.7.2。CUDA 11.8 wheel 需要兼容的 NVIDIA 驱动；纯 CPU
机器请安装 PyTorch 2.5.1 CPU wheel，并在命令中使用 `cpu`。

### 2. 下载并校验 MMPOI-NYC

```bash
./scripts/download_mmpoi_nyc.sh
python scripts/prepare_mmpoi_nyc.py
```

下载脚本使用 MMPOI 作者公开的 Google Drive 文件 ID，并校验两个官方压缩包的
SHA256。数据不会提交到 Git 仓库。

### 3. 复现 MMPOI 基线

```bash
# 参数：epochs、run name、device
./scripts/run_mmpoi_nyc_baseline.sh 200 nyc_fixed_seed618 cuda:0
```

CPU 示例：

```bash
./scripts/run_mmpoi_nyc_baseline.sh 200 nyc_fixed_seed618 cpu
```

该入口会依次执行数据审计、训练、结果文件测试和 checkpoint 权重重评估。

### 4. 训练 IER 评论证据适配器

先确保基线 checkpoint 位于：

```text
runs/mmpoi/nyc_fixed_seed618/checkpoints/best_epoch.state.pt
```

然后执行本次报告所使用的 5-epoch 配置：

```bash
./scripts/run_mmpoi_ier_review.sh 5 evidence_adapter_5ep cuda:0
```

### 5. 鲁棒性评测

```bash
./scripts/evaluate_mmpoi_ier_robustness.sh \
  runs/mmpoi_ier/evidence_adapter_5ep/checkpoints/best_epoch.state.pt \
  evidence_adapter_5ep \
  cuda:0
```

输出位于：

```text
runs/mmpoi_ier_robustness/evidence_adapter_5ep/robustness_summary.json
```

## 数据与实验入口

| 目的 | 入口 |
|---|---|
| MMPOI 下载与 SHA256 校验 | `scripts/download_mmpoi_nyc.sh` |
| MMPOI 审计、图构建与数据链接 | `scripts/prepare_mmpoi_nyc.py` |
| GlobalPop/UserPop/Markov1 | `scripts/run_mmpoi_classical_baselines.py` |
| MMPOI 主基线 | `scripts/run_mmpoi_nyc_baseline.sh` |
| 从权重重新评估基线 | `scripts/evaluate_mmpoi_nyc_checkpoint.sh` |
| 基线产物完整性测试 | `scripts/test_mmpoi_nyc.py` |
| 逐评论证据编码 | `scripts/prepare_mmpoi_review_evidence.py` |
| IER 训练 | `scripts/run_mmpoi_ier_review.sh` |
| IER 模块单元测试 | `scripts/test_mmpoi_review_evidence.py` |
| 五类证据扰动评测 | `scripts/evaluate_mmpoi_ier_robustness.sh` |

通用商品推荐的 Amazon Reviews 2023 下载、审计、处理与 MMRec
BPR/LightGCN 入口也保留在 `scripts/`。完整顺序见
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md)。

## 目录

```text
configs/       固定实验参数清单
docs/          技术路线、数据审计、基线与 IER 报告
scripts/       下载、预处理、训练、测试和鲁棒性入口
third_party/   固定提交的开源基线源码及本项目兼容性修改
data/          本地数据，不提交
artifacts/     仅提交小型审计/指标；证据张量不提交
runs/          checkpoint、日志和预测结果，不提交
```

## 数据与大文件政策

MMPOI/Amazon 原始数据受各自发布方条款约束，因此仓库只提供官方下载和确定性
预处理脚本，不重复分发数据。训练 checkpoint（83–203MB）、图矩阵和逐评论证据
张量也不提交；它们均可由本文档中的命令重新生成。

## 第三方代码

第三方源码固定版本及本地修改见 [`docs/SOURCE_LOCK.md`](docs/SOURCE_LOCK.md)。
MMPOI 原论文与数据使用时请引用其 WWW 2024 论文。第三方目录保留各项目原始
许可证和引用信息。
