# 完整复现说明

## 1. 系统要求

报告结果在 Linux、Python 3.10.19、PyTorch 2.5.1+cu118 上得到。

- 推荐：NVIDIA GPU，至少 8GB 可用显存；本次使用 RTX 3090。
- CPU 可以运行，但完整 200-epoch MMPOI 训练明显更慢。
- 内存建议 16GB 以上。
- 磁盘建议预留 8GB；原始数据、解压数据、图、证据和 checkpoint 均不入 Git。
- 系统命令：`git`、`unzip`、`sha256sum`。

安装：

```bash
git clone git@github.com:huangliqwe2020/IER-POI.git
cd IER-POI
conda env create -f environment.yml
conda activate ier-poi
python scripts/check_environment.py
```

如果机器不能使用 CUDA 11.8 wheel，可先创建 Python 3.10 环境，再按 PyTorch
官方说明安装对应 CUDA/CPU 的 PyTorch 2.5.1，最后安装其余依赖。

所有 shell 入口都使用当前环境中的 `python`。也可以显式指定：

```bash
PYTHON=/path/to/python ./scripts/run_mmpoi_nyc_baseline.sh 200 run_name cuda:0
```

## 2. MMPOI-NYC 数据

```bash
./scripts/download_mmpoi_nyc.sh
```

下载并校验：

| 文件 | Google Drive ID | SHA256 |
|---|---|---|
| processed | `17FbNvkO74xub6AeT2fpm938qDqUDB-04` | `3998e67a1572dc7d368df17de6b94c399cbd0b3eb46f12c7e2fa55642150baca` |
| raw | `1v0BvKs46ixUf1CgjRlk9MY5JsFL7ILxC` | `24591ecd81b9fcb53d76b1e918ebb4124a7d96b06fb5241b59d4bfda85e623e9` |

目录结构：

```text
data/raw/mmpoi/
├── downloads/{NYC_processed.zip,NYC_raw.zip}
├── nyc_processed/NYC/
└── nyc_raw/NYC/
```

然后执行审计和缺失图构建：

```bash
python scripts/prepare_mmpoi_nyc.py
```

该脚本验证官方 train/test、核查多模态覆盖率、在缺失时构建 POI/地理图、
创建 `third_party/MMPOI/NYC` 数据链接，并输出审计 JSON。预期结论为 `GO`：
4,618 个训练 POI，93,147 条训练签到，25,387 条训练轨迹，三种官方聚合
模态向量覆盖所有训练 POI。

## 3. 透明的经典基线

```bash
python scripts/run_mmpoi_classical_baselines.py
```

当前结果：GlobalPop HR@20≈0.0561、Markov1 HR@20≈0.5179、
UserPop HR@20≈0.7506。UserPop 与 MMPOI 的候选约束不同，只作为数据合理性
检查，不能替代论文模型对比。

## 4. MMPOI 论文基线

精确配置位于 `configs/experiment_manifest.yaml` 和
`third_party/MMPOI/param_parser.py`。一键入口：

```bash
./scripts/run_mmpoi_nyc_baseline.sh 200 nyc_fixed_seed618 cuda:0
```

三个位置参数依次是训练 epoch、运行名和设备。单独重评估：

```bash
./scripts/evaluate_mmpoi_nyc_checkpoint.sh \
  runs/mmpoi/nyc_fixed_seed618/checkpoints/best_epoch.state.pt \
  cuda:0
```

产物完整性测试：

```bash
python scripts/test_mmpoi_nyc.py \
  --run-dir runs/mmpoi/nyc_fixed_seed618 \
  --min-hr20 0.65
```

可加载的最佳 checkpoint（zero-based epoch 32）预期为：HR@20=0.700256、
NDCG@20=0.405568、mAP@20=0.317036、MRR=0.319469。

## 5. IER 逐评论证据预处理

```bash
python scripts/prepare_mmpoi_review_evidence.py
python scripts/test_mmpoi_review_evidence.py
```

步骤包括评论清洗和精确去重、TF-IDF、TruncatedSVD-128、L2 归一化以及
POI 顺序对齐。默认每个 POI 最多保留 20 条评论。预期得到 53,989 条证据，
覆盖 3,933/4,618 个 POI；缺失证据的 POI 回退到 MMPOI 官方评论向量。

## 6. IER 第一阶段训练

```bash
./scripts/run_mmpoi_ier_review.sh 5 evidence_adapter_5ep cuda:0
```

关键参数：Top-K=5、temperature=0.2、redundancy weight=0.2、evidence
dropout=0.1、adapter learning rate=1e-3、随机种子 618。模型包含用户条件查询、
可靠性打分、重复密度惩罚、证据 dropout、重复不变性和关键证据敏感性约束。

## 7. 五种证据扰动

```bash
./scripts/evaluate_mmpoi_ier_robustness.sh \
  runs/mmpoi_ier/evidence_adapter_5ep/checkpoints/best_epoch.state.pt \
  evidence_adapter_5ep \
  cuda:0
```

该命令按顺序评估 `clean`、`duplicate`、`noise`、`missing` 和 `sparse`。
预期 HR@20 在五组均约为 0.7003；NDCG@20 相对 Clean 的变化在
-0.0002 到 +0.0001 之间。

## 8. Amazon 通用场景

```bash
./scripts/download_amazon_musical_instruments.sh
python scripts/audit_amazon_musical_instruments.py
python scripts/prepare_amazon_musical_instruments.py
python scripts/audit_processed_musical_instruments.py
python scripts/build_robustness_bench.py
python scripts/run_mostpop.py --dataset amazon_musical_instruments --strict-time-only
python scripts/run_mmrec_baseline.py --dataset amazon_musical_instruments --model BPR --epochs 30 --eval-step 5
python scripts/run_mmrec_baseline.py --dataset amazon_musical_instruments --model LightGCN --epochs 15 --eval-step 5 --train-batch-size 8192 --eval-batch-size 512
```

`Musical_Instruments` 是正式通用数据；`All_Beauty` 仅作为快速 pilot。

## 9. 复现边界

- checkpoint、数据、图片和证据张量未提交，必须按步骤生成。
- MMPOI 原始评论不含 helpful-vote 字段，不能声称使用 helpful vote 监督。
- 当前 TF-IDF+SVD 是第一阶段冻结证据编码器；最终模型仍需加入更强语义编码、
  用户历史兴趣原型和逐图片证据。
- PyTorch/CUDA 的不同设备可能产生末位浮点差异，主要指标应在相同设备上比较。
