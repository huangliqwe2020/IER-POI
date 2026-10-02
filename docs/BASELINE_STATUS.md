# 基线复现状态

## 正式通用场景

数据为 Amazon Reviews 2023 `Musical_Instruments` 官方 5-core。按用户时间排序留最后两条作验证/测试，仅在训练物品上做 full-ranking，排除用户训练期已交互物品。随机种子 `20260930`，以验证集 Recall@20 选轮次。

| 模型 | 训练设置 | Test Recall@10 | Test Recall@20 | Test NDCG@10 | Test NDCG@20 |
|---|---|---:|---:|---:|---:|
| MostPop | deterministic full ranking | 0.0258 | 0.0400 | 0.0133 | 0.0168 |
| BPR | MMRec, 30 epochs, batch=2048, dim=64 | 0.0375 | 0.0578 | 0.0193 | 0.0244 |
| LightGCN | MMRec, 15 epochs, batch=8192, 2 layers, dim=64 | 0.0350 | 0.0554 | 0.0176 | 0.0227 |

MostPop 严格时间子集（去掉 6 个边界同时间戳用户）的测试 Recall@20 为 0.03995，和全 warm 集几乎一致。BPR/LightGCN 当前由 MMRec 原生 evaluator 评估全部 warm 用户；后续 IER 的统一 evaluator 同时报告全 warm 与严格时间子集。

日志：

- `third_party/MMRec/src/log/BPR-amazon_musical_instruments-Sep-30-2026-01-32-02.log`
- `third_party/MMRec/src/log/LightGCN-amazon_musical_instruments-Sep-30-2026-01-38-07.log`

## 快速 pilot（不进入论文主表）

`All_Beauty` 2-core pilot 的 Test Recall@20：MostPop 0.0322、BPR 0.0170、LightGCN 0.0233。该数据个人历史过短、流行度主导，只用于验证代码路径和干预实现。

## 兼容性改动

对 MMRec 做了两处不改变算法定义的最小修补：NumPy 2 移除的 `np.float` 改为 `float`；SciPy 1.14+ 移除私有 DOK `_update` 后，LightGCN 改用公开 COO/CSR API 构图。
