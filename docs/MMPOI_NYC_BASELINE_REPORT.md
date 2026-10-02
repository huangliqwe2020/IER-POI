# MMPOI NYC 数据审计与基线复现报告

## 结论

MMPOI NYC 可以用于 UGC 证据鲁棒多维推荐方案，数据审计结论为 **GO**。它同时提供用户轨迹、POI 图片、评论文本、元数据、地理信息以及三种官方聚合向量，适合先复现多模态下一 POI 推荐，再把“聚合单向量”替换成证据集合建模、证据选择和扰动鲁棒训练。

## 数据与完整性

- 官方 processed 包：`NYC_processed.zip`，SHA256 `3998e67a1572dc7d368df17de6b94c399cbd0b3eb46f12c7e2fa55642150baca`
- 官方 raw 包：`NYC_raw.zip`，SHA256 `24591ecd81b9fcb53d76b1e918ebb4124a7d96b06fb5241b59d4bfda85e623e9`
- 训练：93,147 次签到、1,080 用户、4,618 POI、25,387 轨迹
- 测试：21,987 次签到、1,080 用户、4,083 POI、7,159 轨迹
- 测试集中有 19 个训练未见 POI；官方代码过滤后共移除 138 个冷启动事件
- 图片：45,226 张能匹配训练 POI；3,428 个 POI 至少有 2 张图，3,183 个至少有 5 张图
- 评论：105,432 条能匹配训练 POI；3,529 个 POI 至少有 2 条评论，3,038 个至少有 5 条评论
- 官方 image/review/meta 聚合向量均覆盖全部 4,618 个训练 POI

详细机器可读审计见 `artifacts/data_audit/mmpoi_nyc_audit.json`。

## MMPOI 结果

训练使用官方 NYC 切分、seed=618、batch=256 和默认维度，CPU 全量训练 49 个 epoch。按修正后的标准早停规则，epoch 43 后连续五轮综合分数 `HR@20 + NDCG@20` 未刷新，在 epoch 48 完整写入后结束。

### 可加载的最佳 checkpoint（官方保存准则：`4*HR@1 + HR@20`）

- epoch：32（从 0 计数）
- HR@1：0.182334
- HR@5：0.500940
- HR@10：0.625426
- HR@20：0.700256
- mAP@20：0.317036
- NDCG@20：0.405568
- MRR：0.319469

独立 eval-only 重新前向计算得到完全相同的排序指标。

### 最高 HR@20 轮

- epoch：43
- HR@1：0.166030
- HR@5：0.494630
- HR@10：0.626328
- HR@20：0.717383
- mAP@20：0.307642
- NDCG@20：0.401846
- MRR：0.309855

论文表 3 的 NYC MMPOI 为 HR@20=0.710037、NDCG@20=0.423524。本次最高 HR@20 与论文非常接近并略高 0.00735；checkpoint 的 HR@20 低 0.00978。NDCG@20 比论文低约 0.0180（checkpoint）到 0.0217（最高 HR 轮）。因此复现处于合理区间，但不能声称逐项完全一致。

## 经典 sanity baselines

评估协议为过滤训练未见 POI、至少保留 3 个点的测试轨迹，在最后时间步预测下一个 POI，共评估 3,103 条轨迹。

| 模型 | HR@1 | HR@5 | HR@10 | HR@20 | NDCG@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| GlobalPop | 0.0077 | 0.0280 | 0.0432 | 0.0561 | 0.0258 | 0.0201 |
| Markov-1 | 0.1950 | 0.4115 | 0.4763 | 0.5179 | 0.3426 | 0.2906 |
| UserPop | 0.2056 | 0.5430 | 0.6755 | 0.7506 | 0.4446 | 0.3546 |

UserPop 是非常强的下限，说明 NYC 存在显著的用户重复访问偏好。后续创新模型应同时报告全体样本、非重复访问样本、长尾 POI 和证据扰动子集，避免平均指标被重复访问模式主导。

## 对官方代码的必要修复

复现基于仓库提交 `32f9c76e8768af3b268f0c95679bb7e0c739f766`，发现并修复：

1. A+ 多模态邻接计算后被下一行 `A+A` 覆盖。
2. `pad_sequence(batch_first=True)` 的张量被送入 sequence-first Transformer，导致不同轨迹互相注意；同时 causal mask 误用 batch 大小。
3. 稠密保存的稀疏图每个 batch 重复做稠密乘法；改为数值等价的 sparse tensor。
4. 默认 OneHot 配置仍强制导入未使用的 `sentence_transformers`。
5. checkpoint 遗漏 image/review/meta/geo 和融合层权重，无法独立评估。
6. early stopping 只在两个浮点分数完全相等时累计耐心，几乎永远不会提前停止。
7. 增加 eval-only checkpoint 重评入口。

这些修复恢复了论文描述的 A+ 多模态图和逐轨迹 causal Transformer 语义。未经修复的仓库结果不能作为可靠基线。

## 复现入口

```bash
cd IER-POI

# 全流程：审计、建图、训练、静态校验、checkpoint 重新评估
scripts/run_mmpoi_nyc_baseline.sh

# 只重评现有最佳 checkpoint
scripts/evaluate_mmpoi_nyc_checkpoint.sh

# 校验 checkpoint 完整性和结果文件
python scripts/test_mmpoi_nyc.py \
  --run-dir runs/mmpoi/nyc_fixed_seed618 --min-hr20 0.65

# 经典基线
python scripts/run_mmpoi_classical_baselines.py
```

## 后续创新实验建议

MMPOI raw 数据满足多证据集合建模，但没有 helpful vote 字段。正式的“关键证据正确性”验证可使用：用户历史类别/属性匹配弱标签、多模型一致性弱标签，以及一小批人工标注测试集。建议保留官方聚合向量作为 MMPOI baseline，新模型直接读取 POI 级多图片与多评论集合，并增加 evidence selection、噪声注入、证据删除和跨模态冲突四类实验。
