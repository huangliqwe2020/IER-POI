# 数据与代码版本锁定

## 开源代码

| 目录 | 上游 | 固定提交 |
|---|---|---|
| `third_party/AmazonReviews2023` | `hyp1231/AmazonReviews2023` | `b18fdf54bd46013d60799684f7a4eb80d8501d1a` |
| `third_party/MMRec` | `enoche/MMRec` | `c68ee7c9b0de1ca7c5150258322edb4994be29d9` |
| `third_party/RecBole` | `RUCAIBox/RecBole` | `7b02be5ec80a88310f2d04a27a82adfcbb5dc211` |
| `third_party/MMPOI` | `zzmylq/MMPOI` | `32f9c76e8768af3b268f0c95679bb7e0c739f766` |

MMPOI、MMRec 和 AmazonReviews2023 源码快照随本仓库提交，同时保留原始
许可证和引用信息。RecBole 当前不在已报告实验的执行路径中，只锁定上游提交，
不重复 vendoring。各目录中的嵌套 `.git` 元数据、数据文件和缓存不提交。

MMRec 有四项有意的本地改动：两个 Amazon 数据配置；NumPy 2 的 `np.float`
兼容；SciPy 1.14+ 下 LightGCN 使用公开 COO/CSR API 构造完全相同的二部图
邻接矩阵。

MMPOI 的本地改动包括：延迟加载可选的 `sentence-transformers`；恢复论文预期的
`A+` 图；稀疏图计算；修复 Transformer 的 batch/mask/位置编码；完整 checkpoint
与 eval-only；标准早停；设备选择；以及 IER 的逐评论证据选择、可靠性/冗余聚合、
证据 dropout、重复/敏感性约束和五类扰动入口。

## Amazon Reviews 2023

- `All_Beauty` reviews parquet：Hugging Face 提交 `749931181c9cdd70bfa6408d8fe8b4ddd6107248`，SHA256 `b40ae565f1073e0b4960e2f424fe0c82f3d270b7c4d00bc9272ddd6a8b7411d7`。
- `All_Beauty` metadata parquet：提交 `2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e`，SHA256 `14815cf1312a0d847364866e6876c8c73738993469067242870774c372c04387`。
- `Musical_Instruments` metadata 两片：同一提交 `2b6d039...`，SHA256 为 `867567b573c4dc2860d476d0ac2de30bf9d4b3f77ae18e925d551d8d2170db6d` 与 `d9bf76dd54000c544d81939c1ba7a788c6553ae32745066a8dad668b75690544`。
- `Musical_Instruments` 官方 5-core 交互：SHA256 `e1cabe91651102ada04aee6836e6a890cd557386389da6ee9c3c80e8aa507260`。
- 从官方原始 JSONL 精确匹配 5-core 键得到的评论证据：511,836 个唯一键，SHA256 `5d8f2d694a73e7f69f02b97f4dd17256a419515a386cdee80b5646efe8e0d403`。

数据只用于研究复现；原始文件和处理数据由 `.gitignore` 排除，不重复分发。

## MMPOI NYC

- processed Google Drive ID：`17FbNvkO74xub6AeT2fpm938qDqUDB-04`，SHA256
  `3998e67a1572dc7d368df17de6b94c399cbd0b3eb46f12c7e2fa55642150baca`。
- raw Google Drive ID：`1v0BvKs46ixUf1CgjRlk9MY5JsFL7ILxC`，SHA256
  `24591ecd81b9fcb53d76b1e918ebb4124a7d96b06fb5241b59d4bfda85e623e9`。
- 下载、解压、校验和数据链接统一由 `scripts/download_mmpoi_nyc.sh` 完成。
