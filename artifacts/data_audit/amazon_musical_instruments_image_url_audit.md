# Amazon All_Beauty 图片 URL 可达性抽样

| 来源 | 样本 | 有效图片 | 可达率 |
|---|---:|---:|---:|
| catalog | 100 | 100 | 100.0% |
| ugc_review | 100 | 100 | 100.0% |

成功条件为 HTTP 200/206 且响应 `Content-Type` 为 `image/*`。抽样由 URL 哈希固定，可重复运行。
