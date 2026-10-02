#!/usr/bin/env python3
"""Audit Amazon Reviews 2023 All_Beauty for the IER task.

Run with:
  PYTHONPATH=vendor python scripts/audit_amazon_all_beauty.py

The script is read-only with respect to raw data. It writes JSON and Markdown
reports under artifacts/data_audit/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/amazon_reviews_2023/all_beauty"
OUT = ROOT / "artifacts/data_audit"


def scalar(con: duckdb.DuckDBPyConnection, query: str, params: list[Any] | None = None) -> Any:
    return con.execute(query, params or []).fetchone()[0]


def row_dict(con: duckdb.DuckDBPyConnection, query: str) -> dict[str, Any]:
    cur = con.execute(query)
    names = [d[0] for d in cur.description]
    return dict(zip(names, cur.fetchone()))


def qpath(path: Path) -> str:
    return str(path).replace("'", "''")


def iterative_k_core(con: duckdb.DuckDBPyConnection, k: int) -> dict[str, int]:
    table = f"core_{k}_0"
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        SELECT user_id, parent_asin, min(timestamp) AS timestamp
        FROM reviews
        WHERE user_id IS NOT NULL AND parent_asin IS NOT NULL
        GROUP BY user_id, parent_asin
        """
    )
    previous = scalar(con, f"SELECT count(*) FROM {table}")
    iteration = 0
    while True:
        nxt = f"core_{k}_{iteration + 1}"
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {nxt} AS
            WITH valid_users AS (
                SELECT user_id FROM {table} GROUP BY user_id HAVING count(*) >= {k}
            ), valid_items AS (
                SELECT parent_asin FROM {table} GROUP BY parent_asin HAVING count(*) >= {k}
            )
            SELECT x.* FROM {table} x
            SEMI JOIN valid_users USING (user_id)
            SEMI JOIN valid_items USING (parent_asin)
            """
        )
        current = scalar(con, f"SELECT count(*) FROM {nxt}")
        con.execute(f"DROP TABLE {table}")
        table = nxt
        iteration += 1
        if current == previous:
            break
        previous = current
    result = row_dict(
        con,
        f"""
        SELECT count(*) AS interactions,
               count(DISTINCT user_id) AS users,
               count(DISTINCT parent_asin) AS items
        FROM {table}
        """,
    )
    result["iterations"] = iteration
    con.execute(f"DROP TABLE {table}")
    return result


def split_stats(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "valid", "test"):
        result[split] = row_dict(
            con,
            f"""
            SELECT count(*) AS interactions,
                   count(DISTINCT user_id) AS users,
                   count(DISTINCT parent_asin) AS items,
                   min(timestamp) AS min_timestamp,
                   max(timestamp) AS max_timestamp
            FROM official_{split}
            """,
        )
    result["union"] = row_dict(
        con,
        """
        SELECT count(*) AS interactions,
               count(DISTINCT user_id) AS users,
               count(DISTINCT parent_asin) AS items
        FROM (
          SELECT * FROM official_train
          UNION ALL SELECT * FROM official_valid
          UNION ALL SELECT * FROM official_test
        )
        """,
    )
    result["chronology_violations"] = scalar(
        con,
        """
        WITH t AS (SELECT user_id, max(timestamp) ts FROM official_train GROUP BY user_id),
             v AS (SELECT user_id, max(timestamp) ts FROM official_valid GROUP BY user_id),
             e AS (SELECT user_id, max(timestamp) ts FROM official_test GROUP BY user_id)
        SELECT count(*) FROM t JOIN v USING(user_id) JOIN e USING(user_id)
        WHERE NOT (t.ts <= v.ts AND v.ts <= e.ts)
        """,
    )
    return result


def main() -> None:
    required = [RAW / "reviews.parquet", RAW / "metadata.parquet"]
    required += [RAW / f"official_5core_last_out_{s}.csv" for s in ("train", "valid", "test")]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise SystemExit("Missing files:\n" + "\n".join(missing))

    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute(f"CREATE VIEW reviews AS SELECT * FROM read_parquet('{qpath(RAW / 'reviews.parquet')}')")
    con.execute(f"CREATE VIEW meta AS SELECT * FROM read_parquet('{qpath(RAW / 'metadata.parquet')}')")
    for split in ("train", "valid", "test"):
        con.execute(
            f"CREATE VIEW official_{split} AS SELECT * FROM "
            f"read_csv_auto('{qpath(RAW / f'official_5core_last_out_{split}.csv')}', header=true)"
        )

    report: dict[str, Any] = {
        "dataset": "McAuley-Lab/Amazon-Reviews-2023: All_Beauty",
        "raw_reviews": row_dict(
            con,
            """
            SELECT count(*) AS rows,
                   count(DISTINCT user_id) AS users,
                   count(DISTINCT parent_asin) AS parent_items,
                   count(DISTINCT asin) AS variants,
                   count(*) - count(DISTINCT (user_id, parent_asin)) AS duplicate_user_item_rows,
                   min(timestamp) AS min_timestamp,
                   max(timestamp) AS max_timestamp,
                   count(*) FILTER (WHERE text IS NOT NULL AND length(trim(text)) > 0) AS rows_with_text,
                   count(*) FILTER (WHERE array_length(images) > 0) AS rows_with_review_images,
                   sum(array_length(images)) AS review_image_units,
                   count(*) FILTER (WHERE verified_purchase) AS verified_rows
            FROM reviews
            """,
        ),
        "metadata": row_dict(
            con,
            """
            SELECT count(*) AS rows,
                   count(DISTINCT parent_asin) AS parent_items,
                   count(*) FILTER (WHERE title IS NOT NULL AND length(trim(title)) > 0) AS items_with_title,
                   count(*) FILTER (WHERE array_length(features) > 0) AS items_with_features,
                   count(*) FILTER (WHERE array_length(description) > 0) AS items_with_description,
                   count(*) FILTER (WHERE details IS NOT NULL AND details NOT IN ('', '{}', 'None')) AS items_with_details,
                   count(*) FILTER (WHERE list_count(list_filter(images.large, x -> x IS NOT NULL)) > 0) AS items_with_images,
                   count(*) FILTER (WHERE list_count(list_filter(images.large, x -> x IS NOT NULL)) >= 2) AS items_with_2plus_images,
                   count(*) FILTER (WHERE list_count(list_filter(images.large, x -> x IS NOT NULL)) >= 5) AS items_with_5plus_images,
                   avg(list_count(list_filter(images.large, x -> x IS NOT NULL))) AS mean_product_images,
                   approx_quantile(list_count(list_filter(images.large, x -> x IS NOT NULL)), 0.5) AS median_product_images,
                   max(list_count(list_filter(images.large, x -> x IS NOT NULL))) AS max_product_images
            FROM meta
            """,
        ),
        "official_5core_last_out": split_stats(con),
    }

    report["k_core"] = {str(k): iterative_k_core(con, k) for k in (2, 3, 5)}

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW item_evidence AS
        WITH review_stats AS (
          SELECT parent_asin,
                 count(*) AS review_count,
                 count(*) FILTER (WHERE text IS NOT NULL AND length(trim(text)) > 0) AS text_review_count,
                 count(*) FILTER (WHERE array_length(images) > 0) AS image_review_count,
                 sum(array_length(images)) AS review_image_count
          FROM reviews GROUP BY parent_asin
        ), meta_stats AS (
          SELECT parent_asin,
                 list_count(list_filter(images.large, x -> x IS NOT NULL)) AS product_image_count,
                 CASE WHEN title IS NOT NULL AND length(trim(title)) > 0 THEN 1 ELSE 0 END AS has_title,
                 CASE WHEN array_length(features) > 0 OR array_length(description) > 0 OR
                           (details IS NOT NULL AND details NOT IN ('', '{}', 'None')) THEN 1 ELSE 0 END AS has_catalog_text
          FROM meta
        )
        SELECT r.*, coalesce(m.product_image_count, 0) AS product_image_count,
               coalesce(m.has_title, 0) AS has_title,
               coalesce(m.has_catalog_text, 0) AS has_catalog_text
        FROM review_stats r LEFT JOIN meta_stats m USING(parent_asin)
        """
    )

    report["interaction_item_evidence"] = row_dict(
        con,
        """
        SELECT count(*) AS interacted_items,
               count(*) FILTER (WHERE product_image_count > 0) AS with_product_images,
               count(*) FILTER (WHERE product_image_count >= 2) AS with_2plus_product_images,
               count(*) FILTER (WHERE text_review_count >= 2) AS with_2plus_text_reviews,
               count(*) FILTER (WHERE image_review_count > 0) AS with_ugc_review_images,
               count(*) FILTER (WHERE product_image_count >= 2 AND text_review_count >= 2) AS with_multimodal_multi_evidence,
               count(*) FILTER (WHERE has_catalog_text = 1 AND text_review_count >= 2) AS with_catalog_and_ugc_text,
               avg(review_count) AS mean_reviews_per_item,
               approx_quantile(review_count, 0.5) AS median_reviews_per_item,
               max(review_count) AS max_reviews_per_item
        FROM item_evidence
        """,
    )

    report["official_5core_item_evidence"] = row_dict(
        con,
        """
        WITH used AS (
          SELECT DISTINCT parent_asin FROM (
            SELECT parent_asin FROM official_train
            UNION ALL SELECT parent_asin FROM official_valid
            UNION ALL SELECT parent_asin FROM official_test
          )
        )
        SELECT count(*) AS items,
               count(e.parent_asin) AS items_joined_to_raw,
               count(*) FILTER (WHERE product_image_count > 0) AS with_product_images,
               count(*) FILTER (WHERE product_image_count >= 2) AS with_2plus_product_images,
               count(*) FILTER (WHERE text_review_count >= 2) AS with_2plus_text_reviews,
               count(*) FILTER (WHERE image_review_count > 0) AS with_ugc_review_images,
               count(*) FILTER (WHERE product_image_count >= 2 AND text_review_count >= 2) AS with_multimodal_multi_evidence
        FROM used u LEFT JOIN item_evidence e USING(parent_asin)
        """,
    )

    # Leakage diagnostics: every target interaction has its own review; using the
    # complete review pool therefore leaks the label by construction.
    report["leakage"] = {
        "target_review_leakage_if_full_pool_used": True,
        "future_review_leakage_if_full_pool_used": True,
        "required_policy": (
            "For a target (user, item, t), candidate UGC may only contain reviews with timestamp < t; "
            "exclude the target user's review. Catalog evidence is retained but treated as a static snapshot limitation."
        ),
    }

    raw = report["raw_reviews"]
    official = report["official_5core_last_out"]["union"]
    ev = report["official_5core_item_evidence"]
    decision_reasons: list[str] = []
    if official["users"] < 1000:
        decision_reasons.append(f"official 5-core has only {official['users']} users (<1000)")
    if official["interactions"] < 30000:
        decision_reasons.append(f"official 5-core has only {official['interactions']} interactions (<30000)")
    coverage = (ev["with_multimodal_multi_evidence"] / ev["items"]) if ev["items"] else 0.0
    if coverage < 0.5:
        decision_reasons.append(
            f"only {coverage:.1%} of official items have >=2 product images and >=2 text reviews"
        )
    report["decision"] = {
        "official_5core": "NO-GO" if decision_reasons else "GO",
        "official_5core_reasons": decision_reasons,
        "raw_data_for_custom_core_audit": "GO" if raw["rows"] >= 100000 else "NO-GO",
        "image_url_reachability_pending": True,
        "final_dataset_decision": "PENDING_IMAGE_URL_AUDIT",
    }

    json_path = OUT / "amazon_all_beauty_audit.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    def pct(n: int, d: int) -> str:
        return f"{n / d:.1%}" if d else "n/a"

    md = f"""# Amazon Reviews 2023 / All_Beauty 数据可用性审计

## 结论

- 官方 5-core：**{report['decision']['official_5core']}**。
- 原始数据继续做自定义 core 与时间切分审计：**{report['decision']['raw_data_for_custom_core_audit']}**。
- 最终是否作为 IER 主数据：**{report['decision']['final_dataset_decision']}**（仍需图片 URL 抽样可达率）。

官方 5-core 的否决理由：{'; '.join(decision_reasons) if decision_reasons else '无'}。

## 原始交互与 UGC

| 指标 | 数值 |
|---|---:|
| 评论/交互行 | {raw['rows']:,} |
| 用户 | {raw['users']:,} |
| parent item | {raw['parent_items']:,} |
| 重复 user-item 行 | {raw['duplicate_user_item_rows']:,} |
| 有文本的评论 | {raw['rows_with_text']:,} ({pct(raw['rows_with_text'], raw['rows'])}) |
| 有评论图片的评论 | {raw['rows_with_review_images']:,} ({pct(raw['rows_with_review_images'], raw['rows'])}) |
| 评论图片单元 | {raw['review_image_units']:,} |

## Core 过滤后规模

| Core | 用户 | 物品 | 交互 | 迭代轮数 |
|---:|---:|---:|---:|---:|
"""
    for k, stats in report["k_core"].items():
        md += f"| {k} | {stats['users']:,} | {stats['items']:,} | {stats['interactions']:,} | {stats['iterations']} |\n"

    md += f"""

## 候选证据覆盖（官方 5-core 涉及的物品）

| 指标 | 数值 |
|---|---:|
| 物品 | {ev['items']:,} |
| 可连接原始记录 | {ev['items_joined_to_raw']:,} |
| 至少 1 张商品图 | {ev['with_product_images']:,} ({pct(ev['with_product_images'], ev['items'])}) |
| 至少 2 张商品图 | {ev['with_2plus_product_images']:,} ({pct(ev['with_2plus_product_images'], ev['items'])}) |
| 至少 2 条 UGC 文本评论 | {ev['with_2plus_text_reviews']:,} ({pct(ev['with_2plus_text_reviews'], ev['items'])}) |
| 至少 1 条 UGC 图片评论 | {ev['with_ugc_review_images']:,} ({pct(ev['with_ugc_review_images'], ev['items'])}) |
| 同时有 ≥2 商品图与 ≥2 UGC 文本 | {ev['with_multimodal_multi_evidence']:,} ({pct(ev['with_multimodal_multi_evidence'], ev['items'])}) |

## 必须执行的防泄漏规则

预测 `(user, item, t)` 时，候选物品的评论证据只能来自 `timestamp < t` 的其他历史评论，并排除目标用户自己的目标评论。直接使用全量评论池会同时产生目标标签泄漏与未来信息泄漏。商品目录图/属性只能视为静态快照，并在论文限制中说明缺少历史版本。

## 下一步判断

1. 抽样检测商品图和评论图 URL 的 HTTP 可达率、内容类型与去重后有效图片数。
2. 对 2-core/3-core 中的物品重新统计“在训练截止时间之前”的证据覆盖。
3. 只有规模、时序证据覆盖和图片可达率同时通过，才生成训练数据并开始基线复现。
"""
    (OUT / "amazon_all_beauty_audit.md").write_text(md, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\nWrote {json_path}")


if __name__ == "__main__":
    main()
