#!/usr/bin/env python3
"""Audit the formal Amazon Musical_Instruments candidate for IER."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/amazon_reviews_2023/musical_instruments"
OUT = ROOT / "artifacts/data_audit"


def row(con: duckdb.DuckDBPyConnection, sql: str) -> dict[str, object]:
    cur = con.execute(sql)
    return dict(zip([d[0] for d in cur.description], cur.fetchone()))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=4")
    raw = str(RAW).replace("'", "''")
    con.execute(
        f"CREATE VIEW interactions AS SELECT * FROM read_csv_auto("
        f"'{raw}/official_5core_rating_only.csv.gz', header=true)"
    )
    con.execute(f"CREATE VIEW meta AS SELECT * FROM read_parquet('{raw}/metadata-*.parquet')")

    report = {
        "dataset": "Amazon Reviews 2023 Musical_Instruments official 5-core",
        "interactions": row(
            con,
            """
            SELECT count(*) AS "rows", count(DISTINCT user_id) users,
                   count(DISTINCT parent_asin) items,
                   count(*) - count(DISTINCT (user_id, parent_asin)) duplicate_user_item_rows,
                   min(timestamp) min_timestamp, max(timestamp) max_timestamp,
                   avg(rating) mean_rating
            FROM interactions
            """,
        ),
        "sequence": row(
            con,
            """
            WITH x AS (SELECT user_id, count(*) n FROM interactions GROUP BY user_id)
            SELECT min(n) min_len, avg(n) mean_len,
                   approx_quantile(n, 0.5) median_len,
                   approx_quantile(n, 0.9) p90_len,
                   max(n) max_len,
                   count(*) FILTER (WHERE n >= 10) users_10plus,
                   count(*) FILTER (WHERE n >= 20) users_20plus
            FROM x
            """,
        ),
        "metadata_all": row(
            con,
            """
            SELECT count(*) AS "rows", count(DISTINCT parent_asin) items,
                   count(*) FILTER (WHERE list_count(list_filter(images.large, x -> x IS NOT NULL)) > 0) with_images,
                   count(*) FILTER (WHERE list_count(list_filter(images.large, x -> x IS NOT NULL)) >= 2) with_2plus_images,
                   count(*) FILTER (WHERE list_count(list_filter(images.large, x -> x IS NOT NULL)) >= 5) with_5plus_images,
                   avg(list_count(list_filter(images.large, x -> x IS NOT NULL))) mean_images,
                   approx_quantile(list_count(list_filter(images.large, x -> x IS NOT NULL)), 0.5) median_images,
                   count(*) FILTER (WHERE array_length(features) > 0) with_features,
                   count(*) FILTER (WHERE array_length(description) > 0) with_description,
                   count(*) FILTER (WHERE details IS NOT NULL AND details NOT IN ('', '{}', 'None')) with_details
            FROM meta
            """,
        ),
    }
    con.execute(
        "CREATE TEMP VIEW used AS SELECT DISTINCT parent_asin FROM interactions"
    )
    report["used_item_evidence"] = row(
        con,
        """
        SELECT count(*) items,
               count(m.parent_asin) joined_metadata,
               count(*) FILTER (WHERE list_count(list_filter(images.large, x -> x IS NOT NULL)) > 0) with_images,
               count(*) FILTER (WHERE list_count(list_filter(images.large, x -> x IS NOT NULL)) >= 2) with_2plus_images,
               count(*) FILTER (WHERE list_count(list_filter(images.large, x -> x IS NOT NULL)) >= 5) with_5plus_images,
               count(*) FILTER (WHERE array_length(features) > 0 OR array_length(description) > 0 OR
                                     (details IS NOT NULL AND details NOT IN ('', '{}', 'None'))) with_catalog_text,
               count(*) FILTER (WHERE list_count(list_filter(images.large, x -> x IS NOT NULL)) >= 2 AND
                                     (array_length(features) > 0 OR array_length(description) > 0 OR
                                      (details IS NOT NULL AND details NOT IN ('', '{}', 'None')))) multimodal_multi_evidence,
               avg(list_count(list_filter(images.large, x -> x IS NOT NULL))) mean_images,
               approx_quantile(list_count(list_filter(images.large, x -> x IS NOT NULL)), 0.5) median_images,
               max(list_count(list_filter(images.large, x -> x IS NOT NULL))) max_images
        FROM used u LEFT JOIN meta m USING(parent_asin)
        """,
    )
    report["exact_url_redundancy"] = row(
        con,
        """
        WITH x AS (
          SELECT u.parent_asin, count(url) urls, count(DISTINCT url) distinct_urls
          FROM used u JOIN meta m USING(parent_asin), UNNEST(m.images.large) t(url)
          WHERE url IS NOT NULL GROUP BY u.parent_asin
        )
        SELECT count(*) items_with_images,
               count(*) FILTER (WHERE urls > distinct_urls) items_with_exact_duplicate_urls,
               sum(urls - distinct_urls) duplicate_url_units
        FROM x
        """,
    )

    ugc_path = OUT / "amazon_musical_instruments_ugc_stream.json"
    image_path = OUT / "amazon_musical_instruments_image_url_audit.json"
    ugc = json.loads(ugc_path.read_text(encoding="utf-8")) if ugc_path.is_file() else None
    image = json.loads(image_path.read_text(encoding="utf-8")) if image_path.is_file() else None
    report["ugc_exact_match"] = ugc
    report["image_url_audit"] = image["summary"] if image else None

    inter = report["interactions"]
    ev = report["used_item_evidence"]
    coverage = ev["multimodal_multi_evidence"] / ev["items"] if ev["items"] else 0
    hard_checks = {
        "users_ge_1000": inter["users"] >= 1000,
        "items_ge_300": inter["items"] >= 300,
        "interactions_ge_30000": inter["rows"] >= 30000,
        "metadata_join_ge_95pct": ev["joined_metadata"] / ev["items"] >= 0.95,
        "multimodal_multi_evidence_ge_50pct": coverage >= 0.5,
        "ugc_exact_keys_complete": bool(
            ugc and ugc["matched_rows"] == ugc["distinct_exact_keys"]
            == ugc["expected_exact_5core_rows"] and ugc["missing_exact_rows"] == 0
        ),
        "ugc_text_ge_95pct": bool(
            ugc and ugc["rows_with_text"] / ugc["matched_rows"] >= 0.95
        ),
        "catalog_image_url_ge_90pct": bool(
            image and image["summary"]["catalog"]["reachability"] >= 0.9
        ),
        "ugc_image_url_ge_90pct": bool(
            image and image["summary"]["ugc_review"]["reachability"] >= 0.9
        ),
    }
    final = "GO" if all(hard_checks.values()) else (
        "PENDING_UGC_AND_URL_AUDIT" if not ugc or not image else "NO-GO"
    )
    report["decision"] = {
        "catalog_and_interactions": "GO" if all(hard_checks.values()) else "NO-GO",
        "hard_checks": hard_checks,
        "multimodal_multi_evidence_coverage": coverage,
        "ugc_review_text_pending": ugc is None,
        "image_url_reachability_pending": image is None,
        "final": final,
    }
    (OUT / "amazon_musical_instruments_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    seq = report["sequence"]
    md = f"""# Amazon Musical_Instruments 数据可用性审计

## 阶段结论

- 交互与目录证据：**{report['decision']['catalog_and_interactions']}**
- 最终结论：**{report['decision']['final']}**
- 多图 + 目录文本覆盖率：**{coverage:.1%}**

| 指标 | 数值 |
|---|---:|
| 用户 | {inter['users']:,} |
| 物品 | {inter['items']:,} |
| 交互 | {inter['rows']:,} |
| 序列最小/中位/均值/P90/最大长度 | {seq['min_len']} / {seq['median_len']} / {seq['mean_len']:.2f} / {seq['p90_len']} / {seq['max_len']} |
| 有元数据的实验物品 | {ev['joined_metadata']:,} / {ev['items']:,} |
| 至少 2 张商品图 | {ev['with_2plus_images']:,} / {ev['items']:,} |
| 至少 5 张商品图 | {ev['with_5plus_images']:,} / {ev['items']:,} |
| 有目录文本 | {ev['with_catalog_text']:,} / {ev['items']:,} |
| 同时有 ≥2 图和文本 | {ev['multimodal_multi_evidence']:,} / {ev['items']:,} |
| 精确匹配原始评论 | {ugc['distinct_exact_keys'] if ugc else 'pending'} / {inter['rows']:,} |
| 商品图 URL 固定抽样可达率 | {image['summary']['catalog']['reachability'] if image else 'pending'} |
| UGC 图 URL 固定抽样可达率 | {image['summary']['ugc_review']['reachability'] if image else 'pending'} |

最终判定同时要求：交互和元数据规模通过、511,836 个官方键均精确关联原始评论、文本覆盖率不低于 95%，且商品图与 UGC 图固定抽样可达率均不低于 90%。
"""
    (OUT / "amazon_musical_instruments_audit.md").write_text(md, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
