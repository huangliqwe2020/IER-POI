#!/usr/bin/env python3
"""Audit formal Musical Instruments splits and time-safe evidence availability."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/amazon_musical_instruments"
OUT = ROOT / "artifacts/data_audit"


def sql_path(name: str) -> str:
    return str(DATA / name).replace("'", "''")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute(f"CREATE VIEW x AS SELECT * FROM read_parquet('{sql_path('interactions.parquet')}')")
    con.execute(f"CREATE VIEW e AS SELECT * FROM read_parquet('{sql_path('review_evidence.parquet')}')")
    con.execute(f"CREATE VIEW ci AS SELECT * FROM read_parquet('{sql_path('catalog_images.parquet')}')")
    con.execute(f"CREATE VIEW ct AS SELECT * FROM read_parquet('{sql_path('catalog_text.parquet')}')")

    core = con.execute(
        """
        SELECT count(*) AS n, count(DISTINCT user_idx), count(DISTINCT item_idx),
               count(*) FILTER (WHERE x_label=0),
               count(*) FILTER (WHERE x_label=1),
               count(*) FILTER (WHERE x_label=2),
               count(DISTINCT user_idx) FILTER (WHERE x_label=1),
               count(DISTINCT user_idx) FILTER (WHERE x_label=2),
               count(*) FILTER (WHERE x_label IN (1,2) AND NOT warm_item)
        FROM x
        """
    ).fetchone()
    min_train = con.execute(
        "SELECT min(n) FROM (SELECT user_idx,count(*) n FROM x WHERE x_label=0 GROUP BY user_idx)"
    ).fetchone()[0]
    chronology_violations = con.execute(
        """
        WITH s AS (
          SELECT user_idx,
                 max(timestamp) FILTER (WHERE x_label=0) tr,
                 max(timestamp) FILTER (WHERE x_label=1) va,
                 max(timestamp) FILTER (WHERE x_label=2) te
          FROM x GROUP BY user_idx
        )
        SELECT count(*) FILTER (WHERE tr>va OR va>te),
               count(*) FILTER (WHERE tr=va OR va=te)
        FROM s
        """
    ).fetchone()
    repeat_pairs = con.execute(
        """
        SELECT count(*), coalesce(sum(n-1),0)
        FROM (SELECT user_idx,item_idx,count(*) n FROM x GROUP BY ALL HAVING n>1)
        """
    ).fetchone()
    evidence_integrity = con.execute(
        """
        SELECT count(*) AS n,
               count(DISTINCT evidence_id),
               count(DISTINCT (author_user_idx,item_idx,timestamp)),
               count(*) FILTER (WHERE text IS NOT NULL AND length(trim(text))>0),
               count(*) FILTER (WHERE array_length(images)>0)
        FROM e
        """
    ).fetchone()

    # The join below is the exact eligibility rule the IER loader must use.
    con.execute(
        """
        CREATE TEMP TABLE eval_availability AS
        SELECT t.user_idx,t.item_idx,t.timestamp,t.x_label,t.warm_item,
               count(e.evidence_id) AS prior_reviews,
               count(e.evidence_id) FILTER (
                 WHERE e.text IS NOT NULL AND length(trim(e.text))>0
               ) AS prior_text_reviews,
               coalesce(sum(array_length(e.images)),0) AS prior_review_images
        FROM x t LEFT JOIN e
          ON e.item_idx=t.item_idx
         AND e.timestamp<t.timestamp
         AND e.author_user_idx<>t.user_idx
        WHERE t.x_label IN (1,2)
        GROUP BY ALL
        """
    )

    availability = []
    for row in con.execute(
        """
        SELECT x_label, warm_item, count(*) targets,
               count(*) FILTER (WHERE prior_reviews>=1) has_1,
               count(*) FILTER (WHERE prior_reviews>=2) has_2,
               count(*) FILTER (WHERE prior_reviews>=5) has_5,
               count(*) FILTER (WHERE prior_review_images>=1) has_ugc_image,
               avg(prior_reviews), median(prior_reviews),
               quantile_cont(prior_reviews,0.9), max(prior_reviews)
        FROM eval_availability GROUP BY ALL ORDER BY x_label,warm_item DESC
        """
    ).fetchall():
        availability.append(
            {
                "x_label": row[0], "warm_item": row[1], "targets": row[2],
                "has_at_least_1_prior_review": row[3],
                "has_at_least_2_prior_reviews": row[4],
                "has_at_least_5_prior_reviews": row[5],
                "has_prior_ugc_image": row[6], "mean_prior_reviews": row[7],
                "median_prior_reviews": row[8], "p90_prior_reviews": row[9],
                "max_prior_reviews": row[10],
            }
        )

    catalog = con.execute(
        """
        WITH imgs AS (SELECT item_idx,count(*) n FROM ci GROUP BY item_idx),
        txt AS (
          SELECT item_idx,
                 title IS NOT NULL OR array_length(features)>0 OR array_length(description)>0 AS ok
          FROM ct
        ), eval_items AS (SELECT DISTINCT item_idx FROM x WHERE x_label IN (1,2))
        SELECT count(*) eval_items,
               count(*) FILTER (WHERE coalesce(imgs.n,0)>=1) image_1,
               count(*) FILTER (WHERE coalesce(imgs.n,0)>=2) image_2,
               count(*) FILTER (WHERE coalesce(imgs.n,0)>=5) image_5,
               count(*) FILTER (WHERE txt.ok) text_items,
               count(*) FILTER (WHERE coalesce(imgs.n,0)>=2 AND txt.ok) multimodal_items
        FROM eval_items LEFT JOIN imgs USING(item_idx) LEFT JOIN txt USING(item_idx)
        """
    ).fetchone()

    result = {
        "dataset": "Amazon Reviews 2023 Musical_Instruments official 5-core",
        "integrity": {
            "interactions": core[0], "users": core[1], "items": core[2],
            "train": core[3], "validation": core[4], "test": core[5],
            "validation_users": core[6], "test_users": core[7],
            "cold_eval_interactions": core[8], "minimum_train_interactions_per_user": min_train,
            "chronology_violations": chronology_violations[0],
            "users_with_boundary_timestamp_ties": chronology_violations[1],
            "repeated_user_item_pairs": repeat_pairs[0],
            "extra_interactions_in_repeated_pairs": repeat_pairs[1],
        },
        "review_evidence": {
            "rows": evidence_integrity[0], "distinct_evidence_ids": evidence_integrity[1],
            "distinct_exact_keys": evidence_integrity[2], "rows_with_text": evidence_integrity[3],
            "rows_with_images": evidence_integrity[4],
            "eligibility_rule": "e.item_idx=t.item_idx AND e.timestamp<t.timestamp AND e.author_user_idx<>t.user_idx",
        },
        "eval_prior_evidence_availability": availability,
        "eval_catalog_coverage": {
            "items": catalog[0], "with_at_least_1_image": catalog[1],
            "with_at_least_2_images": catalog[2], "with_at_least_5_images": catalog[3],
            "with_text": catalog[4], "with_text_and_at_least_2_images": catalog[5],
        },
    }
    hard_ok = (
        core[0] == 511836 and core[3] + core[4] + core[5] == core[0]
        and core[4] == core[5] == core[1] and min_train >= 3
        and chronology_violations[0] == 0
        and evidence_integrity[0] == evidence_integrity[1] == evidence_integrity[2] == core[0]
    )
    result["decision"] = "GO" if hard_ok else "NO-GO"
    path = OUT / "amazon_musical_instruments_processed_audit.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
