#!/usr/bin/env python3
"""Prepare the formal Musical_Instruments 5-core dataset for MMRec and IER."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/amazon_reviews_2023/musical_instruments"
OUT = ROOT / "data/processed/amazon_musical_instruments"


def esc(path: Path) -> str:
    return str(path).replace("'", "''")


def main() -> None:
    review_file = RAW / "reviews_5core_exact.jsonl.gz"
    if not review_file.is_file():
        raise SystemExit(f"missing {review_file}; run stream_filter_musical_reviews.py first")
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute(
        f"CREATE VIEW inter AS SELECT * FROM read_csv_auto("
        f"'{esc(RAW / 'official_5core_rating_only.csv.gz')}', header=true)"
    )
    con.execute(f"CREATE VIEW meta AS SELECT * FROM read_parquet('{esc(RAW / 'metadata-*.parquet')}')")
    con.execute(
        f"CREATE VIEW reviews AS SELECT * FROM read_json_auto("
        f"'{esc(review_file)}', format='newline_delimited')"
    )
    con.execute(
        """
        CREATE TEMP TABLE users AS
        SELECT user_id, row_number() OVER (ORDER BY user_id) - 1 AS user_idx
        FROM (SELECT DISTINCT user_id FROM inter)
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE items AS
        SELECT parent_asin, row_number() OVER (ORDER BY parent_asin) - 1 AS item_idx
        FROM (SELECT DISTINCT parent_asin FROM inter)
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE split AS
        WITH x AS (
          SELECT inter.*,
                 row_number() OVER (PARTITION BY user_id ORDER BY timestamp, parent_asin) pos,
                 count(*) OVER (PARTITION BY user_id) seq_len
          FROM inter
        )
        SELECT u.user_idx, i.item_idx, x.* EXCLUDE(user_id, parent_asin),
               x.user_id, x.parent_asin,
               CASE WHEN pos=seq_len THEN 2 WHEN pos=seq_len-1 THEN 1 ELSE 0 END AS x_label
        FROM x JOIN users u USING(user_id) JOIN items i USING(parent_asin)
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE split_flagged AS
        WITH ti AS (SELECT DISTINCT item_idx FROM split WHERE x_label=0),
        uf AS (
          SELECT user_idx,
                 max(timestamp) FILTER (WHERE x_label=0)
                   < max(timestamp) FILTER (WHERE x_label=1)
                 AND max(timestamp) FILTER (WHERE x_label=1)
                   < max(timestamp) FILTER (WHERE x_label=2) AS strict_time_boundary
          FROM split GROUP BY user_idx
        )
        SELECT s.*, ti.item_idx IS NOT NULL AS warm_item, uf.strict_time_boundary
        FROM split s LEFT JOIN ti USING(item_idx) JOIN uf USING(user_idx)
        """
    )

    con.execute(f"COPY users TO '{esc(OUT / 'users.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(f"COPY items TO '{esc(OUT / 'items.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(
        f"COPY split_flagged TO '{esc(OUT / 'interactions.parquet')}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.execute(
        f"""
        COPY (
          SELECT user_idx AS userID, item_idx AS itemID, x_label
          FROM split_flagged WHERE x_label=0 OR warm_item
          ORDER BY user_idx, timestamp, item_idx
        ) TO '{esc(OUT / 'musical_instruments.inter')}' (HEADER, DELIMITER '\t')
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT i.item_idx, m.parent_asin, ordinality-1 evidence_pos,
                 url image_url, list_extract(m.images.variant, ordinality) AS variant,
                 'catalog_image' evidence_type
          FROM items i JOIN meta m USING(parent_asin),
               UNNEST(m.images.large) WITH ORDINALITY AS x(url, ordinality)
          WHERE url IS NOT NULL
        ) TO '{esc(OUT / 'catalog_images.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT i.item_idx, m.parent_asin, m.title, m.features, m.description,
                 m.details, m.categories, m.store, 'catalog_text' evidence_type
          FROM items i JOIN meta m USING(parent_asin)
        ) TO '{esc(OUT / 'catalog_text.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT hash(r.user_id, r.parent_asin, r.asin, r.timestamp, r.title, r.text) evidence_id,
                 i.item_idx, u.user_idx author_user_idx, r.user_id author_user_id,
                 r.parent_asin, r.asin, r.timestamp, r.rating, r.title, r.text,
                 r.images, r.helpful_vote, r.verified_purchase, 'ugc_review' evidence_type
          FROM reviews r JOIN items i USING(parent_asin) JOIN users u USING(user_id)
        ) TO '{esc(OUT / 'review_evidence.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    split_rows = con.execute(
        """
        SELECT x_label, count(*) interactions, count(DISTINCT user_idx) users,
               count(DISTINCT item_idx) items,
               count(*) FILTER (WHERE warm_item) warm_interactions
        FROM split_flagged GROUP BY x_label ORDER BY x_label
        """
    ).fetchall()
    manifest = {
        "source": "Amazon Reviews 2023 Musical_Instruments official 5-core",
        "role": "formal_general_scene",
        "users": con.execute("SELECT count(*) FROM users").fetchone()[0],
        "items": con.execute("SELECT count(*) FROM items").fetchone()[0],
        "interactions": con.execute("SELECT count(*) FROM split").fetchone()[0],
        "split_policy": "Per-user chronological leave-last-two-out; cold eval items flagged and excluded from MMRec clean file.",
        "strict_time_users": con.execute(
            "SELECT count(DISTINCT user_idx) FROM split_flagged WHERE strict_time_boundary"
        ).fetchone()[0],
        "split_counts": [
            {
                "x_label": r[0],
                "interactions": r[1],
                "users": r[2],
                "items": r[3],
                "warm_interactions": r[4],
            }
            for r in split_rows
        ],
        "evidence_policy": (
            "For target time t, UGC evidence requires review.timestamp < t and excludes the target user review. "
            "Catalog images/text are static snapshot evidence."
        ),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
