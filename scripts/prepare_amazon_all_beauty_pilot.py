#!/usr/bin/env python3
"""Prepare the 2-core All_Beauty pilot without temporal evidence leakage.

The pilot is intentionally not the final paper dataset. It is large enough to
debug recommendation and perturbation code while the denser formal category is
prepared.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/amazon_reviews_2023/all_beauty"
OUT = ROOT / "data/processed/amazon_all_beauty_pilot"


def q(path: Path) -> str:
    return str(path).replace("'", "''")


def stats(con: duckdb.DuckDBPyConnection, table: str) -> dict[str, int]:
    cur = con.execute(
        f"SELECT count(*) interactions, count(DISTINCT user_id) users, "
        f"count(DISTINCT parent_asin) items FROM {table}"
    )
    names = [x[0] for x in cur.description]
    return dict(zip(names, cur.fetchone()))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute(f"CREATE VIEW reviews AS SELECT * FROM read_parquet('{q(RAW / 'reviews.parquet')}')")
    con.execute(f"CREATE VIEW meta AS SELECT * FROM read_parquet('{q(RAW / 'metadata.parquet')}')")

    con.execute(
        """
        CREATE TEMP TABLE core AS
        SELECT user_id, parent_asin, min(timestamp) AS timestamp,
               arg_min(rating, timestamp) AS rating
        FROM reviews
        WHERE user_id IS NOT NULL AND parent_asin IS NOT NULL
        GROUP BY user_id, parent_asin
        """
    )
    previous = con.execute("SELECT count(*) FROM core").fetchone()[0]
    iterations = 0
    while True:
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE next_core AS
            WITH users AS (SELECT user_id FROM core GROUP BY user_id HAVING count(*) >= 2),
                 items AS (SELECT parent_asin FROM core GROUP BY parent_asin HAVING count(*) >= 2)
            SELECT core.* FROM core
            SEMI JOIN users USING(user_id)
            SEMI JOIN items USING(parent_asin)
            """
        )
        current = con.execute("SELECT count(*) FROM next_core").fetchone()[0]
        iterations += 1
        if current == previous:
            break
        previous = current
        con.execute("CREATE OR REPLACE TEMP TABLE core AS SELECT * FROM next_core")
    con.execute("CREATE OR REPLACE TEMP TABLE core AS SELECT * FROM next_core")

    con.execute(
        """
        CREATE TEMP TABLE users AS
        SELECT user_id, row_number() OVER (ORDER BY user_id) - 1 AS user_idx
        FROM (SELECT DISTINCT user_id FROM core)
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE items AS
        SELECT parent_asin, row_number() OVER (ORDER BY parent_asin) - 1 AS item_idx
        FROM (SELECT DISTINCT parent_asin FROM core)
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE split AS
        WITH ordered AS (
          SELECT c.*,
                 row_number() OVER (PARTITION BY user_id ORDER BY timestamp, parent_asin) AS pos,
                 count(*) OVER (PARTITION BY user_id) AS seq_len
          FROM core c
        )
        SELECT u.user_idx, i.item_idx, o.user_id, o.parent_asin,
               o.rating, o.timestamp, o.pos, o.seq_len,
               CASE
                 WHEN o.pos = o.seq_len THEN 2
                 WHEN o.seq_len >= 3 AND o.pos = o.seq_len - 1 THEN 1
                 ELSE 0
               END AS x_label
        FROM ordered o JOIN users u USING(user_id) JOIN items i USING(parent_asin)
        """
    )

    # Warm flags are retained rather than silently deleting cold evaluation
    # items; CleanRec can use warm rows and SparseBench can use cold rows.
    con.execute(
        """
        CREATE TEMP TABLE split_flagged AS
        WITH train_users AS (SELECT DISTINCT user_idx FROM split WHERE x_label = 0),
             train_items AS (SELECT DISTINCT item_idx FROM split WHERE x_label = 0)
        SELECT s.*, tu.user_idx IS NOT NULL AS warm_user,
               ti.item_idx IS NOT NULL AS warm_item
        FROM split s
        LEFT JOIN train_users tu USING(user_idx)
        LEFT JOIN train_items ti USING(item_idx)
        """
    )

    con.execute(f"COPY users TO '{q(OUT / 'users.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(f"COPY items TO '{q(OUT / 'items.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(
        f"COPY split_flagged TO '{q(OUT / 'interactions.parquet')}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.execute(
        f"""
        COPY (
          SELECT user_idx AS userID, item_idx AS itemID, x_label
          FROM split_flagged
          WHERE x_label = 0 OR (warm_user AND warm_item)
          ORDER BY user_idx, timestamp, item_idx
        ) TO '{q(OUT / 'all_beauty_pilot.inter')}' (HEADER, DELIMITER '\t')
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT i.item_idx, m.parent_asin, ordinality - 1 AS evidence_pos,
                 url AS image_url, 'catalog_image' AS evidence_type
          FROM items i JOIN meta m USING(parent_asin),
               UNNEST(m.images.large) WITH ORDINALITY AS x(url, ordinality)
          WHERE url IS NOT NULL
        ) TO '{q(OUT / 'catalog_images.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT i.item_idx, m.parent_asin, m.title, m.features, m.description,
                 m.details, m.categories, m.store,
                 'catalog_text' AS evidence_type
          FROM items i JOIN meta m USING(parent_asin)
        ) TO '{q(OUT / 'catalog_text.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT hash(r.user_id, r.parent_asin, r.asin, r.timestamp, r.title, r.text) AS evidence_id,
                 i.item_idx, u.user_idx AS author_user_idx,
                 r.user_id AS author_user_id, r.parent_asin, r.asin,
                 r.timestamp, r.title, r.text,
                 list_transform(r.images, x -> x.large_image_url) AS image_urls,
                 r.helpful_vote, r.verified_purchase,
                 'ugc_review' AS evidence_type
          FROM reviews r JOIN items i USING(parent_asin)
          LEFT JOIN users u USING(user_id)
        ) TO '{q(OUT / 'review_evidence.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    split_counts = con.execute(
        """
        SELECT x_label, count(*) AS interactions,
               count(DISTINCT user_idx) AS users,
               count(DISTINCT item_idx) AS items,
               count(*) FILTER (WHERE warm_user AND warm_item) AS warm_interactions
        FROM split_flagged GROUP BY x_label ORDER BY x_label
        """
    ).fetchall()
    evidence_counts = {
        "catalog_images": con.execute(
            f"SELECT count(*) FROM read_parquet('{q(OUT / 'catalog_images.parquet')}')"
        ).fetchone()[0],
        "catalog_text_items": con.execute(
            f"SELECT count(*) FROM read_parquet('{q(OUT / 'catalog_text.parquet')}')"
        ).fetchone()[0],
        "ugc_reviews_for_core_items": con.execute(
            f"SELECT count(*) FROM read_parquet('{q(OUT / 'review_evidence.parquet')}')"
        ).fetchone()[0],
    }
    manifest = {
        "source": "Amazon Reviews 2023 All_Beauty",
        "role": "pilot_only",
        "core": {"user_min": 2, "item_min": 2, "iterations": iterations, **stats(con, "core")},
        "split_policy": (
            "Per-user chronological: last=test; second-last=valid for seq_len>=3; remaining=train. "
            "MMRec file keeps all train rows and only warm-user/warm-item evaluation rows."
        ),
        "split_counts": [
            {
                "x_label": row[0],
                "interactions": row[1],
                "users": row[2],
                "items": row[3],
                "warm_interactions": row[4],
            }
            for row in split_counts
        ],
        "evidence_counts": evidence_counts,
        "evidence_safety": (
            "At target time t, review_evidence must be filtered to timestamp < t and must exclude "
            "the target user's target review. Files intentionally retain timestamps and author IDs."
        ),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
