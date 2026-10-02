#!/usr/bin/env python3
"""Build deterministic, leakage-safe perturbation targets for IER evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/amazon_musical_instruments"
OUT = DATA / "robustness"


def p(name: str) -> str:
    return str(DATA / name).replace("'", "''")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute(f"CREATE VIEW x AS SELECT * FROM read_parquet('{p('interactions.parquet')}')")
    con.execute(f"CREATE VIEW e AS SELECT * FROM read_parquet('{p('review_evidence.parquet')}')")
    con.execute(
        """
        CREATE TEMP TABLE train_user AS
        SELECT user_idx,count(*) train_history_len FROM x WHERE x_label=0 GROUP BY user_idx
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE train_item AS
        SELECT item_idx,count(*) train_item_pop FROM x WHERE x_label=0 GROUP BY item_idx
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE base AS
        SELECT hash(t.user_idx,t.item_idx,t.timestamp,t.x_label) target_id,
               t.user_idx,t.item_idx,t.timestamp,t.x_label,t.strict_time_boundary,
               tu.train_history_len,ti.train_item_pop,
               count(e.evidence_id) prior_review_count,
               arg_min(e.evidence_id,hash(e.evidence_id,t.user_idx,t.timestamp,'anchor')) anchor_evidence_id
        FROM x t
        JOIN train_user tu USING(user_idx) JOIN train_item ti USING(item_idx)
        LEFT JOIN e
          ON e.item_idx=t.item_idx
         AND e.timestamp<t.timestamp
         AND e.author_user_idx<>t.user_idx
        WHERE t.x_label IN (1,2) AND t.warm_item
        GROUP BY ALL
        """
    )
    n_evidence = con.execute("SELECT count(*) FROM e").fetchone()[0]
    con.execute(
        """
        CREATE TEMP TABLE ranked_noise AS
        SELECT row_number() OVER (ORDER BY hash(evidence_id,'noise-pool')) noise_rank,
               evidence_id,item_idx
        FROM e
        """
    )
    # Five deterministic candidates make a same-item collision negligible; the
    # postcondition below still fails hard rather than silently leaking one.
    joins = []
    for k in range(5):
        joins.append(
            f"LEFT JOIN ranked_noise n{k} ON n{k}.noise_rank="
            f"1+(hash(b.target_id,'noise-{k}') % {n_evidence})"
        )
    noise_id_case = "CASE " + " ".join(
        f"WHEN n{k}.item_idx<>b.item_idx THEN n{k}.evidence_id" for k in range(5)
    ) + " END"
    noise_item_case = "CASE " + " ".join(
        f"WHEN n{k}.item_idx<>b.item_idx THEN n{k}.item_idx" for k in range(5)
    ) + " END"
    output = str(OUT / "targets.parquet").replace("'", "''")
    con.execute(
        f"""
        COPY (
          SELECT b.*,
                 {noise_id_case} noise_evidence_id,
                 {noise_item_case} noise_source_item_idx,
                 CASE WHEN train_history_len=3 THEN 'u3'
                      WHEN train_history_len<=5 THEN 'u4_5' ELSE 'u6_plus' END user_sparse_bucket,
                 CASE WHEN train_item_pop<=5 THEN 'i1_5'
                      WHEN train_item_pop<=20 THEN 'i6_20' ELSE 'i21_plus' END item_sparse_bucket
          FROM base b {' '.join(joins)}
          ORDER BY x_label,user_idx
        ) TO '{output}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    target_path = str(OUT / "targets.parquet").replace("'", "''")
    counts = con.execute(
        f"""
        SELECT count(*),
               count(*) FILTER (WHERE strict_time_boundary),
               count(*) FILTER (WHERE prior_review_count>=1),
               count(*) FILTER (WHERE anchor_evidence_id IS NULL),
               count(*) FILTER (WHERE noise_evidence_id IS NULL),
               count(*) FILTER (WHERE noise_source_item_idx=item_idx),
               count(DISTINCT target_id)
        FROM read_parquet('{target_path}')
        """
    ).fetchone()
    if counts[0] != counts[6] or counts[4] or counts[5]:
        raise RuntimeError(f"robustness target postcondition failed: {counts}")
    protocol = {
        "dataset": "amazon_musical_instruments",
        "seed": 20260930,
        "targets": counts[0],
        "strict_time_targets": counts[1],
        "targets_with_prior_review": counts[2],
        "targets_without_anchor_because_no_prior_review": counts[3],
        "eligibility_rule": "same item; evidence timestamp < target timestamp; evidence author != target user",
        "clean": "unaltered eligible catalog and prior-UGC evidence",
        "duplicate": {
            "source": "anchor_evidence_id",
            "extra_copies": [1, 2, 4, 8],
            "note": "Anchor is selected by deterministic salted hash, not relevance.",
        },
        "noise_random": {
            "source": "noise_evidence_id from a different item",
            "injected_units": [1, 2, 4, 8],
            "note": "Semantic-nearest noise is deferred until evidence embeddings exist.",
        },
        "missing_random": {
            "source": "anchor_evidence_id",
            "remove_units": [1],
        },
        "missing_key": {
            "source": "runtime top evidence under each interest prototype",
            "note": "Must be generated from model relevance, never substituted by the hash anchor.",
        },
        "sparse": {
            "user_buckets": ["u3", "u4_5", "u6_plus"],
            "item_buckets": ["i1_5", "i6_20", "i21_plus"],
        },
        "metrics": ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20", "MRR", "score_drift", "rank_drift", "topk_overlap"],
    }
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
