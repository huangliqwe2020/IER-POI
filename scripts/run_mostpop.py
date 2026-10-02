#!/usr/bin/env python3
"""Deterministic full-ranking MostPop baseline on a processed IER dataset."""

from __future__ import annotations

import json
import math
import argparse
from collections import defaultdict
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/baselines"


def evaluate(rows: list[tuple[int, int]], seen: dict[int, set[int]], ranking: list[int]) -> dict[str, float]:
    sums = {"recall@10": 0.0, "recall@20": 0.0, "ndcg@10": 0.0, "ndcg@20": 0.0}
    for user, target in rows:
        recommendations = []
        user_seen = seen[user]
        for item in ranking:
            if item not in user_seen:
                recommendations.append(item)
                if len(recommendations) == 20:
                    break
        if target in recommendations:
            rank = recommendations.index(target) + 1
            if rank <= 10:
                sums["recall@10"] += 1
                sums["ndcg@10"] += 1 / math.log2(rank + 1)
            sums["recall@20"] += 1
            sums["ndcg@20"] += 1 / math.log2(rank + 1)
    return {k: v / len(rows) for k, v in sums.items()} | {"users": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["amazon_all_beauty_pilot", "amazon_musical_instruments"],
        default="amazon_all_beauty_pilot",
    )
    parser.add_argument(
        "--strict-time-only", action="store_true",
        help="Exclude users whose train/validation/test boundary timestamps tie when the flag exists.",
    )
    args = parser.parse_args()
    data = ROOT / "data/processed" / args.dataset / "interactions.parquet"
    con = duckdb.connect()
    path = str(data).replace("'", "''")
    columns = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()}
    train = con.execute(
        f"SELECT user_idx, item_idx FROM read_parquet('{path}') WHERE x_label=0"
    ).fetchall()
    seen: dict[int, set[int]] = defaultdict(set)
    popularity: dict[int, int] = defaultdict(int)
    for user, item in train:
        seen[user].add(item)
        popularity[item] += 1
    ranking = sorted(popularity, key=lambda item: (-popularity[item], item))
    result = {}
    for name, label in (("valid", 1), ("test", 2)):
        conditions = [f"x_label={label}", "warm_item"]
        if "warm_user" in columns:
            conditions.append("warm_user")
        if args.strict_time_only and "strict_time_boundary" in columns:
            conditions.append("strict_time_boundary")
        rows = con.execute(
            f"""
            SELECT user_idx, item_idx FROM read_parquet('{path}')
            WHERE {' AND '.join(conditions)}
            ORDER BY user_idx
            """
        ).fetchall()
        result[name] = evaluate(rows, seen, ranking)
    result["protocol"] = (
        "Full ranking over warm training items; training-seen items excluded per user; "
        f"strict_time_only={args.strict_time_only}."
    )
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "_strict_time" if args.strict_time_only else ""
    (OUT / f"mostpop_{args.dataset}{suffix}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
