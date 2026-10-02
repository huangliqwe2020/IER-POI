#!/usr/bin/env python3
"""Run transparent next-POI sanity baselines on the official MMPOI NYC split."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/raw/mmpoi/nyc_processed/NYC"
DEFAULT_OUTPUT = ROOT / "runs/mmpoi/classical_nyc_seed618/results.json"


def ranked_metrics(rank: int, ks: tuple[int, ...] = (1, 5, 10, 20)) -> dict[str, float]:
    result = {f"HR@{k}": float(rank <= k) for k in ks}
    result.update({f"NDCG@{k}": (1.0 / np.log2(rank + 1)) if rank <= k else 0.0 for k in ks})
    result["MRR"] = 1.0 / rank
    result["mAP@20"] = (1.0 / rank) if rank <= 20 else 0.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "NYC_train.csv", dtype={"user_id": str, "POI_id": str})
    test = pd.read_csv(args.data_dir / "NYC_test.csv", dtype={"user_id": str, "POI_id": str})
    train_pois = set(train.POI_id)

    global_counts = Counter(train.POI_id)
    user_counts: dict[str, Counter[str]] = defaultdict(Counter)
    transitions: dict[str, Counter[str]] = defaultdict(Counter)
    for user, sub in train.groupby("user_id", sort=False):
        user_counts[user].update(sub.POI_id)
    for _, sub in train.groupby("trajectory_id", sort=False):
        seq = sub.POI_id.tolist()
        for source, target in zip(seq, seq[1:]):
            transitions[source][target] += 1

    global_order = [poi for poi, _ in sorted(global_counts.items(), key=lambda item: (-item[1], item[0]))]
    global_rank = {poi: idx + 1 for idx, poi in enumerate(global_order)}

    sums = {name: Counter() for name in ("GlobalPop", "UserPop", "Markov1")}
    evaluated = 0
    skipped_short = 0
    removed_cold_events = 0
    for trajectory_id, sub in test.groupby("trajectory_id", sort=False):
        original = sub.POI_id.tolist()
        seq = [poi for poi in original if poi in train_pois]
        removed_cold_events += len(original) - len(seq)
        # Matches short_traj_thres=2: at least two input/label pairs.
        if len(seq) < 3:
            skipped_short += 1
            continue
        user = str(trajectory_id).split("_", 1)[0]
        source, target = seq[-2], seq[-1]

        user_order = sorted(
            train_pois,
            key=lambda poi: (-user_counts[user][poi], global_rank[poi]),
        )
        markov_order = sorted(
            train_pois,
            key=lambda poi: (-transitions[source][poi], global_rank[poi]),
        )
        ranks = {
            "GlobalPop": global_rank[target],
            "UserPop": user_order.index(target) + 1,
            "Markov1": markov_order.index(target) + 1,
        }
        for name, rank in ranks.items():
            sums[name].update(ranked_metrics(rank))
        evaluated += 1

    metrics = {
        name: {key: float(value / evaluated) for key, value in values.items()}
        for name, values in sums.items()
    }
    result = {
        "dataset": "MMPOI NYC official split",
        "protocol": "next POI at the last timestep; trajectory-macro average",
        "evaluated_trajectories": evaluated,
        "skipped_short_trajectories": skipped_short,
        "removed_cold_events": removed_cold_events,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
