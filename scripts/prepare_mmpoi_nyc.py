#!/usr/bin/env python3
"""Audit MMPOI NYC evidence and build the official trajectory graphs."""

from __future__ import annotations

import json
import sys
import tarfile
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MMPOI = ROOT / "third_party/MMPOI"
PROCESSED = ROOT / "data/raw/mmpoi/nyc_processed/NYC"
RAW = ROOT / "data/raw/mmpoi/nyc_raw/NYC"
OUT = ROOT / "artifacts/data_audit/mmpoi_nyc_audit.json"


def jsonl_dict(path: Path) -> dict:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            result.update(json.loads(line))
    return result


def main() -> None:
    data_link = MMPOI / "NYC"
    if not data_link.exists():
        data_link.symlink_to(PROCESSED, target_is_directory=True)
    elif data_link.resolve() != PROCESSED.resolve():
        raise RuntimeError(f"{data_link} does not point to {PROCESSED}")

    sys.path.insert(0, str(MMPOI))
    from build_graph import (
        build_global_POI_checkin_graph,
        build_global_POI_geo_graph,
        geohash_add2_df,
        save_geo_graph_to_csv,
        save_graph_to_csv,
    )

    train = pd.read_csv(PROCESSED / "NYC_train.csv")
    test = pd.read_csv(PROCESSED / "NYC_test.csv")
    poi_ids = set(train.POI_id.astype(str))

    image_counts: Counter[str] = Counter()
    with tarfile.open(RAW / "image.tgz", "r:gz") as archive:
        for member in archive.getmembers():
            name = Path(member.name).name
            if not name.endswith(".png"):
                continue
            stem = name[:-4]
            parts = stem.rsplit("_", 2)
            if len(parts) == 3:
                image_counts[parts[1]] += 1

    review_counts: Counter[str] = Counter()
    with (RAW / "review_summary.json").open(encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            for composite, reviews in obj.items():
                poi = composite.split("_", 1)[-1]
                review_counts[poi] += len(reviews)

    image_emb = jsonl_dict(PROCESSED / "modal_image_embedding.json")
    review_emb = jsonl_dict(PROCESSED / "modal_review_embedding.json")
    meta_emb = jsonl_dict(PROCESSED / "modal_meta_embedding.json")

    graph_files = [
        PROCESSED / "graph_A.csv",
        PROCESSED / "graph_X.csv",
        PROCESSED / "geo_graph_A.csv",
        PROCESSED / "geo_graph_X.csv",
    ]
    if not all(path.is_file() for path in graph_files):
        graph = build_global_POI_checkin_graph(train)
        save_graph_to_csv(graph, str(PROCESSED))
        geo_train = geohash_add2_df(train.copy())
        geo_graph = build_global_POI_geo_graph(geo_train)
        save_geo_graph_to_csv(geo_graph, str(PROCESSED))

    def coverage(counter: Counter[str]) -> dict[str, int | float]:
        values = [counter.get(poi, 0) for poi in poi_ids]
        return {
            "units": int(sum(values)),
            "pois_with_1plus": int(sum(v >= 1 for v in values)),
            "pois_with_2plus": int(sum(v >= 2 for v in values)),
            "pois_with_5plus": int(sum(v >= 5 for v in values)),
            "min": int(min(values)),
            "median": float(pd.Series(values).median()),
            "mean": float(pd.Series(values).mean()),
            "max": int(max(values)),
        }

    result = {
        "dataset": "MMPOI NYC official processed + raw",
        "train_rows": len(train),
        "test_rows": len(test),
        "train_users": int(train.user_id.nunique()),
        "test_users": int(test.user_id.nunique()),
        "train_pois": int(train.POI_id.nunique()),
        "test_pois": int(test.POI_id.nunique()),
        "train_trajectories": int(train.trajectory_id.nunique()),
        "test_trajectories": int(test.trajectory_id.nunique()),
        "cold_test_pois": int(len(set(test.POI_id.astype(str)) - poi_ids)),
        "images": coverage(image_counts),
        "reviews": coverage(review_counts),
        "aggregated_embedding_coverage": {
            "image": len(poi_ids & set(image_emb)),
            "review": len(poi_ids & set(review_emb)),
            "meta": len(poi_ids & set(meta_emb)),
        },
        "graph_files": {path.name: path.stat().st_size for path in graph_files},
        "decision": "GO" if all(
            len(poi_ids & set(source)) == len(poi_ids)
            for source in (image_emb, review_emb, meta_emb)
        ) and sum(image_counts.get(poi, 0) >= 2 for poi in poi_ids) / len(poi_ids) >= 0.5
        else "NO-GO",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
