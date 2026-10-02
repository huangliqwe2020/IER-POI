#!/usr/bin/env python3
"""Build a deterministic, POI-aligned bank of individual MMPOI reviews.

The released MMPOI features contain one pooled review vector per POI.  IER
needs the individual pieces of evidence, so this script restores that axis
from the raw review text.  The encoder is deliberately local and frozen:
TF-IDF followed by TruncatedSVD and L2 normalization.  It can later be
replaced by a stronger sentence encoder without changing the saved layout.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


SPACE_RE = re.compile(r"\s+")


def clean_review(text: object) -> str:
    if not isinstance(text, str):
        return ""
    return SPACE_RE.sub(" ", text).strip()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviews",
        type=Path,
        default=root / "data/raw/mmpoi/nyc_raw/NYC/review_summary.json",
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        default=root / "data/raw/mmpoi/nyc_processed/NYC/graph_X.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "artifacts/evidence/mmpoi_nyc_reviews_tfidf_svd128.npz",
    )
    parser.add_argument("--max-evidence", type=int, default=20)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=618)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # The MMPOI release uses JSON Lines: each line is a small mapping rather
    # than one valid top-level JSON object.
    raw: dict[str, list[str]] = {}
    with args.reviews.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object on line {line_number}")
            raw.update(row)

    nodes = pd.read_csv(args.nodes, sep="\t")
    poi_ids = nodes["node_name/poi_id"].astype(str).tolist()

    by_poi: dict[str, list[str]] = {}
    raw_count = 0
    duplicate_count = 0
    for raw_key, values in raw.items():
        poi_id = str(raw_key).split("_", 1)[-1]
        seen: set[str] = set()
        cleaned: list[str] = []
        for value in values:
            text = clean_review(value)
            if not text:
                continue
            raw_count += 1
            key = text.casefold()
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            cleaned.append(text)
        if cleaned:
            by_poi[poi_id] = cleaned

    selected_texts: list[str] = []
    offsets = [0]
    counts: list[int] = []
    for poi_id in poi_ids:
        # Input order is stable; a cap prevents review-rich landmarks from
        # dominating memory and training time.
        texts = by_poi.get(poi_id, [])[: args.max_evidence]
        selected_texts.extend(texts)
        counts.append(len(texts))
        offsets.append(len(selected_texts))

    if not selected_texts:
        raise RuntimeError("No raw reviews matched the processed POI IDs")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=args.max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    tfidf = vectorizer.fit_transform(selected_texts)
    effective_dim = min(args.dim, max(1, tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=effective_dim, random_state=args.seed)
    embeddings = normalize(svd.fit_transform(tfidf), norm="l2").astype(np.float16)
    if effective_dim < args.dim:
        embeddings = np.pad(embeddings, ((0, 0), (0, args.dim - effective_dim)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings=embeddings,
        offsets=np.asarray(offsets, dtype=np.int64),
        counts=np.asarray(counts, dtype=np.int16),
        poi_ids=np.asarray(poi_ids),
        encoder=np.asarray("tfidf_svd_l2_v1"),
        explained_variance=np.asarray(svd.explained_variance_ratio_.sum(), dtype=np.float32),
    )
    text_index_path = args.output.with_suffix(".reviews.jsonl.gz")
    with gzip.open(text_index_path, "wt", encoding="utf-8") as handle:
        for poi_index, poi_id in enumerate(poi_ids):
            start, end = offsets[poi_index], offsets[poi_index + 1]
            handle.write(json.dumps({
                "poi_index": poi_index,
                "poi_id": poi_id,
                "reviews": selected_texts[start:end],
            }, ensure_ascii=False) + "\n")

    stats = {
        "source": str(args.reviews),
        "node_file": str(args.nodes),
        "output": str(args.output),
        "text_index": str(text_index_path),
        "num_pois": len(poi_ids),
        "pois_with_evidence": int(np.count_nonzero(counts)),
        "coverage": float(np.count_nonzero(counts) / len(poi_ids)),
        "raw_nonempty_reviews": raw_count,
        "exact_duplicates_removed": duplicate_count,
        "selected_reviews": len(selected_texts),
        "max_evidence_per_poi": args.max_evidence,
        "embedding_dim": args.dim,
        "tfidf_features": int(tfidf.shape[1]),
        "svd_explained_variance": float(svd.explained_variance_ratio_.sum()),
        "count_percentiles": {
            str(p): float(np.percentile(counts, p)) for p in (0, 25, 50, 75, 90, 100)
        },
    }
    stats_path = args.output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
