#!/usr/bin/env python3
"""Stream the 1.56GB Musical_Instruments JSONL and retain exact 5-core rows.

The source is never materialized on disk. Matching uses the exact
(user_id, parent_asin, timestamp) triple from the official 5-core file.
"""

from __future__ import annotations

import gzip
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/amazon_reviews_2023/musical_instruments"
OUT = RAW / "reviews_5core_exact.jsonl.gz"
STATS = ROOT / "artifacts/data_audit/amazon_musical_instruments_ugc_stream.json"
URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw/review_categories/Musical_Instruments.jsonl?download=true"
)


def main() -> None:
    con = duckdb.connect()
    rating = str(RAW / "official_5core_rating_only.csv.gz").replace("'", "''")
    keys = set(
        con.execute(
            f"SELECT user_id, parent_asin, timestamp FROM read_csv_auto('{rating}', header=true)"
        ).fetchall()
    )
    expected = len(keys)
    remaining = set(keys)
    print(f"loaded {expected:,} exact 5-core keys", flush=True)

    tmp = OUT.with_suffix(OUT.suffix + ".part")
    counts: Counter[str] = Counter()
    item_text = set()
    item_review_images = set()
    sampled_image_urls: list[str] = []
    start = time.time()
    request = urllib.request.Request(URL, headers={"User-Agent": "IER-research-audit/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, gzip.open(
        tmp, "wt", encoding="utf-8", compresslevel=6
    ) as dst:
        for line_no, raw_line in enumerate(response, 1):
            counts["source_rows"] += 1
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                counts["invalid_json"] += 1
                continue
            key = (obj.get("user_id"), obj.get("parent_asin"), obj.get("timestamp"))
            if key not in remaining:
                if line_no % 200_000 == 0:
                    elapsed = time.time() - start
                    print(
                        f"scanned={line_no:,} matched={counts['matched_rows']:,}/{expected:,} "
                        f"elapsed={elapsed:.0f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                continue
            selected = {
                "rating": obj.get("rating"),
                "title": obj.get("title"),
                "text": obj.get("text"),
                "images": obj.get("images") or [],
                "asin": obj.get("asin"),
                "parent_asin": obj.get("parent_asin"),
                "user_id": obj.get("user_id"),
                "timestamp": obj.get("timestamp"),
                "helpful_vote": obj.get("helpful_vote"),
                "verified_purchase": obj.get("verified_purchase"),
            }
            dst.write(json.dumps(selected, ensure_ascii=False, separators=(",", ":")) + "\n")
            remaining.remove(key)
            counts["matched_rows"] += 1
            text = (obj.get("text") or "").strip()
            if text:
                counts["rows_with_text"] += 1
                item_text.add(obj.get("parent_asin"))
            images = obj.get("images") or []
            if images:
                counts["rows_with_images"] += 1
                counts["image_units"] += len(images)
                item_review_images.add(obj.get("parent_asin"))
                for image in images:
                    url = image.get("large_image_url")
                    if url and len(sampled_image_urls) < 200:
                        sampled_image_urls.append(url)
            if not remaining:
                # The exact official triples are unique; after all are found,
                # the remaining source rows cannot contribute to this subset.
                break

    tmp.replace(OUT)
    missing = len(remaining)
    result = {
        "source_url": URL,
        "expected_exact_5core_rows": expected,
        **counts,
        "missing_exact_rows": missing,
        "items_with_text_reviews": len(item_text),
        "items_with_review_images": len(item_review_images),
        "sample_review_image_urls": sampled_image_urls,
        "elapsed_seconds": round(time.time() - start, 2),
        "output": str(OUT),
        "output_bytes": OUT.stat().st_size,
        "decision": "GO" if missing == 0 and counts["rows_with_text"] / expected >= 0.95 else "NO-GO",
    }
    STATS.parent.mkdir(parents=True, exist_ok=True)
    STATS.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
