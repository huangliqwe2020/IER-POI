#!/usr/bin/env python3
"""Deduplicate the first stream result and recover missing exact 5-core rows."""

from __future__ import annotations

import gzip
import json
import time
import urllib.request
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/amazon_reviews_2023/musical_instruments"
SOURCE = RAW / "reviews_5core_exact.jsonl.gz"
TMP = RAW / "reviews_5core_exact.repaired.jsonl.gz"
STATS = ROOT / "artifacts/data_audit/amazon_musical_instruments_ugc_stream.json"
URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw/review_categories/Musical_Instruments.jsonl?download=true"
)


def selected(obj: dict) -> dict:
    return {
        "rating": obj.get("rating"), "title": obj.get("title"), "text": obj.get("text"),
        "images": obj.get("images") or [], "asin": obj.get("asin"),
        "parent_asin": obj.get("parent_asin"), "user_id": obj.get("user_id"),
        "timestamp": obj.get("timestamp"), "helpful_vote": obj.get("helpful_vote"),
        "verified_purchase": obj.get("verified_purchase"),
    }


def key(obj: dict) -> tuple[object, object, object]:
    return obj.get("user_id"), obj.get("parent_asin"), obj.get("timestamp")


def main() -> None:
    con = duckdb.connect()
    rating = str(RAW / "official_5core_rating_only.csv.gz").replace("'", "''")
    official = set(
        con.execute(
            f"SELECT user_id,parent_asin,timestamp FROM read_csv_auto('{rating}',header=true)"
        ).fetchall()
    )
    remaining = set(official)
    seen = set()
    counts = {"duplicate_rows_removed": 0, "recovered_rows": 0}
    with gzip.open(SOURCE, "rt", encoding="utf-8") as src, gzip.open(
        TMP, "wt", encoding="utf-8", compresslevel=6
    ) as dst:
        for line in src:
            obj = json.loads(line)
            k = key(obj)
            if k not in official or k in seen:
                counts["duplicate_rows_removed"] += 1
                continue
            seen.add(k)
            remaining.discard(k)
            dst.write(json.dumps(selected(obj), ensure_ascii=False, separators=(",", ":")) + "\n")
        print(f"deduplicated={len(seen):,}; missing={len(remaining):,}", flush=True)

        # Missing keys occur after the premature stopping point near the end of
        # the source. Start well before that point; fall back to a full stream
        # only if the ranged request does not recover every key.
        for start in (1_150_000_000, 0):
            if not remaining:
                break
            headers = {"User-Agent": "IER-research-audit/1.0"}
            if start:
                headers["Range"] = f"bytes={start}-"
            request = urllib.request.Request(URL, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                print(f"recovery request start={start} status={response.status}", flush=True)
                for raw_line in response:
                    try:
                        obj = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    k = key(obj)
                    if k not in remaining:
                        continue
                    remaining.remove(k)
                    seen.add(k)
                    counts["recovered_rows"] += 1
                    dst.write(
                        json.dumps(selected(obj), ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    if counts["recovered_rows"] % 500 == 0:
                        print(
                            f"recovered={counts['recovered_rows']:,}; remaining={len(remaining):,}",
                            flush=True,
                        )
                    if not remaining:
                        break

    if remaining:
        raise SystemExit(f"repair incomplete: {len(remaining)} keys still missing; kept {TMP}")
    TMP.replace(SOURCE)
    review_path = str(SOURCE).replace("'", "''")
    verified = con.execute(
        f"""
        SELECT count(*) AS "rows",
               count(DISTINCT (user_id,parent_asin,timestamp)) distinct_exact_keys,
               count(*) FILTER (WHERE text IS NOT NULL AND length(trim(text))>0) rows_with_text,
               count(*) FILTER (WHERE array_length(images)>0) rows_with_images,
               sum(array_length(images)) image_units,
               count(DISTINCT parent_asin) FILTER (WHERE text IS NOT NULL AND length(trim(text))>0) items_with_text_reviews,
               count(DISTINCT parent_asin) FILTER (WHERE array_length(images)>0) items_with_review_images
        FROM read_json_auto('{review_path}', format='newline_delimited')
        """
    ).fetchone()
    result = {
        "source_url": URL,
        "expected_exact_5core_rows": len(official),
        "matched_rows": verified[0],
        "distinct_exact_keys": verified[1],
        "rows_with_text": verified[2],
        "rows_with_images": verified[3],
        "image_units": verified[4],
        "missing_exact_rows": len(official) - verified[1],
        "items_with_text_reviews": verified[5],
        "items_with_review_images": verified[6],
        **counts,
        "output": str(SOURCE),
        "output_bytes": SOURCE.stat().st_size,
        "decision": "GO" if verified[0] == verified[1] == len(official) else "NO-GO",
    }
    STATS.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
