#!/usr/bin/env python3
"""Deterministically sample Amazon image URLs and test actual HTTP reachability."""

from __future__ import annotations

import concurrent.futures
import argparse
import json
import subprocess
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/data_audit"
SAMPLE_PER_SOURCE = 100


def fetch(entry: tuple[str, str, str]) -> dict[str, object]:
    source, item, url = entry
    marker = "__IER_CURL__"
    cmd = [
        "curl",
        "--silent",
        "--show-error",
        "--location",
        "--max-time",
        "20",
        "--range",
        "0-2047",
        "--output",
        "/dev/null",
        "--write-out",
        marker + "%{http_code}\t%{content_type}\t%{size_download}\t%{url_effective}",
        url,
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    payload = proc.stdout.rsplit(marker, 1)[-1] if marker in proc.stdout else "0\t\t0\t"
    parts = payload.split("\t", 3)
    while len(parts) < 4:
        parts.append("")
    code, content_type, size, effective_url = parts
    ok = proc.returncode == 0 and code in {"200", "206"} and content_type.startswith("image/")
    return {
        "source": source,
        "parent_asin": item,
        "url": url,
        "ok": ok,
        "curl_exit": proc.returncode,
        "http_code": int(code or 0),
        "content_type": content_type,
        "bytes_received": int(float(size or 0)),
        "effective_url": effective_url,
        "error": proc.stderr.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["all_beauty", "musical_instruments"], default="all_beauty"
    )
    args = parser.parse_args()
    if args.dataset == "all_beauty":
        raw = ROOT / "data/raw/amazon_reviews_2023/all_beauty"
        meta_path = raw / "metadata.parquet"
        review_path = raw / "reviews.parquet"
    else:
        raw = ROOT / "data/raw/amazon_reviews_2023/musical_instruments"
        meta_path = raw / "metadata-*.parquet"
        review_path = raw / "reviews_5core_exact.jsonl.gz"
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    meta = str(meta_path).replace("'", "''")
    product = con.execute(
        f"""
        SELECT DISTINCT 'catalog' AS source, parent_asin, u.url
        FROM read_parquet('{meta}'), UNNEST(images.large) AS u(url)
        WHERE u.url IS NOT NULL AND starts_with(u.url, 'http')
        ORDER BY hash(u.url)
        LIMIT {SAMPLE_PER_SOURCE}
        """
    ).fetchall()
    ugc = []
    if review_path is not None:
        reviews = str(review_path).replace("'", "''")
        review_reader = (
            f"read_parquet('{reviews}')"
            if review_path.suffix == ".parquet"
            else f"read_json_auto('{reviews}', format='newline_delimited')"
        )
        ugc = con.execute(
            f"""
            SELECT DISTINCT 'ugc_review' AS source, parent_asin, u.img.large_image_url AS url
            FROM {review_reader}, UNNEST(images) AS u(img)
            WHERE u.img.large_image_url IS NOT NULL AND starts_with(u.img.large_image_url, 'http')
            ORDER BY hash(u.img.large_image_url)
            LIMIT {SAMPLE_PER_SOURCE}
            """
        ).fetchall()
    entries = product + ugc
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(fetch, entries))

    summary: dict[str, dict[str, object]] = {}
    for source in sorted({r["source"] for r in rows}):
        selected = [r for r in rows if r["source"] == source]
        ok = sum(bool(r["ok"]) for r in selected)
        summary[source] = {
            "sample": len(selected),
            "reachable_images": ok,
            "reachability": ok / len(selected) if selected else 0.0,
            "http_codes": {
                str(code): sum(r["http_code"] == code for r in selected)
                for code in sorted({int(r["http_code"]) for r in selected})
            },
        }
    result = {
        "sample_method": "distinct URLs ordered by deterministic DuckDB hash(url)",
        "range_request": "bytes 0-2047; accepted only HTTP 200/206 with image/* content type",
        "summary": summary,
        "rows": rows,
    }
    (OUT / f"amazon_{args.dataset}_image_url_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md = "# Amazon All_Beauty 图片 URL 可达性抽样\n\n"
    md += "| 来源 | 样本 | 有效图片 | 可达率 |\n|---|---:|---:|---:|\n"
    for source, stats in summary.items():
        md += (
            f"| {source} | {stats['sample']} | {stats['reachable_images']} | "
            f"{stats['reachability']:.1%} |\n"
        )
    md += "\n成功条件为 HTTP 200/206 且响应 `Content-Type` 为 `image/*`。抽样由 URL 哈希固定，可重复运行。\n"
    (OUT / f"amazon_{args.dataset}_image_url_audit.md").write_text(md, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
