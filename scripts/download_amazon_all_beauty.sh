#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${root_dir}/data/raw/amazon_reviews_2023/all_beauty"
mkdir -p "${out_dir}"

base="https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve"

download() {
  local revision="$1"
  local relative="$2"
  local output="$3"
  local expected_sha256="${4:-}"
  if [[ -s "${output}" ]]; then
    echo "exists: ${output}"
  else
    curl --fail --location --retry 5 --continue-at - \
      "${base}/${revision}/${relative}?download=true" --output "${output}.part"
    mv "${output}.part" "${output}"
  fi
  if [[ -n "${expected_sha256}" ]]; then
    echo "${expected_sha256}  ${output}" | sha256sum --check -
  fi
}

download \
  "749931181c9cdd70bfa6408d8fe8b4ddd6107248" \
  "raw_review_All_Beauty/full-00000-of-00001.parquet" \
  "${out_dir}/reviews.parquet" \
  "b40ae565f1073e0b4960e2f424fe0c82f3d270b7c4d00bc9272ddd6a8b7411d7"

download \
  "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e" \
  "raw_meta_All_Beauty/full-00000-of-00001.parquet" \
  "${out_dir}/metadata.parquet" \
  "14815cf1312a0d847364866e6876c8c73738993469067242870774c372c04387"

for split in train valid test; do
  download \
    "main" \
    "benchmark/5core/last_out/All_Beauty.${split}.csv" \
    "${out_dir}/official_5core_last_out_${split}.csv"
done

sha256sum "${out_dir}"/* > "${out_dir}/SHA256SUMS"
du -h "${out_dir}"/*
