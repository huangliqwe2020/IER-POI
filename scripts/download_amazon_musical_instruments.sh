#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${root_dir}/data/raw/amazon_reviews_2023/musical_instruments"
mkdir -p "${out_dir}"

download() {
  local url="$1"
  local output="$2"
  local expected_sha256="${3:-}"
  if [[ ! -s "${output}" ]]; then
    curl --fail --location --retry 5 --continue-at - "${url}" --output "${output}.part"
    mv "${output}.part" "${output}"
  else
    echo "exists: ${output}"
  fi
  if [[ -n "${expected_sha256}" ]]; then
    echo "${expected_sha256}  ${output}" | sha256sum --check -
  fi
}

hf_base="https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"
download \
  "${hf_base}/raw_meta_Musical_Instruments/full-00000-of-00002.parquet?download=true" \
  "${out_dir}/metadata-00000-of-00002.parquet" \
  "867567b573c4dc2860d476d0ac2de30bf9d4b3f77ae18e925d551d8d2170db6d"
download \
  "${hf_base}/raw_meta_Musical_Instruments/full-00001-of-00002.parquet?download=true" \
  "${out_dir}/metadata-00001-of-00002.parquet" \
  "d9bf76dd54000c544d81939c1ba7a788c6553ae32745066a8dad668b75690544"

download \
  "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Musical_Instruments.csv.gz" \
  "${out_dir}/official_5core_rating_only.csv.gz"

sha256sum "${out_dir}"/* > "${out_dir}/SHA256SUMS"
du -h "${out_dir}"/*
