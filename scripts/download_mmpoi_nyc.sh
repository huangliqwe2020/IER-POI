#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DOWNLOAD_DIR="$ROOT/data/raw/mmpoi/downloads"
PROCESSED_DIR="$ROOT/data/raw/mmpoi/nyc_processed"
RAW_DIR="$ROOT/data/raw/mmpoi/nyc_raw"

PROCESSED_ID="17FbNvkO74xub6AeT2fpm938qDqUDB-04"
RAW_ID="1v0BvKs46ixUf1CgjRlk9MY5JsFL7ILxC"
PROCESSED_SHA256="3998e67a1572dc7d368df17de6b94c399cbd0b3eb46f12c7e2fa55642150baca"
RAW_SHA256="24591ecd81b9fcb53d76b1e918ebb4124a7d96b06fb5241b59d4bfda85e623e9"

mkdir -p "$DOWNLOAD_DIR" "$PROCESSED_DIR" "$RAW_DIR"

download_and_check() {
  local file_id="$1"
  local output="$2"
  local expected="$3"
  if [[ ! -f "$output" ]]; then
    "$PYTHON" -m gdown "$file_id" --output "$output"
  fi
  echo "$expected  $output" | sha256sum --check -
}

download_and_check "$PROCESSED_ID" "$DOWNLOAD_DIR/NYC_processed.zip" "$PROCESSED_SHA256"
download_and_check "$RAW_ID" "$DOWNLOAD_DIR/NYC_raw.zip" "$RAW_SHA256"

unzip -q -o "$DOWNLOAD_DIR/NYC_processed.zip" -d "$PROCESSED_DIR"
unzip -q -o "$DOWNLOAD_DIR/NYC_raw.zip" -d "$RAW_DIR"

link="$ROOT/third_party/MMPOI/NYC"
target="$PROCESSED_DIR/NYC"
if [[ -e "$link" && ! -L "$link" ]]; then
  echo "Refusing to replace non-symlink path: $link" >&2
  exit 1
fi
ln -sfn "$target" "$link"

echo "MMPOI NYC downloaded and verified."
echo "Processed: $target"
echo "Raw: $RAW_DIR/NYC"
