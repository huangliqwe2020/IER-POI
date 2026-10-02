#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$ROOT/vendor${PYTHONPATH:+:$PYTHONPATH}"
CHECKPOINT="${1:-$ROOT/runs/mmpoi/nyc_fixed_seed618/checkpoints/best_epoch.state.pt}"
DEVICE="${2:-cuda:0}"

cd "$ROOT/third_party/MMPOI"
"$PYTHON" train.py \
  --device "$DEVICE" \
  --epochs 1 \
  --eval-checkpoint "$CHECKPOINT" \
  --project "$ROOT/runs/mmpoi_eval" \
  --name nyc_checkpoint_eval \
  --exist-ok
