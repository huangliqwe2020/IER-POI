#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
EPOCHS="${1:-200}"
RUN_NAME="${2:-nyc_fixed_seed618}"
DEVICE="${3:-cuda:0}"
export PYTHONPATH="$ROOT/vendor${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" "$ROOT/scripts/prepare_mmpoi_nyc.py"
cd "$ROOT/third_party/MMPOI"
"$PYTHON" train.py \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --save-weights \
  --project "$ROOT/runs/mmpoi" \
  --name "$RUN_NAME" \
  --exist-ok

"$PYTHON" "$ROOT/scripts/test_mmpoi_nyc.py" \
  --run-dir "$ROOT/runs/mmpoi/$RUN_NAME" \
  --output "$ROOT/runs/mmpoi/$RUN_NAME/test_summary.json"

"$ROOT/scripts/evaluate_mmpoi_nyc_checkpoint.sh" \
  "$ROOT/runs/mmpoi/$RUN_NAME/checkpoints/best_epoch.state.pt" "$DEVICE"
