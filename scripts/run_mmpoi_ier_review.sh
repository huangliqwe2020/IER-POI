#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
EPOCHS="${1:-30}"
RUN_NAME="${2:-nyc_ier_review_seed618}"
DEVICE="${3:-cpu}"

export PYTHONPATH="$ROOT/vendor${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" "$ROOT/scripts/prepare_mmpoi_review_evidence.py"
"$PYTHON" "$ROOT/scripts/test_mmpoi_review_evidence.py"

cd "$ROOT/third_party/MMPOI"
"$PYTHON" train.py \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --save-weights \
  --init-checkpoint "$ROOT/runs/mmpoi/nyc_fixed_seed618/checkpoints/best_epoch.state.pt" \
  --review-evidence-enabled \
  --evidence-only-training \
  --review-evidence-path "$ROOT/artifacts/evidence/mmpoi_nyc_reviews_tfidf_svd128.npz" \
  --project "$ROOT/runs/mmpoi_ier" \
  --name "$RUN_NAME" \
  --exist-ok

"$PYTHON" "$ROOT/scripts/test_mmpoi_nyc.py" \
  --run-dir "$ROOT/runs/mmpoi_ier/$RUN_NAME" \
  --output "$ROOT/runs/mmpoi_ier/$RUN_NAME/test_summary.json"
