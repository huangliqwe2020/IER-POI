#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CHECKPOINT="${1:-$ROOT/runs/mmpoi_ier/smoke_review_evidence/checkpoints/best_epoch.state.pt}"
RUN_TAG="${2:-smoke_review_evidence}"
DEVICE="${3:-cpu}"
export PYTHONPATH="$ROOT/vendor${PYTHONPATH:+:$PYTHONPATH}"

for PERTURBATION in clean duplicate noise missing sparse; do
  cd "$ROOT/third_party/MMPOI"
  "$PYTHON" train.py \
    --device "$DEVICE" \
    --epochs 1 \
    --eval-checkpoint "$CHECKPOINT" \
    --review-evidence-enabled \
    --review-evidence-path "$ROOT/artifacts/evidence/mmpoi_nyc_reviews_tfidf_svd128.npz" \
    --evidence-perturbation "$PERTURBATION" \
    --project "$ROOT/runs/mmpoi_ier_robustness/$RUN_TAG" \
    --name "$PERTURBATION" \
    --exist-ok
done

cd "$ROOT"
"$PYTHON" "$ROOT/scripts/summarize_mmpoi_ier_robustness.py" \
  --root "$ROOT/runs/mmpoi_ier_robustness/$RUN_TAG"
