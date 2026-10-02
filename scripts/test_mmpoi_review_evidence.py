#!/usr/bin/env python3
"""Fast correctness tests for the MMPOI individual-review evidence module."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence",
        type=Path,
        default=root / "artifacts/evidence/mmpoi_nyc_reviews_tfidf_svd128.npz",
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        default=root / "data/raw/mmpoi/nyc_processed/NYC/graph_X.csv",
    )
    args = parser.parse_args()

    data = np.load(args.evidence, allow_pickle=False)
    graph_poi_ids = np.loadtxt(args.nodes, delimiter="\t", dtype=str, skiprows=1, usecols=0)
    assert data["poi_ids"].astype(str).tolist() == graph_poi_ids.astype(str).tolist()
    assert len(data["offsets"]) == len(graph_poi_ids) + 1
    assert data["offsets"][-1] == len(data["embeddings"])
    assert np.isfinite(data["embeddings"]).all()

    # model.py parses command-line arguments at import time in the upstream
    # release, so isolate it from this script's arguments.
    sys.path.insert(0, str(root / "third_party/MMPOI"))
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    from model import RobustReviewEvidenceAggregator
    sys.argv = saved_argv

    torch.manual_seed(618)
    model = RobustReviewEvidenceAggregator(128, 128, 128, 128, topk=5)
    user = torch.randn(128, requires_grad=True)
    poi = torch.randn(128, requires_grad=True)
    evidence = torch.from_numpy(data["embeddings"][:20].astype(np.float32))
    valid = torch.ones(20, dtype=torch.bool)
    baseline = torch.randn(128, requires_grad=True)

    model.eval()
    output, red_loss, sens_loss = model(
        user, poi, evidence, valid, baseline, compute_regularizers=True
    )
    # Zero residual projection guarantees exact checkpoint warm-start behavior.
    assert torch.equal(output, baseline)
    assert torch.isfinite(red_loss) and torch.isfinite(sens_loss)

    missing, missing_red, missing_sens = model(
        user, poi, torch.zeros_like(evidence), torch.zeros_like(valid), baseline
    )
    assert torch.equal(missing, baseline)
    assert missing_red.item() == 0 and missing_sens.item() == 0

    with torch.no_grad():
        torch.nn.init.normal_(model.output_projection.weight, std=0.01)
    model.train()
    output, red_loss, sens_loss = model(
        user, poi, evidence, valid, baseline, compute_regularizers=True
    )
    total = output.square().mean() + red_loss + sens_loss
    total.backward()
    assert model.output_projection.weight.grad is not None
    assert model.query.weight.grad is not None
    assert torch.isfinite(model.output_projection.weight.grad).all()

    result = {
        "status": "PASS",
        "poi_alignment": True,
        "num_pois": int(len(graph_poi_ids)),
        "num_evidence": int(len(data["embeddings"])),
        "warm_start_exact": True,
        "missing_evidence_fallback": True,
        "regularizers_finite": True,
        "gradient_flow": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
