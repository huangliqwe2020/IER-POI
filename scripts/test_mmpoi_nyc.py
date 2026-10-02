#!/usr/bin/env python3
"""Validate a completed MMPOI NYC run and emit a machine-readable summary."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/mmpoi/nyc_fixed_seed618"


def parse_metrics(path: Path) -> dict:
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key] = ast.literal_eval(value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--min-hr20", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    metrics_path = args.run_dir / "metrics-val.txt"
    checkpoint_path = args.run_dir / "checkpoints/best_epoch.state.pt"
    if not metrics_path.is_file() or not checkpoint_path.is_file():
        raise SystemExit(f"incomplete run: expected {metrics_path} and {checkpoint_path}")

    metrics = parse_metrics(metrics_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required_states = {
        "poi_embed_state_dict", "geo_embed_state_dict", "image_embed_state_dict",
        "review_embed_state_dict", "meta_embed_state_dict",
        "embed_fuse_multimodal_state_dict", "embed_fuse_model1_state_dict",
        "embed_fuse_model2_state_dict", "user_embed_state_dict",
        "time_embed_state_dict", "cat_embed_state_dict", "seq_model_state_dict",
    }
    missing = sorted(required_states - checkpoint.keys())
    if missing:
        raise SystemExit(f"checkpoint is not independently reusable; missing: {missing}")

    best = {key.removeprefix("epoch_val_"): float(value)
            for key, value in checkpoint["epoch_val_metrics"].items()}
    for key, value in best.items():
        if not math.isfinite(value):
            raise SystemExit(f"non-finite checkpoint metric: {key}={value}")
    if best["top20_acc"] < args.min_hr20:
        raise SystemExit(f"HR@20 {best['top20_acc']:.6f} < required {args.min_hr20:.6f}")

    result = {
        "status": "PASS",
        "run_dir": str(args.run_dir.resolve()),
        "checkpoint_epoch_zero_based": int(checkpoint["epoch"]),
        "checkpoint_metrics": best,
        "reported_best_epoch": int(metrics["best_epoch"]),
        "reported_best": {
            "HR@1": float(metrics["best_top1_acc"]),
            "HR@5": float(metrics["best_top5_acc"]),
            "HR@10": float(metrics["best_top10_acc"]),
            "HR@20": float(metrics["best_top20_acc"]),
            "mAP@20": float(metrics["best_mAP20"]),
            "NDCG@20": float(metrics["best_ndcg20"]),
            "MRR": float(metrics["best_mrr"]),
        },
        "epochs_completed": len(metrics["val_epochs_loss_list"]),
        "checkpoint_state_groups": len(required_states),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
