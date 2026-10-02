#!/usr/bin/env python3
"""Run a constrained, reproducible MMRec baseline on a processed IER dataset."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MMREC_SRC = ROOT / "third_party/MMRec/src"
DATA_PARENT = ROOT / "data/processed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["BPR", "LightGCN"], default="BPR")
    parser.add_argument(
        "--dataset",
        choices=["amazon_all_beauty_pilot", "amazon_musical_instruments"],
        default="amazon_all_beauty_pilot",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--eval-step", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    args = parser.parse_args()

    os.chdir(MMREC_SRC)
    sys.path.insert(0, str(MMREC_SRC))
    from utils.quick_start import quick_start

    config = {
        "use_gpu": False,
        "gpu_id": 0,
        "data_path": str(DATA_PARENT) + "/",
        "epochs": args.epochs,
        "eval_step": args.eval_step,
        "stopping_step": 5,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "embedding_size": 64,
        "learning_rate": 0.001,
        "reg_weight": 1e-4,
        "n_layers": 2,
        "topk": [10, 20],
        "metrics": ["Recall", "NDCG"],
        "valid_metric": "Recall@20",
        "hyper_parameters": ["seed"],
        "seed": [20260930],
        "checkpoint_dir": str(ROOT / "artifacts/checkpoints"),
        "save_recommended_topk": False,
    }
    quick_start(
        model=args.model,
        dataset=args.dataset,
        config_dict=config,
        save_model=False,
    )


if __name__ == "__main__":
    main()
