#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def read_metric_list(path: Path, key: str) -> float:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(key + "="):
            values = ast.literal_eval(line.split("=", 1)[1])
            return float(values[-1])
    raise KeyError(f"{key} not found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for name in ("clean", "duplicate", "noise", "missing", "sparse"):
        metric_file = args.root / name / "metrics-val.txt"
        result[name] = {
            "HR@20": read_metric_list(metric_file, "val_epochs_top20_acc_list"),
            "NDCG@20": read_metric_list(metric_file, "val_epochs_ndcg20_list"),
            "mAP@20": read_metric_list(metric_file, "val_epochs_mAP20_list"),
            "MRR": read_metric_list(metric_file, "val_epochs_mrr_list"),
        }
    clean = result["clean"]
    for name, metrics in result.items():
        metrics["delta_HR@20"] = metrics["HR@20"] - clean["HR@20"]
        metrics["delta_NDCG@20"] = metrics["NDCG@20"] - clean["NDCG@20"]
    output = args.root / "robustness_summary.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
