#!/usr/bin/env python3
"""Report the software and accelerator environment used by IER-POI."""

from __future__ import annotations

import importlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

import torch


PACKAGES = [
    "torch",
    "torchvision",
    "torchaudio",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "networkx",
    "PyYAML",
    "tqdm",
    "matplotlib",
    "Pillow",
    "transformers",
    "gdown",
    "python-geohash",
    "duckdb",
    "lmdb",
]


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> None:
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in PACKAGES},
        "torch": {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
    }
    required_imports = ["numpy", "pandas", "scipy", "sklearn", "networkx", "yaml"]
    report["required_imports_ok"] = all(
        importlib.util.find_spec(name) is not None for name in required_imports
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["required_imports_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
