#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

mapfile -t PYTHON_FILES < <(find "$ROOT/scripts" "$ROOT/third_party/MMPOI" \
  -type f -name '*.py' -not -path '*/__pycache__/*' | sort)
mapfile -t SHELL_FILES < <(find "$ROOT/scripts" -type f -name '*.sh' | sort)

"$PYTHON" -m py_compile "${PYTHON_FILES[@]}"
for script in "${SHELL_FILES[@]}"; do
  bash -n "$script"
done

"$PYTHON" - <<'PY' "$ROOT"
import json
import pathlib
import sys
import yaml

root = pathlib.Path(sys.argv[1])
yaml.safe_load((root / "environment.yml").read_text(encoding="utf-8"))
yaml.safe_load((root / "configs/experiment_manifest.yaml").read_text(encoding="utf-8"))
for path in sorted((root / "artifacts").rglob("*.json")):
    if path.stat().st_size < 1_000_000:
        json.loads(path.read_text(encoding="utf-8"))
print("repository validation: PASS")
PY
