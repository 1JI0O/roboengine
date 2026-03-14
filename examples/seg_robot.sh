#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT"

python "$SCRIPT_DIR/run_roboseg_on_color_dir.py" \
  --color-dir /data/haoxiang/data/airexo2/task_0012/train/scene_0001/cam_105422061350/color \
  --output-dir /data/haoxiang/data/airexo2/task_0012/train/scene_0001/cam_105422061350/roboseg_mask_out \
  --device cuda \
  --anchor-frequency 8