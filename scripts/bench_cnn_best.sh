#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
idx="$root/data/oasis1/index.csv"
spl="$root/data/oasis1"
sheet="$root/data/raw/oasis1/oasis_cross-sectional-5708aa0a98d82080.xlsx"

need() {
  local p="$1"
  if [[ ! -e "$p" ]]; then
    echo "missing path: $p" >&2
    exit 1
  fi
}

if ! command -v uv >/dev/null 2>&1; then
  echo "missing command: uv" >&2
  exit 1
fi

need "$idx"
need "$spl/train.txt"
need "$spl/val.txt"
need "$spl/test.txt"
need "$sheet"

cd "$root"

echo "Training improved CNN baseline with Attention Pooling and Spatial Augmentation..."

uv run obench cnnlit \
  --index data/oasis1/index.csv \
  --sheet data/raw/oasis1/oasis_cross-sectional-5708aa0a98d82080.xlsx \
  --splits data/oasis1 \
  --out reports/cnnlit_best \
  --epochs 50 \
  --bs 8 \
  --lr 1e-4 \
  --axis 1 \
  --ch 3 \
  --arch tiny \
  --pool attn \
  --pick mid \
  --slices 24 \
  --patience 12 \
  --workers 4 \
  --precision auto

uv run obench cal \
  --pred reports/cnnlit_best/run/pred.json \
  --sheet data/raw/oasis1/oasis_cross-sectional-5708aa0a98d82080.xlsx \
  --out reports/cal/cnnlit_best

echo "Generating explainability reports for the improved model..."

uv run obench xaicnn \
  --index data/oasis1/index.csv \
  --sheet data/raw/oasis1/oasis_cross-sectional-5708aa0a98d82080.xlsx \
  --splits data/oasis1 \
  --run reports/cnnlit_best/run \
  --out reports/xai/cnn_improved \
  --n 3 \
  --split test

echo "ready: reports/cnnlit_best/run/metrics.json"
echo "ready: reports/cnnlit_best/run/pred_stats.json"
echo "ready: reports/xai/cnn_improved/summary.md"
