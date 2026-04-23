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

uv run obench cnn2d \
  --index data/oasis1/index.csv \
  --sheet data/raw/oasis1/oasis_cross-sectional-5708aa0a98d82080.xlsx \
  --splits data/oasis1 \
  --out reports/cnn2d_best \
  --epochs 30 \
  --bs 8 \
  --lr 3e-4 \
  --axis 1 \
  --ch 3 \
  --arch tiny \
  --pool mean \
  --pick mid \
  --slices 24

uv run obench cal \
  --pred reports/cnn2d_best/run/pred.json \
  --sheet data/raw/oasis1/oasis_cross-sectional-5708aa0a98d82080.xlsx \
  --out reports/cal/cnn2d_best

echo "ready: reports/cnn2d_best/run/metrics.json"
echo "ready: reports/cnn2d_best/run/pred_stats.json"
echo "ready: reports/cal/cnn2d_best/metrics.json"
