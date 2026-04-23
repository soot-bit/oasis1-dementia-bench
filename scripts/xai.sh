#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
idx="$root/data/oasis1/index.csv"
spl="$root/data/oasis1"
sheet="$root/data/raw/oasis1/oasis_cross-sectional-5708aa0a98d82080.xlsx"

need() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "missing path: $path" >&2
    exit 1
  fi
}

need "$idx"
need "$spl/train.txt"
need "$spl/val.txt"
need "$spl/test.txt"
need "$sheet"

cd "$root"

uv run obench xaitab \
  --index data/oasis1/index.csv \
  --sheet data/raw/oasis1/oasis_cross-sectional-5708aa0a98d82080.xlsx \
  --splits data/oasis1 \
  --out reports/xai/tab \
  --repeats 100 \
  --top 12

cnn_run=""
if [[ -e "$root/reports/cnnlit_best/run/model.pt" ]]; then
  cnn_run="reports/cnnlit_best/run"
elif [[ -e "$root/reports/cnn2d_best/run/model.pt" ]]; then
  cnn_run="reports/cnn2d_best/run"
fi

if [[ -n "$cnn_run" ]]; then
  uv run obench xaicnn \
    --index data/oasis1/index.csv \
    --sheet data/raw/oasis1/oasis_cross-sectional-5708aa0a98d82080.xlsx \
    --splits data/oasis1 \
    --run "$cnn_run" \
    --out reports/xai/cnn \
    --n 2
  echo "ready: reports/xai/cnn/summary.md"
else
  echo "skipping CNN XAI: run scripts/bench_cnn_best.sh first" >&2
fi

echo "ready: reports/xai/tab/summary.md"
