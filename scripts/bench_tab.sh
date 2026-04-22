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

uv run obench eda \
  --index data/oasis1/index.csv \
  --sheet data/raw/oasis1/oasis_cross-sectional-5708aa0a98d82080.xlsx \
  --out reports/eda

uv run obench tab \
  --index data/oasis1/index.csv \
  --sheet data/raw/oasis1/oasis_cross-sectional-5708aa0a98d82080.xlsx \
  --splits data/oasis1 \
  --out reports/tab

uv run obench errtab \
  --errors reports/tab/run/errors.csv \
  --out docs/err/tab

uv run obench cal \
  --pred reports/tab/run/errors.csv \
  --sheet data/raw/oasis1/oasis_cross-sectional-5708aa0a98d82080.xlsx \
  --out reports/cal/tab

echo "ready: reports/eda"
echo "ready: reports/tab/run/summary.csv"
echo "ready: reports/tab/run/compare_auc.png"
echo "ready: docs/err/tab/README.md"
echo "ready: reports/cal/tab/metrics.json"
