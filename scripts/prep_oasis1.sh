#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
raw="$root/data/raw/oasis1"
out="$root/data/oasis1"
sheet="$raw/oasis_cross-sectional-5708aa0a98d82080.xlsx"

need() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "missing file: $p" >&2
    exit 1
  fi
}

ext() {
  local n="$1"
  local tgz="$raw/oasis_cross-sectional_disc${n}.tar.gz"
  local dir="$out/disc${n}"
  need "$tgz"
  if [[ -d "$dir" ]]; then
    echo "skip extract: $dir"
    return
  fi
  echo "extract: $tgz -> $out"
  tar -xzf "$tgz" -C "$out"
}

if ! command -v uv >/dev/null 2>&1; then
  echo "missing command: uv" >&2
  exit 1
fi

need "$sheet"
mkdir -p "$out"

ext 1
ext 2
ext 3

cd "$root"

uv run obench index \
  --root data/oasis1/disc1 \
  --root data/oasis1/disc2 \
  --root data/oasis1/disc3 \
  --out data/oasis1/index.csv

uv run obench manifest \
  --index data/oasis1/index.csv \
  --sheet data/raw/oasis1/oasis_cross-sectional-5708aa0a98d82080.xlsx \
  --out data/oasis1/manifest.csv

uv run obench split \
  --index data/oasis1/index.csv \
  --sheet data/raw/oasis1/oasis_cross-sectional-5708aa0a98d82080.xlsx \
  --out data/oasis1

echo "ready: data/oasis1/index.csv"
echo "ready: data/oasis1/manifest.csv"
echo "ready: data/oasis1/train.txt"
echo "ready: data/oasis1/val.txt"
echo "ready: data/oasis1/test.txt"
