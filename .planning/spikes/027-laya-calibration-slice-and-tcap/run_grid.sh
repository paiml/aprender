#!/usr/bin/env bash
# Spike 027 grid, main cells first (the ~90 min budget cuts from the end of this list).
set -euo pipefail
cd "$(dirname "$0")/../../.."
LOG=models/decide/spike-027/logs
mkdir -p "$LOG"
for cell in "s16 es12" "s16 fixed12" "s64 fixed12" "s64 es12" "s16 r1" "s64 r1" "s16 es12m10" "s64 es12m10"; do
  set -- $cell
  echo "=== $1 $2 $(date +%H:%M:%S)"
  uv run --frozen --project scripts/laya_train python .planning/spikes/027-laya-calibration-slice-and-tcap/run_cell.py \
    --size "$1" --recipe "$2" --seeds 13,17,23 > "$LOG/$1-$2.log" 2>&1
  grep '^RESULT' "$LOG/$1-$2.log" || true
done
echo "GRID DONE $(date +%H:%M:%S)"
