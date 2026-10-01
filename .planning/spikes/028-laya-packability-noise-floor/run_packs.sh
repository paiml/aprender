#!/usr/bin/env bash
# Spike 028: the real `just laya-pack` on each spike-027 candidate, on the gate's test eval set.
# A PACKED line here is a measurement, NOT a deploy and NOT a gate run. .apr files go only to
# models/decide/spike-028/ (gitignored) and are deleted after measuring.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO" || exit 1
BASE="$HOME/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
D27=.planning/spikes/027-laya-calibration-slice-and-tcap/data
OUT=.planning/spikes/028-laya-packability-noise-floor/results/pack
mkdir -p "$OUT" models/decide/spike-028
for spec in s64-es12-seed17-rep:s64 s64-r1-seed13:s64 s16-r1-seed13:s16 s64-es12-seed23-rep:s64; do
  run="${spec%%:*}"; size="${spec##*:}"
  start=$(date +%s)
  just laya-pack "models/decide/spike-027/$run" "$D27/$size" "$BASE" "models/decide/spike-028/$run.apr" \
    > "$OUT/$run.log" 2>&1
  rc=$?
  end=$(date +%s)
  echo "exit=$rc wall_s=$((end - start)) arch=$(uname -m)" >> "$OUT/$run.log"
  echo "$run exit=$rc $(/usr/bin/grep -E '^(PACKED|REFUSED)' "$OUT/$run.log")"
done
