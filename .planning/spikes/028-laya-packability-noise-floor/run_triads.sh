#!/usr/bin/env bash
# Spike 028: Rust dump (zz_triad_dump, temporarily in crates/aprender-decide/examples/) then the torch fp32 +
# float64 triad, per (checkpoint, row set). Rust dumps (final-norm blocks, ~100-200 MB each) go to work/ and
# are deleted once the triad JSON is written.
#   run_triads.sh test|indist
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO" || exit 1
HERE=.planning/spikes/028-laya-packability-noise-floor
BASE="$HOME/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
D27=.planning/spikes/027-laya-calibration-slice-and-tcap/data
BIN="$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys;print(json.load(sys.stdin)["target_directory"])')/release/examples/zz_triad_dump"
set_=${1:?test|indist}; only=${2:-}
if [ "$set_" = test ]; then
  specs="base:s64 s64-es12-seed17-rep:s64 s64-r1-seed13:s64 s16-r1-seed13:s16 s64-es12-seed23-rep:s64"
else
  specs="base:indist s64-es12-seed17-rep:indist s64-es12-seed23-rep:indist s64-r1-seed13:indist"
fi
mkdir -p "$HERE/results/triad" "$HERE/work"
for spec in $specs; do
  [ -n "$only" ] && [ "${spec%%:*}" != "$only" ] && continue
  run="${spec%%:*}"; d="${spec##*:}"
  if [ "$d" = indist ]; then dd="$HERE/data/indist"; else dd="$D27/$d"; fi
  if [ "$run" = base ]; then ck="$BASE"; tck=base; stored="models/decide/spike-027/s64-r1-seed13/zero-shot-probs.json"
  else ck="models/decide/spike-027/$run/checkpoint"; tck="$ck"; stored="models/decide/spike-027/$run/eval-probs.json"; fi
  [ "$set_" = indist ] && stored=""
  extra=""; [ "$run" = s64-r1-seed13 ] && extra="8.758205757575377"
  name="$set_-$run"; w="$HERE/work/$name"
  "$BIN" "$ck" "$dd" "$w" $extra > "$HERE/results/triad/$name.rust.log" 2>&1; rc=$?
  echo "$name rust exit=$rc"
  [ $rc -eq 0 ] || continue
  args=(--ckpt "$tck" --data "$dd" --rust "$w" --out "$HERE/results/triad/$name.json")
  [ -n "$stored" ] && args+=(--stored "$stored")
  [ -n "$extra" ] && args+=(--extra-t "$extra")
  uv run --frozen --project scripts/laya_train python "$HERE/torch_triad.py" "${args[@]}" \
    > "$HERE/results/triad/$name.txt" 2> "$HERE/results/triad/$name.torch.log"; rc=$?
  echo "$name triad exit=$rc"
  [ $rc -eq 0 ] && rm -rf "$w"
done
