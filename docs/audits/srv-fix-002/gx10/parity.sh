#!/usr/bin/env bash
# parity.sh <apr bin> <model> <promptdir> <outdir>: greedy 64 tok, flash vs f32, sequential
set -uo pipefail
B=$1; M=$2; P=$3; O=$4; mkdir -p "$O"; same=0; tot=0
for f in "$P"/*.txt; do
  n=$(basename "$f" .txt)
  for path in flash f32; do
    APR_QWEN35_PREFILL_ATTENTION=$path flock /tmp/apr-gpu.lock "$B" run "$M" --prompt "$(cat "$f")" --max-tokens 64 \
      > "$O/$n.$path.out" 2> "$O/$n.$path.err"; echo "$n $path rc=$?" >> "$O/rc.txt"
  done
  tot=$((tot+1))
  if cmp -s "$O/$n.flash.out" "$O/$n.f32.out"; then same=$((same+1)); echo "$n IDENTICAL"; else echo "$n DIFFER at byte $(cmp "$O/$n.flash.out" "$O/$n.f32.out" | awk '{print $5}')"; fi
done
echo "parity=$same/$tot"
