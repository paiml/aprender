#!/usr/bin/env bash
# KTEST-05 / F-5: prove L5 racecheck can see a missing barrier on THIS device.
# Builds fixtures/f5_smem_reduction.cu twice (with and without the barrier) for the device's own
# arch, runs racecheck on both, and requires: barrier build = 0 hazards, NO_BARRIER build > 0.
# Usage: f5_racecheck_falsifier.sh <outdir>   Env: NVCC, COMPUTE_SANITIZER
# Exit: 0 falsifier works, 1 racecheck is blind (or the clean build races), 2 setup error.
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd)
O=${1:?usage: $0 <outdir>}; mkdir -p "$O" || exit 2
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
CS=${COMPUTE_SANITIZER:-$(command -v compute-sanitizer || echo /usr/local/cuda/bin/compute-sanitizer)}
[ -x "$NVCC" ] && [ -x "$CS" ] || { echo "SETUP: nvcc=$NVCC cs=$CS" >&2; exit 2; }
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
[ -n "$cc" ] || { echo "SETUP: no device" >&2; exit 2; }
{ "$NVCC" --version | tail -n 1; "$CS" --version | tail -n 1; nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader; } > "$O/version.txt" 2>&1
declare -A H
for v in barrier nobarrier; do
  def=(); [ "$v" = nobarrier ] && def=(-DNO_BARRIER)
  "$NVCC" -O2 -arch="sm_$cc" "${def[@]}" -o "$O/f5_$v" "$D/fixtures/f5_smem_reduction.cu" 2> "$O/nvcc_$v.err" \
    || { echo "SETUP: nvcc $v failed (see $O/nvcc_$v.err)" >&2; exit 2; }
  "$CS" --tool racecheck --log-file "$O/racecheck_$v.log" "$O/f5_$v" > "$O/run_$v.out" 2>&1
  n=$(grep -oE 'RACECHECK SUMMARY: [0-9]+' "$O/racecheck_$v.log" | tail -n 1 | grep -oE '[0-9]+$')
  [ -n "$n" ] || { echo "RED_NO_SUMMARY $v"; exit 1; }
  H[$v]=$n
done
echo "sm_$cc barrier=${H[barrier]} nobarrier=${H[nobarrier]} $(cat "$O/run_barrier.out" | tr '\n' ' ')" | tee "$O/verdict.txt"
if [ "${H[barrier]}" -eq 0 ] && [ "${H[nobarrier]}" -gt 0 ]; then echo PASS | tee -a "$O/verdict.txt"; exit 0; fi
echo "RED: racecheck did not separate the builds" | tee -a "$O/verdict.txt"; exit 1
