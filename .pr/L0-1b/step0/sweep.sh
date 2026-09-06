#!/usr/bin/env bash
# L0-1b step 1 — bisection by controls on the 1.5B (lambda). One apr parity run per arm.
set -u
APR=/mnt/nvme-raid0/targets/l0-1b/release/apr
OUT="$(dirname "$0")"
eval "$(grep '^PROMPT=' /home/noah/src/aprender-worktrees/L0-1/scripts/check_model_parity.sh)"
M15=/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
M7=/home/noah/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf
run() { # run <arm> <model> [ENV=VAL ...]
  local arm=$1 model=$2; shift 2
  local t0=$(date +%s)
  env "$@" "$APR" parity "$model" --prompt "$PROMPT" --json > "$OUT/$arm.json" 2> "$OUT/$arm.err"
  echo "$arm rc=$? $(( $(date +%s) - t0 ))s env=[$*]" >> "$OUT/progress.log"
}
: > "$OUT/progress.log"
run A0-baseline        "$M15"
run A1-graph-off       "$M15" SKIP_CUDA_GRAPH=1
run A2-fp8-decode-off  "$M15" FP8_DECODE=0
run A3-fp8-all-off     "$M15" FP8_PREFILL=0 FP8_DECODE=0
run A4-flash-off       "$M15" FLASH_DECODE=0
run A5-fused-gateup-off "$M15" FUSED_GATE_UP=0
run A6-all-off         "$M15" SKIP_CUDA_GRAPH=1 FP8_PREFILL=0 FP8_DECODE=0 FLASH_DECODE=0 FUSED_GATE_UP=0
run R7B-baseline       "$M7"
echo DONE >> "$OUT/progress.log"
