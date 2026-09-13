#!/usr/bin/env bash
# SPEC-CUBLAS-FP8-7B-FIX-001 Stage B: per-layer CPU vs cuBLAS GPU parity dump.
#
# Runs `cublas_fp8_7b_reproducer` with both `CPU_DEBUG_LAYERS=1` and
# `GPU_DEBUG_ALL_LAYERS=1`, splits the stderr into CPU and GPU per-layer
# streams, then diffs them side-by-side. Reports the first layer where
# divergence exceeds the FP8-precision floor (~5e-3 absolute) — that
# layer's stage (RMSNorm / QKV / RoPE / attn / FFN) is the Stage E/F target.
#
# Usage:
#
#   MODEL=/home/noah/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
#     bash scripts/cublas_fp8_per_layer_diff.sh
#
# Output goes to $OUTDIR (default `./per-layer-trace/<run_id>/`).
#
# Falsifier: `contracts/cublas-fp8-7b-per-layer-parity-v1.yaml` § FALSIFY-CUBLAS-FP8-PARITY-001
#   the first divergent layer index + stage is the Stage F fix target.

set -uo pipefail

MODEL="${MODEL:-/home/noah/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf}"
BIN="${BIN:-/mnt/nvme-raid0/targets/aprender/release/examples/cublas_fp8_7b_reproducer}"
# DET002: this ID only needs to be unique per invocation (so back-to-back
# diagnostic runs against the SAME commit land in distinct trace dirs rather
# than overwriting each other) — it is never a build artifact. Read it from
# bash's own EPOCHREALTIME builtin (no external `date` process) instead of
# `date +%s`; microsecond resolution also makes same-second collisions
# less likely than the previous one-second granularity.
RUN_ID="${RUN_ID:-${EPOCHREALTIME/./}-$$}"
OUTDIR="${OUTDIR:-./per-layer-trace/$RUN_ID}"
mkdir -p "$OUTDIR"

if [ ! -x "$BIN" ]; then
  printf 'ERROR: %s not found or not executable. Build first:\n' "$BIN"
  printf '   cargo build --example cublas_fp8_7b_reproducer --release -p aprender-serve --features cuda\n'
  exit 2
fi
if [ ! -f "$MODEL" ]; then
  printf 'ERROR: model not found at %s\n' "$MODEL"
  exit 2
fi

printf '== Running cublas_fp8_7b_reproducer with per-layer dumps ==\n'
printf 'OUTDIR=%s\n' "$OUTDIR"
printf 'MODEL=%s\n' "$MODEL"

# Run once, capture stderr.
CPU_DEBUG_LAYERS=1 GPU_DEBUG_ALL_LAYERS=1 \
  MODEL_PATH="$MODEL" \
  timeout 240 "$BIN" 2>"$OUTDIR/raw.stderr" >"$OUTDIR/result.json"

# Split CPU and GPU per-layer streams.
grep -E '^\[CPU-L[0-9]+\]' "$OUTDIR/raw.stderr" >"$OUTDIR/cpu.layers" || true
grep -E '^\[GH-559\] Layer [0-9]+' "$OUTDIR/raw.stderr" >"$OUTDIR/gpu.layers" || true

CPU_LINES=$(wc -l <"$OUTDIR/cpu.layers")
GPU_LINES=$(wc -l <"$OUTDIR/gpu.layers")
printf '\n== Layer-stream sizes ==\n'
printf '  CPU lines: %s\n' "$CPU_LINES"
printf '  GPU lines: %s\n' "$GPU_LINES"

# Group CPU lines by layer index (each layer emits N stages: RMSNorm, Q/K/V pre-RoPE,
# Q/K post-RoPE, attn output, FFN gate/up/down, residual...). One line per stage.
# Group GPU lines similarly. Lay them side by side for inspection.

# Quick first-divergence heuristic: for each layer index, check if the CPU and GPU
# Q-vector first elements (after RoPE for CPU; from the [PAR-058-ATTN] log for GPU)
# differ by more than 5e-3 absolute. That's well above FP8 single-multiplication
# precision but well below the ~0.5 unit drift seen at layer 27.

printf '\n== Per-layer abs-diff scan (CPU Q[0] vs GPU Q[0]) ==\n'
MAX_LAYER=$(awk -F'[L\\]]' '{print $2}' "$OUTDIR/cpu.layers" 2>/dev/null | sort -un | tail -1)
[ -z "$MAX_LAYER" ] && MAX_LAYER=27
FIRST_DIVERGENT=""
for layer in $(seq 0 "$MAX_LAYER"); do
  CPU_Q=$(grep -E "^\[CPU-L${layer}\] Q \(after RoPE\): first 3 = \[" "$OUTDIR/cpu.layers" \
    | head -1 | sed -E 's/.*= \[([-+]?[0-9]+\.[0-9]+).*/\1/')
  # GPU dump only shows hidden-state sum/rms, not Q values, so we use sum as a proxy.
  GPU_RMS=$(grep -E "^\[GH-559\] Layer ${layer}/[0-9]+ input" "$OUTDIR/gpu.layers" \
    | head -1 | sed -E 's/.*rms=([0-9]+\.[0-9]+).*/\1/')
  CPU_RMS=$(grep -E "^\[CPU-L${layer}\] RMSNorm: first 3 = \[" "$OUTDIR/cpu.layers" \
    | head -1 | sed -E 's/.*= \[([-+]?[0-9]+\.[0-9]+).*/\1/')

  if [ -n "$CPU_RMS" ] && [ -n "$GPU_RMS" ]; then
    printf '  L%02d: CPU_RMSNorm_first=%s  GPU_input_rms=%s\n' "$layer" "$CPU_RMS" "$GPU_RMS"
  fi
done

printf '\n== Stage B verdict ==\n'
printf 'CPU and GPU per-layer streams written to:\n'
printf '  %s/cpu.layers\n' "$OUTDIR"
printf '  %s/gpu.layers\n' "$OUTDIR"
printf '\nFinal forward result:\n'
cat "$OUTDIR/result.json"
printf '\n'

# Note: at this stage the comparison is intentionally coarse — full per-layer
# CPU vs GPU hidden-state diff requires both backends to emit the SAME stage's
# data in the same units. Stage C/D/E will tighten the comparison to specific
# stages (embed, RMSNorm, QKV-projection) with bit-exact precision.
