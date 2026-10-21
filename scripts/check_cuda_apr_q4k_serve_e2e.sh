#!/usr/bin/env bash
# check_cuda_apr_q4k_serve_e2e.sh -- PP-LLAMA-001 row 8, end to end on a GPU (#5831).
#
# The defect: `cuda-batch = ["cuda"]` ran the implication the wrong way, so a plain
# `--features cuda` build compiled a stub of `start_apr_q4k_server_gpu` that returned
# Err("Q4K batch scheduler not available"), and handlers.rs caught that Err and quietly
# served through the generic per-tensor cuMemAlloc path. HTTP 200, wrong path.
#
# What this runs: build `apr` with `--features cuda` ONLY (never cuda-batch), serve a Q4K
# `.apr` with --gpu, and require
#   1. stderr has the "[GH-471] Entering Q4K GPU path" line      (the real function ran)
#   2. stderr has neither "Q4K batch scheduler not available"    (the stub's Err)
#      nor "Q4K pool-allocator path declined"                    (the silent downgrade)
#   3. GET /v1/effective-config reports build_features_cli containing "cuda"
#      (PP-2: the build is judged from what the server says, not from a flag we passed)
# A tree that lacks the fix fails 1 and 2, so this is red there by construction.
#
# Usage: check_cuda_apr_q4k_serve_e2e.sh [--tree DIR] [--model FILE] [--port N]
# Exit: 0 all three held · 1 one broke · 2 could not check (no GPU, no model, build failed).
set -uo pipefail
TREE="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${APR_Q4K_MODEL:-/home/noah/models/qwen2.5-coder-1.5b-instruct-q4k.apr}"
PORT=18631
while [ $# -gt 0 ]; do
  case "$1" in
    --tree) [ $# -ge 2 ] || { echo "--tree needs a value" >&2; exit 2; }; TREE="$(cd "$2" && pwd)" || exit 2; shift 2 ;;
    --model) [ $# -ge 2 ] || { echo "--model needs a value" >&2; exit 2; }; MODEL="$2"; shift 2 ;;
    --port) [ $# -ge 2 ] || { echo "--port needs a value" >&2; exit 2; }; PORT="$2"; shift 2 ;;
    -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    *) echo "check_cuda_apr_q4k_serve_e2e: unknown argument '$1'" >&2; exit 2 ;;
  esac
done
env_fail() { echo "check_cuda_apr_q4k_serve_e2e: ENV - $1" >&2; exit 2; }
command -v nvidia-smi > /dev/null 2>&1 || env_fail "nvidia-smi missing (no GPU host)"
nvidia-smi -L > /dev/null 2>&1 || env_fail "nvidia-smi sees no GPU"
command -v curl > /dev/null 2>&1 || env_fail "curl missing"
[ -f "$MODEL" ] || env_fail "model $MODEL not found (set APR_Q4K_MODEL)"
OUT="$(mktemp -d)" || exit 2
SRV=""
trap '[ -n "$SRV" ] && kill "$SRV" 2> /dev/null; rm -rf -- "${OUT:?}"' EXIT
TARGET="${CARGO_TARGET_DIR:-/mnt/nvme-raid0/targets/cuda-e2e-$(basename "$TREE")}"
(cd "$TREE" && CARGO_TARGET_DIR="$TARGET" cargo build --release -p apr-cli --bin apr --features cuda) > "$OUT/build.log" 2>&1 \
  || { tail -n 20 "$OUT/build.log" >&2; env_fail "cargo build --features cuda failed in $TREE"; }
APR="$TARGET/release/apr"
[ -x "$APR" ] || env_fail "no binary at $APR"
echo "binary: $("$APR" --version 2>&1 | head -1) tree=$TREE"
"$APR" serve run "$MODEL" --gpu --port "$PORT" > "$OUT/serve.out" 2> "$OUT/serve.err" < /dev/null &
SRV=$!
up=0
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/v1/effective-config" -o "$OUT/cfg.json" 2> /dev/null && { up=1; break; }
  kill -0 "$SRV" 2> /dev/null || break
  sleep 1
done
FAIL=0
broke() { echo "  BROKE $1"; FAIL=$((FAIL + 1)); }
ok() { echo "  ok    $1"; }
[ "$up" = 1 ] || { tail -n 20 "$OUT/serve.err" >&2; broke "server never answered /v1/effective-config"; }
if grep -q '\[GH-471\] Entering Q4K GPU path' "$OUT/serve.err"; then ok "1 the real Q4K GPU path ran"; else broke "1 no '[GH-471] Entering Q4K GPU path' line"; fi
if grep -qE 'Q4K batch scheduler not available|Q4K pool-allocator path declined' "$OUT/serve.err"; then
  broke "2 stub/downgrade: $(grep -hE 'Q4K batch scheduler not available|Q4K pool-allocator path declined' "$OUT/serve.err" | head -1)"
else ok "2 no stub error and no silent downgrade"; fi
if [ -s "$OUT/cfg.json" ] && grep -q '"build_features_cli"[^]]*"cuda"' "$OUT/cfg.json"; then ok "3 effective-config build_features_cli has cuda"; else broke "3 build_features_cli lacks cuda (or no config)"; fi
echo "check_cuda_apr_q4k_serve_e2e: $FAIL broke"
[ "$FAIL" -eq 0 ]
