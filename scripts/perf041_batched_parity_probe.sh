#!/usr/bin/env bash
# FALSIFY-CB-006 / CB-008 (contracts/continuous-batching-v1.yaml), executable.
#
# Starts a CUDA server, takes an m=1 reference, then fires concurrent requests
# and checks that the batched output still matches. Exits 1 on a code defect,
# 2 when the run cannot decide (no m>1 batch formed, unstable reference), 0 on
# pass. The 2 matters: a guard that names a code cause for a box it could not
# evaluate has fired three times in this repo in one day.
set -uo pipefail

. scripts/apr_bin.sh || exit 1
BIN="$APR"

MODEL="${APR_MODELS:-$HOME/models}/${PERF041_MODEL:-qwen2.5-coder-1.5b-instruct-q4_k_m.gguf}"
PORT="${PERF041_PORT:-8473}"
OUT="${PERF041_OUT:-/tmp/perf041-parity}"
LOG="$OUT/server.log"

mkdir -p "$OUT"
if [ ! -f "$MODEL" ]; then
    printf 'perf041-parity: model not found: %s\n' "$MODEL" >&2
    exit 2
fi

"$BIN" serve run "$MODEL" --gpu --port "$PORT" --context-length 4096 > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null' EXIT

i=0
while [ "$i" -lt 300 ]; do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null)
    if [ "$code" = "200" ]; then break; fi
    i=$((i + 1))
    sleep 1
done
if [ "$i" -ge 300 ]; then
    printf 'perf041-parity: server never became healthy\n' >&2
    exit 2
fi

printf 'perf041-parity: %s\n' "$("$BIN" --version 2>&1)"
printf 'perf041-parity: model %s\n' "$MODEL"

# Status read directly, never through a pipe.
python3 scripts/perf041_batched_parity_probe.py \
    --url "http://127.0.0.1:${PORT}/v1/chat/completions" \
    --server-log "$LOG" \
    --max-tokens "${PERF041_MAX_TOKENS:-400}"
rc=$?
printf 'perf041-parity: exit %d (0=PASS 1=code defect 2=unmeasurable)\n' "$rc"
exit "$rc"
