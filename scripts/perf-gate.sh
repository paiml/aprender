#!/usr/bin/env bash
# perf-gate.sh — CI throughput gate using probador llm load
#
# CONTRACT: cuda-graph-safety-v1 FALSIFY-CGS-001
# Ensures CUDA context is not poisoned by failed graph capture.
# Minimum decode tok/s must be met WITHOUT CUDA_GRAPH_ENABLE=1.
#
# Usage:
#   ./scripts/perf-gate.sh [model_path] [min_decode_tok_s]
#
# Requirements:
#   - probador with `llm` subcommand
#   - apr serve (from aprender)
#   - NVIDIA GPU with locked clocks
#
# Exit codes:
#   0 = PASS (decode tok/s >= threshold)
#   1 = FAIL (throughput below threshold)
#   2 = ERROR (server failed to start, probador not found, etc.)

set -euo pipefail

MODEL="${1:-/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf}"
MIN_TOK_S="${2:-200}"
PORT=18099
DURATION="10s"
WARMUP="3s"

# Find probador
PROBADOR=$(command -v probador 2>/dev/null || echo "/mnt/nvme-raid0/targets/probar/release/probador")
if [ ! -x "$PROBADOR" ]; then
    echo "ERROR: probador not found. Build from ../probar or install." >&2
    exit 2
fi

# Find apr
APR=$(command -v apr 2>/dev/null || echo "/mnt/nvme-raid0/targets/aprender/release/apr")
if [ ! -x "$APR" ]; then
    echo "ERROR: apr not found. Build from ../aprender or install." >&2
    exit 2
fi

# Start server (NO CUDA_GRAPH_ENABLE — must work without it per contract)
unset CUDA_GRAPH_ENABLE 2>/dev/null || true
"$APR" serve run "$MODEL" --gpu --port "$PORT" > /tmp/perf-gate-server.log 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null" EXIT

# Wait for server
for i in $(seq 1 30); do
    if curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -qi "ok\|healthy\|alive"; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server died during startup. Log:" >&2
        cat /tmp/perf-gate-server.log >&2
        exit 2
    fi
    sleep 1
done

# Check for context poisoning in server log
if grep -q "Graph capture failed\|CUDA_ERROR_UNKNOWN" /tmp/perf-gate-server.log; then
    echo "FAIL: CUDA context poisoned by failed graph capture." >&2
    echo "  CONTRACT VIOLATION: cuda-graph-safety-v1 FALSIFY-CGS-001" >&2
    grep "Graph capture failed\|CUDA_ERROR_UNKNOWN" /tmp/perf-gate-server.log >&2
    exit 1
fi

# Run probador
RESULT=$("$PROBADOR" llm load \
    --url "http://127.0.0.1:$PORT" \
    --model "perf-gate" \
    --concurrency 1 \
    --duration "$DURATION" \
    --warmup "$WARMUP" \
    --max-tokens 64 \
    --stream false \
    --runtime-name "perf-gate" \
    --num-layers 28 \
    2>&1)

# Extract decode tok/s
DECODE_TOK_S=$(echo "$RESULT" | grep "Decode tok/s:" | awk '{print $NF}')
if [ -z "$DECODE_TOK_S" ]; then
    echo "ERROR: Could not extract decode tok/s from probador output." >&2
    echo "$RESULT" >&2
    exit 2
fi

# Gate check
PASS=$(python3 -c "print('PASS' if float('$DECODE_TOK_S') >= float('$MIN_TOK_S') else 'FAIL')")

echo "=== Performance Gate ==="
echo "  Decode tok/s: $DECODE_TOK_S"
echo "  Threshold:    $MIN_TOK_S"
echo "  Result:       $PASS"
echo "  Contract:     cuda-graph-safety-v1 FALSIFY-CGS-001"

if [ "$PASS" = "FAIL" ]; then
    echo ""
    echo "FAIL: Throughput $DECODE_TOK_S < $MIN_TOK_S tok/s" >&2
    echo "  Possible cause: CUDA context poisoned by graph capture." >&2
    echo "  Check: server log at /tmp/perf-gate-server.log" >&2
    exit 1
fi

echo "PASS: $DECODE_TOK_S tok/s >= $MIN_TOK_S tok/s"
