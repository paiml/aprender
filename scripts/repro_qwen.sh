#!/usr/bin/env bash
# PMAT-COR-001-H1: Token 0 Collapse Reproduction Script
# Purpose: Reproduce and trace Token 0 collapse issue for investigation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_PATH="${PROJECT_ROOT}/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
TRACE_OUTPUT="${PROJECT_ROOT}/trace_h1.json"

echo "=== PMAT-COR-001-H1: Token 0 Collapse Investigation ==="
echo "Date: $(date -Iseconds)"
echo "Model: ${MODEL_PATH}"
echo "Trace Output: ${TRACE_OUTPUT}"
echo ""

# Verify model exists
if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "ERROR: Model not found at ${MODEL_PATH}"
    exit 1
fi

echo "=== Step 1: Running apr with transformer block tracing ==="
echo "Command: apr run ${MODEL_PATH} --prompt \"2+2=\" --trace --trace-verbose --trace-output ${TRACE_OUTPUT}"
echo ""

# Run the reproduction with full tracing
cargo run -p apr-cli --release -- run "${MODEL_PATH}" \
    --prompt "2+2=" \
    --trace \
    --trace-verbose \
    --trace-output "${TRACE_OUTPUT}" \
    2>&1 | tee "${PROJECT_ROOT}/repro_output.log"

echo ""
echo "=== Step 2: Checking trace output ==="
if [[ -f "${TRACE_OUTPUT}" ]]; then
    echo "Trace file created: $(ls -la "${TRACE_OUTPUT}")"
    echo ""
    echo "=== First 50 lines of trace ==="
    head -50 "${TRACE_OUTPUT}"
else
    echo "WARNING: Trace output file not created"
fi

echo ""
echo "=== Reproduction complete ==="
echo "Review trace_h1.json and repro_output.log for analysis"
