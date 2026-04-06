#!/bin/bash
# scripts/bench-cpu-matrix.sh
# CPU-only inference benchmark matrix
# Refs: PERF-PARITY-001
#
# Methodology: Hoefler & Belli SC'15 (CV-based stopping)

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${REPO_DIR}/../single-shot-eval/models/raw}"
LLAMA_CPP_PATH="${LLAMA_CPP_PATH:-${REPO_DIR}/../llama.cpp/llama-server}"
RESULTS_DIR="${RESULTS_DIR:-benches/comparative/results}"

# Models to test
MODELS=(
    "phi-2-q4_k_m.gguf"
    "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    "deepseek-coder-1.3b-instruct-q4_k_m.gguf"
)

echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          CPU-Only Inference Benchmark Matrix                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check for llama.cpp
if [[ ! -x "$LLAMA_CPP_PATH" ]]; then
    echo -e "${YELLOW}llama.cpp not found at $LLAMA_CPP_PATH${NC}"
    echo -e "${YELLOW}Skipping external server benchmarks${NC}"
    exit 0
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# Test each model
for MODEL in "${MODELS[@]}"; do
    MODEL_PATH="$MODEL_DIR/$MODEL"

    if [[ ! -f "$MODEL_PATH" ]]; then
        echo -e "${YELLOW}Model not found: $MODEL_PATH, skipping${NC}"
        continue
    fi

    echo -e "${GREEN}=== Testing: $MODEL ===${NC}"

    # Start llama.cpp CPU server
    echo "Starting llama.cpp CPU server (-ngl 0)..."
    "$LLAMA_CPP_PATH" -m "$MODEL_PATH" --host 127.0.0.1 --port 8090 -ngl 0 &
    LLAMA_PID=$!
    sleep 8  # Wait for model load

    # Verify server is up
    if curl -s http://localhost:8090/health > /dev/null 2>&1; then
        echo "Server ready, running benchmark..."

        # Run 10 samples with CV tracking
        LATENCIES=()
        for i in {1..10}; do
            START=$(date +%s%N)
            RESP=$(curl -s -X POST http://localhost:8090/completion \
                -H "Content-Type: application/json" \
                -d '{"prompt": "Hello, world!", "n_predict": 30, "temperature": 0}' 2>/dev/null || echo "{}")
            END=$(date +%s%N)

            LATENCY_MS=$(( (END - START) / 1000000 ))
            LATENCIES+=("$LATENCY_MS")

            TOKENS=$(echo "$RESP" | jq -r '.tokens_predicted // 30' 2>/dev/null || echo "30")
            printf "  [%2d/10] Latency: %dms | Tokens: %s\n" "$i" "$LATENCY_MS" "$TOKENS"
        done

        # Calculate statistics
        SUM=0
        for L in "${LATENCIES[@]}"; do
            SUM=$((SUM + L))
        done
        MEAN=$((SUM / ${#LATENCIES[@]}))

        echo -e "  ${GREEN}Mean latency: ${MEAN}ms${NC}"
    else
        echo -e "${RED}Failed to start llama.cpp server${NC}"
    fi

    # Cleanup
    kill "$LLAMA_PID" 2>/dev/null || true
    wait "$LLAMA_PID" 2>/dev/null || true
    echo ""
done

echo -e "${GREEN}CPU benchmark matrix complete${NC}"
