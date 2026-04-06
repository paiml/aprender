#!/bin/bash
# scripts/bench-gguf-gpu-matrix.sh
# GGUF GPU inference benchmark: realizar vs ollama vs llama.cpp
# Refs: PERF-PARITY-001
#
# Methodology: Hoefler & Belli SC'15 (CV-based stopping)
# Toyota Way: Genchi Genbutsu (measure actual, not theoretical)

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${REPO_DIR}/../single-shot-eval/models/raw}"
LLAMA_CPP_PATH="${LLAMA_CPP_PATH:-${REPO_DIR}/../llama.cpp/llama-server}"
RESULTS_DIR="${RESULTS_DIR:-benches/comparative/results}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Benchmark parameters (per Hoefler & Belli SC'15)
MIN_SAMPLES=10
MAX_SAMPLES=30
CV_THRESHOLD=0.10  # 10% coefficient of variation

# Models to test
MODELS=(
    "phi-2-q4_k_m.gguf"
    "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
)

# Servers to benchmark
declare -A SERVERS=(
    ["llama_gpu"]="8082"
    ["ollama"]="11434"
)

echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          GGUF GPU Inference Benchmark Matrix                   ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Methodology: CV-based stopping (threshold: ${CV_THRESHOLD})"
echo "Samples: min=${MIN_SAMPLES}, max=${MAX_SAMPLES}"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to calculate CV
calculate_cv() {
    local -a samples=("$@")
    local n=${#samples[@]}

    if [[ $n -lt 2 ]]; then
        echo "1.0"
        return
    fi

    # Calculate mean
    local sum=0
    for s in "${samples[@]}"; do
        sum=$((sum + s))
    done
    local mean=$((sum / n))

    # Calculate variance
    local var_sum=0
    for s in "${samples[@]}"; do
        local diff=$((s - mean))
        var_sum=$((var_sum + diff * diff))
    done
    local variance=$((var_sum / (n - 1)))

    # CV = stddev / mean
    local stddev
    stddev=$(echo "scale=4; sqrt($variance)" | bc)
    local cv
    cv=$(echo "scale=4; $stddev / $mean" | bc)

    echo "$cv"
}

# Function to benchmark a server
benchmark_server() {
    local name=$1
    local port=$2
    local endpoint=$3
    local payload=$4

    echo -e "${BLUE}=== Benchmarking: $name ===${NC}"

    # Check server availability
    if ! curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
        if ! curl -s "http://localhost:$port/api/tags" > /dev/null 2>&1; then
            echo -e "${YELLOW}Server not available on port $port, skipping${NC}"
            return 1
        fi
    fi

    local -a latencies=()
    local sample=0
    local cv="1.0"

    while [[ $sample -lt $MAX_SAMPLES ]]; do
        sample=$((sample + 1))

        local start end latency_ms
        start=$(date +%s%N)

        local resp
        resp=$(curl -s -X POST "http://localhost:$port$endpoint" \
            -H "Content-Type: application/json" \
            -d "$payload" 2>/dev/null || echo "{}")

        end=$(date +%s%N)
        latency_ms=$(( (end - start) / 1000000 ))
        latencies+=("$latency_ms")

        # Extract tokens from response
        local tokens
        if echo "$resp" | jq -e '.eval_count' > /dev/null 2>&1; then
            tokens=$(echo "$resp" | jq -r '.eval_count')
        elif echo "$resp" | jq -e '.tokens_predicted' > /dev/null 2>&1; then
            tokens=$(echo "$resp" | jq -r '.tokens_predicted')
        else
            tokens="30"
        fi

        # Calculate CV after minimum samples
        if [[ $sample -ge $MIN_SAMPLES ]]; then
            cv=$(calculate_cv "${latencies[@]}")

            # Check if CV converged
            if (( $(echo "$cv < $CV_THRESHOLD" | bc -l) )); then
                printf "  [%2d/%d] Latency: %dms | Tokens: %s | CV: %.3f ${GREEN}(converged)${NC}\n" \
                    "$sample" "$MAX_SAMPLES" "$latency_ms" "$tokens" "$cv"
                break
            fi
        fi

        printf "  [%2d/%d] Latency: %dms | Tokens: %s | CV: %.3f\n" \
            "$sample" "$MAX_SAMPLES" "$latency_ms" "$tokens" "$cv"
    done

    # Calculate final statistics
    local sum=0 min=999999 max=0
    for l in "${latencies[@]}"; do
        sum=$((sum + l))
        [[ $l -lt $min ]] && min=$l
        [[ $l -gt $max ]] && max=$l
    done
    local mean=$((sum / ${#latencies[@]}))

    # Sort for percentiles
    IFS=$'\n' sorted=($(sort -n <<<"${latencies[*]}")); unset IFS
    local n=${#sorted[@]}
    local p50_idx=$((n / 2))
    local p99_idx=$((n * 99 / 100))
    [[ $p99_idx -ge $n ]] && p99_idx=$((n - 1))

    local p50=${sorted[$p50_idx]}
    local p99=${sorted[$p99_idx]}

    # Calculate throughput (tokens/sec)
    local tps
    if [[ $mean -gt 0 ]]; then
        tps=$(echo "scale=1; 30 * 1000 / $mean" | bc)
    else
        tps="0"
    fi

    echo ""
    echo -e "${GREEN}Results for $name:${NC}"
    echo "  Samples: ${#latencies[@]}"
    echo "  p50 Latency: ${p50}ms"
    echo "  p99 Latency: ${p99}ms"
    echo "  Mean Latency: ${mean}ms"
    echo "  Throughput: ${tps} tok/s"
    echo "  Final CV: $cv"
    echo ""

    # Save results
    cat >> "$RESULTS_DIR/benchmark_gpu_matrix_${TIMESTAMP}.json" << EOF
{
  "runtime": "$name",
  "timestamp": "$(date -Iseconds)",
  "samples": ${#latencies[@]},
  "p50_ms": $p50,
  "p99_ms": $p99,
  "mean_ms": $mean,
  "throughput_tps": $tps,
  "cv": $cv
}
EOF
}

# Project root for realizar
REALIZAR_ROOT="${REALIZAR_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

# Start llama.cpp GPU server if not running
if ! curl -s http://localhost:8082/health > /dev/null 2>&1; then
    if [[ -x "$LLAMA_CPP_PATH" ]] && [[ -f "$MODEL_DIR/${MODELS[0]}" ]]; then
        echo "Starting llama.cpp GPU server..."
        "$LLAMA_CPP_PATH" -m "$MODEL_DIR/${MODELS[0]}" --host 127.0.0.1 --port 8082 -ngl 99 &
        LLAMA_PID=$!
        sleep 8
    fi
fi

# Start realizar GPU server if not running (IMP-093)
REALIZAR_PORT=9999
REALIZAR_PID=""
if ! curl -s http://localhost:$REALIZAR_PORT/health > /dev/null 2>&1; then
    if [[ -f "$MODEL_DIR/${MODELS[1]}" ]]; then
        echo "Starting realizar GPU server..."
        cargo run --release --features gpu --bin realizar --manifest-path "$REALIZAR_ROOT/Cargo.toml" \
            -- serve --model "$MODEL_DIR/${MODELS[1]}" --port $REALIZAR_PORT > /tmp/realizar_bench.log 2>&1 &
        REALIZAR_PID=$!
        echo "Waiting for realizar to load model (this may take 30-60 seconds)..."
        for i in {1..60}; do
            if curl -s http://localhost:$REALIZAR_PORT/health > /dev/null 2>&1; then
                echo "Realizar ready after ${i}s"
                break
            fi
            sleep 1
        done
    fi
fi

# Initialize results file
echo "[" > "$RESULTS_DIR/benchmark_gpu_matrix_${TIMESTAMP}.json"

# Benchmark realizar GPU (IMP-093)
if curl -s http://localhost:$REALIZAR_PORT/health > /dev/null 2>&1; then
    benchmark_server "realizar_gpu" "$REALIZAR_PORT" "/generate" \
        '{"prompt": "Hello, world!", "max_tokens": 30}'
    echo "," >> "$RESULTS_DIR/benchmark_gpu_matrix_${TIMESTAMP}.json"
fi

# Benchmark llama.cpp GPU
benchmark_server "llama_cpp_gpu" "8082" "/completion" \
    '{"prompt": "Hello, world!", "n_predict": 30, "temperature": 0}'

echo "," >> "$RESULTS_DIR/benchmark_gpu_matrix_${TIMESTAMP}.json"

# Benchmark Ollama
benchmark_server "ollama_gpu" "11434" "/api/generate" \
    '{"model": "phi2:2.7b", "prompt": "Hello, world!", "options": {"num_predict": 30, "temperature": 0}, "stream": false}'

echo "]" >> "$RESULTS_DIR/benchmark_gpu_matrix_${TIMESTAMP}.json"

# Cleanup
if [[ -n "${LLAMA_PID:-}" ]]; then
    kill "$LLAMA_PID" 2>/dev/null || true
fi
if [[ -n "${REALIZAR_PID:-}" ]]; then
    kill "$REALIZAR_PID" 2>/dev/null || true
fi

echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          Benchmark Complete                                     ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Results saved to: $RESULTS_DIR/benchmark_gpu_matrix_${TIMESTAMP}.json"
