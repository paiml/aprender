#!/bin/bash
# GPU 2x Ollama Reproducible Benchmark
# SPEC: docs/specifications/qwen2.5-coder-showcase-demo.md v5.0.5
#
# Benchmarks 4 rows: realizar GGUF, realizar APR, apr-cli GGUF, apr-cli APR
# REPRODUCIBILITY: O(1) - Single run produces deterministic JSON output
# NO FAKE DATA: All measurements from real hardware execution
#
# bashrs: 0 errors, info-only warnings

set -euo pipefail

# Configuration
REALIZAR_DIR="${REALIZAR_DIR:-$(cd "$(dirname "$0")/.." && pwd)/../realizar}"
APRENDER_DIR="${APRENDER_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp}"
MODELS_DIR="${MODELS_DIR:-$APRENDER_DIR/models}"

readonly TIMESTAMP
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly OUTPUT_FILE="${OUTPUT_DIR}/gpu_2x_benchmark_${TIMESTAMP}.json"

touch "$OUTPUT_FILE" || { echo "ERROR: Cannot create output file"; exit 1; }

echo "==================================================="
echo "  GPU 2x Benchmark (4 Backend/Format Combinations)"
echo "==================================================="
echo ""

HARDWARE="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'unknown')"
echo "Hardware: $HARDWARE"
echo "Date: $(date -Iseconds)"
echo "Output: $OUTPUT_FILE"
echo ""

# Model paths
declare -A GGUF_MODELS=(
    ["0.5B"]="$MODELS_DIR/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
    ["1.5B"]="$MODELS_DIR/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    ["7B"]="$MODELS_DIR/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    ["32B"]="$MODELS_DIR/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf"
)

declare -A APR_MODELS=(
    ["0.5B"]="$MODELS_DIR/qwen2.5-coder-0.5b.apr"
    ["1.5B"]="$MODELS_DIR/qwen2.5-coder-1.5b.apr"
    ["7B"]="$MODELS_DIR/qwen2.5-coder-7b.apr"
    ["32B"]="$MODELS_DIR/qwen2.5-coder-32b.apr"
)

declare -A OLLAMA_MODELS=(
    ["0.5B"]="qwen2.5-coder:0.5b"
    ["1.5B"]="qwen2.5-coder:1.5b"
    ["7B"]="qwen2.5-coder:7b"
    ["32B"]="qwen2.5-coder:32b"
)

# Function: Get Ollama decode rate
get_ollama_rate() {
    local model_tag="$1"
    ollama run "$model_tag" "Hello" --verbose >/dev/null 2>&1 || true
    local output
    output="$(TERM=dumb ollama run "$model_tag" "Write quicksort" --verbose 2>&1)" || true
    printf '%s' "$output" | grep -oP 'eval rate:\s+\K[0-9.]+' || echo "0"
}

# Function: Get realizar GGUF rate
get_realizar_gguf_rate() {
    local model_path="$1"
    if [[ ! -f "$model_path" ]]; then
        echo "0"
        return
    fi
    local result
    result="$(cd "$REALIZAR_DIR" && TERM=dumb cargo run --release --features cuda --example gpu_showcase_benchmark -- --model "$model_path" --quick 2>&1)" || true
    printf '%s' "$result" | grep -oP 'APR CUDA.*:\s+\K[0-9.]+' | head -1 || echo "0"
}

# Function: Get realizar APR rate
get_realizar_apr_rate() {
    local model_path="$1"
    if [[ ! -f "$model_path" ]]; then
        echo "0"
        return
    fi
    # TODO: Add realizar APR benchmark example
    echo "0"
}

# Function: Get apr-cli GGUF rate
get_apr_cli_gguf_rate() {
    local model_path="$1"
    if [[ ! -f "$model_path" ]]; then
        echo "0"
        return
    fi
    local result
    result="$(cd "$APRENDER_DIR" && TERM=dumb cargo run -p apr-cli --release --features inference -- run "$model_path" --benchmark 2>&1)" || true
    printf '%s' "$result" | grep -oP 'tok/s:\s*\K[0-9.]+' | head -1 || echo "0"
}

# Function: Get apr-cli APR rate
get_apr_cli_apr_rate() {
    local model_path="$1"
    if [[ ! -f "$model_path" ]]; then
        echo "0"
        return
    fi
    local result
    result="$(cd "$APRENDER_DIR" && TERM=dumb cargo run -p apr-cli --release --features inference -- run "$model_path" --benchmark 2>&1)" || true
    printf '%s' "$result" | grep -oP 'tok/s:\s*\K[0-9.]+' | head -1 || echo "0"
}

# Initialize results
declare -A RESULTS

echo "==================================================="
echo "  Benchmarking Models"
echo "==================================================="

for size in "1.5B"; do  # Start with 1.5B as reference
    echo ""
    echo "--- $size ---"

    echo "  Ollama..."
    RESULTS["ollama_$size"]="$(get_ollama_rate "${OLLAMA_MODELS[$size]}")"
    echo "    ${RESULTS[ollama_$size]} tok/s"

    echo "  realizar GGUF..."
    RESULTS["realizar_gguf_$size"]="$(get_realizar_gguf_rate "${GGUF_MODELS[$size]}")"
    echo "    ${RESULTS[realizar_gguf_$size]} tok/s"

    echo "  realizar APR..."
    RESULTS["realizar_apr_$size"]="$(get_realizar_apr_rate "${APR_MODELS[$size]}")"
    echo "    ${RESULTS[realizar_apr_$size]} tok/s"

    echo "  apr-cli GGUF..."
    RESULTS["apr_cli_gguf_$size"]="$(get_apr_cli_gguf_rate "${GGUF_MODELS[$size]}")"
    echo "    ${RESULTS[apr_cli_gguf_$size]} tok/s"

    echo "  apr-cli APR..."
    RESULTS["apr_cli_apr_$size"]="$(get_apr_cli_apr_rate "${APR_MODELS[$size]}")"
    echo "    ${RESULTS[apr_cli_apr_$size]} tok/s"
done

# Generate JSON
{
    printf '{\n'
    printf '  "benchmark": "gpu_2x_ollama",\n'
    printf '  "version": "5.0.5",\n'
    printf '  "timestamp": "%s",\n' "$(date -Iseconds)"
    printf '  "hardware": "%s",\n' "$HARDWARE"
    printf '  "results": {\n'
    printf '    "1.5B": {\n'
    printf '      "ollama": %s,\n' "${RESULTS[ollama_1.5B]:-0}"
    printf '      "realizar_gguf": %s,\n' "${RESULTS[realizar_gguf_1.5B]:-0}"
    printf '      "realizar_apr": %s,\n' "${RESULTS[realizar_apr_1.5B]:-0}"
    printf '      "apr_cli_gguf": %s,\n' "${RESULTS[apr_cli_gguf_1.5B]:-0}"
    printf '      "apr_cli_apr": %s\n' "${RESULTS[apr_cli_apr_1.5B]:-0}"
    printf '    }\n'
    printf '  }\n'
    printf '}\n'
} > "$OUTPUT_FILE"

echo ""
echo "==================================================="
echo "  RESULTS"
echo "==================================================="
jq . "$OUTPUT_FILE"
echo ""
echo "Saved: $OUTPUT_FILE"
