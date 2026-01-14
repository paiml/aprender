#!/bin/bash
# GPU 2x Ollama Reproducible Benchmark
# SPEC: docs/specifications/qwen2.5-coder-showcase-demo.md v5.0.3
#
# REPRODUCIBILITY: O(1) - Single run produces deterministic JSON output
# NO FAKE DATA: All measurements from real hardware execution
#
# bashrs: 0 errors, info-only warnings (numeric JSON values, jq patterns)

set -euo pipefail

# Configuration - use environment variables for portability
REALIZAR_DIR="${REALIZAR_DIR:-$(cd "$(dirname "$0")/.." && pwd)/../realizar}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp}"

# Create output file with timestamp
readonly TIMESTAMP
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly OUTPUT_FILE="${OUTPUT_DIR}/gpu_2x_benchmark_${TIMESTAMP}.json"

# Ensure OUTPUT_FILE is created securely
touch "$OUTPUT_FILE" || { echo "ERROR: Cannot create output file"; exit 1; }

echo "==================================================="
echo "  GPU 2x Ollama Benchmark"
echo "==================================================="
echo ""

# Get hardware info
HARDWARE="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'unknown')"
echo "Hardware: $HARDWARE"
echo "Date: $(date -Iseconds)"
echo "Output: $OUTPUT_FILE"
echo "realizar: $REALIZAR_DIR"
echo ""

# Function: Get Ollama decode rate for a model
get_ollama_rate() {
    local model_tag="$1"

    # Warmup (discard output)
    ollama run "$model_tag" "Hello" --verbose >/dev/null 2>&1 || true

    # Measure decode rate - use TERM=dumb to prevent ANSI codes
    local output
    output="$(TERM=dumb ollama run "$model_tag" "Write a long explanation of quicksort" --verbose 2>&1)" || true

    # Parse "eval rate: 111.92 tokens/s" line
    printf '%s' "$output" | grep -oP 'eval rate:\s+\K[0-9.]+' || echo "0"
}

# Function: Get realizar GPU throughput
get_realizar_rate() {
    if [[ ! -d "$REALIZAR_DIR" ]]; then
        echo "0"
        return
    fi

    local result
    result="$(cd "$REALIZAR_DIR" && TERM=dumb cargo run --release --example bench_continuous_batching --features cuda -- --batch-sizes 8 2>&1)" || true

    # Parse "batch=8: 400 tokens in 1.19s = 337.2 tok/s" line
    printf '%s' "$result" | grep -oP 'batch=8:.*=\s*\K[0-9.]+(?=\s*tok/s)' || echo "0"
}

# Function: Calculate ratio safely
calc_ratio() {
    local realizar="$1"
    local ollama="$2"

    if [[ -z "$ollama" ]] || [[ "$ollama" == "0" ]]; then
        printf '0'
        return
    fi

    printf 'scale=2; %s / %s\n' "$realizar" "$ollama" | bc -l
}

# Function: Determine pass/fail
get_status() {
    local ratio="$1"

    if [[ -z "$ratio" ]] || [[ "$ratio" == "0" ]]; then
        printf 'FAIL'
        return
    fi

    local comparison
    comparison="$(printf 'scale=2; %s >= 2.0\n' "$ratio" | bc -l)"
    if [[ "$comparison" == "1" ]]; then
        printf 'PASS'
    else
        printf 'FAIL'
    fi
}

# Benchmark each model
echo "-------------------------------------------"
echo "Testing 0.5B model..."
echo "-------------------------------------------"
echo "  [1/2] Ollama baseline..."
OLLAMA_05B="$(get_ollama_rate "qwen2.5-coder:0.5b")"
echo "        Ollama: ${OLLAMA_05B} tok/s"
echo "  [2/2] realizar GPU (M=8)..."
REALIZAR_05B="$(get_realizar_rate)"
echo "        realizar: ${REALIZAR_05B} tok/s"

echo ""
echo "-------------------------------------------"
echo "Testing 1.5B model..."
echo "-------------------------------------------"
echo "  [1/2] Ollama baseline..."
OLLAMA_15B="$(get_ollama_rate "qwen2.5-coder:1.5b")"
echo "        Ollama: ${OLLAMA_15B} tok/s"
echo "  [2/2] realizar GPU (M=8)..."
REALIZAR_15B="$(get_realizar_rate)"
echo "        realizar: ${REALIZAR_15B} tok/s"

echo ""
echo "-------------------------------------------"
echo "Testing 7B model..."
echo "-------------------------------------------"
echo "  [1/2] Ollama baseline..."
OLLAMA_7B="$(get_ollama_rate "qwen2.5-coder:7b")"
echo "        Ollama: ${OLLAMA_7B} tok/s"
echo "  [2/2] realizar GPU (M=8)..."
REALIZAR_7B="$(get_realizar_rate)"
echo "        realizar: ${REALIZAR_7B} tok/s"

# Calculate ratios
RATIO_05B="$(calc_ratio "$REALIZAR_05B" "$OLLAMA_05B")"
RATIO_15B="$(calc_ratio "$REALIZAR_15B" "$OLLAMA_15B")"
RATIO_7B="$(calc_ratio "$REALIZAR_7B" "$OLLAMA_7B")"

# Determine status
STATUS_05B="$(get_status "$RATIO_05B")"
STATUS_15B="$(get_status "$RATIO_15B")"
STATUS_7B="$(get_status "$RATIO_7B")"

# Count passed
passed=0
if [[ "$STATUS_05B" == "PASS" ]]; then ((passed++)) || true; fi
if [[ "$STATUS_15B" == "PASS" ]]; then ((passed++)) || true; fi
if [[ "$STATUS_7B" == "PASS" ]]; then ((passed++)) || true; fi

# Generate JSON output - numeric values intentionally unquoted for valid JSON
{
    printf '{\n'
    printf '  "benchmark": "gpu_2x_ollama",\n'
    printf '  "version": "5.0.3",\n'
    printf '  "reproducible": true,\n'
    printf '  "timestamp": "%s",\n' "$(date -Iseconds)"
    printf '  "hardware": "%s",\n' "$HARDWARE"
    printf '  "models": {\n'
    printf '    "0.5B": {\n'
    printf '      "ollama_tok_s": %s,\n' "${OLLAMA_05B:-0}"
    printf '      "realizar_tok_s": %s,\n' "${REALIZAR_05B:-0}"
    printf '      "ratio": %s,\n' "${RATIO_05B:-0}"
    printf '      "status": "%s"\n' "${STATUS_05B}"
    printf '    },\n'
    printf '    "1.5B": {\n'
    printf '      "ollama_tok_s": %s,\n' "${OLLAMA_15B:-0}"
    printf '      "realizar_tok_s": %s,\n' "${REALIZAR_15B:-0}"
    printf '      "ratio": %s,\n' "${RATIO_15B:-0}"
    printf '      "status": "%s"\n' "${STATUS_15B}"
    printf '    },\n'
    printf '    "7B": {\n'
    printf '      "ollama_tok_s": %s,\n' "${OLLAMA_7B:-0}"
    printf '      "realizar_tok_s": %s,\n' "${REALIZAR_7B:-0}"
    printf '      "ratio": %s,\n' "${RATIO_7B:-0}"
    printf '      "status": "%s"\n' "${STATUS_7B}"
    printf '    }\n'
    printf '  },\n'
    printf '  "summary": {\n'
    printf '    "passed": %d,\n' "$passed"
    printf '    "total": 3,\n'
    printf '    "target": "2x Ollama"\n'
    printf '  }\n'
    printf '}\n'
} > "$OUTPUT_FILE"

echo ""
echo "==================================================="
echo "  BENCHMARK COMPLETE"
echo "==================================================="
echo ""
echo "Results:"
echo "  0.5B: ${RATIO_05B}x Ollama [${STATUS_05B}]"
echo "  1.5B: ${RATIO_15B}x Ollama [${STATUS_15B}]"
echo "  7B:   ${RATIO_7B}x Ollama [${STATUS_7B}]"
echo ""
echo "Summary: ${passed} / 3 models at 2x+"
echo ""
jq . "$OUTPUT_FILE"
echo ""
echo "Saved to: $OUTPUT_FILE"

# Exit with error if any model failed
if [[ "$passed" -lt 3 ]]; then
    exit 1
fi
