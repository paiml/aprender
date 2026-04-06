#!/usr/bin/env bash
#
# Reproducible Comparative Benchmark Script
# =========================================
#
# Implements scientific benchmarking methodology per:
# - Hoefler & Belli, "Scientific Benchmarking of Parallel Computing Systems", SC'15
# - Dean & Barroso, "The Tail at Scale", CACM 2013
#
# Usage:
#   ./scripts/bench-compare.sh [--llama-cpp] [--ollama] [--all]
#
# Prerequisites:
#   - llama.cpp server: llama-server -m model.gguf --port 8082 -ngl 99
#   - Ollama: ollama serve (port 11434)
#

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/benches/comparative/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Benchmark parameters (Hoefler & Belli methodology)
MIN_SAMPLES=30          # Minimum samples before CV check
MAX_SAMPLES=200         # Failsafe maximum
CV_THRESHOLD=0.05       # 5% coefficient of variation target
WARMUP_ITERATIONS=5     # Warmup to stabilize caches

# Model configuration
MODEL_NAME="phi-2"
PROMPT="Explain the concept of machine learning in one sentence."
MAX_TOKENS=50
TEMPERATURE=0.7

# Server endpoints
LLAMA_CPP_URL="http://localhost:8082"
OLLAMA_URL="http://localhost:11434"
REALIZAR_URL="http://localhost:8080"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[FAIL]${NC} $*"; }

# Calculate statistics from array of values
calc_stats() {
    local values=("$@")
    local n=${#values[@]}

    if [[ $n -eq 0 ]]; then
        echo "0 0 0 0 0"
        return
    fi

    # Calculate mean
    local sum=0
    for v in "${values[@]}"; do
        sum=$(echo "$sum + $v" | bc -l)
    done
    local mean=$(echo "scale=4; $sum / $n" | bc -l)

    # Calculate std_dev
    local sq_sum=0
    for v in "${values[@]}"; do
        local diff=$(echo "$v - $mean" | bc -l)
        sq_sum=$(echo "$sq_sum + ($diff * $diff)" | bc -l)
    done
    local variance=$(echo "scale=4; $sq_sum / ($n - 1)" | bc -l 2>/dev/null || echo "0")
    local std_dev=$(echo "scale=4; sqrt($variance)" | bc -l 2>/dev/null || echo "0")

    # Calculate CV
    local cv=0
    if [[ $(echo "$mean > 0.0001" | bc -l) -eq 1 ]]; then
        cv=$(echo "scale=4; $std_dev / $mean" | bc -l)
    fi

    # Sort for percentiles
    local sorted=($(printf '%s\n' "${values[@]}" | sort -n))
    local p50_idx=$((n / 2))
    local p99_idx=$((n * 99 / 100))
    [[ $p99_idx -ge $n ]] && p99_idx=$((n - 1))

    local p50=${sorted[$p50_idx]}
    local p99=${sorted[$p99_idx]}

    echo "$mean $std_dev $cv $p50 $p99"
}

# Check if server is healthy
check_server() {
    local url=$1
    local name=$2
    local health_endpoint=$3

    if curl -s --max-time 2 "${url}${health_endpoint}" > /dev/null 2>&1; then
        log_success "$name server is healthy at $url"
        return 0
    else
        log_warn "$name server not available at $url"
        return 1
    fi
}

# ============================================================================
# Benchmark Functions
# ============================================================================

# Benchmark llama.cpp server
bench_llama_cpp() {
    local output_file="${OUTPUT_DIR}/llama_cpp_${TIMESTAMP}.json"
    log_info "Benchmarking llama.cpp at ${LLAMA_CPP_URL}..."

    if ! check_server "$LLAMA_CPP_URL" "llama.cpp" "/health"; then
        return 1
    fi

    # Warmup
    log_info "Warmup phase ($WARMUP_ITERATIONS iterations)..."
    for ((i=1; i<=WARMUP_ITERATIONS; i++)); do
        curl -s -X POST "${LLAMA_CPP_URL}/completion" \
            -H "Content-Type: application/json" \
            -d "{\"prompt\": \"$PROMPT\", \"n_predict\": $MAX_TOKENS, \"temperature\": $TEMPERATURE, \"stream\": false}" \
            > /dev/null
    done

    # Measurement phase with CV-based stopping
    log_info "Measurement phase (CV threshold: ${CV_THRESHOLD})..."
    local latencies=()
    local tokens_per_sec=()
    local iteration=0
    local cv=1.0

    while [[ $iteration -lt $MAX_SAMPLES ]]; do
        iteration=$((iteration + 1))

        local start_ns=$(date +%s%N)
        local response=$(curl -s -X POST "${LLAMA_CPP_URL}/completion" \
            -H "Content-Type: application/json" \
            -d "{\"prompt\": \"$PROMPT\", \"n_predict\": $MAX_TOKENS, \"temperature\": $TEMPERATURE, \"stream\": false}")
        local end_ns=$(date +%s%N)

        local latency_ms=$(echo "scale=3; ($end_ns - $start_ns) / 1000000" | bc -l)
        local tokens=$(echo "$response" | jq -r '.tokens_predicted // 0')
        local tps=0
        if [[ $tokens -gt 0 ]]; then
            tps=$(echo "scale=2; $tokens / ($latency_ms / 1000)" | bc -l)
        fi

        latencies+=("$latency_ms")
        tokens_per_sec+=("$tps")

        # Check CV after minimum samples
        if [[ $iteration -ge $MIN_SAMPLES ]]; then
            local stats=($(calc_stats "${latencies[@]}"))
            cv=${stats[2]}

            if [[ $(echo "$cv < $CV_THRESHOLD" | bc -l) -eq 1 ]]; then
                log_success "CV stable at ${cv} after ${iteration} iterations"
                break
            fi
        fi

        printf "\r  [%3d/%d] Latency: %6.1fms | Tokens: %3d | TPS: %5.1f | CV: %.3f" \
            "$iteration" "$MAX_SAMPLES" "$latency_ms" "$tokens" "$tps" "$cv"
    done
    echo ""

    # Calculate final statistics
    local lat_stats=($(calc_stats "${latencies[@]}"))
    local tps_stats=($(calc_stats "${tokens_per_sec[@]}"))

    # Output JSON results
    cat > "$output_file" << EOF
{
  "benchmark": "llama.cpp",
  "timestamp": "$(date -Iseconds)",
  "config": {
    "url": "${LLAMA_CPP_URL}",
    "model": "${MODEL_NAME}",
    "prompt": "${PROMPT}",
    "max_tokens": ${MAX_TOKENS},
    "temperature": ${TEMPERATURE}
  },
  "methodology": {
    "type": "CV-based stopping (Hoefler & Belli SC'15)",
    "min_samples": ${MIN_SAMPLES},
    "max_samples": ${MAX_SAMPLES},
    "cv_threshold": ${CV_THRESHOLD},
    "warmup_iterations": ${WARMUP_ITERATIONS}
  },
  "results": {
    "iterations": ${iteration},
    "final_cv": ${cv},
    "latency_ms": {
      "mean": ${lat_stats[0]},
      "std_dev": ${lat_stats[1]},
      "p50": ${lat_stats[3]},
      "p99": ${lat_stats[4]}
    },
    "tokens_per_sec": {
      "mean": ${tps_stats[0]},
      "std_dev": ${tps_stats[1]},
      "p50": ${tps_stats[3]},
      "p99": ${tps_stats[4]}
    }
  },
  "hardware": {
    "cpu": "$(lscpu 2>/dev/null | grep 'Model name' | cut -d: -f2 | xargs || echo 'unknown')",
    "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'none')"
  }
}
EOF

    log_success "Results saved to: $output_file"

    # Print summary
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                    llama.cpp Benchmark Results                 ║"
    echo "╠════════════════════════════════════════════════════════════════╣"
    printf "║  Iterations: %-48d ║\n" "$iteration"
    printf "║  Final CV: %-50.4f ║\n" "$cv"
    printf "║  Latency (p50): %-43.2f ms ║\n" "${lat_stats[3]}"
    printf "║  Latency (p99): %-43.2f ms ║\n" "${lat_stats[4]}"
    printf "║  Throughput (mean): %-39.1f tok/s ║\n" "${tps_stats[0]}"
    echo "╚════════════════════════════════════════════════════════════════╝"
}

# Benchmark Ollama server
bench_ollama() {
    local output_file="${OUTPUT_DIR}/ollama_${TIMESTAMP}.json"
    log_info "Benchmarking Ollama at ${OLLAMA_URL}..."

    if ! check_server "$OLLAMA_URL" "Ollama" "/api/tags"; then
        return 1
    fi

    # Warmup
    log_info "Warmup phase ($WARMUP_ITERATIONS iterations)..."
    for ((i=1; i<=WARMUP_ITERATIONS; i++)); do
        curl -s -X POST "${OLLAMA_URL}/api/generate" \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"phi\", \"prompt\": \"$PROMPT\", \"stream\": false, \"options\": {\"num_predict\": $MAX_TOKENS, \"temperature\": $TEMPERATURE}}" \
            > /dev/null
    done

    # Measurement phase
    log_info "Measurement phase (CV threshold: ${CV_THRESHOLD})..."
    local latencies=()
    local tokens_per_sec=()
    local iteration=0
    local cv=1.0

    while [[ $iteration -lt $MAX_SAMPLES ]]; do
        iteration=$((iteration + 1))

        local start_ns=$(date +%s%N)
        local response=$(curl -s -X POST "${OLLAMA_URL}/api/generate" \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"phi\", \"prompt\": \"$PROMPT\", \"stream\": false, \"options\": {\"num_predict\": $MAX_TOKENS, \"temperature\": $TEMPERATURE}}")
        local end_ns=$(date +%s%N)

        local latency_ms=$(echo "scale=3; ($end_ns - $start_ns) / 1000000" | bc -l)
        local tokens=$(echo "$response" | jq -r '.eval_count // 0')
        local tps=0
        if [[ $tokens -gt 0 ]]; then
            tps=$(echo "scale=2; $tokens / ($latency_ms / 1000)" | bc -l)
        fi

        latencies+=("$latency_ms")
        tokens_per_sec+=("$tps")

        # Check CV after minimum samples
        if [[ $iteration -ge $MIN_SAMPLES ]]; then
            local stats=($(calc_stats "${latencies[@]}"))
            cv=${stats[2]}

            if [[ $(echo "$cv < $CV_THRESHOLD" | bc -l) -eq 1 ]]; then
                log_success "CV stable at ${cv} after ${iteration} iterations"
                break
            fi
        fi

        printf "\r  [%3d/%d] Latency: %6.1fms | Tokens: %3d | TPS: %5.1f | CV: %.3f" \
            "$iteration" "$MAX_SAMPLES" "$latency_ms" "$tokens" "$tps" "$cv"
    done
    echo ""

    # Calculate final statistics
    local lat_stats=($(calc_stats "${latencies[@]}"))
    local tps_stats=($(calc_stats "${tokens_per_sec[@]}"))

    # Output JSON results
    cat > "$output_file" << EOF
{
  "benchmark": "ollama",
  "timestamp": "$(date -Iseconds)",
  "config": {
    "url": "${OLLAMA_URL}",
    "model": "phi",
    "prompt": "${PROMPT}",
    "max_tokens": ${MAX_TOKENS},
    "temperature": ${TEMPERATURE}
  },
  "methodology": {
    "type": "CV-based stopping (Hoefler & Belli SC'15)",
    "min_samples": ${MIN_SAMPLES},
    "max_samples": ${MAX_SAMPLES},
    "cv_threshold": ${CV_THRESHOLD},
    "warmup_iterations": ${WARMUP_ITERATIONS}
  },
  "results": {
    "iterations": ${iteration},
    "final_cv": ${cv},
    "latency_ms": {
      "mean": ${lat_stats[0]},
      "std_dev": ${lat_stats[1]},
      "p50": ${lat_stats[3]},
      "p99": ${lat_stats[4]}
    },
    "tokens_per_sec": {
      "mean": ${tps_stats[0]},
      "std_dev": ${tps_stats[1]},
      "p50": ${tps_stats[3]},
      "p99": ${tps_stats[4]}
    }
  },
  "hardware": {
    "cpu": "$(lscpu 2>/dev/null | grep 'Model name' | cut -d: -f2 | xargs || echo 'unknown')",
    "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'none')"
  }
}
EOF

    log_success "Results saved to: $output_file"

    # Print summary
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                    Ollama Benchmark Results                    ║"
    echo "╠════════════════════════════════════════════════════════════════╣"
    printf "║  Iterations: %-48d ║\n" "$iteration"
    printf "║  Final CV: %-50.4f ║\n" "$cv"
    printf "║  Latency (p50): %-43.2f ms ║\n" "${lat_stats[3]}"
    printf "║  Latency (p99): %-43.2f ms ║\n" "${lat_stats[4]}"
    printf "║  Throughput (mean): %-39.1f tok/s ║\n" "${tps_stats[0]}"
    echo "╚════════════════════════════════════════════════════════════════╝"
}

# ============================================================================
# Main
# ============================================================================

main() {
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║          Realizar Comparative Benchmark Suite v1.1             ║"
    echo "║                                                                ║"
    echo "║  Methodology: CV-based stopping (Hoefler & Belli, SC'15)       ║"
    echo "║  Statistical threshold: CV < ${CV_THRESHOLD} (${WARMUP_ITERATIONS} warmup iterations)              ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Parse arguments
    local run_llama_cpp=false
    local run_ollama=false

    if [[ $# -eq 0 ]] || [[ "$1" == "--all" ]]; then
        run_llama_cpp=true
        run_ollama=true
    else
        while [[ $# -gt 0 ]]; do
            case $1 in
                --llama-cpp) run_llama_cpp=true ;;
                --ollama) run_ollama=true ;;
                --all)
                    run_llama_cpp=true
                    run_ollama=true
                    ;;
                *)
                    log_error "Unknown option: $1"
                    echo "Usage: $0 [--llama-cpp] [--ollama] [--all]"
                    exit 1
                    ;;
            esac
            shift
        done
    fi

    # Run benchmarks
    local results=()

    if $run_llama_cpp; then
        echo ""
        if bench_llama_cpp; then
            results+=("llama.cpp: PASS")
        else
            results+=("llama.cpp: SKIP (server not available)")
        fi
    fi

    if $run_ollama; then
        echo ""
        if bench_ollama; then
            results+=("Ollama: PASS")
        else
            results+=("Ollama: SKIP (server not available)")
        fi
    fi

    # Summary
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                         Summary                                ║"
    echo "╠════════════════════════════════════════════════════════════════╣"
    for result in "${results[@]}"; do
        printf "║  %-60s ║\n" "$result"
    done
    echo "║                                                                ║"
    printf "║  Results directory: %-41s ║\n" "$OUTPUT_DIR"
    echo "╚════════════════════════════════════════════════════════════════╝"
}

main "$@"
