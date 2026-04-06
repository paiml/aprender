#!/usr/bin/env bash
#
# Backend Benchmark Matrix Script
# ================================
#
# Creates a reproducible benchmark matrix comparing:
# - Runtimes: realizar, llama.cpp, ollama
# - Backends: CPU, GPU (wgpu/CUDA)
#
# Methodology: CV-based stopping (Hoefler & Belli, SC'15)
#
# Usage:
#   ./scripts/bench-matrix.sh [OPTIONS]
#
# Options:
#   --quick     Quick run (5 iterations, no CV check)
#   --full      Full run (CV-based stopping)
#   --output    Output directory (default: benches/comparative/results)

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/benches/comparative/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Benchmark parameters (Hoefler & Belli methodology)
MIN_SAMPLES=5       # Minimum samples before CV check (quick mode)
MAX_SAMPLES=30      # Maximum samples
CV_THRESHOLD=0.10   # 10% CV target for quick mode
WARMUP_ITERATIONS=2 # Warmup iterations

# Model configuration
PROMPT="Explain the concept of machine learning in one sentence."
MAX_TOKENS=50
TEMPERATURE=0.7

# Server endpoints
LLAMA_CPP_CPU_PORT=8083
LLAMA_CPP_GPU_PORT=8082
OLLAMA_URL="http://localhost:11434"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[FAIL]${NC} $*"; }
log_header() { echo -e "${CYAN}$*${NC}"; }

# Calculate statistics from array
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
    local mean=$(echo "scale=6; $sum / $n" | bc -l)

    # Calculate std_dev
    local sq_sum=0
    for v in "${values[@]}"; do
        local diff=$(echo "$v - $mean" | bc -l)
        sq_sum=$(echo "$sq_sum + ($diff * $diff)" | bc -l)
    done
    local variance=$(echo "scale=6; $sq_sum / ($n - 1)" | bc -l 2>/dev/null || echo "0")
    local std_dev=$(echo "scale=6; sqrt($variance)" | bc -l 2>/dev/null || echo "0")

    # Calculate CV
    local cv=0
    if [[ $(echo "$mean > 0.0001" | bc -l) -eq 1 ]]; then
        cv=$(echo "scale=6; $std_dev / $mean" | bc -l)
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
        log_success "$name available"
        return 0
    else
        log_warn "$name not available"
        return 1
    fi
}

# ============================================================================
# Benchmark Functions
# ============================================================================

# Benchmark llama.cpp with specified GPU layers
bench_llama_cpp() {
    local backend=$1  # "cpu" or "gpu"
    local port=$2
    local ngl=$3
    local output_file="${OUTPUT_DIR}/llama_cpp_${backend}_${TIMESTAMP}.json"

    log_header "=== llama.cpp (${backend^^}) ==="
    log_info "GPU layers: $ngl"

    local url="http://localhost:${port}"

    if ! check_server "$url" "llama.cpp ($backend)" "/health"; then
        echo "{\"runtime\": \"llama-cpp\", \"backend\": \"$backend\", \"available\": false}" > "$output_file"
        return 1
    fi

    # Warmup
    log_info "Warmup ($WARMUP_ITERATIONS iterations)..."
    for ((i=1; i<=WARMUP_ITERATIONS; i++)); do
        curl -s -X POST "${url}/completion" \
            -H "Content-Type: application/json" \
            -d "{\"prompt\": \"$PROMPT\", \"n_predict\": $MAX_TOKENS, \"temperature\": $TEMPERATURE, \"stream\": false}" \
            > /dev/null
    done

    # Measurement
    log_info "Measuring (min=$MIN_SAMPLES, max=$MAX_SAMPLES, CV<$CV_THRESHOLD)..."
    local latencies=()
    local throughputs=()
    local cold_start=0
    local iteration=0

    while [[ $iteration -lt $MAX_SAMPLES ]]; do
        iteration=$((iteration + 1))

        local start_ns=$(date +%s%N)
        local response=$(curl -s -X POST "${url}/completion" \
            -H "Content-Type: application/json" \
            -d "{\"prompt\": \"$PROMPT\", \"n_predict\": $MAX_TOKENS, \"temperature\": $TEMPERATURE, \"stream\": false}")
        local end_ns=$(date +%s%N)

        local latency_ms=$(echo "scale=3; ($end_ns - $start_ns) / 1000000" | bc -l)
        local tokens=$(echo "$response" | jq -r '.tokens_predicted // 0')
        local tps=0
        if [[ $tokens -gt 0 ]]; then
            tps=$(echo "scale=2; $tokens / ($latency_ms / 1000)" | bc -l)
        fi

        # First iteration is cold start
        if [[ $iteration -eq 1 ]]; then
            cold_start=$latency_ms
        fi

        latencies+=("$latency_ms")
        throughputs+=("$tps")

        # Check CV after minimum samples
        if [[ $iteration -ge $MIN_SAMPLES ]]; then
            local stats=($(calc_stats "${latencies[@]}"))
            local cv=${stats[2]}

            if [[ $(echo "$cv < $CV_THRESHOLD" | bc -l) -eq 1 ]]; then
                log_success "CV stable at ${cv} after ${iteration} iterations"
                break
            fi
        fi

        printf "\r  [%3d/%d] Latency: %6.1fms | TPS: %5.1f" \
            "$iteration" "$MAX_SAMPLES" "$latency_ms" "$tps"
    done
    echo ""

    # Calculate final stats
    local lat_stats=($(calc_stats "${latencies[@]}"))
    local tps_stats=($(calc_stats "${throughputs[@]}"))

    # Output JSON
    cat > "$output_file" << EOF
{
  "runtime": "llama-cpp",
  "backend": "$backend",
  "available": true,
  "model": "phi-2-q4_k_m",
  "url": "$url",
  "iterations": $iteration,
  "latency_ms": {
    "mean": ${lat_stats[0]},
    "p50": ${lat_stats[3]},
    "p99": ${lat_stats[4]},
    "samples": [$(IFS=,; echo "${latencies[*]}")]
  },
  "throughput_tokens_per_sec": ${tps_stats[0]},
  "cold_start_ms": $cold_start,
  "cv_at_stop": ${lat_stats[2]},
  "ngl": $ngl
}
EOF

    log_success "Saved: $output_file"
    printf "  p50: %.1fms | p99: %.1fms | TPS: %.1f\n" "${lat_stats[3]}" "${lat_stats[4]}" "${tps_stats[0]}"
}

# Benchmark Ollama
bench_ollama() {
    local backend=$1  # Always "gpu" for ollama
    local output_file="${OUTPUT_DIR}/ollama_${backend}_${TIMESTAMP}.json"

    log_header "=== Ollama (${backend^^}) ==="

    if ! check_server "$OLLAMA_URL" "Ollama" "/api/tags"; then
        echo "{\"runtime\": \"ollama\", \"backend\": \"$backend\", \"available\": false}" > "$output_file"
        return 1
    fi

    # Warmup
    log_info "Warmup ($WARMUP_ITERATIONS iterations)..."
    for ((i=1; i<=WARMUP_ITERATIONS; i++)); do
        curl -s -X POST "${OLLAMA_URL}/api/generate" \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"phi\", \"prompt\": \"$PROMPT\", \"stream\": false, \"options\": {\"num_predict\": $MAX_TOKENS, \"temperature\": $TEMPERATURE}}" \
            > /dev/null 2>&1 || true
    done

    # Measurement
    log_info "Measuring (min=$MIN_SAMPLES, max=$MAX_SAMPLES, CV<$CV_THRESHOLD)..."
    local latencies=()
    local throughputs=()
    local cold_start=0
    local iteration=0

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

        # First iteration is cold start
        if [[ $iteration -eq 1 ]]; then
            cold_start=$latency_ms
        fi

        latencies+=("$latency_ms")
        throughputs+=("$tps")

        # Check CV after minimum samples
        if [[ $iteration -ge $MIN_SAMPLES ]]; then
            local stats=($(calc_stats "${latencies[@]}"))
            local cv=${stats[2]}

            if [[ $(echo "$cv < $CV_THRESHOLD" | bc -l) -eq 1 ]]; then
                log_success "CV stable at ${cv} after ${iteration} iterations"
                break
            fi
        fi

        printf "\r  [%3d/%d] Latency: %6.1fms | TPS: %5.1f" \
            "$iteration" "$MAX_SAMPLES" "$latency_ms" "$tps"
    done
    echo ""

    # Calculate final stats
    local lat_stats=($(calc_stats "${latencies[@]}"))
    local tps_stats=($(calc_stats "${throughputs[@]}"))

    # Output JSON
    cat > "$output_file" << EOF
{
  "runtime": "ollama",
  "backend": "$backend",
  "available": true,
  "model": "phi",
  "url": "$OLLAMA_URL",
  "iterations": $iteration,
  "latency_ms": {
    "mean": ${lat_stats[0]},
    "p50": ${lat_stats[3]},
    "p99": ${lat_stats[4]},
    "samples": [$(IFS=,; echo "${latencies[*]}")]
  },
  "throughput_tokens_per_sec": ${tps_stats[0]},
  "cold_start_ms": $cold_start,
  "cv_at_stop": ${lat_stats[2]}
}
EOF

    log_success "Saved: $output_file"
    printf "  p50: %.1fms | p99: %.1fms | TPS: %.1f\n" "${lat_stats[3]}" "${lat_stats[4]}" "${tps_stats[0]}"
}

# Generate matrix summary
generate_matrix_summary() {
    local output_file="${OUTPUT_DIR}/benchmark_matrix_${TIMESTAMP}.json"

    log_header "=== Generating Matrix Summary ==="

    # Collect all results
    local entries=""

    for file in "${OUTPUT_DIR}"/llama_cpp_*_${TIMESTAMP}.json "${OUTPUT_DIR}"/ollama_*_${TIMESTAMP}.json; do
        if [[ -f "$file" ]]; then
            if [[ -n "$entries" ]]; then
                entries="${entries},"
            fi
            entries="${entries}$(cat "$file")"
        fi
    done

    # Get hardware info
    local cpu_model=$(lscpu 2>/dev/null | grep 'Model name' | cut -d: -f2 | xargs || echo 'unknown')
    local gpu_model=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'none')

    cat > "$output_file" << EOF
{
  "version": "1.1",
  "timestamp": "$(date -Iseconds)",
  "methodology": "CV-based stopping (Hoefler & Belli SC'15)",
  "cv_threshold": $CV_THRESHOLD,
  "hardware": {
    "cpu": "$cpu_model",
    "gpu": "$gpu_model"
  },
  "entries": [$entries]
}
EOF

    log_success "Matrix saved: $output_file"

    # Generate markdown table
    local md_file="${OUTPUT_DIR}/benchmark_matrix_${TIMESTAMP}.md"
    cat > "$md_file" << 'EOF'
# Benchmark Matrix Results

**Methodology:** CV-based stopping (Hoefler & Belli SC'15)

| Runtime | Backend | p50 Latency | p99 Latency | Throughput | Cold Start |
|---------|---------|-------------|-------------|------------|------------|
EOF

    for file in "${OUTPUT_DIR}"/*_${TIMESTAMP}.json; do
        if [[ -f "$file" ]] && [[ "$file" != *"matrix"* ]]; then
            local runtime=$(jq -r '.runtime' "$file")
            local backend=$(jq -r '.backend' "$file")
            local available=$(jq -r '.available' "$file")

            if [[ "$available" == "true" ]]; then
                local p50=$(jq -r '.latency_ms.p50' "$file")
                local p99=$(jq -r '.latency_ms.p99' "$file")
                local tps=$(jq -r '.throughput_tokens_per_sec' "$file")
                local cold=$(jq -r '.cold_start_ms' "$file")
                printf "| **%s** | %s | %.1fms | %.1fms | %.1f tok/s | %.0fms |\n" \
                    "$runtime" "$backend" "$p50" "$p99" "$tps" "$cold" >> "$md_file"
            else
                printf "| %s | %s | - | - | - | - |\n" "$runtime" "$backend" >> "$md_file"
            fi
        fi
    done

    log_success "Markdown saved: $md_file"
    echo ""
    cat "$md_file"
}

# ============================================================================
# Main
# ============================================================================

main() {
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║          Realizar Benchmark Matrix v1.1                        ║"
    echo "║                                                                ║"
    echo "║  Methodology: CV-based stopping (Hoefler & Belli, SC'15)       ║"
    echo "║  Matrix: runtime × backend (CPU, GPU)                          ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick)
                MIN_SAMPLES=5
                MAX_SAMPLES=10
                CV_THRESHOLD=0.20
                ;;
            --full)
                MIN_SAMPLES=30
                MAX_SAMPLES=200
                CV_THRESHOLD=0.05
                ;;
            --output)
                OUTPUT_DIR="$2"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Usage: $0 [--quick|--full] [--output DIR]"
                exit 1
                ;;
        esac
        shift
    done

    log_info "Parameters: MIN=$MIN_SAMPLES, MAX=$MAX_SAMPLES, CV<$CV_THRESHOLD"
    echo ""

    # Run benchmarks

    # llama.cpp GPU (ngl=99)
    bench_llama_cpp "gpu" "$LLAMA_CPP_GPU_PORT" 99 || true
    echo ""

    # llama.cpp CPU (ngl=0)
    bench_llama_cpp "cpu" "$LLAMA_CPP_CPU_PORT" 0 || true
    echo ""

    # Ollama (GPU by default)
    bench_ollama "gpu" || true
    echo ""

    # Generate matrix summary
    generate_matrix_summary

    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                         Complete                               ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
}

main "$@"
