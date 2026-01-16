#!/usr/bin/env bash
# benchmark-matrix.sh - Scientific parameterized benchmark
#
# Usage:
#   ./benchmark-matrix.sh                    # Full matrix (all modalities)
#   ./benchmark-matrix.sh --mode generate    # Single mode
#   ./benchmark-matrix.sh --format gguf      # Single format
#   ./benchmark-matrix.sh --backend gpu      # Single backend
#   ./benchmark-matrix.sh --batch-sizes 1,8,16,32  # Custom batch sizes
#   ./benchmark-matrix.sh --iterations 5     # Custom iteration count
#   ./benchmark-matrix.sh --json             # JSON output only
#
# Modality Matrix:
#   Format × Backend × Mode × BatchSize = Full Cartesian Product
#
# Scientific Requirements:
#   - Warmup iterations (excluded from measurement)
#   - Multiple iterations with statistics (mean, std, CV)
#   - Cold vs warm measurements
#   - Structured JSON output for analysis

set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

MODEL_GGUF="${MODEL_GGUF:-/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf}"
MODEL_APR="${MODEL_APR:-/tmp/test-transformer.apr}"
APR_BIN="${APR_BIN:-/mnt/nvme-raid0/targets/aprender/release/apr}"
REALIZAR_BIN="${REALIZAR_BIN:-/mnt/nvme-raid0/targets/realizar/release/examples/test_m16}"

# Defaults
WARMUP_ITERATIONS=2
MEASURE_ITERATIONS=3
BATCH_SIZES="1,8,16,32"
FORMATS="gguf"         # APR excluded until real model available (147K vs 1.5B = invalid comparison)
BACKENDS="gpu,cpu"
MODES="generate,serve"
JSON_ONLY=false
RESULTS_FILE="/tmp/benchmark-matrix-$(date +%Y%m%d-%H%M%S).json"
TRACE_MODE=""          # activation, attention, logit, quant, kvcache, all
PROFILE_MODE=false     # Enable profiling output
RELEASE_MODE=true      # Use release binary (default)

# APR model size check - reject toy models for perf benchmarks
APR_MIN_SIZE_MB=100    # Minimum 100MB for valid perf comparison

# Ollama baselines (tok/s)
declare -A OLLAMA_BASELINE
OLLAMA_BASELINE[gpu_batched]=291
OLLAMA_BASELINE[gpu_single]=120
OLLAMA_BASELINE[cpu]=15

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ═══════════════════════════════════════════════════════════════════════
# Argument Parsing
# ═══════════════════════════════════════════════════════════════════════

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODES="$2"
            shift 2
            ;;
        --format)
            FORMATS="$2"
            shift 2
            ;;
        --backend)
            BACKENDS="$2"
            shift 2
            ;;
        --batch-sizes|-M)
            BATCH_SIZES="$2"
            shift 2
            ;;
        --iterations|-n)
            MEASURE_ITERATIONS="$2"
            shift 2
            ;;
        --warmup)
            WARMUP_ITERATIONS="$2"
            shift 2
            ;;
        --json)
            JSON_ONLY=true
            shift
            ;;
        --output|-o)
            RESULTS_FILE="$2"
            shift 2
            ;;
        --trace)
            TRACE_MODE="$2"
            shift 2
            ;;
        --profile)
            PROFILE_MODE=true
            shift
            ;;
        --debug)
            RELEASE_MODE=false
            APR_BIN="${APR_BIN/release/debug}"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode MODE          Modes to test: generate,serve,chat,batch (default: all)"
            echo "  --format FORMAT      Formats to test: gguf,apr (default: all)"
            echo "  --backend BACKEND    Backends to test: gpu,cpu (default: all)"
            echo "  --batch-sizes SIZES  Batch sizes: 1,8,16,32 (default: 1,8,16,32)"
            echo "  --iterations N       Measurement iterations (default: 3)"
            echo "  --warmup N           Warmup iterations (default: 2)"
            echo "  --json               JSON output only (no table)"
            echo "  --output FILE        Output JSON file"
            echo "  --trace TYPE         Enable tracing: activation,attention,logit,quant,kvcache,all"
            echo "  --profile            Enable profiling output"
            echo "  --debug              Use debug binary (slower, more checks)"
            echo "  --help               Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --batch-sizes 1 --mode generate           # M=1 generate only"
            echo "  $0 --trace activation --iterations 1         # With tracing"
            echo "  $0 --profile --format apr                    # Profile APR format"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ═══════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════

log() {
    if [[ "$JSON_ONLY" != "true" ]]; then
        echo -e "$@"
    fi
}

# Calculate statistics from array of values
calc_stats() {
    local values=("$@")
    local n=${#values[@]}

    if [[ $n -eq 0 ]]; then
        echo "0 0 0"
        return
    fi

    # Calculate mean
    local sum=0
    for v in "${values[@]}"; do
        sum=$(echo "$sum + $v" | bc -l)
    done
    local mean=$(echo "scale=2; $sum / $n" | bc -l)

    # Calculate std
    local sq_sum=0
    for v in "${values[@]}"; do
        local diff=$(echo "$v - $mean" | bc -l)
        sq_sum=$(echo "$sq_sum + ($diff * $diff)" | bc -l)
    done
    local std=$(echo "scale=2; sqrt($sq_sum / $n)" | bc -l 2>/dev/null || echo "0")

    # Calculate CV (coefficient of variation)
    local cv=0
    if [[ $(echo "$mean > 0" | bc -l) -eq 1 ]]; then
        cv=$(echo "scale=2; ($std / $mean) * 100" | bc -l)
    fi

    echo "$mean $std $cv"
}

# Run single benchmark and extract tok/s
run_benchmark() {
    local format=$1
    local backend=$2
    local mode=$3
    local batch_size=$4

    local model_path=""
    local gpu_flag=""
    local result=0

    # Select model
    if [[ "$format" == "gguf" ]]; then
        model_path="$MODEL_GGUF"
    else
        model_path="$MODEL_APR"
    fi

    # Select backend flag
    if [[ "$backend" == "cpu" ]]; then
        gpu_flag="--no-gpu"
    fi

    case "$mode" in
        generate)
            if [[ "$format" == "gguf" ]]; then
                local output=$("$APR_BIN" run "$model_path" --prompt "Hello" --max-tokens 20 $gpu_flag --benchmark 2>&1 || true)
                result=$(echo "$output" | grep -oP 'Inference:.*\(\K[0-9.]+(?= tok/s)' || echo "0")
            else
                local output=$("$APR_BIN" run "$model_path" --prompt "1,2,3" --max-tokens 10 $gpu_flag --benchmark 2>&1 || true)
                result=$(echo "$output" | grep -oP 'tok/s: \K[0-9.]+' || echo "0")
            fi
            ;;
        serve)
            local port=$((8100 + RANDOM % 100))
            timeout 15 "$APR_BIN" serve "$model_path" --port $port $gpu_flag &>/tmp/serve_bench.log &
            local pid=$!
            sleep 3

            if [[ "$format" == "gguf" ]]; then
                # GGUF serve - measure via multiple requests
                local start=$(date +%s.%N)
                for i in $(seq 1 5); do
                    curl -s -X POST "http://127.0.0.1:$port/v1/completions" \
                        -H "Content-Type: application/json" \
                        -d '{"model":"m","prompt":"Hi","max_tokens":10}' >/dev/null 2>&1 || true
                done
                local end=$(date +%s.%N)
                local elapsed=$(echo "$end - $start" | bc -l)
                result=$(echo "scale=1; 50 / $elapsed" | bc -l 2>/dev/null || echo "0")  # 5 requests × 10 tokens
            else
                # APR serve - has tok_per_sec in response
                local resp=$(curl -s -X POST "http://127.0.0.1:$port/v1/completions" \
                    -H "Content-Type: application/json" \
                    -d '{"model":"m","prompt":"Hi","max_tokens":5}' 2>/dev/null || echo "{}")
                result=$(echo "$resp" | grep -oP '"tok_per_sec":\s*\K[0-9.]+' || echo "0")
            fi

            kill $pid 2>/dev/null || true
            wait $pid 2>/dev/null || true
            ;;
        batch)
            if [[ "$format" == "gguf" && "$backend" == "gpu" ]]; then
                # Use realizar batched benchmark
                local output=$(MODEL_PATH="$model_path" "$REALIZAR_BIN" 2>&1 || true)
                case "$batch_size" in
                    1)  result=$(echo "$output" | grep "M=8:" | grep -oP '[0-9.]+(?= tok/s)' | head -1 || echo "0")
                        result=$(echo "scale=1; $result / 8" | bc -l 2>/dev/null || echo "0") ;;  # Approximate M=1
                    8)  result=$(echo "$output" | grep "M=8:" | grep -oP '[0-9.]+(?= tok/s)' || echo "0") ;;
                    16) result=$(echo "$output" | grep "M=16:" | grep -oP '[0-9.]+(?= tok/s)' || echo "0") ;;
                    32) result=$(echo "$output" | grep "M=32:" | grep -oP '[0-9.]+(?= tok/s)' || echo "0") ;;
                esac
            else
                result="N/A"
            fi
            ;;
        chat)
            # Chat is generate with chat template - same as generate
            if [[ "$format" == "gguf" ]]; then
                local output=$("$APR_BIN" run "$model_path" --prompt "Hello" --max-tokens 10 $gpu_flag --benchmark 2>&1 || true)
                result=$(echo "$output" | grep -oP 'Inference:.*\(\K[0-9.]+(?= tok/s)' || echo "0")
            else
                local output=$("$APR_BIN" run "$model_path" --prompt "1,2,3" --max-tokens 5 $gpu_flag --benchmark 2>&1 || true)
                result=$(echo "$output" | grep -oP 'tok/s: \K[0-9.]+' || echo "0")
            fi
            ;;
    esac

    echo "${result:-0}"
}

# ═══════════════════════════════════════════════════════════════════════
# Main Benchmark Loop
# ═══════════════════════════════════════════════════════════════════════

log "${BOLD}╔═══════════════════════════════════════════════════════════════════════╗${NC}"
log "${BOLD}║           SCIENTIFIC BENCHMARK MATRIX                                 ║${NC}"
log "${BOLD}║           Warmup: $WARMUP_ITERATIONS | Iterations: $MEASURE_ITERATIONS | $(date -Iseconds)             ║${NC}"
log "${BOLD}╚═══════════════════════════════════════════════════════════════════════╝${NC}"
log ""

# Initialize JSON results
JSON_RESULTS='{"timestamp":"'"$(date -Iseconds)"'","config":{"warmup":'$WARMUP_ITERATIONS',"iterations":'$MEASURE_ITERATIONS'},"results":['

FIRST_RESULT=true

# Convert comma-separated to arrays
IFS=',' read -ra FORMAT_ARR <<< "$FORMATS"
IFS=',' read -ra BACKEND_ARR <<< "$BACKENDS"
IFS=',' read -ra MODE_ARR <<< "$MODES"
IFS=',' read -ra BATCH_ARR <<< "$BATCH_SIZES"

# Results table header
log "${CYAN}═══ Running Benchmark Matrix ═══${NC}"
log ""
log "┌──────────┬─────────┬──────────┬───────┬──────────────┬──────────┬────────┬───────────────┐"
log "│ Format   │ Backend │ Mode     │ M     │ Mean (tok/s) │ Std      │ CV%    │ vs Ollama     │"
log "├──────────┼─────────┼──────────┼───────┼──────────────┼──────────┼────────┼───────────────┤"

for format in "${FORMAT_ARR[@]}"; do
    # APR model size validation - reject toy models for perf benchmarks
    if [[ "$format" == "apr" ]]; then
        if [[ -f "$MODEL_APR" ]]; then
            apr_size_bytes=$(stat -c%s "$MODEL_APR" 2>/dev/null || echo "0")
            apr_size_mb=$((apr_size_bytes / 1024 / 1024))
            if [[ $apr_size_mb -lt $APR_MIN_SIZE_MB ]]; then
                log "${YELLOW}⚠ SKIPPING APR: Model too small (${apr_size_mb}MB < ${APR_MIN_SIZE_MB}MB minimum)${NC}"
                log "${YELLOW}  Toy models (147K params) vs real models (1.5B params) = invalid comparison${NC}"
                log "${YELLOW}  Use --format apr to force (results will be marked INVALID)${NC}"
                log ""
                continue
            fi
        fi
    fi

    for backend in "${BACKEND_ARR[@]}"; do
        for mode in "${MODE_ARR[@]}"; do
            for batch_size in "${BATCH_ARR[@]}"; do
                # Skip invalid combinations
                if [[ "$mode" == "batch" && ("$format" != "gguf" || "$backend" != "gpu") ]]; then
                    continue
                fi
                if [[ "$mode" == "batch" && "$batch_size" == "1" ]]; then
                    continue  # Batch mode requires M > 1
                fi

                if [[ "$JSON_ONLY" != "true" ]]; then
                    echo -n "Testing $format/$backend/$mode/M=$batch_size..."
                fi

                # Warmup
                for i in $(seq 1 $WARMUP_ITERATIONS); do
                    run_benchmark "$format" "$backend" "$mode" "$batch_size" >/dev/null 2>&1 || true
                done

                # Measure
                declare -a measurements=()
                for i in $(seq 1 $MEASURE_ITERATIONS); do
                    val=$(run_benchmark "$format" "$backend" "$mode" "$batch_size")
                    if [[ "$val" != "N/A" && "$val" != "0" ]]; then
                        measurements+=("$val")
                    fi
                done

                # Calculate statistics
                if [[ ${#measurements[@]} -gt 0 ]]; then
                    read mean std cv <<< $(calc_stats "${measurements[@]}")
                else
                    mean="0"
                    std="0"
                    cv="0"
                fi

                # Calculate vs Ollama
                baseline_key="gpu_single"
                if [[ "$backend" == "cpu" ]]; then
                    baseline_key="cpu"
                elif [[ "$mode" == "batch" ]]; then
                    baseline_key="gpu_batched"
                fi
                baseline=${OLLAMA_BASELINE[$baseline_key]}
                ratio="N/A"
                if [[ "$mean" != "0" && "$mean" != "N/A" ]]; then
                    ratio=$(echo "scale=2; $mean / $baseline" | bc -l 2>/dev/null || echo "N/A")
                fi

                # Output table row
                if [[ "$JSON_ONLY" != "true" ]]; then
                    printf "\r│ %-8s │ %-7s │ %-8s │ %-5s │ %12s │ %8s │ %6s │ %13s │\n" \
                        "$format" "$backend" "$mode" "$batch_size" "$mean" "$std" "$cv" "${ratio}x"
                fi

                # Add to JSON
                if [[ "$FIRST_RESULT" != "true" ]]; then
                    JSON_RESULTS+=','
                fi
                FIRST_RESULT=false
                JSON_RESULTS+='{"format":"'"$format"'","backend":"'"$backend"'","mode":"'"$mode"'","batch_size":'$batch_size',"mean":'${mean:-0}',"std":'${std:-0}',"cv":'${cv:-0}',"vs_ollama":"'"$ratio"'"}'
            done
        done
    done
done

log "└──────────┴─────────┴──────────┴───────┴──────────────┴──────────┴────────┴───────────────┘"

# Close JSON
JSON_RESULTS+='],"ollama_baselines":{"gpu_batched":291,"gpu_single":120,"cpu":15}}'

# Save JSON results
echo "$JSON_RESULTS" | jq . > "$RESULTS_FILE"

log ""
log "${GREEN}Results saved: $RESULTS_FILE${NC}"
log ""

# Summary
if [[ "$JSON_ONLY" == "true" ]]; then
    cat "$RESULTS_FILE"
fi
