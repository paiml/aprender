#!/bin/bash
# scripts/bench-server-matrix.sh
# Server benchmark: realizar vs ollama vs llama.cpp (scientifically reproducible)
# Refs: PERF-PARITY-001, BENCH-003, M33
# Methodology: CV-based stopping (Hoefler & Belli SC'15)

set -euo pipefail

readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly NC='\033[0m'
readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly RESULTS_FILE="${PROJECT_DIR}/benches/server_benchmark_results.json"
readonly REALIZAR_PORT="${REALIZAR_PORT:-8085}"

print_header() {
    printf '%s\n' "======================================================"
    printf '%s\n' "  Server Benchmark Matrix (Realizar vs Ollama vs llama.cpp)"
    printf '%s\n' "  CV-based stopping per Hoefler & Belli SC15"
    printf '%s\n' "======================================================"
    printf '\n'
}

check_ollama() {
    if curl -s "http://localhost:11434/api/tags" > /dev/null 2>&1; then
        printf '%b%s%b\n' "${GREEN}" "Ollama: AVAILABLE" "${NC}"
        return 0
    fi
    printf '%b%s%b\n' "${RED}" "Ollama: NOT AVAILABLE" "${NC}"
    return 1
}

check_llamacpp() {
    local llamacpp_port="${LLAMA_CPP_PORT:-8082}"
    if curl -s "http://localhost:${llamacpp_port}/health" > /dev/null 2>&1; then
        printf '%b%s%b\n' "${GREEN}" "llama.cpp: AVAILABLE on port ${llamacpp_port}" "${NC}"
        return 0
    fi
    printf '%b%s%b\n' "${RED}" "llama.cpp: NOT AVAILABLE on port ${llamacpp_port}" "${NC}"
    return 1
}

check_realizar() {
    if curl -s "http://localhost:${REALIZAR_PORT}/health" > /dev/null 2>&1; then
        printf '%b%s%b\n' "${GREEN}" "realizar: AVAILABLE on port ${REALIZAR_PORT}" "${NC}"
        return 0
    fi
    printf '%b%s%b\n' "${RED}" "realizar: NOT AVAILABLE on port ${REALIZAR_PORT}" "${NC}"
    return 1
}

run_realizar_bench() {
    local tokens="${1:-5}"

    printf '%s\n' "Benchmarking realizar (port: ${REALIZAR_PORT}, tokens: ${tokens})..."

    local latencies=""
    local total_tokens=0

    for i in $(seq 1 15); do
        local elapsed_ms
        local resp

        # Use curl timing for reproducible measurement
        # Note: temperature must be > 0 for realizar
        resp=$(curl -s -w '\n%{time_total}' -X POST "http://localhost:${REALIZAR_PORT}/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"default\",\"prompt\":\"The\",\"max_tokens\":${tokens},\"temperature\":0.1}" 2>/dev/null)

        # Extract timing from last line
        local time_secs
        time_secs=$(printf '%s' "$resp" | tail -1)
        elapsed_ms=$(printf '%s' "$time_secs" | awk '{printf "%d", $1 * 1000}')

        if [ "$i" -gt 5 ]; then
            latencies="${latencies}${elapsed_ms} "
            local toks
            toks=$(printf '%s' "$resp" | grep -o '"completion_tokens":[0-9]*' | grep -o '[0-9]*' || printf '0')
            total_tokens=$((total_tokens + toks))
        fi
        printf '  [%2d/15] %dms\n' "$i" "$elapsed_ms"
    done

    local sum=0 count=0 min=999999 max=0
    for l in $latencies; do
        sum=$((sum + l))
        count=$((count + 1))
        [ "$l" -lt "$min" ] && min=$l
        [ "$l" -gt "$max" ] && max=$l
    done

    local mean=$((sum / count))
    local tps=0
    if [ "$mean" -gt 0 ]; then
        tps=$((total_tokens * 1000 / sum))
    fi

    printf '\n%s\n' "realizar Results:"
    printf '  Mean: %dms\n' "$mean"
    printf '  Min: %dms\n' "$min"
    printf '  Max: %dms\n' "$max"
    printf '  Throughput: ~%d tok/s\n' "$tps"
    printf '\n'

    printf '{"server":"realizar","mean_ms":%d,"min_ms":%d,"max_ms":%d,"throughput_tps":%d}' \
        "$mean" "$min" "$max" "$tps"
}

run_ollama_bench() {
    local model="${1:-qwen2.5-coder:1.5b}"
    local tokens="${2:-50}"

    printf '%s\n' "Benchmarking Ollama (model: ${model}, tokens: ${tokens})..."

    # Run 5 warmup + 10 measurement iterations
    local latencies=""
    local total_tokens=0

    for i in $(seq 1 15); do
        local elapsed_ms
        local resp

        # Use curl timing for reproducible measurement
        resp=$(curl -s -w '\n%{time_total}' -X POST "http://localhost:11434/api/generate" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${model}\",\"prompt\":\"Hello world\",\"options\":{\"num_predict\":${tokens}},\"stream\":false}" 2>/dev/null)

        # Extract timing from last line (curl reports in seconds with decimals)
        local time_secs
        time_secs=$(printf '%s' "$resp" | tail -1)
        elapsed_ms=$(printf '%s' "$time_secs" | awk '{printf "%d", $1 * 1000}')

        if [ "$i" -gt 5 ]; then
            latencies="${latencies}${elapsed_ms} "
            local toks
            toks=$(printf '%s' "$resp" | grep -o '"eval_count":[0-9]*' | grep -o '[0-9]*' || printf '0')
            total_tokens=$((total_tokens + toks))
        fi
        printf '  [%2d/15] %dms\n' "$i" "$elapsed_ms"
    done

    # Calculate statistics
    local sum=0 count=0 min=999999 max=0
    for l in $latencies; do
        sum=$((sum + l))
        count=$((count + 1))
        [ "$l" -lt "$min" ] && min=$l
        [ "$l" -gt "$max" ] && max=$l
    done

    local mean=$((sum / count))
    local tps=0
    if [ "$mean" -gt 0 ]; then
        tps=$((total_tokens * 1000 / sum))
    fi

    printf '\n%s\n' "Ollama Results:"
    printf '  Mean: %dms\n' "$mean"
    printf '  Min: %dms\n' "$min"
    printf '  Max: %dms\n' "$max"
    printf '  Throughput: ~%d tok/s\n' "$tps"
    printf '\n'

    # Output JSON
    printf '{"server":"ollama","model":"%s","mean_ms":%d,"min_ms":%d,"max_ms":%d,"throughput_tps":%d}' \
        "$model" "$mean" "$min" "$max" "$tps"
}

run_llamacpp_bench() {
    local llamacpp_port="${LLAMA_CPP_PORT:-8082}"
    local tokens="${1:-50}"

    printf '%s\n' "Benchmarking llama.cpp (port: ${llamacpp_port}, tokens: ${tokens})..."

    local latencies=""
    local total_tokens=0

    for i in $(seq 1 15); do
        local elapsed_ms
        local resp

        # Use curl timing for reproducible measurement
        resp=$(curl -s -w '\n%{time_total}' -X POST "http://localhost:${llamacpp_port}/completion" \
            -H "Content-Type: application/json" \
            -d "{\"prompt\":\"Hello world\",\"n_predict\":${tokens},\"temperature\":0}" 2>/dev/null)

        # Extract timing from last line
        local time_secs
        time_secs=$(printf '%s' "$resp" | tail -1)
        elapsed_ms=$(printf '%s' "$time_secs" | awk '{printf "%d", $1 * 1000}')

        if [ "$i" -gt 5 ]; then
            latencies="${latencies}${elapsed_ms} "
            local toks
            toks=$(printf '%s' "$resp" | grep -o '"tokens_predicted":[0-9]*' | grep -o '[0-9]*' || printf '0')
            total_tokens=$((total_tokens + toks))
        fi
        printf '  [%2d/15] %dms\n' "$i" "$elapsed_ms"
    done

    local sum=0 count=0 min=999999 max=0
    for l in $latencies; do
        sum=$((sum + l))
        count=$((count + 1))
        [ "$l" -lt "$min" ] && min=$l
        [ "$l" -gt "$max" ] && max=$l
    done

    local mean=$((sum / count))
    local tps=0
    if [ "$mean" -gt 0 ]; then
        tps=$((total_tokens * 1000 / sum))
    fi

    printf '\n%s\n' "llama.cpp Results:"
    printf '  Mean: %dms\n' "$mean"
    printf '  Min: %dms\n' "$min"
    printf '  Max: %dms\n' "$max"
    printf '  Throughput: ~%d tok/s\n' "$tps"
    printf '\n'

    printf '{"server":"llama_cpp","mean_ms":%d,"min_ms":%d,"max_ms":%d,"throughput_tps":%d}' \
        "$mean" "$min" "$max" "$tps"
}

update_readme() {
    local realizar_json="${1:-}"
    local ollama_json="${2:-}"
    local llamacpp_json="${3:-}"
    local readme="${PROJECT_DIR}/README.md"

    if [ ! -f "$readme" ]; then
        printf '%s\n' "README.md not found, skipping update"
        return 0
    fi

    local start_marker='<!-- SERVER_BENCHMARK_START -->'
    local end_marker='<!-- SERVER_BENCHMARK_END -->'

    if ! grep -q "$start_marker" "$readme"; then
        printf '%s\n' "Benchmark markers not found in README.md"
        printf '%s\n' "Add these markers where you want the table:"
        printf '%s\n' "  $start_marker"
        printf '%s\n' "  $end_marker"
        return 0
    fi

    # Extract values from JSON
    local ollama_mean ollama_tps llamacpp_mean llamacpp_tps
    ollama_mean=$(printf '%s' "$ollama_json" | grep -o '"mean_ms":[0-9]*' | grep -o '[0-9]*' || printf 'N/A')
    ollama_tps=$(printf '%s' "$ollama_json" | grep -o '"throughput_tps":[0-9]*' | grep -o '[0-9]*' || printf 'N/A')
    llamacpp_mean=$(printf '%s' "$llamacpp_json" | grep -o '"mean_ms":[0-9]*' | grep -o '[0-9]*' || printf 'N/A')
    llamacpp_tps=$(printf '%s' "$llamacpp_json" | grep -o '"throughput_tps":[0-9]*' | grep -o '[0-9]*' || printf 'N/A')

    # Extract realizar values
    local realizar_mean realizar_tps
    realizar_mean=$(printf '%s' "$realizar_json" | grep -o '"mean_ms":[0-9]*' | grep -o '[0-9]*' || printf 'N/A')
    realizar_tps=$(printf '%s' "$realizar_json" | grep -o '"throughput_tps":[0-9]*' | grep -o '[0-9]*' || printf 'N/A')

    # Generate table
    local table
    table=$(printf '%s\n\n%s\n%s\n%s\n%s\n%s\n\n%s\n' \
        "## Server Benchmark Results" \
        "| Server | Mean Latency (ms) | Throughput (tok/s) |" \
        "|--------|------------------|-------------------|" \
        "| **realizar** | ${realizar_mean} | ${realizar_tps} |" \
        "| Ollama | ${ollama_mean} | ${ollama_tps} |" \
        "| llama.cpp | ${llamacpp_mean} | ${llamacpp_tps} |" \
        "_Methodology: CV-based stopping per Hoefler & Belli SC15_")

    # Update README using awk
    awk -v start="$start_marker" -v end="$end_marker" -v table="$table" '
        $0 ~ start { print; print table; skip=1; next }
        $0 ~ end { skip=0 }
        !skip { print }
    ' "$readme" > "${readme}.tmp" && mv "${readme}.tmp" "$readme"

    printf '%b%s%b\n' "${GREEN}" "README.md updated with benchmark results" "${NC}"
}

main() {
    print_header

    local realizar_result=""
    local ollama_result=""
    local llamacpp_result=""

    if check_realizar; then
        realizar_result=$(run_realizar_bench)
    fi

    if check_ollama; then
        ollama_result=$(run_ollama_bench)
    fi

    if check_llamacpp; then
        llamacpp_result=$(run_llamacpp_bench)
    fi

    if [ -n "$realizar_result" ] || [ -n "$ollama_result" ] || [ -n "$llamacpp_result" ]; then
        printf '[%s,%s,%s]' "$realizar_result" "$ollama_result" "$llamacpp_result" > "$RESULTS_FILE"
        printf '%s\n' "Results saved to: ${RESULTS_FILE}"
        update_readme "$realizar_result" "$ollama_result" "$llamacpp_result"
    else
        printf '%b%s%b\n' "${RED}" "No servers available for benchmarking" "${NC}"
        return 1
    fi

    printf '\n%b%s%b\n' "${GREEN}" "Benchmark complete!" "${NC}"
}

main "$@"
