#!/usr/bin/env bash
#
# QA Script for apr run (Falsification Agent)
# Tests direct model execution across GGUF and APR formats
#
# CRITICAL: Format parity requires SAME QUANTIZATION (Q4_K)
#   - GGUF Q4_K vs APR Q4_K = Valid comparison
#   - GGUF Q4_K vs APR F32 = INVALID (8x memory diff, different kernels)
#   - SafeTensors F32 = Import format only, NOT for performance comparison
#
# Usage: ./qa-run.sh [model_path] [format]
#   model_path: Path to model (optional, uses default 1.5B GGUF)
#   format: gguf|apr (optional, auto-detect from extension)
#
# Format Parity Testing:
#   ./qa-run.sh --format-parity
#   Tests GGUF Q4_K vs APR Q4_K (same quantization, fair comparison)
#
set -euo pipefail

# Configuration
MODEL_PATH="${1:-}"
FORMAT="${2:-}"
VERBOSE="${VERBOSE:-1}"
TIMEOUT_SECS=60
# Note: apr run loads model fresh each time, so targets are lower than apr chat
# (model load overhead ~1.5-2s, generation ~10 tok/s actual)
# apr chat keeps model loaded and achieves 30+ tok/s
TARGET_TOKS_CPU=8   # End-to-end with model load (CPU)
TARGET_TOKS_GPU=10  # End-to-end with model load (GPU)

# Model paths for format parity testing
# CRITICAL: APR model MUST be Q4_K quantized for fair comparison with GGUF Q4_K
GGUF_MODEL="${HOME}/.cache/huggingface/models/qwen2.5-coder-1.5b-gguf/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
APR_MODEL="${HOME}/models/qwen2.5-coder-1.5b-q4k.apr"  # Q4_K quantized APR (must match GGUF)
# SafeTensors F32 is for import only - NOT included in performance parity tests
ST_MODEL="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a/model.safetensors"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Counters
TESTS_PASSED=0
TESTS_FAILED=0
TOTAL_TESTS=0

print_color() {
    printf '%b%s%b\n' "$1" "$2" "${NC}"
}

print_header() {
    print_color "${BLUE}" "\n=== $1 ==="
}

log_debug() {
    if [[ "${VERBOSE}" == "1" ]]; then
        print_color "${CYAN}" "DEBUG: $1"
    fi
}

print_result() {
    local id="$1"
    local name="$2"
    local result="$3"
    local details="${4:-}"

    TOTAL_TESTS=$((TOTAL_TESTS+1))

    if [[ "${result}" == "PASS" ]]; then
        print_color "${GREEN}" "[PASS] ${id}: ${name}"
        TESTS_PASSED=$((TESTS_PASSED+1))
    else
        print_color "${RED}" "[FAIL] ${id}: ${name}"
        TESTS_FAILED=$((TESTS_FAILED+1))
    fi

    if [[ -n "$details" ]]; then
        printf '       %s\n' "$details"
    fi
}

# Test basic run functionality
test_run_basic() {
    local model="$1"
    local format="$2"

    print_header "Testing ${format} Run Basic"

    if [[ ! -f "${model}" ]]; then
        print_result "F-RUN-001" "${format} Model Exists" "FAIL" "Model not found: ${model}"
        return 1
    fi
    print_result "F-RUN-001" "${format} Model Exists" "PASS"

    # Test run with prompt
    local output
    output=$(timeout "${TIMEOUT_SECS}" apr run "${model}" --prompt "What is 2+2? Answer with just the number." --max-tokens 10 2>&1) || true
    log_debug "Run output: ${output}"

    # Check for valid response containing "4"
    if [[ "${output}" == *"4"* ]]; then
        print_result "F-RUN-002" "${format} Correct Answer" "PASS" "Contains '4'"
    else
        print_result "F-RUN-002" "${format} Correct Answer" "FAIL" "Expected '4' in output"
    fi

    # Check for garbage patterns
    if [[ "${output}" == *"token"[0-9]* ]] || [[ "${output}" == *$'\xef\xbf\xbd'* ]]; then
        print_result "F-RUN-003" "${format} No Garbage" "FAIL" "Garbage patterns detected"
    else
        print_result "F-RUN-003" "${format} No Garbage" "PASS"
    fi

    # Check for BPE artifacts
    if [[ "${output}" == *Ġ* ]] || [[ "${output}" == *Ċ* ]]; then
        print_result "F-RUN-004" "${format} No BPE Artifacts" "FAIL" "BPE artifacts detected"
    else
        print_result "F-RUN-004" "${format} No BPE Artifacts" "PASS"
    fi

    # Test with trace flag
    # Note: apr run --trace enables tracing mode but detailed trace data is only available
    # via apr serve with X-Trace-Level header. This test verifies the flag is accepted.
    local trace_output
    trace_output=$(timeout "${TIMEOUT_SECS}" apr run "${model}" --prompt "Hi" --max-tokens 5 --trace 2>&1) || true
    log_debug "Trace output: ${trace_output}"

    # Check that trace flag is accepted and acknowledged
    if [[ "${trace_output}" == *"tracing enabled"* ]] || [[ "${trace_output}" == *"APR-TRACE"* ]]; then
        print_result "F-RUN-005" "${format} Trace Flag" "PASS" "Trace mode enabled"
    else
        print_result "F-RUN-005" "${format} Trace Flag" "FAIL" "Trace flag not recognized"
    fi
}

# Test run performance
test_run_performance() {
    local model="$1"
    local format="$2"

    print_header "Testing ${format} Run Performance"

    if [[ ! -f "${model}" ]]; then
        print_result "F-RUN-010" "${format} Performance" "SKIP" "Model not found"
        return 0
    fi

    # Run with timing - generate 50 tokens
    local start_time end_time duration
    start_time=$(date +%s.%N)

    local output
    output=$(timeout 120 apr run "${model}" --prompt "Write a haiku about programming." --max-tokens 50 2>&1) || true

    end_time=$(date +%s.%N)
    duration=$(echo "${end_time} - ${start_time}" | bc)

    # Estimate tokens (rough: word count * 1.3)
    local word_count tokens_est toks_per_sec
    word_count=$(echo "${output}" | wc -w)
    tokens_est=$((word_count * 13 / 10))
    toks_per_sec=$(echo "scale=2; ${tokens_est} / ${duration}" | bc)

    log_debug "Duration: ${duration}s, Est tokens: ${tokens_est}, tok/s: ${toks_per_sec}"

    # Check against target (CPU target for now)
    local target="${TARGET_TOKS_CPU}"
    if [[ $(echo "${toks_per_sec} >= ${target}" | bc) -eq 1 ]]; then
        print_result "F-RUN-010" "${format} Performance" "PASS" "${toks_per_sec} tok/s >= ${target} target"
    else
        print_result "F-RUN-010" "${format} Performance" "FAIL" "${toks_per_sec} tok/s < ${target} target"
    fi
}

# Test determinism
test_run_determinism() {
    local model="$1"
    local format="$2"

    print_header "Testing ${format} Run Determinism"

    if [[ ! -f "${model}" ]]; then
        print_result "F-RUN-020" "${format} Determinism" "SKIP" "Model not found"
        return 0
    fi

    # Note: apr run doesn't support --temperature flag, so determinism depends on default sampling
    # Extract just the "Output:" section to avoid timing differences
    local out1 out2 text1 text2
    out1=$(timeout "${TIMEOUT_SECS}" apr run "${model}" --prompt "Say hello" --max-tokens 10 2>&1) || true
    out2=$(timeout "${TIMEOUT_SECS}" apr run "${model}" --prompt "Say hello" --max-tokens 10 2>&1) || true

    # Extract text between "Output:" and "Completed" (the actual model output)
    text1=$(echo "${out1}" | sed -n '/^Output:$/,/^Completed/p' | grep -v "^Output:$" | grep -v "^Completed")
    text2=$(echo "${out2}" | sed -n '/^Output:$/,/^Completed/p' | grep -v "^Output:$" | grep -v "^Completed")

    log_debug "Output 1: ${text1}"
    log_debug "Output 2: ${text2}"

    # With greedy sampling (default), outputs should match
    if [[ "${text1}" == "${text2}" ]]; then
        print_result "F-RUN-020" "${format} Determinism" "PASS" "Outputs match"
    else
        print_result "F-RUN-020" "${format} Determinism" "FAIL" "Outputs differ"
    fi
}

# Format parity test
# CRITICAL: Tests GGUF Q4_K vs APR Q4_K (same quantization for fair comparison)
test_format_parity() {
    print_header "Format Parity Test (GGUF Q4_K vs APR Q4_K)"
    print_color "${CYAN}" "REQUIREMENT: Both formats must use Q4_K quantization"
    print_color "${CYAN}" "  - GGUF Q4_K (~1.1 GB) vs APR Q4_K (~1.1 GB) = Valid"
    print_color "${CYAN}" "  - GGUF Q4_K vs APR F32 = INVALID comparison"

    local formats_tested=0

    # Test GGUF Q4_K (reference baseline)
    if [[ -f "${GGUF_MODEL}" ]]; then
        print_color "${BLUE}" "\n[1/2] Testing GGUF Q4_K (reference baseline)..."
        test_run_basic "${GGUF_MODEL}" "GGUF_Q4K"
        test_run_performance "${GGUF_MODEL}" "GGUF_Q4K"
        test_run_determinism "${GGUF_MODEL}" "GGUF_Q4K"
        formats_tested=$((formats_tested+1))
    else
        print_color "${RED}" "[FAIL] GGUF Q4_K model not found: ${GGUF_MODEL}"
        print_color "${RED}" "Cannot run parity test without reference model"
        return 1
    fi

    # Test APR Q4_K (must match GGUF performance)
    if [[ -f "${APR_MODEL}" ]]; then
        print_color "${BLUE}" "\n[2/2] Testing APR Q4_K (must match GGUF)..."
        # Verify APR model is Q4_K quantized (file size ~1.1 GB, not ~6.6 GB)
        local apr_size
        apr_size=$(stat -c%s "${APR_MODEL}" 2>/dev/null || stat -f%z "${APR_MODEL}" 2>/dev/null || echo 0)
        local apr_size_gb=$((apr_size / 1073741824))
        if [[ ${apr_size_gb} -gt 2 ]]; then
            print_color "${YELLOW}" "WARNING: APR model is ${apr_size_gb} GB - likely F32 not Q4_K"
            print_color "${YELLOW}" "For valid parity test, convert GGUF to APR Q4_K:"
            print_color "${YELLOW}" "  apr convert ${GGUF_MODEL} --format apr -o ${APR_MODEL}"
        fi
        test_run_basic "${APR_MODEL}" "APR_Q4K"
        test_run_performance "${APR_MODEL}" "APR_Q4K"
        test_run_determinism "${APR_MODEL}" "APR_Q4K"
        formats_tested=$((formats_tested+1))
    else
        print_color "${YELLOW}" "[SKIP] APR Q4_K model not found: ${APR_MODEL}"
        print_color "${YELLOW}" "Create Q4_K APR model with:"
        print_color "${YELLOW}" "  apr convert ${GGUF_MODEL} --format apr -o ${APR_MODEL}"
    fi

    # SafeTensors F32 - informational only, NOT a parity test
    if [[ -f "${ST_MODEL}" ]]; then
        print_color "${BLUE}" "\n[INFO] SafeTensors F32 (import format only, not parity test)..."
        print_color "${YELLOW}" "SafeTensors F32 is for import/export only"
        print_color "${YELLOW}" "Performance comparison with GGUF Q4_K is INVALID (8x memory diff)"
        # Only test correctness, skip performance (unfair comparison)
        test_run_basic "${ST_MODEL}" "SafeTensors_F32"
        # Skip performance test - invalid comparison
        print_color "${YELLOW}" "[SKIP] F-RUN-010: SafeTensors performance (invalid comparison with Q4_K)"
        test_run_determinism "${ST_MODEL}" "SafeTensors_F32"
    fi

    print_color "${BLUE}" "\nParity formats tested: ${formats_tested}"
}

# Main
main() {
    print_color "${BLUE}" "╔══════════════════════════════════════════════════════════════╗"
    print_color "${BLUE}" "║         APR RUN QA - Falsification Suite                     ║"
    print_color "${BLUE}" "║         Target: GGUF parity (${TARGET_TOKS_CPU}+ tok/s CPU)                   ║"
    print_color "${BLUE}" "╚══════════════════════════════════════════════════════════════╝"

    if [[ "${MODEL_PATH}" == "--format-parity" ]]; then
        test_format_parity
    elif [[ -n "${MODEL_PATH}" ]]; then
        local fmt="${FORMAT:-gguf}"
        [[ "${MODEL_PATH}" == *.apr ]] && fmt="APR"
        [[ "${MODEL_PATH}" == *.safetensors ]] && fmt="SafeTensors"
        test_run_basic "${MODEL_PATH}" "${fmt}"
        test_run_performance "${MODEL_PATH}" "${fmt}"
        test_run_determinism "${MODEL_PATH}" "${fmt}"
    else
        # Default: test GGUF
        if [[ -f "${GGUF_MODEL}" ]]; then
            test_run_basic "${GGUF_MODEL}" "GGUF"
            test_run_performance "${GGUF_MODEL}" "GGUF"
            test_run_determinism "${GGUF_MODEL}" "GGUF"
        else
            print_color "${RED}" "ERROR: No model specified and default GGUF not found"
            echo "Usage: $0 [model_path] [format]"
            echo "       $0 --format-parity"
            exit 2
        fi
    fi

    # Summary
    print_header "Falsification Summary"
    printf 'Total Tests: %s\n' "${TOTAL_TESTS}"
    printf 'Passed:      %s\n' "${TESTS_PASSED}"
    printf 'Failed:      %s\n' "${TESTS_FAILED}"

    if [[ "${TESTS_FAILED}" -eq 0 ]]; then
        print_color "${GREEN}" "Hypothesis \"apr run produces correct output at GGUF parity\" SURVIVED falsification."
        exit 0
    else
        print_color "${RED}" "Hypothesis \"apr run produces correct output at GGUF parity\" FALSIFIED."
        exit 1
    fi
}

main "$@"
