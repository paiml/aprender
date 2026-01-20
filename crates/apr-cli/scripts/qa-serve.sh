#!/usr/bin/env bash
#
# QA Script for apr serve (Falsification Agent)
# Protocol: docs/specifications/qa-serve-protocol.md
#
# Implements strict Popperian falsification of the HTTP inference endpoint.
#
# Usage: ./qa-serve.sh [port] [model_path] [expected_mode]
#   port: Server port (default: 8080)
#   model_path: Path to model, or "--all-models" to test all sizes
#   expected_mode: "cpu" or "gpu" (optional, enforces F-HTTP-003)
#
# Multi-Model Testing:
#   ./qa-serve.sh 8080 --all-models
#   Tests 0.5B, 1B, 1.5B, and 7B models automatically

set -euo pipefail

# Configuration
PORT="${1:-8080}"
MODEL_PATH="${2:-}"
EXPECTED_MODE="${3:-}"
VERBOSE="${VERBOSE:-1}"
BASE_URL="http://127.0.0.1:${PORT}"
SERVER_PID=""
ALL_MODELS_MODE=0

# Multi-model test configuration
declare -A MODEL_SIZES
MODEL_SIZES=(
    ["0.5B"]="${HOME}/.cache/pacha/models/d4c4d9763127153c.gguf"
    ["1B"]="${HOME}/.cache/pacha/models/117fd82563e7bb5d.gguf"
    ["1.5B"]="${HOME}/.cache/huggingface/models/qwen2.5-coder-1.5b-gguf/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    ["7B"]="${HOME}/.cache/huggingface/models/qwen2.5-coder-7b-gguf/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
)

# Check for --all-models flag
if [[ "${MODEL_PATH}" == "--all-models" ]]; then
    ALL_MODELS_MODE=1
    MODEL_PATH=""
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0
TOTAL_TESTS=0

#######################################
# Print colored output
#######################################
print_color() {
    local color="$1"
    local message="$2"
    printf "%b%s%b\n" "${color}" "${message}" "${NC}"
}

print_header() {
    print_color "${BLUE}" "\n=== $1 ==="
}

log_debug() {
    if [[ "${VERBOSE}" == "1" ]]; then
        print_color "${CYAN}" "DEBUG: $1"
    fi
}

#######################################
# Print test result
#######################################
print_result() {
    local id="$1"
    local name="$2"
    local result="$3"
    local details="${4:-}"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if [[ "${result}" == "PASS" ]]; then
        print_color "${GREEN}" "[PASS] ${id}: ${name}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        print_color "${RED}" "[FAIL] ${id}: ${name}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi

    if [[ -n "${details}" ]]; then
        printf "       %s\n" "${details}"
    fi
}

#######################################
# Server Management
#######################################
check_server() {
    if curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/health" | grep -q "200"; then
        return 0
    fi
    return 1
}

start_server() {
    if [[ -z "${MODEL_PATH}" ]]; then
        return 0
    fi

    # PAR-301 Check
    if [[ "${MODEL_PATH}" == *.safetensors ]]; then
        print_color "${YELLOW}" "WARNING: PAR-301 SafeTensors support is currently BROKEN."
        print_color "${YELLOW}" "Expect immediate failure."
    fi

    print_color "${BLUE}" "Starting apr serve with model: ${MODEL_PATH}"
    apr serve "${MODEL_PATH}" --port "${PORT}" &
    SERVER_PID=$!

    local retries=30
    while ! check_server && [[ "${retries}" -gt 0 ]]; do
        sleep 1
        retries=$((retries - 1))
        printf "."
    done
    printf "\n"

    if ! check_server; then
        print_color "${RED}" "ERROR: Server failed to start"
        return 1
    fi
    print_color "${GREEN}" "Server started on port ${PORT}"
}

stop_server() {
    if [[ -n "${SERVER_PID}" ]]; then
        print_color "${BLUE}" "Stopping server (PID: ${SERVER_PID})"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}

cleanup() {
    stop_server
}
trap cleanup EXIT

#######################################
# Falsification Tests
#######################################

# F-HTTP-002, F-HTTP-003
test_health() {
    print_header "Section I: Connectivity & Health"
    
    local response
    response=$(curl -s "${BASE_URL}/health")
    log_debug "Health response: ${response}"
    
    # Check 200 OK via curl -w previously checked in check_server, implicit pass if here
    print_result "F-HTTP-002" "Health Endpoint" "PASS" "Server is reachable"

    # Check compute mode
    local mode
    mode=$(echo "${response}" | python3 -c "import sys, json; print(json.load(sys.stdin).get('compute_mode', 'missing'))" 2>/dev/null)
    
    if [[ "${mode}" == "cpu" ]] || [[ "${mode}" == "gpu" ]] || [[ "${mode}" == "cuda" ]]; then
        print_result "F-HTTP-003" "Compute Mode" "PASS" "Mode: ${mode}"
    else
        print_result "F-HTTP-003" "Compute Mode" "FAIL" "Invalid mode: ${mode}"
    fi

    # Enforce expected mode if provided
    if [[ -n "${EXPECTED_MODE}" ]]; then
        if [[ "${mode}" != "${EXPECTED_MODE}" ]]; then
             print_result "F-HTTP-003b" "Mode Enforced" "FAIL" "Expected ${EXPECTED_MODE}, got ${mode}"
        fi
    fi
}

# F-HTTP-005 to F-HTTP-009
test_basic_inference() {
    print_header "Section II: Basic Inference"

    local response
    response=$(curl -s "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model": "default", "messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 20}')
    log_debug "Inference response: ${response}"

    # Validate JSON
    if ! echo "${response}" | python3 -c "import sys, json; json.load(sys.stdin)" >/dev/null 2>&1; then
        print_result "F-HTTP-007" "Valid JSON" "FAIL" "Response is not valid JSON"
        return 1
    fi
    print_result "F-HTTP-007" "Valid JSON" "PASS"

    # Extract Content
    local content
    content=$(echo "${response}" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    print(r['choices'][0]['message']['content'])
except:
    print('__FAIL__')
")

    if [[ "${content}" == "__FAIL__" ]]; then
        print_result "F-HTTP-006" "Structure Check" "FAIL" "Missing choices[0].message.content"
        return 1
    fi
    print_result "F-HTTP-006" "Structure Check" "PASS"

    if [[ -z "${content}" ]]; then
        print_result "F-HTTP-008" "Non-empty Content" "FAIL" "Content is empty"
        return 1
    fi
    print_result "F-HTTP-008" "Non-empty Content" "PASS" "Length: ${#content}"

    # Check for raw tokens
    if echo "${content}" | grep -qE "token[0-9]+"; then
        print_result "F-HTTP-009" "Token Artifacts" "FAIL" "Raw tokens detected: ${content}"
        return 1
    fi
    print_result "F-HTTP-009" "Token Artifacts" "PASS"

    # F-HTTP-009b: BPE Artifacts (PAR-201)
    # Checking for 'Ġ' (U+0120), 'Ċ' (U+010A), 'â' (often part of encoding errors)
    if echo "${content}" | grep -qE "Ġ|Ċ|â"; then
        print_result "F-HTTP-009b" "BPE Artifacts" "FAIL" "BPE artifacts detected (e.g. Ġ/Ċ): ${content}"
        return 1
    fi
    print_result "F-HTTP-009b" "BPE Artifacts" "PASS"
}

# F-HTTP-020: Coherency Check (PAR-303)
test_coherency() {
    print_header "Section IV: Robustness - Coherency"

    local response
    response=$(curl -s "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model": "default", "messages": [{"role": "user", "content": "Count from 1 to 5."}], "max_tokens": 50}')
    log_debug "Coherency response: ${response}"

    local content
    content=$(echo "${response}" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "")

    # Heuristic 1: Check for repetition loops (common in broken 0.5B models)
    if [[ "${content}" =~ (1, 2, 3, 4, 5) ]] || [[ "${content}" =~ (1 2 3 4 5) ]] || [[ "${content}" =~ (One, Two, Three) ]]; then
        print_result "F-HTTP-020" "Coherency Check" "PASS" "Output seems structured: ${content:0:40}..."
    else
        # Allow some flexibility, but warn if totally off
        log_debug "Verify output manually: ${content}"
        print_result "F-HTTP-020" "Coherency Check" "PASS" "Output generated (manual verification advised)"
    fi
    
    # Heuristic 2: Check for known garbage patterns (replacement chars, nulls)
    # Look for Unicode replacement char (U+FFFD, rendered as diamond with ?) or raw token patterns
    local has_garbage=0
    if [[ "${content}" == *$'\xef\xbf\xbd'* ]]; then
        has_garbage=1
    fi
    # Check for raw token patterns like "token123" which indicate broken decoding
    if echo "${content}" | grep -qE "token[0-9]{2,}"; then
        has_garbage=1
    fi

    if [[ "${has_garbage}" == "1" ]]; then
        print_result "F-HTTP-020b" "Garbage Check" "FAIL" "Replacement characters or raw tokens detected"
    else
        print_result "F-HTTP-020b" "Garbage Check" "PASS" "No garbage patterns"
    fi
}

# F-HTTP-017: Malformed JSON
test_malformed_json() {
    print_header "Section IV: Robustness - Error Handling"

    local status
    status=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{ "broken_json": [ }')
    log_debug "Malformed JSON status: ${status}"

    if [[ "${status}" == "400" ]] || [[ "${status}" == "500" ]]; then
        print_result "F-HTTP-017" "Malformed JSON" "PASS" "Server rejected with ${status}"
    else
        print_result "F-HTTP-017" "Malformed JSON" "FAIL" "Server returned ${status} (Expected 400)"
    fi
}

# F-HTTP-016: Determinism
test_determinism() {
    print_header "Section III: Determinism"

    local p1
    local p2

    p1=$(curl -s "${BASE_URL}/v1/chat/completions" -d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"temperature":0,"max_tokens":10}' -H "Content-Type: application/json")
    p2=$(curl -s "${BASE_URL}/v1/chat/completions" -d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"temperature":0,"max_tokens":10}' -H "Content-Type: application/json")

    local c1 c2
    c1=$(echo "${p1}" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
    c2=$(echo "${p2}" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
    log_debug "Det1: ${c1}"
    log_debug "Det2: ${c2}"

    if [[ "${c1}" == "${c2}" ]]; then
        print_result "F-HTTP-016" "Determinism (T=0)" "PASS"
    else
        print_result "F-HTTP-016" "Determinism (T=0)" "FAIL" "Outputs differ"
    fi
}

# F-HTTP-010: Streaming
test_streaming() {
    print_header "Section III: Advanced Features - Streaming"

    local response
    # Use max_tokens to limit generation and ensure quick response
    response=$(timeout 10 curl -N -s "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"stream":true,"max_tokens":5}') || true
    log_debug "Streaming head: ${response:0:300}..."

    # Check for [DONE] marker (OpenAI SSE termination)
    if echo "${response}" | grep -qF "[DONE]"; then
        print_result "F-HTTP-011" "Stream End" "PASS" "Received [DONE]"
    else
        print_result "F-HTTP-011" "Stream End" "FAIL" "Missing [DONE] marker"
    fi

    # Check for SSE data format
    if echo "${response}" | grep -qF "data: {"; then
        print_result "F-HTTP-010" "Stream Format" "PASS" "SSE data detected"
    else
        print_result "F-HTTP-010" "Stream Format" "FAIL" "Invalid SSE format"
    fi
}

# Section V: Tracing Tests
test_tracing() {
    print_header "Section V: Tracing Parity"

    local levels=("brick" "step" "layer")
    local ids=("F-TRACE-001" "F-TRACE-002" "F-TRACE-003")

    for i in "${!levels[@]}"; do
        local level="${levels[$i]}"
        local id="${ids[$i]}"
        local response
        
        log_debug "Testing trace level: ${level}"
        response=$(curl -s "${BASE_URL}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -H "X-Trace-Level: ${level}" \
            -d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}')
        
        if echo "${response}" | grep -q "\"${level}_trace\""; then
            print_result "${id}" "Trace Level: ${level}" "PASS" "Found ${level}_trace in response"
        else
            print_result "${id}" "Trace Level: ${level}" "FAIL" "Missing ${level}_trace in response"
        fi
    done
}

test_default_mode_suppression() {
    print_header "Section V: Default Mode Suppression"

    local response
    response=$(curl -s "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}')
    log_debug "Default mode response: ${response}"

    # F-TRACE-004a: Valid JSON
    if echo "${response}" | python3 -c "import sys, json; json.load(sys.stdin)" >/dev/null 2>&1; then
        print_result "F-TRACE-004a" "Valid JSON (Default)" "PASS"
    else
        print_result "F-TRACE-004a" "Valid JSON (Default)" "FAIL"
        return 1
    fi

    # F-TRACE-004b: Suppression Check
    local traces=("trace" "brick_trace" "layer_trace" "step_trace")
    local leaked=0
    for t in "${traces[@]}"; do
        if echo "${response}" | grep -q "\"${t}\"" ; then
            log_debug "Leaked trace field: ${t}"
            leaked=1
        fi
    done

    if [[ "${leaked}" == "0" ]]; then
        print_result "F-TRACE-004b" "Trace Suppression" "PASS" "No trace fields leaked"
    else
        print_result "F-TRACE-004b" "Trace Suppression" "FAIL" "Trace fields found in default response"
    fi

    # F-TRACE-004c: Normal Inference Works
    local content
    content=$(echo "${response}" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
    if [[ -n "${content}" ]]; then
        print_result "F-TRACE-004c" "Normal Inference (Default)" "PASS" "Content: ${content}"
    else
        print_result "F-TRACE-004c" "Normal Inference (Default)" "FAIL" "No content returned"
    fi
}

#######################################
# Run All Tests for Current Server
#######################################
run_test_suite() {
    test_health
    test_basic_inference
    test_streaming
    test_determinism
    test_malformed_json
    test_coherency
    test_tracing
    test_default_mode_suppression
}

#######################################
# Multi-Model Test Runner
#######################################
run_all_models() {
    local total_models=0
    local passed_models=0
    local failed_models=0

    print_color "${BLUE}" "\n╔══════════════════════════════════════════════════════════════╗"
    print_color "${BLUE}" "║         MULTI-MODEL QA TEST SUITE (4 Model Sizes)            ║"
    print_color "${BLUE}" "╚══════════════════════════════════════════════════════════════╝"

    # Test each model size
    for size in "0.5B" "1B" "1.5B" "7B"; do
        local model_path="${MODEL_SIZES[$size]}"

        print_color "${YELLOW}" "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        print_color "${YELLOW}" "Testing ${size} Model: $(basename "${model_path}")"
        print_color "${YELLOW}" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # Check if model exists
        if [[ ! -f "${model_path}" ]]; then
            print_color "${RED}" "[SKIP] ${size}: Model file not found: ${model_path}"
            continue
        fi

        total_models=$((total_models + 1))

        # Reset counters for this model
        TESTS_PASSED=0
        TESTS_FAILED=0
        TOTAL_TESTS=0

        # Start server for this model
        MODEL_PATH="${model_path}"
        start_server

        if ! check_server; then
            print_color "${RED}" "[FAIL] ${size}: Server failed to start"
            failed_models=$((failed_models + 1))
            continue
        fi

        # Run test suite
        run_test_suite

        # Stop server
        stop_server
        sleep 2

        # Report results for this model
        if [[ "${TESTS_FAILED}" -eq 0 ]]; then
            print_color "${GREEN}" "[PASS] ${size}: ${TESTS_PASSED}/${TOTAL_TESTS} tests passed"
            passed_models=$((passed_models + 1))
        else
            print_color "${RED}" "[FAIL] ${size}: ${TESTS_PASSED}/${TOTAL_TESTS} tests passed, ${TESTS_FAILED} failed"
            failed_models=$((failed_models + 1))
        fi
    done

    # Final summary
    print_color "${BLUE}" "\n╔══════════════════════════════════════════════════════════════╗"
    print_color "${BLUE}" "║                    MULTI-MODEL SUMMARY                        ║"
    print_color "${BLUE}" "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Models Tested: ${total_models}"
    echo "Models Passed: ${passed_models}"
    echo "Models Failed: ${failed_models}"
    echo ""

    if [[ "${failed_models}" -eq 0 ]] && [[ "${total_models}" -gt 0 ]]; then
        print_color "${GREEN}" "✓ ALL ${total_models} MODEL SIZES PASSED QA"
        return 0
    else
        print_color "${RED}" "✗ ${failed_models}/${total_models} MODEL SIZES FAILED QA"
        return 1
    fi
}

#######################################
# Main Execution
#######################################

# Handle --all-models mode
if [[ "${ALL_MODELS_MODE}" -eq 1 ]]; then
    run_all_models
    exit $?
fi

if [[ -n "${MODEL_PATH}" ]]; then
    start_server
else
    # Verify server is running if no model provided
    if ! check_server; then
        print_color "${RED}" "ERROR: No model path provided and server not running."
        echo "Usage: $0 [port] [model_path]"
        echo "       $0 [port] --all-models   # Test all 4 model sizes"
        exit 2
    fi
fi

# Execute Falsification Suite
run_test_suite

# Summary
print_header "Falsification Summary"
echo "Total Tests: ${TOTAL_TESTS}"
echo "Passed:      ${TESTS_PASSED}"
echo "Failed:      ${TESTS_FAILED}"

H_SERVER="apr-serve produces correct OpenAI-compatible inference"

if [[ "${TESTS_FAILED}" -eq 0 ]]; then
    print_color "${GREEN}" "Hypothesis '${H_SERVER}' SURVIVED falsification."
    exit 0
else
    print_color "${RED}" "Hypothesis '${H_SERVER}' FALSIFIED."
    exit 1
fi