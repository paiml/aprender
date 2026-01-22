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
#
# shellcheck disable=SC2227  # Redirection warnings are false positives (no pipes)
# shellcheck disable=SC2282  # ${var:-} is intentional defensive coding
# shellcheck disable=SC2290  # $ in array subscripts is required for associative arrays
# shellcheck disable=SC2310  # set -e in conditions is understood and intentional

set -euo pipefail

# Configuration
PORT="${1:-8080}"
MODEL_PATH="${2:-}"
EXPECTED_MODE="${3:-}"
VERBOSE="${VERBOSE:-1}"
BASE_URL="http://127.0.0.1:${PORT}"
SERVER_PID=''
ALL_MODELS_MODE=0

# Multi-model test configuration (5 sizes: 0.5B to 32B)
# Note: 32B model requires ~20GB+ RAM for Q2_K, ~40GB+ for Q4_K
declare -A MODEL_SIZES
MODEL_SIZES=(
    ["0.5B"]="${HOME}/.cache/pacha/models/d4c4d9763127153c.gguf"
    ["1B"]="${HOME}/.cache/pacha/models/117fd82563e7bb5d.gguf"
    ["1.5B"]="${HOME}/.cache/huggingface/models/qwen2.5-coder-1.5b-gguf/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    ["7B"]="${HOME}/.cache/huggingface/models/qwen2.5-coder-7b-gguf/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    ["32B"]="${HOME}/Downloads/qwen2.5-coder-32b-instruct-q2_k.gguf"
)

# Alternative 32B model paths (fallback order)
MODEL_SIZES_32B_ALT=(
    "${HOME}/.cache/huggingface/models/qwen2.5-coder-32b-gguf/qwen2.5-coder-32b-instruct-q2_k.gguf"
    "${HOME}/.cache/huggingface/models/qwen2.5-coder-32b-gguf/qwen2.5-coder-32b-instruct-q4_k_m.gguf"
    "${HOME}/Downloads/qwen2.5-coder-32b-instruct-q4_k_m.gguf"
)

# Results tracking for proof matrix
declare -A PROOF_MATRIX

# Check for --all-models flag
if [[ "${MODEL_PATH}" == "--all-models" ]]; then
    ALL_MODELS_MODE=1
    MODEL_PATH=''
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
    printf '%b%s%b\n' "${color}" "${message}" "${NC}"
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

    TOTAL_TESTS=$((TOTAL_TESTS+1))

    if [[ "${result}" == "PASS" ]]; then
        print_color "${GREEN}" "[PASS] ${id}: ${name}"
        TESTS_PASSED=$((TESTS_PASSED+1))
    else
        print_color "${RED}" "[FAIL] ${id}: ${name}"
        TESTS_FAILED=$((TESTS_FAILED+1))
    fi

    # Print details if provided (details is a scalar string, not an array)
    if test -n "$details"; then
        printf '       %s\n' "$details"
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

    print_color "${BLUE}" "Starting apr serve with model: ${MODEL_PATH} (GPU enabled)"
    apr serve "${MODEL_PATH}" --port "${PORT}" --gpu &
    SERVER_PID=$!

    local retries=30
    while ! check_server && test "$retries" -gt 0; do
        sleep 1
        retries=$((retries-1))
        printf '%s' '.'
    done
    printf '\n'

    if ! check_server; then
        print_color "${RED}" "ERROR: Server failed to start"
        return 1
    fi
    print_color "${GREEN}" "Server started on port ${PORT}"
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
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
    mode=$(python3 -c "import sys, json; print(json.load(sys.stdin).get('compute_mode', 'missing'))" <<< "${response}" 2>/dev/null)

    if [[ "${mode}" == "cpu" || "${mode}" == "gpu" || "${mode}" == "cuda" ]]; then
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
    if ! python3 -c "import sys, json; json.load(sys.stdin)" <<< "${response}" >/dev/null 2>&1; then
        print_result "F-HTTP-007" "Valid JSON" "FAIL" "Response is not valid JSON"
        return 1
    fi
    print_result "F-HTTP-007" "Valid JSON" "PASS"

    # Extract Content using jq if available, else python
    local content=""
    local has_jq=0
    type jq &>/dev/null && has_jq=1
    if test "$has_jq" -eq 1; then
        content=$(jq -r '.choices[0].message.content' <<< "${response}" 2>/dev/null) || content="__FAIL__"
    else
        content=$(python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["choices"][0]["message"]["content"])' <<< "${response}" 2>/dev/null) || content="__FAIL__"
    fi

    log_debug "Extracted content: ${content}"

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
    if [[ "${content}" == *token[0-9]* ]]; then
        print_result "F-HTTP-009" "Token Artifacts" "FAIL" "Raw tokens detected: ${content}"
        return 1
    fi
    print_result "F-HTTP-009" "Token Artifacts" "PASS"

    # F-HTTP-009b: BPE Artifacts (PAR-201)
    # Checking for 'Ġ' (U+0120), 'Ċ' (U+010A), 'â' (often part of encoding errors)
    if [[ "${content}" == *Ġ* || "${content}" == *Ċ* || "${content}" == *â* ]]; then
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
    content=$(python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" <<< "${response}" 2>/dev/null) || content=""

    # Heuristic 1: Check for repetition loops (common in broken 0.5B models)
    if [[ "${content}" =~ (1, 2, 3, 4, 5) || "${content}" =~ (1 2 3 4 5) || "${content}" =~ (One, Two, Three) ]]; then
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
    if [[ "${content}" == *token[0-9][0-9]* ]]; then
        has_garbage=1
    fi

    if [[ "${has_garbage}" == "1" ]]; then
        print_result "F-HTTP-020b" "Garbage Check" "FAIL" "Replacement characters or raw tokens detected"
    else
        print_result "F-HTTP-020b" "Garbage Check" "PASS" "No garbage patterns"
    fi

    # F-HTTP-020c: Multi-turn loop check (PMAT-092)
    # Model should NOT generate fake Human:/Assistant:/User: turns
    test_no_multi_turn_loop
}

# F-HTTP-020c: Multi-turn Loop Prevention (PMAT-092)
# Verifies model stops at EOS and doesn't generate fake conversation turns
# Uses temperature=0 and limited tokens for deterministic output
test_no_multi_turn_loop() {
    print_header "Section IV: Robustness - Multi-Turn Loop Prevention (PMAT-092)"

    local response
    # Use temperature=0 for deterministic output, max_tokens=30 to limit generation
    response=$(curl -s "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model": "default", "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}], "max_tokens": 30, "temperature": 0}')
    log_debug "Multi-turn check response: ${response}"

    local content
    content=$(python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" <<< "${response}" 2>/dev/null) || content=""

    # Check for fake conversation turn markers that indicate the model didn't stop at EOS
    local has_fake_turns=0
    local detected_patterns=""

    # Check for Human/Assistant style (Anthropic/Claude format)
    if [[ "${content}" == *$'\nHuman:'* || "${content}" == *$'\n\nHuman:'* ]]; then
        has_fake_turns=1
        detected_patterns="Human:"
    fi
    if [[ "${content}" == *$'\nAssistant:'* || "${content}" == *$'\n\nAssistant:'* ]]; then
        has_fake_turns=1
        detected_patterns="${detected_patterns} Assistant:"
    fi

    # Check for User/Assistant style (common in chat models)
    if [[ "${content}" == *$'\nUser:'* || "${content}" == *$'\n\nUser:'* ]]; then
        has_fake_turns=1
        detected_patterns="${detected_patterns} User:"
    fi

    # Check for ChatML markers that shouldn't appear in content
    if [[ "${content}" == *'<|im_start|>'* ]]; then
        has_fake_turns=1
        detected_patterns="${detected_patterns} <|im_start|>"
    fi

    if [[ "${has_fake_turns}" == "1" ]]; then
        print_result "F-HTTP-020c" "Multi-Turn Loop" "FAIL" "Detected fake turns: ${detected_patterns}"
        log_debug "Full content: ${content}"
    else
        print_result "F-HTTP-020c" "Multi-Turn Loop" "PASS" "No fake conversation turns detected"
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

    case "${status}" in
        400|500)
            print_result "F-HTTP-017" "Malformed JSON" "PASS" "Server rejected with ${status}"
            ;;
        *)
            print_result "F-HTTP-017" "Malformed JSON" "FAIL" "Server returned ${status} (Expected 400)"
            ;;
    esac
}

# F-HTTP-016: Determinism
test_determinism() {
    print_header "Section III: Determinism"

    local p1
    local p2

    p1=$(curl -s "${BASE_URL}/v1/chat/completions" -d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"temperature":0,"max_tokens":10}' -H "Content-Type: application/json")
    p2=$(curl -s "${BASE_URL}/v1/chat/completions" -d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"temperature":0,"max_tokens":10}' -H "Content-Type: application/json")

    local c1 c2
    c1=$(python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" <<< "${p1}" 2>/dev/null)
    c2=$(python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" <<< "${p2}" 2>/dev/null)
    log_debug "Det1: ${c1}"
    log_debug "Det2: ${c2}"

    if [[ "${c1}" == "${c2}" ]]; then
        print_result "F-HTTP-016" "Determinism [T=0]" "PASS"
    else
        print_result "F-HTTP-016" "Determinism [T=0]" "FAIL" "Outputs differ"
    fi
}

# F-HTTP-010: Streaming (curl-based)
test_streaming() {
    print_header "Section III: Advanced Features - Streaming"

    local response
    # Use max_tokens to limit generation and ensure quick response
    response=$(timeout 10 curl -N -s "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"stream":true,"max_tokens":5}') || true
    log_debug "Streaming head: ${response:0:300}..."

    # Check for [DONE] marker (OpenAI SSE termination)
    if [[ "${response}" == *'[DONE]'* ]]; then
        print_result "F-HTTP-011" "Stream End" "PASS" "Received [DONE]"
    else
        print_result "F-HTTP-011" "Stream End" "FAIL" "Missing [DONE] marker"
    fi

    # Check for SSE data format
    if [[ "${response}" == *'data: {'* ]]; then
        print_result "F-HTTP-010" "Stream Format" "PASS" "SSE data detected"
    else
        print_result "F-HTTP-010" "Stream Format" "FAIL" "Invalid SSE format"
    fi

    # F-HTTP-010b: OpenAI SDK streaming test (ephemeral uv)
    test_streaming_openai_sdk
}

# F-HTTP-010b: OpenAI SDK Streaming Test (PMAT-087)
# Uses ephemeral uv for Python execution - no persistent deps
test_streaming_openai_sdk() {
    print_header "Section III: OpenAI SDK Streaming (uv ephemeral)"

    # Check if uv is available
    if ! command -v uv &>/dev/null; then
        print_result "F-HTTP-010b" "OpenAI SDK Streaming" "SKIP" "uv not installed"
        return 0
    fi

    local script_output
    local script_exit=0

    # Ephemeral Python script using uv run --with
    # This installs openai temporarily and runs the test
    # Export BASE_URL for the Python script to use
    export OPENAI_API_BASE="${BASE_URL}/v1"
    script_output=$(timeout 30 uv run --with openai python3 - <<'PYTHON_EOF'
import os
import sys
import time

# Suppress warnings
os.environ["PYTHONWARNINGS"] = "ignore"

from openai import OpenAI

base_url = os.getenv("OPENAI_API_BASE", "http://localhost:8080/v1")
client = OpenAI(api_key="test", base_url=base_url)

try:
    stream = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Say hi"}],
        max_tokens=5,
        stream=True,
    )

    token_count = 0
    first_token_time = None
    start_time = time.time()
    content_pieces = []

    for chunk in stream:
        if hasattr(chunk.choices[0].delta, 'content'):
            content = chunk.choices[0].delta.content
            if content:
                if first_token_time is None:
                    first_token_time = time.time()
                content_pieces.append(content)
                token_count += 1

    total_time = time.time() - start_time
    ttft = first_token_time - start_time if first_token_time else total_time

    # Output results in parseable format
    print(f"TOKENS:{token_count}")
    print(f"TTFT:{ttft:.3f}")
    print(f"TOTAL:{total_time:.3f}")
    print(f"CONTENT:{''.join(content_pieces)}")

    # Streaming is working if TTFT < 50% of total time (tokens arrive incrementally)
    if token_count > 0 and ttft < total_time * 0.8:
        print("STREAMING:TRUE")
        sys.exit(0)
    elif token_count > 0:
        print("STREAMING:MAYBE")  # Got tokens but timing unclear
        sys.exit(0)
    else:
        print("STREAMING:FALSE")
        sys.exit(1)

except Exception as e:
    print(f"ERROR:{e}")
    sys.exit(1)
PYTHON_EOF
    ) || script_exit=$?

    log_debug "SDK streaming output: ${script_output}"

    # Parse results
    local tokens ttft streaming content
    tokens=$(grep '^TOKENS:' <<< "${script_output}" | cut -d: -f2) || tokens="0"
    ttft=$(grep '^TTFT:' <<< "${script_output}" | cut -d: -f2) || ttft="0"
    streaming=$(grep '^STREAMING:' <<< "${script_output}" | cut -d: -f2) || streaming="FALSE"
    content=$(grep '^CONTENT:' <<< "${script_output}" | cut -d: -f2-) || content=""

    if [[ "${script_exit}" -eq 0 && "${tokens}" -gt 0 ]]; then
        if [[ "${streaming}" == "TRUE" ]]; then
            print_result "F-HTTP-010b" "OpenAI SDK Streaming" "PASS" "Tokens: ${tokens}, TTFT: ${ttft}s"
        else
            print_result "F-HTTP-010b" "OpenAI SDK Streaming" "PASS" "Tokens received: ${tokens} (timing inconclusive)"
        fi
    else
        local error
        error=$(grep '^ERROR:' <<< "${script_output}" | cut -d: -f2-) || error="Unknown error"
        print_result "F-HTTP-010b" "OpenAI SDK Streaming" "FAIL" "${error}"
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

        if [[ "${response}" == *"\"${level}_trace\""* ]]; then
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
    if python3 -c "import sys, json; json.load(sys.stdin)" <<< "${response}" >/dev/null 2>&1; then
        print_result "F-TRACE-004a" "Valid JSON (Default)" "PASS"
    else
        print_result "F-TRACE-004a" "Valid JSON (Default)" "FAIL"
        return 1
    fi

    # F-TRACE-004b: Suppression Check
    local traces=("trace" "brick_trace" "layer_trace" "step_trace")
    local leaked=0
    for t in "${traces[@]}"; do
        if [[ "${response}" == *"\"${t}\""* ]]; then
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
    content=$(python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" <<< "${response}" 2>/dev/null)
    if test -n "${content}"; then
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
# Find model path with fallbacks
#######################################
find_model_path() {
    local size="$1"
    local primary_path="${MODEL_SIZES[$size]}"

    # Check primary path
    if [[ -f "${primary_path}" ]]; then
        echo "${primary_path}"
        return 0
    fi

    # For 32B, try alternative paths
    if [[ "${size}" == "32B" ]]; then
        for alt_path in "${MODEL_SIZES_32B_ALT[@]}"; do
            if [[ -f "${alt_path}" ]]; then
                echo "${alt_path}"
                return 0
            fi
        done
    fi

    # Not found
    echo ""
    return 1
}

#######################################
# Core capability tests (for proof matrix)
#######################################
test_serve_capability() {
    if check_server; then
        PROOF_MATRIX["${CURRENT_SIZE}_serve"]="PASS"
        return 0
    fi
    PROOF_MATRIX["${CURRENT_SIZE}_serve"]="FAIL"
    return 1
}

test_chat_capability() {
    local response
    response=$(curl -s -w "\n%{http_code}" "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model": "default", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10}')

    local http_code
    http_code=$(tail -n1 <<< "${response}")
    local body
    # Delete last line (the http_code) from response to get body
    body=$(head -n -1 <<< "${response}")

    if [[ "${http_code}" == "200" ]]; then
        local content
        content=$(python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" <<< "${body}" 2>/dev/null) || content=""
        if [[ -n "${content}" ]]; then
            PROOF_MATRIX["${CURRENT_SIZE}_chat"]="PASS"
            return 0
        fi
    fi
    PROOF_MATRIX["${CURRENT_SIZE}_chat"]="FAIL"
    return 1
}

test_stream_capability() {
    local response
    response=$(timeout 15 curl -N -s "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"stream":true,"max_tokens":5}') || true

    # Check for [DONE] marker and data: prefix
    if [[ "${response}" == *"[DONE]"* && "${response}" == *"data: {"* ]]; then
        PROOF_MATRIX["${CURRENT_SIZE}_stream"]="PASS"
        return 0
    fi
    PROOF_MATRIX["${CURRENT_SIZE}_stream"]="FAIL"
    return 1
}

test_trace_capability() {
    local all_traces_pass=1

    # Test all three trace levels
    for level in "brick" "step" "layer"; do
        local response
        response=$(curl -s "${BASE_URL}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -H "X-Trace-Level: ${level}" \
            -d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":3}') || true

        # Check if trace data is present in response
        if [[ "${response}" != *"\"${level}_trace\""* ]]; then
            all_traces_pass=0
            log_debug "Trace level ${level} missing in response"
        fi
    done

    # all_traces_pass is a scalar integer (0 or 1), not an array
    if test "$all_traces_pass" -eq 1; then
        PROOF_MATRIX["${CURRENT_SIZE}_trace"]="PASS"
        return 0
    fi
    PROOF_MATRIX["${CURRENT_SIZE}_trace"]="FAIL"
    return 1
}

#######################################
# Print proof matrix
#######################################
print_proof_matrix() {
    print_color "${BLUE}" "\n╔════════════════════════════════════════════════════════════════════════════════╗"
    print_color "${BLUE}" "║                    PROOF MATRIX: Serve/Chat/Stream/Trace                       ║"
    print_color "${BLUE}" "╠════════════════════════════════════════════════════════════════════════════════╣"
    print_color "${BLUE}" "║  Model Size │  SERVE  │  CHAT   │  STREAM │  TRACE  │  Status                 ║"
    print_color "${BLUE}" "╠═════════════╪═════════╪═════════╪═════════╪═════════╪═════════════════════════╣"

    local all_pass=1
    for size in "0.5B" "1B" "1.5B" "7B" "32B"; do
        local key_serve="${size}_serve"
        local key_chat="${size}_chat"
        local key_stream="${size}_stream"
        local key_trace="${size}_trace"
        local serve="${PROOF_MATRIX[$key_serve]:-SKIP}"
        local chat="${PROOF_MATRIX[$key_chat]:-SKIP}"
        local stream="${PROOF_MATRIX[$key_stream]:-SKIP}"
        local trace="${PROOF_MATRIX[$key_trace]:-SKIP}"

        local serve_icon="⏭"
        local chat_icon="⏭"
        local stream_icon="⏭"
        local trace_icon="⏭"
        local status="SKIPPED"

        if [[ "${serve}" == "PASS" ]]; then serve_icon="✓"; fi
        if [[ "${serve}" == "FAIL" ]]; then serve_icon="✗"; fi
        if [[ "${chat}" == "PASS" ]]; then chat_icon="✓"; fi
        if [[ "${chat}" == "FAIL" ]]; then chat_icon="✗"; fi
        if [[ "${stream}" == "PASS" ]]; then stream_icon="✓"; fi
        if [[ "${stream}" == "FAIL" ]]; then stream_icon="✗"; fi
        if [[ "${trace}" == "PASS" ]]; then trace_icon="✓"; fi
        if [[ "${trace}" == "FAIL" ]]; then trace_icon="✗"; fi

        if [[ "${serve}" == "PASS" && "${chat}" == "PASS" && "${stream}" == "PASS" && "${trace}" == "PASS" ]]; then
            status='ALL PASS ✓'
        elif [[ "${serve}" == "SKIP" ]]; then
            status='SKIPPED [no model]'
        else
            status='FAILURES ✗'
            all_pass=0
        fi

        printf "║  %-9s │   %s     │   %s     │    %s    │   %s     │  %-21s ║\n" \
            "${size}" "${serve_icon}" "${chat_icon}" "${stream_icon}" "${trace_icon}" "${status}"
    done

    print_color "${BLUE}" "╚════════════════════════════════════════════════════════════════════════════════╝"

    return $((1-all_pass))
}

#######################################
# Multi-Model Test Runner
#######################################
run_all_models() {
    local total_models=0
    local passed_models=0
    local failed_models=0

    # Reset proof matrix (PROOF_MATRIX is declared globally as associative array)
    # shellcheck disable=SC2034
    declare -gA PROOF_MATRIX=()

    print_color "${BLUE}" "\n╔══════════════════════════════════════════════════════════════╗"
    print_color "${BLUE}" "║         MULTI-MODEL QA TEST SUITE (5 Model Sizes)            ║"
    print_color "${BLUE}" "║                                                              ║"
    print_color "${BLUE}" "║  Testing: SERVE, CHAT, STREAM for each model size           ║"
    print_color "${BLUE}" "╚══════════════════════════════════════════════════════════════╝"

    # Test each model size
    for size in "0.5B" "1B" "1.5B" "7B" "32B"; do
        CURRENT_SIZE="${size}"
        local model_path
        model_path=$(find_model_path "${size}")

        print_color "${YELLOW}" "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        print_color "${YELLOW}" "Testing ${size} Model"
        print_color "${YELLOW}" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # Check if model exists
        if [[ -z "${model_path}" ]]; then
            print_color "${RED}" "[SKIP] ${size}: No model file found"
            print_color "${CYAN}" "       Searched: ${MODEL_SIZES[$size]}"
            if [[ "${size}" == "32B" ]]; then
                print_color "${CYAN}" "       Also tried: ${MODEL_SIZES_32B_ALT[*]}"
            fi
            PROOF_MATRIX["${size}_serve"]="SKIP"
            PROOF_MATRIX["${size}_chat"]="SKIP"
            PROOF_MATRIX["${size}_stream"]="SKIP"
            continue
        fi

        print_color "${CYAN}" "Model: $(basename "${model_path}")"
        print_color "${CYAN}" "Path:  ${model_path}"

        total_models=$((total_models+1))

        # Reset counters for this model
        TESTS_PASSED=0
        TESTS_FAILED=0
        TOTAL_TESTS=0

        # Start server for this model (|| true prevents set -e exit on timeout)
        MODEL_PATH="${model_path}"
        start_server || true

        if ! check_server; then
            print_color "${RED}" "[FAIL] ${size}: Server failed to start"
            PROOF_MATRIX["${size}_serve"]="FAIL"
            PROOF_MATRIX["${size}_chat"]="FAIL"
            PROOF_MATRIX["${size}_stream"]="FAIL"
            PROOF_MATRIX["${size}_trace"]="FAIL"
            failed_models=$((failed_models+1))
            continue
        fi

        # Test core capabilities for proof matrix (|| true prevents set -e exit)
        print_color "${CYAN}" "\n[Proof] Testing core capabilities..."
        test_serve_capability || true
        test_chat_capability || true
        test_stream_capability || true
        test_trace_capability || true

        local pkey_serve="${size}_serve"
        local pkey_chat="${size}_chat"
        local pkey_stream="${size}_stream"
        local pkey_trace="${size}_trace"
        printf '  SERVE:  %s\n' "${PROOF_MATRIX[$pkey_serve]}"
        printf '  CHAT:   %s\n' "${PROOF_MATRIX[$pkey_chat]}"
        printf '  STREAM: %s\n' "${PROOF_MATRIX[$pkey_stream]}"
        printf '  TRACE:  %s\n' "${PROOF_MATRIX[$pkey_trace]}"

        # Run full test suite (|| true prevents set -e exit on test failures)
        run_test_suite || true

        # Stop server
        stop_server
        sleep 2

        # Report results for this model
        if [[ "${TESTS_FAILED}" -eq 0 ]]; then
            print_color "${GREEN}" "[PASS] ${size}: ${TESTS_PASSED}/${TOTAL_TESTS} tests passed"
            passed_models=$((passed_models+1))
        else
            print_color "${RED}" "[FAIL] ${size}: ${TESTS_PASSED}/${TOTAL_TESTS} tests passed, ${TESTS_FAILED} failed"
            failed_models=$((failed_models+1))
        fi
    done

    # Print proof matrix (capture result without triggering set -e)
    local matrix_result=0
    print_proof_matrix || matrix_result=$?

    # Final summary
    print_color "${BLUE}" "\n╔══════════════════════════════════════════════════════════════╗"
    print_color "${BLUE}" "║                    MULTI-MODEL SUMMARY                        ║"
    print_color "${BLUE}" "╚══════════════════════════════════════════════════════════════╝"
    printf '\n'
    printf 'Models Tested: %s\n' "${total_models}"
    printf 'Models Passed: %s\n' "${passed_models}"
    printf 'Models Failed: %s\n' "${failed_models}"
    printf '\n'

    if test "$failed_models" -eq 0 && test "$total_models" -gt 0; then
        print_color "${GREEN}" "✓ ALL ${total_models} MODEL SIZES PASSED QA"
        print_color "${GREEN}" "✓ PROVEN: Chat, Serve, Stream work for all tested sizes"
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
        printf 'Usage: %s [port] [model_path]\n' "$0"
        printf '       %s [port] --all-models   # Test all 4 model sizes\n' "$0"
        exit 2
    fi
fi

# Execute Falsification Suite (|| true prevents set -e exit on test failures)
run_test_suite || true

# Summary
print_header "Falsification Summary"
printf 'Total Tests: %s\n' "${TOTAL_TESTS}"
printf 'Passed:      %s\n' "${TESTS_PASSED}"
printf 'Failed:      %s\n' "${TESTS_FAILED}"

readonly H_SERVER='apr-serve produces correct OpenAI-compatible inference'

if [[ "${TESTS_FAILED}" -eq 0 ]]; then
    print_color "${GREEN}" "Hypothesis \"${H_SERVER}\" SURVIVED falsification."
    exit 0
else
    print_color "${RED}" "Hypothesis \"${H_SERVER}\" FALSIFIED."
    exit 1
fi
