#!/usr/bin/env bash
# scripts/verify-chat-models.sh
# Verified with: bashrs lint && bashrs purify
#
# Model Chat Template Verification Matrix
# Toyota Way: Genchi Genbutsu - test with real models

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly APR_BIN="${PROJECT_ROOT}/target/release/apr"
readonly MODELS_DIR="${HOME}/.apr/models"
readonly RESULTS_FILE="${PROJECT_ROOT}/target/model-verification-results.json"
readonly README_FILE="${PROJECT_ROOT}/README.md"

# Test prompt (simple, deterministic)
readonly TEST_PROMPT="What is 2+2?"
readonly MAX_TOKENS=16
readonly TIMEOUT_SECS=30

# ============================================================================
# Model Registry (Source of Truth)
# ============================================================================

declare -A MODEL_URLS=(
    ["tinyllama"]="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    ["mistral"]="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    ["openhermes"]="https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q4_K_M.gguf"
    ["vicuna"]="https://huggingface.co/TheBloke/vicuna-7B-v1.5-GGUF/resolve/main/vicuna-7b-v1.5.Q4_K_M.gguf"
    ["yi"]="https://huggingface.co/TheBloke/Yi-6B-Chat-GGUF/resolve/main/yi-6b-chat.Q4_K_M.gguf"
)

declare -A MODEL_FILES=(
    ["tinyllama"]="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    ["mistral"]="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    ["openhermes"]="openhermes-2.5-mistral-7b.Q4_K_M.gguf"
    ["vicuna"]="vicuna-7b-v1.5.Q4_K_M.gguf"
    ["yi"]="yi-6b-chat.Q4_K_M.gguf"
)

declare -A MODEL_TEMPLATES=(
    ["tinyllama"]="llama2"
    ["mistral"]="mistral"
    ["openhermes"]="chatml"
    ["vicuna"]="llama2"
    ["yi"]="chatml"
)

declare -A MODEL_SIZES=(
    ["tinyllama"]="638M"
    ["mistral"]="4.1G"
    ["openhermes"]="4.1G"
    ["vicuna"]="4.1G"
    ["yi"]="3.5G"
)

# ============================================================================
# Utility Functions
# ============================================================================

log_info() {
    printf "\033[0;34m[INFO]\033[0m %s\n" "$1"
}

log_success() {
    printf "\033[0;32m[PASS]\033[0m %s\n" "$1"
}

log_failure() {
    printf "\033[0;31m[FAIL]\033[0m %s\n" "$1"
}

log_skip() {
    printf "\033[0;33m[SKIP]\033[0m %s\n" "$1"
}

# ============================================================================
# Verification Functions
# ============================================================================

verify_apr_binary() {
    if [[ ! -x "${APR_BIN}" ]]; then
        log_info "Building apr-cli with inference feature..."
        cargo build --release -p apr-cli --features inference \
            --manifest-path "${PROJECT_ROOT}/Cargo.toml"
    fi

    if [[ ! -x "${APR_BIN}" ]]; then
        log_failure "apr binary not found at ${APR_BIN}"
        return 1
    fi

    log_success "apr binary ready: ${APR_BIN}"
}

download_model() {
    local model_key="$1"
    local url="${MODEL_URLS[$model_key]}"
    local filename="${MODEL_FILES[$model_key]}"
    local filepath="${MODELS_DIR}/${filename}"

    if [[ -f "${filepath}" ]]; then
        log_info "Model already cached: ${filename}"
        return 0
    fi

    log_info "Downloading ${model_key} (${MODEL_SIZES[$model_key]})..."
    mkdir -p "${MODELS_DIR}"

    if curl -L -o "${filepath}" "${url}" 2>/dev/null; then
        log_success "Downloaded: ${filename}"
        return 0
    else
        log_failure "Failed to download: ${filename}"
        return 1
    fi
}

verify_model() {
    local model_key="$1"
    local filename="${MODEL_FILES[$model_key]}"
    local filepath="${MODELS_DIR}/${filename}"
    local template="${MODEL_TEMPLATES[$model_key]}"

    if [[ ! -f "${filepath}" ]]; then
        log_skip "${model_key}: model file not present"
        echo "skip"
        return 0
    fi

    log_info "Testing ${model_key} (template: ${template})..."

    # Run chat with timeout and capture output
    local output
    local exit_code=0

    output=$(echo -e "${TEST_PROMPT}\n/quit" | \
        timeout "${TIMEOUT_SECS}" "${APR_BIN}" chat "${filepath}" \
            --max-tokens "${MAX_TOKENS}" \
            --temperature 0.1 2>&1) || exit_code=$?

    # Check for success indicators
    if [[ ${exit_code} -eq 0 ]] && \
       echo "${output}" | grep -qiE "(loaded|format|model)" && \
       echo "${output}" | grep -qiE "(goodbye|quit|exit|4)"; then
        log_success "${model_key}: chat completed successfully"
        echo "pass"
        return 0
    else
        log_failure "${model_key}: chat failed (exit=${exit_code})"
        echo "fail"
        return 1
    fi
}

# ============================================================================
# Rust Unit Test Verification
# ============================================================================

verify_chat_template_tests() {
    log_info "Running chat_template unit tests..."

    if cargo test --lib chat_template --manifest-path "${PROJECT_ROOT}/Cargo.toml" 2>&1; then
        log_success "All chat_template tests passed"
        return 0
    else
        log_failure "Some chat_template tests failed"
        return 1
    fi
}

# ============================================================================
# Matrix Generation
# ============================================================================

generate_results_json() {
    local results_json="{"
    results_json+='"timestamp": "'"$(date -Iseconds)"'",'
    results_json+='"models": {'

    local first=true
    for model_key in "${!MODEL_FILES[@]}"; do
        local result
        result=$(verify_model "${model_key}")

        if [[ "${first}" == "true" ]]; then
            first=false
        else
            results_json+=','
        fi

        results_json+='"'"${model_key}"'": {'
        results_json+='"file": "'"${MODEL_FILES[$model_key]}"'",'
        results_json+='"template": "'"${MODEL_TEMPLATES[$model_key]}"'",'
        results_json+='"size": "'"${MODEL_SIZES[$model_key]}"'",'
        results_json+='"status": "'"${result}"'"'
        results_json+='}'
    done

    results_json+='}}'

    mkdir -p "$(dirname "${RESULTS_FILE}")"
    echo "${results_json}" > "${RESULTS_FILE}"
    log_success "Results written to: ${RESULTS_FILE}"
}

update_readme_matrix() {
    if [[ ! -f "${RESULTS_FILE}" ]]; then
        log_failure "Results file not found: ${RESULTS_FILE}"
        return 1
    fi

    log_info "Updating README.md model matrix..."

    # Generate markdown table from results
    local table="| Model | Format | Template | Size | Status |\n"
    table+="|-------|--------|----------|------|--------|\n"

    for model_key in tinyllama mistral openhermes vicuna yi; do
        local status
        if command -v jq >/dev/null 2>&1; then
            status=$(jq -r ".models.${model_key}.status // \"unknown\"" "${RESULTS_FILE}")
        else
            # Fallback without jq
            status="unknown"
        fi
        local template="${MODEL_TEMPLATES[$model_key]}"
        local size="${MODEL_SIZES[$model_key]}"

        local status_emoji
        case "${status}" in
            pass) status_emoji="Pass" ;;
            fail) status_emoji="Fail" ;;
            skip) status_emoji="Skip" ;;
            *)    status_emoji="Unknown" ;;
        esac

        table+="| ${model_key} | GGUF | ${template} | ${size} | ${status_emoji} |\n"
    done

    # Update README between markers (if markers exist)
    if grep -q "<!-- MODEL_MATRIX_START -->" "${README_FILE}"; then
        sed -i '/<!-- MODEL_MATRIX_START -->/,/<!-- MODEL_MATRIX_END -->/c\<!-- MODEL_MATRIX_START -->\n'"$(echo -e "${table}")"'\n<!-- MODEL_MATRIX_END -->' "${README_FILE}"
        log_success "README.md matrix updated"
    else
        log_info "README.md markers not found, skipping update"
    fi
}

# ============================================================================
# Main Entry Point
# ============================================================================

main() {
    local cmd="${1:-verify}"

    case "${cmd}" in
        verify)
            verify_apr_binary
            generate_results_json
            ;;
        download)
            local model="${2:-all}"
            if [[ "${model}" == "all" ]]; then
                for key in "${!MODEL_URLS[@]}"; do
                    download_model "${key}" || true
                done
            else
                download_model "${model}"
            fi
            ;;
        update-readme)
            update_readme_matrix
            ;;
        test)
            verify_chat_template_tests
            ;;
        full)
            verify_apr_binary
            for key in tinyllama; do  # Start with smallest
                download_model "${key}" || true
            done
            generate_results_json
            update_readme_matrix
            ;;
        *)
            echo "Usage: $0 {verify|download [model]|update-readme|test|full}"
            echo ""
            echo "Commands:"
            echo "  verify        Build apr and run model verification"
            echo "  download      Download model(s) (all or specific: tinyllama, mistral, etc.)"
            echo "  update-readme Update README.md with verification results"
            echo "  test          Run chat_template Rust unit tests"
            echo "  full          Download tinyllama, verify, and update README"
            exit 1
            ;;
    esac
}

main "$@"
