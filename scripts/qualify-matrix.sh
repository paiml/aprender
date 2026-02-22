#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC2036,SC2086,SC2096,SC2154,SC2199

# qualify-matrix.sh — Run `apr qualify --json` across local models, generate matrix.
#
# Usage:
#   bash scripts/qualify-matrix.sh                          # All 8 models
#   bash scripts/qualify-matrix.sh ~/models/TinyLlama*.gguf # Single model
#
# Prerequisites: apr binary on PATH (cargo install --path crates/apr-cli)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/docs/qualify-results"
MATRIX_FILE="${REPO_ROOT}/docs/qualify-matrix.md"
TIMEOUT=180

# Find apr binary on PATH
if ! command -v apr >/dev/null 2>&1; then
    echo "ERROR: apr not found on PATH. Run:"
    echo "  cargo install --path crates/apr-cli"
    exit 1
fi
APR_BIN="$(command -v apr)"

# Model list (ordered smallest-first for fast feedback)
ALL_MODELS=(
    "${HOME}/models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf"
    "${HOME}/models/qwen2.5-coder-0.5b-instruct/model.safetensors"
    "${HOME}/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    "${HOME}/models/qwen2.5-coder-1.5b-instruct-q4k.apr"
    "${HOME}/models/qwen2.5-coder-1.5b.apr"
    "${HOME}/models/qwen2.5-coder-7b-instruct-q4k.gguf"
    "${HOME}/models/qwen2.5-coder-7b-instruct-exported.gguf"
    "${HOME}/models/qwen2.5-coder-7b-instruct.apr"
)

# Slug derivation: basename without extension, lowercased
slug_for() {
    local path="$1"
    local base
    base="$(basename "${path}")"
    base="${base%.gguf}"
    base="${base%.apr}"
    base="${base%.safetensors}"
    printf '%s' "${base}" | tr '[[:upper:]]' '[[:lower:]]'
}

# Format detection from file extension
format_for() {
    local path="$1"
    case "${path}" in
        *.gguf) printf '%s' "GGUF" ;;
        *.apr)  printf '%s' "APR" ;;
        *.safetensors) printf '%s' "SafeTensors" ;;
        *) printf '%s' "Unknown" ;;
    esac
}

# Human-readable file size
size_for() {
    local path="$1"
    local bytes
    bytes="$(stat --format='%s' "${path}" 2>/dev/null || stat -f '%z' "${path}" 2>/dev/null || printf '0')"
    if (( bytes >= 1073741824 )); then
        printf '%.1fG' "$(bc <<< "scale=1; ${bytes} / 1073741824")"
    else
        printf '%dM' "$(( bytes / 1048576 ))"
    fi
}

# Status emoji mapping
emoji_for() {
    case "$1" in
        PASS)    printf '%s' "✅" ;;
        FAIL)    printf '%s' "❌" ;;
        SKIP)    printf '%s' "⏭️" ;;
        PANIC)   printf '%s' "💥" ;;
        TIMEOUT) printf '%s' "⏰" ;;
        *)       printf '%s' "❓" ;;
    esac
}

# Determine which models to run
if [[ $# -gt 0 ]]; then
    MODELS=("$@")
else
    MODELS=("${ALL_MODELS[@]}")
fi

mkdir -p "${RESULTS_DIR}"

# Run qualify on each model
for model in "${MODELS[@]}"; do
    if [[ ! -f "${model}" ]]; then
        echo "WARN: ${model} not found, skipping"
        continue
    fi

    slug="$(slug_for "${model}")"
    json_out="${RESULTS_DIR}/${slug}.json"
    err_out="${RESULTS_DIR}/${slug}.err"
    echo "==> Qualifying: $(basename "${model}") -> ${slug}.json"

    # JSON goes to stdout; stderr goes to .err file for debugging
    if "${APR_BIN}" qualify "${model}" --json --timeout "${TIMEOUT}" >"${json_out}" 2>"${err_out}"; then
        echo "    PASSED"
    else
        echo "    FAILED (see ${json_out})"
    fi
done

# Generate matrix from all JSON results
echo ""
echo "==> Generating matrix: ${MATRIX_FILE}"

GATE_NAMES=("inspect" "validate" "validate_quality" "tensors" "lint" "debug" "tree" "hex" "flow" "explain" "check")
GATE_HEADERS=("Inspect" "Validate" "Val.Quality" "Tensors" "Lint" "Debug" "Tree" "Hex" "Flow" "Explain" "Check")

# jq filter for extracting gate status by name (uses jq --arg, not shell variable)
# shellcheck disable=SC2016
JQ_GATE_FILTER='.gates[] | select(.name == $n) | .status'

generate_matrix() {
    printf '%s\n' "# APR Qualify Matrix"
    printf '\n'
    printf '%s\n' "Cross-subcommand smoke test results across all local models."
    printf 'Each cell shows whether `apr <subcommand>` completes without crashing on the model file.\n'
    printf '\n'
    printf '**Tool:** `apr qualify` (11-gate cross-subcommand smoke test)\n'
    printf '**Last Updated:** %s\n' "$(date -u '+%Y-%m-%d %H:%M UTC')"
    printf '\n'
    printf '%s\n' "## Legend"
    printf '\n'
    printf '%s\n' "| Symbol | Meaning |"
    printf '%s\n' "|--------|---------|"
    printf '%s\n' "| ✅ | PASS — subcommand completed successfully |"
    printf '%s\n' "| ❌ | FAIL — subcommand returned an error |"
    printf '%s\n' "| ⏭️ | SKIP — gate skipped (e.g. missing feature flag) |"
    printf '%s\n' "| 💥 | PANIC — subcommand panicked (crash bug) |"
    printf '%s\n' "| ⏰ | TIMEOUT — exceeded ${TIMEOUT}s deadline |"
    printf '\n'
    printf '%s\n' "## Results"
    printf '\n'

    # Table header
    printf '%s\n' "<!-- QUALIFY_MATRIX_START -->"
    printf '%s' "| Model | Format | Size |"
    for h in "${GATE_HEADERS[@]}"; do
        printf ' %s |' "${h}"
    done
    printf ' %s\n' "Score | Duration |"

    # Separator
    printf '%s' "|-------|--------|------|"
    for _ in "${GATE_HEADERS[@]}"; do
        printf '%s' "------|"
    done
    printf '%s\n' "-------|----------|"

    # Rows — iterate over ALL models to keep consistent order
    local slug json_file fmt sz display_name
    local pass_count total_count gate_status duration_ms duration_s
    for model in "${ALL_MODELS[@]}"; do
        slug="$(slug_for "${model}")"
        json_file="${RESULTS_DIR}/${slug}.json"

        # Skip missing or empty results
        if [[ ! -f "${json_file}" ]] || [[ ! -s "${json_file}" ]]; then
            continue
        fi
        if ! jq empty "${json_file}" 2>/dev/null; then
            continue
        fi

        fmt="$(format_for "${model}")"
        sz="$(size_for "${model}")"
        display_name="$(basename "${model}")"
        # Truncate long names
        if (( ${#display_name} > 45 )); then
            display_name="${display_name:0:42}..."
        fi

        # Extract gate statuses
        pass_count=0
        total_count=0
        printf '| %s | %s | %s |' "${display_name}" "${fmt}" "${sz}"

        for gate_name in "${GATE_NAMES[@]}"; do
            gate_status="$(jq -r --arg n "${gate_name}" "${JQ_GATE_FILTER}" "${json_file}" 2>/dev/null)" || gate_status="SKIP"
            if [[ -z "${gate_status}" ]]; then
                gate_status="SKIP"
            fi
            printf ' %s |' "$(emoji_for "${gate_status}")"
            total_count=$((total_count+1))
            if [[ "${gate_status}" == "PASS" ]]; then
                pass_count=$((pass_count+1))
            fi
        done

        duration_ms="$(jq -r '.total_duration_ms // 0' "${json_file}" 2>/dev/null)" || duration_ms=0
        if [[ -z "${duration_ms}" ]] || [[ "${duration_ms}" == "null" ]]; then
            duration_ms=0
        fi
        duration_s="$(bc <<< "scale=1; ${duration_ms} / 1000")"

        printf ' %d/%d | %ss |\n' "${pass_count}" "${total_count}" "${duration_s}"
    done

    printf '%s\n' "<!-- QUALIFY_MATRIX_END -->"
    printf '\n'
    printf '%s\n' "## Running"
    printf '\n'
    printf '%s\n' '```bash'
    printf '%s\n' "# Build apr-cli with inference support"
    printf '%s\n' "cargo build -p apr-cli --features inference --release"
    printf '\n'
    printf '%s\n' "# Run all models"
    printf '%s\n' "bash scripts/qualify-matrix.sh"
    printf '\n'
    printf '%s\n' "# Run a single model"
    printf '%s\n' "bash scripts/qualify-matrix.sh ~/models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf"
    printf '%s\n' '```'
    printf '\n'
    printf '%s\n' "## Per-Model JSON"
    printf '\n'
    printf 'Raw JSON results are saved in [`docs/qualify-results/`](qualify-results/).\n'
    printf 'Each file contains the full gate-by-gate breakdown from `apr qualify --json`.\n'
}

generate_matrix > "${MATRIX_FILE}"

echo "==> Done. Results in ${MATRIX_FILE}"
