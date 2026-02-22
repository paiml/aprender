#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC2036,SC2086,SC2096,SC2154,SC2199

# qualify-matrix.sh — Run `apr qualify --json` across local models, generate matrix.
#
# Usage:
#   bash scripts/qualify-matrix.sh                          # All local + cached models
#   bash scripts/qualify-matrix.sh ~/models/TinyLlama*.gguf # Single model
#   bash scripts/qualify-matrix.sh --cached-only             # Only cached models
#
# Prerequisites: apr binary on PATH (cargo install --path crates/apr-cli)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/docs/qualify-results"
MATRIX_FILE="${REPO_ROOT}/docs/qualify-matrix.md"
TIMEOUT=180
CACHED_ONLY=false

# Find apr binary on PATH
if ! command -v apr >/dev/null 2>&1; then
    echo "ERROR: apr not found on PATH. Run:"
    echo "  cargo install --path crates/apr-cli"
    exit 1
fi
APR_BIN="$(command -v apr)"

# Local models (ordered smallest-first)
LOCAL_MODELS=(
    "${HOME}/models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf"
    "${HOME}/models/qwen2.5-coder-0.5b-instruct/model.safetensors"
    "${HOME}/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    "${HOME}/models/qwen2.5-coder-1.5b-instruct-q4k.apr"
    "${HOME}/models/qwen2.5-coder-1.5b.apr"
    "${HOME}/models/qwen2.5-coder-7b-instruct-q4k.gguf"
    "${HOME}/models/qwen2.5-coder-7b-instruct-exported.gguf"
    "${HOME}/models/qwen2.5-coder-7b-instruct.apr"
)

# Slug derivation: use NAME_MAP display name if available, else basename
slug_for() {
    local path="$1"
    local mapped="${NAME_MAP[$path]:-}"
    local base
    if [[ -n "${mapped}" ]]; then
        # Convert display name to slug: org/repo:file -> org-repo-file
        base="${mapped}"
        base="${base//\//-}"
        base="${base//:/-}"
    else
        base="$(basename "${path}")"
    fi
    base="${base%.gguf}"
    base="${base%.apr}"
    base="${base%.safetensors}"
    base="${base%.converted}"
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

# Build name→path mapping from apr cache and write to temp files
# Creates: ORDERED_PATHS array, NAME_MAP associative array
declare -A NAME_MAP
ORDERED_PATHS=()

build_cache_map() {
    # Try manifest.json first (contains ALL cached models), fall back to apr list
    local manifest="${HOME}/.cache/pacha/models/manifest.json"
    local cache_json

    if [[ -f "${manifest}" ]]; then
        cache_json="$(cat "${manifest}")"
        while IFS=$'\t' read -r name path; do
            if [[ -f "${path}" ]]; then
                local stripped="${name#hf_}"
                local org="${stripped%%_*}"
                stripped="${stripped#*_}"
                local repo="${stripped%%_*}"
                local filename="${stripped#*_}"
                local display="${org}/${repo}:${filename}"
                NAME_MAP["${path}"]="${display}"
                ORDERED_PATHS+=("${path}")
            fi
        done < <(printf '%s' "${cache_json}" | jq -r '.[] | [.name, .path] | @tsv')
    else
        cache_json="$("${APR_BIN}" list --json 2>/dev/null)" || return 0
        while IFS=$'\t' read -r name path; do
            if [[ -f "${path}" ]]; then
                local stripped="${name#hf_}"
                local org="${stripped%%_*}"
                stripped="${stripped#*_}"
                local repo="${stripped%%_*}"
                local filename="${stripped#*_}"
                local display="${org}/${repo}:${filename}"
                NAME_MAP["${path}"]="${display}"
                ORDERED_PATHS+=("${path}")
            fi
        done < <(printf '%s' "${cache_json}" | jq -r '.models[] | [.name, .path] | @tsv')
    fi
}

# Get display name: check NAME_MAP first, then fallback to basename
display_name_for() {
    local path="$1"
    local mapped="${NAME_MAP[$path]:-}"
    if [[ -n "${mapped}" ]]; then
        printf '%s' "${mapped}"
    else
        basename "${path}"
    fi
}

# Parse args
EXPLICIT_MODELS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cached-only) CACHED_ONLY=true; shift ;;
        *) EXPLICIT_MODELS+=("$1"); shift ;;
    esac
done

# Build cache mapping
build_cache_map

# Determine which models to run
if [[ ${#EXPLICIT_MODELS[@]} -gt 0 ]]; then
    MODELS=("${EXPLICIT_MODELS[@]}")
elif [[ "${CACHED_ONLY}" == "true" ]]; then
    MODELS=("${ORDERED_PATHS[@]}")
else
    # Local models first, then cached models (dedup by path)
    MODELS=("${LOCAL_MODELS[@]}")
    declare -A SEEN
    for m in "${LOCAL_MODELS[@]}"; do
        SEEN["${m}"]=1
    done
    for m in "${ORDERED_PATHS[@]}"; do
        was_seen="${SEEN[$m]:-}"
        if [[ -z "${was_seen}" ]]; then
            MODELS+=("${m}")
            SEEN["${m}"]=1
        fi
    done
fi

# ALL_MODELS used for matrix generation (determines row order)
ALL_MODELS=("${MODELS[@]}")

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
    local_name="$(display_name_for "${model}")"
    echo "==> Qualifying: ${local_name} -> ${slug}.json"

    # Skip if JSON already exists and is non-empty (incremental runs)
    if [[ -s "${json_out}" ]] && jq empty "${json_out}" 2>/dev/null; then
        local_passed="$(jq -r '.passed' "${json_out}" 2>/dev/null)"
        if [[ "${local_passed}" == "true" ]]; then
            echo "    CACHED (already passed)"
            continue
        fi
    fi

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
    printf '%s\n' "Cross-subcommand smoke test results across all local and cached models."
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
    local slug json_file fmt sz dname
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
        dname="$(display_name_for "${model}")"
        # Truncate long names
        if (( ${#dname} > 50 )); then
            dname="${dname:0:47}..."
        fi

        # Extract gate statuses
        pass_count=0
        total_count=0
        printf '| %s | %s | %s |' "${dname}" "${fmt}" "${sz}"

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
    printf '%s\n' "# Run all local + cached models"
    printf '%s\n' "bash scripts/qualify-matrix.sh"
    printf '\n'
    printf '%s\n' "# Run only cached models (from apr pull)"
    printf '%s\n' "bash scripts/qualify-matrix.sh --cached-only"
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
