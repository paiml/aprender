#!/usr/bin/env bash
# pp_quant_exit_fixtures.sh - real-file exit check for #3091/#3432 (PP-QUANT-001).
#
# NON-REQUIRED LANE. This script is not wired into any GitHub workflow and is
# not part of any gate. It is a manual/nightly probe run by hand or from an
# ad-hoc cron, never blocking. Refs #3429, #3432, #3091. Expected RED on main
# until #3432 lands (the IQ2_XXS/mixed-qtype refusal this pins).
#
# Downloads (if not already cached) two Qwen3.5-0.8B GGUF quantizations at a
# pinned unsloth revision, verifies their byte size + sha256, then runs `apr
# run --no-gpu` against each and reports one PASS/FAIL line per file.
#
# Usage:
#   scripts/pp_quant_exit_fixtures.sh               # verify+download, then run
#   scripts/pp_quant_exit_fixtures.sh --verify-only  # check cache only, no run
#
# Exit status: 0 iff both files PASS, 1 if any file's `apr run` failed or was
# missing from the cache in --verify-only mode, 2 on a cache integrity failure
# (size or sha256 mismatch on a cached or freshly downloaded file).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT

readonly PP_QUANT_EXIT_REV="6ab461498e2023f6e3c1baea90a8f0fe38ab64d0"
readonly PP_QUANT_EXIT_REPO="unsloth/Qwen3.5-0.8B-GGUF"

CACHE_DIR="${PP_QUANT_EXIT_CACHE:-"$HOME/.cache/aprender/pp-quant-exit"}"
readonly CACHE_DIR

# name|bytes|sha256
PP_QUANT_EXIT_FIXTURES=(
    "Qwen3.5-0.8B-UD-IQ2_XXS.gguf|338227456|a369165c8ec45a92d55cbad98b5377e2c32ca8dfc824c899d9d05b33b6b53e54"
    "Qwen3.5-0.8B-UD-Q4_K_XL.gguf|558772480|3177ebd67afe4438374da19e690bc1b98756f7e0fea9240e1be404336156a7b5"
)
readonly PP_QUANT_EXIT_FIXTURES

verify_only=0
for arg in "$@"; do
    case "$arg" in
        --verify-only)
            verify_only=1
            ;;
        *)
            echo "pp-quant-exit: unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

# file_size <path> -> bytes on stdout
file_size() {
    local target_path="$1"
    stat -c '%s' "$target_path" 2>/dev/null || stat -f '%z' "$target_path"
}

# file_sha256 <path> -> hex digest on stdout
file_sha256() {
    local target_path="$1"
    local digest_line
    digest_line="$(sha256sum "$target_path")"
    printf '%s\n' "${digest_line%% *}"
}

# verify_cached_file <path> <expected_bytes> <expected_sha256>
# Returns 0 if both match, 1 if the path is absent, 2 on a size/sha mismatch.
verify_cached_file() {
    local target_path="$1"
    local want_bytes="$2"
    local want_sha="$3"

    if [[ ! -f "$target_path" ]]; then
        return 1
    fi

    local have_bytes
    have_bytes="$(file_size "$target_path")"
    # SC2199 below is a bashrs false positive: have_bytes is a scalar.
    if [[ "$have_bytes" != "$want_bytes" ]]; then  # bashrs disable-line=SC2199
        echo "pp-quant-exit: $(basename "$target_path") size mismatch: expected $want_bytes, got $have_bytes" >&2
        return 2
    fi

    local have_sha
    have_sha="$(file_sha256 "$target_path")"
    if [[ "$have_sha" != "$want_sha" ]]; then
        echo "pp-quant-exit: $(basename "$target_path") sha256 mismatch: expected $want_sha, got $have_sha" >&2
        return 2
    fi

    return 0
}

# download_file <name> <dest_path>
download_file() {
    local fixture_name="$1"
    local dest_path="$2"
    local url="https://huggingface.co/${PP_QUANT_EXIT_REPO}/resolve/${PP_QUANT_EXIT_REV}/${fixture_name}"
    local part_path="${dest_path}.part"

    echo "pp-quant-exit: downloading $fixture_name..." >&2
    curl -fL --progress-bar -o "$part_path" "$url"
    mv "$part_path" "$dest_path"
}

mkdir -p "$CACHE_DIR"

if [[ "$verify_only" -eq 0 ]]; then
    # Pin the binary once: never a bare `apr` (CLAUDE.md Step 0).
    . "${REPO_ROOT}/scripts/apr_bin.sh" || exit 1  # bashrs disable-line=SC1091
fi

overall_status=0

for entry in "${PP_QUANT_EXIT_FIXTURES[@]}"; do
    fixture_name="${entry%%|*}"
    rest="${entry#*|}"
    expected_bytes="${rest%%|*}"
    expected_sha="${rest#*|}"
    dest_path="${CACHE_DIR}/${fixture_name}"

    verify_rc=0
    verify_cached_file "$dest_path" "$expected_bytes" "$expected_sha" || verify_rc=$?

    if [[ "$verify_rc" -eq 2 ]]; then
        echo "pp-quant-exit: $fixture_name FAIL (cache integrity failure)" >&2
        exit 2
    fi

    if [[ "$verify_rc" -eq 1 ]]; then
        if [[ "$verify_only" -eq 1 ]]; then
            echo "pp-quant-exit: $fixture_name sha256=(absent) exit=- FAIL"
            overall_status=1
            continue
        fi
        download_file "$fixture_name" "$dest_path"
        download_verify_rc=0
        verify_cached_file "$dest_path" "$expected_bytes" "$expected_sha" || download_verify_rc=$?
        if [[ "$download_verify_rc" -ne 0 ]]; then
            echo "pp-quant-exit: $fixture_name FAIL (downloaded file failed integrity check)" >&2
            exit 2
        fi
    fi

    sha_prefix="$(file_sha256 "$dest_path" | cut -c1-16)"  # bashrs disable-line=PERF002

    if [[ "$verify_only" -eq 1 ]]; then
        echo "pp-quant-exit: $fixture_name sha256=${sha_prefix} exit=- PASS"  # bashrs disable-line=SC2178
        continue
    fi

    # The log is KEPT: a FAIL line without the refusal text is not evidence.
    # SC2178 on the report lines is a bashrs false positive: it parses the
    # literal `sha256=` inside the quoted message as an assignment.
    log_path="${CACHE_DIR}/${fixture_name}.run.log"
    run_rc=0
    "$APR" run "$dest_path" --prompt "2+2=" --max-tokens 8 --no-gpu >"$log_path" 2>&1 || run_rc=$?

    if [[ "$run_rc" -eq 0 ]]; then
        echo "pp-quant-exit: $fixture_name sha256=${sha_prefix} exit=${run_rc} PASS"  # bashrs disable-line=SC2178
    else
        echo "pp-quant-exit: $fixture_name sha256=${sha_prefix} exit=${run_rc} FAIL log=${log_path}"  # bashrs disable-line=SC2178
        overall_status=1
    fi
done

exit "$overall_status"
