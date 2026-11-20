#!/usr/bin/env bash
# check_apr_cli_no_default_features.sh - `cargo build -p apr-cli --no-default-features` must compile (#7294).
#
# `realizar` is linked only by the `inference` feature, so a module that names it must be gated on that
# feature. Exit 0 when the build succeeds, 1 when it fails (the first errors are printed), 2 when cargo could
# not be run (not found, or it exited without a rustc diagnostic: not_measured, never a verdict), 64 on
# a bad flag. LAB check: not wired into a PR, merge-queue, release or verdict job.
#
# Usage: bash scripts/check_apr_cli_no_default_features.sh [--target-dir DIR]
set -euo pipefail

usage() { echo "Usage: bash scripts/check_apr_cli_no_default_features.sh [--target-dir DIR]" >&2; }

target_dir="${CARGO_TARGET_DIR:-}"
case "${1:-}" in
    "") ;;
    -h | --help)
        usage
        exit 0
        ;;
    --target-dir)
        target_dir="${2:?--target-dir needs a directory}"
        ;;
    *)
        usage
        exit 64
        ;;
esac
if ! command -v cargo > /dev/null 2>&1; then
    echo "not_measured: cargo not found" >&2
    exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."

log="$(mktemp "${TMPDIR:-/tmp}/apr-cli-nodefault.XXXXXX")"
trap 'rm -f "$log"' EXIT
[[ -n "$target_dir" ]] && export CARGO_TARGET_DIR="$target_dir"

# One cargo step, classified: a rustc error is a verdict (1), anything else is not_measured (2).
run_step() {
    local label="$1"
    shift
    local rc=0
    nice -n 10 cargo "$@" -p apr-cli --no-default-features -j 8 > "$log" 2>&1 || rc=$?
    [[ "$rc" -eq 0 ]] && return 0
    if grep -qE "^error(\[E[0-9]+\])?:" "$log"; then
        echo "FAIL: $label exited $rc" >&2
        grep -E "^error" "$log" | head -n 20 >&2 || true
        exit 1
    fi
    echo "not_measured: $label exited $rc with no rustc error (cargo could not run)" >&2
    tail -n 5 "$log" >&2 || true
    exit 2
}

run_step "cargo build -p apr-cli --no-default-features" build
# Not checked here: `cargo test -p apr-cli --no-default-features --lib --no-run` has 70+ pre-existing
# errors in other test modules (golden_output, check, tokenize, ...); it joins this script once they are fixed.
echo "OK: apr-cli builds with --no-default-features"
