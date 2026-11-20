#!/usr/bin/env bash
# check_apr_cli_no_default_features.sh - `cargo build -p apr-cli --no-default-features` must compile (#7294).
#
# `realizar` is linked only by the `inference` feature, so a module that names it must be gated on that
# feature. Exit 0 when the build succeeds, 1 when it fails (the first errors are printed), 2 when cargo
# could not be run. LAB check: not wired into a PR, merge-queue or release job.
#
# Usage: bash scripts/check_apr_cli_no_default_features.sh [--target-dir DIR]
set -euo pipefail

target_dir="${CARGO_TARGET_DIR:-}"
if [[ "${1:-}" == "--target-dir" ]]; then
    target_dir="${2:?--target-dir needs a directory}"
fi
if ! command -v cargo > /dev/null 2>&1; then
    echo "not_measured: cargo not found" >&2
    exit 2
fi

log="$(mktemp "${TMPDIR:-/tmp}/apr-cli-nodefault.XXXXXX")"
trap 'rm -f "$log"' EXIT

rc=0
if [[ -n "$target_dir" ]]; then
    CARGO_TARGET_DIR="$target_dir" nice -n 10 cargo build -p apr-cli --no-default-features -j 8 > "$log" 2>&1 || rc=$?
else
    nice -n 10 cargo build -p apr-cli --no-default-features -j 8 > "$log" 2>&1 || rc=$?
fi

if [[ "$rc" -ne 0 ]]; then
    echo "FAIL: nice -n 10 cargo build -p apr-cli --no-default-features exited $rc" >&2
    grep -E "^error" "$log" | head -n 20 >&2 || true
    exit 1
fi
echo "OK: apr-cli builds with --no-default-features"
