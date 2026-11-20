#!/usr/bin/env bash
# check_apr_cli_no_default_features.sh - `cargo build -p apr-cli --no-default-features` must compile (#7294).
#
# `realizar` is linked only by the `inference` feature, so a module that names it must be gated on that
# feature. Exit 0 when the build succeeds, 1 when it fails (the first errors are printed), 2 when cargo could
# not be run (not found, or it exited without a rustc diagnostic: not_measured, never a verdict), 64 on
# a bad flag. LAB check: not wired into a PR, merge-queue, release or verdict job.
#
# Usage: bash scripts/check_apr_cli_no_default_features.sh [--target-dir DIR | --self-test]  (self-test: arg-parsing rows, no cargo)
set -euo pipefail

usage() { echo "Usage: bash scripts/check_apr_cli_no_default_features.sh [--target-dir DIR]" >&2; }

target_dir="${CARGO_TARGET_DIR:-}"
# parse_args: sets target_dir; exits 0 on --help, 64 on any bad flag, a bare --target-dir, or trailing args.
parse_args() {
    # No arguments: default run. A lone empty-string argument is a bad flag, not "no arguments".
    [[ $# -eq 0 ]] && return 0
    case "$1" in
        -h | --help)
            [[ $# -eq 1 ]] || { usage; exit 64; }
            usage
            exit 0
            ;;
        --target-dir)
            if [[ $# -ne 2 || -z "${2:-}" ]]; then
                usage
                exit 64
            fi
            target_dir="$2"
            ;;
        *)
            usage
            exit 64
            ;;
    esac
}

# --self-test: the argument-parsing rows (no cargo run); exit 0 all pass, 1 a row failed.
if [[ "${1:-}" == "--self-test" ]]; then
    me="${BASH_SOURCE[0]}"
    fails=0
    row() {
        local want="$1" got=0
        shift
        bash "$me" "$@" > /dev/null 2>&1 < /dev/null || got=$?
        if [[ "$got" -ne "$want" ]]; then
            echo "FAIL row: args [$*] want rc=$want got rc=$got" >&2
            fails=$((fails + 1))
        fi
    }
    row 0 --help
    row 64 --bogus
    row 64 --target-dir
    row 64 --target-dir ""
    row 64 --target-dir /tmp/x extra
    row 64 --target-dir /tmp/x --help
    row 64 "" extra
    row 64 --help extra
    row 64 ""
    # fake-cargo rows: classification of a failing cargo (no real cargo run)
    fake_row() {
        local want="$1" msg="$2" d got=0
        d="$(mktemp -d "${TMPDIR:-/tmp}/apr-cli-nodefault-st.XXXXXX")"
        printf '#!/bin/sh\necho "%s" >&2\nexit 101\n' "$msg" > "$d/cargo"
        chmod +x "$d/cargo"
        PATH="$d:$PATH" bash "$me" > /dev/null 2>&1 < /dev/null || got=$?
        rm -rf "$d"
        if [[ "$got" -ne "$want" ]]; then
            echo "FAIL fake-cargo row: [$msg] want rc=$want got rc=$got" >&2
            fails=$((fails + 1))
        fi
    }
    fake_row 2 "error: failed to acquire package cache lock"
    fake_row 2 "error: failed to get \`anyhow\` as a dependency of package \`apr-cli\`"
    fake_row 2 "error: no matching package named \`arrow-array\` found"
    fake_row 1 "error[E0433]: failed to resolve: use of undeclared crate realizar"
    fake_row 1 "error: could not compile \`apr-cli\` (lib) due to 3 previous errors"
    [[ "$fails" -eq 0 ]] && echo "OK: self-test rows pass" && exit 0
    exit 1
fi
parse_args "$@"
if ! command -v cargo > /dev/null 2>&1; then
    echo "not_measured: cargo not found" >&2
    exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."

log="$(mktemp "${TMPDIR:-/tmp}/apr-cli-nodefault.XXXXXX")"
trap 'rm -f "$log"' EXIT
[[ -n "$target_dir" ]] && export CARGO_TARGET_DIR="$target_dir"

# One cargo step, classified: a rustc error (error[E....] or `could not compile`) is a verdict (1), anything else is not_measured (2).
run_step() {
    local label="$1"
    shift
    local rc=0
    nice -n 10 cargo "$@" -p apr-cli --no-default-features -j 8 > "$log" 2>&1 || rc=$?
    [[ "$rc" -eq 0 ]] && return 0
    # A rustc diagnostic is `error[E....]:` or cargo's own `error: could not compile`; every other `error:`
    # line (package cache lock, failed to get, no matching package, ...) is cargo/host, not a verdict.
    if grep -qE "^error\[E[0-9]+\]:|^error: could not compile" "$log"; then
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
# errors in other test modules (golden_output, check, tokenize, ...); unfiled follow-up (no ticket, backlog capped); it joins this script once they are fixed.
echo "OK: apr-cli builds with --no-default-features"
