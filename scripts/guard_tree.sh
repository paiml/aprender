#!/usr/bin/env bash
#
# guard_tree.sh -- run every scripts/check_*.sh guard and report EVERY
# failure, not just the first (BSE-01, PMAT-1062).
#
# WHY THIS EXISTS
# ---------------
# .github/workflows/ci.yml's `guard-runner-labels` job invokes ~110
# scripts/check_*.sh guards as ~110 separate GitHub Actions steps. GitHub
# Actions steps are fail-fast by construction: the first failing step stops
# the job, so a PR that breaks guard #3 never learns whether guards #4-110
# also broke. This script removes that ordering dependency: it runs every
# guard in the universe, collects every result, prints one row per guard,
# and exits 1 only after every guard has had a chance to run.
#
# THE UNIVERSE IS NEVER A WRITTEN LIST
# -------------------------------------
# `guard_universe()` is exactly `git ls-files 'scripts/check_*.sh'`. A
# hand-maintained list drifts the moment someone adds or removes a guard
# script without also touching the list -- this script cannot go stale that
# way because it has no list to go stale.
#
# THE CARGO-FREE SUBSET
# ----------------------
# `--no-cargo` (and `--list --no-cargo`) restrict the universe to guards
# whose OWN source never contains a bare `cargo ` token:
# `grep -LE '(^|[^a-z_-])cargo '`. This is a purely textual test over the
# guard's source -- it does not execute anything to classify it -- so ci.yml
# can run this subset on a bare runner with no cargo/docker toolchain and
# guard_tree.sh's classification can never independently drift from a
# hand-copied list living in the workflow.
#
# SELF-TEST DETECTION
# --------------------
# A guard "advertises --self-test" when its own `--help` output contains the
# literal substring `self-test`. Detection captures `"$g" --help 2>&1` into a
# variable and greps the variable with `grep -c` -- NEVER `<producer> |
# grep -q`, which SIGPIPEs the producer under `pipefail` and can read a real
# match as a false negative (paiml/infra feedback_pipefail_...). A guard that
# advertises self-test gets TWO rows: `[self-test]` (run first) and `[run]`.
# A guard that does not gets one `[run]` row.
#
# USAGE
# -----
#   scripts/guard_tree.sh                     run every guard
#   scripts/guard_tree.sh --no-cargo          run only the cargo-free subset
#   scripts/guard_tree.sh --cargo-only        run only the cargo-using subset
#   scripts/guard_tree.sh --list              print the guard universe
#   scripts/guard_tree.sh --list --no-cargo   print the cargo-free subset
#   scripts/guard_tree.sh --list --cargo-only print the cargo-using subset

set -uo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)" || exit 1
cd "$REPO_ROOT" || exit 1

# A bare `cargo ` token: not preceded by a lowercase letter, underscore or
# hyphen (so `sccache`, `rustc-sccache`, `cargo-ci` in a path do not count),
# and followed by a space (so `cargo` alone, e.g. a comment fragment, does
# not count either -- it must look like an invocation).
CARGO_RE='(^|[^a-z_-])cargo '

guard_universe() {
    git ls-files 'scripts/check_*.sh'
}

cargo_free_universe() {
    # xargs -r: never invoke grep with zero files (empty universe).
    guard_universe | xargs -r grep -LE "$CARGO_RE"
}

cargo_only_universe() {
    guard_universe | xargs -r grep -lE "$CARGO_RE"
}

usage() {
    printf 'usage: %s [--list] [--no-cargo | --cargo-only]\n' "$0" >&2
    exit 2
}

list_mode=0
subset=all
for arg in "$@"; do
    case "$arg" in
        --list) list_mode=1 ;;
        --no-cargo) subset=no-cargo ;;
        --cargo-only) subset=cargo-only ;;
        *) usage ;;
    esac
done

universe_for_subset() {
    case "$subset" in
        no-cargo) cargo_free_universe ;;
        cargo-only) cargo_only_universe ;;
        all) guard_universe ;;
    esac
}

if [ "$list_mode" -eq 1 ]; then
    universe_for_subset
    exit 0
fi

# advertises_self_test G -- does guard G's --help output mention "self-test"?
#
# A guard that does not understand --help at all just runs its normal body
# (most guards fall through an argv `case` to a usage/die message that itself
# echoes the flags it accepts, per this repo's convention -- see
# check_contract_test_binding.sh's `case "${1:-}" in ... *) die "usage: $0
# [--self-test | --update-baseline]" ;; esac`). Either way this call's
# output is captured and never executed a second time for detection alone.
advertises_self_test() {
    g="$1"
    help_out="$("$g" --help 2>&1)"
    n="$(grep -c -- 'self-test' <<<"$help_out")"
    [ "${n:-0}" -gt 0 ]
}

TMP_OUT="$(mktemp)" || exit 1
trap 'rm -f "$TMP_OUT"' EXIT

total=0
failed=0
fail_rows=""

run_row() {
    # $1 = row label, remaining args = the command to run
    label="$1"
    shift
    total=$((total + 1))
    if "$@" >"$TMP_OUT" 2>&1; then
        printf 'PASS  %s\n' "$label"
    else
        failed=$((failed + 1))
        fail_rows="${fail_rows}${label}
"
        printf 'FAIL  %s\n' "$label"
        sed 's/^/      | /' "$TMP_OUT"
    fi
}

guards="$(universe_for_subset)"

while IFS= read -r g; do
    [ -n "$g" ] || continue
    if advertises_self_test "$g"; then
        run_row "$g [self-test]" "$g" --self-test
        run_row "$g [run]" "$g"
    else
        run_row "$g [run]" "$g"
    fi
done <<<"$guards"

printf '%d checks, %d failed\n' "$total" "$failed"
if [ "$failed" -gt 0 ]; then
    printf 'FAILED:\n%s' "$fail_rows" >&2
    exit 1
fi
exit 0
