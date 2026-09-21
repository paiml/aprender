#!/usr/bin/env bash
# dogfood_comparator_env_tests.sh -- the comparator-consumer tests, run WITH the pinned llama.cpp (#3740).
#
# WHY. In-tree Rust tests that compare apr against llama.cpp read ONLY the environment
# scripts/llama_bin.sh exports after it has proved the pinned build ($LLAMA_CLI, $LLAMA_SERVER):
# no PATH lookup, no list of candidate paths, no second resolver. Without that environment they
# SKIP and print the command that runs them -- which makes a skip the easy path. The cop's
# condition on #3740 (2026-09-21): the pre-publish dogfood runs them WITH the environment set, so
# the skip can never be the release path. This script is that run, declared as a dogfood gate in
# Cargo.toml's [package.metadata.dogfood].
#
# A test that returns early prints `... ok` exactly like one that ran, so the verdict is read from
# what the test SAID, not from its exit status alone: a "skipped —" line is a FAIL, and so is a
# run in which the named test did not appear at all (a filter that matched nothing).
#
# THE GPU RULE (rule rev 5): the build runs outside the lock; the run itself queues through
# `gpu-q --prio 1` (release-blocking; DOGFOOD_GPU_PRIO overrides it for a proof run) where gpu-q exists.
#
# exit 0  every listed test ran against the pinned build and passed
# exit 1  the pin does not resolve on this host, a test did not build, failed, skipped or did not run
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

# The comparator-consumer tests: "<package> <test target> <test name>", one per line.
TESTS='aprender-serve qwen3_moe_argmax_parity f_qw3_moe_parity_002_argmax_vs_llama_cpp'

# shellcheck source=scripts/llama_bin.sh
. scripts/llama_bin.sh; pin_rc=$?
if [ "$pin_rc" -ne 0 ]; then
    printf 'FAIL  the pinned llama.cpp does not resolve on %s (scripts/llama_bin.sh rc=%s): the comparator-consumer tests cannot run against it\n' "$(uname -n)" "$pin_rc"
    exit 1
fi
# the test compares a RAW completion, so its comparator is llama-completion (LOAD.md at the pin bump)
if [ -z "${LLAMA_COMPLETION:-}" ] || [ ! -x "$LLAMA_COMPLETION" ]; then
    printf 'FAIL  scripts/llama_bin.sh proved the pin but exported no executable LLAMA_COMPLETION (%s): build the llama-completion target\n' "${LLAMA_COMPLETION:-<unset>}"
    exit 1
fi
printf 'ok    pinned llama-completion: %s (%s)\n' "$LLAMA_COMPLETION" "${LLAMA_BUILD:-?}"

# priority 1 at release; a first-green proof outside a train passes DOGFOOD_GPU_PRIO=8 (cop, rule rev 5)
prio="${DOGFOOD_GPU_PRIO:-1}"
case "$prio" in [0-9]) ;; *) printf 'FAIL  DOGFOOD_GPU_PRIO=%s is not 0-9\n' "$prio"; exit 1 ;; esac
WRAP=""
command -v gpu-q > /dev/null 2>&1 && WRAP="gpu-q --prio $prio --"
log=$(mktemp) || exit 2
trap 'rm -f "${log:?}"' EXIT
rc=0; n=0
while read -r pkg target name; do
    [ -n "$pkg" ] || continue
    n=$((n + 1))
    if ! cargo test -p "$pkg" --test "$target" --no-run > "$log" 2>&1; then
        printf 'FAIL  %s --test %s did not build: %s\n' "$pkg" "$target" "$(grep -m1 -E '^error' "$log")"
        rc=1; continue
    fi
    # shellcheck disable=SC2086
    $WRAP cargo test -p "$pkg" --test "$target" -- --ignored --exact "$name" --nocapture > "$log" 2>&1; trc=$?
    if grep -q 'skipped —' "$log"; then
        printf 'FAIL  %s SKIPPED with the pinned build set -- a skip is not the release path: %s\n' "$name" "$(grep -m1 'skipped —' "$log")"
        rc=1
    elif [ "$trc" -ne 0 ]; then
        printf 'FAIL  %s exited %s: %s\n' "$name" "$trc" "$(grep -m1 -E 'panicked|FAILED|^error' "$log")"
        rc=1
    elif ! grep -qE "^test ${name} \.\.\. ok$" "$log"; then
        printf 'FAIL  %s did not run (no "test %s ... ok" line): the filter matched nothing\n' "$name" "$name"
        rc=1
    else
        printf 'ok    %s ran against the pinned llama-completion and passed\n' "$name"
    fi
done <<< "$TESTS"
[ "$n" -gt 0 ] || { printf 'FAIL  no comparator-consumer test is listed: a gate over nothing is not a pass\n'; exit 1; }
[ "$rc" -eq 0 ] && printf 'PASS  %s comparator-consumer test(s) ran against the pinned llama.cpp\n' "$n"
exit "$rc"
