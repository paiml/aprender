#!/usr/bin/env bash
# check_no_literal_thread_count.sh -- no workflow carries a literal thread count (#6151).
#
# Operator, 2026-10-23 11:26Z: thread caps come from the runner, not from the file.
# Every CARGO_BUILD_JOBS / RAYON_NUM_THREADS / NEXTEST_TEST_THREADS / RUST_TEST_THREADS
# in ci/sections.yml and .github/workflows/*.yml is "${RUNNER_THREAD_CAP:-4}". A literal
# (=16, =8, =4) is how 16 outran a 4-core runner. `--test-threads 1` is serialization,
# not a cap, and is not matched.
#
#   bash scripts/check_no_literal_thread_count.sh              scan, then the planted cases
#   bash scripts/check_no_literal_thread_count.sh --self-test  planted cases only
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RE='(CARGO_BUILD_JOBS|RAYON_NUM_THREADS|NEXTEST_TEST_THREADS|RUST_TEST_THREADS)[[:space:]]*[=:][[:space:]]*"?[0-9]+'

scan() { # scan FILE... -> prints offending lines, rc 1 if any
    local hits
    hits=$(grep -nHE "$RE" "$@" 2>/dev/null | grep -vE '^[^:]+:[0-9]+:[[:space:]]*#' || true)
    if [ -n "$hits" ]; then printf 'FAIL: literal thread count (use "${RUNNER_THREAD_CAP:-4}"):\n%s\n' "$hits"; return 1; fi
}

self_test() {
    local td rc=0 r
    td=$(mktemp -d)
    printf '  -e CARGO_BUILD_JOBS="${RUNNER_THREAD_CAP:-4}" -e RAYON_NUM_THREADS="${RUNNER_THREAD_CAP:-4}"\n  run-tests -- --test-threads 1\n# CARGO_BUILD_JOBS=16 in a comment\n' > "$td/good.yml"
    printf '  -e CARGO_BUILD_JOBS=16 \\\n' > "$td/bad16.yml"
    printf '  env:\n    NEXTEST_TEST_THREADS: 8\n' > "$td/badenv.yml"
    printf '  -e RUST_TEST_THREADS="4" \\\n' > "$td/badq.yml"
    scan "$td/good.yml" >/dev/null && r=0 || r=1
    if [ "$r" = 0 ]; then echo "ok   self-test: cap references and --test-threads 1 pass"; else echo "FAIL self-test: good fixture refused"; rc=1; fi
    for f in bad16 badenv badq; do
        scan "$td/$f.yml" >/dev/null && r=0 || r=1
        if [ "$r" = 1 ]; then echo "ok   self-test: planted literal ($f) is RED"; else echo "FAIL self-test: planted literal ($f) passed"; rc=1; fi
    done
    rm -rf "${td:?}"
    return "$rc"
}

[ "${1:-}" = "--self-test" ] && { self_test; exit $?; }
scan "$root/ci/sections.yml" "$root"/.github/workflows/*.yml
self_test
echo "check_no_literal_thread_count: PASS"
