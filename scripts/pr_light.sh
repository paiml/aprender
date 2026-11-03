#!/usr/bin/env bash
# pr_light.sh -- `make pr-light`: the cargo-free half of CI's `pr-light` job, run locally BEFORE you push.
#
# Operator C2 (2026-11-03 07:05Z): "checks before review". A PR that is red in pr-light cannot be armed, and the
# usual reds (a new `grep | grep -q` site, a literal count in scripts/release, a guard case table) are found in
# seconds here instead of ~10 minutes in CI. Every step runs and every failure is reported, like guard_tree.sh.
#
#   bash scripts/pr_light.sh              # run every step, exit 1 if any failed
#   bash scripts/pr_light.sh --list       # print the steps and exit 0
#   bash scripts/pr_light.sh --self-test
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# name|command -- the steps mirror .github/workflows/ci.yml `pr-light` (cheap checks + the cargo-free guard tree).
steps() {
    cat << 'STEPS'
fmt|cargo fmt --all -- --check
fail-fast self-test|python3 scripts/ci/fail_fast.py self-test
falsification ids|python3 scripts/ci/fail_fast.py ids --base origin/main
guard-tree case table|bash scripts/tests/guard_tree_test.sh
guard-tree (cargo-free)|bash scripts/guard_tree.sh --no-cargo
STEPS
}

run_steps() { # run_steps -> prints one row per step; returns 1 if any failed
    local failed=0 n=0 name cmd rc
    while IFS='|' read -r name cmd; do
        [ -n "$name" ] || continue
        n=$((n + 1))
        set +e
        (cd "$REPO_ROOT" && bash -c "$cmd") > /dev/null 2>&1 < /dev/null
        rc=$?
        set -e
        if [ "$rc" -eq 0 ]; then printf 'ok    %s\n' "$name"; else printf 'FAIL  %s (rc=%s) -- rerun: %s\n' "$name" "$rc" "$cmd"; failed=1; fi
    done
    [ "$n" -gt 0 ] || { printf 'NO-GO: zero steps ran; that is not a pass\n' >&2; return 2; }
    return "$failed"
}

self_test() {
    local pass=0 fail=0 out rc
    row() { if [ "$2" = "$3" ]; then pass=$((pass + 1)); printf '  ok    %s\n' "$1"; else fail=$((fail + 1)); printf '  FAIL  %s want=%s got=%s\n' "$1" "$3" "$2"; fi; }
    row "the guard tree is a step" "$(steps | grep -c 'guard_tree.sh --no-cargo')" 1
    row "fmt is a step" "$(steps | grep -c '^fmt|')" 1
    set +e; out="$(printf 'good|true\nbad|false\nlast|true\n' | run_steps)"; rc=$?; set -e
    row "one failing step makes the run fail" "$rc" 1
    row "a failure does not stop later steps" "$(printf '%s\n' "$out" | grep -c '^ok    last')" 1
    row "the failing step is named" "$(printf '%s\n' "$out" | grep -c '^FAIL  bad')" 1
    set +e; printf 'good|true\n' | run_steps > /dev/null; rc=$?; set -e
    row "all green exits 0" "$rc" 0
    set +e; printf '' | run_steps > /dev/null; rc=$?; set -e
    row "zero steps is not a verdict" "$rc" 2
    set +e; out="$(printf 'eat|cat\nlast|true\n' | run_steps)"; set -e
    row "a step that reads stdin cannot swallow the next steps" "$(printf '%s\n' "$out" | grep -c '^ok    last')" 1
    printf 'pr_light self-test: %s passed, %s failed\n' "$pass" "$fail"
    [ "$fail" -eq 0 ]
}

case "${1:-}" in
    --self-test) self_test ;;
    --list) steps | cut -d'|' -f1 ;;
    '') printf '== pr-light (local, cargo-free half of CI) ==\n'; steps | run_steps ;;
    *) printf 'usage: %s [--list|--self-test]\n' "$(basename "$0")" >&2; exit 2 ;;
esac
