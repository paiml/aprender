#!/usr/bin/env bash
#
# check_gate_scope_not_narrowed.sh - the required `ci / gate` must have nothing
# left to narrow (aprender#2734).
#
# WHY THIS EXISTS
# ---------------
# `ci / gate` is a required status check on main, produced by the reusable
# workflow `paiml/.github/.github/workflows/sovereign-ci.yml@main`. Its lint and
# test steps both end in a fallback chain that narrows SCOPE on failure instead
# of reporting it (sovereign-ci.yml:528-530 and :312-322):
#
#   cargo clippy $CLIPPY_ARGS -- -D warnings || \
#   cargo clippy -p "$REPO_NAME" -- -D warnings || \
#   { echo "::error::Clippy failed - check workspace path dependencies"; exit 1; }
#
# The second command is not a retry of the same work; it is a strictly smaller
# scope, and taking it emits no `::warning::` at all. So a green `ci / gate`
# does not distinguish "the configured scope passed" from "the configured scope
# failed and a smaller one passed". That is the class this repo names most
# often: an assertion that cannot exclude the outcome it exists to exclude.
#
# The fallback is UPSTREAM. Nothing in this repo can disarm it, and #2734 says
# so. What this repo can do is hold the difference between the two scopes at
# zero, so a silent narrowing has nothing to hide.
#
# WHAT IT CHECKS  (both rules in scripts/lib/gate_scope.py)
#   S1 the CLIPPY window is empty - the units the fallback compiles are exactly
#      the units the primary command compiles, read from `cargo metadata`.
#   S2 the TEST window is empty - this caller does not pass `test_workspace`,
#      which would make TEST_SCOPE `--workspace --lib` against a `--lib -p
#      <repo>` fallback.
#
# IS THE WINDOW OPEN TODAY? NO - AND THAT IS THE POINT
# ----------------------------------------------------
# MEASURED on 50d2bc2bb (aprender 0.64.0): the root manifest is both
# [workspace] and [package] and declares no `default-members`, so
# `workspace_default_members` is 1 - the `aprender` facade alone - and that
# package declares exactly two selectable targets, `lib aprender` and
# `bin apr`. `--all-targets` adds test/bench/example targets that DO NOT EXIST
# (there is no root benches/ or examples/, and root tests/ holds one YAML
# fixture and no .rs). So the primary and fallback commands select an identical
# two-unit set and the fallback cannot currently mask anything.
#
# #2734 reads this the other way round - "the first command lints the facade's
# lib + bin + tests + benches + examples" - which overstates aprender's
# exposure. The mechanism is real; the live window on THIS repo is empty.
#
# It is empty by accident. Nothing prevents a root-level tests/*.rs, and the day
# one appears the required check silently starts hiding findings in it with no
# diff to review. This guard is what turns the accident into an invariant, which
# is the only part of #2734 that is fixable here.
#
# The same reasoning is why `--all-targets` in scripts/check_clippy_current_stable.sh
# adds nothing over `make tier2`'s bare `cargo clippy` today; that script's
# header said it was a superset and has been corrected in the same commit.
#
# RED-TURNING MUTATION (verified, not argued)
#   $ touch tests/smoke.rs && bash scripts/check_gate_scope_not_narrowed.sh
#     S1 CLIPPY WINDOW IS OPEN: 1 unit(s) ... aprender :: test :: smoke
#     exit 1
#   The --self-test table below runs that mutation and five others against
#   committed fixtures, including two must-NOT-match rows: the commented
#   `test_workspace: true` line that has sat in ci.yml since the sccache pilot,
#   and a second facade bin (which both commands select, so it is not a window).
#
#   bash scripts/check_gate_scope_not_narrowed.sh              # self-test, then check
#   bash scripts/check_gate_scope_not_narrowed.sh --self-test  # case table only
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCOPE="${REPO_ROOT}/scripts/lib/gate_scope.py"
CASES="${REPO_ROOT}/scripts/lib/gate_scope_cases"
CI_WORKFLOW="${REPO_ROOT}/.github/workflows/ci.yml"

# cargo exits non-zero for two categories of reason and only one is about the
# code. ENV still exits non-zero here; only the CLAIM changes.
. "${REPO_ROOT}/scripts/cargo_classify.sh" || exit 1

fails=0

run_case() { # name want_rc metadata workflow repo needle
    local name="$1" want="$2" md="$3" wf="$4" repo="$5" needle="$6" out rc
    out="$( python3 "$SCOPE" "$CASES/$md" "$CASES/$wf" "$repo" 2>&1 )"; rc=$?
    if [ "$rc" != "$want" ]; then
        printf 'FAIL  %s: exit %s, expected %s\n%s\n' "$name" "$rc" "$want" "$out"
        fails=1
        return
    fi
    if [ -n "$needle" ] && ! grep -q -- "$needle" <<< "$out"; then
        printf 'FAIL  %s: exit %s as expected but did not name %s\n%s\n' \
            "$name" "$rc" "$needle" "$out"
        fails=1
        return
    fi
    printf 'ok    %s\n' "$name"
}

self_test() {
    printf '=== case table (scripts/lib/gate_scope_cases) ===\n'
    # MUST NOT MATCH - the three ways this guard could be over-strict and
    # block work that does not open a window.
    run_case 'row 1  todays shape passes (lib+bin facade, no test_workspace)' \
        0 md_good.json wf_good.yml aprender ''
    run_case 'row 2  a COMMENTED test_workspace does not trip S2' \
        0 md_good.json wf_good.yml aprender 'S2 test window:   EMPTY'
    run_case 'row 3  a second facade bin is not a window (both commands select bins)' \
        0 md_facade_second_bin.json wf_good.yml aprender ''
    # MUST MATCH - each way the window actually opens.
    run_case 'row 4  a root tests/*.rs opens the clippy window' \
        1 md_facade_has_test.json wf_good.yml aprender 'aprender :: test :: smoke'
    run_case 'row 5  a root bench/example opens the clippy window' \
        1 md_facade_has_bench_example.json wf_good.yml aprender 'S1 CLIPPY WINDOW IS OPEN: 2'
    run_case 'row 6  default-members beyond the facade opens the window' \
        1 md_default_members_many.json wf_good.yml aprender 'aprender-core :: lib'
    run_case 'row 7  test_workspace: true opens the test window' \
        1 md_good.json wf_test_workspace.yml aprender 'S2 TEST WINDOW IS OPEN'
    # REFUSALS - the model must not silently keep its old meaning.
    run_case 'row 8  an overridden clippy_args is refused, not guessed' \
        1 md_good.json wf_clippy_args_override.yml aprender 'S1 UNMODELLED'
    run_case 'row 9  empty workspace_default_members is VACUOUS, not clean' \
        1 md_no_default_members.json wf_good.yml aprender 'S1 VACUOUS'
    run_case 'row 10 a workflow that calls no sovereign-ci is a fail mode' \
        2 md_good.json wf_no_caller.yml aprender 'pass over nothing'
    run_case 'row 11 a fallback naming no package is reported, not ignored' \
        1 md_good.json wf_good.yml nosuchcrate 'cargo clippy -p nosuchcrate'

    # The cargo failure CLASSIFIER is a different surface, so it gets its own
    # table rather than inheriting rows 1-11's green.
    cargo_classify_selftest --quiet || fails=1

    if [ "$fails" -ne 0 ]; then
        printf '\nSELF-TEST FAILED\n'
        return 1
    fi
    printf '\nSELF-TEST PASSED (11 rows + classifier table)\n'
    return 0
}

self_test || exit 1
if [ "${1:-}" = '--self-test' ]; then
    exit 0
fi

printf '\n=== live check: is there anything left for `ci / gate` to narrow? ===\n'

cd "$REPO_ROOT" || exit 1

# The fallback is `cargo clippy -p "$REPO_NAME"`, where REPO_NAME is the GitHub
# repository name the caller passes as `repo:`. Derived from the remote rather
# than hardcoded, so a rename is a RED here instead of a guard that quietly
# measures the wrong package.
#
# No nested double quotes inside the command substitution and no pipe: bashrs
# reports SC1078 on the former, and a status read through the latter is the
# first entry in CLAUDE.md Verification Discipline. Parameter expansion only.
REMOTE_URL="$(git remote get-url origin 2>/dev/null)"
REPO_NAME="${REMOTE_URL##*[/:]}"
REPO_NAME="${REPO_NAME%.git}"
if [ -z "$REPO_NAME" ]; then
    printf 'FAIL: no `origin` remote, so the fallback command `cargo clippy -p <repo>`\n'
    printf '      cannot be resolved. Refusing to guess the package it names.\n'
    exit 1
fi
printf 'repo name (from origin): %s\n' "$REPO_NAME"

MD="$( mktemp )" || exit 1
CARGOLOG="$( mktemp )" || exit 1
trap 'rm -f "${MD:?}" "${CARGOLOG:?}"' EXIT

# `--no-deps` performs no dependency resolution, so this cannot rewrite
# Cargo.lock behind check_lockfile_current.sh (verified: sha256 unchanged).
cargo metadata --no-deps --format-version 1 > "$MD" 2> "$CARGOLOG"
rc=$?
if [ "$rc" -ne 0 ] || [ ! -s "$MD" ]; then
    if [ "$( classify_cargo_failure "$CARGOLOG" )" = 'ENV' ]; then
        report_cargo_env_failure "$CARGOLOG" 'the required check comparison'
        exit 1
    fi
    printf 'FAIL: `cargo metadata --no-deps` produced no usable document (exit %s).\n' "$rc"
    sed 's/^/      | /' "$CARGOLOG"
    printf '      A scope that cannot be measured is not a scope known to be safe.\n'
    exit 1
fi

if python3 "$SCOPE" "$MD" "$CI_WORKFLOW" "$REPO_NAME"; then
    printf '\n'
    printf '%s\n' '✓ check_gate_scope_not_narrowed: the fallback in the required'
    printf '%s\n' '  check is a no-op. It selects exactly what the configured command'
    printf '%s\n' '  selects, so a silent narrowing has nothing to mask (aprender#2734).'
    exit 0
fi

printf '\n'
printf 'FAIL: `ci / gate` can now narrow its own scope without saying so.\n'
printf '      See aprender#2734 and the header of this script.\n'
exit 1
