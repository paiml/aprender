#!/usr/bin/env bash
# check_guards_are_wired.sh — every scripts/check_*.sh must be named by at least
# one GitHub workflow.
#
# WHY THIS EXISTS
# ---------------
# A guard that no workflow invokes is a file that looks like enforcement and is
# not reachable by any automated path. Four were found this way, by accident,
# while looking for something else (#2512):
#
#   check_contract_test_binding.sh   ci=0  makefile=2
#   check_wasm32_core_builds.sh      ci=0  makefile=1
#   check_book_examples_executable.sh ci=0 makefile=0   <- invoked by NOTHING
#   check_package_includes.sh        ci=0  makefile=0   <- invoked by NOTHING
#
# Makefile-only means `make tier3`, which is not run in CI. The bottom two were
# reachable from nothing at all.
#
# `check_package_includes.sh` is the sharp one: it is the CB-510 guard, written
# because a `models/` pattern matched `src/models/` and hid source from
# crates.io. Its own header instructs the reader to run it after any `.gitignore`
# or `Cargo.toml` exclude change. Its sibling `check_include_files.sh` IS wired.
# Nothing enforced the instruction.
#
# This is the meta-guard: without it, the next one to go dark is found the same
# way these were.
#
# A shrink-only baseline holds any deliberate exemption, so a guard that
# genuinely should not run in CI can be recorded rather than argued about — but
# the list may only get shorter.
#
#   bash scripts/check_guards_are_wired.sh              # check
#   bash scripts/check_guards_are_wired.sh --self-test  # case table
#   bash scripts/check_guards_are_wired.sh --update     # re-baseline

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="${REPO_ROOT}/scripts/unwired_guards_baseline.txt"

# Guards named by no workflow, one per line, sorted.
unwired_in() {
    local root="$1" g base
    for g in "$root"/scripts/check_*.sh; do
        [ -f "$g" ] || continue
        base=$(basename "$g")
        if ! grep -rqF -- "$base" "$root"/.github/workflows/ 2>/dev/null; then
            printf '%s\n' "$base"
        fi
    done | LC_ALL=C sort
}

# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    TD=$(mktemp -d) || exit 1
    trap 'rm -rf "${TD:?}"' EXIT
    fails=0
    mkdir -p "$TD/scripts" "$TD/.github/workflows"
    : > "$TD/scripts/check_wired.sh"
    : > "$TD/scripts/check_dark.sh"
    printf 'jobs:\n  x:\n    steps:\n      - run: bash scripts/check_wired.sh\n' \
        > "$TD/.github/workflows/ci.yml"

    got=$(unwired_in "$TD" | tr '\n' ' ')
    if [ "$got" = "check_dark.sh " ]; then
        printf 'ok    row 1 the unwired guard is reported, the wired one is not\n'
    else
        printf 'FAIL  row 1 got [%s], expected [check_dark.sh ]\n' "$got"; fails=1
    fi

    # Row 2 is the control: wire it up and the report must go EMPTY. Without
    # this, row 1 passes even if the scan reported every guard it saw.
    printf '      - run: bash scripts/check_dark.sh\n' >> "$TD/.github/workflows/ci.yml"
    if [ -z "$(unwired_in "$TD")" ]; then
        printf 'ok    row 2 wiring it clears the report\n'
    else
        printf 'FAIL  row 2 still reports: %s\n' "$(unwired_in "$TD" | tr '\n' ' ')"; fails=1
    fi

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
    printf '\nSELF-TEST PASSED (2/2)\n'
    exit 0
fi

printf '=== every check_*.sh must be named by a workflow (check_guards_are_wired.sh) ===\n'

total=$(find "$REPO_ROOT/scripts" -maxdepth 1 -name 'check_*.sh' | wc -l | tr -d ' ')

# Vacuity: a glob that matched nothing would report zero unwired guards and look
# like a pass. That is the exact failure mode this guard is about.
if [ "$total" -lt 20 ]; then
    printf '\nFAIL (vacuity): only %s guard(s) found under scripts/, expected 20+.\n' "$total"
    printf 'The scan is broken, not the wiring. Fix it rather than this number.\n'
    exit 1
fi

FOUND=$(unwired_in "$REPO_ROOT")
count=$(printf '%s\n' "$FOUND" | grep -c . || true)

printf '%s guard(s) scanned, %s named by no workflow\n' "$total" "$count"

if [ "${1:-}" = "--update" ]; then
    printf '%s\n' "$FOUND" | grep . > "$BASELINE" || : > "$BASELINE"
    printf 'baseline set to %s\n' "$count"
    exit 0
fi

if [ ! -f "$BASELINE" ]; then
    printf 'FAIL: %s missing. Run --update once to establish it.\n' "$BASELINE"
    exit 1
fi
baseline_count=$(grep -cvE '^\s*(#|$)' "$BASELINE" || true)

if [ "$count" -gt "$baseline_count" ]; then
    printf '\nFAIL: unwired guards grew %s -> %s.\n' "$baseline_count" "$count"
    printf 'A guard was added or unwired. Name it in a workflow, or record the\n'
    printf 'exemption in %s with a reason.\n\n' "$(basename "$BASELINE")"
    comm -13 <(grep -vE '^\s*(#|$)' "$BASELINE" | LC_ALL=C sort) \
            <(printf '%s\n' "$FOUND" | grep .) | sed 's|^|  NEW: |'
    exit 1
fi

if [ "$count" -lt "$baseline_count" ]; then
    printf '\nImproved: %s -> %s. Run --update to record it.\n' "$baseline_count" "$count"
fi

printf 'PASS (ratcheted)\n'
exit 0
