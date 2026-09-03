#!/usr/bin/env bash
# check_shell_lint_ratchet.sh — bashrs must see every script in scripts/, and
# the error count may only fall.
#
# WHY THIS EXISTS
# ---------------
# CLAUDE.md mandates bashrs over shellcheck. CI honoured that for SEVEN of the
# 72 scripts:
#
#     .github/workflows/book.yml:58
#       bashrs lint scripts/check_book_*.sh
#
# `apr_bin.sh` was not among them. That is the script every gate in this repo
# sources to resolve the `apr` binary — 462 references — and no CI job linted
# it. Neither did anything else:
#
#   * `make lint-scripts` exists, but tier3 is not run in CI.
#   * pmat DOES accept the file and reports
#       Functions: 0, Max Cyclomatic: 0
#     for 227 lines holding 4 functions and heavy branching. It does not parse
#     shell. That is worse than no coverage, because it looks like a pass —
#     the vacuous-scan class this repo keeps closing.
#
# WHY A RATCHET AND NOT A FIX
# ---------------------------
# Extending the glob to `scripts/*.sh` surfaces ~847 error lines. They are
# dominated by bashrs's known false positives on HAND-WRITTEN bash: it parses an
# embedded heredoc as shell (so TOML `name = "x"` is SC1007), reads parens in a
# string sharing a line with `[ ]` as an unescaped test expression (SC1028),
# and treats em-dashes in prose as SC1100. Turning the gate on outright would
# be a 847-item triage, and a gate that cannot go green gets disabled.
#
# So the count is baselined and may only SHRINK. New scripts cannot add errors,
# and the existing debt is visible rather than hidden behind a glob.
#
# The real answer is upstream of this file: bashrs is a Rust-to-POSIX
# TRANSPILER, and shell it generates does not trip its own parser. Hand-written
# bash is the thing being linted here, and the fleet's own tooling says not to
# write it. That is a larger change than a gate.
#
#   bash scripts/check_shell_lint_ratchet.sh              # check
#   bash scripts/check_shell_lint_ratchet.sh --update     # re-baseline (shrink only)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="${REPO_ROOT}/scripts/shell_lint_baseline.txt"

cd "$REPO_ROOT" || exit 1

if ! command -v bashrs >/dev/null 2>&1; then
    printf 'SKIP: bashrs is not installed; install it with `cargo install bashrs --locked`.\n' >&2
    printf 'This is a hard failure in CI, where the workflow installs it first.\n' >&2
    [ "${CI:-}" = "true" ] && exit 1
    exit 0
fi

scanned=$(find scripts -maxdepth 1 -name '*.sh' | wc -l | tr -d ' ')

# Vacuity. A glob that matched nothing would report zero errors and look like a
# pass — which is exactly how the previous gate covered 7 of 72 without anyone
# noticing.
if [ "$scanned" -lt 60 ]; then
    printf 'FAIL (vacuity): only %s script(s) found under scripts/, expected 60+.\n' "$scanned"
    printf 'The glob is broken, not the scripts. Fix it rather than this number.\n'
    exit 1
fi

LOG=$(mktemp) || exit 1
trap 'rm -f "${LOG:?}"' EXIT
# ONE FILE PER INVOCATION. bashrs 7.0.1 cross-contaminates files linted in one
# call: the same tree counted 13 errors on the fleet and 53 on a workstation,
# and a branch that REMOVED 180 findings counted 57 on the fleet and 12 here,
# so the single-invocation number was noise the ratchet could not distinguish
# from growth (PMAT-936). Linting each script alone is deterministic; the file
# name is prefixed because a single-file run prints none.
: > "$LOG"
# A BROKEN TOOL IS NOT A CLEAN TREE. bashrs exits 0 on a clean file and 1 on
# findings, 2 on errors; anything above is the tool failing, and a
# tool that printed no [error] lines because it died must not read as an
# improvement (measured: a stub exiting 101 produced "Improved: 9 -> 0", PASS).
tool_failed=0
while IFS= read -r script; do
    out=$(bashrs lint "$script" 2>&1); rc=$?
    printf '%s\n' "$out" | sed "s|^|${script}: |" >> "$LOG"
    # bashrs: 0 clean, 1 warnings, 2 errors -- all three are the tool RUNNING.
    if [ "$rc" -gt 2 ]; then
        printf 'FAIL: bashrs exited %s on %s; a lint that could not run is not a lint that found nothing.\n' "$rc" "$script"
        tool_failed=1
    fi
done < <(find scripts -maxdepth 1 -name '*.sh' | LC_ALL=C sort)
if [ "$tool_failed" -ne 0 ]; then
    exit 1
fi
if [ ! -s "$LOG" ]; then
    printf 'FAIL: bashrs produced no output over %s script(s); the tool did not run.\n' "$scanned"
    exit 1
fi
errors=$(grep -cE '\[error\]' "$LOG" || true)

printf '=== bashrs must see every script in scripts/ (check_shell_lint_ratchet.sh) ===\n'
printf '%s script(s) scanned, %s error line(s)\n' "$scanned" "$errors"

if [ "${1:-}" = "--update" ]; then
    printf '%s\n' "$errors" > "$BASELINE"
    printf 'baseline set to %s\n' "$errors"
    exit 0
fi

# THE RATCHET IS A PROPERTY OF THE DIFF, NOT OF THE TREE.
#
# Everything above compares the scan against the baseline AS IT STANDS IN THE
# WORKING TREE, and that is not a ratchet. NEW (a finding with no entry) and
# STALE (an entry with no finding) are the only two properties a working tree
# can answer, and a commit that appends one line AND lands the matching
# violation satisfies both at once: not new, because it is baselined; not
# stale, because the finding is real.
#
# Measured, not argued: appending one entry cloned from this file's own last
# real entry returned rc=0 from this guard, under its own words:
#     "the count is baselined and may only SHRINK"
# Twelve guards in scripts/ failed the same probe.
#
# So growth is now compared against merge-base(HEAD, origin/main), falling
# back to the origin/main TIP because CI checks out shallow — a ref this
# branch cannot rewrite, and never the branch against itself.
# shellcheck source=scripts/lib_baseline_ratchet.sh
. "${REPO_ROOT}/scripts/lib_baseline_ratchet.sh" || exit 1
baseline_ratchet_check "${REPO_ROOT}" scripts/shell_lint_baseline.txt count || exit 1

if [ ! -f "$BASELINE" ]; then
    printf 'FAIL: %s missing. Run --update once to establish it.\n' "$BASELINE"
    exit 1
fi
baseline=$(tr -d '[:space:]' < "$BASELINE")

printf 'baseline %s\n' "$baseline"

if [ "$errors" -gt "$baseline" ]; then
    printf '\nFAIL: bashrs errors grew %s -> %s.\n' "$baseline" "$errors"
    printf 'A new or edited script added shell-quality errors. The top rules:\n\n'
    grep -oE '\[error\] [A-Z]+[0-9]+' "$LOG" | sort | uniq -c | sort -rn | head -6 | sed 's|^|  |'
    exit 1
fi

if [ "$errors" -lt "$baseline" ]; then
    printf '\nImproved: %s -> %s. Run --update to record it.\n' "$baseline" "$errors"
fi

printf 'PASS (ratcheted)\n'
exit 0
