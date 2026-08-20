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
bashrs lint scripts/*.sh > "$LOG" 2>&1
errors=$(grep -cE '\[error\]' "$LOG" || true)

printf '=== bashrs must see every script in scripts/ (check_shell_lint_ratchet.sh) ===\n'
printf '%s script(s) scanned, %s error line(s)\n' "$scanned" "$errors"

if [ "${1:-}" = "--update" ]; then
    printf '%s\n' "$errors" > "$BASELINE"
    printf 'baseline set to %s\n' "$errors"
    exit 0
fi

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
