#!/usr/bin/env bash
# check_build_ledger_collector.sh — run scripts/collect_build_ledger.sh's case table
# in CI.
#
# WHY A SHIM EXISTS AT ALL
# ------------------------
# `guard_tree.sh`'s universe is exactly `git ls-files 'scripts/check_*.sh'`, and it
# runs each one BARE. The collector is a tool, not a guard, so it is not in that
# universe and its 32-row case table would run nowhere — a self-test nothing executes
# is the same dark target as a `tests/*.rs` file never added to ci.yml's --test line.
# Naming the tool `check_…` to get it collected would be worse: `guard_tree.sh` would
# then run a COLLECTOR bare in CI, which means network calls from inside a guard.
#
# So the tool stays a tool, and this five-line guard is what the tree collects.
#
# exit: 0 the case table passed · 1 it failed · 2 the box cannot answer (collector
#       missing, or no jq — an unrun case table is never a passing one).

set -uo pipefail

PROG=${0##*/}
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL="$ROOT/scripts/collect_build_ledger.sh"

[ -f "$TOOL" ] || { echo "$PROG: ENV $TOOL is absent" >&2; exit 2; }
command -v jq > /dev/null 2>&1 || { echo "$PROG: ENV jq is required by the case table" >&2; exit 2; }

# Bare, --self-test and --help all mean the same thing here: there is nothing else
# this guard does. guard_tree.sh runs it bare.
case "${1:-}" in
    ""|--self-test|-h|--help) ;;
    *) echo "usage: $PROG [--self-test]" >&2; exit 2 ;;
esac

out=$(bash "$TOOL" --self-test 2>&1); rc=$?
printf '%s\n' "$out"
if [ "$rc" -ne 0 ]; then
    echo "$PROG: FAIL — collect_build_ledger.sh --self-test exited $rc" >&2
    exit 1
fi
rows=$(printf '%s\n' "$out" | grep -c '^  OK   ')
# Vacuity: a case table that passes with no rows is the `0 violations over 0 files`
# signature. The count is a floor, not the exact number, so adding rows never fails.
if [ "$rows" -lt 20 ]; then
    echo "$PROG: FAIL — case table reported only $rows row(s); expected at least 20" >&2
    exit 1
fi
echo "$PROG: PASS — collect_build_ledger.sh case table, $rows row(s)"
