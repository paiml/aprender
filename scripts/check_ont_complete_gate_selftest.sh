#!/usr/bin/env bash
# check_ont_complete_gate_selftest.sh -- keep G-ONT's case table on every PR (#4429).
#
# G-ONT (scripts/release/ont_complete_gate.sh) judges ONT-001 against a PINNED infra
# checkout. A PR has neither, so the gate cannot measure there: it lived at
# scripts/check_ont_complete.sh, guard_tree.sh ran every scripts/check_*.sh bare, and the
# bare run exited 2 (usage) -- a red guard-tree row on the 0.70.0 car (#4429) for a gate
# that was never given its input. It now lives beside fleet_cells_gate.sh, the other rc
# gate that needs host inputs, and is run with --infra/--pin at the rc decision.
#
# What stays per-PR is what CAN be measured without infra: the gate's own case table and
# mutants. This guard runs them, and pins the move itself so it cannot silently undo:
#   self-test     the gate's --self-test (rows + mutants) passes
#   bare-refuses  the gate run with no input exits 2 -- it is not a per-PR guard
#   not-in-tree   guard_tree.sh's universe does not contain it (a rename back to
#                 scripts/check_*.sh turns THIS row red, not the car's guard-tree)
#
# exit 0 all rows hold . 1 a row failed
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
GATE=scripts/release/ont_complete_gate.sh
bad=0
row() { # row <name> <ok?>
    if [ "$2" = 0 ]; then echo "ok    $1"; else echo "FAIL  $1"; bad=1; fi
}

out=$(bash "$GATE" --self-test 2>&1); rc=$?
grep -q 'ont_complete_gate self-test: PASS' <<< "$out"; row self-test $((rc | $?))
[ "$rc" = 0 ] || grep -E '^FAIL' <<< "$out" | head -10 | sed 's/^/        /'

bash "$GATE" > /dev/null 2>&1; rc=$?
[ "$rc" = 2 ]; row bare-refuses $?

git ls-files 'scripts/check_*.sh' | grep -q 'ont_complete\.sh$\|check_ont_complete_gate\.sh$'
[ $? = 1 ]; row not-in-tree $?

echo "check_ont_complete_gate_selftest: $([ "$bad" = 0 ] && echo PASS || echo FAIL)"
exit "$bad"
