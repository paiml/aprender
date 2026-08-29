#!/usr/bin/env bash
# Guard: the complexity DELTA gate (#2766) must stay wired into every surface
# that installs or runs a pre-commit hook.
#
# The live hook is generated into .git/ and is not tracked, so CI cannot inspect
# it. What CI CAN inspect is the tracked wiring that puts it there. A guard that
# only checked "the script exists" would be theater: the script existing while
# nothing calls it is exactly the failure mode this repo keeps hitting.
#
# Surfaces checked:
#   1. Makefile hooks-install  -> must re-splice after `pmat hooks install`
#      regenerates the hook, or the absolute scan silently comes back.
#   2. Makefile hooks-verify   -> must assert the splice is actually present.
#   3. .githooks/pre-commit    -> the documented `core.hooksPath .githooks`
#      install path must run the gate too.
#   4. The tracked fixture     -> must still carry the installer's anchors.

set -uo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT" || exit 1

fail=0
ok() { printf '  ok   %s\n' "$1"; }
bad() {
    fail=$((fail + 1))
    printf '  FAIL %s\n' "$1"
}

# recipe <target>: the tab-indented body of one Makefile target.
recipe() {
    awk -v t="$1:" '
        index($0, t) == 1 { inr = 1; next }
        inr && /^\t/ { print; next }
        inr && !/^\t/ && NF { inr = 0 }
    ' Makefile
}

echo "== 1. make hooks-install re-splices the delta gate =="
if recipe hooks-install | grep -q 'install_complexity_delta_gate\.sh'; then
    ok "hooks-install calls the installer"
else
    bad "hooks-install does NOT call scripts/install_complexity_delta_gate.sh"
    bad "  -> 'pmat hooks install' regenerates the hook and the #2766 freeze returns"
fi

echo "== 2. make hooks-verify asserts the splice is present =="
if recipe hooks-verify | grep -q 'install_complexity_delta_gate\.sh --check'; then
    ok "hooks-verify runs --check"
else
    bad "hooks-verify does NOT run install_complexity_delta_gate.sh --check"
fi

echo "== 3. .githooks/pre-commit runs the gate =="
if [ -f .githooks/pre-commit ]; then
    if grep -q 'complexity_delta_gate\.sh' .githooks/pre-commit; then
        ok ".githooks/pre-commit runs the delta gate"
    else
        bad ".githooks/pre-commit does NOT run scripts/complexity_delta_gate.sh"
    fi
else
    bad ".githooks/pre-commit is missing"
fi

echo "== 4. the shipped pieces exist and are executable =="
for f in scripts/complexity_delta_gate.sh scripts/install_complexity_delta_gate.sh; do
    if [ -x "$f" ]; then ok "$f is executable"; else bad "$f missing or not executable"; fi
done
if [ -f scripts/lib/complexity_delta_violations.jq ]; then
    ok "scripts/lib/complexity_delta_violations.jq present"
else
    bad "scripts/lib/complexity_delta_violations.jq missing (the gate cannot measure without it)"
fi

echo "== 5. the tracked fixture still carries the installer anchors =="
FIX=scripts/tests/fixtures/pmat_generated_pre_commit.fixture
if [ -f "$FIX" ]; then
    st=$(grep -c -E '^# 1\. Complexity analysis' "$FIX")
    sline=$(grep -n -E '^# 1\. Complexity analysis' "$FIX" | cut -d: -f1 | awk 'NR == 1')
    en=$(awk -v s="${sline:-0}" \
        'NR > s && /Complexity check\.\.\. ⏭.*no source files staged/ { n++ } END { print n + 0 }' "$FIX")
    if [ "$st" = 1 ] && [ "$en" = 1 ]; then
        ok "fixture matches START and END anchors exactly once each"
    else
        bad "fixture anchors drifted (start=$st end=$en) - re-anchor the installer"
    fi
else
    bad "$FIX missing - the installer has nothing to be tested against"
fi

if [ "$fail" -eq 0 ]; then
    echo "complexity delta gate wiring: OK"
    exit 0
fi
printf 'complexity delta gate wiring: %d FAILURE(S)\n' "$fail" >&2
exit 1
