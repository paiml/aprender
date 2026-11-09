#!/usr/bin/env bash
# scripts/lab/spec_sweep_rules.sh -- LAB (C324 items 4-5), spec-sweep 2026-10-31, epic #6809.
# Enforces five spec rules that had no blocking check: #6610 (PP-6) #6613 (slice-5 floor) #6614 (FT-12 state)
# #6615 (T-5 reconcile blocks release) #6616 (plan-quorum apr lane advisory). Each rule is a function that prints
# RED lines; a rule whose input file is missing is RED (the universe may not empty itself). A rule that is
# true of the spec but not yet true of the code (6615) is RED too: that is the finding, one ticket, one owner.
# Never in a PR, merge-queue or release job; run nightly or on demand.
# Usage: scripts/lab/spec_sweep_rules.sh [repo-root] | --self-test      Exit: 0 green, 1 RED, 2 usage
set -euo pipefail
PM=scripts/perf-matrix.yaml
EP74=docs/specifications/EPIC-0.74-any-model-plan.md
DR=docs/specifications/DEBT-RATCHET-001-070-074.md
FT=docs/specifications/release-0.70-fast-train-plan.md
AE=docs/specifications/APR-EPIC-001-themed-epics-and-issue-cap.md
need() { [ -f "$1/$2" ] || { echo "RED $2: input missing"; return 1; }; }

# 6610 PP-6: an arm with comparator_required: true declares phase: release (never merge/both/reporting).
pp6() {
  need "$1" "$PM" || return 0
  awk '
    /^arms:/ { in_arms = 1; next }
    in_arms && /^[A-Za-z]/ { in_arms = 0 }
    in_arms && /^  [A-Za-z0-9_]+:[[:space:]]*$/ { flush(); arm = $1; sub(/:$/, "", arm); ph = ""; cmp = ""; seen++ ; next }
    in_arms && /^    phase:/ { ph = $2 }
    in_arms && /^    comparator_required:/ { cmp = $2 }
    function flush() { if (arm != "" && cmp == "true" && ph != "release") printf "RED PP-6: arm %s is a comparator but phase=%s (want release)\n", arm, (ph == "" ? "<none>" : ph); if (arm != "" && cmp == "true") cmps++ }
    END { flush(); if (seen == 0) print "RED PP-6: no arms parsed"; else if (cmps == 0) print "RED PP-6: no comparator arm found (vacuous)" }
  ' "$1/$PM"
}

# 6613 slice 5: the A-pillar floor in EPIC-0.74 s4 equals B + round(0.8*(9500-B)), B read from DEBT-RATCHET A-2.
slice5() {
  need "$1" "$EP74" || return 0; need "$1" "$DR" || return 0
  local b floor want
  b=$(grep -E 'With `B = [0-9]+`' "$1/$DR" | head -n 1 | sed -E 's/.*With `B = ([0-9]+)`.*/\1/')
  floor=$(grep -E '^\| A: P' "$1/$EP74" | head -n 1 | sed -E 's/.*≥ ([0-9],[0-9]+).*/\1/' | tr -d ',')
  if ! [[ "$b" =~ ^[0-9]+$ && "$floor" =~ ^[0-9]+$ ]]; then echo "RED slice-5: cannot read B ('$b') or floor ('$floor')"; return 0; fi
  want=$(awk -v b="$b" 'BEGIN { printf "%d", b + int(0.8 * (9500 - b) + 0.5) }')
  [ "$floor" -eq "$want" ] || echo "RED slice-5: A floor $floor != B($b)+80% of gap = $want"
}

# 6614 FT-12: the debt-ratchet slice-1 header carries exactly one state token, PROPOSED or RULED.
ft12() {
  need "$1" "$FT" || return 0
  local h n
  h=$(grep -E '^### FT-12 ' "$1/$FT" | head -n 1)
  [ -n "$h" ] || { echo "RED FT-12: header missing"; return 0; }
  n=$(printf '%s' "$h" | grep -o -E 'PROPOSED|RULED' | wc -l)
  [ "$n" -eq 1 ] || echo "RED FT-12: header has $n state tokens (want exactly 1 of PROPOSED|RULED)"
}

# 6615 T-5: a release workflow runs scripts/check_reconcile.sh and does not mask its exit.
t5() {
  local f hit=0 w="$1/.github/workflows"
  [ -d "$w" ] || { echo "RED T-5: no workflows dir"; return 0; }
  for f in "$w"/binary-release.yml "$w"/release-gates-nightly.yml; do
    [ -f "$f" ] || continue
    if grep -q 'check_reconcile.sh' "$f"; then
      hit=$((hit + 1))
      grep -q -E '\|\|[[:space:]]*true|continue-on-error' <<<"$(grep 'check_reconcile.sh' "$f")" && echo "RED T-5: $(basename "$f") masks check_reconcile.sh"
    fi
  done
  [ "$hit" -gt 0 ] || echo "RED T-5: no release workflow runs scripts/check_reconcile.sh"
}

# 6616 plan quorum: the rule line says the apr lane is advisory and a do-not-implement verdict is a STOP.
pq() {
  need "$1" "$AE" || return 0
  local l; l=$(grep -E '^[0-9]+\. \*\*Plan quorum\.\*\*' "$1/$AE" | head -n 1)
  [ -n "$l" ] || { echo "RED plan-quorum: rule line missing"; return 0; }
  grep -q 'apr lane is advisory' <<<"$l" || echo "RED plan-quorum: apr lane not stated advisory"
  grep -q 'do-not-implement' <<<"$l" || echo "RED plan-quorum: no do-not-implement STOP"
}

run() {
  local root=$1 out reds=0 r
  [ -d "$root" ] || { echo "usage: no such dir $root" >&2; return 2; }
  for r in pp6 slice5 ft12 t5 pq; do
    out=$($r "$root"); if [ -n "$out" ]; then echo "$out"; reds=$((reds + 1)); fi
  done
  if [ "$reds" -eq 0 ]; then echo "GREEN: 5 rules"; return 0; fi
  echo "RED: $reds of 5 rules"; return 1
}

selftest() {
  local t rc fail=0
  t=$(mktemp -d)
  mk() {
    rm -rf "$t"; mkdir -p "$t/scripts" "$t/docs/specifications" "$t/.github/workflows"
    printf 'arms:\n  A:\n    phase: release\n    comparator_required: false\n  L3:\n    phase: release\n    comparator_required: true\n' > "$t/$PM"
    printf '| A: P₀ bp | ≥ 9,364 (x) |\n' > "$t/$EP74"
    printf 'With `B = 8819`, gap = 681\n' > "$t/$DR"
    printf '### FT-12 · slice 1 PROPOSED (x)\n' > "$t/$FT"
    printf '3. **Plan quorum.** the apr lane is advisory per x. Any `do-not-implement` verdict is a STOP.\n' > "$t/$AE"
    printf 'run: bash scripts/check_reconcile.sh\n' > "$t/.github/workflows/binary-release.yml"
  }
  expect() { rc=0; run "$t" >/dev/null || rc=$?; [ "$rc" -eq "$1" ] || { echo "FAIL $2 rc=$rc want $1"; fail=1; }; }
  mk; expect 0 "good tree"
  mk; printf 'arms:\n  L3:\n    phase: merge\n    comparator_required: true\n' > "$t/$PM"; expect 1 "comparator at merge"
  mk; printf 'arms:\n  A:\n    phase: release\n    comparator_required: false\n' > "$t/$PM"; expect 1 "vacuous no comparator"
  mk; printf '| A: P₀ bp | ≥ 9,300 (x) |\n' > "$t/$EP74"; expect 1 "wrong floor"
  mk; printf '### FT-12 · slice 1 (x)\n' > "$t/$FT"; expect 1 "no state token"
  mk; printf '### FT-12 · PROPOSED RULED\n' > "$t/$FT"; expect 1 "two state tokens"
  mk; printf 'run: echo hi\n' > "$t/.github/workflows/binary-release.yml"; expect 1 "reconcile not run"
  mk; printf 'run: bash scripts/check_reconcile.sh || true\n' > "$t/.github/workflows/binary-release.yml"; expect 1 "reconcile masked"
  mk; printf '3. **Plan quorum.** all lanes vote.\n' > "$t/$AE"; expect 1 "quorum rule weakened"
  mk; rm "$t/$DR"; expect 1 "input missing"
  if [ "$fail" -eq 0 ]; then echo "self-test: 10 cases ok"; fi
  return "$fail"
}
case "${1:-}" in
  --self-test) selftest ;;
  *) run "${1:-.}" ;;
esac
