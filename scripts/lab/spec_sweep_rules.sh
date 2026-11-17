#!/usr/bin/env bash
# scripts/lab/spec_sweep_rules.sh -- LAB (C324 items 4-5), spec-sweep 2026-10-31, epic #6809.
# Enforces eight spec rules that had no blocking check: #6610 (PP-6) #6613 (slice-5 floor) #6614 (FT-12 state)
# #6615 (T-5 reconcile blocks release) #6616 (plan-quorum apr lane advisory) #6641 (nemawashi advisory-until-ack) #6617 (Track D non-blocking). Each rule is a function that prints
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
MG=docs/specifications/aprender-distribute/multi-gpu-cpu-machine-tasks.md
PP66=docs/specifications/PP-066-release-spec.md
RULES=(pp6 slice5 ft12 t5 pq nemawashi trackd)
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

# 6615 T-5: release.yml (the release decision surface) really runs scripts/check_reconcile.sh and does not mask or
# disable it. "Runs" = the script named inside a step's `run:` body, comments stripped (a step name or a comment
# naming it is not a run). RED when: no such step in release.yml; the step has `|| true`, continue-on-error or an
# `if:` containing false; or a step that runs one_command.sh / publish_tagged.sh / cargo publish comes before it.
# binary-release.yml and release-gates-nightly.yml are scanned for masking too, never as a stand-in for release.yml.
t5_scan() {
  awk '
    { l = $0; sub(/^[ \t]*#.*$/, "", l); sub(/[ \t]+#.*$/, "", l) }
    match(l, /^ *- /) { if (!started || RLENGTH - 2 <= ind) { step++; ind = RLENGTH - 2; started = 1; inrun = 0 } }
    l ~ /^ *(- )?run:/ { inrun = 1; runind = index(l, "run:") - 1; txt = l }
    !(l ~ /^ *(- )?run:/) && inrun { if (l ~ /[^ \t]/) { match(l, /^ */); if (RLENGTH > runind) txt = l; else { inrun = 0; txt = "" } } else txt = "" }
    !inrun { txt = "" }
    txt ~ /check_reconcile\.sh/ { runs[step] = 1; if (!first) first = step; if (txt ~ /\|\|[ \t]*true/) masked = 1 }
    txt ~ /(one_command|publish_tagged)\.sh|cargo +publish/ { if (!pub) pub = step }
    l ~ /^ *(- )?if:.*false/ { iff[step] = 1 }
    l ~ /continue-on-error/ { coe[step] = 1 }
    END {
      if (first) print "RUNS"
      if (masked) print "MASKED"
      for (k in runs) { if (coe[k]) print "COE"; if (iff[k]) print "IFFALSE" }
      if (first && pub && pub < first) print "LATE"
    }' "$1"
}
t5() {
  local f out w="$1/.github/workflows"
  [ -d "$w" ] || { echo "RED T-5: no workflows dir"; return 0; }
  if [ -f "$w/release.yml" ]; then
    out=$(t5_scan "$w/release.yml")
    grep -q '^RUNS$' <<<"$out" || echo "RED T-5: release.yml has no step that runs scripts/check_reconcile.sh"
    grep -q '^LATE$' <<<"$out" && echo "RED T-5: release.yml runs check_reconcile.sh after a tag/publish step"
  else
    echo "RED T-5: release.yml missing"
  fi
  for f in "$w"/release.yml "$w"/binary-release.yml "$w"/release-gates-nightly.yml; do
    [ -f "$f" ] || continue
    out=$(t5_scan "$f")
    grep -q '^MASKED$' <<<"$out" && echo "RED T-5: $(basename "$f") masks check_reconcile.sh"
    grep -q '^COE$' <<<"$out" && echo "RED T-5: $(basename "$f") step runs check_reconcile.sh with continue-on-error"
    grep -q '^IFFALSE$' <<<"$out" && echo "RED T-5: $(basename "$f") step runs check_reconcile.sh under an if: that is false"
  done
  return 0
}

# 6616 plan quorum: the rule line says the apr lane is advisory and a do-not-implement verdict is a STOP.
pq() {
  need "$1" "$AE" || return 0
  local l; l=$(grep -E '^[0-9]+\. \*\*Plan quorum\.\*\*' "$1/$AE" | head -n 1)
  [ -n "$l" ] || { echo "RED plan-quorum: rule line missing"; return 0; }
  grep -q 'apr lane is advisory' <<<"$l" || echo "RED plan-quorum: apr lane not stated advisory"
  grep -q 'do-not-implement' <<<"$l" || echo "RED plan-quorum: no do-not-implement STOP"
}

# 6641 nemawashi: the multi-gpu spec still says task assignment is advisory until the worker acknowledges.
# Spec-line half only: no deterministic definition of "no blocking-before-ack branch" exists for the code, so none is checked.
nemawashi() {
  need "$1" "$MG" || return 0
  grep -q -F -- "Task assignment is advisory until worker acknowledges" "$1/$MG" || echo "RED nemawashi: advisory-until-ack line missing from $MG"
}


# 6617 trackd: PP-066 Track D rows (D-Ndoc) are non-blocking: the Track D heading says so and no other row's
# blockers cell (table column 4, or a "blockers:" card line) names one. Never a merge or release hold.
trackd() {
  need "$1" "$PP66" || return 0
  awk -F'|' '
    /^#+ / { d = ($0 ~ /^### Track D/); if (d && $0 ~ /non-blocking/) nb = 1 }
    d && /^\| D-[0-9]+doc \|/ { ids++ }
    !d && /^\|/ && $4 ~ /D-[0-9]+doc/ { printf "RED track-d: line %d row%s names a Track D row as a blocker\n", NR, $2 }
    !d && /blockers: [^·]*D-[0-9]+doc/ { printf "RED track-d: line %d card names a Track D row as a blocker\n", NR }
    END { if (!nb) print "RED track-d: no Track D heading marked non-blocking"; if (!ids) print "RED track-d: no Track D rows found (vacuous)" }
  ' "$1/$PP66"
}

run() {
  local root=$1 out reds=0 r
  [ -d "$root" ] || { echo "usage: no such dir $root" >&2; return 2; }
  for r in "${RULES[@]}"; do
    out=$($r "$root"); if [ -n "$out" ]; then echo "$out"; reds=$((reds + 1)); fi
  done
  if [ "$reds" -eq 0 ]; then echo "GREEN: ${#RULES[@]} rules"; return 0; fi
  echo "RED: $reds of ${#RULES[@]} rules"; return 1
}

selftest() {
  local t rc fail=0
  t=$(mktemp -d)
  _rm() {
    local v="${t:-}"
    case "$v" in /tmp/?*|/var/folders/?*|"${TMPDIR:-/tmp}"/?*) ;; *) return 0 ;; esac
    [ -n "$v" ] && [ "$v" != "/" ] && rm -rf -- "$v" || :
  }
  mk() {
    _rm; mkdir -p "$t/scripts" "$t/docs/specifications" "$t/.github/workflows"
    printf 'arms:\n  A:\n    phase: release\n    comparator_required: false\n  L3:\n    phase: release\n    comparator_required: true\n' > "$t/$PM"
    printf '| A: P₀ bp | ≥ 9,364 (x) |\n' > "$t/$EP74"
    printf 'With `B = 8819`, gap = 681\n' > "$t/$DR"
    printf '### FT-12 · slice 1 PROPOSED (x)\n' > "$t/$FT"
    printf '3. **Plan quorum.** the apr lane is advisory per x. Any `do-not-implement` verdict is a STOP.\n' > "$t/$AE"
    printf 'jobs:\n  r:\n    steps:\n      - name: reconcile\n        run: bash scripts/check_reconcile.sh "v$V"\n' > "$t/.github/workflows/release.yml"
    mkdir -p "$t/${MG%/*}"; printf -- '- Task assignment is advisory until worker acknowledges\n' > "$t/$MG"
    mkdir -p "$t/${PP66%/*}"; printf '### Track D\n| id | title | blockers |\n|---|---|---|\n| D-1doc | x | G-11 |\n### Track D - docs (non-blocking)\n## s6\n| W-B | x | S-3 |\n' > "$t/$PP66"
  }
  expect() { rc=0; run "$t" >/dev/null || rc=$?; [ "$rc" -eq "$1" ] || { echo "FAIL $2 rc=$rc want $1"; fail=1; }; }
  mk; expect 0 "good tree"
  mk; printf 'arms:\n  L3:\n    phase: merge\n    comparator_required: true\n' > "$t/$PM"; expect 1 "comparator at merge"
  mk; printf 'arms:\n  A:\n    phase: release\n    comparator_required: false\n' > "$t/$PM"; expect 1 "vacuous no comparator"
  mk; printf '| A: P₀ bp | ≥ 9,300 (x) |\n' > "$t/$EP74"; expect 1 "wrong floor"
  mk; printf '### FT-12 · slice 1 (x)\n' > "$t/$FT"; expect 1 "no state token"
  mk; printf '### FT-12 · PROPOSED RULED\n' > "$t/$FT"; expect 1 "two state tokens"
  mk; printf 'run: echo hi\n' > "$t/.github/workflows/release.yml"; expect 1 "reconcile not run in release.yml"
  mk; printf 'run: bash scripts/check_reconcile.sh || true\n' > "$t/.github/workflows/release.yml"; expect 1 "reconcile masked in release.yml"
  mk; printf 'run: echo hi\n' > "$t/.github/workflows/release.yml"; printf 'run: bash scripts/check_reconcile.sh\n' > "$t/.github/workflows/release-gates-nightly.yml"; expect 1 "nightly hit does not stand in for release.yml"
  mk; printf 'jobs:\n  r:\n    steps:\n      - name: reconcile\n        continue-on-error: true\n        run: bash scripts/check_reconcile.sh "v$V"\n      - name: next\n        run: echo hi\n' > "$t/.github/workflows/release.yml"; expect 1 "step-level continue-on-error"
  mk; printf 'jobs:\n  r:\n    steps:\n      - name: other\n        continue-on-error: true\n        run: echo hi\n      - name: reconcile\n        run: bash scripts/check_reconcile.sh "v$V"\n' > "$t/.github/workflows/release.yml"; expect 0 "continue-on-error on another step is fine"
  mk; printf '# check_reconcile.sh R1\njobs:\n  r:\n    steps:\n      - name: check_reconcile.sh gate\n        run: echo hi # scripts/check_reconcile.sh\n' > "$t/.github/workflows/release.yml"; expect 1 "mention only in comment and step name"
  mk; printf 'jobs:\n  r:\n    steps:\n      - name: reconcile\n        if: false\n        run: bash scripts/check_reconcile.sh "v$V"\n' > "$t/.github/workflows/release.yml"; expect 1 "reconcile step with if: false"
  mk; printf 'jobs:\n  r:\n    steps:\n      - name: release\n        run: bash scripts/release/one_command.sh "$V"\n      - name: reconcile\n        run: bash scripts/check_reconcile.sh "v$V"\n' > "$t/.github/workflows/release.yml"; expect 1 "reconcile after the release step"
  mk; printf 'jobs:\n  r:\n    steps:\n      - name: reconcile\n        run: |\n          set -u\n          bash scripts/check_reconcile.sh "v$V"\n      - name: release\n        run: bash scripts/release/one_command.sh "$V"\n' > "$t/.github/workflows/release.yml"; expect 0 "multi-line run body before the release step"
  mk; printf '3. **Plan quorum.** all lanes vote.\n' > "$t/$AE"; expect 1 "quorum rule weakened"
  mk; rm "$t/$DR"; expect 1 "input missing"
  mk; printf -- '- Task assignment is final\n' > "$t/$MG"; expect 1 "advisory-until-ack line gone"
  mk; rm "$t/$MG"; expect 1 "multi-gpu spec missing"
  mk; printf '| W-B | x | D-1doc |\n' >> "$t/$PP66"; expect 1 "Track D row named as blocker"
  mk; printf -- '- a - blockers: D-1doc - x\n' >> "$t/$PP66"; expect 1 "Track D named in card blockers"
  mk; printf '### Track D\n| D-1doc | x | G-11 |\n' > "$t/$PP66"; expect 1 "heading not non-blocking"
  mk; printf '### Track D - docs (non-blocking)\n| W-B | x | S-3 |\n' > "$t/$PP66"; expect 1 "no Track D rows (vacuous)"
  if [ "$fail" -eq 0 ]; then echo "self-test: 23 cases ok"; fi
  return "$fail"
}
case "${1:-}" in
  --self-test) selftest ;;
  *) run "${1:-.}" ;;
esac
