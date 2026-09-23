#!/usr/bin/env bash
# check_ont_complete.sh -- G-ONT, a MUST-RED gate of 0.70.0 (#4045): ONT-001 is FINISHED, on both sides.
#
# Operator, verbatim via the release cop (2026-09-23): "the ontology spec MUST be finished in .7". Every ONT row of
# infra/docs/specifications/paiml-ontology.md is closed with its done_when MEASURED:
#   1. the spec is read at a PINNED infra commit (--pin <sha>): an unpinned or moved spec is RED (#3269's lesson: a
#      spec that exists on one box only makes every verdict unverifiable);
#   2. infra's own scripts/ont/precondition-lint.sh over that spec and the ONT ledger reports rows > 0, unbound=0 and
#      violations=0 -- its summary line is REQUIRED (0 rows is the fleet's signature vacuous pass: a decline);
#   3. EVERY scripts/ont/done_when/ONT-*.sh probe exits 0 against the release commit (WT) and the pinned infra
#      (WT_INFRA). A probe that exits 2 DECLINED: not done, RED, never skipped.
#
#   bash scripts/check_ont_complete.sh --infra <infra checkout> --pin <infra sha> [--wt <aprender tree>] [--ledger <f>]
#   bash scripts/check_ont_complete.sh --self-test
# exit 0 = ONT-001 complete . 1 = not complete (every open row named) . 2 = usage / vacuous
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
INFRA=""; PIN=""; WT="$(pwd)"; LEDGER=""; SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --infra) INFRA="$2"; shift 2 ;;
    --pin) PIN="$2"; shift 2 ;;
    --wt) WT="$2"; shift 2 ;;
    --ledger) LEDGER="$2"; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    *) echo "usage: $0 --infra <dir> --pin <sha> [--wt <dir>] [--ledger <file>] | --self-test" >&2; exit 2 ;;
  esac
done

judge() { # judge <infra> <pin> <wt> <ledger> -> ok/FAIL lines; rc 0 complete, 1 not, 2 vacuous/usage
  local infra=$1 pin=$2 wt=$3 ledger=$4 head spec lint rows bound unbound viol p rc bad=0 n=0
  [ -d "$infra/.git" ] || [ -f "$infra/.git" ] || { echo "decline: --infra $infra is not a git checkout"; return 2; }
  head=$(git -C "$infra" rev-parse HEAD 2> /dev/null)
  if [ -z "$pin" ] || [ "$head" != "$(git -C "$infra" rev-parse --verify --quiet "$pin^{commit}" 2> /dev/null)" ]; then
    echo "FAIL  the ONT spec is not read at the pinned infra commit (HEAD ${head:0:9}, --pin '${pin:-none}') -- an unpinned spec is unverifiable"
    return 1
  fi
  spec="$infra/docs/specifications/paiml-ontology.md"
  [ -f "$spec" ] || { echo "decline: $spec not found at ${head:0:9}"; return 2; }
  [ -n "$ledger" ] || ledger="$infra/docs/audits/ONT-001/ledger.jsonl"
  echo "note  spec $(sha256sum "$spec" | cut -c1-16) at infra ${head:0:9}; ledger $ledger"
  lint=$(cd "$infra" && bash scripts/ont/precondition-lint.sh docs/specifications/paiml-ontology.md --ledger "$ledger" 2>&1)
  if [[ "$lint" =~ rows=([0-9]+)\ bound=([0-9]+)\ unbound=([0-9]+)\ .*violations=([0-9]+) ]]; then
    rows=${BASH_REMATCH[1]}; bound=${BASH_REMATCH[2]}; unbound=${BASH_REMATCH[3]}; viol=${BASH_REMATCH[4]}
  else
    echo "decline: precondition-lint printed no rows=/bound=/unbound=/violations= summary -- a count it did not print is not a pass"
    return 2
  fi
  [ "$rows" -gt 0 ] || { echo "decline: precondition-lint parsed 0 rows -- the vacuous pass"; return 2; }
  if [ "$unbound" != 0 ] || [ "$viol" != 0 ]; then
    echo "FAIL  ONT rows: $rows, bound $bound, UNBOUND $unbound, violations $viol -- every row must be bound in the ledger"
    grep -E '^(BLOCKED-EXTERNAL|precondition-lint: ONT-)' <<< "$lint" | head -20 | sed 's/^/        /'
    bad=1
  else
    echo "ok    ONT rows: $rows, all bound, 0 violations"
  fi
  for p in "$infra"/scripts/ont/done_when/ONT-*.sh; do
    [ -f "$p" ] || continue
    n=$((n + 1))
    (WT="$wt" WT_INFRA="$infra" LEDGER="$ledger" bash "$p" > /dev/null 2>&1); rc=$?
    case "$rc" in
      0) : ;;
      2) echo "FAIL  $(basename "$p" .sh): its done_when DECLINED (exit 2) -- a declined probe is not done"; bad=1 ;;
      *) echo "FAIL  $(basename "$p" .sh): its done_when is not met (exit $rc)"; bad=1 ;;
    esac
  done
  [ "$n" -gt 0 ] || { echo "decline: no scripts/ont/done_when/ONT-*.sh probe at ${head:0:9}"; return 2; }
  # EVERY spec row is measured: a row the spec names with no probe file would otherwise never be run (quorum, Fable)
  local r
  for r in $(grep -oE '^\*\*ONT-[A-Za-z0-9]+\*\* ' "$spec" | tr -d '* '); do
    [ -f "$infra/scripts/ont/done_when/$r.sh" ] || { echo "FAIL  $r: the spec names this row and it has no done_when probe -- an unmeasured row is not done"; bad=1; }
  done
  [ "$bad" = 0 ] && echo "ok    ONT-001 complete: $n done_when probe(s) pass at infra ${head:0:9}"
  # the LAST line names the pinned infra commit: the watch keeps a gate's last line in its verdict
  echo "G-ONT $([ "$bad" = 0 ] && echo complete || echo NOT-complete) at infra $head (spec $(sha256sum "$spec" | cut -c1-16))"
  return "$bad"
}

if [ "$SELF_TEST" = 1 ]; then
  exec 3>&1
  T=$(mktemp -d); tbad=0
  fixture() { # fixture <summary line|-> <probe rcs...>: a scratch infra repo
    local f i=0; f=$(mktemp -d -p "$T" infra-XXXXXX)
    mkdir -p "$f/scripts/ont/done_when" "$f/docs/specifications" "$f/docs/audits/ONT-001"
    echo "# ONT-001" > "$f/docs/specifications/paiml-ontology.md"; : > "$f/docs/audits/ONT-001/ledger.jsonl"
    [ "${FX_EXTRA_ROW:-0}" = 1 ] && printf '**ONT-99** `aprender` \xc2\xb7 "a row with no probe"\n' >> "$f/docs/specifications/paiml-ontology.md"
    if [ "$1" = - ]; then printf '#!/bin/bash\necho nothing\n' > "$f/scripts/ont/precondition-lint.sh"
    else printf '#!/bin/bash\necho "%s"\n' "$1" > "$f/scripts/ont/precondition-lint.sh"; fi
    shift
    for rc in "$@"; do i=$((i + 1)); printf '#!/bin/bash\nexit %s\n' "$rc" > "$f/scripts/ont/done_when/ONT-$i.sh"
      printf '**ONT-%s** `aprender` \xc2\xb7 "row %s"\n' "$i" "$i" >> "$f/docs/specifications/paiml-ontology.md"; done
    git -C "$f" init -q && git -C "$f" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t add -A && \
      git -C "$f" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm f
    printf '%s' "$f"
  }
  row() { # row <name> <want rc> <needle> <infra> [pin]
    local out rc pin=${5:-$(git -C "$4" rev-parse HEAD)}
    out=$(judge "$4" "$pin" "$T" ""); rc=$?
    if [ "$rc" = "$2" ] && grep -qF -- "$3" <<< "$out"; then echo "ok    $1"
    else echo "FAIL  $1 -- rc $rc: $(tr '\n' ' ' <<< "$out" | cut -c1-200)"; tbad=1; fi
  }
  GOOD="precondition-lint: rows=3 bound=3 unbound=0 probe_paths=2 declared=1 violations=0"
  row "complete" 0 "ONT-001 complete: 2 done_when" "$(fixture "$GOOD" 0 0)"
  row "unbound-row-is-red" 1 "UNBOUND 1" "$(fixture "precondition-lint: rows=3 bound=2 unbound=1 probe_paths=2 declared=1 violations=0" 0 0)"
  row "violation-is-red" 1 "violations 2" "$(fixture "precondition-lint: rows=3 bound=3 unbound=0 probe_paths=2 declared=1 violations=2" 0 0)"
  row "probe-unmet-is-red" 1 "ONT-2: its done_when is not met (exit 1)" "$(fixture "$GOOD" 0 1)"
  row "probe-declined-is-red" 1 "ONT-1: its done_when DECLINED" "$(fixture "$GOOD" 2 0)"
  row "no-summary-declines" 2 "no rows=/bound=" "$(fixture - 0)"
  row "zero-rows-declines" 2 "parsed 0 rows" "$(fixture "precondition-lint: rows=0 bound=0 unbound=0 probe_paths=0 declared=0 violations=0" 0)"
  f=$(fixture "$GOOD" 0); row "unpinned-is-red" 1 "not read at the pinned infra commit" "$f" 0000000000000000000000000000000000000000
  row "no-probes-declines" 2 "no scripts/ont/done_when" "$(fixture "$GOOD")"
  row "row-without-probe-is-red" 1 "ONT-99: the spec names this row and it has no done_when probe" "$(FX_EXTRA_ROW=1 fixture "$GOOD" 0 0)"
  row "verdict-names-the-pinned-infra" 0 "G-ONT complete at infra" "$(fixture "$GOOD" 0 0)"
  if [ -n "$T" ] && [ "$T" != "/" ] && [ -d "$T" ]; then rm -rf -- "$T"; fi
  if [ "${ONT_MUTANTS:-1}" = 1 ] && [ "$tbad" = 0 ]; then   # each rule deleted in a copy; its NAMED row must go RED
    M=$(mktemp -d)
    while IFS='~' read -r label must old new; do
      [ -n "$label" ] || continue
      # the anchor is mutated in the JUDGE (the code above the self-test), never in this table, which quotes it
      python3 -c 'import sys
s = open(sys.argv[1]).read(); code, cut, rest = s.partition("\nif [ \"$SELF_TEST\" = 1 ]; then")
assert code.count(sys.argv[3]) == 1
open(sys.argv[2], "w").write(code.replace(sys.argv[3], sys.argv[4]) + cut + rest)' \
        "$0" "$M/$label.sh" "$old" "$new" 2> /dev/null || { echo "FAIL  mutant $label did not apply"; tbad=1; continue; }
      mo=$(ONT_MUTANTS=0 bash "$M/$label.sh" --self-test 2>&1)
      if grep -q "^FAIL  $must " <<< "$mo"; then echo "ok    mutant $label killed by $must"; else echo "FAIL  mutant $label SURVIVED $must"; tbad=1; fi
    done <<'MUT'
unbound-ignored~unbound-row-is-red~if [ "$unbound" != 0 ] || [ "$viol" != 0 ]; then~if [ "$viol" != 0 ]; then
violation-ignored~violation-is-red~if [ "$unbound" != 0 ] || [ "$viol" != 0 ]; then~if [ "$unbound" != 0 ]; then
decline-accepted~probe-declined-is-red~      2) echo "FAIL  $(basename "$p" .sh): its done_when DECLINED (exit 2) -- a declined probe is not done"; bad=1 ;;~      2) : ;;
unmet-accepted~probe-unmet-is-red~      *) echo "FAIL  $(basename "$p" .sh): its done_when is not met (exit $rc)"; bad=1 ;;~      *) : ;;
no-summary-passes~no-summary-declines~    echo "decline: precondition-lint printed no rows=~    return 0; echo "decline: precondition-lint printed no rows=
zero-rows-pass~zero-rows-declines~  [ "$rows" -gt 0 ] || { echo "decline: precondition-lint parsed 0 rows -- the vacuous pass"; return 2; }~  :
pin-ignored~unpinned-is-red~  if [ -z "$pin" ] || [ "$head" != "$(git -C "$infra" rev-parse --verify --quiet "$pin^{commit}" 2> /dev/null)" ]; then~  if false; then
row-probe-unchecked~row-without-probe-is-red~    [ -f "$infra/scripts/ont/done_when/$r.sh" ] || { echo~    true || { echo
no-probe-pass~no-probes-declines~  [ "$n" -gt 0 ] || { echo "decline: no scripts/ont/done_when/ONT-*.sh probe at ${head:0:9}"; return 2; }~  :
MUT
    if [ -n "$M" ] && [ "$M" != "/" ] && [ -d "$M" ]; then rm -rf -- "$M"; fi
  fi
  echo "check_ont_complete self-test: $([ "$tbad" = 0 ] && echo PASS || echo FAIL)"
  exit "$tbad"
fi

if [ -z "$INFRA" ]; then
  # CI (guard_tree's bare run): REPORT-only, per the release cop -- G-ONT is RED by design until ONT-001 is done and
  # GATES only at the 0.70.0 release, where it runs with --infra/--pin. The self-test (the gate's own proof) runs in CI.
  echo "REPORT G-ONT: no --infra/--pin given -- the live gate runs at the 0.70.0 release (#4045); its self-test is the CI row"
  exit 0
fi
judge "$INFRA" "$PIN" "$WT" "$LEDGER"
