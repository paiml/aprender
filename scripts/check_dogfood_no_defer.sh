#!/usr/bin/env bash
# check_dogfood_no_defer.sh — no dogfood row may be DEFERRED (#3957 F1b).
#
# WHY. Operator doctrine 2026-09-23: "no defer" -- a capability is ORACLE-PROVEN or RED, with
# no third state. scripts/dogfood.sh had three: `mark coverage DEFER` (#3839, deferred WORK),
# `mark publish-dry-run DEFER` and a generic hatch that turned any declared gate's
# `DEFERRED: ` log line into DEFER, legal in the pre-publish phase. Operator ruling (a),
# 2026-09-23: the hatch goes, coverage is RED, and the two rows that are unmeasurable by
# CONSTRUCTION before a publish -- `publish-dry-run` (a workspace root cannot dry-run before
# its members are on the registry) and `declared:check_multiplatform_dogfood` (no host can
# install a version that is not on crates.io) -- become a NAMED post-publish
# obligation, status OPEN: legal in --phase pre-publish only, only for those two names,
# listed on the receipt as OPEN and never as passed. The post-publish dogfood runs the same
# rows with measurements, and mark() refuses OPEN in any other phase, so an unmet obligation
# FAILs there.
#
# HOW. The three functions that decide -- strip_ansi, mark, classify_declared -- are LIFTED
# from dogfood.sh and driven by a case table, so this tests the shipped code. Plus two static
# rules: dogfood.sh calls `mark <row> DEFER` nowhere, and no declared gate prints `DEFERRED: `.
#
# Exit: 0 all cases as expected · 1 a case landed wrong · 2 could not check.
#       --self-test: 0 when each planted mutation turns this RED, 1 otherwise.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
SCRIPT="$ROOT/scripts/dogfood.sh"
SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --script) [ $# -ge 2 ] || { echo "--script needs a value" >&2; exit 2; }; SCRIPT="$2"; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    *) echo "check_dogfood_no_defer: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

lift() { # lift <script> -> the obligation list and the three deciding functions, or return 2
  local src="$1" fn body out=""
  [ -f "$src" ] || { echo "  cannot read $src" >&2; return 2; }
  # The closed obligation list is part of the rule: lifted with it, never re-declared here.
  out=$(grep -m1 '^POST_PUBLISH_OBLIGATIONS=' "$src") || { echo "  $src declares no POST_PUBLISH_OBLIGATIONS" >&2; return 2; }
  out="$out
"
  for fn in strip_ansi mark classify_declared; do
    body=$(awk -v F="^$fn\\\\(\\\\) \\\\{" '$0 ~ F {f=1} f{print} f && (/^\}$/ || /; }$/){exit}' "$src")
    [ -n "$body" ] || { echo "  $src defines no $fn() -- the rule this checks is gone" >&2; return 2; }
    out="$out$body
"
  done
  printf '%s' "$out"
}

# run_case <script> <kind> <phase> <row name> <status-or-rc> <log text> -> prints "<RESULT> <bad>"
#   kind mark:     mark <row name> <status> "note"          bad = FAILED after the call
#   kind declared: classify_declared <row name> gate.sh <rc> <log>   bad = its return status
run_case() {
  local src="$1" kind="$2" phase="$3" name="$4" arg="$5" text="$6" tmp log rc
  tmp=$(mktemp) || return 2; log=$(mktemp) || { rm -f "$tmp"; return 2; }
  lift "$src" > "$tmp" || { rm -f "$tmp" "$log"; return 2; }
  printf '%s\n' "$text" > "$log"
  cat >> "$tmp" <<'RUN'
NAMES=(); RESULTS=(); NOTES=(); FAILED=0
if [ "$KIND" = mark ]; then mark "$NAME" "$ARG" "note" > /dev/null; bad=$FAILED
else classify_declared "$NAME" gate.sh "$ARG" "$LOG" > /dev/null; bad=$?; fi
printf '%s %s\n' "${RESULTS[${#RESULTS[@]}-1]:-NONE}" "$bad"
RUN
  KIND="$kind" DOGFOOD_PHASE="$phase" NAME="$name" ARG="$arg" LOG="$log" bash "$tmp" 2> /dev/null; rc=$?
  rm -f "$tmp" "$log"
  return $rc
}

# name|kind|phase|row|status-or-rc|log text|want "<RESULT> <bad>"
cases() {
cat <<'CASES'
hatch-deferred-line-pre-publish|declared|pre-publish|declared:check_multiplatform_dogfood|0|DEFERRED: 0.69.1 is not on crates.io yet|FAIL 1
hatch-deferred-line-full|declared|full|declared:check_multiplatform_dogfood|0|DEFERRED: 0.69.1 is not on crates.io yet|FAIL 1
hatch-deferred-any-gate|declared|pre-publish|declared:check_model_ladder|0|DEFERRED: something|FAIL 1
obligation-allowlisted-pre-publish|declared|pre-publish|declared:check_multiplatform_dogfood|0|OPEN-OBLIGATION: 0.69.1 is not on crates.io yet|OPEN 0
obligation-other-gate-refused|declared|pre-publish|declared:check_model_ladder|0|OPEN-OBLIGATION: trust me|FAIL 1
obligation-post-publish-refused|declared|post-publish|declared:check_multiplatform_dogfood|0|OPEN-OBLIGATION: still not measured|FAIL 1
obligation-with-nonzero-rc-refused|declared|pre-publish|declared:check_multiplatform_dogfood|1|OPEN-OBLIGATION: and it also failed|FAIL 1
declared-rc0-pass|declared|pre-publish|declared:check_model_ladder|0|ok    every required rung green|PASS 0
declared-rc1-fail|declared|pre-publish|declared:check_model_ladder|1|RED   the release claims a capability|FAIL 1
mark-defer-pre-publish-is-fail|mark|pre-publish|coverage|DEFER||FAIL 1
mark-defer-full-is-fail|mark|full|publish-dry-run|DEFER||FAIL 1
mark-open-publish-dry-run-pre-publish|mark|pre-publish|publish-dry-run|OPEN||OPEN 0
mark-open-coverage-refused|mark|pre-publish|coverage|OPEN||FAIL 1
mark-open-post-publish-refused|mark|post-publish|publish-dry-run|OPEN||FAIL 1
CASES
}

run_table() { # -> 0 all as expected, 1 a case wrong, 2 could not check
  local src="$1" rc=0 name kind phase row arg text want got
  while IFS='|' read -r name kind phase row arg text want; do
    [ -n "$name" ] || continue
    got=$(run_case "$src" "$kind" "$phase" "$row" "$arg" "$text") || return 2
    if [ "$got" = "$want" ]; then printf '  ok    %-40s %s\n' "$name" "$got"
    else printf '  FAIL  %-40s got %s, want %s\n' "$name" "$got" "$want"; rc=1; fi
  done < <(cases)
  return $rc
}

static_rules() { # <dogfood.sh> -> 0 when no row is DEFERred and no declared gate prints DEFERRED:
  local src="$1" rc=0 hits g
  # An UNCOMMENTED `mark <row> DEFER` call. `[^#]*` keeps a comment that names it from counting.
  hits=$(grep -nE '^[^#]*\bmark[[:space:]]+[^[:space:]]+[[:space:]]+DEFER\b' "$src" || true)
  if [ -n "$hits" ]; then printf '  FAIL  %s still marks a row DEFER:\n%s\n' "$src" "$hits"; rc=1
  else printf '  ok    %s marks no row DEFER\n' "$(basename "$src")"; fi
  for g in "$ROOT"/scripts/check_*.sh; do
    [ "$g" = "$ROOT/scripts/check_dogfood_no_defer.sh" ] && continue
    [ "$g" = "$ROOT/scripts/check_model_ladder.sh" ] && continue   # names the marker only to pin its absence
    if grep -qE "^[^#]*printf[^#]*'DEFERRED: " "$g"; then printf '  FAIL  %s prints a DEFERRED: line\n' "$g"; rc=1; fi
  done
  [ "$rc" = 0 ] && printf '  ok    no gate script prints a DEFERRED: line\n'
  return $rc
}

if [ "$SELF_TEST" = 1 ]; then
  run_table "$SCRIPT" > /dev/null && static_rules "$SCRIPT" > /dev/null \
    || { echo "SELF-TEST FAILED: the shipped dogfood.sh is already red" >&2; exit 1; }
  echo "self-test: the shipped dogfood.sh -- GREEN (expected)"
  bad=0
  mut() { # mut <label> <table|static> <sed expression>
    local m; m=$(mktemp) || exit 2
    sed "$3" "$SCRIPT" > "$m"
    if cmp -s "$SCRIPT" "$m"; then echo "SELF-TEST INCONCLUSIVE: mutant $1 changed nothing" >&2; bad=1; rm -f "$m"; return; fi
    if { [ "$2" = table ] && run_table "$m" > /dev/null 2>&1; } || { [ "$2" = static ] && static_rules "$m" > /dev/null 2>&1; }; then
      echo "SELF-TEST FAILED: mutant $1 SURVIVED"; bad=1
    else echo "self-test: mutant $1 -- RED (expected)"; fi
    rm -f "$m"
  }
  mut hatch-restored  table  's/^  if \[ -n "\$defer" \]; then$/  if [ -n "$defer" ] \&\& false; then/'
  mut defer-legal     table  's/^  if \[ "\$st" = DEFER \]; then$/  if false; then/'
  mut any-obligation  table  's/^    case " \$POST_PUBLISH_OBLIGATIONS " in \*" \$1 "\*) ;; \*) st=FAIL/    case " $POST_PUBLISH_OBLIGATIONS " in *) ;; *) st=FAIL/'
  mut open-any-phase  table  's/^  if \[ "\$st" = OPEN \] && \[ "\$DOGFOOD_PHASE" != pre-publish \]; then$/  if false; then/'
  mut coverage-defer  static 's/^    mark coverage FAIL "measured/    mark coverage DEFER "measured/'
  [ "$bad" = 0 ] && echo "self-test: PASS -- red when the hatch returns, when DEFER is legal, when any row may be OPEN, when OPEN outlives pre-publish, and when coverage defers"
  exit "$bad"
fi

echo "dogfood: no row is DEFERRED; only the two named post-publish obligations may be OPEN (#3957 F1b)"
rc=0
run_table "$SCRIPT" || rc=$?
[ "$rc" = 2 ] && exit 2
static_rules "$SCRIPT" || rc=1
[ "$rc" = 0 ] && echo "OK" || echo "FAIL: a dogfood row can still be deferred (#3957 F1b)"
exit "$rc"
