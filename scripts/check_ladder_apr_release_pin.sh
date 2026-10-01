#!/usr/bin/env bash
# check_ladder_apr_release_pin.sh — the release cells run the apr UNDER TEST, never a PATH lookup (#3715 A3).
#
# WHY. Operator 2026-09-28 17:28Z: "cells and every gate step invoke the binary UNDER TEST by absolute path
# (the rc.1 artifact) with a sha256 check; each receipt records `apr --version` SHA == rc.1 tag SHA.
# Falsifier: plant a different-sha apr first on PATH -> cell refuses. PATH lookup of apr on the gate path = 0."
# Four `apr` binaries once coexisted on one box and a bare `apr` resolved to a 26-day-old one. A cell measured
# on the wrong binary is a confident verdict about code nobody is shipping.
#
# WHAT. model_ladder.sh step 0, release mode (APR_RELEASE_SHA256 / APR_RELEASE_COMMIT set), driven through
# `--lock-probe` -- the same apr_locked every cell goes through -- in a temp root whose PATH has a PLANTED apr
# first. Every fake apr logs each invocation, so "the planted apr ran zero times" is observed, not assumed.
#
# Exit: 0 every case as expected and every mutant killed · 1 a case or mutant landed wrong · 2 could not check.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
LADDER_SH=scripts/model_ladder.sh
for t in sha256sum flock choom; do command -v "$t" > /dev/null || { echo "check_ladder_apr_release_pin: $t absent" >&2; exit 2; }; done
[ -f "$LADDER_SH" ] || { echo "check_ladder_apr_release_pin: $LADDER_SH absent" >&2; exit 2; }

W=$(mktemp -d "${TMPDIR:-/tmp}/apr-pin.XXXXXX") || exit 2
trap 'rm -rf -- "${W:?}"' EXIT
GOOD=abcdef1234567890abcdef1234567890abcdef12   # the "release commit"
OTHER=0123456789abcdef0123456789abcdef01234567
mkdir -p "$W/root" "$W/rc" "$W/path" "$W/imp"
for d in contracts scripts evidence Cargo.toml; do ln -s "$PWD/$d" "$W/root/$d"; done
LOG="$W/calls"
fake() { # <file> <label> <commit-it-claims>
  printf '#!/bin/sh\necho "%s $*" >> "%s"\n[ "$1" = --version ] && { echo "apr 0.70.0 (%s)"; exit 0; }\nexit 0\n' "$2" "$LOG" "${3:0:9}" > "$1"
  chmod +x "$1"
}
fake "$W/rc/apr" rc "$GOOD"          # the artifact under test
fake "$W/path/apr" planted "$OTHER"  # a different-sha apr, first on PATH
fake "$W/imp/apr" impostor "$GOOD"   # different bytes, but CLAIMS the release commit
# A bare name is checked in one place and executed from another: the ladder's cwd holds the real bytes (so
# -f and sha256 pass on them), while exec resolves the same name through PATH -- to an impostor claiming the
# release commit. Only the ABSOLUTE-path rule stops this one: bytes checked != bytes run.
cp "$W/rc/apr" "$W/root/rc-apr"; chmod +x "$W/root/rc-apr"
fake "$W/path/rc-apr" impostor "$GOOD"
sum() { local s; s=$(sha256sum -- "$1") && printf '%s' "${s%% *}"; }
RC_SUM=$(sum "$W/rc/apr"); PL_SUM=$(sum "$W/path/apr")

# probe <script> <APR> <sha|-> <commit|-> -> sets RC, CALLS (non---version invocations by label)
probe() {
  : > "$LOG"
  local env=(PATH="$W/path:$PATH" MODEL_LADDER_ROOT="$W/root" MODEL_LADDER_GPU_LOCK="$W/lock" MODEL_LADDER_LOCK_WAIT=5 APR="$2")
  [ "$3" != - ] && env+=(APR_RELEASE_SHA256="$3")
  [ "$4" != - ] && env+=(APR_RELEASE_COMMIT="$4")
  env "${env[@]}" timeout 60 bash "$1" --lock-probe qa cell > "$W/out" 2>&1; RC=$?
  CALLS=$(grep -v -- ' --version' "$LOG" | cut -d' ' -f1 | sort | uniq -c | tr -s ' ' | tr '\n' ';')
}

# name | APR | sha | commit | want rc | want the ONLY cell invocation (label or "none")
CASES="green-planted-on-path|$W/rc/apr|$RC_SUM|$GOOD|0|rc
bare-apr-is-a-path-lookup|apr|$RC_SUM|$GOOD|2|none
bare-name-checked-in-cwd-run-from-path|rc-apr|$RC_SUM|$GOOD|2|none
absolute-planted|$W/path/apr|$RC_SUM|$GOOD|2|none
impostor-claims-the-commit|$W/imp/apr|$RC_SUM|$GOOD|2|none
right-bytes-wrong-commit|$W/path/apr|$PL_SUM|$GOOD|2|none
half-pin-sha-only|$W/rc/apr|$RC_SUM|-|2|none
half-pin-commit-only|$W/rc/apr|-|$GOOD|2|none
sha-not-hex|$W/rc/apr|nothex|$GOOD|2|none"

run_cases() { # <script> [quiet] -> number of cases that landed wrong
  local s="$1" q="${2:-}" bad=0 name apr sha commit wrc wcall want
  while IFS='|' read -r name apr sha commit wrc wcall; do
    probe "$s" "$apr" "$sha" "$commit"
    want=$([ "$wcall" = none ] && echo "" || echo " 1 $wcall;")
    if [ "$RC" = "$wrc" ] && [ "$CALLS" = "$want" ]; then [ -z "$q" ] && echo "ok    $name (rc $RC, cell ran: ${wcall})"
    else bad=$((bad + 1)); [ -z "$q" ] && { echo "FAIL  $name: rc $RC (want $wrc), cell invocations '${CALLS}' (want '${want}')"; sed 's/^/        /' "$W/out" | tail -n 3; }
    fi
  done <<< "$CASES"
  return "$bad"
}

echo "== cases (planted apr first on PATH in every case)"
run_cases "$LADDER_SH"; bad=$?
rows=$(printf '%s\n' "$CASES" | wc -l)

echo "== mutants (each must turn a case RED)"
mutant() { # <name> <python-literal old> <new>
  local m="$W/mut-$1.sh"
  python3 - "$LADDER_SH" "$m" "$2" "$3" <<'PY' || { echo "FAIL  mutant $1: anchor not found once -- the check no longer tracks $LADDER_SH"; bad=$((bad + 1)); return; }
import sys
src, dst, old, new = sys.argv[1:]
s = open(src).read()
sys.exit(1) if s.count(old) != 1 else open(dst, "w").write(s.replace(old, new, 1))
PY
  if run_cases "$m" quiet; then echo "FAIL  mutant $1 SURVIVED"; bad=$((bad + 1)); else echo "ok    mutant $1 killed"; fi
}
mutant no-release-pin '  release_pin || exit 2' '  true || exit 2'
mutant no-absolute-check '    /*) ;;
    *) echo "decline: release mode needs' '    *) ;;
    /*) echo "decline: release mode needs'
mutant no-sha256-check '[ "$got" = "$want" ] ||' 'true ||'
mutant no-version-check '*"(${commit:0:9})"*) ;;' '*) ;;'
mutant path-lookup-in-cell 'choom -n 1000 -- "$APR" "$@"; }' 'choom -n 1000 -- apr "$@"; }'

[ "$rows" -ge 9 ] || { echo "VACUOUS $rows case(s), fewer than the 9 declared" >&2; exit 1; }
[ "$bad" = 0 ] && { echo "PASS $rows cases + 5 mutants: release cells run only the pinned artifact; PATH lookups of apr = 0"; exit 0; }
echo "RED $bad"; exit 1
