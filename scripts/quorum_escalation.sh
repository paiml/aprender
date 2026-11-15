#!/usr/bin/env bash
# quorum_escalation.sh - #3562 "two consecutive non-passing plan quorums must escalate
# automatically", built (PMAT-6101). A LAB check: it runs at night, it cannot block a merge.
#
#   bash scripts/quorum_escalation.sh <receipts-dir> <escalations-file> [owner]
#   bash scripts/quorum_escalation.sh --self-test
#
# Reads gate-quorum receipts (`triage-quorum-<N>-gate-*.md`) oldest first by mtime. A chain is
# the PMAT-<n> ticket named in the receipt's first line (else the PR number). A receipt PASSes
# only on an N/M tally with N = M and no FAIL / NOT PASS beside it; a tally that cannot be read
# is NOT MEASURED, never a pass. On the SECOND consecutive non-PASS receipt of a chain, one line
#   ESCALATE ticket=<chain> receipts=<first>,<second> owner=<owner>
# is appended to <escalations-file> (once: a rerun does not repeat it).
# EXIT: 0 nothing new to escalate; 1 an escalation was written; 2 not measured (empty dir,
# or an unreadable receipt in a chain).
set -uo pipefail

verdict() { # receipt file -> PASS | FAIL | NM
  local l
  l=$(grep -m1 -E '(TALLY|VERDICT|RESULT)[^0-9]*[0-9]/[0-9]|(PASS|FAIL)[^0-9]*[0-9]/[0-9]' "$1") || { echo NM; return; }
  case "$l" in *"NOT PASS"* | *FAIL*) echo FAIL; return ;; esac
  if [[ $l =~ ([0-9])/([0-9]) ]] && [ "${BASH_REMATCH[1]}" = "${BASH_REMATCH[2]}" ]; then echo PASS; else echo FAIL; fi
}

chain() { # receipt file -> ticket id
  local h b
  h=$(head -n1 "$1")
  if [[ $h =~ (PMAT-[0-9]+) ]]; then echo "${BASH_REMATCH[1]}"; return; fi
  b=$(basename "$1")
  if [[ $b =~ ^triage-quorum-([0-9]+)- ]]; then echo "pr${BASH_REMATCH[1]}"; else echo "$b"; fi
}

run() { # dir esc owner
  local dir=$1 esc=$2 owner=${3:-cop} f c v line rc=0 nm=0 n=0
  local -A prev streak
  touch "$esc"
  while IFS= read -r f; do
    n=$((n + 1)); c=$(chain "$f"); v=$(verdict "$f")
    if [ "$v" = NM ]; then
      nm=1
      streak[$c]=0
      continue
    fi
    if [ "$v" = PASS ]; then
      streak[$c]=0
      continue
    fi
    streak[$c]=$(( ${streak[$c]:-0} + 1 ))
    if [ "${streak[$c]}" -eq 2 ]; then
      line="ESCALATE ticket=$c receipts=${prev[$c]},$(basename "$f") owner=$owner"
      if ! grep -qxF "$line" "$esc"; then echo "$line" >>"$esc"; rc=1; fi  # ESCALATE
    fi
    prev[$c]=$(basename "$f")
  done < <(ls -tr "$dir"/triage-quorum-*-gate-*.md 2>/dev/null)
  [ "$n" -eq 0 ] && return 2
  [ "$rc" -eq 0 ] && [ "$nm" -eq 1 ] && return 2
  return "$rc"
}

self_test() { # $1 = script under test
  local t d bad=0 s=$1
  t=$(mktemp -d "${TMPDIR:-/tmp}/qe.XXXXXX") || return 2
  mk() { printf '# Quorum receipt - aprender #%s (%s)\n\n**TALLY: %s**\n' "$2" "$3" "$4" >"$t/triage-quorum-$2-gate-$1.md"; touch -d "2026-10-23 0$1:00" "$t/triage-quorum-$2-gate-$1.md"; }
  clr() { rm -f "${t:?}"/triage-quorum-* "${t:?}/esc"; }
  chk() { local want=$1 got; shift; bash "$s" "$t" "$t/esc" x >/dev/null 2>&1; got=$?; [ "$got" = "$want" ] || { echo "FAIL: $*: rc $got want $want"; bad=1; }; }
  mk 1 100 PMAT-1 "2/3 PASS"; chk 0 "one FAIL";                      clr
  mk 2 101 PMAT-1 "3/3 PASS"; chk 0 "FAIL then PASS";                clr
  clr; mk 3 102 PMAT-1 "2/3 PASS"; mk 4 103 PMAT-1 "1/3 PASS"; chk 1 "two in a row"
  grep -q 'ticket=PMAT-1 receipts=triage-quorum-102-gate-3.md,triage-quorum-103-gate-4.md owner=x' "$t/esc" || { echo "FAIL: line wrong"; bad=1; }
  chk 0 "rerun is idempotent"; clr
  mk 5 200 PMAT-2 "2/3 PASS"; mk 6 201 PMAT-3 "2/3 PASS"; mk 7 100 PMAT-1 "3/3 PASS"; mk 8 103 PMAT-1 "3/3 PASS"
  chk 0 "two chains, one FAIL each"; clr
  clr; mk 9 300 PMAT-4 "unreadable"; chk 2 "unreadable tally is not measured"
  clr; mk 1 400 PMAT-5 "2/3 PASS"; mk 2 401 PMAT-5 "3/3 PASS"; mk 3 402 PMAT-5 "2/3 PASS"; chk 0 "PASS resets the streak"
  clr; mk 1 410 PMAT-6 "2/3 PASS"; mk 2 411 PMAT-6 "unreadable"; mk 3 412 PMAT-6 "2/3 PASS"; chk 2 "NM resets the streak"
  clr; mk 1 420 PMAT-7 "3/3 FAIL"; mk 2 421 PMAT-7 "3/3 FAIL"; chk 1 "N=M beside FAIL is a FAIL"
  clr; mk 1 430 PMAT-12 "1/3 PASS"; mk 2 431 PMAT-12 "1/3 PASS"; chk 1 "two-digit ticket"
  grep -q 'ticket=PMAT-12 ' "$t/esc" || { echo "FAIL: PMAT-12 truncated"; bad=1; }
  rm -f "${t:?}/esc"; bash "$s" "$t" "$t/esc" >/dev/null 2>&1; grep -q 'owner=cop$' "$t/esc" || { echo "FAIL: default owner"; bad=1; }
  clr; mk 9 300 PMAT-4 "unreadable"
  d=$(mktemp -d "${TMPDIR:-/tmp}/qe0.XXXXXX"); bash "$s" "$d" "$d/esc" >/dev/null 2>&1; [ "$?" = 2 ] || { echo "FAIL: empty dir"; bad=1; }
  rm -rf "${d:?}" "${t:?}"
  return "$bad"
}

if [ "${1:-}" = "--self-test" ]; then
  self_test "$0" || exit 1
  m=$(mktemp "${TMPDIR:-/tmp}/qe-mut.XXXXXX") || exit 2
  grep -v '# ESCALATE$' "$0" >"$m"   # mutation: remove the escalation write
  if self_test "$m" >/dev/null 2>&1; then echo "FAIL: case table stayed green with the escalation removed"; rm -f "${m:?}"; exit 1; fi
  rm -f "${m:?}"; echo "quorum_escalation self-test: OK (mutation turns it red)"; exit 0
fi
[ "$#" -ge 2 ] || { echo "usage: $0 <receipts-dir> <escalations-file> [owner] | --self-test" >&2; exit 2; }
run "$@"
