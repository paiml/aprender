#!/usr/bin/env bash
# quorum_escalation.sh - #3562 "two consecutive non-passing plan quorums must escalate
# automatically", built (PMAT-6101). A LAB check: it runs at night, it cannot block a merge.
#
#   bash scripts/quorum_escalation.sh <receipts-dir> <escalations-file> [owner]
#   bash scripts/quorum_escalation.sh --self-test
#
# Reads gate-quorum receipts (`triage-quorum-<PR>-gate-<hhmm>Z.md`, never the -A/-B/-C/-prompt lane
# files) oldest first by mtime. CHAIN = the ticket the receipt is about: its `PMAT-Ticket:` trailer when
# that names one ticket, else the first PMAT-<n> / #<n> / "pace <n>" on line 1 (not the receipt's own
# PR); no id = not measured. A triage-quorum-* file whose name is not a receipt or lane name = not measured. VERDICT = the first PASS / NOT PASS / FAIL token of the receipt's last headline
# (TALLY / RESULT / OVERALL / Final, else VERDICT, else a "verdict X" on lines 1-3), after the last
# arrow; a token beside an N/M tally passes only if N = M. No token = NOT MEASURED: it neither
# resets a chain's streak nor counts toward it. On the SECOND consecutive non-PASS receipt of a
# chain, one line
#   ESCALATE ticket=<chain> receipts=<first>,<second> owner=<owner>
# is appended to <escalations-file> (once: a rerun does not repeat it).
# EXIT: 0 nothing new to escalate; 1 an escalation was written; 2 not measured (empty dir, or an
# unreadable receipt and nothing escalated); 3 the escalations file could not be written.
set -uo pipefail

verdict() { # receipt file -> PASS | FAIL | NM
  local l b tok rest pre u
  l=$(grep -i -E '^[^A-Za-z]*(tally|result|overall|final)\b.*(PASS|FAIL)' "$1" | tail -n1)
  [ -n "$l" ] || l=$(grep -i -E '^[^A-Za-z]*verdict\b.*(PASS|FAIL)' "$1" | tail -n1)
  [ -n "$l" ] || l=$(head -n3 "$1" | grep -i -m1 -oE 'verdict:? *(NOT PASS|PASS|FAIL).*')
  [ -n "$l" ] || { echo NM; return; }
  b=${l#*:}
  [[ ${b##*→} =~ (PASS|FAIL) ]] && b=${b##*→}   # a verdict after the last arrow wins
  tok=$(grep -oE 'NOT PASS|PASS|FAIL' <<<"$b" | head -n1)
  [ -n "$tok" ] || { echo NM; return; }
  [ "$tok" = PASS ] || { echo FAIL; return; }
  rest=${b#*"$tok"}; pre=${b%%"$tok"*}
  u="${pre^^} ${rest^^}"   # 'fail=1' counts as 'FAIL=1'
  [[ "$u" =~ (^|[^0-9])[1-9][0-9]*\ *(LANES?\ +)?FAIL ]] && { echo FAIL; return; }   # a counted FAIL beside PASS
  [[ "$u" =~ FAIL[A-Za-z]*[:\ =]+[1-9] ]] && { echo FAIL; return; }   # 'FAIL: 1', 'FAIL 1'
  if [[ $rest =~ ^[^A-Za-z0-9/]{0,3}([0-9]+)(/|\ +of\ +)([0-9]+) ]] || [[ $pre =~ ([0-9]+)(/|\ +of\ +)([0-9]+)[^A-Za-z0-9/]*$ ]]; then
    [ "${BASH_REMATCH[1]}" = "${BASH_REMATCH[3]}" ] && echo PASS || echo FAIL
    return
  fi
  echo PASS
}

chain() { # receipt file -> ticket id
  local b own="" n
  b=$(basename "$1")
  [[ $b =~ ^triage-quorum-([0-9]+)- ]] && own=${BASH_REMATCH[1]}
  n=$(grep -oE 'PMAT-Ticket: PMAT-[0-9]+' "$1" | grep -oE '[0-9]+$' | sort -u)   # the trailer wins when it names ONE ticket
  case $n in *[!0-9]*) n="" ;; esac
  [ -n "$n" ] || n=$(head -n1 "$1" | grep -oE 'PMAT-[0-9]+|#[0-9]{4,}|pace [0-9]{4,}' | grep -oE '[0-9]+' | grep -vx "$own" | head -n1)
  [ -z "$n" ] || echo "PMAT-$n"   # never the most-mentioned id in the body: that keyed one ticket two ways. No id = NM.
}

run() { # dir esc owner
  local dir=$1 esc=$2 owner=${3:-cop} f c v line rc=0 nm=0 n=0
  local -A prev streak
  touch "$esc" 2>/dev/null || { echo "cannot write $esc" >&2; return 3; }
  while IFS= read -r f; do
    n=$((n + 1)); c=$(chain "$f"); v=$(verdict "$f")
    [[ $f =~ /triage-quorum-[0-9]+-gate-[A-Za-z0-9-]*Z\.md$ ]] || v=NM   # a renamed receipt is not measured, never dropped
    [ -n "$c" ] || v=NM
    if [ "$v" = NM ]; then nm=1; continue; fi   # not measured: neither resets nor counts
    if [ "$v" = PASS ]; then
      streak[$c]=0
      continue
    fi
    streak[$c]=$(( ${streak[$c]:-0} + 1 ))
    if [ "${streak[$c]}" -eq 2 ]; then
      line="ESCALATE ticket=$c receipts=${prev[$c]},$(basename "$f") owner=$owner"
      if ! grep -qxF "$line" "$esc"; then
        echo "$line" >>"$esc" || { echo "cannot write $esc" >&2; return 3; }  # ESCALATE
        rc=1
      fi
    fi
    prev[$c]=$(basename "$f")
  done < <(ls -tr "$dir"/triage-quorum-* 2>/dev/null | grep -vE '/triage-quorum-[0-9]+-gate-[A-Za-z0-9-]*Z-(A|B|C|prompt)\.[^/]*$')
  [ "$n" -eq 0 ] && return 2
  [ "$rc" -eq 0 ] && [ "$nm" -eq 1 ] && return 2
  return "$rc"
}

self_test() { # $1 = script under test
  local t d bad=0 s=$1
  t=$(mktemp -d "${TMPDIR:-/tmp}/qe.XXXXXX") || return 2
  # mk <pr> <hhmm> <line1> <headline>: a receipt shaped like the real handoff ones
  mk() { local f="$t/triage-quorum-$1-gate-$2Z.md"; printf '%s\n\nLanes: A opus, B haiku, C argon.\n%s\n' "$3" "$4" >"$f"; touch -d "2026-10-23 $2" "$f"; }
  clr() { rm -f "${t:?}"/triage-quorum-* "${t:?}/esc"; }
  chk() { local want=$1 got; shift; bash "$s" "$t" "$t/esc" x >/dev/null 2>&1; got=$?; [ "$got" = "$want" ] || { echo "FAIL: $*: rc $got want $want"; bad=1; }; }
  local F="**VERDICT: FAIL (1/3).** Not armed." P="RESULT: PASS 3/3 (all lanes valid)."
  mk 100 0100 "# Quorum #100 (PMAT-1)" "$F"; chk 0 "one FAIL"; clr
  mk 100 0100 "# Quorum #100 (PMAT-1)" "$F"; mk 101 0200 "# Quorum #101 (PMAT-1)" "$P"; chk 0 "FAIL then PASS"; clr
  mk 102 0300 "# Quorum #102 (PMAT-1)" "$F"; mk 103 0400 "# Quorum #103 (PMAT-1)" "$F"; chk 1 "two in a row"
  grep -q 'ticket=PMAT-1 receipts=triage-quorum-102-gate-0300Z.md,triage-quorum-103-gate-0400Z.md owner=x' "$t/esc" || { echo "FAIL: line wrong"; bad=1; }
  chk 0 "rerun is idempotent"; clr
  mk 100 0100 "# Quorum #100 (PMAT-1)" "$F"; mk 101 0200 "# Quorum #101 (PMAT-1)" "$P"; mk 102 0300 "# Quorum #102 (PMAT-1)" "$F"; chk 0 "PASS resets the streak"; clr
  mk 200 0100 "# Quorum #200 (PMAT-2)" "$F"; mk 201 0200 "# Quorum #201 (PMAT-3)" "$F"; chk 0 "two chains, one FAIL each"; clr
  mk 9 0100 "# Quorum #9 (PMAT-4)" "no verdict here"; chk 2 "unreadable tally is not measured"; clr
  # f1: chain from receipt content as the real PMAT-5920 receipts carry it (pace on line 1, PMAT in body)
  mk 6085 0100 "# Quorum #6085 @cdb (pace 5920 v13, gate-class), 05:45Z" "$F"
  mk 6092 0200 "# Quorum receipt — paiml/aprender PR #6092 (C324)" "- trailer \`PMAT-Ticket: PMAT-5920\`
## OVERALL: FAIL (0/3 PASS)"
  chk 1 "f1 chain keyed by ticket, not by line 1 or PR"
  grep -q 'ticket=PMAT-5920 ' "$t/esc" || { echo "FAIL: f1 ticket not PMAT-5920"; bad=1; }; clr
  # f2: the verdict is the token, never a FAIL substring
  mk 1 0100 "# Quorum #1 (PMAT-8)" "$F"; mk 2 0200 "# Quorum #2 (PMAT-8)" "RESULT: PASS 3/3. Adds hard FAIL(ENV) exit 1."; mk 3 0300 "# Quorum #3 (PMAT-8)" "$F"; chk 0 "f2 PASS beside FAIL(ENV) is a PASS"; clr
  mk 1 0100 "# Quorum #1 (PMAT-8)" "$F"; mk 2 0200 "# Quorum #2 (PMAT-8)" "RESULT: NOT PASS 2/3.
**RESULT: PASS (3/3 valid lanes PASS).** This supersedes the NOT PASS above."; mk 3 0300 "# Quorum #3 (PMAT-8)" "$F"; chk 0 "f2 superseding PASS wins"; clr
  mk 1 0100 "# Quorum #1 (PMAT-8)" "$F"; mk 2 0200 "# Quorum #2 (PMAT-8)" "RESULT: NOT PASS 2/3. Fix it → re-seat both + new quorum."; chk 1 "f2 trailing arrow keeps the headline verdict"; clr
  mk 1 0100 "# Quorum #1 (PMAT-8)" "$F"; mk 2 0200 "# Quorum #2 (PMAT-8)" "RESULT: NOT PASS 2/3 → PASS 3/3 after the re-seat."; mk 3 0300 "# Quorum #3 (PMAT-8)" "$F"; chk 0 "f2 a verdict after the arrow wins"; clr
  mk 1 0100 "# Quorum #1 (PMAT-8)" "$F"; mk 2 0200 "# Quorum #2 (PMAT-8)" "RESULT: PASS 2/3 (one lane errored)."; chk 1 "f2 PASS beside a 2/3 tally is not a pass"; clr
  mk 1 0100 "# Quorum #1 (PMAT-8)" "$F"; mk 2 0200 "# Quorum #2 (PMAT-8)" "RESULT: NOT PASS (lane B errored)."; chk 1 "f2 NOT PASS without a tally is not a pass"; clr
  mk 1 0100 "# Quorum #1 (PMAT-8)" "$F"; mk 2 0200 "# Quorum #2 (PMAT-8)" "RESULT: PASS (3 of 4 lanes)."; chk 1 "f10 PASS beside '3 of 4' is not a pass"; clr
  mk 1 0100 "# Quorum #1 (PMAT-8)" "$F"; mk 2 0200 "# Quorum #2 (PMAT-8)" "RESULT: PASS (2 lanes PASS, 1 FAIL)."; chk 1 "f10 PASS beside a counted FAIL is not a pass"; clr
  mk 1 0100 "# Quorum #1 (PMAT-8)" "$F"; mk 2 0200 "# Quorum #2 (PMAT-8)" "TALLY: 9/10 PASS"; chk 1 "f10 a 9/10 tally before PASS is not a pass"; clr
  mk 1 0100 "# Quorum #1 (PMAT-8)" "$F"; mk 2 0200 "# Quorum #2 (PMAT-8)" "RESULT: PASS (4 of 4 lanes)."; mk 3 0300 "# Quorum #3 (PMAT-8)" "$F"; chk 0 "f10 '4 of 4' is a PASS"; clr
  mk 1 0100 "# Quorum #1 (PMAT-8)" "$F"; mk 2 0200 "# Quorum #2 (PMAT-8)" "RESULT: PASS (lanes: 3/3 PASS; FAIL: 1)"; chk 1 "f11 PASS beside 'FAIL: 1' is not a pass"; clr
  mk 1 0100 "# Quorum #1 (PMAT-8)" "$F"; mk 2 0200 "# Quorum #2 (PMAT-8)" "RESULT: PASS (FAIL 1)"; chk 1 "f11 PASS beside 'FAIL 1' is not a pass"; clr
  mk 1 0100 "# Quorum #1 (PMAT-8)" "$F"; mk 2 0200 "# Quorum #2 (PMAT-8)" "RESULT: PASS (3/3 lanes, 0 FAIL)"; mk 3 0300 "# Quorum #3 (PMAT-8)" "$F"; chk 0 "f11 '0 FAIL' beside PASS is a pass"; clr
  mk 1 0100 "# Quorum #1 (PMAT-8)" "$F"; mk 2 0200 "# Quorum #2 (PMAT-8)" "RESULT: PASS (FAIL: 0)"; mk 3 0300 "# Quorum #3 (PMAT-8)" "$F"; chk 0 "f11 'FAIL: 0' beside PASS is a pass"; clr
  mk 1 0100 "# Quorum #1 (PMAT-8)" "$F"; mk 2 0200 "# Quorum #2 (PMAT-8)" "RESULT: PASS (fail=1)"; chk 1 "f12 PASS beside lower-case 'fail=1' is not a pass"; clr
  mk 7078 0100 "# Quorum #7078" "$F"; mk 7078 0200 "# Quorum #7078" "$F"; chk 2 "f12 a receipt naming only its own 4-digit PR has no chain: not measured"; clr
  mk 1 0100 "# Quorum #1 (PMAT-8)" "$F"; mk 2 0200 "# Quorum #2 (PMAT-8)" "RESULT: PASS (1 lane FAIL)"; chk 1 "f13 PASS beside lower-case '1 lane FAIL' is not a pass"; clr
  mk 1 0100 "# Quorum #1 (PMAT-21)" "PMAT-Ticket: PMAT-30
PMAT-Ticket: PMAT-31
$F"; mk 2 0200 "# Quorum #2 (PMAT-21)" "$F"; chk 1 "f13 two different trailers fall back to line 1"; grep -q 'ticket=PMAT-21 ' "$t/esc" || { echo "FAIL: f13 ticket not PMAT-21"; bad=1; }; clr
  # f3: an unreadable receipt neither resets nor counts
  mk 1 0100 "# Quorum #1 (PMAT-9)" "$F"; mk 2 0200 "# Quorum #2 (PMAT-9)" "no verdict here"; mk 3 0300 "# Quorum #3 (PMAT-9)" "$F"; chk 1 "f3 NM keeps the streak"; clr
  # f4: lane and prompt files are not receipts
  mk 1 0100 "# Quorum #1 (PMAT-10)" "$F"
  for x in A.md B.txt prompt.md prompt.Z.md; do cp "$t/triage-quorum-1-gate-0100Z.md" "$t/triage-quorum-1-gate-0100Z-$x"; touch -d "2026-10-23 0200" "$t/triage-quorum-1-gate-0100Z-$x"; done
  chk 0 "f4 lane/prompt files never count"; clr
  # f7 renamed receipt = NM; f8 no chain id = NM; f9 trailer and line 1 key ONE chain, body mentions do not
  mk 1 0100 "# Quorum #1 (PMAT-13)" "$F"; mk 2 0200 "# Quorum #2 (PMAT-13)" "$F"; mv "$t/triage-quorum-2-gate-0200Z.md" "$t/triage-quorum-2-gate-0200Z-final.md"; mk 3 0300 "# Quorum #3 (PMAT-13)" "$P"; chk 2 "f7 renamed FAIL is NM, not a silent pass"; clr
  mk 1 0100 "# Gate quorum: PR #1 at abc" "$F"; mk 2 0200 "# Gate quorum: PR #2 at def" "$F"; chk 2 "f8 no chain id is NM"; clr
  mk 1 0100 "# Quorum #1 (PMAT-14)" "$F"; mk 2 0200 "# Quorum receipt PR #2 (C324)" "- trailer PMAT-Ticket: PMAT-14 (see PMAT-99, PMAT-99)
$F"; chk 1 "f9 one ticket keyed by line 1 and by trailer is one chain"; grep -q 'ticket=PMAT-14 ' "$t/esc" || { echo "FAIL: f9 ticket not PMAT-14"; bad=1; }; clr
  # f5: multi-digit tally
  mk 1 0100 "# Quorum #1 (PMAT-11)" "$F"; mk 2 0200 "# Quorum #2 (PMAT-11)" "TALLY: 10/10 PASS"; mk 3 0300 "# Quorum #3 (PMAT-11)" "$F"; chk 0 "f5 10/10 is a PASS"; clr
  # f6: a failed write is not a clean exit
  mk 1 0100 "# Quorum #1 (PMAT-12)" "$F"; mk 2 0200 "# Quorum #2 (PMAT-12)" "$F"
  bash "$s" "$t" "$t/nodir/esc" >/dev/null 2>&1; [ "$?" = 3 ] || { echo "FAIL: f6 unwritable escalations file"; bad=1; }
  rm -f "${t:?}/esc"; bash "$s" "$t" "$t/esc" >/dev/null 2>&1; grep -q 'ticket=PMAT-12 .*owner=cop$' "$t/esc" || { echo "FAIL: default owner / PMAT-12"; bad=1; }
  clr; d=$(mktemp -d "${TMPDIR:-/tmp}/qe0.XXXXXX"); bash "$s" "$d" "$d/esc" >/dev/null 2>&1; [ "$?" = 2 ] || { echo "FAIL: empty dir"; bad=1; }
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
exit $?
