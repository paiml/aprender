#!/usr/bin/env bash
# stopline_report.sh -- hourly STOPLINE line (PMAT-5836, #5836). Operator 08:09Z: "The STOPLINE
# report runs late by hand -> make it a scheduled job, not a cop chore."
#
# One line per run: merges/h, gate p95, open and non-draft PRs vs the WIP cap, and each fix's
# metric with its predicted and measured value. Timer: scripts/stopline/stopline.timer (*:05).
#
#   stopline_report.sh            live: print the line and append it to the cop inbox
#   stopline_report.sh --print    live reads, print only, write nothing
#   stopline_report.sh --cases    decision case table (no network)
#
# Reads: GitHub GraphQL search only (issueCount, 3 points per run; GH-2), the gate p95 script
# (default: aprender-flow-p95.sh, REST, skipped below 1000 remaining), and the fixes ledger
# (scripts/stopline/fixes.tsv). A read that fails prints not_measured; it is never guessed (L23).
set -u
PATH=$PATH:/usr/local/bin:/usr/bin
REPO=${SL_REPO:-paiml/aprender}
CAP=${SL_CAP:-7}
INBOX=${SL_INBOX:-/mnt/nvme-raid0/cop-inbox/inbox.md}
HERE=$(cd "$(dirname "$0")" && pwd)
FIXES=${SL_FIXES:-$HERE/stopline/fixes.tsv}
FLOW010=${SL_FLOW010:-$HERE/stopline/flow010.tsv}
P95=${SL_P95:-/mnt/nvme-raid0/cop-inbox/handoff/aprender-flow-p95.sh}

# count_q <search-qualifiers> -> issueCount, or "nm" when the read fails
count_q() {
  gh api graphql -f q="repo:$REPO is:pr $1" \
    -f query='query($q:String!){search(query:$q,type:ISSUE,first:1){issueCount}}' \
    --jq .data.search.issueCount 2>/dev/null || echo nm
}

# verdict <predicted> <measured> <dir>   dir: le (measured must be <= predicted) | ge | none
# -> MEET | MISS | MEASURING (no number yet) | -
verdict() {
  case "$2" in nm | "" | not_measured*) echo MEASURING; return ;; esac
  case "$3" in
    le) awk -v p="$1" -v m="$2" 'BEGIN{print (m <= p) ? "MEET" : "MISS"}' ;;
    ge) awk -v p="$1" -v m="$2" 'BEGIN{print (m >= p) ? "MEET" : "MISS"}' ;;
    *) echo - ;;
  esac
}

# fixes_part <file> -> "name metric pred X, meas Y VERDICT | ..." (no network when no cmd is set)
# columns: name<TAB>metric<TAB>predicted<TAB>dir<TAB>measure-cmd (stdout = one number)<TAB>note (optional)
fixes_part() {
  local out="" name metric pred dir cmd note meas v
  [ -f "$1" ] || { echo "no fixes ledger"; return; }
  while IFS=$'\t' read -r name metric pred dir cmd note; do
    case "$name" in "" | \#*) continue ;; esac
    meas=nm
    [ -n "${cmd:-}" ] && [ "$cmd" != - ] && meas=$(timeout 60 bash -c "$cmd" 2>/dev/null | head -n1)
    [ -n "$meas" ] || meas=nm
    v=$(verdict "$pred" "$meas" "$dir")
    out="${out:+$out | }$name: $metric pred $pred, meas ${meas/nm/not_measured} $v${note:+ [$note]}"
  done <"$1"
  echo "${out:-no fixes}"
}

# cpu_part -> latest "cpu-min/batch ..." from flow's :02 gate line in the inbox, or not_measured
cpu_part() {
  local l
  l=$(grep -h "gate | aprender-flow" "$INBOX" 2>/dev/null | grep -o "cpu-min/batch [^;]*" | tail -n1)
  echo "${l:-cpu-min/batch not_measured}"
}

# flow010_part <file> -> one row per phase: owner, due, state, exit test
flow010_part() {
  local p o d s e
  [ -f "$1" ] || { echo "FLOW-010 (design): no phase ledger"; return; }
  while IFS=$'\t' read -r p o d s e; do
    case "$p" in "" | \#*) continue ;; esac
    printf 'FLOW-010 (design) %s | owner %s | due %s | state %s | exit %s\n' "$p" "$o" "$d" "$s" "$e"
  done <"$1"
}

# cpu_part -> latest "cpu-min/batch ..." from flow's :02 gate line in the inbox, or not_measured
cpu_part() {
  local l
  l=$(grep -h "gate | aprender-flow" "$INBOX" 2>/dev/null | grep -o 'cpu-min/batch [^;]*' | tail -n1)
  echo "${l:-cpu-min/batch not_measured}"
}

# flow010_part <file> -> one row per phase: owner, due, state, exit test
flow010_part() {
  local p o d s e
  [ -f "$1" ] || { echo "FLOW-010 (design): no phase ledger"; return; }
  while IFS=$'\t' read -r p o d s e; do
    case "$p" in "" | \#*) continue ;; esac
    printf 'FLOW-010 (design) %s | owner %s | due %s | state %s | exit %s\n' "$p" "$o" "$d" "$s" "$e"
  done <"$1"
}

# line <merged1h> <merged2h> <p95> <open> <nondraft> <fixes>
line() {
  local m1=$1 m2=$2 p95=$3 op=$4 nd=$5 fx=$6 rate cap
  if [ "$m2" = nm ]; then rate=nm; else rate=$(awk -v n="$m2" 'BEGIN{printf "%.1f", n/2}'); fi
  cap=nm
  if [ "$nd" != nm ]; then if [ "$nd" -le "$CAP" ]; then cap=ok; else cap=OVER; fi; fi
  printf '%sZ | aprender-stopline | STOPLINE | merges/h 1h=%s 2h-avg=%s | gate p95 %s | open %s, non-draft %s vs %s (%s) | fixes: %s' \
    "$(date -u +%H:%M)" "${m1/nm/not_measured}" "${rate/nm/not_measured}" "${p95:-not_measured}" \
    "${op/nm/not_measured}" "${nd/nm/not_measured}" "$CAP" "${cap/nm/not_measured}" "$fx"
}

cases() {
  local bad=0 t
  chk() { [ "$2" = "$3" ] || { echo "FAIL $1: got '$2' want '$3'"; bad=1; }; }
  chk V1 "$(verdict 20 12 ge)" MISS
  chk V2 "$(verdict 20 20 ge)" MEET
  chk V3 "$(verdict 15 22 le)" MISS
  chk V4 "$(verdict 15 15 le)" MEET
  chk V5 "$(verdict 15 nm le)" MEASURING
  chk V6 "$(verdict 15 not_measured le)" MEASURING
  chk V7 "$(verdict 15 3 none)" -
  t=$(mktemp)
  printf '# c\na\tmerges/h\t2\tge\techo 1\nb\tp95\t15\tle\techo 22\nc\tx\t1\tle\t-\tn\n' >"$t"
  chk F1 "$(fixes_part "$t")" "a: merges/h pred 2, meas 1 MISS | b: p95 pred 15, meas 22 MISS | c: x pred 1, meas not_measured MEASURING [n]"
  chk F2 "$(fixes_part /nonexistent)" "no fixes ledger"
  printf '# only\n' >"$t"; chk F3 "$(fixes_part "$t")" "no fixes"
  printf '# h\nP9\to\td\ts\te\n' >"$t"
  chk P1 "$(flow010_part "$t")" "FLOW-010 (design) P9 | owner o | due d | state s | exit e"
  chk P2 "$(flow010_part /nonexistent)" "FLOW-010 (design): no phase ledger"
  printf 'x gate | aprender-flow | wall p95; cpu-min/batch mean=178 med=180 n=13 window=24h; reuse 0%%\n' >"$t"
  chk C1 "$(INBOX=$t cpu_part)" "cpu-min/batch mean=178 med=180 n=13 window=24h"
  chk C2 "$(INBOX=/nonexistent cpu_part)" "cpu-min/batch not_measured"
  rm -f "${t:?}"
  chk L1 "$(line 1 2 'p95=98min' 45 8 x | cut -d'|' -f4-)" " merges/h 1h=1 2h-avg=1.0 | gate p95 p95=98min | open 45, non-draft 8 vs 7 (OVER) | fixes: x"
  chk L2 "$(line nm nm '' nm nm x | cut -d'|' -f5,6)" " gate p95 not_measured | open not_measured, non-draft not_measured vs 7 (not_measured) "
  chk L3 "$(line 3 4 p 9 7 x | grep -o '(ok)')" "(ok)"
  [ "$bad" = 0 ] && echo "stopline cases: all pass"
  return "$bad"
}

case "${1:-}" in
  --cases) cases; exit $? ;;
  "" | --print) ;;
  *) echo "usage: $0 [--print|--cases]" >&2; exit 2 ;;
esac

now=$(date -u +%s)
s1=$(date -u -d "@$((now - 3600))" +%Y-%m-%dT%H:%M:%SZ)
s2=$(date -u -d "@$((now - 7200))" +%Y-%m-%dT%H:%M:%SZ)
m1=$(count_q "is:merged base:main merged:>=$s1")
m2=$(count_q "is:merged base:main merged:>=$s2")
op=$(count_q "is:open")
nd=$(count_q "is:open draft:false")
p95=""
if [ -x "$P95" ] || [ -f "$P95" ]; then
  p95=$(bash "$P95" 24 2>/dev/null | head -n1)
fi
out=$(line "$m1" "$m2" "$p95" "$op" "$nd" "$(fixes_part "$FIXES")")
out="$out | $(cpu_part)
$(flow010_part "$FLOW010" | sed "s/^/$(date -u +%H:%M)Z | aprender-stopline | /")"
printf '%s\n' "$out"
[ "${1:-}" = "--print" ] || printf '%s\n' "$out" | cut -c1-600 >>"$INBOX"
