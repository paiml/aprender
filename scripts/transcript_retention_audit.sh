#!/usr/bin/env bash
# PRA-001 §1 G14 / T0: read-only transcript-retention audit, one row per fleet host.
#
#   make transcript-retention-audit                 # the five fleet hosts
#   HOSTS="gx10 mini" make transcript-retention-audit
#
# Row: host  cleanupPeriodDays  files_older_30d  files_total  oldest_mtime_utc
#   cleanupPeriodDays: the value in ~/.claude/settings.json, `unset` (Claude Code's default of 30 applies) or
#   `no-settings`. Files are regular files under ~/.claude/projects. A host with no ~/.claude/projects reports 0 0 -.
#
# Nothing is written on any host. Fails closed: a host that cannot be reached, or whose row does not parse,
# prints `UNREACHABLE` and makes the exit status 1, so a missing row can never read as a compliant one.
# Portable to macOS (mini): no `find -printf`, no jq, `stat` probed for GNU or BSD form.
set -euo pipefail

HOSTS="${HOSTS:-lambda-labs intel gx10 mini yoga}"
SELF="$(uname -n)"

# Runs on the audited host under sh. Prints exactly one line: cpd older total oldest_epoch
PROBE='
p="$HOME/.claude/projects"; s="$HOME/.claude/settings.json"
if [ -f "$s" ]; then
  cpd=$(tr -d "\n " < "$s" | sed -n "s/.*\"cleanupPeriodDays\":\([0-9][0-9]*\).*/\1/p")
  [ -n "$cpd" ] || cpd=unset
else
  cpd=no-settings
fi
if [ -d "$p" ]; then
  older=$(find "$p" -type f -mtime +30 | wc -l | tr -d " ")
  total=$(find "$p" -type f | wc -l | tr -d " ")
  if stat -c %Y / >/dev/null 2>&1; then fmt="-c %Y"; else fmt="-f %m"; fi
  oldest=$(find "$p" -type f -exec stat $fmt {} + 2>/dev/null | sort -n | head -1)
  [ -n "$oldest" ] || oldest=-
else
  older=0; total=0; oldest=-
fi
echo "ROW $cpd $older $total $oldest"
'

to_utc() {
  if [ "$1" = "-" ]; then
    printf '%s' "-"
  else
    date -u -d "@$1" +%Y-%m-%dT%H:%MZ
  fi
}

rc=0
printf '%s\t%s\t%s\t%s\t%s\n' host cleanupPeriodDays files_older_30d files_total oldest_mtime_utc
read -r -a hosts <<<"$HOSTS"
for h in "${hosts[@]}"; do
  if [ "$h" = "$SELF" ]; then
    out=$(sh -s <<<"$PROBE" 2>/dev/null) || out=""
  else
    out=$(timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" sh -s <<<"$PROBE" 2>/dev/null) || out=""
  fi
  row=$(printf '%s\n' "$out" | sed -n 's/^ROW //p' | tail -1)
  if [ -z "$row" ]; then
    printf '%s\tUNREACHABLE\t-\t-\t-\n' "$h"
    rc=1
    continue
  fi
  read -r cpd older total oldest <<<"$row"
  printf '%s\t%s\t%s\t%s\t%s\n' "$h" "$cpd" "$older" "$total" "$(to_utc "$oldest")"
done
exit "$rc"
