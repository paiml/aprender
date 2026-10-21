#!/usr/bin/env bash
# check_lab_timeouts.sh -- PMAT-5847. LAB/nightly (schedule:) workflows are bounded: every job has timeout-minutes,
# and the cron start plus the longest job timeout ends by 05:00Z (a start before 05:00Z ends by 05:00Z that day,
# a later start by 05:00Z the next day). Operator, verbatim, 11:04Z: "Lab/nightly workflows get timeout-minutes
# and must end by 05:00Z". A LAB check: run on demand or at night, never inside a PR or release job (C324).
# A job that only calls a reusable workflow (uses:) cannot carry timeout-minutes; the callee is scanned itself.
# Usage: check_lab_timeouts.sh [workflows-dir] | --self-test       Exit: 0 green, 1 RED
set -euo pipefail
LIMIT=300 DAY=1440
scan() { # one workflow file -> RED lines
  local f="$1" j longest m h s end
  grep -qE '^[[:space:]]+schedule:' "$f" || return 0
  for j in $(awk '/^jobs:/{in_jobs=1;next} in_jobs&&/^  [A-Za-z0-9_-]+:[[:space:]]*(#.*)?$/{if(name&&runs&&!seen)print name; name=$1; runs=0; seen=0; next} in_jobs&&/^    runs-on:/{runs=1} in_jobs&&/^    timeout-minutes:/{seen=1} END{if(name&&runs&&!seen)print name}' "$f"); do
    echo "RED $f job ${j%:}: no timeout-minutes"
  done
  longest=$(awk '/^jobs:/{in_jobs=1} in_jobs&&/^    timeout-minutes:/{v=$2+0; if(v>m)m=v} END{print m+0}' "$f")
  grep -E '^[[:space:]]+- cron:' "$f" | sed -E "s/.*cron:[[:space:]]*['\"]?([^'\"#]*).*/\1/" | while read -r m h _; do
    case "$m$h" in *[!0-9]*) continue ;; esac   # */10 or a range is not one nightly start
    s=$((h * 60 + m)); end=$((s + longest))
    if [ "$s" -lt "$LIMIT" ]; then [ "$end" -le "$LIMIT" ] || echo "RED $f: cron $m $h + ${longest}m ends after 05:00Z"
    else [ "$end" -le $((DAY + LIMIT)) ] || echo "RED $f: cron $m $h + ${longest}m ends after 05:00Z"; fi
  done
}
check() { local bad=0 out f; for f in "$1"/*.yml; do out=$(scan "$f"); [ -z "$out" ] || { echo "$out"; bad=1; }; done; return "$bad"; }
if [ "${1:-}" = --self-test ]; then
  t=$(mktemp -d); trap 'rm -rf -- "${t:?}"' EXIT; fail=0
  w() { rm -f "${t:?}"/*.yml; printf 'on:\n  schedule:\n    - cron: "%s"\njobs:\n  a:\n    runs-on: x\n%s    steps: []\n' "$1" "$2" > "$t/w.yml"; }
  w '5 23 * * *' $'    timeout-minutes: 60\n';  check "$t" > /dev/null || { echo "FAIL bounded row is RED"; fail=1; }
  w '5 23 * * *' '';                            check "$t" > /dev/null && { echo "FAIL no-timeout row not RED"; fail=1; }
  w '5 23 * * *' $'    timeout-minutes: 420\n'; check "$t" > /dev/null && { echo "FAIL ends-after-05Z row not RED"; fail=1; }
  w '5 23 * * *' $'    timeout-minutes: 355\n'; check "$t" > /dev/null || { echo "FAIL ends-exactly-05:00 row is RED"; fail=1; }
  w '20 3 * * *' $'    timeout-minutes: 120\n'; check "$t" > /dev/null && { echo "FAIL 03:20+120m row not RED"; fail=1; }
  rm -f "${t:?}"/*.yml; printf 'on:\n  pull_request:\njobs:\n  a:\n    runs-on: x\n    steps: []\n' > "$t/pr.yml"
  check "$t" > /dev/null || { echo "FAIL non-schedule workflow flagged"; fail=1; }
  [ "$fail" -eq 0 ] && echo "ok: self-test (6 rows)"; exit "$fail"
fi
check "${1:-.github/workflows}"
