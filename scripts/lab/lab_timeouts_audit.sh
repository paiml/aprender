#!/usr/bin/env bash
# scripts/lab/lab_timeouts_audit.sh -- PMAT-5847. LAB/nightly (schedule:) workflows are bounded: every job has timeout-minutes,
# and the cron start plus the longest job timeout ends by 05:00Z (a start before 05:00Z ends by 05:00Z that day,
# a later start by 05:00Z the next day). Operator, verbatim, 11:04Z: "Lab/nightly workflows get timeout-minutes
# and must end by 05:00Z". A LAB check: run on demand or at night, never inside a PR or release job (C324).
# A job that only calls a reusable workflow (uses:) cannot carry timeout-minutes; the callee is scanned itself.
# Usage: scripts/lab/lab_timeouts_audit.sh [workflows-dir] | --self-test       Exit: 0 green, 1 RED
set -euo pipefail
LIMIT=300 DAY=1440
DAY_FROM=300 DAY_TO=1080   # a start in 05:00-18:00Z runs in the working day
# Exceptions: none by default. release-rehearsal-nightly.yml (cron 20 5, 455 min lower bound) is RED here, one owner, on
# PMAT-5847; exempting it is pending quorum (see PMAT-5847). LAB_TIMEOUT_EXCEPT names basenames to skip, for a ruling or a test.
EXCEPT="${LAB_TIMEOUT_EXCEPT-}"
PATH_AWK=$(cat <<'AWK'
function rec(j,   k, best, v, n, a, i) {
  if (j in memo) return memo[j]
  best = 0; n = split(need[j], a, " ")
  for (i = 1; i <= n; i++) if (a[i] in tm) { v = rec(a[i]); if (v > best) best = v }
  return memo[j] = best + tm[j]
}
/^jobs:/ { in_jobs = 1; next }
in_jobs && /^  [A-Za-z0-9_-]+:[[:space:]]*(#.*)?$/ { name = $1; sub(/:$/, "", name); tm[name] = 0; need[name] = ""; innd = 0; next }
in_jobs && name && /^    timeout-minutes:/ { tm[name] = $2 + 0; innd = 0; next }
in_jobs && name && /^    needs:/ { s = $0; sub(/^    needs:[[:space:]]*/, "", s); gsub(/[\[\],]/, " ", s); need[name] = s; innd = (s ~ /^[[:space:]]*$/); next }
in_jobs && name && innd && /^      - / { s = $0; sub(/^      - /, "", s); need[name] = need[name] " " s; next }
in_jobs && name && /^    [A-Za-z]/ { innd = 0 }
END { m = 0; for (j in tm) { v = rec(j); if (v > m) m = v } print m }
AWK
)
scan() { # one workflow file -> RED lines
  local f="$1" j longest m h s end
  grep -qE '^[[:space:]]+schedule:' "$f" || return 0
  case " $EXCEPT " in *" $(basename "$f") "*) return 0 ;; esac
  for j in $(awk '/^jobs:/{in_jobs=1;next} in_jobs&&/^  [A-Za-z0-9_-]+:[[:space:]]*(#.*)?$/{if(name&&runs&&!seen)print name; name=$1; runs=0; seen=0; next} in_jobs&&/^    runs-on:/{runs=1} in_jobs&&/^    timeout-minutes:/{seen=1} END{if(name&&runs&&!seen)print name}' "$f"); do
    echo "RED $f job ${j%:}: no timeout-minutes"
  done
  longest=$(awk "$PATH_AWK" "$f")   # longest needs: chain, not the longest single job
  grep -E '^[[:space:]]+- cron:' "$f" | sed -E "s/.*cron:[[:space:]]*['\"]?([^'\"#]*).*/\1/" | while read -r m h _; do
    case "$m$h" in *[!0-9]*) continue ;; esac   # */10 or a range is not one nightly start
    s=$((h * 60 + m)); end=$((s + longest))
    if [ "$s" -ge "$DAY_FROM" ] && [ "$s" -lt "$DAY_TO" ]; then echo "RED $f: cron $m $h starts in the working day (05:00-18:00Z)"; continue; fi
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
  rm -f "${t:?}"/*.yml; printf 'on:\n  schedule:\n    - cron: "5 22 * * *"\njobs:\n  a:\n    runs-on: x\n    timeout-minutes: 200\n  b:\n    needs: a\n    runs-on: x\n    timeout-minutes: 200\n  c:\n    needs: [a, b]\n    runs-on: x\n    timeout-minutes: 100\n' > "$t/chain.yml"
  check "$t" > /dev/null && { echo "FAIL chain overrun (a+b+c=500) not RED"; fail=1; }
  w '0 12 * * *' $'    timeout-minutes: 10\n'; check "$t" > /dev/null && { echo "FAIL daytime start not RED"; fail=1; }
  rm -f "${t:?}"/*.yml; printf 'on:\n  pull_request:\njobs:\n  a:\n    runs-on: x\n    steps: []\n' > "$t/pr.yml"
  check "$t" > /dev/null || { echo "FAIL non-schedule workflow flagged"; fail=1; }
  # Pin: the exemption is by basename and defaults to empty. The same overrun is exempt only under its own name.
  rm -f "${t:?}"/*.yml; printf 'on:\n  schedule:\n    - cron: "20 5 * * *"\njobs:\n  a:\n    runs-on: x\n    timeout-minutes: 300\n' > "$t/release-rehearsal-nightly.yml"
  cp "$t/release-rehearsal-nightly.yml" "$t/other.yml"
  EXCEPT=""; check "$t" > /dev/null && { echo "FAIL default (empty) exemption: rehearsal overrun not RED"; fail=1; }
  EXCEPT="release-rehearsal-nightly.yml"; out=$(check "$t" || true)
  case "$out" in *other.yml*) ;; *) echo "FAIL exemption by name: other.yml not RED"; fail=1 ;; esac
  case "$out" in *release-rehearsal-nightly.yml*) echo "FAIL exemption by name: the named file still RED"; fail=1 ;; esac
  [ "$fail" -eq 0 ] && echo "ok: self-test (11 rows)"; exit "$fail"
fi
check "${1:-.github/workflows}"
