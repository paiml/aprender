#!/usr/bin/env bash
# #7277 (slice of #7181): dogfood-nightly is a LAB lane red 7+ nights. Guard that (a) its workflow has no
# `schedule:` trigger while stopped, (b) every step carries timeout-minutes, so a hung step ends with the
# runner's timeout line instead of a silent 88-minute exit 1. Restoring the cron is the owner's act:
# delete the STOPPED marker line in the workflow and this check's (a) with it.
# Usage: check_dogfood_nightly_lab.sh [WORKFLOW]   --self-test runs the planted-defect cases.
set -uo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)

check() { # workflow -> prints findings, rc 0 clean / 1 found
  local wf=$1 out
  out=$(awk '
    /^[[:space:]]*#/ { next }
    /^  schedule:/ { sched = 1 }
    /^      - (uses|name):/ { if (step && !hastm) miss = miss " " name; step = 1; hastm = 0; name = $0; sub(/^ *- (uses|name): */, "", name); name = substr(name, 1, 30); gsub(/ /, "_", name) }
    /^      - / && !/^      - (uses|name):/ { }
    /^        timeout-minutes:/ { hastm = 1 }
    /^  [a-z_-]+:$/ && !/^  (schedule|workflow_dispatch):/ { if (step && !hastm) miss = miss " " name; step = 0 }
    END { if (step && !hastm) miss = miss " " name
          if (sched) print "schedule trigger present while lane is STOPPED"
          if (miss != "") print "steps without timeout-minutes:" miss }' "$wf")
  [ -z "$out" ] && return 0
  printf '%s\n' "$out"; return 1
}

if [ "${1:-}" = --self-test ]; then
  t=$(mktemp -d) || exit 1; [ -n "$t" ] && [ -d "$t" ] || exit 1
  trap 'rm -rf -- "${t:?}"' EXIT
  good=$here/../.github/workflows/dogfood-nightly.yml
  check "$good" >/dev/null || { echo "FAIL real workflow not clean"; exit 1; }
  sed 's/^  workflow_dispatch:/  schedule:\n    - cron: "5 20 * * *"\n  workflow_dispatch:/' "$good" > "$t/a.yml"
  check "$t/a.yml" >/dev/null && { echo "FAIL planted schedule not caught"; exit 1; }
  grep -v '^        timeout-minutes:' "$good" > "$t/b.yml"
  check "$t/b.yml" >/dev/null && { echo "FAIL planted missing timeouts not caught"; exit 1; }
  echo "PASS dogfood-nightly LAB guard: 2 planted defects caught, real workflow clean"; exit 0
fi
wf=${1:-$here/../.github/workflows/dogfood-nightly.yml}
check "$wf"; rc=$?
[ "$rc" -eq 0 ] && echo "PASS dogfood-nightly stopped and every step has a timeout" || echo "FAIL dogfood-nightly LAB guard (#7277)"
exit "$rc"
