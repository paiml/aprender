#!/usr/bin/env bash
# The three arch-neutral required jobs must carry X64 in runs-on: gx10 aarch64 runners die at job
# start (0 steps) and a dead required job marks the whole merge-queue group UNMERGEABLE (#6200).
# Usage: check_ci_required_x64.sh [ci.yml]   |   check_ci_required_x64.sh --self-test
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS="mg-reuse-self-test ci-gate gate"

# Print each job in $JOBS whose runs-on line lacks X64 or that is missing.
bad_jobs() {
  local f=$1 j
  for j in $JOBS; do
    awk -v j="$j" '
      $0 ~ "^  "j":[[:space:]]*$" { in_j=1; found=1; next }
      in_j && /^  [A-Za-z0-9_-]+:[[:space:]]*$/ { in_j=0 }
      in_j && /^    runs-on:/ { seen=1; l=$0; sub(/#.*/,"",l); if (l !~ /X64/) bad=1 }
      END { if (!found || !seen || bad) print j }' "$f"
  done
}

if [ "${1:-}" = "--self-test" ]; then
  t=$(mktemp -d); trap 'rm -rf "${t:?}"' EXIT
  base=.github/workflows/ci.yml
  [ -z "$(bad_jobs "$base")" ] || { echo "FAIL: real ci.yml is not clean"; exit 1; }
  for j in $JOBS; do
    # mutation: drop X64 from this job's runs-on only
    awk -v j="$j" '$0 ~ "^  "j":[[:space:]]*$" {m=1} m && /^    runs-on:/ { sub(/X64, /,""); m=0 } {print}' "$base" > "$t/m.yml"
    [ "$(bad_jobs "$t/m.yml")" = "$j" ] || { echo "FAIL: dropping X64 from $j not caught"; exit 1; }
  done
  grep -v '^  gate:' "$base" > "$t/m.yml"   # mutation: job renamed/removed
  [ "$(bad_jobs "$t/m.yml")" = "gate" ] || { echo "FAIL: missing gate job not caught"; exit 1; }
  echo "ok: self-test (3 per-job mutations + 1 missing job)"; exit 0
fi

out=$(bad_jobs "${1:-.github/workflows/ci.yml}")
[ -z "$out" ] || { echo "FAIL: runs-on lacks X64 (or job missing): $out" >&2; exit 1; }
echo "ok: mg-reuse-self-test, ci-gate and gate all pin X64"
