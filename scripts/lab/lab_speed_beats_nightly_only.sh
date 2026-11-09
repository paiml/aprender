#!/usr/bin/env bash
# scripts/lab/lab_speed_beats_nightly_only.sh -- aprender#6608 (spec-sweep, fable-architectural-review.md rule 7).
# Rule: "speed beats are nightly, non-blocking". A workflow that runs a speed beat (beat_*speed*) triggers ONLY on
# schedule / workflow_dispatch, and no per-PR fragment (ci/explicit-test-commands.d) names one.
# A LAB check: run on demand or at night, never inside a PR, merge-queue or release job (C324).
# Usage: scripts/lab/lab_speed_beats_nightly_only.sh [repo-root] | --self-test      Exit: 0 green, 1 RED, 2 cannot measure
set -euo pipefail
BEAT='beat_[a-z0-9_]*speed'
BAD_TRIGGERS='push|pull_request|pull_request_target|merge_group|workflow_call'

triggers() { # workflow file -> its top-level on: keys, one per line
  awk '/^on:/{o=1;next} o&&/^[^[:space:]#]/{o=0} o&&/^  [A-Za-z_]+:/{k=$1;sub(/:$/,"",k);print k}' "$1"
}
scan() { # repo-root -> RED lines; prints "SCANNED n" on the last line
  local root="$1" f t n=0 bad=0
  [ -d "$root/.github/workflows" ] || { echo "cannot measure: no $root/.github/workflows" >&2; return 2; }
  for f in "$root"/.github/workflows/*.yml; do
    grep -qE "$BEAT" "$f" || continue
    n=$((n + 1))
    for t in $(triggers "$f"); do
      if printf '%s' "$t" | grep -qE "^($BAD_TRIGGERS)$"; then echo "RED $(basename "$f"): speed beat in a $t workflow"; bad=1; fi
    done
  done
  if [ -d "$root/ci/explicit-test-commands.d" ]; then
    for f in "$root"/ci/explicit-test-commands.d/*; do
      [ -f "$f" ] || continue
      grep -qE "$BEAT" "$f" && { echo "RED ci/explicit-test-commands.d/$(basename "$f"): per-PR fragment names a speed beat"; bad=1; }
    done
  fi
  echo "SCANNED $n speed-beat workflow(s)"
  [ "$n" -gt 0 ] || { echo "cannot measure: 0 speed-beat workflows scanned" >&2; return 2; }
  return "$bad"
}
if [ "${1:-}" = --self-test ]; then
  t=$(mktemp -d); trap 'rm -rf -- "${t:?}"' EXIT; fail=0
  mk() { rm -rf -- "${t:?}"/*; mkdir -p "$t/.github/workflows" "$t/ci/explicit-test-commands.d"; printf 'on:\n%s\njobs: {}\n# cargo test beat_x_speed\n' "$1" > "$t/.github/workflows/w.yml"; }
  mk $'  schedule:\n    - cron: "5 3 * * *"\n  workflow_dispatch:'
  scan "$t" > /dev/null || { echo "FAIL nightly-only workflow is RED"; fail=1; }
  for trig in push pull_request merge_group; do   # planted violations
    mk $'  schedule:\n    - cron: "5 3 * * *"\n  '"$trig:"
    scan "$t" > /dev/null && { echo "FAIL $trig trigger passes (mutant)"; fail=1; }
  done
  mk $'  schedule:\n    - cron: "5 3 * * *"'; echo 'cargo test beat_y_speed' > "$t/ci/explicit-test-commands.d/010-x.cmd"
  scan "$t" > /dev/null && { echo "FAIL per-PR fragment passes (mutant)"; fail=1; }
  rm -rf -- "${t:?}/.github" "${t:?}/ci"; scan "$t" > /dev/null 2>&1 && { echo "FAIL empty root passes"; fail=1; }
  [ "$fail" = 0 ] && echo "lab_speed_beats_nightly_only: self-test PASS (6 rows)"; exit "$fail"
fi
scan "${1:-.}"
