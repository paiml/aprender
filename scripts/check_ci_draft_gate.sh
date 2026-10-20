#!/usr/bin/env bash
# D1 (FLOW-010 runner-wait): a DRAFT pull request takes no runner slot.
#
# Every job of ci.yml that asks for the clean-room or mq pool must either
#   - carry `pull_request.draft` in its `if:` (skipped on a draft), or
#   - never run on a pull_request (`push`-only, or `event_name != 'pull_request'`), or
#   - be one of the AGGREGATORS below, which stay `if: always()` so the three
#     required check names exist and FAIL on a draft (a skipped required check
#     counts as passing on GitHub, so skipping them would let an armed draft merge).
# The workflow must also re-run on `ready_for_review`, or a draft flipped to
# ready would never get CI.
#
# Usage: check_ci_draft_gate.sh [ci.yml]   |   check_ci_draft_gate.sh --self-test
set -euo pipefail

AGGREGATORS=" workspace-test ci-gate gate "

# jobfield FILE JOB KEY -> the single-line value of `    KEY:` inside job JOB (empty if absent)
jobfield() {
  awk -v job="  $2:" -v key="    $3:" '
    $0 == job {f = 1; next}
    f && /^  [A-Za-z0-9_-]+:[ ]*$/ {exit}
    f && index($0, key) == 1 {v = substr($0, length(key) + 1); sub(/^ +/, "", v); print v; exit}' "$1"
}

# The only two draft-guard forms ci.yml uses; anything else (a flipped polarity, a bare
# `pull_request.draft`) is RED. G_ALL covers the jobs that run on every event, G_PQ the
# `!cancelled() && ...` jobs.
G_ALL="github.event_name != 'pull_request' || !github.event.pull_request.draft"
G_PQ="!(github.event_name == 'pull_request' && github.event.pull_request.draft)"

check() {
  local f="$1" bad=0 job runs iff pool=0 aggs=0
  # no yq on the runner: the pull_request `types:` line is the only 4-space `types:` key in ci.yml
  grep -qE '^    types: \[.*ready_for_review' "$f" \
    || { echo "FAIL: on.pull_request.types lacks ready_for_review"; bad=1; }
  while IFS= read -r job; do
    iff=$(jobfield "$f" "$job" if)
    if [[ "$AGGREGATORS" == *" $job "* ]]; then
      aggs=$((aggs + 1))
      [ "$iff" = "always()" ] || { echo "FAIL: aggregator $job must be if: always() (got '$iff')"; bad=1; }
      continue
    fi
    runs=$(jobfield "$f" "$job" runs-on)
    case "$runs" in *clean-room* | *mq*) ;; *) continue ;; esac
    pool=$((pool + 1))
    case "$iff" in
      *"$G_ALL"* | *"$G_PQ"* | *"&& github.event_name != 'pull_request' }}"* | *"event_name == 'push'"*) ;;
      *) echo "FAIL: job $job takes a runner slot on a draft PR, or its draft guard is not one of the two allowed forms (if: '$iff')"; bad=1 ;;
    esac
  done < <(awk '/^jobs:/ {j = 1; next} j && /^[^ #]/ {j = 0} j && /^  [A-Za-z0-9_-]+:[ ]*$/ {sub(/^  /, ""); sub(/:.*/, ""); print}' "$f")
  # a parse that finds nothing must not pass
  [ "$aggs" -eq 3 ] || { echo "FAIL: saw $aggs of the 3 aggregators (workspace-test, ci-gate, gate)"; bad=1; }
  [ "$pool" -ge 1 ] || { echo "FAIL: no clean-room/mq job found — parser saw nothing"; bad=1; }
  return "$bad"
}

if [ "${1:-}" = "--self-test" ]; then
  src="${CI_YML:-.github/workflows/ci.yml}"
  check "$src" || { echo "self-test: real workflow is not draft-gated"; exit 1; }
  t=$(mktemp -d); trap 'rm -rf "${t:?}"' EXIT
  # Mutation 1: drop the draft guard from x86-main -> must be RED.
  sed "s/ && !(github.event_name == 'pull_request' \&\& github.event.pull_request.draft)//" "$src" >"$t/m1.yml"
  if check "$t/m1.yml" >/dev/null; then echo "self-test: mutation 1 (no draft guard) not caught"; exit 1; fi
  # Mutation 2: drop ready_for_review -> must be RED.
  sed 's/, ready_for_review//' "$src" >"$t/m2.yml"
  if check "$t/m2.yml" >/dev/null; then echo "self-test: mutation 2 (no ready_for_review) not caught"; exit 1; fi
  # Mutation 3: unscope the draft guard (draft is empty on merge_group -> would still run, but the scope rule must fire).
  sed "s/!(github.event_name == 'pull_request' && github.event.pull_request.draft)/!github.event.pull_request.draft/" "$src" >"$t/m3.yml"
  if check "$t/m3.yml" >/dev/null; then echo "self-test: mutation 3 (unscoped guard) not caught"; exit 1; fi
  # Mutation 4: flip the polarity of the guard (job would run ONLY on drafts) -> must be RED.
  sed "s/!(github.event_name == 'pull_request' \&\& github.event.pull_request.draft)/(github.event_name == 'pull_request' \&\& github.event.pull_request.draft)/" "$src" >"$t/m4.yml"
  if check "$t/m4.yml" >/dev/null; then echo "self-test: mutation 4 (flipped polarity) not caught"; exit 1; fi
  # Mutation 5: an aggregator loses if: always() -> must be RED.
  awk '/^  gate:$/ {g = 1} g && /^    if: always\(\)/ {sub(/always\(\)/, "success()"); g = 0} {print}' "$src" >"$t/m5.yml"
  if check "$t/m5.yml" >/dev/null; then echo "self-test: mutation 5 (aggregator not always) not caught"; exit 1; fi
  # Mutation 6: nothing parses (empty jobs) -> must be RED, not vacuously green.
  printf 'on:\n  pull_request:\n    types: [ready_for_review]\njobs:\n' >"$t/m6.yml"
  if check "$t/m6.yml" >/dev/null; then echo "self-test: mutation 6 (empty parse) not caught"; exit 1; fi
  echo "check_ci_draft_gate: self-test OK (6 mutations caught)"
  exit 0
fi
check "${1:-.github/workflows/ci.yml}" && echo "check_ci_draft_gate: OK"
