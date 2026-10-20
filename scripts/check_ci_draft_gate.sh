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

# draft_guard_ok IF -> 0 when the whole `if:` value is an allowed form. The comparison is on the WHOLE
# expression (the `${{ }}` wrapper stripped), never a substring: a padded `... || true`, a trailing
# `# comment` or `push || true` must not pass. A form is the exact guard, or a conjunction that adds
# only `&&`-ed terms (no `||`, no `#`) in front of it, so extra terms can only narrow when the job runs.
draft_guard_ok() {
  local e="$1"
  e="${e#\$\{\{ }"; e="${e% \}\}}"
  [[ "$e" == *"||"* && "$e" != "$G_ALL" ]] && return 1
  [[ "$e" == *"#"* || "$e" == *"!!"* ]] && return 1
  [ "$e" = "$G_ALL" ] && return 0
  [[ "$e" == *"$G_PQ" ]] && return 0
  [[ "$e" == *" && github.event_name != 'pull_request'" ]] && return 0
  [[ "$e" == "github.event_name == 'push' && "* ]] && return 0
  return 1
}

check() {
  local f="$1" bad=0 job runs iff pool=0 aggs=0
  # no yq on the runner: the pull_request `types:` line is the only 4-space `types:` key in ci.yml
  grep -qE '^    types: \[([^]]*, ?)?ready_for_review(,[^]]*)?\]' "$f" \
    || { echo "FAIL: on.pull_request.types lacks ready_for_review"; bad=1; }
  while IFS= read -r job; do
    iff=$(jobfield "$f" "$job" if)
    if [[ "$AGGREGATORS" == *" $job "* ]]; then
      aggs=$((aggs + 1))
      [ "$iff" = "always()" ] || { echo "FAIL: aggregator $job must be if: always() (got '$iff')"; bad=1; }
      continue
    fi
    runs=$(jobfield "$f" "$job" runs-on)
    # a runs-on the single-line parser cannot read (block list) on a job that is not a reusable call
    # would be skipped silently, so it is RED, not "not a pool job"
    if [ -z "$runs" ] && [ -z "$(jobfield "$f" "$job" uses)" ]; then
      echo "FAIL: job $job has no single-line runs-on and no uses: — the draft gate cannot read it"; bad=1; continue
    fi
    case "$runs" in *clean-room* | *mq*) ;; *) continue ;; esac
    pool=$((pool + 1))
    if ! draft_guard_ok "$iff"; then
      echo "FAIL: job $job takes a runner slot on a draft PR, or its draft guard is not one of the allowed exact forms (if: '$iff')"; bad=1
    fi
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
  # Mutation 7: a padded guard (`|| true`) on a G_PQ job and on a G_ALL job -> must be RED.
  sed "s/draft) }}/draft) || true }}/" "$src" >"$t/m7.yml"
  if check "$t/m7.yml" >/dev/null; then echo "self-test: mutation 7 (guard padded with || true) not caught"; exit 1; fi
  sed "s/!github.event.pull_request.draft }}/!github.event.pull_request.draft || true }}/" "$src" >"$t/m7b.yml"
  if check "$t/m7b.yml" >/dev/null; then echo "self-test: mutation 7b (G_ALL padded with || true) not caught"; exit 1; fi
  # Mutation 8: a push-looking if that is true on a draft -> must be RED.
  awk '/^  workspace-test-shard:/ {g = 1} g && /^    if:/ {print "    if: ${{ github.event_name == '"'"'push'"'"' || true }}"; g = 0; next} {print}' "$src" >"$t/m8.yml"
  if check "$t/m8.yml" >/dev/null; then echo "self-test: mutation 8 (push || true) not caught"; exit 1; fi
  # Mutation 9: a near-miss ready_for_review token -> must be RED.
  sed 's/ready_for_review\]/ready_for_review_x]/' "$src" >"$t/m9.yml"
  if check "$t/m9.yml" >/dev/null; then echo "self-test: mutation 9 (ready_for_review_x) not caught"; exit 1; fi
  # Mutation 10: a job whose runs-on is a block list the parser cannot read -> must be RED.
  awk '/^  mg-reuse-self-test:/ {g = 1} g && /^    runs-on:/ {print "    runs-on:"; print "      - self-hosted"; print "      - clean-room"; g = 0; next} {print}' "$src" >"$t/m10.yml"
  if check "$t/m10.yml" >/dev/null; then echo "self-test: mutation 10 (block-list runs-on) not caught"; exit 1; fi
  # Mutation 6: nothing parses (empty jobs) -> must be RED, not vacuously green.
  printf 'on:\n  pull_request:\n    types: [ready_for_review]\njobs:\n' >"$t/m6.yml"
  if check "$t/m6.yml" >/dev/null; then echo "self-test: mutation 6 (empty parse) not caught"; exit 1; fi
  echo "check_ci_draft_gate: self-test OK (10 mutations caught)"
  exit 0
fi
check "${1:-.github/workflows/ci.yml}" && echo "check_ci_draft_gate: OK"
