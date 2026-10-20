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

check() {
  local f="$1" bad=0 job runs iff
  yq -e '.on.pull_request.types | contains(["ready_for_review"])' "$f" >/dev/null 2>&1 \
    || { echo "FAIL: on.pull_request.types lacks ready_for_review"; bad=1; }
  while IFS= read -r job; do
    runs=$(yq -r ".jobs.\"$job\".\"runs-on\" | tostring" "$f")
    case "$runs" in *clean-room* | *mq*) ;; *) continue ;; esac
    iff=$(yq -r ".jobs.\"$job\".if // \"\"" "$f")
    if [[ "$AGGREGATORS" == *" $job "* ]]; then
      [ "$iff" = "always()" ] || { echo "FAIL: aggregator $job must be if: always() (got '$iff')"; bad=1; }
      continue
    fi
    # merge_group untouched: a draft guard must be qualified by the pull_request event.
    if [[ "$iff" == *pull_request.draft* && "$iff" != *"event_name == 'pull_request'"* && "$iff" != *"event_name != 'pull_request'"* ]]; then
      echo "FAIL: job $job draft guard is not scoped to the pull_request event (would skip merge_group)"; bad=1
    fi
    case "$iff" in
      *pull_request.draft* | *"event_name == 'push'"* | *"event_name != 'pull_request'"*) ;;
      *) echo "FAIL: job $job takes a runner slot on a draft PR (if: '$iff')"; bad=1 ;;
    esac
  done < <(yq -r '.jobs | keys | .[]' "$f")
  return "$bad"
}

if [ "${1:-}" = "--self-test" ]; then
  src="${CI_YML:-.github/workflows/ci.yml}"
  check "$src" || { echo "self-test: real workflow is not draft-gated"; exit 1; }
  t=$(mktemp -d); trap 'rm -rf -- "${t:?}"' EXIT
  # Mutation 1: drop the draft guard from x86-main -> must be RED.
  sed "s/ && !(github.event_name == 'pull_request' \&\& github.event.pull_request.draft)//" "$src" >"$t/m1.yml"
  if check "$t/m1.yml" >/dev/null; then echo "self-test: mutation 1 (no draft guard) not caught"; exit 1; fi
  # Mutation 2: drop ready_for_review -> must be RED.
  sed 's/, ready_for_review//' "$src" >"$t/m2.yml"
  if check "$t/m2.yml" >/dev/null; then echo "self-test: mutation 2 (no ready_for_review) not caught"; exit 1; fi
  # Mutation 3: unscope the draft guard (draft is empty on merge_group -> would still run, but the scope rule must fire).
  sed "s/!(github.event_name == 'pull_request' && github.event.pull_request.draft)/!github.event.pull_request.draft/" "$src" >"$t/m3.yml"
  if check "$t/m3.yml" >/dev/null; then echo "self-test: mutation 3 (unscoped guard) not caught"; exit 1; fi
  echo "check_ci_draft_gate: self-test OK (3 mutations caught)"
  exit 0
fi
check "${1:-.github/workflows/ci.yml}" && echo "check_ci_draft_gate: OK"
