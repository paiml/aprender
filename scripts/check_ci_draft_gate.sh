#!/usr/bin/env bash
# D1 (FLOW-010 runner-wait): a DRAFT pull request takes no runner slot.
#
# EQUALITY, not pattern matching. Three rounds of "the pattern accepts X" ended here: each job
# of ci.yml has exactly one canonical `if:` below, and the check is string equality with that
# value (a trailing ` # comment` stripped), byte for byte. The set of jobs in ci.yml must equal
# the CANON list, so a new job is RED until it is added here on purpose with its draft behaviour.
#   - jobs that ask for a clean-room/mq runner on a pull_request carry the draft guard in their `if:`;
#   - the three AGGREGATORS (workspace-test, ci-gate, gate) stay `if: always()`: they must exist and
#     FAIL on a draft (a skipped required check counts as passing on GitHub);
#   - on.pull_request.types must be exactly the canonical list, which includes ready_for_review.
#
# Usage: check_ci_draft_gate.sh [ci.yml]   |   check_ci_draft_gate.sh --self-test
set -euo pipefail

TYPES='types: [opened, synchronize, reopened, ready_for_review]'

# canon JOB -> the one allowed `if:` value (empty output = job not in the list)
canon() {
  case "$1" in
    mg-reuse) printf '%s' "github.event_name == 'push' && github.ref == 'refs/heads/main'" ;;
    mg-reuse-self-test | workspace-test-shard) printf '%s' '${{ github.event_name != '"'pull_request'"' || !github.event.pull_request.draft }}' ;;
    x86-main) printf '%s' '${{ !cancelled() && needs.mg-reuse.outputs.x86 != '"'1'"' && !(github.event_name == '"'pull_request'"' && github.event.pull_request.draft) }}' ;;
    x86-main-advisories) printf '%s' '${{ !cancelled() && needs.mg-reuse.outputs.x86 == '"'1'"' && !(github.event_name == '"'pull_request'"' && github.event.pull_request.draft) }}' ;;
    determinism) printf '%s' '${{ !cancelled() && needs.mg-reuse.outputs.det != '"'1'"' && github.event_name != '"'pull_request'"' }}' ;;
    workspace-test | ci-gate | gate) printf '%s' 'always()' ;;
  esac
}
JOBS="mg-reuse mg-reuse-self-test x86-main x86-main-advisories determinism workspace-test-shard workspace-test ci-gate gate"

# strip a trailing ` # comment` and trailing blanks
nocomment() { sed -E 's/[[:space:]]+#.*$//; s/[[:space:]]+$//'; }

# jobs_of FILE -> every 2-space key under `jobs:` (a trailing comment on the header is allowed)
jobs_of() {
  awk '/^jobs:/ {j = 1; next} j && /^[^ #]/ {j = 0} j && /^  [^ #][^:]*:/ {sub(/^  /, ""); sub(/:.*/, ""); print}' "$1"
}

# ifof FILE JOB -> the job's `    if:` value (comment stripped), or nothing
ifof() {
  awk -v job="$2" '
    /^  [^ #][^:]*:/ { n = $0; sub(/^  /, "", n); sub(/:.*/, "", n); f = (n == job); next }
    f && index($0, "    if: ") == 1 { v = substr($0, 9); print v; exit }' "$1" | nocomment
}

check() {
  local f="$1" bad=0 job want got seen=0
  # the pull_request `types:` line is the only 4-space `types:` key in ci.yml
  got=$(/usr/bin/grep -E '^    types: ' "$f" | nocomment | sed -E 's/^ +//' || true)
  [ "$got" = "$TYPES" ] || { echo "FAIL: on.pull_request.types is '$got', want '$TYPES'"; bad=1; }
  while IFS= read -r job; do
    seen=$((seen + 1))
    want=$(canon "$job")
    [ -n "$want" ] || { echo "FAIL: job $job is not in the CANON list of check_ci_draft_gate.sh (add it with its draft behaviour)"; bad=1; continue; }
    got=$(ifof "$f" "$job")
    [ "$got" = "$want" ] || { echo "FAIL: job $job: if: '$got' != canonical '$want'"; bad=1; }
  done < <(jobs_of "$f")
  # every listed job must exist: a parse that finds fewer must not pass
  for job in $JOBS; do
    jobs_of "$f" | /usr/bin/grep -qx "$job" || { echo "FAIL: listed job $job not found in $f"; bad=1; }
  done
  [ "$seen" -ge 1 ] || { echo "FAIL: parser saw no jobs"; bad=1; }
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
  # Mutation 11: `&& true != !(...)` — a comparison that makes the guard vacuous -> must be RED.
  sed "s/ && !(github.event_name == 'pull_request'/ \&\& true != !(github.event_name == 'pull_request'/" "$src" >"$t/m11.yml"
  if check "$t/m11.yml" >/dev/null; then echo "self-test: mutation 11 (true != !(...)) not caught"; exit 1; fi
  # Mutation 12: a double negation `! !(...)` -> must be RED.
  sed "s/ && !(github.event_name == 'pull_request'/ \&\& ! !(github.event_name == 'pull_request'/" "$src" >"$t/m12.yml"
  if check "$t/m12.yml" >/dev/null; then echo "self-test: mutation 12 (! !(...)) not caught"; exit 1; fi
  # Mutation 13: a job whose header carries a comment suffix and has lost its `if:` -> must be RED.
  awk '/^  workspace-test-shard:/ {print "  workspace-test-shard: # no guard here"; g = 1; next} g && /^    if:/ {g = 0; next} {print}' "$src" >"$t/m13.yml"
  if check "$t/m13.yml" >/dev/null; then echo "self-test: mutation 13 (comment-suffixed unguarded job) not caught"; exit 1; fi
  # Mutation 14: a brand-new unguarded pool job (not in CANON), header with a comment -> must be RED.
  awk '{print} /^jobs:/ {print "  extra-job: # unguarded"; print "    runs-on: [self-hosted, Linux, clean-room]"; print "    steps: []"}' "$src" >"$t/m14.yml"
  if check "$t/m14.yml" >/dev/null; then echo "self-test: mutation 14 (new job outside CANON) not caught"; exit 1; fi
  # Mutation 15: a trailing comment is stripped, but a comment hiding a padded guard must not help -> RED.
  sed "s/draft) }}\$/draft) || true }} # fine/" "$src" >"$t/m15.yml"
  if check "$t/m15.yml" >/dev/null; then echo "self-test: mutation 15 (|| true then comment) not caught"; exit 1; fi
  # Mutation 6: nothing parses (empty jobs) -> must be RED, not vacuously green.
  printf 'on:\n  pull_request:\n    types: [ready_for_review]\njobs:\n' >"$t/m6.yml"
  if check "$t/m6.yml" >/dev/null; then echo "self-test: mutation 6 (empty parse) not caught"; exit 1; fi
  echo "check_ci_draft_gate: self-test OK (15 mutations caught)"
  exit 0
fi
check "${1:-.github/workflows/ci.yml}" && echo "check_ci_draft_gate: OK"
