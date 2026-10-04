#!/usr/bin/env bash
# check_ci_push_reuse.sh - the case table for T36: a push to main reuses the merge queue's
# workspace-test result on the same commit.
#
# The merge queue tests a commit, then fast-forwards main to that very commit; the push run on
# main used to re-test it (about 103 runner-min per merge, most of it the three workspace-test
# shards). Now the push run's tier step asks GitHub for the merge_group workspace-test result on
# its own sha (scripts/ci_mg_workspace_result.sh) and scripts/ci_test_tier.sh reuses it only
# when that result is a success for exactly this commit. Any doubt is today's tier.
#
#   L rows  the lookup, against a stub `gh` serving canned JSON (no network)
#   D rows  the decision, scripts/ci_test_tier.sh --event push on throwaway repositories
#   W rows  the tier step in ci/sections.yml calls the lookup on push, and survives its failure
#   A rows  the workspace-test aggregator in ci.yml still refuses a split tier
#
#   bash scripts/check_ci_push_reuse.sh              # the table
#   bash scripts/check_ci_push_reuse.sh --self-test  # delete each `# R-*` refusal line in a copy
#                                                    # of the subjects; each must turn the table RED
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOOKUP="${PUSH_REUSE_LOOKUP:-$ROOT/scripts/ci_mg_workspace_result.sh}"
TIER="${PUSH_REUSE_TIER:-$ROOT/scripts/ci_test_tier.sh}"

case "${1:-}" in -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

if [ "${1:-}" = "--self-test" ]; then
  # A red table kills every mutant for free: the unmutated table must be green first.
  bash "$0" >/dev/null 2>&1 || { echo "check_ci_push_reuse: the unmutated table is red - mutants not_measured"; exit 1; }
  work="$(mktemp -d)"; killed=0; total=0
  for subj in "$ROOT/scripts/ci_mg_workspace_result.sh" "$ROOT/scripts/ci_test_tier.sh"; do
    mapfile -t markers < <(grep -o '# R-[A-Z]*$' "$subj" | sed 's/^# //')
    [ "${#markers[@]}" -gt 0 ] || { echo "check_ci_push_reuse: no R-* markers in ${subj##*/} (vacuous)"; rm -rf "${work:?}"; exit 1; }
    for m in "${markers[@]}"; do
      total=$((total+1))
      grep -v "# $m\$" "$subj" >"$work/subject.sh"
      # A mutation that changes nothing is an error, never a survivor and never a kill.
      if cmp -s "$subj" "$work/subject.sh"; then echo "  ERROR    ${subj##*/} $m (the mutation changed nothing)"; continue; fi
      if [ "${subj##*/}" = ci_mg_workspace_result.sh ]; then PUSH_REUSE_LOOKUP="$work/subject.sh" timeout 120 bash "$0" >/dev/null 2>&1; rc=$?
      else PUSH_REUSE_TIER="$work/subject.sh" timeout 120 bash "$0" >/dev/null 2>&1; rc=$?; fi
      if [ "$rc" -eq 0 ]; then echo "  SURVIVED ${subj##*/} $m"; else killed=$((killed+1)); echo "  killed   ${subj##*/} $m"; fi
    done
  done
  rm -rf "${work:?}"
  echo "check_ci_push_reuse: mutants $killed/$total killed"
  [ "$killed" -eq "$total" ]; exit $?
fi

for f in "$LOOKUP" "$TIER"; do [ -f "$f" ] || { echo "check_ci_push_reuse: missing $f"; exit 1; }; done

T="$(mktemp -d)"
trap 'rm -rf "${T:?}"' EXIT
fails=0
ok()    { printf '  ok    %s\n' "$1"; }
bad()   { fails=$((fails+1)); printf '  FAIL  %s\n     expected: %s\n     actual:   %s\n' "$1" "$2" "$3"; }
want()  { if [ "$2" = "$3" ]; then ok "$1"; else bad "$1" "$2" "$3"; fi; }
has()   { case "$3" in *"$2"*) ok "$1";; *) bad "$1" "output containing '$2'" "$3";; esac; }
hasnt() { case "$3" in *"$2"*) bad "$1" "no '$2'" "$3";; *) ok "$1";; esac; }

# -- the stub gh: `gh api <path> --jq <expr>` -> jq over $T/canned/<suites|runs>.json; a FAIL file fails it
mkdir -p "$T/bin" "$T/canned"
cat >"$T/bin/gh" <<'STUB'
#!/usr/bin/env bash
c="$(dirname "$(dirname "$0")")/canned"
[ "$1" = api ] || exit 9
case "$2" in
  */check-suites*) k=suites ;;
  */check-runs*)   k=runs ;;
  *) echo "stub gh: unexpected path $2" >&2; exit 9 ;;
esac
echo "$2" >>"$c/calls"
[ -f "$c/$k.FAIL" ] && { echo "gh: HTTP 502" >&2; exit 1; }
[ "$3" = --jq ] || exit 9
jq -r "$4" <"$c/$k.json"
STUB
chmod +x "$T/bin/gh"
export PATH="$T/bin:$PATH"

SHA=1111111111111111111111111111111111111111
OTHER=2222222222222222222222222222222222222222
QUEUE=gh-readonly-queue/main/pr-4700-0000000000000000000000000000000000000000
suite() { printf '{"id":%s,"head_branch":"%s","app":{"slug":"%s"}}' "$1" "$2" "${3:-github-actions}"; }
crun()  { printf '{"status":"%s","conclusion":%s,"head_sha":"%s","completed_at":"%s","check_suite":{"id":%s},"details_url":"https://github.com/o/r/actions/runs/%s/job/9"}' "$1" "$2" "$3" "$4" "$5" "$6"; }
canned() { # canned <suites-json-list> <runs-json-list>
  rm -f "${T:?}/canned/suites.FAIL" "${T:?}/canned/runs.FAIL" "${T:?}/canned/calls"
  printf '{"check_suites":[%s]}' "$1" >"$T/canned/suites.json"
  printf '{"check_runs":[%s]}' "$2" >"$T/canned/runs.json"
}
look() { OUT="$(bash "$LOOKUP" o/r "${1:-$SHA}" 2>"$T/err")"; RC=$?; ERR="$(cat "$T/err")"; }
flat() { tr '\n' ' ' <<<"$1" | sed 's/ $//'; }

echo "check_ci_push_reuse: T36, a push to main reuses the merge queue's workspace-test on the same sha"

# -- L: the lookup ---------------------------------------------------------------------------------
canned "$(suite 50 "$QUEUE"),$(suite 60 main)" \
       "$(crun completed '"success"' $SHA 2026-10-04T10:00:00Z 50 7001),$(crun completed '"failure"' $SHA 2026-10-04T11:00:00Z 60 7002)"
look
want "L1 a merge-queue workspace-test success on the sha -> found, exit 0" 0 "$RC"
want "L1 prints the run, the sha the API reports, the conclusion (the push suite's later failure is not the queue's)" \
     "--mg-run 7001 --mg-sha $SHA --mg-conclusion success" "$(flat "$OUT")"
want "L1 costs exactly 2 API calls" 2 "$(wc -l <"$T/canned/calls" | tr -d ' ')"

canned "$(suite 50 "$QUEUE")" "$(crun completed '"failure"' $SHA 2026-10-04T10:00:00Z 50 7003)"
look
want "L2 the merge-queue workspace-test failed -> still printed (the decision refuses it)" \
     "--mg-run 7003 --mg-sha $SHA --mg-conclusion failure" "$(flat "$OUT")"

canned "$(suite 50 "$QUEUE")" "$(crun completed '"cancelled"' $SHA 2026-10-04T10:00:00Z 50 7004),$(crun completed '"success"' $SHA 2026-10-04T09:00:00Z 50 7005)"
look
want "L3 the LATEST merge-queue result wins (a cancelled re-run over an older success)" \
     "--mg-run 7004 --mg-sha $SHA --mg-conclusion cancelled" "$(flat "$OUT")"

canned "$(suite 60 main)" "$(crun completed '"success"' $SHA 2026-10-04T10:00:00Z 60 7006)"
look
want "L4 only a push suite (not the merge queue) -> nothing, exit 1" "1:" "$RC:$OUT"
has "L4 says so" "no merge-queue check suite" "$ERR"

canned "$(suite 50 "$QUEUE")" "$(crun in_progress null $SHA 2026-10-04T10:00:00Z 50 7007),$(crun completed '"success"' $SHA 2026-10-04T10:00:00Z 60 7008)"
look
want "L5 the queue's workspace-test still running (another suite's success only) -> nothing, exit 1" "1:" "$RC:$OUT"
has "L5 says so" "no completed merge-queue workspace-test" "$ERR"

canned "$(suite 50 "$QUEUE" some-other-app)" "$(crun completed '"success"' $SHA 2026-10-04T10:00:00Z 50 7009)"
look
want "L6 a queue-branch suite from another app -> nothing, exit 1" "1:" "$RC:$OUT"

canned "$(suite 50 "$QUEUE")" "$(crun completed '"success"' $SHA 2026-10-04T10:00:00Z 50 7010)"
: >"$T/canned/suites.FAIL"
look
want "L7 the check-suites call fails -> nothing, exit 1 (a failed lookup never reuses)" "1:" "$RC:$OUT"
has "L7 names the failed call" "check-suites lookup failed" "$ERR"

canned "$(suite 50 "$QUEUE")" "$(crun completed '"success"' $SHA 2026-10-04T10:00:00Z 50 7011)"
: >"$T/canned/runs.FAIL"
look
want "L8 the check-runs call fails -> nothing, exit 1" "1:" "$RC:$OUT"
has "L8 names the failed call" "check-runs lookup failed" "$ERR"

canned "$(suite 50 "$QUEUE")" "$(crun completed '"success"' $SHA 2026-10-04T10:00:00Z 50 'x')"
look
want "L9 a details_url with no run id -> nothing, exit 1" "1:" "$RC:$OUT"

canned "$(suite 50 "$QUEUE")" "$(crun completed '"success"' $OTHER 2026-10-04T10:00:00Z 50 7012)"
look
want "L10 the API's head_sha is reported, never the sha asked about" \
     "--mg-run 7012 --mg-sha $OTHER --mg-conclusion success" "$(flat "$OUT")"

canned "$(suite 50 "$QUEUE")" "$(crun completed '"success"' $SHA 2026-10-04T10:00:00Z 50 7013)"
OUT="$(bash "$LOOKUP" o/r 'not-a-sha' 2>&1)"; RC=$?
want "L11 a sha that is not 40 hex -> exit 1" 1 "$RC"
want "L11 and no API call was made" 0 "$( [ -f "$T/canned/calls" ] && wc -l <"$T/canned/calls" | tr -d ' ' || echo 0)"
OUT="$(bash "$LOOKUP" 'not a repo' "$SHA" 2>&1)"; RC=$?
want "L12 a repository that is not owner/repo -> exit 1" 1 "$RC"
want "L12 and no API call was made" 0 "$( [ -f "$T/canned/calls" ] && wc -l <"$T/canned/calls" | tr -d ' ' || echo 0)"

canned "$(suite 50 "$QUEUE")" "$(crun completed '"success"' abc 2026-10-04T10:00:00Z 50 7014)"
look
want "L13 the check run reports a head_sha that is not a sha -> nothing, exit 1" "1:" "$RC:$OUT"

# -- D: the decision -------------------------------------------------------------------------------
mkrepo() { # mkrepo <dir> -> a two-commit repo whose push diff is docs-only (today's tier: none, no cargo)
  git init -q -b main "$1"
  git -C "$1" -c user.name=t -c user.email=t@t commit -q --allow-empty -m base
  mkdir -p "$1/docs/roadmaps"; printf 'x: 1\n' >"$1/docs/roadmaps/x.yaml"
  git -C "$1" add docs; git -C "$1" -c user.name=t -c user.email=t@t commit -q -m docs
}
mkrepo "$T/r"; H="$(git -C "$T/r" rev-parse HEAD)"
tier() { (cd "$ROOT" && env -u GITHUB_SHA ${GSHA:+GITHUB_SHA=$GSHA} bash "$TIER" --event push --repo-root "$T/r" "$@" 2>&1); }
BASE="$(tier)"
want "D0 push with no merge_group result -> today's tier (docs-only: none)" "tier=none" "$(grep '^tier=' <<<"$BASE")"
hasnt "D0 and no note about a merge_group (no --mg-* given: as before)" "merge_group" "$BASE"

OUT="$(tier --mg-run 7001 --mg-sha "$H" --mg-conclusion success)"
want "D1 push + merge_group success on this very commit -> reuse" "tier=reuse" "$(grep '^tier=' <<<"$OUT")"
want "D1 cites the merge_group run id" "cite=7001" "$(grep '^cite=' <<<"$OUT")"
has "D1 the reason names the run and the commit" "merge_group run 7001 tested this very commit" "$OUT"

for c in failure cancelled timed_out ''; do
  OUT="$(tier --mg-run 7001 --mg-sha "$H" --mg-conclusion "$c")"
  want "D2 merge_group workspace-test concluded '${c:-<empty>}' -> today's tier" "tier=none" "$(grep '^tier=' <<<"$OUT")"
  has "D2 '${c:-<empty>}' the reason says why it was not reused" "not success" "$OUT"
done

OUT="$(tier --mg-run 7001 --mg-sha "$OTHER" --mg-conclusion success)"
want "D3 the cited run tested another sha -> today's tier" "tier=none" "$(grep '^tier=' <<<"$OUT")"
has "D3 says which sha it tested" "tested $OTHER, not HEAD" "$OUT"

OUT="$(GSHA=$OTHER tier --mg-run 7001 --mg-sha "$H" --mg-conclusion success)"
want "D4 GITHUB_SHA is not HEAD -> today's tier" "tier=none" "$(grep '^tier=' <<<"$OUT")"
has "D4 says so" "GITHUB_SHA $OTHER is not HEAD" "$OUT"

OUT="$(GSHA=$H tier --mg-run 7001 --mg-sha "$H" --mg-conclusion success)"
want "D5 GITHUB_SHA equal to HEAD -> reuse" "tier=reuse" "$(grep '^tier=' <<<"$OUT")"

OUT="$(tier --mg-run 'x7' --mg-sha "$H" --mg-conclusion success)"
want "D6 a run id that is not a number -> today's tier" "tier=none" "$(grep '^tier=' <<<"$OUT")"
OUT="$(tier --mg-sha "$H" --mg-conclusion success)"
want "D7 no run id at all -> today's tier" "tier=none" "$(grep '^tier=' <<<"$OUT")"

# a root commit: today's tier is full (no diff can be derived); a refused result keeps it full
git init -q -b main "$T/root"; git -C "$T/root" -c user.name=t -c user.email=t@t commit -q --allow-empty -m only
OUT="$(cd "$ROOT" && env -u GITHUB_SHA bash "$TIER" --event push --repo-root "$T/root" --mg-run 7001 --mg-sha "$H" --mg-conclusion success 2>&1)"
want "D8 a refused result on a root commit keeps today's full" "tier=full" "$(grep '^tier=' <<<"$OUT")"

# merge_group and pull_request never read --mg-*; merge_group's own reuse is unchanged
git -C "$T/r" -c user.name=t -c user.email=t@t commit -q --allow-empty -m "queue, same tree"
P="$(git -C "$T/r" rev-parse HEAD~1)"; Q="$(git -C "$T/r" rev-parse HEAD)"
OUT="$(cd "$ROOT" && bash "$TIER" --event merge_group --repo-root "$T/r" --pr-head "$P" --pr-head-conclusion success 2>&1)"
want "D9 merge_group same tree + PR head success -> reuse citing the PR head (unchanged)" "cite=$P" "$(grep '^cite=' <<<"$OUT")"
OUT="$(cd "$ROOT" && bash "$TIER" --event merge_group --repo-root "$T/r" --pr-head "$P" --pr-head-conclusion failure --mg-run 7001 --mg-sha "$Q" --mg-conclusion success 2>&1)"
hasnt "D10 merge_group ignores --mg-*: a PR-head failure never reuses" "tier=reuse" "$OUT"
printf 'docs/audits/a.md\n' >"$T/d.txt"
OUT="$(cd "$ROOT" && bash "$TIER" --event pull_request --repo-root "$T/r" --diff-from "$T/d.txt" --mg-run 7001 --mg-sha "$Q" --mg-conclusion success 2>&1)"
hasnt "D11 pull_request ignores --mg-*: never reuse" "tier=reuse" "$OUT"

# -- W: the CI wiring ------------------------------------------------------------------------------
SEC="$ROOT/ci/sections.yml"
want "W1 the tier step's push arm calls the lookup for GITHUB_SHA" 1 \
     "$(grep -cF 'bash scripts/ci_mg_workspace_result.sh "${GITHUB_REPOSITORY}" "${GITHUB_SHA}"' "$SEC")"
want "W2 a failed lookup is not fatal to the step (it falls to today's tier)" 1 \
     "$(grep -cF '"${GITHUB_SHA}") || mg=""' "$SEC")"
W3PAT="push) printf 'cited merge_group run: %s"
if grep -qF "$W3PAT" "$SEC"; then w3=1; else w3=0; fi
want "W3 the reuse step labels a push citation as a merge_group run, not a PR head" 1 "$w3"

# -- A: the aggregator still refuses a split tier --------------------------------------------------
CI="$ROOT/.github/workflows/ci.yml"
agg="$(awk '/set -- \$tiers/{p=1} p{print} /is not owed/{exit}' "$CI" | sed 's/^ *//')"
has "A0 the aggregator's tier rule was found in ci.yml" 'the shards disagree' "$agg"
# run the extract as its own script (no eval): its `exit` ends that script, and $tiers is its env
printf '%s\n' "$agg" >"$T/agg.sh"
agg_run() { OUT="$(tiers=" $*" bash "$T/agg.sh" 2>&1)"; RC=$?; }
agg_run reuse reuse quick; want "A1 a split tier (reuse reuse quick) -> RED" 1 "$RC"
agg_run reuse full reuse;  want "A2 a split tier (reuse full reuse) -> RED" 1 "$RC"
agg_run reuse reuse reuse; want "A3 all three shards reuse -> green, Σ not owed" 0 "$RC"

if [ "$fails" -eq 0 ]; then
  echo "check_ci_push_reuse: OK - push reuses only a merge-queue success on its own sha; every doubt is today's tier"
  exit 0
fi
echo "check_ci_push_reuse: $fails assertion(s) FAILED"
exit 1
