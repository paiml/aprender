#!/usr/bin/env bash
# check_ci_push_reuse.sh - the case table for T36 and #4748: a push to main reuses the merge
# queue's result on the same commit, for workspace-test and for x86-main + determinism.
#
# The merge queue tests a commit, then fast-forwards main to that very commit; the push run on
# main used to re-test it (about 103 runner-min per merge, most of it the three workspace-test
# shards). Now the push run's tier step asks GitHub for the merge_group workspace-test result on
# its own sha (scripts/ci_mg_workspace_result.sh) and scripts/ci_test_tier.sh reuses it only
# when that result is a success for exactly this commit. #4748 does the same for x86-main and
# determinism (scripts/ci_mg_reuse.sh decide-push), and only when the queue run on that sha RAN
# them -- a queue run that itself reused them is never cited. Any doubt runs the job.
#
#   L rows  the lookup, against a stub `gh` serving canned JSON (no network)
#   D rows  the decision, scripts/ci_test_tier.sh --event push on throwaway repositories
#   W rows  the tier step in ci/sections.yml calls the lookup on push, and survives its failure
#   A rows  the workspace-test aggregator in ci.yml still refuses a split tier
#   P rows  decide-push for x86-main + determinism, on a throwaway repository and canned lists
#   R rows  resolve-push against the stub `gh`: 2 calls, and a failed call runs the jobs
#   X rows  the ci.yml wiring of the push arm
#
#   bash scripts/check_ci_push_reuse.sh              # the table
#   bash scripts/check_ci_push_reuse.sh --self-test  # delete each `# R-*` refusal line in a copy
#                                                    # of the subjects; each must turn the table RED
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOOKUP="${PUSH_REUSE_LOOKUP:-$ROOT/scripts/ci_mg_workspace_result.sh}"
TIER="${PUSH_REUSE_TIER:-$ROOT/scripts/ci_test_tier.sh}"
MG="${PUSH_REUSE_MG:-$ROOT/scripts/ci_mg_reuse.sh}"

case "${1:-}" in -h|--help) sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

if [ "${1:-}" = "--self-test" ]; then
  # A red table kills every mutant for free: the unmutated table must be green first.
  bash "$0" >/dev/null 2>&1 || { echo "check_ci_push_reuse: the unmutated table is red - mutants not_measured"; exit 1; }
  # The cap stops a hung table, nothing more: a table run took 33 s at load 7, and a
  # loaded runner once took a 120 s cap past it. A timeout (rc 124) is never a kill.
  RUN_TIMEOUT="${PUSH_REUSE_RUN_TIMEOUT:-600}"
  work="$(mktemp -d)"; killed=0; total=0
  # run_as SUBJECT COPY: the table with COPY standing in for SUBJECT; prints the table's rc
  run_as() {
    case "${1##*/}" in
      ci_mg_workspace_result.sh) PUSH_REUSE_LOOKUP="$2" timeout "$RUN_TIMEOUT" bash "$0" >/dev/null 2>&1 ;;
      ci_test_tier.sh) PUSH_REUSE_TIER="$2" timeout "$RUN_TIMEOUT" bash "$0" >/dev/null 2>&1 ;;
      ci_mg_reuse.sh) PUSH_REUSE_MG="$2" timeout "$RUN_TIMEOUT" bash "$0" >/dev/null 2>&1 ;;
      *) echo 9; return ;;
    esac
    echo $?
  }
  for subj in "$ROOT/scripts/ci_mg_workspace_result.sh" "$ROOT/scripts/ci_test_tier.sh" "$ROOT/scripts/ci_mg_reuse.sh"; do
    mapfile -t markers < <(grep -o '# R-[A-Z]*$' "$subj" | sed 's/^# //')
    [ "${#markers[@]}" -gt 0 ] || { echo "check_ci_push_reuse: no R-* markers in ${subj##*/} (vacuous)"; rm -rf "${work:?}"; exit 1; }
    # Control: an UNMUTATED copy in the work dir must keep the table green, or a copy that
    # breaks for its location alone would read as every mutant killed.
    cp "$subj" "$work/subject.sh"
    rc="$(run_as "$subj" "$work/subject.sh")"
    [ "$rc" -ne 124 ] || { echo "check_ci_push_reuse: the table timed out after ${RUN_TIMEOUT}s on an unmutated copy of ${subj##*/} - mutants not_measured"; rm -rf "${work:?}"; exit 1; }
    [ "$rc" -eq 0 ] || { echo "check_ci_push_reuse: an unmutated copy of ${subj##*/} turns the table red (rc $rc) - mutants not_measured"; rm -rf "${work:?}"; exit 1; }
    for m in "${markers[@]}"; do
      total=$((total+1))
      grep -v "# $m\$" "$subj" >"$work/subject.sh"
      # A mutation that changes nothing is an error, never a survivor and never a kill.
      if cmp -s "$subj" "$work/subject.sh"; then echo "  ERROR    ${subj##*/} $m (the mutation changed nothing)"; continue; fi
      rc="$(run_as "$subj" "$work/subject.sh")"
      if [ "$rc" -eq 124 ]; then echo "  ERROR    ${subj##*/} $m (timed out after ${RUN_TIMEOUT}s)"; continue; fi
      if [ "$rc" -eq 0 ]; then echo "  SURVIVED ${subj##*/} $m"; else killed=$((killed+1)); echo "  killed   ${subj##*/} $m"; fi
    done
  done
  rm -rf "${work:?}"
  echo "check_ci_push_reuse: mutants $killed/$total killed"
  [ "$killed" -eq "$total" ]; exit $?
fi

for f in "$LOOKUP" "$TIER" "$MG"; do [ -f "$f" ] || { echo "check_ci_push_reuse: missing $f"; exit 1; }; done

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
  */actions/runs/*/jobs*) k=mgjobs ;;
  */actions/runs\?*) k=mgruns ;;
  *) echo "stub gh: unexpected path $2" >&2; exit 9 ;;
esac
echo "$2" >>"$c/calls"
# the merge_group run and job lists (#4748) are served whole; a failing one still prints its body
case "$k" in mg*) [ -f "$c/$k.FAIL" ] && { cat "$c/$k.json"; echo "gh: HTTP 502" >&2; exit 1; }; cat "$c/$k.json"; exit 0 ;; esac
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

# -- P: x86-main + determinism on push (#4748), scripts/ci_mg_reuse.sh decide-push ----------------
# A repo with base A, parent B, pushed commit S, a side commit X and a merge M of B and X.
git init -q -b main "$T/pr"
gc() { git -C "$T/pr" -c user.name=t -c user.email=t@t -c commit.gpgsign=false "$@"; }
gc commit -q --allow-empty -m A; gc commit -q --allow-empty -m B; PB0="$(git -C "$T/pr" rev-parse HEAD)"
gc commit -q --allow-empty -m S; PS0="$(git -C "$T/pr" rev-parse HEAD)"
PA0="$(git -C "$T/pr" rev-parse HEAD~2)"
PX0="$(gc commit-tree -p "$PA0" -m X "$(git -C "$T/pr" rev-parse "$PA0^{tree}")")"
PM0="$(gc commit-tree -p "$PB0" -p "$PX0" -m M "$(git -C "$T/pr" rev-parse "$PB0^{tree}")")"
mkdir -p "$T/p"
mgrun() { # mgrun <id> <sha> <status> <branch-base> <created> [event]
  printf '{"id":%s,"path":".github/workflows/ci.yml","event":"%s","head_sha":"%s","status":"%s","run_attempt":1,"head_branch":"gh-readonly-queue/main/pr-4748-%s","created_at":"%s"}' \
    "$1" "${6:-merge_group}" "$2" "$3" "$4" "$5"
}
runsf() { printf '[%s]' "$2" | jq --argjson e "$1" '{total_count: (length + $e), workflow_runs: .}' >"$T/p/runs.json"; }
jobsf() { # jobsf <run-id> <sha> <extra-total> < "name<TAB>conclusion" lines
  jq -R -s --argjson rid "$1" --arg h "$2" --argjson e "$3" '
    [split("\n")[] | select(length > 0) | split("\t")
     | {name: .[0], conclusion: .[1], status: "completed", run_id: $rid, head_sha: $h}]
    | {total_count: (length + $e), jobs: .}' >"$T/p/jobs.json"
}
REAL=$'mg-reuse\tsuccess\nx86-main\tsuccess\nguards\tsuccess\nx86-main-advisories\tskipped\ndeterminism\tsuccess\nmac-check\tsuccess\ngate\tsuccess\nci / gate\tsuccess'
real() { printf '%s\n' "$REAL" | sed "s|^$1\t.*|$1\t$2|"; }
green() { runsf 0 "$(mgrun 9001 "${1:-$PS0}" completed "${2:-$PB0}" 2026-10-06T09:00:00Z)"; printf '%s\n' "$REAL" | jobsf 9001 "${1:-$PS0}" 0; }
pd() { # pd [event] [ref] [sha] [before]
  OUT="$(bash "$MG" decide-push --event "${1:-push}" --ref "${2:-refs/heads/main}" --sha "${3:-$PS0}" --before "${4:-$PB0}" \
    --runs "$T/p/runs.json" --jobs "$T/p/jobs.json" --repo-root "$T/pr" 2>&1)"; RC=$?
  FL="$(grep '^x86=' <<<"$OUT" | cut -d= -f2)$(grep '^det=' <<<"$OUT" | cut -d= -f2)$(grep '^mac=' <<<"$OUT" | cut -d= -f2)"
}
no() { want "$1" "0:000" "$RC:$FL"; has "$1 (the reason)" "$2" "$OUT"; }

green; pd
want "P1 the queue RAN x86-main + determinism on this commit and all passed -> reuse both, never mac" "0:110" "$RC:$FL"
has "P1 cites the merge_group run" "cite=merge_group run 9001 attempt 1 on ${PS0:0:10}" "$OUT"
green; pd pull_request;                    no "P2 a pull_request event never reuses" "P1:"
green; pd push refs/heads/release/0.70;    no "P3 a push to another branch never reuses" "P1:"
green; pd push refs/heads/main "$PS0" "$PA0"; no "P4 the push moved main from a commit that is not S's parent" "P2: the push moved main"
green "$PM0" "$PB0"; pd push refs/heads/main "$PM0" "$PB0"; no "P5 a two-parent (merge) commit never reuses" "single-parent"
green "$PS0" "$PA0"; pd;                   no "P6 the queue branch names another base" "does not name base"
green; pd push refs/heads/main 3333333333333333333333333333333333333333; no "P7 a commit absent from the clone" "P2:"
runsf 1 "$(mgrun 9001 "$PS0" completed "$PB0" 2026-10-06T09:00:00Z)"; printf '%s\n' "$REAL" | jobsf 9001 "$PS0" 0; pd
no "P8 the run list has an unread page" "unread page"
runsf 0 "$(mgrun 9001 "$PS0" completed "$PB0" 2026-10-06T09:00:00Z),$(mgrun 9002 "$PS0" in_progress "$PB0" 2026-10-06T09:30:00Z)"
printf '%s\n' "$REAL" | jobsf 9001 "$PS0" 0; pd
no "P9 the newest merge_group run on S is still running (an older real run does not count)" "'in_progress' (run 9002)"
runsf 0 "$(mgrun 9001 "$PS0" completed "$PB0" 2026-10-06T09:00:00Z pull_request)"; printf '%s\n' "$REAL" | jobsf 9001 "$PS0" 0; pd
no "P10 only a pull_request run on S (no merge_group run)" "'absent'"
green; printf '%s\n' "$REAL" | jobsf 9001 "$PS0" 1; pd; no "P11 the job list has an unread page" "unread page"
green; printf '%s\n' "$REAL" | jobsf 9002 "$PS0" 0; pd; no "P12 the jobs belong to another run" "not run 9001's"
green; printf '%s\n' "$REAL" | jobsf 9001 "$PB0" 0; pd; no "P13 the jobs ran on another commit" "not run 9001's"
for j in x86-main guards determinism gate 'ci / gate'; do
  green; real "$j" failure | jobsf 9001 "$PS0" 0; pd; no "P14 the queue run's $j failed" "$j is failure"
done
green; real x86-main-advisories success | jobsf 9001 "$PS0" 0; pd
no "P15 x86-main-advisories ran: the queue run reused rather than ran" "reused rather than ran"
green; printf '%s\n' "$REAL" | sed 's/^\(x86-main\|determinism\|guards\)\tsuccess$/\1\tskipped/; s/^x86-main-advisories\tskipped$/x86-main-advisories\tsuccess/' | jobsf 9001 "$PS0" 0; pd
no "P16 a queue run that itself reused x86-main + determinism (both skipped) is never cited" "x86-main is skipped"
green; printf '%s\n' "$REAL" | grep -v '^determinism' | jobsf 9001 "$PS0" 0; pd; no "P17 the queue run has no determinism job" "determinism is count=0"

# -- R: resolve-push, against the stub gh (2 calls); a failing call prints a body, so only its exit counts
mgcanned() { cp "$T/p/runs.json" "$T/canned/mgruns.json"; cp "$T/p/jobs.json" "$T/canned/mgjobs.json"; rm -f "${T:?}/canned/calls" "${T:?}/canned/mgruns.FAIL" "${T:?}/canned/mgjobs.FAIL"; }
rp() { OUT="$(bash "$MG" resolve-push --event push --ref refs/heads/main --sha "${1:-$PS0}" --before "$PB0" --repo o/r --repo-root "$T/pr" 2>&1)"; RC=$?
  FL="$(grep '^x86=' <<<"$OUT" | cut -d= -f2)$(grep '^det=' <<<"$OUT" | cut -d= -f2)$(grep '^mac=' <<<"$OUT" | cut -d= -f2)"
  CALLS="$( [ -f "$T/canned/calls" ] && wc -l <"$T/canned/calls" | tr -d ' ' || echo 0)"; }
green; mgcanned; rp
want "R1 resolve-push reads both lists and reuses on a real green queue run" "0:110" "$RC:$FL"
want "R1 costs exactly 2 API calls" 2 "$CALLS"
green; mgcanned; : >"$T/canned/mgruns.FAIL"; rp; no "R2 the runs call fails (body or not) -> run in full" "lookup failed: the merge_group runs"
green; mgcanned; : >"$T/canned/mgjobs.FAIL"; rp; no "R3 the jobs call fails (body or not) -> run in full" "lookup failed: the jobs of merge_group run 9001"
green; mgcanned; rp not-a-sha; no "R4 a sha that is not 40 hex" "not a 40-hex sha"
want "R4 and no API call was made" 0 "$CALLS"
OUT="$(bash "$MG" resolve-push --event pull_request --ref refs/heads/main --sha "$PS0" --before "$PB0" --repo o/r 2>&1)"
has "R5 a non-push event refuses without a call" "x86=0" "$OUT"

# -- X: the ci.yml wiring for the push arm (its verdict and job wiring also run in ci_mg_reuse.sh --self-test)
want "X1 mg-reuse runs on a push to main" 1 \
     "$(grep -cF "if: github.event_name == 'merge_group' || (github.event_name == 'push' && github.ref == 'refs/heads/main')" "$CI")"
want "X2 the push calls resolve-push with the pushed sha and before" 1 \
     "$(grep -cF 'bash scripts/ci_mg_reuse.sh resolve-push --event "$EVT" --ref "$GITHUB_REF" --sha "$GITHUB_SHA" --before "$PBEFORE"' "$CI")"
want "X3 both verdict copies (ci / gate and gate) accept a push reuse only with the advisories green" 2 \
     "$(grep -cF 'merge_group:success:1:1:success | push:success:1:1:success) REUSE=1' "$CI")"
bash "$ROOT/scripts/ci_mg_reuse.sh" --self-test >"$T/mgst" 2>&1; rc=$?
want "X4 ci_mg_reuse.sh --self-test (verdict rows incl. push, job wiring) is green" 0 "$rc"

if [ "$fails" -eq 0 ]; then
  echo "check_ci_push_reuse: OK - push reuses only a merge-queue success on its own sha (x86-main + determinism: only a real run); every doubt runs it"
  exit 0
fi
echo "check_ci_push_reuse: $fails assertion(s) FAILED"
exit 1
