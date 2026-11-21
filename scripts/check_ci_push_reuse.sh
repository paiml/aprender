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
#   T rows  a tag push (#5172) is full: never reused, never refined
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

case "${1:-}" in -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

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
      grep -v "# $m\$" "$subj" >"$work/m$total.sh"
      printf '%s\t%s\n' "$subj" "$m" >"$work/m$total.what"
    done
  done
  # #4919: the 29 mutant tables ran one after another, 1157-2330 s of the guard-tree budget. They
  # are independent (each its own mutated copy and its own mktemp table), so they run
  # PUSH_REUSE_JOBS at a time, and a mutant table stops at its first failed row
  # (PUSH_REUSE_FAIL_FAST): a kill needs only rc != 0, and a SURVIVOR still runs every row.
  # Verdicts are printed in marker order after all have finished. No case is dropped.
  JOBS="${PUSH_REUSE_JOBS:-4}"
  case "$JOBS" in ''|*[!0-9]*|0) echo "check_ci_push_reuse: PUSH_REUSE_JOBS='$JOBS' is not a positive count"; rm -rf "${work:?}"; exit 1 ;; esac
  i=0
  while [ "$i" -lt "$total" ]; do
    i=$((i+1))
    IFS=$'\t' read -r subj m <"$work/m$i.what"
    # A mutation that changes nothing is an error, never a survivor and never a kill.
    if cmp -s "$subj" "$work/m$i.sh"; then echo nochange >"$work/m$i.rc"; continue; fi
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
    ( export PUSH_REUSE_FAIL_FAST=1; run_as "$subj" "$work/m$i.sh" >"$work/m$i.rc" ) &
  done
  wait
  i=0
  while [ "$i" -lt "$total" ]; do
    i=$((i+1))
    IFS=$'\t' read -r subj m <"$work/m$i.what"
    rc="$(cat "$work/m$i.rc" 2>/dev/null)"
    case "$rc" in
      nochange) echo "  ERROR    ${subj##*/} $m (the mutation changed nothing)" ;;
      124) echo "  ERROR    ${subj##*/} $m (timed out after ${RUN_TIMEOUT}s)" ;;
      0) echo "  SURVIVED ${subj##*/} $m" ;;
      ''|*[!0-9]*) echo "  ERROR    ${subj##*/} $m (no verdict was written)" ;;
      *) killed=$((killed+1)); echo "  killed   ${subj##*/} $m" ;;
    esac
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
bad()   { fails=$((fails+1)); printf '  FAIL  %s\n     expected: %s\n     actual:   %s\n' "$1" "$2" "$3"; [ "${PUSH_REUSE_FAIL_FAST:-0}" != 1 ] || exit 1; }
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
sjobs() { # sjobs <n>... -> a jobs body whose workspace-test-shard (n, 4) jobs all succeeded (the queue run really tested)
  local n; { for n in "$@"; do printf '{"name":"workspace-test-shard (%s, 4)","conclusion":"success"}\n' "$n"; done; printf '{"name":"gate","conclusion":"success"}\n'; } | jq -s '{jobs: .}'
}
canned() { # canned <suites-json-list> <runs-json-list>
  rm -f "${T:?}/canned/suites.FAIL" "${T:?}/canned/runs.FAIL" "${T:?}/canned/mgjobs.FAIL" "${T:?}/canned/calls"
  printf '{"check_suites":[%s]}' "$1" >"$T/canned/suites.json"
  printf '{"check_runs":[%s]}' "$2" >"$T/canned/runs.json"
  sjobs 1 2 3 4 >"$T/canned/mgjobs.json"
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
want "L1 costs exactly 3 API calls (suites, check-runs, the run's jobs)" 3 "$(wc -l <"$T/canned/calls" | tr -d ' ')"

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

# L14-L18: a queue run whose green `workspace-test` did not run its own shards is a citation, not a measurement (finding: transitive reuse)
canned "$(suite 50 "$QUEUE")" "$(crun completed '"success"' $SHA 2026-10-04T10:00:00Z 50 7015)"
sjobs 1 2 | jq '.jobs += [{"name":"workspace-test-shard (3, 4)","conclusion":"skipped"},{"name":"workspace-test-shard (4, 4)","conclusion":"skipped"}]' >"$T/canned/mgjobs.json"
look
want "L14 the queue run skipped its shards (it cited a PR run) -> nothing, exit 1" "1:" "$RC:$OUT"
has "L14 says why" "did not run and pass its own workspace-test shards" "$ERR"

canned "$(suite 50 "$QUEUE")" "$(crun completed '"success"' $SHA 2026-10-04T10:00:00Z 50 7016)"
sjobs 1 2 3 | jq '.jobs += [{"name":"workspace-test-shard (4, 4)","conclusion":"failure"}]' >"$T/canned/mgjobs.json"
look
want "L15 one shard failed under a green verdict -> nothing, exit 1" "1:" "$RC:$OUT"

canned "$(suite 50 "$QUEUE")" "$(crun completed '"success"' $SHA 2026-10-04T10:00:00Z 50 7017)"
sjobs >"$T/canned/mgjobs.json"
look
want "L16 the queue run has no shard job at all -> nothing, exit 1" "1:" "$RC:$OUT"

canned "$(suite 50 "$QUEUE")" "$(crun completed '"success"' $SHA 2026-10-04T10:00:00Z 50 7018)"
: >"$T/canned/mgjobs.FAIL"
look
want "L17 the jobs call fails -> nothing, exit 1" "1:" "$RC:$OUT"
has "L17 names the failed call" "jobs lookup failed" "$ERR"

canned "$(suite 50 "$QUEUE")" "$(crun completed '"failure"' $SHA 2026-10-04T10:00:00Z 50 7019)"
sjobs >"$T/canned/mgjobs.json"
look
want "L18 a failed verdict is still printed without a jobs call (only a success is checked for citation)" \
     "--mg-run 7019 --mg-sha $SHA --mg-conclusion failure" "$(flat "$OUT")"
want "L18 and it cost 2 API calls" 2 "$(wc -l <"$T/canned/calls" | tr -d ' ')"

# -- D: the decision -------------------------------------------------------------------------------
mkrepo() { # mkrepo <dir> -> a two-commit repo whose push diff is docs-only (today's tier: none, no cargo)
  git init -q -b main "$1"
  git -C "$1" -c user.name=t -c user.email=t@t commit -q --allow-empty -m base
  mkdir -p "$1/docs/roadmaps"; printf 'x: 1\n' >"$1/docs/roadmaps/x.yaml"
  git -C "$1" add docs; git -C "$1" -c user.name=t -c user.email=t@t commit -q -m docs
}
mkrepo "$T/r"; H="$(git -C "$T/r" rev-parse HEAD)"
tier() { (cd "$ROOT" && env -u GITHUB_SHA -u GITHUB_REF PUSH_BEFORE="$(git -C "$T/r" rev-parse HEAD^1)" ${GSHA:+GITHUB_SHA=$GSHA} ${GREF:+GITHUB_REF=$GREF} bash "$TIER" --event push --repo-root "$T/r" "$@" 2>&1); }
BASE="$(tier)"
want "D0 push with no merge_group result -> today's tier (docs-only: none)" "tier=none" "$(grep '^tier=' <<<"$BASE")"
OUT="$(pushd "$ROOT" >/dev/null && env -u GITHUB_SHA -u GITHUB_REF PUSH_BEFORE="$H" bash "$TIER" --event push --repo-root "$T/r" 2>&1)"
want "D0e push whose before == HEAD (empty diff) -> full, never none" "tier=full" "$(grep '^tier=' <<<"$OUT")"
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

# #5172: a tag push (the release tag) runs the whole workspace: never reused, never refined.
# The v0.71.0 tag push reused the queue result on its commit and ran no workspace test.
OUT="$(GREF=refs/tags/v9.9.9 GSHA=$H tier --mg-run 7001 --mg-sha "$H" --mg-conclusion success)"
want "T1 a tag push with a merge_group success on its commit -> full, never reuse" "tier=full" "$(grep '^tier=' <<<"$OUT")"
has "T1 the reason names the tag" "push of tag v9.9.9" "$OUT"
OUT="$(GREF=refs/tags/v9.9.9 tier)"
want "T2 a tag push with no merge_group result -> full, never the docs-only diff tier" "tier=full" "$(grep '^tier=' <<<"$OUT")"
OUT="$(GREF=refs/heads/main GSHA=$H tier --mg-run 7001 --mg-sha "$H" --mg-conclusion success)"
want "T3 a branch push (refs/heads/main) still reuses" "tier=reuse" "$(grep '^tier=' <<<"$OUT")"

# a root commit: today's tier is full (no diff can be derived); a refused result keeps it full
git init -q -b main "$T/root"; git -C "$T/root" -c user.name=t -c user.email=t@t commit -q --allow-empty -m only
OUT="$(pushd "$ROOT" >/dev/null && env -u GITHUB_SHA PUSH_BEFORE=0000000000000000000000000000000000000000 bash "$TIER" --event push --repo-root "$T/root" --mg-run 7001 --mg-sha "$H" --mg-conclusion success 2>&1)"
want "D8 a refused result on a root commit keeps today's full" "tier=full" "$(grep '^tier=' <<<"$OUT")"

# merge_group and pull_request never read --mg-*; merge_group's own reuse is unchanged
git -C "$T/r" -c user.name=t -c user.email=t@t commit -q --allow-empty -m "queue, same tree"
P="$(git -C "$T/r" rev-parse HEAD~1)"; Q="$(git -C "$T/r" rev-parse HEAD)"
OUT="$(pushd "$ROOT" >/dev/null && bash "$TIER" --event merge_group --repo-root "$T/r" --pr-head "$P" --pr-head-conclusion success 2>&1)"
want "D9 merge_group + PR head success, SAME tree (empty queue diff) -> full: the PR green is never reused (RED against the pre-PR script, which answered reuse)" "tier=full" "$(grep '^tier=' <<<"$OUT")"
OUT="$(pushd "$ROOT" >/dev/null && bash "$TIER" --event merge_group --repo-root "$T/r" --pr-head "$P" --pr-head-conclusion success --mg-run 7001 --mg-sha "$Q" --mg-conclusion success 2>&1)"
want "D10 merge_group + PR head success + a green --mg-* run, SAME tree -> full: --mg-* is ignored (RED against the pre-PR script)" "tier=full" "$(grep '^tier=' <<<"$OUT")"
printf "[workspace]\n" > "$T/r/Cargo.toml"; git -C "$T/r" add Cargo.toml; git -C "$T/r" -c user.name=t -c user.email=t@t commit -q -m "queue, root manifest (non-empty diff)"
P="$(git -C "$T/r" rev-parse HEAD~1)"; Q="$(git -C "$T/r" rev-parse HEAD)"
OUT="$(pushd "$ROOT" >/dev/null && bash "$TIER" --event merge_group --repo-root "$T/r" --pr-head "$P" --pr-head-conclusion success 2>&1)"
want "D9b merge_group + PR head success, NON-EMPTY diff (root manifest) -> full" "tier=full" "$(grep '^tier=' <<<"$OUT")"
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
agg_run() { OUT="$(SHARDS=$# tiers=" $*" bash "$T/agg.sh" 2>&1)"; RC=$?; }  # SHARDS = the tiers given, so a row tests the DISAGREEMENT rule, not the shard count
agg_run reuse reuse quick; want "A1 a split tier (reuse reuse quick) -> RED" 1 "$RC"
agg_run reuse full reuse;  want "A2 a split tier (reuse full reuse) -> RED" 1 "$RC"
agg_run reuse reuse reuse; want "A3 all three shards reuse -> green, Σ not owed" 0 "$RC"
SHARDS=3 tiers=" reuse reuse" bash "$T/agg.sh" >/dev/null 2>&1; want "A4 fewer tier files than shards -> RED (a shard that never reported)" 1 "$?"

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
REAL=$'mg-reuse\tsuccess\nx86-main\tsuccess\nx86-main-advisories\tskipped\ndeterminism\tsuccess\nmac-check\tsuccess\ngate\tsuccess\nci / gate\tsuccess'
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
for j in x86-main determinism gate 'ci / gate'; do
  green; real "$j" failure | jobsf 9001 "$PS0" 0; pd; no "P14 the queue run's $j failed" "$j is failure"
done
green; real x86-main-advisories success | jobsf 9001 "$PS0" 0; pd
no "P15 x86-main-advisories ran: the queue run reused rather than ran" "reused rather than ran"
green; printf '%s\n' "$REAL" | sed 's/^\(x86-main\|determinism\)\tsuccess$/\1\tskipped/; s/^x86-main-advisories\tskipped$/x86-main-advisories\tsuccess/' | jobsf 9001 "$PS0" 0; pd
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
# X1 is scoped to the job: mg-reuse's own 4-space `if:` key, never a string found anywhere in the file.
mgif() { awk '/^  [A-Za-z0-9_-]+:$/ {j = $1} j == "mg-reuse:" && /^    if:/ {sub(/^    /, ""); print; exit}' "$1"; }
X1WANT="if: github.event_name == 'push' && github.ref == 'refs/heads/main'"
want "X1 mg-reuse's own job-level if is push-to-main only (S1: skipped on merge_group, x86-main asks for its runner at t=0)" "$X1WANT" "$(mgif "$CI")"
# RED mutants (each must differ from the real answer AND carry the string, so the control is live):
# the string as a real `if:` of ANOTHER job (what an unanchored grep would count), and the old form.
X1ESC="${X1WANT//&/\\&}"
sed "/^    if: github.event_name == 'push' && github.ref == 'refs\/heads\/main'/d" "$CI" | sed "s|^  x86-main:\$|  x86-main:\n    $X1ESC|" > "$T/x1a.yml"
want "X1m1 control: the mutant really carries the string" 1 "$(grep -cF "$X1WANT" "$T/x1a.yml")"
want "X1m1 the string as another job's if: does not satisfy X1" 1 "$([ "$(mgif "$T/x1a.yml")" != "$X1WANT" ] && echo 1 || echo 0)"
sed "s/^    if: github.event_name == 'push' && github.ref == 'refs\/heads\/main'/    if: github.event_name == 'merge_group' || (github.event_name == 'push' \&\& github.ref == 'refs\/heads\/main')/" "$CI" > "$T/x1b.yml"
want "X1m2 the old merge_group form does not satisfy X1" 1 "$([ "$(mgif "$T/x1b.yml")" != "$X1WANT" ] && echo 1 || echo 0)"
want "X2 the push calls resolve-push with the pushed sha and before" 1 \
     "$(grep -cF 'bash scripts/ci_mg_reuse.sh resolve-push --event "$EVT" --ref "$GITHUB_REF" --sha "$GITHUB_SHA" --before "$PBEFORE"' "$CI")"
want "X3 both verdict copies (ci / gate and gate) accept a push reuse only, never a merge_group one" 2 \
     "$(grep -cE '^ +push:success:1:1:success\) REUSE=1' "$CI")"
want "X3b no verdict copy still names merge_group:success:1:1" 0 "$(grep -cF 'merge_group:success:1:1' "$CI")"
bash "$ROOT/scripts/ci_mg_reuse.sh" --self-test >"$T/mgst" 2>&1; rc=$?
want "X4 ci_mg_reuse.sh --self-test (verdict rows incl. push, job wiring) is green" 0 "$rc"

# -- Y: the PR-run arm (#7298; its decision table is scripts/ci_push_reuse.sh --self-test). Structure only, on the surfaces that table cannot see.
FACTS="$ROOT/scripts/ci_push_reuse_facts.sh"
want "Y1 the facts reader never derives a fact from workflow text (no tier_reuse, no grep of ci.yml for receipt-judge)" 0 \
     "$(grep -cE 'tier_reuse|grep .*receipt-judge' "$FACTS")"
want "Y2 C6 reads the run's creation (created_at, which a re-run never resets), never run_started_at (a re-run resets it) or updated_at" "1:0:0" "$(grep -c 'created_epoch: (.created_at | fromdateiso8601)' "$FACTS"):$(grep -c 'run_started_at | fromdateiso8601' "$FACTS"):$(grep -c 'updated_at' "$FACTS")"
want "Y3 the facts carry the PR own base_ref and head_ref, the run head_branch and sov_test; never run.pull_requests (empty after the merge)" "1:1:1:1:0" "$(grep -c "base_ref: " "$FACTS"):$(grep -c "head_ref: " "$FACTS"):$(grep -c "head_branch: " "$FACTS"):$(grep -c "sov_test" "$FACTS" | sed "s/^[2-9].*/1/;s/^1[0-9].*/1/"):$(grep -c "pull_requests" "$FACTS")"
want "Y4 the stamp log is read whole, a failed read is a refusal, and one stamp line is required" 3 \
     "$(grep -cE 'stamp job log read failed|has \$nstamps stamp lines|stamp and an unavailable line' "$FACTS")"
want "Y5 the facts reader pipes nothing into grep -q" 0 "$(grep -cE '\| *grep -[a-zA-Z]*q' "$FACTS")"
# x86=1 / det=1 / wt=1 are written once by the PR-run arm, directly inside the exact-`reuse=yes` branch
want "Y6 the PR-run arm writes x86=1 once" 1 "$(grep -cF '{ echo "x86=1"; echo "det=1"; echo "cite=$cite"; echo "wt=1"; echo "wt_cite=$cite"; }' "$CI")"
want "Y7 that write sits directly under the exact reuse=yes test" 1 \
     "$(grep -B2 -F '{ echo "x86=1"; echo "det=1"; echo "cite=$cite"; echo "wt=1"; echo "wt_cite=$cite"; }' "$CI" | grep -cF 'if [ "$pr" = "reuse=yes" ]; then')"
want "Y8 a failed facts read leaves the outputs alone (reuse=no, never yes)" 1 "$(grep -cF 'pr="reuse=no reason=the facts could not be read"' "$CI")"
want "Y9 workspace-test-ran still records the stamp, once" 1 "$(grep -c 'echo "PUSH-REUSE-STAMP tested_sha=' "$CI")"
want "Y10 workspace-test accepts wt=1 only from a job that succeeded" 1 "$(grep -cF '[ "${WT_SRC_RESULT:-}" = success ]' "$CI")"
FX="$ROOT/scripts/fixtures/ci_push_reuse"
want "Y11 a real-shape stamp job log (BOM, CRLF, colour-prefixed echoed script, one emitted stamp) yields the stamp" 1 \
     "$(bash "$FACTS" --parse-stamp-log "$FX/stamp-job-with-stamp.fixture" | jq -r 'select(.tree != null and (.sha|length)==40) | 1')"
want "Y12 a log that only echoes the unavailable text and emits it yields no stamp (null), never a stamp" null \
     "$(bash "$FACTS" --parse-stamp-log "$FX/stamp-job-unavailable.fixture")"
want "Y13 a log with no stamp line yields null" null "$(bash "$FACTS" --parse-stamp-log "$FX/stamp-job-no-stamp.fixture")"
want "Y14 a PR run of x86-main that skipped sov.test reads skipped" '"skipped"' "$(bash "$FACTS" --parse-section-log "$FX/x86-main-pr-sections.fixture" sov.test)"
want "Y15 a run that ran sov.test reads success" '"success"' "$(bash "$FACTS" --parse-section-log "$FX/x86-main-push-sections.fixture" sov.test)"
want "Y16 a section the log does not carry reads null" null "$(bash "$FACTS" --parse-section-log "$FX/x86-main-push-sections.fixture" no.such.section)"
# Y17-Y19 (#7314): the SEC010 refusal of a ".." path is exercised (deleting it turns a row red), and a flag with no value exits 2, never loops.
OUT=$(bash "$FACTS" --parse-stamp-log "$FX/../ci_push_reuse/stamp-job-no-stamp.fixture" 2>/dev/null); RC=$?
want "Y17 --parse-stamp-log refuses a path with .. (exit 1, nothing printed, the readable file is not parsed)" "1:" "$RC:$OUT"
OUT=$(bash "$FACTS" --parse-section-log "$FX/../ci_push_reuse/x86-main-push-sections.fixture" sov.test 2>/dev/null); RC=$?
want "Y18 --parse-section-log refuses a path with .. (exit 1, nothing printed)" "1:" "$RC:$OUT"
for fl in --repo --sha --before --ref --event --now; do
  RC=0; timeout 10 bash "$FACTS" "$fl" >/dev/null 2>&1 || RC=$?
  want "Y19 $fl with no value exits 2 (124 would be the old endless loop)" 2 "$RC"
  RC=0; timeout 10 bash "$FACTS" --repo a/b --sha 0 --before 0 --ref r --event e "$fl" >/dev/null 2>&1 || RC=$?
  want "Y19 $fl as the LAST flag with no value exits 2" 2 "$RC"
done

# Y20 (#7314 lane 1 on cb90e8c): a ROOT commit (no parent) dies at "no parent" with zero gh calls; without `cut -s` the sha itself read as its parent.
RG="$T/rootrepo"; mkdir -p "$RG/bin"; git init -q -b main "$RG/r" && git -C "$RG/r" -c user.name=t -c user.email=t@t commit -q --allow-empty -m root
printf '#!/bin/sh\necho called >> "%s/gh.calls"\nexit 1\n' "$RG" > "$RG/bin/gh"; chmod +x "$RG/bin/gh"
RSHA=$(git -C "$RG/r" rev-parse HEAD)
OUT=$(pushd "$RG/r" >/dev/null && PATH="$RG/bin:$PATH" bash "$FACTS" --repo o/r --sha "$RSHA" --before 0000000000000000000000000000000000000000 --ref refs/heads/main --event push --now 1000 2>&1); RC=$?
want "Y20 a root commit is refused as 'no parent' before any gh call" "1:1:0" "$RC:$(grep -c 'no parent' <<< "$OUT"):$(wc -l < "$RG/gh.calls" 2> /dev/null || echo 0)"

# -- Z: round 3 of #7314 (fail-closed facts, fork guard on every PR-run self-hosted job, the --now wiring)
# Z1-Z8: the facts reader's draft and freshness derivation, on saved API objects (no network).
ZP="$T/zpr.json"
zpr() { printf '%s' "$1" > "$ZP"; bash "$FACTS" --derive-pr-facts o/r "$ZP" 2>&1; }
ZBASE='"number":7,"merged":true,"merge_commit_sha":"a","base":{"ref":"main"},"head":{"ref":"f","sha":"b","repo":{"full_name":"o/r"}}'
want "Z1 draft false reads false" "false" "$(zpr "{$ZBASE,\"draft\":false}" | jq -c '.draft')"
want "Z2 draft true reads true" "true" "$(zpr "{$ZBASE,\"draft\":true}" | jq -c '.draft')"
want "Z3 a missing draft reads null (the decision refuses), never false" "null" "$(zpr "{$ZBASE}" | jq -c '.draft')"
want "Z4 a null draft reads null" "null" "$(zpr "{$ZBASE,\"draft\":null}" | jq -c '.draft')"
want "Z5 the string \"true\" reads null" "null" "$(zpr "{$ZBASE,\"draft\":\"true\"}" | jq -c '.draft')"
want "Z5b the number 0 reads null" "null" "$(zpr "{$ZBASE,\"draft\":0}" | jq -c '.draft')"
ZR="$T/zruns.json"
zrun() { printf '{"workflow_runs":[{"id":5,"event":"pull_request","path":"p","head_sha":"h","head_branch":"b","status":"completed","conclusion":"success","run_attempt":%s,"created_at":"2026-11-18T10:00:00Z","run_started_at":"%s"}]}' "$1" "$2" > "$ZR"
  bash "$FACTS" --derive-run-facts "$ZR" 5 5 2>&1; }
want "Z6 the run's freshness is its created_at (2026-11-18T10:00:00Z = 1794996000)" "1794996000" "$(zrun 1 2026-11-18T10:00:00Z | jq -r '.created_epoch')"
want "Z7 a re-run (attempt 3) with a run_started_at of a day later does NOT move the freshness time" "1794996000" "$(zrun 3 2026-11-19T09:00:00Z | jq -r '.created_epoch')"
want "Z8 the run facts carry no started_epoch" "null" "$(zrun 3 2026-11-19T09:00:00Z | jq -r '.started_epoch')"
printf '{"workflow_runs":[{"id":5,"run_attempt":1,"run_started_at":"2026-11-18T10:00:00Z"}]}' > "$ZR"
RC=0; bash "$FACTS" --derive-run-facts "$ZR" 5 5 > /dev/null 2>&1 || RC=$?
want "Z9 a run with no created_at is unreadable (non-zero), never a fresh run" "1" "$([ "$RC" -ne 0 ] && echo 1 || echo 0)"
# Z10-Z14: every self-hosted job that a pull_request now runs carries the fork guard, and removing it from any one is caught.
FORKG="(github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository)"
FORKW="github.event.pull_request.head.repo.full_name == github.repository"
jobif() { awk -v J="$2:" '/^  [A-Za-z0-9_-]+:$/ {j = $1} j == J && /^    if:/ {print; exit}' "$1"; }
forkok() { # forkok FILE JOB: 1 when the job's own if: carries the fork guard
  case "$(jobif "$1" "$2")" in *"$FORKW"*) echo 1 ;; *) echo 0 ;; esac; }
for j in x86-main x86-main-advisories determinism workspace-test-shard workspace-test-ran workspace-test-sigma; do
  want "Z10 $j carries the fork guard in its own if:" 1 "$(forkok "$CI" "$j")"
  sed "/^  $j:\$/,/^    runs-on:/s/ \&\& (github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository)//; /^  $j:\$/,/^    runs-on:/s/ \&\& github.event.pull_request.head.repo.full_name == github.repository//" "$CI" > "$T/zf-$j.yml"
  want "Z11 mutant: $j without the fork guard is caught" 0 "$(forkok "$T/zf-$j.yml" "$j")"
done
want "Z12 control: runner-smells keeps the same guard form (the one the others copy)" 1 "$(forkok "$CI" runner-smells)"
# a fork PR still fails closed: the verdict jobs read the skipped heavy jobs as red (workspace-test: shard result skipped; ci / gate: no sov.gate result)
want "Z13 workspace-test refuses a non-success shard result (a fork PR skips the shards)" 1 "$(grep -cF '[ "$SHARD_RESULT" = success ] || { echo "::error::workspace-test: shard matrix result' "$CI")"
# #7314: a fork PR skips workspace-test-sigma, and workspace-test refuses anything but its success
want "Z13b workspace-test refuses a non-success workspace-test-sigma result (a fork PR skips it)" 1 "$(grep -cF '[ "${SIGMA:-}" = success ] || { echo "::error::workspace-test: workspace-test-sigma is' "$CI")"
want "Z14 ci / gate reads a missing sov.gate result as red" 1 "$(grep -cF 'ci / gate: section result is' "$CI")"
# Z15: --now is passed to the facts reader (deleting it must turn this red; the reader itself refuses without it)
NOWARG='--now "$(date +%s)"'
want "Z15 ci.yml passes $NOWARG to ci_push_reuse_facts.sh" 1 "$(grep -cF "ci_push_reuse_facts.sh --repo \"\$REPO\" --sha \"\$GITHUB_SHA\" --before \"\$PBEFORE\" --ref \"\$GITHUB_REF\" --event \"\$EVT\" $NOWARG" "$CI")"
sed 's| --now "$(date +%s)"||' "$CI" > "$T/znow.yml"
want "Z16 mutant: ci.yml without --now is caught" 0 "$(grep -cF "ci_push_reuse_facts.sh --repo \"\$REPO\" --sha \"\$GITHUB_SHA\" --before \"\$PBEFORE\" --ref \"\$GITHUB_REF\" --event \"\$EVT\" $NOWARG" "$T/znow.yml")"
RC=0; bash "$FACTS" --repo a/b --sha 0000000000000000000000000000000000000000 --before 0 --ref r --event push > /dev/null 2>&1 || RC=$?
ERR=$(bash "$FACTS" --repo a/b --sha 0000000000000000000000000000000000000000 --before 0 --ref r --event push 2>&1 >/dev/null); RC=$?
case "$ERR" in *"--now (epoch seconds"*) NOWMSG=named ;; *) NOWMSG="other: $ERR" ;; esac
want "Z17 the facts reader without --now refuses (exit 1) BECAUSE of --now, not a later check (deleting the --now line turns this red)" "1:named" "$RC:$NOWMSG"

# Y40-Y45 (#7314 lane 2): the stamp / section uniqueness rules RUN on logs with two lines (the one-line fixtures cannot see a relaxed count).
STAMP_LINE=$(grep -E '^[0-9TZ:.-]+ PUSH-REUSE-STAMP tested_sha=' "$FX/stamp-job-with-stamp.fixture" | tr -d '\r')
UNAV_LINE=$(grep -E '^[0-9TZ:.-]+ PUSH-REUSE-STAMP unavailable' "$FX/stamp-job-unavailable.fixture" | tr -d '\r')
printf '%s\n%s\n' "$STAMP_LINE" "$STAMP_LINE" > "$T/two-stamps.log"
printf '%s\n%s\n%s\n' "$STAMP_LINE" "$STAMP_LINE" "$STAMP_LINE" > "$T/three-stamps.log"
printf '%s\n%s\n' "$STAMP_LINE" "$UNAV_LINE" > "$T/stamp-and-unavailable.log"
printf '%s\n' "$STAMP_LINE" > "$T/one-stamp.log"
OUT=$(bash "$FACTS" --parse-stamp-log "$T/one-stamp.log" 2>&1); RC=$?
want "Y40 control: a log with exactly one stamp line parses (exit 0, one JSON object)" "0:1" "$RC:$(grep -c '"tree"' <<< "$OUT")"
OUT=$(bash "$FACTS" --parse-stamp-log "$T/two-stamps.log" 2>&1); RC=$?
want "Y41 two stamp lines are a refusal (exit 1, named, no JSON) - a count relaxed to -le 2 turns this red" "1:named:0" "$RC:$(case "$OUT" in *"has 2 stamp lines"*) echo named;; *) echo "other";; esac):$(grep -c '"tree"' <<< "$OUT")"
OUT=$(bash "$FACTS" --parse-stamp-log "$T/three-stamps.log" 2>&1); RC=$?
want "Y42 three stamp lines are a refusal too" "1" "$RC"
OUT=$(bash "$FACTS" --parse-stamp-log "$T/stamp-and-unavailable.log" 2>&1); RC=$?
want "Y43 a stamp AND an emitted unavailable line is a refusal (exit 1, named, no JSON)" "1:named:0" "$RC:$(case "$OUT" in *"a stamp and an unavailable line"*) echo named;; *) echo "other";; esac):$(grep -c '"tree"' <<< "$OUT")"
SEC_LINE=$(grep -E '^[0-9TZ:.-]+ section sov\.test: ' "$FX/x86-main-push-sections.fixture" | tr -d '\r')
printf '%s\n%s\n' "$SEC_LINE" "$SEC_LINE" > "$T/two-sections.log"
want "Y44 a section line that appears twice reads null, never a result (exactly-one rule)" null "$(bash "$FACTS" --parse-section-log "$T/two-sections.log" sov.test)"
printf '%s\n' "$SEC_LINE" > "$T/one-section.log"
want "Y45 control: the same section line once reads its result" '"success"' "$(bash "$FACTS" --parse-section-log "$T/one-section.log" sov.test)"

if [ "$fails" -eq 0 ]; then
  echo "check_ci_push_reuse: OK - a push reuses only a merge-queue success on its own sha, or the PR's own green run when scripts/ci_push_reuse.sh C1-C6 hold (workspace-test, x86-main, determinism: only a real run); every doubt runs the jobs"
  exit 0
fi
echo "check_ci_push_reuse: $fails assertion(s) FAILED"
exit 1
