#!/usr/bin/env bash
# ci_mg_reuse.sh -- may a merge_group run REUSE the PR head's green x86-main,
# determinism and mac-check instead of re-running them? (T43, Refs #4678)
#
# WHY. On five consecutive merges the merge_group commit's tree was byte-identical
# to the PR head's, and on four of them the head had already passed all three jobs:
# 131.5 of the 163.5 runner-min those jobs spent in the queue repeated a green on
# the same tree. The fifth (#4655) is why the key reads the head's JOB, never "the
# head was tested": its head x86-main FAILED and the merge_group run was its first
# green. That merge is the planted must-run row of the case table below.
#
# THE KEY (contracts/apr-required-checks-v1.yaml, equation t43_merge_group_reuse_key).
# A job is reused only when every one of these is READ, from git or the API:
#   K1  the queue head M has exactly one parent, and it is the queue base B
#   K2  B has exactly one parent: main moves only by squash commits
#   K3  B is an ancestor of the PR head H
#   K4  M^{tree} == H^{tree}: the queue tests exactly the tree the head tested
#   K5  the newest ci.yml pull_request run on H is completed; the run and job
#       lists are complete (no unread page), and every job is that run's, on H
#   K6  the same-named job succeeded there. x86-main and determinism are ONE unit:
#       determinism-compare waits on x86-main's X64 artifact and the gate reads
#       both jobs' section results, so they are reused together, and only when
#       the head run's `gate` and `ci / gate` succeeded too. mac-check stands alone.
# Why K1-K3 make the head run's origin/main comparand equal B: H contains B, so
# every run on H started after B was on main; B was main's tip when the queue
# entry was cut, and main only moves forward. So main's tip was B for the whole
# head run, and the ratchets that compare against origin/main compared against B.
# Anything not proven -- a failed or partial lookup included -- answers 0, i.e.
# run the job (L25). The RustSec advisory steps are never reused (their input
# changes with no commit): ci.yml's x86-main-advisories job runs them whenever
# x86-main is skipped.
#
#   ci_mg_reuse.sh decide  --event E --queue-head M --base B --pr-head H
#                          --runs FILE --jobs FILE [--repo-root DIR]
#   ci_mg_reuse.sh resolve --event E --ref QUEUE_REF --queue-head M --base B
#                          --repo OWNER/NAME [--repo-root DIR]   (3 gh api calls)
#   ci_mg_reuse.sh decide-push  --event E --ref REF --sha S --before P
#                          --runs FILE --jobs FILE [--repo-root DIR]
#   ci_mg_reuse.sh resolve-push --event E --ref REF --sha S --before P
#                          --repo OWNER/NAME [--repo-root DIR]   (2 gh api calls)
#   ci_mg_reuse.sh --self-test [--workflow FILE]
# decide/resolve print x86= det= mac= cite= pr_head= reason= and exit 0; 2 is usage.
# Each refusal line carries `# R:<id>`: the mutation harness drops it, and the
# case table must then go RED.
set -uo pipefail

CI_WORKFLOW_PATH=".github/workflows/ci.yml"
PRH=""

emit() { printf 'x86=%s\ndet=%s\nmac=%s\ncite=%s\npr_head=%s\nreason=%s\n' "$1" "$2" "$3" "$4" "$5" "$6"; }
refuse() { emit 0 0 0 "" "$PRH" "$1"; }
usage() { echo "usage: ci_mg_reuse.sh decide|resolve|decide-push|resolve-push [opts] | --self-test [--workflow FILE]" >&2; exit 2; }

# The newest ci.yml pull_request run on $2 in runs file $1: "id<TAB>status<TAB>attempt".
pick_run() {
  jq -r --arg h "$2" --arg p "$CI_WORKFLOW_PATH" '
    [.workflow_runs[]? | select(.path == $p and .event == "pull_request" and .head_sha == $h)]
    | sort_by([.created_at, .id]) | last // empty
    | "\(.id)\t\(.status)\t\(.run_attempt // 1)"' "$1" 2> /dev/null
}

# The conclusion of the ONE job named $2 in jobs file $1; any other count is not success.
concl() {
  jq -r --arg n "$2" '[.jobs[]? | select(.name == $n)]
    | if length == 1 then (.[0].conclusion // "none") else "count=\(length)" end' "$1" 2> /dev/null
}

decide() {
  local ev="" m="" b="" h="" runs="" jobs="" root="."
  while [ $# -gt 0 ]; do
    case "$1" in
      --event) ev="${2-}" ;;
      --queue-head) m="${2-}" ;;
      --base) b="${2-}" ;;
      --pr-head) h="${2-}" ;;
      --runs) runs="${2-}" ;;
      --jobs) jobs="${2-}" ;;
      --repo-root) root="${2-}" ;;
      *) echo "ci_mg_reuse: unknown argument '$1'" >&2; return 2 ;;
    esac
    shift 2 || return 2
  done
  PRH="$h"
  [ "$ev" = merge_group ] || { refuse "event '$ev' is not merge_group: nothing to reuse"; return 0; } # R:event
  local mp bp B tm th mpa bpa
  mp=$(git -C "$root" rev-list --parents -n 1 "$m" 2> /dev/null)
  read -r -a mpa <<< "$mp"
  B=$(git -C "$root" rev-parse --verify --quiet "$b^{commit}" 2> /dev/null)
  [ "${#mpa[@]}" -eq 2 ] || { refuse "K1: the queue head has $(( ${#mpa[@]} > 0 ? ${#mpa[@]} - 1 : 0 )) parents (or is not in this clone), not 1"; return 0; } # R:k1-one-parent
  [ -n "$B" ] && [ "${mpa[1]}" = "$B" ] || { refuse "K1: the queue head's parent ${mpa[1]:0:10} is not the base '${b:0:10}'"; return 0; } # R:k1-parent-is-base
  bp=$(git -C "$root" rev-list --parents -n 1 "$B" 2> /dev/null)
  read -r -a bpa <<< "$bp"
  [ "${#bpa[@]}" -eq 2 ] || { refuse "K2: the base ${B:0:10} is not a single-parent (squash) commit"; return 0; } # R:k2-base-squash
  git -C "$root" merge-base --is-ancestor "$B" "$h" 2> /dev/null || { refuse "K3: the base ${B:0:10} is not an ancestor of the PR head '${h:0:10}'"; return 0; } # R:k3-ancestor
  tm=$(git -C "$root" rev-parse --verify --quiet "$m^{tree}" 2> /dev/null)
  th=$(git -C "$root" rev-parse --verify --quiet "$h^{tree}" 2> /dev/null)
  [ -n "$tm" ] && [ "$tm" = "$th" ] || { refuse "K4: queue tree ${tm:0:10} != PR head tree ${th:0:10}"; return 0; } # R:k4-tree
  h=$(git -C "$root" rev-parse --verify --quiet "$h^{commit}" 2> /dev/null)
  PRH="$h"
  jq -e '(.total_count // -1) == ([.workflow_runs[]?] | length)' "$runs" > /dev/null 2>&1 || { refuse "K5: the run list for the PR head is unreadable or has an unread page"; return 0; } # R:k5-runs-complete
  local rid st att
  IFS=$'\t' read -r rid st att <<< "$(pick_run "$runs" "$h")"
  [ "$st" = completed ] || { refuse "K5: the newest ci.yml pull_request run on the PR head is '${st:-absent}' (run ${rid:-none})"; return 0; } # R:k5-completed
  jq -e '(.total_count // -1) == ([.jobs[]?] | length)' "$jobs" > /dev/null 2>&1 || { refuse "K5: the job list of run $rid is unreadable or has an unread page"; return 0; } # R:k5-jobs-complete
  jq -e --arg id "$rid" --arg h "$h" '([.jobs[]?] | length) > 0 and all(.jobs[]; (.run_id | tostring) == $id and .head_sha == $h)' "$jobs" > /dev/null 2>&1 || { refuse "K5: the job list is not run $rid's on the PR head"; return 0; } # R:k5-jobs-run
  local why="" x mc
  [ "$(concl "$jobs" x86-main)" = success ] || why="${why:+$why; }x86-main is $(concl "$jobs" x86-main)" # R:k6-x86
  [ "$(concl "$jobs" guards)" = success ] || why="${why:+$why; }guards is $(concl "$jobs" guards)" # R:k6-guards
  [ "$(concl "$jobs" determinism)" = success ] || why="${why:+$why; }determinism is $(concl "$jobs" determinism)" # R:k6-det
  [ "$(concl "$jobs" gate)" = success ] || why="${why:+$why; }gate is $(concl "$jobs" gate)" # R:k6-gate
  [ "$(concl "$jobs" 'ci / gate')" = success ] || why="${why:+$why; }ci / gate is $(concl "$jobs" 'ci / gate')" # R:k6-cigate
  x=1
  [ -z "$why" ] || x=0
  mc=1
  [ "$(concl "$jobs" mac-check)" = success ] || mc=0 # R:k6-mac
  emit "$x" "$x" "$mc" "run $rid attempt $att on ${h:0:10}" "$h" \
    "K1-K5 hold; x86-main+determinism: $([ "$x" = 1 ] && echo reuse || echo "run ($why)"); mac-check: $([ "$mc" = 1 ] && echo reuse || echo "run ($(concl "$jobs" mac-check))")"
}

resolve() {
  local ev="" ref="" m="" b="" repo="" root="."
  while [ $# -gt 0 ]; do
    case "$1" in
      --event) ev="${2-}" ;;
      --ref) ref="${2-}" ;;
      --queue-head) m="${2-}" ;;
      --base) b="${2-}" ;;
      --repo) repo="${2-}" ;;
      --repo-root) root="${2-}" ;;
      *) echo "ci_mg_reuse: unknown argument '$1'" >&2; return 2 ;;
    esac
    shift 2 || return 2
  done
  [ "$ev" = merge_group ] || { decide --event "$ev"; return; }
  # Every lookup below that fails answers "run it": decide re-reads what it gets.
  local n h w rid
  n=$(printf '%s\n' "$ref" | sed -n 's|^\(refs/heads/\)\{0,1\}gh-readonly-queue/[^/]*/pr-\([0-9][0-9]*\)-.*|\2|p')
  [ -n "$n" ] || { refuse "lookup failed: no PR number in the queue ref '$ref'"; return 0; }
  h=$(gh api "repos/$repo/pulls/$n" --jq .head.sha 2> /dev/null) && [ -n "$h" ] || { refuse "lookup failed: repos/$repo/pulls/$n"; return 0; }
  PRH="$h"
  git -C "$root" cat-file -e "$h^{commit}" 2> /dev/null \
    || git -C "$root" fetch --quiet --no-tags origin "+refs/pull/$n/head" > /dev/null 2>&1 || true
  w=$(mktemp -d "${TMPDIR:-/tmp}/mg-reuse.XXXXXX") || { refuse "lookup failed: no temp dir"; return 0; }
  if ! gh api "repos/$repo/actions/runs?head_sha=$h&event=pull_request&per_page=100" > "$w/runs.json" 2> /dev/null; then
    refuse "lookup failed: the PR head's runs"; rm -rf "${w:?}"; return 0
  fi
  IFS=$'\t' read -r rid _ _ <<< "$(pick_run "$w/runs.json" "$h")"
  if [ -z "$rid" ] || ! gh api "repos/$repo/actions/runs/$rid/jobs?filter=latest&per_page=100" > "$w/jobs.json" 2> /dev/null; then
    refuse "lookup failed: the jobs of run '${rid:-none}'"; rm -rf "${w:?}"; return 0
  fi
  decide --event "$ev" --queue-head "$m" --base "$b" --pr-head "$h" \
    --runs "$w/runs.json" --jobs "$w/jobs.json" --repo-root "$root"
  rm -rf "${w:?}"
}

# ---------------------------------------------------------------- the push arm
# The merge queue tests a commit S, then fast-forwards main to S, and the push run on main used to
# re-run x86-main and determinism on S in full. decide-push lets the push run reuse the queue run's
# result instead, the same way workspace-test already does (T36), and only when that queue run was
# a REAL run of both jobs, never itself a reuse. Every key is READ from git or the API:
#   P1  the event is a push to refs/heads/main
#   P2  S has exactly one parent B, and the push moved main from B (before = B): one squash commit
#   P3  the run and job lists are complete; the newest ci.yml merge_group run on S is completed, its
#       queue branch names B as its base, and every job is that run's, on S
#   P4  that run's x86-main, determinism, gate and `ci / gate` each succeeded (one job of that
#       name), and its x86-main-advisories was skipped: the queue's own mg-reuse said "run", so
#       x86-main and determinism ran there for real. A queue run that reused them cites nothing.
# mac-check is never reused on a push. Anything not proven answers 0 and the jobs run (L25).
# Each refusal ends in a `# R-<NAME>` marker: scripts/check_ci_push_reuse.sh --self-test deletes
# it, and that table must then go RED.

# The newest ci.yml merge_group run on $2 in runs file $1: "id<TAB>status<TAB>attempt<TAB>branch".
pick_mg_run() {
  jq -r --arg h "$2" --arg p "$CI_WORKFLOW_PATH" '
    [.workflow_runs[]? | select(.path == $p and .event == "merge_group" and .head_sha == $h)]
    | sort_by([.created_at, .id]) | last // empty
    | "\(.id)\t\(.status)\t\(.run_attempt // 1)\t\(.head_branch // "")"' "$1" 2> /dev/null
}

decide_push() {
  local ev="" ref="" s="" before="" runs="" jobs="" root="."
  while [ $# -gt 0 ]; do
    case "$1" in
      --event) ev="${2-}" ;;
      --ref) ref="${2-}" ;;
      --sha) s="${2-}" ;;
      --before) before="${2-}" ;;
      --runs) runs="${2-}" ;;
      --jobs) jobs="${2-}" ;;
      --repo-root) root="${2-}" ;;
      *) echo "ci_mg_reuse: unknown argument '$1'" >&2; return 2 ;;
    esac
    shift 2 || return 2
  done
  PRH=""
  [ "$ev" = push ] && [ "$ref" = refs/heads/main ] || { refuse "P1: event '$ev' on '$ref' is not a push to main"; return 0; } # R-PUSHMAIN
  local S sp spa B P
  S=$(git -C "$root" rev-parse --verify --quiet "$s^{commit}" 2> /dev/null)
  [ -n "$S" ] || { refuse "P2: the pushed commit '${s:0:10}' is not in this clone"; return 0; }  # unmarked: an absent commit also fails R-PUSHPARENT
  sp=$(git -C "$root" rev-list --parents -n 1 "$S" 2> /dev/null)
  read -r -a spa <<< "$sp"
  [ "${#spa[@]}" -eq 2 ] || { refuse "P2: the pushed commit ${S:0:10} is not a single-parent (squash) commit"; return 0; } # R-PUSHPARENT
  B="${spa[1]}"
  P=$(git -C "$root" rev-parse --verify --quiet "$before^{commit}" 2> /dev/null)
  [ -n "$P" ] && [ "$P" = "$B" ] || { refuse "P2: the push moved main from '${before:0:10}', not from ${S:0:10}'s parent ${B:0:10}"; return 0; } # R-PUSHBEFORE
  jq -e '(.total_count // -1) == ([.workflow_runs[]?] | length)' "$runs" > /dev/null 2>&1 || { refuse "P3: the merge_group run list for ${S:0:10} is unreadable or has an unread page"; return 0; } # R-PUSHRUNS
  local rid st att br
  IFS=$'\t' read -r rid st att br <<< "$(pick_mg_run "$runs" "$S")"
  [ "$st" = completed ] || { refuse "P3: the newest ci.yml merge_group run on ${S:0:10} is '${st:-absent}' (run ${rid:-none})"; return 0; } # R-PUSHDONE
  case "$br" in gh-readonly-queue/main/pr-[0-9]*-"$B") ;; *) refuse "P3: merge_group run $rid's queue branch '$br' does not name base ${B:0:10}"; return 0 ;; esac # R-PUSHQBASE
  jq -e '(.total_count // -1) == ([.jobs[]?] | length)' "$jobs" > /dev/null 2>&1 || { refuse "P3: the job list of run $rid is unreadable or has an unread page"; return 0; } # R-PUSHJOBS
  jq -e --arg id "$rid" --arg h "$S" '([.jobs[]?] | length) > 0 and all(.jobs[]; (.run_id | tostring) == $id and .head_sha == $h)' "$jobs" > /dev/null 2>&1 || { refuse "P3: the job list is not run $rid's on ${S:0:10}"; return 0; } # R-PUSHJOBRUN
  local why=""
  [ "$(concl "$jobs" x86-main)" = success ] || why="${why:+$why; }x86-main is $(concl "$jobs" x86-main)" # R-PUSHX
  [ "$(concl "$jobs" guards)" = success ] || why="${why:+$why; }guards is $(concl "$jobs" guards)" # R-PUSHGUARDS
  [ "$(concl "$jobs" determinism)" = success ] || why="${why:+$why; }determinism is $(concl "$jobs" determinism)" # R-PUSHDET
  [ "$(concl "$jobs" gate)" = success ] || why="${why:+$why; }gate is $(concl "$jobs" gate)" # R-PUSHGATE
  [ "$(concl "$jobs" 'ci / gate')" = success ] || why="${why:+$why; }ci / gate is $(concl "$jobs" 'ci / gate')" # R-PUSHCIGATE
  [ "$(concl "$jobs" x86-main-advisories)" = skipped ] || why="${why:+$why; }x86-main-advisories is $(concl "$jobs" x86-main-advisories), so the queue run reused rather than ran" # R-PUSHREAL
  [ -z "$why" ] || { refuse "P4: merge_group run $rid on ${S:0:10}: $why"; return 0; }
  emit 1 1 0 "merge_group run $rid attempt $att on ${S:0:10}" "" \
    "P1-P4 hold: merge_group run $rid ran x86-main + determinism on this very commit and both passed; mac-check: run (never reused on a push)"
}

resolve_push() {
  local ev="" ref="" s="" before="" repo="" root="."
  while [ $# -gt 0 ]; do
    case "$1" in
      --event) ev="${2-}" ;;
      --ref) ref="${2-}" ;;
      --sha) s="${2-}" ;;
      --before) before="${2-}" ;;
      --repo) repo="${2-}" ;;
      --repo-root) root="${2-}" ;;
      *) echo "ci_mg_reuse: unknown argument '$1'" >&2; return 2 ;;
    esac
    shift 2 || return 2
  done
  PRH=""
  [ "$ev" = push ] && [ "$ref" = refs/heads/main ] || { decide_push --event "$ev" --ref "$ref"; return; }
  [[ $s =~ ^[0-9a-f]{40}$ ]] || { refuse "lookup failed: '$s' is not a 40-hex sha"; return 0; } # R-PUSHSHAFORM
  # Two API calls. Every lookup that fails answers "run it": decide-push re-reads what it gets.
  # A failing call can still print a body, so only its exit status says it failed.
  local w rid rc
  w=$(mktemp -d "${TMPDIR:-/tmp}/mg-reuse.XXXXXX") || { refuse "lookup failed: no temp dir"; return 0; }
  gh api "repos/$repo/actions/runs?head_sha=$s&event=merge_group&per_page=100" > "$w/runs.json" 2> /dev/null; rc=$?
  [ "$rc" -eq 0 ] || { refuse "lookup failed: the merge_group runs on ${s:0:10}"; rm -rf "${w:?}"; return 0; } # R-PUSHRUNSAPI
  IFS=$'\t' read -r rid _ _ _ <<< "$(pick_mg_run "$w/runs.json" "$s")"
  [ -n "$rid" ] || { refuse "lookup failed: no ci.yml merge_group run on ${s:0:10}"; rm -rf "${w:?}"; return 0; }
  gh api "repos/$repo/actions/runs/$rid/jobs?filter=latest&per_page=100" > "$w/jobs.json" 2> /dev/null; rc=$?
  [ "$rc" -eq 0 ] || { refuse "lookup failed: the jobs of merge_group run $rid"; rm -rf "${w:?}"; return 0; } # R-PUSHJOBSAPI
  decide_push --event "$ev" --ref "$ref" --sha "$s" --before "$before" \
    --runs "$w/runs.json" --jobs "$w/jobs.json" --repo-root "$root"
  rm -rf "${w:?}"
}

# ---------------------------------------------------------------- case table
# The jobs of run 37040341824, #4655's only PR-head run (db86acf021), verbatim
# from the API (two GPU-host leg job names replaced; the key never reads them):
# x86-main failed while gate and ci / gate passed.
PLANTED_4655_JOBS=$'gpu-touched\tsuccess\nmutants-table-scope\tsuccess\ndeterminism\tsuccess\nworkspace-test-shard (1)\tsuccess\nmac-check\tsuccess\nworkspace-test-shard (3)\tsuccess\nworkspace-test-shard (2)\tsuccess\nx86-main\tfailure\nmutants-table\tskipped\nmutants-shard\tskipped\ngpu-host-leg-a\tsuccess\ngpu-host-leg-b\tsuccess\nworkspace-test\tsuccess\nci / gate\tsuccess\ngate\tsuccess'
GREEN_JOBS=$'x86-main\tsuccess\nguards\tsuccess\ndeterminism\tsuccess\nmac-check\tsuccess\nworkspace-test\tsuccess\nci / gate\tsuccess\ngate\tsuccess'

# jobs_json HEAD RUN_ID EXTRA_TOTAL < "name<TAB>conclusion" lines
jobs_json() {
  jq -R -s --arg h "$1" --argjson rid "$2" --argjson extra "$3" '
    [split("\n")[] | select(length > 0) | split("\t")
     | {name: .[0], conclusion: .[1], status: "completed", run_id: $rid, head_sha: $h}]
    | {total_count: (length + $extra), jobs: .}'
}
# run_obj ID HEAD STATUS [PATH] [EVENT] [CREATED]
run_obj() {
  jq -n --argjson id "$1" --arg h "$2" --arg s "$3" --arg p "${4:-$CI_WORKFLOW_PATH}" \
    --arg e "${5:-pull_request}" --arg c "${6:-2026-10-01T00:00:00Z}" \
    '{id: $id, path: $p, event: $e, status: $s, head_sha: $h, run_attempt: 1, created_at: $c}'
}
# runs_json EXTRA_TOTAL < run objects
runs_json() { jq -s --argjson extra "$1" '{total_count: (length + $extra), workflow_runs: .}'; }

kv() { printf '%s\n' "$2" | sed -n "s/^$1=//p"; }

self_test() {
  local wf="$CI_WORKFLOW_PATH"
  [ "${1-}" = --workflow ] && wf="${2-}"
  local T pass=0 fail=0
  T=$(mktemp -d "${TMPDIR:-/tmp}/mg-reuse-st.XXXXXX") || return 1
  local R="$T/r"
  git init -q "$R" || return 1
  gc() { git -C "$R" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@example.invalid "$@"; }
  ct() { gc commit-tree "$@"; }
  echo 0 > "$R/f"; gc add f; gc commit -q -m r0
  local R0 A B H X BM H2 B2 M M_TWO M_TREE M_K3 M_K2
  R0=$(gc rev-parse HEAD)
  echo a > "$R/f"; gc commit -q -am a; A=$(gc rev-parse HEAD)
  echo b > "$R/f"; gc commit -q -am b; B=$(gc rev-parse HEAD)
  echo h > "$R/g"; gc add g; gc commit -q -m h; H=$(gc rev-parse HEAD)
  X=$(ct -p "$R0" -m x "$R0^{tree}")
  BM=$(ct -p "$A" -p "$X" -m merge-base "$B^{tree}")
  H2=$(ct -p "$BM" -m h2 "$H^{tree}")
  B2=$(ct -p "$A" -m b2 "$B^{tree}")
  M=$(ct -p "$B" -m queue "$H^{tree}")
  M_TWO=$(ct -p "$B" -p "$H" -m queue2 "$H^{tree}")
  M_TREE=$(ct -p "$B" -m queue3 "$B^{tree}")
  M_K3=$(ct -p "$B2" -m queue4 "$H^{tree}")
  M_K2=$(ct -p "$BM" -m queue5 "$H2^{tree}")
  local F="$T/fx"; mkdir -p "$F"
  run_obj 1001 "$H" completed | runs_json 0 > "$F/runs.ok"
  jobs_json "$H" 1001 0 <<< "$GREEN_JOBS" > "$F/jobs.ok"
  jobs_json "$H" 1002 0 <<< "$GREEN_JOBS" > "$F/jobs.running"
  run_obj 1001 "$H2" completed | runs_json 0 > "$F/runs.h2"
  jobs_json "$H2" 1001 0 <<< "$GREEN_JOBS" > "$F/jobs.h2"
  run_obj 37040341824 "$H" completed | runs_json 0 > "$F/runs.4655"
  jobs_json "$H" 37040341824 0 <<< "$PLANTED_4655_JOBS" > "$F/jobs.4655"
  { run_obj 1001 "$H" completed; run_obj 1002 "$H" in_progress "" "" 2026-10-02T00:00:00Z; } | runs_json 0 > "$F/runs.newer-running"
  run_obj 1001 "$H" completed | runs_json 1 > "$F/runs.partial"
  run_obj 1001 "$H" completed .github/workflows/other.yml | runs_json 0 > "$F/runs.otherwf"
  run_obj 1001 "$H" completed "" push | runs_json 0 > "$F/runs.push"
  jobs_json "$H" 1001 1 <<< "$GREEN_JOBS" > "$F/jobs.partial"
  jobs_json "$H" 999 0 <<< "$GREEN_JOBS" > "$F/jobs.otherrun"
  jobs_json "$B" 1001 0 <<< "$GREEN_JOBS" > "$F/jobs.otherhead"
  printf '{"total_count":0,"jobs":[]}\n' > "$F/jobs.empty"
  printf 'not json\n' > "$F/garbage"
  local name
  for name in determinism gate 'ci / gate' mac-check guards; do
    sed "s|^$name	success\$|$name	failure|" <<< "$GREEN_JOBS" | jobs_json "$H" 1001 0 > "$F/jobs.fail-${name//[ \/]/_}"
  done
  sed '/^determinism	/d' <<< "$GREEN_JOBS" | jobs_json "$H" 1001 0 > "$F/jobs.no-det"
  sed '/^guards	/d' <<< "$GREEN_JOBS" | jobs_json "$H" 1001 0 > "$F/jobs.no-guards"
  { cat <<< "$GREEN_JOBS"; printf 'x86-main\tsuccess\n'; } | jobs_json "$H" 1001 0 > "$F/jobs.dup-x86"

  row() { # row NAME "x86 det mac" decide-args...
    local n="$1" want="$2" out got; shift 2
    out=$(decide "$@" --repo-root "$R")
    got="$(kv x86 "$out") $(kv det "$out") $(kv mac "$out")"
    if [ "$got" = "$want" ]; then pass=$((pass + 1)); else
      fail=$((fail + 1)); echo "FAIL $n: want '$want' got '$got' -- $(kv reason "$out")"; fi
  }
  dr() { local n="$1" want="$2" ev="$3" m="$4" b="$5" h="$6" r="$7" j="$8"
    row "$n" "$want" --event "$ev" --queue-head "$m" --base "$b" --pr-head "$h" --runs "$F/$r" --jobs "$F/$j"; }

  dr planted-4655-must-run      "0 0 1" merge_group "$M" "$B" "$H" runs.4655 jobs.4655
  dr positive-reuse             "1 1 1" merge_group "$M" "$B" "$H" runs.ok jobs.ok
  dr positive-short-shas        "1 1 1" merge_group "${M:0:12}" "${B:0:12}" "${H:0:12}" runs.ok jobs.ok
  dr regress-pull_request       "0 0 0" pull_request "$M" "$B" "$H" runs.ok jobs.ok
  dr regress-push               "0 0 0" push "$M" "$B" "$H" runs.ok jobs.ok
  dr regress-empty-event        "0 0 0" "" "$M" "$B" "$H" runs.ok jobs.ok
  dr k1-two-parent-queue-head   "0 0 0" merge_group "$M_TWO" "$B" "$H" runs.ok jobs.ok
  dr k1-base-not-the-parent     "0 0 0" merge_group "$M" "$A" "$H" runs.ok jobs.ok
  dr k1-queue-head-absent       "0 0 0" merge_group 0123456789abcdef0123456789abcdef01234567 "$B" "$H" runs.ok jobs.ok
  dr k1-base-empty              "0 0 0" merge_group "$M" "" "$H" runs.ok jobs.ok
  dr k2-base-is-a-merge         "0 0 0" merge_group "$M_K2" "$BM" "$H2" runs.h2 jobs.h2
  dr k3-base-not-ancestor       "0 0 0" merge_group "$M_K3" "$B2" "$H" runs.ok jobs.ok
  dr k3-pr-head-absent          "0 0 0" merge_group "$M" "$B" 0123456789abcdef0123456789abcdef01234567 runs.ok jobs.ok
  dr k4-tree-differs            "0 0 0" merge_group "$M_TREE" "$B" "$H" runs.ok jobs.ok
  dr k5-newest-run-in-progress  "0 0 0" merge_group "$M" "$B" "$H" runs.newer-running jobs.ok
  dr k5-in-progress-own-jobs    "0 0 0" merge_group "$M" "$B" "$H" runs.newer-running jobs.running
  dr k5-run-list-unread-page    "0 0 0" merge_group "$M" "$B" "$H" runs.partial jobs.ok
  dr k5-only-other-workflow     "0 0 0" merge_group "$M" "$B" "$H" runs.otherwf jobs.ok
  dr k5-only-push-run           "0 0 0" merge_group "$M" "$B" "$H" runs.push jobs.ok
  dr k5-runs-garbage            "0 0 0" merge_group "$M" "$B" "$H" garbage jobs.ok
  dr k5-job-list-unread-page    "0 0 0" merge_group "$M" "$B" "$H" runs.ok jobs.partial
  dr k5-jobs-of-another-run     "0 0 0" merge_group "$M" "$B" "$H" runs.ok jobs.otherrun
  dr k5-jobs-on-another-head    "0 0 0" merge_group "$M" "$B" "$H" runs.ok jobs.otherhead
  dr k5-jobs-empty              "0 0 0" merge_group "$M" "$B" "$H" runs.ok jobs.empty
  dr k5-jobs-garbage            "0 0 0" merge_group "$M" "$B" "$H" runs.ok garbage
  dr k6-determinism-failed      "0 0 1" merge_group "$M" "$B" "$H" runs.ok jobs.fail-determinism
  dr k6-determinism-missing     "0 0 1" merge_group "$M" "$B" "$H" runs.ok jobs.no-det
  dr k6-head-gate-failed        "0 0 1" merge_group "$M" "$B" "$H" runs.ok jobs.fail-gate
  dr k6-head-ci-gate-failed     "0 0 1" merge_group "$M" "$B" "$H" runs.ok jobs.fail-ci___gate
  dr k6-x86-main-twice          "0 0 1" merge_group "$M" "$B" "$H" runs.ok jobs.dup-x86
  dr k6-mac-failed              "1 1 0" merge_group "$M" "$B" "$H" runs.ok jobs.fail-mac-check
  dr k6-guards-failed           "0 0 1" merge_group "$M" "$B" "$H" runs.ok jobs.fail-guards
  dr k6-guards-missing          "0 0 1" merge_group "$M" "$B" "$H" runs.ok jobs.no-guards

  # resolve, end to end, through a stub gh serving canned JSON. A failed lookup never reuses.
  local SB="$T/bin"; mkdir -p "$SB"
  cat > "$SB/gh" <<'STUB'
#!/usr/bin/env bash
# A failing call still prints its body (gh api writes an error response to stdout),
# so only the exit status says it failed: a dropped status check is caught.
rc=0
case "$2" in
  repos/o/r/pulls/*) printf '%s\n' "$GH_STUB_HEAD" ;;
  repos/o/r/actions/runs\?*) cat "$GH_STUB_DIR/runs.ok" ;;
  repos/o/r/actions/runs/*/jobs\?*) cat "$GH_STUB_DIR/jobs.ok" ;;
  *) rc=1 ;;
esac
[ -n "${GH_STUB_FAIL:-}" ] && case "$*" in *"$GH_STUB_FAIL"*) exit 1 ;; esac
exit "$rc"
STUB
  chmod +x "$SB/gh"
  rrow() { # rrow NAME "x86 det mac" FAIL_PATTERN HEAD REF
    local n="$1" want="$2" out got
    out=$(PATH="$SB:$PATH" GH_STUB_DIR="$F" GH_STUB_FAIL="$3" GH_STUB_HEAD="$4" \
      resolve --event merge_group --ref "$5" --queue-head "$M" --base "$B" --repo o/r --repo-root "$R")
    got="$(kv x86 "$out") $(kv det "$out") $(kv mac "$out")"
    if [ "$got" = "$want" ]; then pass=$((pass + 1)); else
      fail=$((fail + 1)); echo "FAIL resolve-$n: want '$want' got '$got' -- $(kv reason "$out")"; fi
  }
  local QREF="refs/heads/gh-readonly-queue/main/pr-4655-${B}"
  rrow positive          "1 1 1" ""        "$H" "$QREF"
  rrow short-ref         "1 1 1" ""        "$H" "gh-readonly-queue/main/pr-4655-${B}"
  rrow no-pr-number      "0 0 0" ""        "$H" "refs/heads/main"
  rrow gh-pulls-fails    "0 0 0" "pulls/"  "$H" "$QREF"
  rrow gh-runs-fails     "0 0 0" "head_sha=" "$H" "$QREF"
  rrow gh-jobs-fails     "0 0 0" "/jobs"   "$H" "$QREF"
  rrow head-unfetchable  "0 0 0" ""        0123456789abcdef0123456789abcdef01234567 "$QREF"

  # The verdict block, run exactly as ci.yml carries it in ci-gate and gate.
  local blocks nb
  blocks=$(awk '/# --- MG-REUSE-VERDICT-BEGIN/{f=1} f{sub(/^ +/, ""); print} /# --- MG-REUSE-VERDICT-END/{f=0}' "$wf" 2> /dev/null)
  nb=$(grep -c 'MG-REUSE-VERDICT-BEGIN' <<< "$blocks")
  local blk; blk=$(awk '/MG-REUSE-VERDICT-BEGIN/{n++} n==1{print} /MG-REUSE-VERDICT-END/ && n==1{exit}' <<< "$blocks")
  if [ "$nb" != 2 ] || [ "$(awk '/MG-REUSE-VERDICT-BEGIN/{n++} n==2{print}' <<< "$blocks")" != "$blk" ]; then
    fail=$((fail + 1)); echo "FAIL verdict-block: want 2 identical copies (ci-gate, gate) in $wf, found $nb"
  fi
  vrow() { # vrow NAME WANT(rc:REUSE) EVT MG MG_X86 MG_DET ADV
    local n="$1" want="$2" got
    got=$(EVT="$3" MG="$4" MG_X86="$5" MG_DET="$6" ADV="$7" MG_CITE=t \
      bash -c 'set -euo pipefail; '"$blk"$'\necho "REUSE=$REUSE"' 2> /dev/null | sed -n 's/^REUSE=//p'; echo ":${PIPESTATUS[0]}")
    got="$(tr -d '\n' <<< "$got")"
    case "$want" in
      1) [ "$got" = "1:0" ] ;;
      0) [ "$got" = "0:0" ] ;;
      red) [ "${got##*:}" != 0 ] ;;
    esac && pass=$((pass + 1)) || { fail=$((fail + 1)); echo "FAIL verdict-$n: want $want got '$got'"; }
  }
  vrow reuse-ok             1   merge_group success 1 1 success
  vrow live-pull_request    0   pull_request skipped "" "" skipped
  vrow live-mg-said-run     0   merge_group success 0 0 skipped
  vrow live-mg-job-failed   0   merge_group failure "" "" skipped
  vrow red-advisories-fail  red merge_group success 1 1 failure
  vrow red-advisories-skip  red merge_group success 1 1 skipped
  vrow red-split-x86-only   red merge_group success 1 0 success
  vrow red-split-det-only   red merge_group success 0 1 skipped
  vrow red-flag-off-event   red pull_request success 1 1 success
  vrow red-mg-failed-flag1  red merge_group failure 1 1 success
  vrow red-advisories-ran-live red merge_group success 0 0 success
  vrow reuse-push-ok        1   push success 1 1 success
  vrow live-push-said-run   0   push success 0 0 skipped
  vrow live-push-mg-failed  0   push failure "" "" skipped
  vrow live-push-pre-t43    0   push skipped "" "" skipped
  vrow red-push-adv-fail    red push success 1 1 failure
  vrow red-push-adv-skip    red push success 1 1 skipped
  vrow red-push-split       red push success 1 0 success
  vrow red-flag-workflow-dispatch red workflow_dispatch success 1 1 success

  # The wiring: each skip and each refusal is where ci.yml decides it.
  job() { awk -v j="  $1:" '$0 == j {f = 1; next} f && /^  [A-Za-z0-9_-]+:$/ {exit} f' "$wf" 2> /dev/null; }
  wire() { # wire JOB FIXED_STRING
    # Capture, then a here-string: never pipe into grep -q here. grep -q exits on its first
    # match, job's awk takes SIGPIPE (141), and under this file's pipefail the pipeline is
    # 141, a false FAIL though grep matched (scripts/check_no_pipe_into_grep_q.sh).
    local t; t=$(job "$1")
    if grep -qF -- "$2" <<<"$t"; then pass=$((pass + 1)); else
      fail=$((fail + 1)); echo "FAIL wiring: job '$1' lacks: $2"; fi
  }
  wire mg-reuse "if: github.event_name == 'merge_group' || (github.event_name == 'push' && github.ref == 'refs/heads/main')"
  wire mg-reuse 'bash scripts/ci_mg_reuse.sh --self-test'
  wire mg-reuse 'bash scripts/ci_mg_reuse.sh resolve --event "$EVT" --ref "$QREF"'
  wire mg-reuse 'bash scripts/ci_mg_reuse.sh resolve-push --event "$EVT" --ref "$GITHUB_REF" --sha "$GITHUB_SHA" --before "$PBEFORE"'
  wire mg-reuse 'PBEFORE: ${{ github.event.before }}'
  wire mg-reuse-self-test 'bash scripts/ci_mg_reuse.sh --self-test'
  wire x86-main 'needs: [mg-reuse]'
  wire x86-main "if: \${{ !cancelled() && needs.mg-reuse.outputs.x86 != '1' }}"
  wire guards 'needs: [mg-reuse]'  # #3668: the guards job is part of the x86-main unit
  wire guards "if: \${{ !cancelled() && needs.mg-reuse.outputs.x86 != '1' }}"
  wire gate 'needs: [x86-main, guards,'
  wire gate 'for pair in GRD:guard-tree GRD:guard-cargo X86:sov.gate DET:determinism-compare; do'
  wire determinism 'needs: [mg-reuse, x86-main]'  # T42: determinism also waits on x86-main
  wire determinism "if: \${{ !cancelled() && needs.mg-reuse.outputs.det != '1' }}"
  wire mac-check 'needs: [mg-reuse]'
  wire mac-check "if: \${{ !cancelled() && needs.mg-reuse.outputs.mac != '1' }}"
  wire x86-main-advisories "if: \${{ !cancelled() && needs.mg-reuse.outputs.x86 == '1' }}"
  wire x86-main-advisories 'cargo deny check advisories'
  wire x86-main-advisories 'bash scripts/check_deny_exemptions_live.sh'
  wire ci-gate 'MG-REUSE-VERDICT-BEGIN'
  wire ci-gate 'needs: [x86-main, mg-reuse, x86-main-advisories]'
  wire gate 'MG-REUSE-VERDICT-BEGIN'
  wire gate 'mg-reuse, x86-main-advisories, mg-reuse-self-test]'
  wire gate '[ "$MST" = success ]'

  # Positive control for wire itself: one job of more than 64 KiB (a pipe buffer) whose
  # wanted line comes first. A pipe from job into grep -q takes SIGPIPE on this every
  # time, so if wire ever pipes again this row FAILs deterministically.
  local wf_real="$wf"
  wf="$T/wire-big.yml"
  { printf 'jobs:\n  wire-big:\n    needs: [mg-reuse]\n    steps:\n'
    awk 'BEGIN { for (i = 0; i < 3000; i++) printf "      - run: echo wire-control-filler-%05d\n", i }'
    printf '  after-big:\n    needs: [wire-big]\n'
  } > "$wf"
  wire wire-big 'needs: [mg-reuse]'
  wf="$wf_real"

  rm -rf "${T:?}"
  echo "ci_mg_reuse self-test: $pass passed, $fail failed"
  [ "$fail" -eq 0 ] && [ "$pass" -gt 0 ]
}

case "${1-}" in
  decide) shift; decide "$@" ;;
  resolve) shift; resolve "$@" ;;
  decide-push) shift; decide_push "$@" ;;
  resolve-push) shift; resolve_push "$@" ;;
  --self-test) shift; self_test "$@" ;;
  *) usage ;;
esac
