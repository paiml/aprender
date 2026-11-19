#!/usr/bin/env bash
# ci_push_reuse_facts.sh - gathers the facts scripts/ci_push_reuse.sh decides on (#7298).
# Git and API reads only; no decision lives here. It prints one JSON object on stdout and exits 0
# only when EVERY fact was read; any failed or partial read prints the reason on stderr and exits 1,
# and the caller then runs the jobs (reuse=no). About 6 REST calls: the PR of the commit, the PR, the
# run list, the job list, the stamp job's log, x86-main's log.
#
#   ci_push_reuse_facts.sh --repo OWNER/NAME --sha S --before P --ref REF --event EVT --now EPOCH
#     (--now is supplied by the caller, `date +%s`; this script reads no clock, so it is deterministic)
#   ci_push_reuse_facts.sh --parse-stamp-log FILE        # the stamp parser alone, on a saved job log (the tests use it)
#   ci_push_reuse_facts.sh --parse-section-log FILE NAME # one section's result line from a saved x86-main log
#
# Log lines: GitHub prefixes every log line with a timestamp, and ECHOES each line of a `run:` script
# (colour-coded, after the timestamp) before running it. Only lines the step EMITTED count: the
# timestamp, one space, then the marker at column 0. The echoed script text never matches.
#
# Needs a clone with full history (the checkout of mg-reuse uses fetch-depth 0), gh, git, jq.
set -uo pipefail
die() { echo "facts: $*" >&2; exit 1; }

TS='[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z'
# normalise a job log: no byte-order mark, no carriage returns
clean_log() { local l="$1"; l="${l#$'\xEF\xBB\xBF'}"; printf '%s\n' "${l//$'\r'/}"; }

# stamp_from_log LOG: prints the stamp as JSON {sha,tree,base,head}, or `null` when the step emitted none.
# Dies on two stamps, or on a stamp plus an emitted `unavailable` line.
stamp_from_log() {
  local log stamps nstamps nun
  log=$(clean_log "$1")
  stamps=$(grep -E "^${TS} PUSH-REUSE-STAMP tested_sha=[0-9a-f]{40} tested_tree=[0-9a-f]{40} tested_base=[0-9a-f]{40} tested_head=[0-9a-f]{40}\$" <<< "$log") || stamps=""
  nstamps=$(grep -c . <<< "$stamps") || nstamps=0
  nun=$(grep -cE "^${TS} PUSH-REUSE-STAMP unavailable" <<< "$log") || nun=0
  [ "$nstamps" -le 1 ] || die "the stamp job log has $nstamps stamp lines"
  if [ "$nstamps" -eq 1 ] && [ "$nun" -gt 0 ]; then die "the stamp job log has a stamp and an unavailable line"; fi
  if [ "$nstamps" -eq 0 ]; then echo null; return 0; fi
  sed -E "s/^${TS} PUSH-REUSE-STAMP tested_sha=([0-9a-f]+) tested_tree=([0-9a-f]+) tested_base=([0-9a-f]+) tested_head=([0-9a-f]+)\$/\1 \2 \3 \4/" <<< "$stamps" \
    | jq -R -c 'split(" ") | {sha: .[0], tree: .[1], base: .[2], head: .[3]}' || die "stamp unreadable"
}

# section_result LOG NAME: prints success|failure|skipped|cancelled when fat_driver emitted exactly ONE
# `section NAME: <result> in <n>s` line, else `null` (none, or more than one).
section_result() {
  local log lines n
  [[ $2 =~ ^[A-Za-z0-9._-]+$ ]] || die "bad section name"
  log=$(clean_log "$1")
  lines=$(grep -E "^${TS} section ${2//./\\.}: (success|failure|skipped|cancelled) in [0-9]+s\$" <<< "$log") || lines=""
  n=$(grep -c . <<< "$lines") || n=0
  if [ "$n" -ne 1 ]; then echo null; return 0; fi
  sed -E "s/^${TS} section [^:]+: ([a-z]+) in [0-9]+s\$/\"\1\"/" <<< "$lines"
}

case "${1-}" in
  --parse-stamp-log) case "${2-}" in *..*) echo "facts: refusing a log path with \"..\"" >&2; exit 1 ;; esac  # bashrs SEC010: refuse the path, never read it
    [ -r "${2-}" ] || die "no readable log file"; stamp_from_log "$(< "$2")"; exit $? ;;
  --parse-section-log) case "${2-}" in *..*) echo "facts: refusing a log path with \"..\"" >&2; exit 1 ;; esac  # bashrs SEC010
    [ -r "${2-}" ] && [ -n "${3-}" ] || die "usage: --parse-section-log FILE NAME"; section_result "$(< "$2")" "$3"; exit $? ;;
esac

repo="" sha="" before="" ref="" event="" now=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo|--sha|--before|--ref|--event|--now) [ "$#" -ge 2 ] || { echo "facts: $1 needs a value" >&2; exit 2; } ;;  # a bare `shift 2` fails and never advances: that looped forever
  esac
  case "$1" in
    --repo) repo="${2-}"; shift 2 ;; --sha) sha="${2-}"; shift 2 ;; --before) before="${2-}"; shift 2 ;;
    --ref) ref="${2-}"; shift 2 ;; --event) event="${2-}"; shift 2 ;;
    --now) now="${2-}"; shift 2 ;;
    *) echo "facts: unknown argument $1" >&2; exit 1 ;;
  esac
done
for v in repo sha before ref event; do [ -n "${!v}" ] || die "--$v is required"; done
[[ $now =~ ^[0-9]+$ ]] || die "--now (epoch seconds, supplied by the caller) is required"
[[ $sha =~ ^[0-9a-f]{40}$ ]] || die "sha is not 40-hex"
command -v jq > /dev/null && command -v gh > /dev/null && command -v git > /dev/null || die "jq, gh and git are required"

tree=$(git rev-parse "$sha^{tree}" 2> /dev/null) || die "commit $sha not in this clone"
parents=$(git rev-list --parents -n 1 "$sha" | cut -d' ' -f2- | jq -R 'split(" ") | map(select(length > 0))') || die "cannot read parents"
b=$(jq -r '.[0] // empty' <<< "$parents")
[ -n "$b" ] || die "no parent"

# S's PR: the merged PR whose merge_commit_sha is S (1 call)
prs=$(gh api "repos/$repo/commits/$sha/pulls" 2> /dev/null) || die "commits/$sha/pulls read failed"
n=$(jq -r --arg s "$sha" '[.[] | select(.merge_commit_sha == $s)] | if length == 1 then .[0].number else empty end' <<< "$prs") || die "pulls list unreadable"
[ -n "$n" ] || die "no single PR names $sha as its merge commit"
prj=$(gh api "repos/$repo/pulls/$n" 2> /dev/null) || die "pulls/$n read failed"
h=$(jq -r '.head.sha // empty' <<< "$prj")
[[ $h =~ ^[0-9a-f]{40}$ ]] || die "PR head sha unreadable"
# base_ref and head_ref come from the PR object, not from the run: a run's own PR list is computed live from OPEN
# PRs and is empty after the merge, so it can never bind a run to a merged PR. What survives the merge is (a) the PR's
# own head.sha and base.ref, read here, and (b) the stamp the run recorded (the base and head of its test merge).
pr=$(jq -c --arg repo "$repo" '{number: .number, merged: (.merged == true), merge_commit_sha: .merge_commit_sha, base_ref: .base.ref, head_ref: .head.ref,
  is_fork: ((.head.repo.full_name // "") != $repo), head_repo: (.head.repo.full_name // null), draft: (.draft == true), head_sha: .head.sha}' <<< "$prj") || die "PR facts unreadable"

# H's ci.yml pull_request runs (1 call): the newest by id, and the newest success to cite
runs=$(gh api "repos/$repo/actions/workflows/ci.yml/runs?head_sha=$h&event=pull_request&per_page=100" 2> /dev/null) || die "run list read failed"
runs_complete=$(jq '(.total_count // -1) == ([.workflow_runs[]?] | length)' <<< "$runs") || die "run list unreadable"
newest=$(jq -r '[.workflow_runs[]? | select(.head_sha == "'"$h"'")] | map(.id) | max // empty' <<< "$runs")
cite=$(jq -r '[.workflow_runs[]? | select(.head_sha == "'"$h"'" and .conclusion == "success")] | map(.id) | max // empty' <<< "$runs")
[ -n "$newest" ] || die "no pull_request run on $h"
rid="${cite:-$newest}"
run=$(jq -c --argjson id "$rid" --argjson nw "$newest" '.workflow_runs[] | select(.id == $id)
  | {id: .id, event: .event, path: .path, head_sha: .head_sha, head_branch: .head_branch, status: .status, conclusion: .conclusion, newest_id: $nw,
     started_epoch: (.run_started_at | fromdateiso8601)}' <<< "$runs") || die "run object unreadable"

jobsj=$(gh api "repos/$repo/actions/runs/$rid/jobs?per_page=100&filter=latest" 2> /dev/null) || die "job list read failed"
jobs=$(jq -c '[.jobs[]? | {name: .name, conclusion: .conclusion, run_id: .run_id, head_sha: .head_sha}]' <<< "$jobsj") || die "job list unreadable"
jobs_total=$(jq '.total_count // -1' <<< "$jobsj") || die "job count unreadable"

# the stamp the PR run's workspace-test-ran job wrote (1 call). It is MANDATORY: absent = null and the decision refuses.
# The log is read whole (a failed read is a refusal) and matched with here-strings, never a pipe; only EMITTED lines count.
stamp=null; m_derived=null
wtid=$(jq -r '[.jobs[]? | select(.name == "workspace-test-ran")] | if length == 1 then .[0].id else empty end' <<< "$jobsj")
if [ -n "$wtid" ]; then
  log=$(gh api "repos/$repo/actions/jobs/$wtid/logs" 2> /dev/null) || die "stamp job log read failed"
  stamp=$(stamp_from_log "$log") || exit 1
  ms=$(jq -r '.sha // empty' <<< "$stamp")
  if [ -n "$ms" ] && { git cat-file -e "$ms^{commit}" 2> /dev/null || git fetch -q --no-tags origin "$ms" 2> /dev/null; } && git cat-file -e "$ms^{commit}" 2> /dev/null; then
    m_derived=$(jq -n -c --arg t "$(git rev-parse "$ms^{tree}")" --arg b "$(git rev-parse "$ms^1" 2> /dev/null)" --arg h "$(git rev-parse "$ms^2" 2> /dev/null)" '{tree: $t, base: $b, head: $h}') || m_derived=null
  fi
fi

# The sovereign-ci `test` section: ci/sections.yml sets heavy_on 'train', so on an ordinary PR x86-main SKIPS sov.test (printed
# NOT MEASURED) while a push to main owes it. Whether the cited run ran it is read from THAT run's own x86-main job log
# (fat_driver's emitted `section sov.test: <result> in <n>s` line), never from the branch name or a label. 1 call.
sov_test=null
xid=$(jq -r '[.jobs[]? | select(.name == "x86-main")] | if length == 1 then .[0].id else empty end' <<< "$jobsj")
if [ -n "$xid" ]; then
  xlog=$(gh api "repos/$repo/actions/jobs/$xid/logs" 2> /dev/null) || die "x86-main job log read failed"
  sov_test=$(section_result "$xlog" sov.test) || exit 1
fi

# B..S changed paths; the push matrix of S. Whether the CITED RUN reused a receipt is
# read by the decision from that run's own job list (C4-not-a-reuse), never from workflow text.
paths=$(git diff --no-renames --name-only "$b" "$sha" | jq -R -s -c 'split("\n") | map(select(length > 0))') || die "diff unreadable"
wf=$(git show "$sha:.github/workflows/ci.yml" 2> /dev/null) || die "ci.yml of $sha unreadable"
exp=$(printf '%s\n' "$wf" | sed -n "s/^[[:space:]]*shards: .*|| '\[\([0-9][0-9]*\)\]').*/\1/p" | head -n 1)
[[ $exp =~ ^[0-9]+$ ]] || die "push shard count not readable from ci.yml"

jq -n -c --arg event "$event" --arg ref "$ref" --arg sha "$sha" --arg tree "$tree" --argjson parents "$parents" --arg before "$before" --arg repo "$repo" \
  --argjson pr "$pr" --argjson stamp "$stamp" --argjson m_derived "$m_derived" \
  --argjson expect_shards "$exp" --argjson runs_complete "$runs_complete" --argjson run "$run" --argjson jobs "$jobs" --argjson jobs_total "$jobs_total" --argjson sov_test "$sov_test" \
  --argjson paths "$paths" --argjson now "$now" \
  '{event: $event, ref: $ref, sha: $sha, tree: $tree, parents: $parents, before: $before, repo: $repo, pr: $pr,
    stamp: $stamp, m_derived: $m_derived, expect_shards: $expect_shards, runs_complete: $runs_complete,
    run: ($run + {jobs: $jobs, jobs_total: $jobs_total, sov_test: $sov_test}), paths: $paths, now: $now}' || die "facts could not be assembled"
