#!/usr/bin/env bash
# ci_push_reuse_facts.sh - gathers the facts scripts/ci_push_reuse.sh decides on (#7298).
# Git and API reads only; no decision lives here. It prints one JSON object on stdout and exits 0
# only when EVERY fact was read; any failed or partial read prints the reason on stderr and exits 1,
# and the caller then runs the jobs (reuse=no). About 5 REST calls: the PR of the commit, the PR, the
# run list, the job list, the stamp job's log.
#
#   ci_push_reuse_facts.sh --repo OWNER/NAME --sha S --before P --ref REF --event EVT
#
# Needs a clone with full history (the checkout of mg-reuse uses fetch-depth 0), gh, git, jq.
set -uo pipefail
repo="" sha="" before="" ref="" event=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo) repo="${2-}"; shift 2 ;; --sha) sha="${2-}"; shift 2 ;; --before) before="${2-}"; shift 2 ;;
    --ref) ref="${2-}"; shift 2 ;; --event) event="${2-}"; shift 2 ;;
    *) echo "facts: unknown argument $1" >&2; exit 1 ;;
  esac
done
die() { echo "facts: $*" >&2; exit 1; }
for v in repo sha before ref event; do [ -n "${!v}" ] || die "--$v is required"; done
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
pr=$(jq -c --arg repo "$repo" '{number: .number, merged: (.merged == true), merge_commit_sha: .merge_commit_sha,
  is_fork: ((.head.repo.full_name // "") != $repo), head_repo: (.head.repo.full_name // null), draft: (.draft == true), head_sha: .head.sha}' <<< "$prj") || die "PR facts unreadable"

# H's objects: the PR head ref outlives the branch
git cat-file -e "$h^{commit}" 2> /dev/null || git fetch -q --no-tags origin "refs/pull/$n/head" 2> /dev/null || true
git cat-file -e "$h^{commit}" 2> /dev/null || die "PR head $h not fetchable"
head_tree=$(git rev-parse "$h^{tree}") || die "head tree unreadable"
git merge-base --is-ancestor "$b" "$h" 2> /dev/null; arc=$?
case "$arc" in 0) base_in_head=true ;; 1) base_in_head=false ;; *) die "ancestor test failed (rc $arc)" ;; esac

# H's ci.yml pull_request runs (1 call): the newest by id, and the newest success to cite
runs=$(gh api "repos/$repo/actions/workflows/ci.yml/runs?head_sha=$h&event=pull_request&per_page=100" 2> /dev/null) || die "run list read failed"
runs_complete=$(jq '(.total_count // -1) == ([.workflow_runs[]?] | length)' <<< "$runs") || die "run list unreadable"
newest=$(jq -r '[.workflow_runs[]? | select(.head_sha == "'"$h"'")] | map(.id) | max // empty' <<< "$runs")
cite=$(jq -r '[.workflow_runs[]? | select(.head_sha == "'"$h"'" and .conclusion == "success")] | map(.id) | max // empty' <<< "$runs")
[ -n "$newest" ] || die "no pull_request run on $h"
rid="${cite:-$newest}"
run=$(jq -c --argjson id "$rid" --argjson nw "$newest" '.workflow_runs[] | select(.id == $id)
  | {id: .id, event: .event, path: .path, head_sha: .head_sha, status: .status, conclusion: .conclusion, newest_id: $nw,
     started_epoch: (.run_started_at | fromdateiso8601),
     pull_requests: [(.pull_requests // [])[] | {number: .number, base_ref: .base.ref}]}' <<< "$runs") || die "run object unreadable"

jobsj=$(gh api "repos/$repo/actions/runs/$rid/jobs?per_page=100&filter=latest" 2> /dev/null) || die "job list read failed"
jobs=$(jq -c '[.jobs[]? | {name: .name, conclusion: .conclusion, run_id: .run_id, head_sha: .head_sha}]' <<< "$jobsj") || die "job list unreadable"
jobs_total=$(jq '.total_count // -1' <<< "$jobsj") || die "job count unreadable"

# the stamp the PR run's workspace-test-ran job wrote (1 call); absent = null (path b only)
stamp=null; m_derived=null
wtid=$(jq -r '[.jobs[]? | select(.name == "workspace-test-ran")] | if length == 1 then .[0].id else empty end' <<< "$jobsj")
if [ -n "$wtid" ]; then
  # The log is read whole (a failed read is a refusal: a stamp that might contradict path b must never be skipped),
  # then matched with here-strings, never a pipe. Exactly ONE stamp line is accepted, and none when the same log
  # also carries the unavailable line, so a line planted before the real one can never win.
  log=$(gh api "repos/$repo/actions/jobs/$wtid/logs" 2> /dev/null) || die "stamp job log read failed"
  stamps=$(grep -Eo 'PUSH-REUSE-STAMP tested_sha=[0-9a-f]{40} tested_tree=[0-9a-f]{40} tested_base=[0-9a-f]{40} tested_head=[0-9a-f]{40}' <<< "$log") || stamps=""
  nstamps=$(grep -c . <<< "$stamps") || nstamps=0
  [ "$nstamps" -le 1 ] || die "the stamp job log has $nstamps stamp lines"
  if [ "$nstamps" -eq 1 ] && grep -qF 'PUSH-REUSE-STAMP unavailable' <<< "$log"; then die "the stamp job log has a stamp and an unavailable line"; fi
  line="$stamps"
  if [ -n "$line" ]; then
    stamp=$(printf '%s\n' "$line" | sed -E 's/.*tested_sha=([0-9a-f]+) tested_tree=([0-9a-f]+) tested_base=([0-9a-f]+) tested_head=([0-9a-f]+)/\1 \2 \3 \4/' \
      | jq -R -c 'split(" ") | {sha: .[0], tree: .[1], base: .[2], head: .[3]}') || stamp=null
    ms=$(jq -r '.sha // empty' <<< "$stamp")
    if [ -n "$ms" ] && { git cat-file -e "$ms^{commit}" 2> /dev/null || git fetch -q --no-tags origin "$ms" 2> /dev/null; } && git cat-file -e "$ms^{commit}" 2> /dev/null; then
      m_derived=$(jq -n -c --arg t "$(git rev-parse "$ms^{tree}")" --arg b "$(git rev-parse "$ms^1" 2> /dev/null)" --arg h "$(git rev-parse "$ms^2" 2> /dev/null)" '{tree: $t, base: $b, head: $h}') || m_derived=null
    fi
  fi
fi

# B..S changed paths; the push matrix of S. Whether the CITED RUN reused a receipt is
# read by the decision from that run's own job list (C4-not-a-reuse), never from workflow text.
paths=$(git diff --no-renames --name-only "$b" "$sha" | jq -R -s -c 'split("\n") | map(select(length > 0))') || die "diff unreadable"
wf=$(git show "$sha:.github/workflows/ci.yml" 2> /dev/null) || die "ci.yml of $sha unreadable"
exp=$(printf '%s\n' "$wf" | sed -n "s/^[[:space:]]*shards: .*|| '\[\([0-9][0-9]*\)\]').*/\1/p" | head -n 1)
[[ $exp =~ ^[0-9]+$ ]] || die "push shard count not readable from ci.yml"

jq -n -c --arg event "$event" --arg ref "$ref" --arg sha "$sha" --arg tree "$tree" --argjson parents "$parents" --arg before "$before" --arg repo "$repo" \
  --argjson pr "$pr" --arg head_tree "$head_tree" --argjson base_in_head "$base_in_head" --argjson stamp "$stamp" --argjson m_derived "$m_derived" \
  --argjson expect_shards "$exp" --argjson runs_complete "$runs_complete" --argjson run "$run" --argjson jobs "$jobs" --argjson jobs_total "$jobs_total" \
  --argjson paths "$paths" --argjson now "$(date +%s)" \
  '{event: $event, ref: $ref, sha: $sha, tree: $tree, parents: $parents, before: $before, repo: $repo, pr: $pr, head_tree: $head_tree,
    base_in_head: $base_in_head, stamp: $stamp, m_derived: $m_derived, expect_shards: $expect_shards, runs_complete: $runs_complete,
    run: ($run + {jobs: $jobs, jobs_total: $jobs_total}), paths: $paths, now: $now}' || die "facts could not be assembled"
