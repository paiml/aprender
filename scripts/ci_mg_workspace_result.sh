#!/usr/bin/env bash
# ci_mg_workspace_result.sh - the merge queue's workspace-test result for one commit (T36).
#
# The merge queue tests a commit, then main is moved to that very commit and the push run starts.
# The push run's tier step calls this to ask GitHub what the queue already measured, and hands the
# answer to scripts/ci_test_tier.sh, which reuses it only when it is a success for HEAD itself.
#
#   ci_mg_workspace_result.sh <owner/repo> <sha>
#
# Prints, one per line, the arguments for ci_test_tier.sh:
#   --mg-run <actions run id>  --mg-sha <the sha GitHub reports>  --mg-conclusion <conclusion>
# taken from the LATEST completed `workspace-test` check run whose check suite is the GitHub
# Actions suite of a merge-queue branch (gh-readonly-queue/...). It reports what it found, a
# failure included; judging it is ci_test_tier.sh's job.
#
# Three API calls: two on the checks endpoints (the ones the merge_group arm already reads) and, for a success, one
# on the found run's jobs. A queue run that SKIPPED its workspace-test shards (it cited a PR run, mg-reuse-wt #5884)
# also reports a green `workspace-test`; reusing that on the push would cite a citation, so the lookup refuses
# unless every `workspace-test-shard (N, M)` job of that run succeeded.
# Anything that is not a clean answer prints NOTHING on stdout, one reason on stderr, and exits 1:
# the caller then decides the tier from the push's own diff, as before T36.
#
# Every refusal is ONE line ending in a `# R-<NAME>` marker, so scripts/check_ci_push_reuse.sh can
# delete it and prove its table turns red without it.
set -uo pipefail

repo=${1:-}; sha=${2:-}
no() { echo "ci_mg_workspace_result: $*" >&2; exit 1; }

[[ $repo =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || no "'$repo' is not owner/repo" # R-REPO
[[ $sha =~ ^[0-9a-f]{40}$ ]] || no "'$sha' is not a 40-hex commit sha" # R-SHAFORM

# Call 1: the merge-queue suites on this commit. A push suite on the same sha is not a measurement
# the queue made, and a suite from another app is not GitHub Actions.
SUITES_JQ='.check_suites[] | select(.app.slug == "github-actions") | select((.head_branch // "") | startswith("gh-readonly-queue/")) | .id'
suites=$(gh api "repos/$repo/commits/$sha/check-suites?per_page=100" --jq "$SUITES_JQ"); rc=$?
[ "$rc" -eq 0 ] || no "the check-suites lookup failed for $sha" # R-SUITESAPI
[ "${#suites}" -gt 0 ] || no "no merge-queue check suite on $sha" # R-NOQUEUE

# Call 2: the completed workspace-test check runs on this commit, latest first.
RUNS_JQ='[.check_runs[] | select(.status == "completed")] | sort_by(.completed_at) | reverse | .[] | "\(.check_suite.id) \(.head_sha) \(.conclusion) \(.details_url)"'
runs=$(gh api "repos/$repo/commits/$sha/check-runs?check_name=workspace-test&per_page=100" --jq "$RUNS_JQ"); rc=$?
[ "$rc" -eq 0 ] || no "the check-runs lookup failed for $sha" # R-RUNSAPI

found=""
while read -r sid hsha concl url; do
    [ -n "${sid:-}" ] || continue
    grep -qx -- "$sid" <<<"$suites" || continue # R-QUEUESUITE
    found="$hsha $concl $url"; break
done <<<"$runs"
[ -n "$found" ] || no "no completed merge-queue workspace-test on $sha" # R-NORUN

read -r hsha concl url <<<"$found"
[[ $hsha =~ ^[0-9a-f]{40}$ ]] || no "the workspace-test check run reports no sha" # R-HEADSHA
[[ $url =~ /actions/runs/([0-9]+)(/|$) ]] || no "the workspace-test details_url names no run id: $url" # R-RUNID
rid=${BASH_REMATCH[1]}

# Call 3 (a success only): the shards of that run. A skipped, failed or absent shard means the green
# `workspace-test` was a citation of another run (or tested nothing), never this tree's own measurement.
if [ "$concl" = success ]; then
    jobs=$(gh api "repos/$repo/actions/runs/$rid/jobs?per_page=100&filter=latest"); rc=$?
    [ "$rc" -eq 0 ] || no "the jobs lookup failed for run $rid" # R-JOBSAPI
    shards=$(jq -r '[.jobs[]? | select(.name | test("^workspace-test-shard \\([0-9]+, [0-9]+\\)$"))] | if length == 0 then "none" elif all(.conclusion == "success") then "ok" else "no" end' <<<"$jobs" 2>/dev/null)
    [ "$shards" = ok ] || no "run $rid did not run and pass its own workspace-test shards (${shards:-unreadable}): its green verdict may be a citation" # R-SHARDS
fi
printf -- '--mg-run\n%s\n--mg-sha\n%s\n--mg-conclusion\n%s\n' "$rid" "$hsha" "$concl"
