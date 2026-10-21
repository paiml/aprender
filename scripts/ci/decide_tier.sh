#!/usr/bin/env bash
# decide_tier.sh -- which test tier this run owes (BSE-17, PMAT-1077); the one body of the
# "Decide the test tier" step, shared by the workspace-test-shard section and by the
# workspace-test-arch job that places the shards (a full tier needs ONE architecture: the
# cross-shard Σ compares every shard with one universe list). Env: GITHUB_EVENT_NAME,
# GITHUB_SHA, GITHUB_REPOSITORY, GITHUB_BASE_REF, MG_BASE_SHA, GH_TOKEN, RUNNER_TEMP, GITHUB_OUTPUT.
set -euo pipefail
args=(--event "${GITHUB_EVENT_NAME}")
# PMAT-1098 67-E2: the queue and a push to main re-derive their own
# diff from HEAD's FIRST PARENT (HEAD^1..HEAD is the PR's diff on the
# new base). actions/checkout is shallow, so that parent is not in the
# object store until we deepen -- and without it ci_test_tier.sh falls
# closed to `full` every single time, which would make the whole
# optimisation inert while looking like it worked. Best effort by
# design: if the deepen fails the decision still falls closed, and the
# warning below says so instead of the run silently paying the hour.
case "${GITHUB_EVENT_NAME}" in
  merge_group)
    # Operator 2026-10-17: the queue runs the quick tier over the entry's diff (base_sha = prior entry's head),
    # MG_BASE_SHA..HEAD. Fetch the base; if that fails the tier falls closed to full.
    bash scripts/ci/fetch_p6.sh --no-tags --depth=1 origin "${GITHUB_SHA}" "${MG_BASE_SHA}" 2>/dev/null \
      || bash scripts/ci/fetch_p6.sh --no-tags --depth=1 origin "${MG_BASE_SHA}" 2>/dev/null \
      || true
    git cat-file -e "${MG_BASE_SHA}^{commit}" 2>/dev/null \
      || printf '::warning::merge_group base sha unavailable -- the tier falls closed to full\n'
    ;;
  push)
    bash scripts/ci/fetch_p6.sh --no-tags --depth=2 origin "${GITHUB_SHA}" 2>/dev/null \
      || bash scripts/ci/fetch_p6.sh --no-tags --deepen=1 origin 2>/dev/null \
      || true
    if git rev-parse -q --verify 'HEAD^1' >/dev/null 2>&1; then
      printf 'first parent available: %s\n' "$(git rev-parse HEAD^1)"
    else
      printf '::warning::first parent unavailable after deepen -- the tier falls closed to full\n'
    fi
    ;;
esac
case "${GITHUB_EVENT_NAME}" in
  pull_request)
    bash scripts/ci/fetch_p6.sh --no-tags --depth=1 origin "+refs/heads/${GITHUB_BASE_REF}:refs/remotes/origin/${GITHUB_BASE_REF}"
    # #5391: the PR's change is the diff from merge-base(base, HEAD), not the
    # base tip -- main's later commits are not the PR's. --no-renames (#3664)
    # lives in the helper.
    bash scripts/ci/pr_touched_paths.sh "origin/${GITHUB_BASE_REF}" HEAD > "$RUNNER_TEMP/touched.txt"
    args+=(--diff-from "$RUNNER_TEMP/touched.txt")
    ;;
  merge_group)
    # Operator 2026-10-17 (cop window), verbatim: "only gating on tags/release".
    # ci_test_tier.sh diffs MG_BASE_SHA..HEAD (env above) and selects like a PR.
    ;;
  push)
    # T36: the merge queue already tested this very sha. Hand its
    # workspace-test result to the tier script, which reuses it only
    # on a success for HEAD itself. A failed lookup is the push's own
    # tier, as before -- never a reuse, never a red step.
    mg=$(bash scripts/ci_mg_workspace_result.sh "${GITHUB_REPOSITORY}" "${GITHUB_SHA}") || mg=""
    if [ -n "$mg" ]; then
      mapfile -t mgargs <<<"$mg"
      args+=("${mgargs[@]}")
    else
      printf 'no merge_group workspace-test result for %s: the tier is decided from the push diff\n' "${GITHUB_SHA}"
    fi
    ;;
esac
bash scripts/ci_test_tier.sh "${args[@]}" | tee "$RUNNER_TEMP/tier.txt"
grep -E '^(tier|crates|targets|check_workspace|reason|cite)=' "$RUNNER_TEMP/tier.txt" >> "$GITHUB_OUTPUT"
