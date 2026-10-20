#!/usr/bin/env bash
# advisory_gate.sh -- the advisories step of guard-cargo (PMAT-5792, contract advisory-gate-diff-v1).
#
#   pull_request / merge_group  fail only on advisories the PR INTRODUCES against the pinned build base
#                               (refs/remotes/origin/main, pinned by pin_build_base.sh); an advisory
#                               already on the base is reported, never a block.
#   any other event (push)      the full `cargo deny check advisories`, as before: a red there is the
#                               main-resident andon, not a PR block.
# Fails closed: anything that cannot be checked exits 75 and fails the step. Both sides are scanned with the
# BASE deny.toml, from the BASE worktree's script, over ONE advisory-db fetched once by this job.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EV="${GITHUB_EVENT_NAME:-}"
case "$EV" in
    pull_request | merge_group) ;;
    *) exec cargo deny check advisories ;;
esac

base_sha() { # the build base the pin step named
    git rev-parse --verify -q refs/remotes/origin/main^{commit} || { echo "COULD-NOT-CHECK: origin/main is not pinned" >&2; return 75; }
}
BASE=$(base_sha) || exit 75
# merge_group: the pin must equal merge_group.base_sha; a drifted pin is refused, not guessed around.
if [ "$EV" = merge_group ] && [ -n "${MG_BASE_SHA:-}" ] && [ "$BASE" != "$MG_BASE_SHA" ]; then
    echo "COULD-NOT-CHECK: pinned base $BASE != merge_group base_sha $MG_BASE_SHA" >&2
    exit 75
fi
bash "$HERE/advisory_diff.sh" base-ref "$EV" "${MG_BASE_SHA:-}" "$BASE" >/dev/null || exit 75

W="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/advgate-base-$$"
trap 'git worktree remove --force "${W:?}" >/dev/null 2>&1 || true; rm -f "${W:?}"/../advgate-pr-$$.ndjson "${W:?}"/../advgate-base-$$.ndjson' EXIT
git worktree add --detach "$W" "$BASE" >/dev/null 2>&1 || { echo "COULD-NOT-CHECK: no worktree for base $BASE (shallow fetch?)" >&2; exit 75; }

# One DB for both scans: fetched with the BASE config (the db-path both scans read), HEAD logged.
cargo deny fetch --config "$W/deny.toml" db || { echo "COULD-NOT-CHECK: advisory-db fetch failed" >&2; exit 75; }
for d in "$HOME"/.cargo/advisory-dbs/*; do
    [ -d "$d/.git" ] && echo "advisory-db $(basename "$d") HEAD $(git -C "$d" rev-parse --short HEAD 2>/dev/null) $(git -C "$d" log -1 --format=%cI 2>/dev/null)"
done

# The BASE script when it has one; the PR's copy only to bootstrap the merge that introduces the script.
GATE="$W/scripts/ci/advisory_diff.sh"
[ -f "$GATE" ] || GATE="$HERE/advisory_diff.sh"

bash "$GATE" ignores-added "$W/deny.toml" deny.toml || exit 1

scan() { # TREE OUT.ndjson -> prints the cargo-deny rc
    local rc=0
    (cd "$1" && cargo deny --locked --all-features --format json check --config "$W/deny.toml" --disable-fetch advisories) \
        >/dev/null 2>"$2" || rc=$?
    echo "$rc"
}
PRRC=$(scan "$PWD" "$W/../advgate-pr-$$.ndjson")
BASERC=$(scan "$W" "$W/../advgate-base-$$.ndjson")
bash "$GATE" verdict "$W/../advgate-pr-$$.ndjson" "$PRRC" "$W/../advgate-base-$$.ndjson" "$BASERC"
