#!/usr/bin/env bash
# arm_pr_automerge.sh -- arm auto-merge ONLY when guard-tree is green on the pull
# request's CURRENT head. #3292, APR-RELEASE-001 (revised 2026-09-16), "arming
# precondition".
#
# THE DEFECT, measured 2026-09-16. #3278 was armed while its guard-tree was RED.
# The merge queue took it, put it at POSITION 1, and when it failed there it
# ejected the batch queued behind it. An arming decision is not a local one: a PR
# armed on a red guard costs every PR behind it a rebuild, and the queue is deepest
# exactly when that hurts most.
#
# WHY THIS IS A SEPARATE SCRIPT FROM pr_review_quorum_arm.sh. That script was
# searched for first and it IS an arming path, but a different one: it evaluates
# PR-REVIEW-SKILL-002 S13's review-quorum predicate (classes Q1..Q10) and REFUSES
# with Q1 when there is no signed review receipt, so it cannot be the general
# arming entry point. It also reads `statusCheckRollup`, which is keyed by check
# NAME -- see THE STALE-CHECK TRAP below for why that is the wrong surface for this
# question. The two compose: a quorum PERMIT still has to clear this precondition,
# and this precondition says nothing about review.
#
# THE STALE-CHECK TRAP IS THE WHOLE POINT. "guard-tree is green on this PR" is a
# claim about a SHA, not about a PR. A PR whose guard-tree passed two pushes ago
# and has not been re-run is exactly as unsafe as one that failed, and every
# name-keyed surface (`gh pr checks`, `statusCheckRollup`) will happily show the
# older success next to the newer head. So the head sha is read FIRST, the checks
# are read from `repos/{owner}/{repo}/commits/<that sha>/check-runs` -- an endpoint
# that cannot answer about another commit -- and every row is filtered on
# `head_sha == <that sha>` again anyway. A guard-tree success that exists only on
# an older sha is reported as the named state `stale`, not silently as `absent`,
# because the two want different fixes (push again vs re-run).
#
# ABSENT IS A REFUSAL, NOT A PASS. A missing guard-tree is the fail-open shape this
# repository keeps shipping: no row, no failure, arm. Here it is `absent` and it
# exits 3. In-progress refuses too -- "not yet red" is not green.
#
# Usage
#   bash scripts/arm_pr_automerge.sh 3278                  arm, or refuse and exit 3
#   bash scripts/arm_pr_automerge.sh 3278 --dry-run        decide, arm nothing
#   bash scripts/arm_pr_automerge.sh --self-test           the case table, no network
#   bash scripts/arm_pr_automerge.sh --help
#
# Exit codes
#   0  armed (or --dry-run would arm)
#   2  ENV: bad usage, gh missing, the pull request or its checks unreadable
#   3  REFUSED: guard-tree is not green on the current head
set -uo pipefail

REPO_DEFAULT="paiml/aprender"
GUARD_CHECK_NAME="${GUARD_CHECK_NAME:-guard-tree}"

usage() {
    cat <<'USAGE'
arm_pr_automerge.sh -- arm auto-merge only when guard-tree is green on the CURRENT head.

  PR_NUMBER             the pull request to arm
  --repo OWNER/NAME     the repository (default paiml/aprender)
  --dry-run             print the decision; never arm
  --self-test           run the committed case table (fixtures, no network)
  --help

  exit 0 armed, 2 ENV, 3 refused.
USAGE
}

# guard_tree_verdict SHA CHECKRUNS_FILE -> "ARM <why>" or "REFUSE <state> -- <why>",
# where <state> is one of: absent, stale, queued, in_progress, failure, cancelled,
# timed_out, skipped, neutral, action_required, unreadable.
#
# Pure: a sha and a check-runs payload in, a verdict out. The network lives in the
# caller, so every row below is pinned against a fixture and the live path runs the
# same evaluator (one predicate, two producers).
guard_tree_verdict() {
    local sha="${1:-}" f="${2:-}" name="${GUARD_CHECK_NAME:-guard-tree}"
    local total on_sha elsewhere pending concl
    if [ -z "$sha" ]; then
        printf 'REFUSE unreadable -- no head sha was given; the claim is about a sha, never about a PR\n'
        return 0
    fi
    if [ -z "$f" ] || [ ! -r "$f" ]; then
        printf 'REFUSE unreadable -- the check runs for %s could not be read\n' "$sha"
        return 0
    fi
    total=$(jq '[.check_runs[]?] | length' "$f" 2>/dev/null) || total=""
    case "$total" in ''|*[!0-9]*)
        printf 'REFUSE unreadable -- the check-runs payload is not the shape the API returns\n'
        return 0 ;;
    esac
    on_sha=$(jq --arg n "$name" --arg s "$sha" \
        '[.check_runs[]? | select(.name == $n) | select(.head_sha == $s)] | length' "$f" 2>/dev/null) || on_sha=""
    elsewhere=$(jq --arg n "$name" --arg s "$sha" \
        '[.check_runs[]? | select(.name == $n) | select(.head_sha != $s)] | length' "$f" 2>/dev/null) || elsewhere=""
    case "$on_sha$elsewhere" in ''|*[!0-9]*)
        printf 'REFUSE unreadable -- the check-runs payload could not be queried\n'
        return 0 ;;
    esac
    if [ "$on_sha" -eq 0 ]; then
        # THE TRAP, NAMED. A success that exists only on another commit is `stale`,
        # and it is a refusal: the head it passed on is not the head that will merge.
        if [ "$elsewhere" -gt 0 ]; then
            printf 'REFUSE stale -- %s ran on %s other commit(s) but not on this head\n' "$name" "$elsewhere"
            return 0
        fi
        printf 'REFUSE absent -- no %s check run exists on this head; a missing guard is not a green one\n' "$name"
        return 0
    fi
    # A re-run in flight beside an older success on the SAME sha still refuses:
    # "not yet red" is not green, and the re-run exists because something changed.
    pending=$(jq -r --arg n "$name" --arg s "$sha" \
        '[.check_runs[]? | select(.name == $n) | select(.head_sha == $s)
          | select(.status != "completed") | .status] | first // ""' "$f" 2>/dev/null) || pending=""
    if [ -n "$pending" ]; then
        printf 'REFUSE %s -- %s has not finished on this head\n' "$pending" "$name"
        return 0
    fi
    concl=$(jq -r --arg n "$name" --arg s "$sha" \
        '[.check_runs[]? | select(.name == $n) | select(.head_sha == $s)]
         | sort_by(.completed_at // "") | last | (.conclusion // "")' "$f" 2>/dev/null) || concl=""
    if [ "$concl" != "success" ]; then
        printf 'REFUSE %s -- %s concluded %s on this head\n' \
            "${concl:-unreadable}" "$name" "${concl:-<none>}"
        return 0
    fi
    printf 'ARM %s is success on this head (%s run(s) on it)\n' "$name" "$on_sha"
}

# refusal_line SHA VERDICT -> the one line a refusal prints. The state AND the sha
# are both in it: "refused" with neither is undiagnosable, and an undiagnosable
# refusal gets routed around.
refusal_line() {
    local sha="${1:-}" verdict="${2:-}" state
    state="$( printf '%s' "$verdict" | cut -d' ' -f2 )"
    printf 'refuse: %s is %s on %s\n' "$GUARD_CHECK_NAME" "$state" "$sha"
}

# fetch_check_runs REPO SHA OUTFILE -> 0 on success. The endpoint is keyed on the
# COMMIT, so it cannot answer about another one. `--paginate -q` yields one object
# per line across pages; jq -s puts them back into the single document the
# predicate reads. The rc is taken from gh directly, never through a pipe.
fetch_check_runs() {
    local repo="$1" sha="$2" out="$3" nd rc
    nd="$out.ndjson"
    gh api "repos/$repo/commits/$sha/check-runs?per_page=100" --paginate \
        -q '.check_runs[]' > "$nd" 2>/dev/null
    rc=$?
    [ "$rc" -eq 0 ] || return "$rc"
    jq -s '{check_runs: .}' "$nd" > "$out" 2>/dev/null || return 1
    return 0
}

self_test() {
    local fails=0 rows=0 td sha_new sha_old
    td="$(mktemp -d)" || return 2
    sha_new='eb883a2f2d52f50e5436b22a926c7b0087817a6d'   # #3278's head, read 2026-09-16
    sha_old='1111111111111111111111111111111111111111'

    _eq() { # _eq LABEL WANT GOT
        rows=$(( rows + 1 ))
        if [ "$3" = "$2" ]; then printf 'ok    %s\n' "$1"
        else printf 'FAIL  %s: got "%s", wanted "%s"\n' "$1" "$3" "$2"; fails=1; fi
    }

    # THE FIXTURES ARE THE MEASURED SHAPE. Taken from
    # repos/paiml/aprender/commits/eb883a2f.../check-runs at 2026-09-16, where
    # guard-tree read status=in_progress, conclusion=null, head_sha=eb883a2f... --
    # row A3 is that payload in the fields the predicate reads. Every other row is
    # the same document with ONE field moved, so a row differs from its neighbour by
    # exactly the thing it claims to test.
    _mk() { # _mk FILE STATUS CONCLUSION HEAD_SHA
        jq -n --arg st "$2" --arg cc "$3" --arg sh "$4" '
            {check_runs: [
                {name: "ci / lint", status: "completed", conclusion: "success", head_sha: $sh,
                 started_at: "2026-09-16T14:20:00Z", completed_at: "2026-09-16T14:24:00Z"},
                {name: "guard-tree", status: $st,
                 conclusion: (if $cc == "" then null else $cc end), head_sha: $sh,
                 started_at: "2026-09-16T14:26:47Z",
                 completed_at: (if $st == "completed" then "2026-09-16T14:40:00Z" else null end)},
                {name: "workspace-test", status: "completed", conclusion: "success", head_sha: $sh,
                 started_at: "2026-09-16T14:20:00Z",
                 completed_at: "2026-09-16T14:55:00Z"}]}' > "$1"
    }

    _mk "$td/green.json"      completed   success "$sha_new"
    _mk "$td/failure.json"    completed   failure "$sha_new"
    _mk "$td/inprogress.json" in_progress ''      "$sha_new"
    _mk "$td/queued.json"     queued      ''      "$sha_new"
    _mk "$td/skipped.json"    completed   skipped "$sha_new"
    _mk "$td/staleonly.json"  completed   success "$sha_old"
    jq '{check_runs: [.check_runs[] | select(.name != "guard-tree")]}' \
        "$td/green.json" > "$td/absent.json"
    # A re-run in flight on the SAME sha, beside the earlier success.
    jq --arg s "$sha_new" '.check_runs += [{name: "guard-tree", status: "in_progress",
        conclusion: null, head_sha: $s, started_at: "2026-09-16T15:10:00Z",
        completed_at: null}]' "$td/green.json" > "$td/rerun.json"

    # A1 IS THE POSITIVE CONTROL. Without it every row below is satisfied by a
    # function that returns REFUSE unconditionally, and the helper would be a gate
    # that can never let anything through.
    _eq 'A1 guard-tree success ON THIS HEAD -> ARM' \
        'ARM' "$( guard_tree_verdict "$sha_new" "$td/green.json" | head -1 | cut -d' ' -f1 )"
    _eq 'A2 guard-tree FAILED on this head -> refuse (state failure)' \
        'REFUSE failure' "$( guard_tree_verdict "$sha_new" "$td/failure.json" | head -1 | cut -d' ' -f1,2 )"
    # A3 IS THE MEASURED PAYLOAD of #3278 at 14:26Z. "Not yet red" is not green.
    _eq 'A3 guard-tree in_progress (the live #3278 shape) -> refuse' \
        'REFUSE in_progress' "$( guard_tree_verdict "$sha_new" "$td/inprogress.json" | head -1 | cut -d' ' -f1,2 )"
    _eq 'A4 guard-tree queued -> refuse' \
        'REFUSE queued' "$( guard_tree_verdict "$sha_new" "$td/queued.json" | head -1 | cut -d' ' -f1,2 )"
    # A5: the fail-open shape. No row, no failure -- and a name-keyed reading calls
    # that "nothing red".
    _eq 'A5 no guard-tree check run at all -> refuse (absent is not green)' \
        'REFUSE absent' "$( guard_tree_verdict "$sha_new" "$td/absent.json" | head -1 | cut -d' ' -f1,2 )"
    # A6 IS THE STALE-CHECK TRAP, and it is the row that separates this helper from
    # every name-keyed surface: guard-tree is SUCCESS, it is just success on a
    # commit that is not the head that will merge.
    _eq 'A6 success only on an OLDER sha -> refuse (state stale, not absent)' \
        'REFUSE stale' "$( guard_tree_verdict "$sha_new" "$td/staleonly.json" | head -1 | cut -d' ' -f1,2 )"
    _eq 'A7 a re-run in flight beside an earlier success on the same sha -> refuse' \
        'REFUSE in_progress' "$( guard_tree_verdict "$sha_new" "$td/rerun.json" | head -1 | cut -d' ' -f1,2 )"
    # `skipped` is a conclusion, and it is not `success`. A gate that reads
    # "completed and not failure" arms on it.
    _eq 'A8 guard-tree skipped -> refuse (skipped is a conclusion, not a pass)' \
        'REFUSE skipped' "$( guard_tree_verdict "$sha_new" "$td/skipped.json" | head -1 | cut -d' ' -f1,2 )"
    _eq 'A9 an unreadable payload -> refuse, never arm' \
        'REFUSE unreadable' "$( guard_tree_verdict "$sha_new" "$td/does-not-exist.json" | head -1 | cut -d' ' -f1,2 )"
    _eq 'A10 no head sha -> refuse (the claim is about a sha)' \
        'REFUSE unreadable' "$( guard_tree_verdict '' "$td/green.json" | head -1 | cut -d' ' -f1,2 )"
    _eq 'A11 the refusal line names the state AND the sha' \
        "refuse: guard-tree is failure on $sha_new" \
        "$( refusal_line "$sha_new" "$( guard_tree_verdict "$sha_new" "$td/failure.json" )" )"
    # A12: the fetch normalises `--paginate -q` NDJSON back into one document. The
    # predicate is fed that shape and nothing else, so the shape is a row.
    printf '%s\n' '{"name":"guard-tree","status":"completed","conclusion":"success","head_sha":"'"$sha_new"'","completed_at":"2026-09-16T14:40:00Z"}' > "$td/pages.ndjson"
    jq -s '{check_runs: .}' "$td/pages.ndjson" > "$td/slurped.json"
    _eq 'A12 the paginated NDJSON, slurped, is the document the predicate reads' \
        'ARM' "$( guard_tree_verdict "$sha_new" "$td/slurped.json" | head -1 | cut -d' ' -f1 )"

    rm -rf "${td:?}"
    printf '\n%s row(s), %s\n' "$rows" \
        "$( [ "$fails" -eq 0 ] && echo '0 red / FALSIFIER GREEN' || echo 'RED' )"
    return "$fails"
}

arm() {
    local repo="$1" pr="$2" dry="$3" tmp sha verdict out rc
    command -v gh > /dev/null 2>&1 || { printf 'ENV: gh is not on PATH\n' >&2; return 2; }
    command -v jq > /dev/null 2>&1 || { printf 'ENV: jq is not on PATH\n' >&2; return 2; }
    tmp="$(mktemp -d)" || return 2
    trap 'rm -rf "${tmp:?}"' RETURN

    # THE HEAD SHA IS READ FIRST, AND EVERYTHING AFTER IT IS ABOUT THAT SHA.
    sha=$(gh pr view "$pr" --repo "$repo" --json headRefOid -q '.headRefOid' 2>/dev/null) || sha=""
    if [ -z "$sha" ]; then
        printf 'ENV: could not read the current head of %s#%s\n' "$repo" "$pr" >&2
        return 2
    fi
    if ! fetch_check_runs "$repo" "$sha" "$tmp/checks.json"; then
        printf 'ENV: could not read the check runs for %s\n' "$sha" >&2
        return 2
    fi
    verdict="$( guard_tree_verdict "$sha" "$tmp/checks.json" )"
    case "$verdict" in
        REFUSE*)
            refusal_line "$sha" "$verdict"
            printf '        %s\n' "${verdict#REFUSE }"
            return 3 ;;
    esac
    if [ "$dry" = 1 ]; then
        printf 'WOULD-ARM %s#%s on %s -- %s\n' "$repo" "$pr" "$sha" "${verdict#ARM }"
        return 0
    fi
    out=$(gh pr merge "$pr" --repo "$repo" --squash --auto 2>&1)
    rc=$?
    if [ "$rc" -ne 0 ]; then
        printf 'ENV: arming failed (exit %s): %s\n' \
            "$rc" "$( printf '%s' "$out" | tr '\n' ' ' | cut -c1-200 )" >&2
        return 2
    fi
    printf 'ARMED %s#%s on %s -- %s\n' "$repo" "$pr" "$sha" "${verdict#ARM }"
    return 0
}

MODE=""; DRY=0; REPO="$REPO_DEFAULT"; PR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --self-test|--selftest) MODE=self; shift ;;
        --dry-run)              DRY=1; shift ;;
        --repo)                 REPO="${2:-$REPO_DEFAULT}"; shift 2 ;;
        --help|-h)              usage; exit 0 ;;
        -*) printf 'arm_pr_automerge.sh: unknown argument %s\n' "$1" >&2; usage >&2; exit 2 ;;
        *)  PR="$1"; MODE="${MODE:-arm}"; shift ;;
    esac
done

# A BARE invocation runs the case table, never an arming. The guard tree runs every
# scripts/*.sh that carries a --self-test arm; a helper whose no-argument behaviour
# is to take an irreversible action on a pull request must not be what that reaches.
case "${MODE:-self}" in
    self) self_test; exit $? ;;
    arm)
        case "$PR" in ''|*[!0-9]*)
            printf 'arm_pr_automerge.sh: PR_NUMBER must be a number, got "%s"\n' "$PR" >&2
            usage >&2; exit 2 ;;
        esac
        arm "$REPO" "$PR" "$DRY"; exit $? ;;
    *)    usage >&2; exit 2 ;;
esac
