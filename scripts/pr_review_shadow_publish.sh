#!/usr/bin/env bash
# pr_review_shadow_publish.sh - put the quorum verdict ON THE PULL REQUEST, as an
# idempotent marked comment.
#
# OPERATOR INSTRUCTION, 2026-09-01: "confirm that these quorum reviews are always added
# TO THE PULL REQUEST itself as metadata". A verdict that exists only in a job log is a
# verdict nobody reads: logs expire (this repository has already lost a diagnosis that
# way - the check-run annotation was the only surviving truth), and a reviewer looking
# at a pull request cannot see a step summary from a job they did not open. So the
# sample is written where the decision is made.
#
# IDEMPOTENT BY MARKER, NOT BY POSITION. A push updates the SAME comment rather than
# adding one, because a lane that appends is a lane people mute, and a muted lane is an
# unread lane. `gh pr comment --edit-last` was rejected for this: it edits the last
# comment by this actor whatever that comment is, so a concurrent lane's post becomes
# the thing this one overwrites. The marker is an HTML comment, invisible when rendered
# and exact when matched.
#
# IT CANNOT MERGE. The job invoking this holds `contents: read`, which is what
# `gh pr merge` needs and does not have; `pull-requests: write` buys a comment and
# nothing else. The capability is absent, not merely unused.
#
# EXIT
#   0  the comment is on the pull request (created or updated)
#   1  it is not, and that is loud: a publisher that silently fails to publish leaves
#      the verdict exactly as invisible as having no publisher at all.
#
# USAGE
#   pr_review_shadow_publish.sh --pr <N> --body-file <f> [--marker <m>]
#   pr_review_shadow_publish.sh --self-test
#
# ENVIRONMENT
#   PR_REVIEW_GH  the gh binary (default: gh). Pointed at a stub in --self-test; no
#                 value of it turns publishing off - an absent gh is exit 1, not a skip.

set -euo pipefail

PROG=${0##*/}
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
GH=${PR_REVIEW_GH:-gh}
MARKER_DEFAULT='<!-- s13-shadow-verdict -->'

ST_ROOT=''
safe_rm_scratch() {
    local victim=${1:-} must=${2:-}
    [ -n "$victim" ] || return 0
    [ -n "$must" ]   || return 0
    [ "$victim" != "/" ] || return 0
    case "$victim" in
      *"$must"*) if [ -n "$victim" ] && [ "$victim" != "/" ]; then rm -rf -- "$victim"; fi ;;
      *) return 0 ;;
    esac
}
cleanup_self_test() { safe_rm_scratch "$ST_ROOT" 'shadowpub-selftest.'; }
trap cleanup_self_test EXIT

fail() { echo "$PROG: FAIL - $*" >&2; exit 1; }

publish() {
    local pr=$1 body_file=$2 marker=$3
    local existing='' rc=0 body

    [ -f "$body_file" ] || fail "body file not found: $body_file"
    command -v "$GH" >/dev/null 2>&1 || [ -x "$GH" ] || fail "gh not available at '$GH'; a publisher that cannot publish is exit 1, never a skip"

    body="$marker"$'\n'"$(cat "$body_file")"

    # Find a comment already carrying the marker. Status read straight off the
    # command, never through a pipe (Verification Discipline #1).
    set +e
    existing=$("$GH" api "repos/{owner}/{repo}/issues/$pr/comments" \
                 --jq "[.[] | select(.body != null) | select(.body | contains(\"$marker\")) | .id] | first // empty" 2>/dev/null)
    rc=$?
    set -e
    [ "$rc" -eq 0 ] || fail "could not list comments on #$pr (gh rc $rc)"

    if [ -n "$existing" ]; then
        set +e
        "$GH" api --method PATCH "repos/{owner}/{repo}/issues/comments/$existing" \
              -f body="$body" >/dev/null 2>&1
        rc=$?
        set -e
        [ "$rc" -eq 0 ] || fail "could not update comment $existing on #$pr (gh rc $rc)"
        echo "updated existing comment $existing on #$pr"
    else
        set +e
        "$GH" api --method POST "repos/{owner}/{repo}/issues/$pr/comments" \
              -f body="$body" >/dev/null 2>&1
        rc=$?
        set -e
        [ "$rc" -eq 0 ] || fail "could not create a comment on #$pr (gh rc $rc)"
        echo "created a comment on #$pr"
    fi
}

self_test() {
    local fails=0 pass_n=0 td
    td=$(mktemp -d -t shadowpub-selftest.XXXXXX) || fail "mktemp"
    ST_ROOT=$td
    echo "$td/calls.log" > "$td/logpath"
    printf 'S13-SHADOW pr=1 head=abc verdict=REFUSE class=Q1 arm_rc=1\n' > "$td/body.txt"

    mk_gh() { # <name> <list-output> <mutate-rc>
        local f="$td/$1"
        {
            echo '#!/usr/bin/env bash'
            echo "echo \"\$*\" >> \"$td/calls.log\""
            echo 'if [ "$1" = api ] && [ "${2:-}" != --method ]; then'
            printf '  cat <<%s\n%s\n%s\n' "'IDS'" "$2" "IDS"
            echo '  exit 0'
            echo 'fi'
            echo "exit $3"
        } > "$f"; chmod +x "$f"; printf '%s' "$f"
    }

    row() { # desc expect gh-stub [expect-verb] [body-file]
        local desc=$1 expect=$2 stub=$3 verb=${4:-} bf=${5:-$td/body.txt}
        local out rc=0
        : > "$td/calls.log"
        set +e
        out=$(PR_REVIEW_GH="$stub" bash "$HERE/$PROG" --pr 1 --body-file "$bf" 2>&1)
        rc=$?
        set -e
        local ok=0
        if [ "$expect" = PASS ] && [ "$rc" -eq 0 ]; then ok=1; fi
        if [ "$expect" = FAIL ] && [ "$rc" -ne 0 ]; then ok=1; fi
        if [ "$ok" -eq 1 ] && [ -n "$verb" ]; then
            grep -q -- "$verb" "$td/calls.log" || ok=0
        fi
        if [ "$ok" -eq 1 ]; then printf 'ok   %s\n' "$desc"; pass_n=$((pass_n+1))
        else printf 'FAIL %s (expected %s, rc=%s)\n     out: %s\n     calls: %s\n' \
             "$desc" "$expect" "$rc" "$out" "$(tr '\n' ';' < "$td/calls.log")"; fails=$((fails+1)); fi
    }

    echo "== pr_review_shadow_publish.sh --self-test =="
    row 'no existing comment -> CREATE (POST)'     PASS "$(mk_gh gh-none '' 0)"       'POST'
    row 'existing marked comment -> UPDATE (PATCH)' PASS "$(mk_gh gh-has '4242' 0)"   'PATCH'
    row 'update targets the id the marker matched'  PASS "$(mk_gh gh-has2 '99' 0)"    'issues/comments/99'
    row 'a failing write is LOUD, never a silent skip' FAIL "$(mk_gh gh-wfail '' 7)"
    row 'an absent gh is exit 1, not a skip'        FAIL "$td/no-such-gh"
    row 'a missing body file is exit 1, never an empty post' FAIL "$(mk_gh gh-ok '' 0)" '' "$td/absent.txt"

    printf '\n%s passed, %s failed\n' "$pass_n" "$fails"
    [ "$fails" -eq 0 ] || { echo 'SELF-TEST FAILED'; exit 1; }
    exit 0
}

PR=''; BODY=''; MARKER=$MARKER_DEFAULT
while [ $# -gt 0 ]; do
  case "$1" in
    --self-test) self_test ;;
    --pr)        PR=${2:-};     shift 2 ;;
    --body-file) BODY=${2:-};   shift 2 ;;
    --marker)    MARKER=${2:-}; shift 2 ;;
    -h|--help)   sed -n '2,32p' "$0"; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done
[ -n "$PR" ]   || fail "--pr is required"
[ -n "$BODY" ] || fail "--body-file is required"
publish "$PR" "$BODY" "$MARKER"
