#!/usr/bin/env bash
# review_receipts_nightly.sh - receipt_presence over the pull requests that MERGED
# to main (PR-REVIEW-SKILL-002 v2 S8: PRs merged carrying a valid signed receipt /
# PRs merged). A LAB check: it runs at night and cannot block anything.
#
# WHY AT NIGHT AND NOT ON THE PR (#4602, #4688)
# ---------------------------------------------
# `pr-review-quorum / present` judged each PR's own receipt on the PR and again in
# the merge queue. It is not a required context, so its red never stopped a merge:
# 8 of the last 30 queue runs failed (2026-10-06..09) and all 8 PRs merged, and
# #4944 merged with it red on the PR as well. A check that cannot block a merge does
# not run on the PR or merge-queue path. It runs here, over what actually merged,
# and a red night updates one ticket instead of sitting on PRs as a red nobody has
# to read.
#
# WHAT IT DOES
#   Walks main's first-parent commits in the window (default: the last 26 hours;
#   the 2 extra hours absorb a late start, and a PR judged on two nights costs
#   nothing). Each is a merge-queue squash whose subject ends in "(#N)". For each,
#   Arm 4 (check_pr_review_arm4.sh, unchanged) judges PR N with the squash as a
#   QUEUE subject: a receipt must bind <squash>^1..<squash>, as it did in the queue.
#
#   A commit with no "(#N)" landed without a PR, so no receipt can bind it: RED.
#   Arm 4 rc 1 is RED. Any other non-zero rc (2 = ENV, a killed run) is NOT MEASURED.
#
#   bash scripts/review_receipts_nightly.sh [--since <git date>]   # judge the window
#   bash scripts/review_receipts_nightly.sh --self-test            # case table
#   bash scripts/review_receipts_nightly.sh --seven-nights <conclusion>...
#       The LAB stop: given the newest-first conclusions of this check's nightly
#       runs, prints `stop` and exits 0 when there are at least 7 and none of the
#       newest 7 is `success` (red or not measured for seven nights: it stops until
#       its owner turns it back on). Otherwise prints `keep` and exits 1.
#
# EXIT (judge mode): 0 every commit in the window passed and there was at least one;
#   1 any RED; 2 nothing red but something not measured. An EMPTY window is 2: it
#   measured nothing, and nothing measured is never a pass.
#
# The table goes to stdout; Arm 4's own output goes to stderr (the job log).

set -uo pipefail

PROG=${0##*/}
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Not read from the environment: only --self-test points this at a stub, so there
# is no value of any variable that swaps the judge out of a real run.
JUDGE_CMD="$REPO_ROOT/scripts/check_pr_review_arm4.sh"

# walk <repo> <since> - print the table for <repo>'s first-parent commits since
# <since>, return 0/1/2 as documented above.
walk() {
    local repo=$1 since=$2 sha subj pr rc n=0 pass=0 red=0 nm=0 log
    log=$(git -C "$repo" log --first-parent --format='%H %s' --since="$since" HEAD) \
        || { echo "$PROG: ENV - git log failed in $repo" >&2; return 2; }
    printf '| verdict | PR | commit |\n|---|---|---|\n'
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        sha=${line%% *}; subj=${line#* }
        n=$((n + 1))
        pr=$(printf '%s\n' "$subj" | sed -n 's/.*(#\([0-9][0-9]*\))[[:space:]]*$/\1/p')
        if [ -z "$pr" ]; then
            printf '| RED: no PR number in the subject, so no receipt can bind it | - | %s |\n' "${sha:0:10}"
            red=$((red + 1)); continue
        fi
        echo "== PR #$pr, squash $sha" >&2
        PR_REVIEW_SUBJECT_KIND=queue PR_NUMBER="$pr" PR_HEAD_SHA="$sha" bash "$JUDGE_CMD" >&2
        rc=$?
        case "$rc" in
          0) printf '| PASS | #%s | %s |\n' "$pr" "${sha:0:10}"; pass=$((pass + 1)) ;;
          1) printf '| RED (Arm 4 rc 1) | #%s | %s |\n' "$pr" "${sha:0:10}"; red=$((red + 1)) ;;
          *) printf '| NOT MEASURED (Arm 4 rc %s) | #%s | %s |\n' "$rc" "$pr" "${sha:0:10}"; nm=$((nm + 1)) ;;
        esac
    done <<< "$log"
    printf '\nreceipt_presence: %s/%s merged in the window passed (red %s, not measured %s), since "%s", main %s\n' \
        "$pass" "$n" "$red" "$nm" "$since" "$(git -C "$repo" rev-parse --short=10 HEAD)"
    [ "$red" -eq 0 ] || return 1
    [ "$n" -gt 0 ] && [ "$nm" -eq 0 ] || return 2
    return 0
}

seven_nights() {
    local i=0 c
    for c in "$@"; do
        [ "$i" -lt 7 ] || break
        [ "$c" != success ] || { echo keep; return 1; }
        i=$((i + 1))
    done
    if [ "$i" -ge 7 ]; then echo stop; return 0; fi
    echo keep; return 1
}

self_test() {
    local st stub fail=0 rows=0
    st=$(mktemp -d "${TMPDIR:-/tmp}/review-nightly-st.XXXXXX") || { echo "$PROG: ENV - mktemp failed" >&2; return 2; }
    stub="$st/judge.sh"
    # The stub answers from a "<pr> <rc>" map, and answers 3 (not measured) unless
    # it was called as Arm 4 must be called: a QUEUE subject that is a commit.
    cat > "$stub" <<'STUB'
kind=${PR_REVIEW_SUBJECT_KIND:-}
[ "$kind" = queue ] || exit 3
git -C "$REPO" cat-file -e "${PR_HEAD_SHA:-x}^{commit}" 2>/dev/null || exit 3
echo "$PR_NUMBER" >> "$CALLS"
rc=$(sed -n "s/^$PR_NUMBER //p" "$MAP")
exit "${rc:-3}"
STUB
    # mkrepo <dir> <subject>... - a root commit dated 2000 (outside the window), then
    # one commit per subject dated now.
    mkrepo() {
        local d=$1 s; shift
        # A throwaway fixture repo: no signing, no hooks from the host's global config.
        git init -q "$d" && git -C "$d" config user.email t@t && git -C "$d" config user.name t \
            && git -C "$d" config commit.gpgsign false && git -C "$d" config core.hooksPath /dev/null
        GIT_AUTHOR_DATE='2000-01-01T00:00:00Z' GIT_COMMITTER_DATE='2000-01-01T00:00:00Z' \
            git -C "$d" commit -q --allow-empty -m 'root (#1)'
        for s in "$@"; do git -C "$d" commit -q --allow-empty -m "$s"; done
    }
    # row <id> <want rc> <map> <subject>...
    row() {
        local id=$1 want=$2 map=$3 d got; shift 3
        d="$st/$id"; mkrepo "$d" "$@" || { echo "$PROG: ENV - fixture $id" >&2; fail=1; return; }
        printf '%b' "$map" > "$d.map"; : > "$d.calls"
        (JUDGE_CMD=$stub; export REPO="$d" MAP="$d.map" CALLS="$d.calls"; walk "$d" '1 day ago') > "$d.out" 2>&1
        got=$?
        rows=$((rows + 1))
        if [ "$got" -eq "$want" ]; then printf '  ok   %-22s rc %s\n' "$id" "$got"
        else printf '  FAIL %-22s rc %s, want %s\n' "$id" "$got" "$want"; sed 's/^/       /' "$d.out"; fail=1; fi
    }
    row all-pass             0 '11 0\n12 0\n'         'feat: a (#11)' 'fix: b (#12)'
    row one-red              1 '11 0\n12 1\n'         'feat: a (#11)' 'fix: b (#12)'
    row not-measured         2 '11 0\n12 2\n'         'feat: a (#11)' 'fix: b (#12)'
    row red-beats-unmeasured 1 '11 1\n12 2\n'         'feat: a (#11)' 'fix: b (#12)'
    row killed-judge         2 '11 0\n12 137\n'       'feat: a (#11)' 'fix: b (#12)'
    row empty-window         2 ''
    row no-pr-number         1 '11 0\n'               'feat: a (#11)' 'chore: pushed with no PR'
    row last-number-wins     0 '11 1\n13 0\n'         'revert: a (#11) (#13)'
    row number-mid-subject   1 '11 0\n'               'fix (#11): trailing text'
    # The judge must have been called once per PR, with the last number.
    rows=$((rows + 1))
    if [ "$(tr '\n' ' ' < "$st/last-number-wins.calls")" = '13 ' ]; then echo '  ok   judged-13-only'
    else echo '  FAIL judged-13-only: calls were' "$(tr '\n' ' ' < "$st/last-number-wins.calls")"; fail=1; fi
    seven() {
        local id=$1 want=$2 got; shift 2
        rows=$((rows + 1))
        got=$(seven_nights "$@")
        if [ "$got" = "$want" ]; then printf '  ok   %-22s %s\n' "$id" "$got"
        else printf '  FAIL %-22s %s, want %s\n' "$id" "$got" "$want"; fail=1; fi
    }
    seven seven-red          stop failure failure failure failure failure failure failure
    seven six-red            keep failure failure failure failure failure failure
    seven green-inside-seven keep failure failure failure failure failure failure success
    seven green-eighth       stop failure cancelled failure timed_out failure failure failure success
    seven green-newest       keep success failure failure failure failure failure failure failure
    printf '%s: --self-test %s rows, %s\n' "$PROG" "$rows" "$([ "$fail" -eq 0 ] && echo PASS || echo FAIL)"
    case "$st" in "${TMPDIR:-/tmp}"/review-nightly-st.*) rm -rf -- "${st:?}" ;; esac
    return "$fail"
}

SINCE='26 hours ago'
case "${1:-}" in
  --self-test)    self_test; exit $? ;;
  --seven-nights) shift; seven_nights "$@"; exit $? ;;
  --since)        SINCE=${2:?--since needs a git date}; [ "$#" -eq 2 ] || { echo "usage: $PROG [--since <git date>]" >&2; exit 2; } ;;
  '')             ;;
  *)              echo "usage: $PROG [--since <git date>] | --self-test | --seven-nights <conclusion>..." >&2; exit 2 ;;
esac
walk "$REPO_ROOT" "$SINCE"
