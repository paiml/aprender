#!/usr/bin/env bash
# check_ci_concurrency_cancel.sh -- which CI run a newer run cancels (#5381).
#
# THE DEFECT, measured 2026-10-15. A merge-queue entry that is kicked and then
# re-enqueued on the SAME base gets the SAME ref,
# `gh-readonly-queue/main/pr-<N>-<base oid>`. Run 38693636752 built
# `pr-5369-8f601caff2...` (06:38:37Z) and kept running after #5369's kick; the
# re-enqueue run 38694775126 carried the identical ref (06:49:44Z), landed in the
# same concurrency group `ci-${{ github.event.pull_request.number || github.ref }}`,
# and -- `cancel-in-progress` being true for `pull_request` only -- QUEUED behind
# the dead run for about 31 minutes.
#
# WHY CANCELLING IS SAFE FOR merge_group. The ref names exactly one queue entry:
# (PR number, base oid) -- see check_ci_unwedge.sh rule 3, which measured that the
# 40 hex IS the entry's baseCommit.oid. GitHub only dispatches a second run for an
# existing ref when that entry was rebuilt, so the older run on that ref is building
# a verdict nobody will read. Distinct stacked entries have distinct refs, so they
# never share a group and never cancel each other (rows M2/M3). What #3292 warned
# against -- cancelling a LIVE queue build by hand -- cannot happen through this
# expression: only a newer run in the SAME group cancels.
#
# THE MODEL. GitHub evaluates `cancel-in-progress` on the NEW run: it cancels the
# in-progress run of its group when the expression is true, and otherwise queues
# behind it. A run of another group is untouched. This guard reads the two
# expressions out of .github/workflows/ci.yml, refuses any shape it cannot model,
# and replays the case table below against them.
#
# Usage
#   bash scripts/check_ci_concurrency_cancel.sh              check ci.yml
#   bash scripts/check_ci_concurrency_cancel.sh --self-test  the table must also
#                                                            go RED on two mutants
#   bash scripts/check_ci_concurrency_cancel.sh --help
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CI_YML="${CI_YML:-$ROOT/.github/workflows/ci.yml}"
WANT_GROUP='ci-${{ github.event.pull_request.number || github.ref }}'

Q='refs/heads/gh-readonly-queue/main'
B1='8f601caff2eeef7224c3aa58ac3382bb3ab688f0'
B2='a95ebe9ad8b85859bc66814174eeba5b3321b125'
B3='0c6932fd543b78c46dcb18998bf974960401b7c9'

# id | old event | old PR | old ref | new event | new PR | new ref | expected
CASES="$(cat <<EOF
P1|pull_request|5369|refs/pull/5369/merge|pull_request|5369|refs/pull/5369/merge|CANCELS
P2|pull_request|5369|refs/pull/5369/merge|pull_request|5370|refs/pull/5370/merge|INDEPENDENT
M1|merge_group||$Q/pr-5369-$B1|merge_group||$Q/pr-5369-$B1|CANCELS
M2|merge_group||$Q/pr-5369-$B1|merge_group||$Q/pr-5370-$B2|INDEPENDENT
M3|merge_group||$Q/pr-5370-$B2|merge_group||$Q/pr-5371-$B3|INDEPENDENT
M4|merge_group||$Q/pr-5369-$B1|merge_group||$Q/pr-5369-$B3|INDEPENDENT
X1|pull_request|5369|refs/pull/5369/merge|merge_group||$Q/pr-5369-$B1|INDEPENDENT
X2|merge_group||$Q/pr-5369-$B1|pull_request|5369|refs/pull/5369/merge|INDEPENDENT
H1|push||refs/heads/main|push||refs/heads/main|QUEUES
H2|push||refs/heads/main|workflow_dispatch||refs/heads/main|QUEUES
H3|workflow_dispatch||refs/heads/main|push||refs/heads/main|QUEUES
T1|push||refs/tags/v0.72.0|push||refs/tags/v0.72.0|QUEUES
EOF
)"

usage() {
    sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# top_level_key FILE KEY -> the value of KEY inside the top-level `concurrency:` block.
top_level_key() {
    awk -v key="$2" '
        /^concurrency:[[:space:]]*$/ { inblk = 1; next }
        inblk && /^[^[:space:]#]/      { inblk = 0 }
        inblk && $1 == key ":" {
            sub("^[[:space:]]*" key ":[[:space:]]*", "")
            print
            exit
        }' "$1"
}

# cancel_events EXPR -> the events the expression is true for, one per line, or
# `ALL` for a literal `true`. rc 1 on any shape this guard cannot model.
cancel_events() {
    local expr="$1" inner term
    case "$expr" in
        true) echo ALL; return 0 ;;
        false) return 0 ;;
    esac
    [[ "$expr" =~ ^\$\{\{[[:space:]]*(.*[^[:space:]])[[:space:]]*\}\}$ ]] || return 1
    inner="${BASH_REMATCH[1]}"
    inner="${inner//||/$'\n'}"
    while IFS= read -r term; do
        [[ "$term" =~ ^[[:space:]]*github\.event_name[[:space:]]*==[[:space:]]*\'([a-z_]+)\'[[:space:]]*$ ]] || return 1
        echo "${BASH_REMATCH[1]}"
    done <<<"$inner"
}

# group EVENT PR REF -> the evaluated WANT_GROUP. pull_request.number only exists
# on a pull_request event.
group() {
    if [[ "$1" == pull_request && -n "$2" ]]; then
        echo "ci-$2"
    else
        echo "ci-$3"
    fi
}

# replay EVENTS -> one row per case, rc 1 when any row differs from its expectation.
replay() {
    local events="$1" bad=0 id oe op oref ne np nref want got
    while IFS='|' read -r id oe op oref ne np nref want; do
        if [[ "$(group "$oe" "$op" "$oref")" != "$(group "$ne" "$np" "$nref")" ]]; then
            got=INDEPENDENT
        elif [[ "$events" == ALL ]] || grep -qxF -- "$ne" <<<"$events"; then
            got=CANCELS
        else
            got=QUEUES
        fi
        if [[ "$got" == "$want" ]]; then
            printf '  ok    %-3s %s -> %s: %s\n' "$id" "$oe" "$ne" "$got"
        else
            printf '  RED   %-3s %s -> %s: %s, want %s\n' "$id" "$oe" "$ne" "$got" "$want"
            bad=1
        fi
    done <<<"$CASES"
    return "$bad"
}

check_file() {
    local g c events
    g="$(top_level_key "$CI_YML" group)"
    c="$(top_level_key "$CI_YML" cancel-in-progress)"
    if [[ "$g" != "$WANT_GROUP" ]]; then
        printf 'RED: %s concurrency group is %q, the table models %q\n' "$CI_YML" "$g" "$WANT_GROUP"
        return 1
    fi
    if ! events="$(cancel_events "$c")"; then
        printf 'RED: cancel-in-progress %q is not a shape this guard can model\n' "$c"
        return 1
    fi
    printf 'cancel-in-progress: %s\n' "$c"
    replay "$events"
}

self_test() {
    local bad=0 m out
    echo "-- the table on the shipped expression --"
    replay "$(cancel_events "\${{ github.event_name == 'pull_request' || github.event_name == 'merge_group' }}")" || bad=1
    for m in "\${{ github.event_name == 'pull_request' }}" true; do
        echo "-- mutant must go RED: $m --"
        if out="$(replay "$(cancel_events "$m")")"; then
            echo "  RED   the table passed the mutant; it cannot tell the two apart"
            bad=1
        else
            grep '^  RED' <<<"$out" | sed 's/RED /killed/'
        fi
    done
    echo "-- shapes it must refuse --"
    for m in "\${{ github.ref != 'refs/heads/main' }}" "\${{ github.event_name == 'pull_request' && true }}"; do
        if cancel_events "$m" >/dev/null; then
            echo "  RED   modelled an expression it does not understand: $m"
            bad=1
        else
            echo "  ok    refused: $m"
        fi
    done
    return "$bad"
}

case "${1:-}" in
    --help|-h) usage ;;
    --self-test) self_test && echo "PASS self-test" || { echo "FAIL self-test"; exit 1; } ;;
    "") check_file && echo "PASS $CI_YML" || { echo "FAIL $CI_YML"; exit 1; } ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
esac
