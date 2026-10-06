#!/usr/bin/env bash
# sha_checks.sh -- are the named CI checks green for exactly this commit? (#4672, C316 2c)
#
#   bash scripts/release/sha_checks.sh SHA SPEC [SPEC...]   # one verdict line per SPEC
#   bash scripts/release/sha_checks.sh --self-test          # planted rows must stay red
#
# WHY. `dogfood --phase pre-publish` re-ran fmt, clippy, the whole test suite and coverage on
# the release commit, hours of work CI had already done for that same commit. The release
# commit reaches main through the merge queue, so `ci / gate` and `workspace-test` have
# already run on it, and coverage runs on it as `Coverage Nightly / coverage` (COV_FLOOR)
# and on its tag as `CI / ci / coverage`. This script reads those results instead of
# re-measuring them. It reads; it never re-runs and never waits.
#
# SPEC  "<workflow>:<check name>", e.g. "CI:ci / gate". Alternatives joined by "|" are
#       satisfied by any one of them: "CI:ci / coverage|Coverage Nightly:coverage".
#
# THE RULE. A SPEC is green only when its newest check run (by start time) ON THIS SHA,
# in a run of the named workflow, completed with conclusion SUCCESS. Everything else is red,
# never a skip: a red or cancelled newest run, a run still in progress, no run at all, a
# green run on a DIFFERENT commit, and a GitHub read that failed (not measured is red).
#
# GH-1: one GraphQL call per invocation, however many SPECs.
# ENV  GH (default gh) · SHA_CHECKS_REPO (paiml/aprender) · SHA_CHECKS_TSV (a file of rows to
#      judge instead of asking GitHub; the self-test plants rows through it).
# OUT  per SPEC: "ok <spec> <detail>" or "bad <spec> <why>".
# EXIT 0 every SPEC green · 1 any SPEC red, or the read failed · 2 usage.
set -uo pipefail
GH=${GH:-gh}
REPO=${SHA_CHECKS_REPO:-paiml/aprender}

# fetch SHA -> TSV rows "<sha>\t<workflow>\t<check>\t<status>\t<conclusion>\t<startedAt>".
fetch() {
    local q='query($o:String!,$n:String!,$s:GitObjectID!){repository(owner:$o,name:$n){object(oid:$s){... on Commit{oid checkSuites(first:100){nodes{workflowRun{workflow{name}} checkRuns(first:100){nodes{name status conclusion startedAt}}}}}}}}'
    "$GH" api graphql -f query="$q" -f o="${REPO%%/*}" -f n="${REPO#*/}" -f s="$1" \
        --jq '.data.repository.object as $c | $c.checkSuites.nodes[] | select(.workflowRun)
              | .workflowRun.workflow.name as $w | .checkRuns.nodes[]
              | [$c.oid, $w, .name, .status, (.conclusion // ""), (.startedAt // "")] | @tsv'
}

# decide SHA ALT < TSV -> "ok <detail>" or "bad <why>" for ONE alternative "<wf>:<check>".
decide() {
    local sha=$1 wf=${2%%:*} name=${2#*:} row
    row=$(awk -F'\t' -v s="$sha" -v w="$wf" -v n="$name" \
        '$1 == s && $2 == w && $3 == n { print $6 "\t" $4 "\t" $5 }' | LC_ALL=C sort | tail -n 1)
    [ -n "$row" ] || { echo "bad no '$name' run of workflow '$wf' on ${sha:0:10}"; return; }
    local st cc; st=$(cut -f2 <<<"$row"); cc=$(cut -f3 <<<"$row")
    if [ "$st" = COMPLETED ] && [ "$cc" = SUCCESS ]; then echo "ok $wf / $name SUCCESS on ${sha:0:10}"
    else echo "bad newest '$name' ($wf) on ${sha:0:10} is ${st}${cc:+ $cc}"; fi
}

# judge SHA SPEC < TSV -> "ok|bad <spec> <detail>"; any alternative green makes the SPEC green.
judge() {
    local sha=$1 spec=$2 rows alt v why=""
    rows=$(cat)
    IFS='|' read -r -a alts <<<"$spec"
    for alt in "${alts[@]}"; do
        v=$(decide "$sha" "$alt" <<<"$rows")
        case "$v" in ok\ *) echo "ok $spec -- ${v#ok }"; return ;; esac
        why="${why:+$why; }${v#bad }"
    done
    echo "bad $spec -- $why"
}

main() {
    local sha=$1 rows spec v rc=0; shift
    [[ "$sha" =~ ^[0-9a-f]{40}$ ]] || { echo "sha_checks: need a full 40-hex SHA, got '$sha'" >&2; return 2; }
    [ $# -ge 1 ] || { echo "sha_checks: need at least one SPEC" >&2; return 2; }
    if [ -n "${SHA_CHECKS_TSV:-}" ]; then rows=$(cat "$SHA_CHECKS_TSV") || rows=""
    elif ! rows=$(fetch "$sha" 2>&1); then
        for spec in "$@"; do echo "bad $spec -- the GitHub read failed, so not measured: ${rows:0:120}"; done
        return 1
    fi
    for spec in "$@"; do
        v=$(judge "$sha" "$spec" <<<"$rows"); echo "$v"
        case "$v" in ok\ *) ;; *) rc=1 ;; esac
    done
    return "$rc"
}

self_test() {
    local d A B fail=0 n=0 got want
    d=$(mktemp -d) || return 2
    A=$(printf 'a%.0s' {1..40}); B=$(printf 'b%.0s' {1..40})
    t() { # t <name> <want rc> <spec> <rows...>
        local name=$1 w=$2 spec=$3; shift 3
        printf '%s\n' "$@" | sed 's/ | /\t/g' > "$d/rows.tsv"
        SHA_CHECKS_TSV="$d/rows.tsv" main "$A" "$spec" > "$d/out" 2>&1; got=$?
        n=$((n + 1))
        if [ "$got" = "$w" ]; then printf 'ok    %-34s rc=%s %s\n' "$name" "$got" "$(head -c 90 "$d/out")"
        else printf 'FAIL  %-34s rc=%s want %s: %s\n' "$name" "$got" "$w" "$(cat "$d/out")"; fail=1; fi
    }
    local G="CI:ci / gate"
    t green_control              0 "$G" "$A | CI | ci / gate | COMPLETED | SUCCESS | 2026-10-06T10:00:00Z"
    t red_check_for_this_sha     1 "$G" "$A | CI | ci / gate | COMPLETED | FAILURE | 2026-10-06T10:00:00Z"
    t missing_check              1 "$G" "$A | CI | workspace-test | COMPLETED | SUCCESS | 2026-10-06T10:00:00Z"
    t green_for_a_different_sha  1 "$G" "$B | CI | ci / gate | COMPLETED | SUCCESS | 2026-10-06T10:00:00Z"
    t no_rows_at_all             1 "$G"
    t in_progress                1 "$G" "$A | CI | ci / gate | IN_PROGRESS |  | 2026-10-06T10:00:00Z"
    t cancelled                  1 "$G" "$A | CI | ci / gate | COMPLETED | CANCELLED | 2026-10-06T10:00:00Z"
    t skipped_is_not_green       1 "$G" "$A | CI | ci / gate | COMPLETED | SKIPPED | 2026-10-06T10:00:00Z"
    t same_name_other_workflow   1 "$G" "$A | Other | ci / gate | COMPLETED | SUCCESS | 2026-10-06T10:00:00Z"
    t newer_red_beats_older_green 1 "$G" "$A | CI | ci / gate | COMPLETED | SUCCESS | 2026-10-06T10:00:00Z" \
                                         "$A | CI | ci / gate | COMPLETED | FAILURE | 2026-10-06T11:00:00Z"
    t newer_green_rerun_wins     0 "$G" "$A | CI | ci / gate | COMPLETED | FAILURE | 2026-10-06T10:00:00Z" \
                                         "$A | CI | ci / gate | COMPLETED | SUCCESS | 2026-10-06T11:00:00Z"
    local C="CI:ci / coverage|Coverage Nightly:coverage"
    t coverage_nightly_green     0 "$C" "$A | Coverage Nightly | coverage | COMPLETED | SUCCESS | 2026-10-06T10:00:00Z"
    t coverage_tag_green         0 "$C" "$A | CI | ci / coverage | COMPLETED | SUCCESS | 2026-10-06T10:00:00Z"
    t coverage_red               1 "$C" "$A | Coverage Nightly | coverage | COMPLETED | FAILURE | 2026-10-06T10:00:00Z"
    t coverage_other_sha_green   1 "$C" "$B | Coverage Nightly | coverage | COMPLETED | SUCCESS | 2026-10-06T10:00:00Z"
    t coverage_missing           1 "$C" "$A | CI | ci / gate | COMPLETED | SUCCESS | 2026-10-06T10:00:00Z"
    # Two SPECs: one red makes the whole call red.
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$A" CI "ci / gate" COMPLETED SUCCESS 2026-10-06T10:00:00Z > "$d/two.tsv"
    SHA_CHECKS_TSV="$d/two.tsv" main "$A" "$G" "CI:workspace-test" > "$d/out" 2>&1; got=$?; n=$((n + 1))
    if [ "$got" = 1 ] && grep -q '^bad CI:workspace-test' "$d/out"; then echo "ok    one_of_two_red                     rc=1"
    else echo "FAIL  one_of_two_red rc=$got: $(cat "$d/out")"; fail=1; fi
    # A failed GitHub read is red for every SPEC, never a skip.
    GH=false main "$A" "$G" > "$d/out" 2>&1; got=$?; n=$((n + 1))
    if [ "$got" = 1 ] && grep -q 'not measured' "$d/out"; then echo "ok    github_read_failed                 rc=1"
    else echo "FAIL  github_read_failed rc=$got: $(cat "$d/out")"; fail=1; fi
    main "abc" "$G" > /dev/null 2>&1; got=$?; n=$((n + 1))
    if [ "$got" = 2 ]; then echo "ok    short_sha_is_usage                 rc=2"; else echo "FAIL  short_sha_is_usage rc=$got"; fail=1; fi
    rm -rf "${d:?}"
    if [ "$fail" -eq 0 ]; then echo "sha_checks self-test: $n/$n"; return 0; fi
    echo "sha_checks self-test: FAILED"; return 1
}

if [ "${1:-}" = --self-test ]; then self_test; exit $?; fi
[ $# -ge 2 ] || { sed -n '2,5p' "$0" >&2; exit 2; }
main "$@"
