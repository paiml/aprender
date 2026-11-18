#!/usr/bin/env bash
# lab_stop.sh — the stop step of a nightly LAB lane: disable its workflow when none of its last 7 nights was green
# (epic #7149 rule R3, slice S0.5, #7158). No workflow runs this yet; the stop-check job of S0.6 will only call it.
#
# THE RULE. A lane's workflow is disabled when none of its last 7 nights was green. A night is green only when
#   scripts/release/nightly_greens.sh says green: failure, cancelled, timed out, retried and not started are all not
#   green. This script reads one run history (the TSV nightly_greens.sh reads), cuts it to the last 7 dates ending at
#   --as-of, asks nightly_greens.sh for the lane's check (lab-<lane>, need 7) and stops when it reports total=0.
#   No second counter is written.
#   A lane with fewer than 7 nights (its first run on main is later than --as-of minus 6 days) is never disabled.
#   Tonight's own run is still pending when the check runs, so --tonight success (the cell job's result) keeps the
#   lane: tonight is a green night.
#   Only a run on main stops a lane: on any other ref the script decides nothing and makes no call.
#
# STATED, NOT PROVED HERE: that the real `gh workflow disable` call succeeds with the job's token. The selftest runs
#   against a stub that records its arguments; the call is checked by what it is asked to do, not by what GitHub does.
#
# USAGE
#   lab_stop.sh --lane edge|edge-opt|wasm|box --workflow FILE --history TSV --as-of YYYY-MM-DD --ref REF [--tonight RESULT]
#   lab_stop.sh --self-test     the case table, with gh replaced by a stub that records its arguments
#   lab_stop.sh --mutants       each planted mutant of this script must turn the case table RED
# ENV   LAB_STOP_GH  the gh command (default gh); the selftest points it at its stub
# EXIT  0 decided (kept or disabled) · 1 the disable call failed · 2 not_measured (the judge could not read the history)
#       3 caller error
set -uo pipefail

PROG="${0##*/}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$HERE/${BASH_SOURCE[0]##*/}"
GREENS="$HERE/nightly_greens.sh"
WINDOW=7

caller_error() { printf 'FAIL  LAB-STOP %s: caller error: %s\n' "$PROG" "$*"; exit 3; }

# decide LANE WORKFLOW HISTORY ASOF REF TONIGHT -> keep or stop on stdout, then the gh call when stopping
decide() {
    local lane=$1 workflow=$2 history=$3 asof=$4 ref=$5 tonight=$6
    local gh="${LAB_STOP_GH:-gh}" start first cut verdict total rc=0
    if [ "$ref" != refs/heads/main ] && [ "$ref" != main ]; then
        printf 'keep  LAB-STOP %s: ref %s is not main, nothing decided\n' "$lane" "$ref"; return 0
    fi
    start=$(date -u -d "$asof - $((WINDOW - 1)) days" +%F) || caller_error "--as-of \"$asof\" is not a date"
    first=$(awk -F'\t' 'NR > 1 && $3 == "main" && ($4 == "schedule" || $4 == "workflow_dispatch") { d = substr($2, 1, 10); if (m == "" || d < m) m = d } END { print m }' "$history")
    if [ -z "$first" ] || [[ "$first" > "$start" ]]; then
        printf 'keep  LAB-STOP %s: fewer than %d nights (first run on main %s, window starts %s)\n' "$lane" "$WINDOW" "${first:-none}" "$start"; return 0
    fi
    if [ "$tonight" = success ]; then
        printf 'keep  LAB-STOP %s: tonight is green\n' "$lane"; return 0
    fi
    cut=$(mktemp) || caller_error "mktemp failed"
    awk -F'\t' -v s="$start" -v e="$asof" 'NR == 1 || (substr($2, 1, 10) >= s && substr($2, 1, 10) <= e)' "$history" > "$cut"
    verdict=$(bash "$GREENS" --check "lab-$lane" --history "$cut" --as-of "$asof" 2>&1) || rc=$?
    rm -f "${cut:?}"
    if [ "$rc" -eq 3 ]; then printf '%s\n' "$verdict"; return 3; fi
    if [[ "$verdict" =~ total=([0-9]+)\  ]]; then total=${BASH_REMATCH[1]}
    else printf 'not_measured  LAB-STOP %s: the judge printed no count\n%s\n' "$lane" "$verdict"; return 2; fi
    if [ "$total" -ne 0 ]; then
        printf 'keep  LAB-STOP %s: %s green night(s) in the last %d\n' "$lane" "$total" "$WINDOW"; return 0
    fi
    printf 'stop  LAB-STOP %s: none of the last %d nights was green, disabling %s\n' "$lane" "$WINDOW" "$workflow"
    "$gh" workflow disable "$workflow" || { printf 'FAIL  LAB-STOP %s: gh workflow disable %s failed\n' "$lane" "$workflow"; return 1; }
    return 0
}

main() {
    local lane="" workflow="" history="" asof="" ref="" tonight=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --lane|--workflow|--history|--as-of|--ref|--tonight)
                [ $# -ge 2 ] || caller_error "$1 needs a value"
                case "$1" in
                    --lane) lane="$2" ;; --workflow) workflow="$2" ;; --history) history="$2" ;;
                    --as-of) asof="$2" ;; --ref) ref="$2" ;; *) tonight="$2" ;;
                esac
                shift 2 ;;
            *) caller_error "unknown option $1" ;;
        esac
    done
    case "$lane" in edge|edge-opt|wasm|box) : ;; *) caller_error "--lane '$lane' is not edge, edge-opt, wasm or box" ;; esac
    case "$workflow" in ''|*[!A-Za-z0-9._-]*) caller_error "--workflow '$workflow' must be a workflow file name" ;; esac
    [ -n "$asof" ] || caller_error "--as-of is required"
    [ -n "$ref" ] || caller_error "--ref is required"
    { [ -n "$history" ] && [ -f "$history" ] && [ -r "$history" ]; } || caller_error "--history '$history' is not a readable file"
    decide "$lane" "$workflow" "$history" "$asof" "$ref" "$tonight"
}

# --------------------------------------------------------------- selftest ---
selftest_cleanup() { case "${tmp:-}" in ''|/) return 0 ;; *) rm -rf -- "${tmp:?}" ;; esac; }
selftest() {
    local tmp pass=0 fail=0 D=2026-10 script=${1:-$SCRIPT_PATH}
    tmp="$(mktemp -d)"
    case "$tmp" in ?*/tmp.?*) : ;; *) echo "mktemp gave '${tmp:-}', refusing"; return 2 ;; esac
    trap selftest_cleanup RETURN
    local wf=edge-fast-verbs-nightly.yml
    printf '#!/usr/bin/env bash\nprintf "%%s\\n" "$*" >> "%s/calls"\n[ -z "${STUB_FAIL:-}" ]\n' "$tmp" > "$tmp/gh"
    chmod +x "$tmp/gh"
    fx() { # name, then runs "id date conclusion [attempt [branch [event]]]"
        local f="$tmp/$1" r id d co at br ev; shift
        printf 'run_id\tcreated_at\tbranch\tevent\tconclusion\tattempt\n' > "$f"
        for r in "$@"; do
            read -r id d co at br ev <<< "$r"
            printf '%s\t%sT02:17:00Z\t%s\t%s\t%s\t%s\n' "$id" "$d" "${br:-main}" "${ev:-schedule}" "$co" "${at:-1}" >> "$f"
        done
    }
    row() { # name expect-rc expect-call(none|disable) needle history ref [tonight [STUB_FAIL]]
        local name=$1 erc=$2 ecall=$3 needle=$4 hist=$5 ref=$6 tn=${7:-} sf=${8:-} o rc=0 calls=none
        : > "$tmp/calls"
        o=$(LAB_STOP_GH="$tmp/gh" STUB_FAIL="$sf" bash "$script" --lane edge --workflow "$wf" --history "$tmp/$hist" --as-of "$D-07" --ref "$ref" --tonight "$tn" 2>&1) || rc=$?
        [ -s "$tmp/calls" ] && calls=$(< "$tmp/calls")
        [ "$ecall" = disable ] && ecall="workflow disable $wf"
        if [ "$rc" = "$erc" ] && [ "$calls" = "$ecall" ] && [[ "$o" == *"$needle"* ]]; then
            printf '  ok    %s\n' "$name"; pass=$((pass + 1))
        else
            printf '  BROKE %s: want rc=%s call=[%s] needle=[%s]; got rc=%s call=[%s]\n%s\n' "$name" "$erc" "$ecall" "$needle" "$rc" "$calls" "$o"; fail=$((fail + 1))
        fi
    }
    local m=refs/heads/main
    fx seven "1 $D-01 failure" "2 $D-02 failure" "3 $D-03 failure" "4 $D-04 failure" "5 $D-05 failure" "6 $D-06 failure" "7 $D-07 failure"
    fx six "1 $D-01 success" "2 $D-02 failure" "3 $D-03 failure" "4 $D-04 failure" "5 $D-05 failure" "6 $D-06 failure" "7 $D-07 failure"
    fx one_green "1 $D-01 failure" "2 $D-02 failure" "3 $D-03 failure" "4 $D-04 success" "5 $D-05 failure" "6 $D-06 failure" "7 $D-07 failure"
    fx last_green "1 $D-01 failure" "2 $D-02 failure" "3 $D-03 failure" "4 $D-04 failure" "5 $D-05 failure" "6 $D-06 failure" "7 $D-07 success"
    fx young "5 $D-05 failure" "6 $D-06 failure" "7 $D-07 failure"
    fx cancelled "1 $D-01 cancelled" "2 $D-02 cancelled" "3 $D-03 failure" "4 $D-04 timed_out" "5 $D-05 cancelled" "6 $D-06 failure" "7 $D-07 failure"
    fx retried "1 $D-01 failure" "2 $D-02 failure" "3 $D-03 success 2" "4 $D-04 failure" "5 $D-05 failure" "6 $D-06 failure" "7 $D-07 failure"
    fx branch_green "1 $D-01 failure" "2 $D-02 failure" "3 $D-03 success 1 feat/x" "4 $D-04 failure" "5 $D-05 failure" "6 $D-06 failure" "7 $D-07 failure"
    fx gap "1 $D-01 failure" "3 $D-03 failure" "4 $D-04 failure" "5 $D-05 failure" "6 $D-06 failure" "7 $D-07 failure"
    fx old_green "0 2026-09-20 success" "1 $D-01 failure" "2 $D-02 failure" "3 $D-03 failure" "4 $D-04 failure" "5 $D-05 failure" "6 $D-06 failure" "7 $D-07 failure"
    row seven_not_green_nights_disable            0 disable "stop  LAB-STOP edge" seven "$m"
    row ref_may_be_plain_main                     0 disable "stop  LAB-STOP edge" seven main
    row six_not_green_and_one_green_keep          0 none "1 green night(s)" six "$m"
    row one_green_in_the_middle_keeps             0 none "1 green night(s)" one_green "$m"
    row tonight_green_keeps                       0 none "1 green night(s)" last_green "$m"
    row tonight_green_result_keeps_seven_red      0 none "tonight is green" seven "$m" success
    row tonight_failed_result_still_disables      0 disable "stop  LAB-STOP edge" seven "$m" failure
    row fewer_than_seven_nights_keep              0 none "fewer than 7 nights" young "$m"
    row cancelled_and_timed_out_disable           0 disable "stop  LAB-STOP edge" cancelled "$m"
    row a_retried_green_is_not_green_disables     0 disable "stop  LAB-STOP edge" retried "$m"
    row a_green_on_another_branch_is_not_counted  0 disable "stop  LAB-STOP edge" branch_green "$m"
    row a_night_with_no_run_is_not_green          0 disable "stop  LAB-STOP edge" gap "$m"
    row a_green_older_than_the_window_is_ignored  0 disable "stop  LAB-STOP edge" old_green "$m"
    row a_branch_ref_makes_no_call                0 none "is not main, nothing decided" seven refs/heads/feat/x
    row a_pull_ref_makes_no_call                  0 none "is not main" seven refs/pull/1/merge
    row a_failed_disable_call_exits_one           1 disable "failed" seven "$m" "" 1
    row an_unreadable_history_is_a_caller_error   3 none "caller error" nope "$m"
    printf -- '--- %s/%s rows ---\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ]
}

# ---------------------------------------------------------------- mutants ---
# Each mutant is "name sed-script" run on a copy of this script. It must change the copy, still parse, and turn the
# case table RED with at least one BROKE row.
mutants() {
    local tmp pass=0 fail=0 name expr copy o rc
    tmp="$(mktemp -d)"
    case "$tmp" in ?*/tmp.?*) : ;; *) echo "mktemp gave '${tmp:-}', refusing"; return 2 ;; esac
    trap selftest_cleanup RETURN
    while read -r name expr; do
        [ -n "$name" ] || continue
        copy="$tmp/$name.sh"
        sed -e "$expr" "$SCRIPT_PATH" > "$copy"
        if cmp -s "$SCRIPT_PATH" "$copy"; then
            printf '  BROKE %-34s changed nothing: its pattern no longer matches\n' "$name"; fail=$((fail + 1)); continue
        fi
        if ! bash -n "$copy" 2>/dev/null; then
            printf '  BROKE %-34s does not parse\n' "$name"; fail=$((fail + 1)); continue
        fi
        rc=0; o="$(bash "$SCRIPT_PATH" --self-test-of "$copy" < /dev/null 2>&1)" || rc=$?
        case "$rc:$o" in
            0:*) printf '  BROKE %-34s SURVIVED: the case table stayed green\n' "$name"; fail=$((fail + 1)) ;;
            *"  BROKE "*) printf '  ok    %-34s killed, %s row(s) broke\n' "$name" "$(awk '/^  BROKE /{n++} END{print n+0}' <<< "$o")"; pass=$((pass + 1)) ;;
            *) printf '  BROKE %-34s exit %s with no broken row: not a kill\n' "$name" "$rc"; fail=$((fail + 1)) ;;
        esac
    done <<'MUTANTS'
drop_the_disable_call       s#^    "\$gh" workflow disable "\$workflow" ||#    true ||#
drop_the_ref_check          s|if \[ "\$ref" != refs/heads/main \] && \[ "\$ref" != main \]; then|if false; then|
change_the_workflow_name    s|"\$gh" workflow disable "\$workflow"|"$gh" workflow disable "${workflow}x"|
drop_the_young_lane_check   s#if \[ -z "\$first" \] || \[\[ "\$first" > "\$start" \]\]; then#if [ -z "$first" ]; then#
ignore_tonight              s|if \[ "\$tonight" = success \]; then|if false; then|
stop_on_one_green           s|if \[ "\$total" -ne 0 \]; then|if [ "$total" -gt 1 ]; then|
window_of_three             s|^WINDOW=7|WINDOW=3|
history_not_cut_to_window   s#--history "\$cut" --as-of#--history "$history" --as-of#
MUTANTS
    printf -- '--- %s/%s mutants killed ---\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ]
}

case "${1:-}" in
    --self-test) selftest ;;
    --self-test-of) selftest "${2:?}" ;;
    --mutants) mutants ;;
    -h|--help) awk 'NR > 1 && /^set -uo pipefail$/ { exit } NR > 1' "$0" ;;
    *) main "$@" ;;
esac
