#!/usr/bin/env bash
# nightly_greens.sh — has a check's input been produced green on `main`, nightly, three times? (BLD-001 row 10)
#
# THE RULE. Operator ruling C280 item 12 (2026-10-03), the P0 for 0.70.2: "no check may be enforced at publish until
#   its inputs have been produced green on main, nightly, three times." The publish preflight's R5, R7 and R8 judge
#   inputs that an earlier run produced (scripts/check_publish_preflight.sh, the R5, R7 and R8 rules in its header).
#   When this was written (2026-10-03, main @ 316dee2cd4) no workflow with a `schedule:` trigger produced them. This
#   script is the count: it reads one producer's run history, answers whether three nights on `main` were green, and
#   prints their run IDs.
#
# A NIGHT is a UTC date: schedule the producer well away from 00:00 UTC, since a run created after midnight counts for
#   the next night. Only rows with branch `main` and event `schedule` count: a push, a pull request, a manual dispatch
#   or a run on another branch never does. Rows dated after --as-of are not counted, so a replay over a longer history
#   gives the same answer. A night is
#     red      a run ended failure, timed_out or startup_failure, or succeeded only at attempt 2 or later (a retried
#              green is not a green: docs/specifications/FLOW-003-queue-and-release-cycle-model.md, G4, plans CI
#              retries 2 -> 0);
#     pending  else, a run has no conclusion yet;
#     green    else, a run succeeded at attempt 1;
#     void     else (cancelled, skipped, neutral, stale, action_required): it measured nothing;
#     absent   no run at all.
#
# VERDICT. LAST is the newest night that is green or red. Walking back from it one date at a time, the streak
#   counts green nights up to the first night that is not green.
#     ready         streak >= NEED, and LAST is --as-of or the day before. The NEED newest run IDs are printed.
#     not ready     the walk stopped at a red night, or at the start of the history (fewer than NEED greens exist).
#     not_measured  the walk stopped at an absent, void or pending night ("a night with no run is never green"); or
#                   no night is green or red; or LAST is older than the day before --as-of (stale); or a line of
#                   the history is unreadable.
#   BOTH COUNTS. C280 item 12 says "three times"; the 0.70.2 exit bar says "three nights in a row" (BLD-001 §4 Q8,
#   open). Every verdict line prints streak= (in a row) and total= (green nights up to --as-of). The verdict uses the
#   streak, the stricter reading, until the operator answers.
#   NEED IS A CONSTANT, not an option and not an environment value: a threshold a caller can lower is theater.
#   Changing it is a reviewed commit, and --selftest pins it.
#   ONE CHECK IS ONE NIGHT. release-rehearsal, the streak rehearse.sh --streak reads before a release pass starts,
#   needs REHEARSAL_NEED = 1 green night, not three (operator C355 item 6, 2026-10-10: the 3-night check goes to one
#   night; the count is lowered, the check stays). Every other check keeps NEED. Also a constant, pinned the same way.
#
# HISTORY. A TSV file. Its first line is exactly these six tab-separated names; then one run per line:
#     run_id  created_at (YYYY-MM-DDTHH:MM:SSZ, UTC)  branch  event  conclusion (empty while running)  attempt
#   A workflow run from the REST API maps onto it one to one: id, created_at, head_branch, event, conclusion (null
#   while running), run_attempt.
#
# EXIT  0 ready · 1 not ready · 2 not_measured (a stop: unknown is not a pass) · 3 caller error (a wiring defect)
#
# USAGE
#   nightly_greens.sh --check NAME --history FILE --as-of YYYY-MM-DD
#   nightly_greens.sh --selftest    the case table
#   nightly_greens.sh --mutants     each planted mutant of judge() or NEED must turn the case table RED
set -uo pipefail

PROG="${0##*/}"
SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/${BASH_SOURCE[0]##*/}"
# C280 item 12: "three times". A constant: never an option, never taken from the environment.
NEED=3
# C355 item 6: the release-rehearsal check needs one night. A constant, keyed on the check's name, never an option.
# An exact string compare, never a glob: no near-miss spelling inherits the one night.
REHEARSAL_NEED=1
need_of() { if [ "$1" = release-rehearsal ]; then printf '%s\n' "$REHEARSAL_NEED"; else printf '%s\n' "$NEED"; fi; }

caller_error() { printf 'FAIL  NIGHTLY %s: caller error: %s\n' "$PROG" "$*"; exit 3; }

# judge check history as-of -> the verdict line and one context line on stdout; exits 0/1/2/3 as above
judge() {
    awk -v check="$1" -v asof="$3" -v need="$(need_of "$1")" '
    function days(y, m, d,    era, yoe, doy, doe, mp) {    # civil date -> days since 1970-01-01 (H. Hinnant)
        y -= (m <= 2)
        era = int(y / 400)
        yoe = y - era * 400
        mp = (m + 9) % 12
        doy = int((153 * mp + 2) / 5) + d - 1
        doe = yoe * 365 + int(yoe / 4) - int(yoe / 100) + doy
        return era * 146097 + doe - 719468
    }
    function civil(z,    era, doe, yoe, y, doy, mp, d, m) {    # days since 1970-01-01 -> YYYY-MM-DD
        z += 719468
        era = int(z / 146097)
        doe = z - era * 146097
        yoe = int((doe - int(doe / 1460) + int(doe / 36524) - int(doe / 146096)) / 365)
        y = yoe + era * 400
        doy = doe - (365 * yoe + int(yoe / 4) - int(yoe / 100))
        mp = int((5 * doy + 2) / 153)
        d = doy - int((153 * mp + 2) / 5) + 1
        m = (mp < 10) ? mp + 3 : mp - 9
        return sprintf("%04d-%02d-%02d", y + (m <= 2), m, d)
    }
    function dayno(s) { return days(substr(s, 1, 4) + 0, substr(s, 6, 2) + 0, substr(s, 9, 2) + 0) }
    function isdate(s,    m, d) {    # a real calendar date, written YYYY-MM-DD, year 0001 or later
        if (s !~ /^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]$/) return 0
        m = substr(s, 6, 2) + 0; d = substr(s, 9, 2) + 0
        if (substr(s, 1, 4) + 0 < 1 || m < 1 || m > 12 || d < 1 || d > 31) return 0
        return (civil(dayno(s)) == s)
    }
    BEGIN {
        FS = "\t"
        want = "run_id\tcreated_at\tbranch\tevent\tconclusion\tattempt"
        rank["void"] = 1; rank["green"] = 2; rank["pending"] = 3; rank["red"] = 4; rank["retried"] = 4
        if (!isdate(asof)) { err = "--as-of \"" asof "\" is not a real date written YYYY-MM-DD"; exit }
        asof_n = dayno(asof)
    }
    NR == 1 {
        if ($0 != want) { err = "the history header is not the six tab-separated columns this judge reads"; exit }
        next
    }
    {
        if (NF != 6) { bad = "line " NR " has " NF " fields, not 6"; exit }
        id = $1; ts = $2; co = $5; at = $6
        if (id !~ /^[A-Za-z0-9._-]+$/) { bad = "line " NR " run_id \"" id "\""; exit }
        if (ts !~ /^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z$/ || !isdate(substr(ts, 1, 10))) {
            bad = "line " NR " created_at \"" ts "\""; exit
        }
        if (at !~ /^[1-9][0-9]*$/) { bad = "line " NR " attempt \"" at "\""; exit }
        if (co == "") s = "pending"
        else if (co == "success") s = (at == 1) ? "green" : "retried"
        else if (co == "failure" || co == "timed_out" || co == "startup_failure") s = "red"
        else if (co == "cancelled" || co == "skipped" || co == "neutral" || co == "stale" || co == "action_required") s = "void"
        else { bad = "line " NR " conclusion \"" co "\""; exit }
        if ($3 != "main" || $4 != "schedule") { other++; next }
        n = dayno(ts)
        if (n > asof_n) { later++; next }
        runs++
        if (!(n in st) || rank[s] > rank[st[n]] || (rank[s] == rank[st[n]] && ts > tsof[n])) {
            st[n] = s; rid[n] = id; tsof[n] = ts; con[n] = co; att[n] = at
        }
    }
    END {
        if (err != "") { printf "FAIL  NIGHTLY %s caller error: %s\n", check, err; exit 3 }
        if (NR == 0) { printf "FAIL  NIGHTLY %s not_measured: the history file is empty\n", check; exit 2 }
        if (bad != "") { printf "FAIL  NIGHTLY %s not_measured: unreadable history, %s\n", check, bad; exit 2 }
        last = ""; first = ""; total = 0
        for (k in st) {
            k += 0
            if (first == "" || k < first) first = k
            if (st[k] == "green") total++
            if ((st[k] == "green" || st[k] == "red" || st[k] == "retried") && (last == "" || k > last)) last = k
        }
        info = sprintf("      NIGHTLY %s counted %d scheduled run(s) on main up to %s; not counted: %d other row(s), %d after --as-of", check, runs, asof, other, later)
        npend = 0; pmax = 0
        for (k in st) if (st[k] == "pending" && (last == "" || k + 0 > last)) { npend++; if (k + 0 > pmax) pmax = k + 0 }
        if (npend > 0) info = info sprintf("; still running: %d night(s), newest %s", npend, civil(pmax))
        if (last == "") {
            printf "FAIL  NIGHTLY %s not_measured: no green or red nightly run on main up to %s (streak=0 total=0 need=%d)\n", check, asof, need
            print info; exit 2
        }
        streak = 0; k = last
        while ((k in st) && st[k] == "green") { streak++; ids[streak] = rid[k]; k-- }
        stop = (k in st) ? st[k] : ((k < first) ? "start" : "absent")
        counts = sprintf("streak=%d total=%d need=%d", streak, total, need)
        if (last < asof_n - 1) {
            printf "FAIL  NIGHTLY %s not_measured: stale, the newest green or red night is %s and --as-of %s needs %s or later (%s)\n", check, civil(last), asof, civil(asof_n - 1), counts
            print info; exit 2
        }
        if (streak >= need) {
            out = ""
            for (i = 1; i <= need; i++) out = out " " ids[i]
            printf "ok    NIGHTLY %s ready: %d green nights in a row on main, newest %s: runs%s (%s)\n", check, need, civil(last), out, counts
            print info; exit 0
        }
        if (stop == "red") {
            printf "FAIL  NIGHTLY %s not ready: night %s was red (run %s, %s) (%s)\n", check, civil(k), rid[k], con[k], counts
            print info; exit 1
        }
        if (stop == "retried") {
            printf "FAIL  NIGHTLY %s not ready: night %s was red (run %s, success only at attempt %s: a retried green is not a green) (%s)\n", check, civil(k), rid[k], att[k], counts
            print info; exit 1
        }
        if (stop == "start") {
            printf "FAIL  NIGHTLY %s not ready: the history on main starts %s, with %d green night(s) in a row (%s)\n", check, civil(first), streak, counts
            print info; exit 1
        }
        why = (stop == "absent") ? "no run" : ((stop == "pending") ? "run " rid[k] " has not finished" : "run " rid[k] " was " con[k])
        printf "FAIL  NIGHTLY %s not_measured: night %s has no completed run on main (%s); a night with no run is never green (%s)\n", check, civil(k), why, counts
        print info; exit 2
    }' "$2"
}

main() {
    local check="" history="" asof=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --check|--history|--as-of)
                [ $# -ge 2 ] || caller_error "$1 needs a value"
                case "$1" in --check) check="$2" ;; --history) history="$2" ;; *) asof="$2" ;; esac
                shift 2 ;;
            *) caller_error "unknown option $1 (usage: $PROG --check NAME --history FILE --as-of YYYY-MM-DD)" ;;
        esac
    done
    case "$check" in ''|*[!A-Za-z0-9._-]*) caller_error "--check '$check' must be a name of letters, digits, '.', '_' or '-'" ;; esac
    [ -n "$asof" ] || caller_error "--as-of is required: a judgment is made for a named date, never an implicit today"
    { [ -n "$history" ] && [ -f "$history" ] && [ -r "$history" ]; } || caller_error "--history '$history' is not a readable file"
    judge "$check" "$history" "$asof"
}

# --------------------------------------------------------------- selftest ---
selftest_cleanup() { case "${tmp:-}" in ''|/) return 0 ;; *) rm -rf -- "${tmp:?}" ;; esac; }
selftest() {
    local tmp pass=0 fail=0 D=2026-10
    tmp="$(mktemp -d)"
    case "$tmp" in ?*/tmp.?*) : ;; *) echo "mktemp gave '${tmp:-}', refusing"; return 2 ;; esac
    # SEC011: validate before rm -rf. $tmp is this function's own mktemp -d (checked above).
    trap selftest_cleanup RETURN
    fx() { # name, then runs "id date conclusion [attempt [branch [event [time]]]]"; conclusion - = still running
        local f="$tmp/$1" r id d co at br ev tm; shift
        printf 'run_id\tcreated_at\tbranch\tevent\tconclusion\tattempt\n' > "$f"
        for r in "$@"; do
            read -r id d co at br ev tm <<< "$r"
            [ "$co" = - ] && co=""
            printf '%s\t%sT%sZ\t%s\t%s\t%s\t%s\n' "$id" "$d" "${tm:-02:17:00}" "${br:-main}" "${ev:-schedule}" "$co" "${at:-1}" >> "$f"
        done
    }
    row() { # name expect-rc needle [VAR=value ...] -- script-args ...
        local name="$1" expect="$2" needle="$3" o rc=0; shift 3
        local -a cmd=(env)
        while [ $# -gt 0 ] && [ "$1" != -- ]; do cmd+=("$1"); shift; done
        shift
        cmd+=(bash "$SCRIPT_PATH" "$@")
        o="$("${cmd[@]}" < /dev/null 2>&1)" || rc=$?
        if [ "$rc" != "$expect" ]; then printf '  BROKE %-50s expected exit %s got %s\n%s\n' "$name" "$expect" "$rc" "$o"; fail=$((fail + 1)); return 0; fi
        case "$o" in
            *"$needle"*) printf '  ok    %-50s exit=%s\n' "$name" "$rc"; pass=$((pass + 1)) ;;
            *) printf '  BROKE %-50s exit %s but never said: %s\n%s\n' "$name" "$rc" "$needle" "$o"; fail=$((fail + 1)) ;;
        esac
    }
    j() { row "$1" "$2" "$3" -- --check R7 --history "$tmp/$4" --as-of "${5:-$D-06}"; }

    fx three "104 $D-04 success" "105 $D-05 success" "106 $D-06 success"
    j three_green_nights_are_ready                     0 "ok    NIGHTLY R7 ready: 3 green nights in a row on main, newest $D-06: runs 106 105 104 (streak=3 total=3 need=3)" three
    j the_context_line_counts_what_was_read            0 "NIGHTLY R7 counted 3 scheduled run(s) on main up to $D-06; not counted: 0 other row(s), 0 after --as-of" three
    fx shuffled "105 $D-05 success" "106 $D-06 success" "104 $D-04 success"
    j row_order_does_not_matter                        0 "runs 106 105 104 (streak=3" shuffled
    fx red_then_two "104 $D-04 failure" "105 $D-05 success" "106 $D-06 success"
    j two_greens_after_a_red_are_not_ready             1 "FAIL  NIGHTLY R7 not ready: night $D-04 was red (run 104, failure) (streak=2 total=2 need=3)" red_then_two
    fx two "105 $D-05 success" "106 $D-06 success"
    j two_greens_from_the_start_are_not_ready          1 "not ready: the history on main starts $D-05, with 2 green night(s) in a row (streak=2 total=2 need=3)" two
    fx gap "103 $D-03 success" "105 $D-05 success" "106 $D-06 success"
    j a_night_with_no_run_is_not_measured              2 "FAIL  NIGHTLY R7 not_measured: night $D-04 has no completed run on main (no run); a night with no run is never green (streak=2 total=3 need=3)" gap
    fx tonight "103 $D-03 success" "104 $D-04 success" "105 $D-05 success" "106 $D-06 -"
    j a_run_still_going_tonight_uses_the_three_before  0 "runs 105 104 103 (streak=3 total=3 need=3)" tonight
    j the_running_night_is_named                       0 "; still running: 1 night(s), newest $D-06" tonight
    fx yesterday "103 $D-03 success" "104 $D-04 success" "105 $D-05 success"
    j the_day_before_as_of_is_fresh                    0 "newest $D-05: runs 105 104 103" yesterday
    j two_days_before_as_of_is_stale                   2 "not_measured: stale, the newest green or red night is $D-05 and --as-of $D-07 needs $D-06 or later" yesterday "$D-07"
    fx retried "104 $D-04 success 2" "105 $D-05 success" "106 $D-06 success"
    j a_retried_green_is_not_green                     1 "night $D-04 was red (run 104, success only at attempt 2: a retried green is not a green)" retried
    fx branch "103 $D-03 success" "104 $D-04 success 1 feature" "105 $D-05 success" "106 $D-06 success"
    j a_green_on_another_branch_does_not_count         2 "night $D-04 has no completed run on main (no run)" branch
    j the_branch_row_is_reported_as_not_counted        2 "not counted: 1 other row(s), 0 after --as-of" branch
    fx dispatch "103 $D-03 success" "104 $D-04 success 1 main workflow_dispatch" "105 $D-05 success" "106 $D-06 success"
    j a_manual_dispatch_does_not_count                 2 "night $D-04 has no completed run on main (no run)" dispatch
    fx red_last "103 $D-03 success" "104 $D-04 success" "105 $D-05 success" "106 $D-06 failure"
    j a_red_newest_night_is_not_ready                  1 "not ready: night $D-06 was red (run 106, failure) (streak=0 total=3 need=3)" red_last
    j rows_after_as_of_are_not_counted                 0 "runs 105 104 103 (streak=3 total=3 need=3)" red_last "$D-05"
    j the_later_row_is_reported_as_not_counted         0 "not counted: 0 other row(s), 1 after --as-of" red_last "$D-05"
    fx double "104 $D-04 success" "105 $D-05 success" "150 $D-05 failure 1 main schedule 03:40:00" "106 $D-06 success"
    j one_red_run_makes_its_night_red                  1 "night $D-05 was red (run 150, failure)" double
    fx double_green "104 $D-04 success" "105 $D-05 success" "155 $D-05 success 1 main schedule 03:40:00" "106 $D-06 success"
    j a_night_counts_once_whatever_its_runs            0 "runs 106 155 104 (streak=3 total=3 need=3)" double_green
    fx half_done "103 $D-03 success" "104 $D-04 success" "140 $D-04 - 1 main schedule 03:40:00" "105 $D-05 success" "106 $D-06 success"
    j a_night_with_a_run_still_going_is_not_green_yet  2 "night $D-04 has no completed run on main (run 140 has not finished)" half_done
    fx apart "102 $D-02 success" "103 $D-03 success" "104 $D-04 failure" "105 $D-05 success" "106 $D-06 success"
    j greens_not_in_a_row_print_both_counts            1 "(streak=2 total=4 need=3)" apart
    fx stuck "103 $D-03 success" "104 $D-04 -" "105 $D-05 success" "106 $D-06 success"
    j a_run_that_never_finished_is_not_measured        2 "night $D-04 has no completed run on main (run 104 has not finished)" stuck
    fx cancelled "103 $D-03 success" "104 $D-04 cancelled" "105 $D-05 success" "106 $D-06 success"
    j a_cancelled_night_is_not_measured                2 "night $D-04 has no completed run on main (run 104 was cancelled)" cancelled
    fx cancelled_last "103 $D-03 success" "104 $D-04 success" "105 $D-05 success" "106 $D-06 cancelled"
    j a_cancelled_newest_night_keeps_the_streak        0 "runs 105 104 103" cancelled_last
    fx timed_out "104 $D-04 timed_out" "105 $D-05 success" "106 $D-06 success"
    j a_timed_out_night_is_red                         1 "night $D-04 was red (run 104, timed_out)" timed_out
    fx none
    j no_run_at_all_is_not_measured                    2 "not_measured: no green or red nightly run on main up to $D-06 (streak=0 total=0 need=3)" none
    : > "$tmp/empty"
    j an_empty_file_is_not_measured                    2 "not_measured: the history file is empty" empty
    fx bogus "104 $D-04 bogus"
    j an_unknown_conclusion_is_not_measured            2 "not_measured: unreadable history, line 2 conclusion \"bogus\"" bogus
    printf 'run_id\tcreated_at\tbranch\tevent\tconclusion\tattempt\n104\t%s-04T02:17:00Z\tmain\tschedule\tsuccess\n' "$D" > "$tmp/short"
    j a_short_line_is_not_measured                     2 "not_measured: unreadable history, line 2 has 5 fields, not 6" short
    fx badday "104 2026-09-31 success"
    j an_impossible_run_date_is_not_measured           2 "unreadable history, line 2 created_at \"2026-09-31T02:17:00Z\"" badday
    printf 'run_id\tcreated_at\tbranch\tevent\tattempt\tconclusion\n' > "$tmp/header"
    j a_wrong_header_is_a_caller_error                 3 "FAIL  NIGHTLY R7 caller error: the history header" header
    row no_as_of_is_a_caller_error                     3 "caller error: --as-of is required" -- --check R7 --history "$tmp/three"
    row an_impossible_as_of_is_a_caller_error          3 "caller error: --as-of \"2026-02-30\" is not a real date" -- --check R7 --history "$tmp/three" --as-of 2026-02-30
    row a_check_name_with_a_space_is_a_caller_error    3 "caller error: --check 'R 7'" -- --check "R 7" --history "$tmp/three" --as-of "$D-06"
    row a_missing_history_is_a_caller_error            3 "caller error: --history '$tmp/nope' is not a readable file" -- --check R7 --history "$tmp/nope" --as-of "$D-06"
    row need_is_not_an_option                          3 "caller error: unknown option --need" -- --check R7 --history "$tmp/two" --as-of "$D-06" --need 2
    row need_is_not_read_from_the_environment          1 "(streak=2 total=2 need=3)" NEED=2 -- --check R7 --history "$tmp/two" --as-of "$D-06"
    row help_prints_the_header                         0 "nightly_greens.sh --mutants" -- --help
    fx year "201 2026-12-31 success" "202 2027-01-01 success" "203 2027-01-02 success"
    j dates_cross_a_year_end                           0 "runs 203 202 201" year 2027-01-02
    fx leap "301 2028-02-28 success" "302 2028-02-29 success" "303 2028-03-01 success"
    j dates_cross_a_leap_day                           0 "runs 303 302 301" leap 2028-03-01
    fx noleap "400 2027-02-27 success" "401 2027-02-28 success" "402 2027-03-01 success"
    j a_february_without_a_leap_day_has_28_days        0 "runs 402 401 400" noleap 2027-03-01
    if [ "$NEED" = 3 ]; then
        printf '  ok    %-50s NEED=3\n' committed_need_is_three; pass=$((pass + 1))
    else
        printf '  BROKE %-50s NEED=%s, the ruling says three\n' committed_need_is_three "$NEED"; fail=$((fail + 1))
    fi
    # C355 item 6: release-rehearsal needs one night; every other check still needs three
    jr() { row "$1" "$2" "$3" -- --check release-rehearsal --history "$tmp/$4" --as-of "${5:-$D-06}"; }
    fx one "106 $D-06 success"
    jr rehearsal_one_green_night_is_ready              0 "ok    NIGHTLY release-rehearsal ready: 1 green nights in a row on main, newest $D-06: runs 106 (streak=1 total=1 need=1)" one
    jr rehearsal_a_red_newest_night_is_not_ready       1 "not ready: night $D-06 was red (run 106, failure) (streak=0 total=3 need=1)" red_last
    jr rehearsal_no_night_is_not_measured              2 "(streak=0 total=0 need=1)" none
    j  other_checks_still_need_three_after_one_night   1 "(streak=1 total=1 need=3)" one
    # The one-night need is keyed on the exact name: a sibling check or a near-miss name still needs three.
    jn() { row "$1" "$2" "$3" -- --check "$4" --history "$tmp/one" --as-of "$D-06"; }
    jn nightly_train_still_needs_three_after_one_night  1 "(streak=1 total=1 need=3)" nightly-train
    jn near_miss_release_lanes_needs_three              1 "(streak=1 total=1 need=3)" release-lanes
    jn near_miss_rehearsal_suffix_needs_three           1 "(streak=1 total=1 need=3)" release-rehearsal-old
    jn near_miss_other_prefix_rehearsal_needs_three     1 "(streak=1 total=1 need=3)" x-rehearsal
    jn near_miss_capital_release_rehearsal_needs_three  1 "(streak=1 total=1 need=3)" Release-rehearsal
    jn near_miss_last_letter_needs_three                1 "(streak=1 total=1 need=3)" release-rehearsaX
    if [ "$REHEARSAL_NEED" = 1 ]; then
        printf '  ok    %-50s REHEARSAL_NEED=1\n' committed_rehearsal_need_is_one; pass=$((pass + 1))
    else
        printf '  BROKE %-50s REHEARSAL_NEED=%s, C355 item 6 says one\n' committed_rehearsal_need_is_one "$REHEARSAL_NEED"; fail=$((fail + 1))
    fi
    printf -- '--- %s/%s rows ---\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ]
}

# ---------------------------------------------------------------- mutants ---
# Each mutant is "name sed-script". It must change this file, still parse, and turn --selftest RED with at least
# one BROKE row. A pattern that no longer matches is reported, never skipped: a mutant that changed nothing proves
# nothing.
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
            printf '  BROKE %-44s changed nothing: its pattern no longer matches\n' "$name"; fail=$((fail + 1)); continue
        fi
        if ! bash -n "$copy" 2>/dev/null; then
            printf '  BROKE %-44s does not parse: a RED from it would prove nothing\n' "$name"; fail=$((fail + 1)); continue
        fi
        rc=0; o="$(bash "$copy" --selftest < /dev/null 2>&1)" || rc=$?
        case "$rc:$o" in
            0:*) printf '  BROKE %-44s SURVIVED: the case table stayed green\n' "$name"; fail=$((fail + 1)) ;;
            *"  BROKE "*) printf '  ok    %-44s killed, %s row(s) broke\n' "$name" "$(printf '%s\n' "$o" | awk '/^  BROKE /{n++} END{print n+0}')"; pass=$((pass + 1)) ;;
            *) printf '  BROKE %-44s exit %s with no broken row: not a kill\n%s\n' "$name" "$rc" "$o"; fail=$((fail + 1)) ;;
        esac
    done <<'MUTANTS'
need_is_two                 /^NEED=3/s/^NEED=3/NEED=2/
need_from_env               /^NEED=3/s/^NEED=3/NEED=${NEED:-3}/
rehearsal_need_is_three     /^REHEARSAL_NEED=1/s/^REHEARSAL_NEED=1/REHEARSAL_NEED=3/
rehearsal_need_is_zero      /^REHEARSAL_NEED=1/s/^REHEARSAL_NEED=1/REHEARSAL_NEED=0/
rehearsal_name_not_keyed    /^need_of() /s/= release-rehearsal ]/= rehearsal ]/
rehearsal_and_train_keyed   /^need_of() /s/\[ "\$1" = release-rehearsal \]/{ [ "$1" = release-rehearsal ] || [ "$1" = nightly-train ]; }/
any_release_check_keyed     /^need_of() /s/\[ "\$1" = release-rehearsal \]/[[ $1 == release-* ]]/
rehearsal_prefix_keyed      /^need_of() /s/\[ "\$1" = release-rehearsal \]/[[ $1 == release-rehearsal* ]]/
rehearsal_suffix_keyed      /^need_of() /s/\[ "\$1" = release-rehearsal \]/[[ $1 == *rehearsal ]]/
one_letter_glob_keyed       /^need_of() /s/\[ "\$1" = release-rehearsal \]/[[ $1 == release-rehearsa? ]]/
case_glob_keyed             /^need_of() /s/\[ "\$1" = release-rehearsal \]/[[ $1 == [rR]elease-rehearsal ]]/
one_night_for_every_check   /^need_of() /s/"\$NEED"/"$REHEARSAL_NEED"/
retried_counts_as_green     /^judge() {$/,/^}$/s/? "green" : "retried"/? "green" : "green"/
failure_is_not_red          /^judge() {$/,/^}$/s/co == "failure" || co == "timed_out"/co == "never" || co == "timed_out"/
void_counts_as_green        /^judge() {$/,/^}$/s/) s = "void"/) s = "green"/
red_ranks_below_green       /^judge() {$/,/^}$/s/rank\["red"\] = 4/rank["red"] = 0/
pending_ranks_below_green   /^judge() {$/,/^}$/s/rank\["pending"\] = 3/rank["pending"] = 0/
tie_keeps_the_older_run     /^judge() {$/,/^}$/s/ts > tsof\[n\]/ts < tsof[n]/
every_branch_counts         /^judge() {$/,/^}$/s/if (\$3 != "main" || \$4 != "schedule")/if (0)/
later_rows_count            /^judge() {$/,/^}$/s/if (n > asof_n)/if (0)/
newest_red_is_skipped       /^judge() {$/,/^}$/s/st\[k\] == "green" || st\[k\] == "red" || st\[k\] == "retried"/st[k] == "green"/
no_staleness_bound          /^judge() {$/,/^}$/s/if (last < asof_n - 1)/if (0)/
absent_night_is_the_start   /^judge() {$/,/^}$/s/: ((k < first) ? "start" : "absent")/: "start"/
streak_off_by_one           /^judge() {$/,/^}$/s/if (streak >= need)/if (streak >= need - 1)/
leap_years_ignored          /^judge() {$/,/^}$/s/doe = yoe \* 365 + int(yoe \/ 4) - int(yoe \/ 100) + doy/doe = yoe * 365 + doy/
MUTANTS
    printf -- '--- %s/%s mutants killed ---\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ]
}

case "${1:-}" in
    --selftest) selftest ;;
    --mutants) mutants ;;
    -h|--help) awk 'NR > 1 && /^set -uo pipefail$/ { exit } NR > 1' "$0" ;;
    *) main "$@" ;;
esac
