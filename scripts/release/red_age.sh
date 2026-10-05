#!/usr/bin/env bash
# red_age.sh — how long has each check on `main` been red? An andon at 24 h, and the receipt for E2 (BLD-001 row 2)
#
# THE RULE. Operator ruling C277 item 6 (2026-10-03) lists the bars 0.70.2 ships on; one is "no check has been red on
#   main for more than 24 h". BLD-001 row 2 asks for a meter: for each check that reports on a `main` commit,
#   scheduled ones included, the age of its current red stretch (the first red after the last green, to now). At 24 h
#   the meter exits red and prints the check and its age. Its value at the tag is the receipt for BLD-001's exit bar E2
#   ("the longest red stretch in the 7 days before the tag is <= 24 h"). This script is that meter. It reads one run
#   history and judges every check in it.
#
# A RESULT. Only rows with branch `main` count, from any event except a pull request: a pull request from a fork's
#   branch named `main` reports on a merge commit, not on `main`. Rows dated after --as-of are not counted, so a replay
#   over a longer history gives the same answer. Each counted run is
#     green    it succeeded at attempt 1;
#     red      it ended failure, timed_out or startup_failure, or succeeded only at attempt 2 or later (a retried
#              green is not a green: docs/specifications/FLOW-003-queue-and-release-cycle-model.md, G4, plans CI
#              retries 2 -> 0);
#     pending  it has no conclusion yet;
#     void     cancelled, skipped, neutral, stale or action_required: it measured nothing.
#   Pending and void runs neither start nor end a red stretch.
#
# A STRETCH. A check's current red stretch starts at its oldest red result after its newest green one and runs to
#   --as-of. A red in the same second as a green counts as after it. A green ends a stretch. Row order in the file does
#   not matter, and the check lines come out in name order.
#
# VERDICT, per check; the first that holds wins:
#     not_measured  it has no green or red result in the WINDOW (7 days) to --as-of: a check with no result is never
#                   green;
#     red           its current stretch is LIMIT (24 h) or older. With no green before the stretch in the history its
#                   age is only a lower bound ("at least"), and it is still red;
#     not_measured  its current stretch has no green before it in the history and is under LIMIT: it may be older;
#     ok            red for under LIMIT, or green.
#   Each check line ends with max7d: the longest stretch open at any time in the WINDOW, measured from its true start,
#   never clipped to the WINDOW. ">=" marks a lower bound (no green before that stretch in the history).
#
# E2. One more line. E2 is met when no check's max7d is over LIMIT. It is "not met" when one is, and the line names
#   each such check and its stretch. It is not_measured when a check has no result in the WINDOW, or only a lower
#   bound under LIMIT. At exactly LIMIT the andon is red (row 2: "at 24 h it exits red") and E2 is still met (C277: "for
#   more than 24 h"). The E2 line is the receipt at the tag; it never changes the exit status, the andon does.
#   LIMIT AND WINDOW ARE CONSTANTS, not options and not environment values: a threshold a caller can raise is theater.
#   Changing one is a reviewed commit, and --selftest pins both.
#
# HISTORY. A TSV file. Its first line is exactly these seven tab-separated names; then one run per line:
#     check (letters, digits, '.', '_', '/', '-')  run_id  created_at (YYYY-MM-DDTHH:MM:SSZ, UTC)  branch  event
#     conclusion (empty while running)  attempt
#   A workflow run from the REST API maps onto it one to one after the check name: id, created_at, head_branch, event,
#   conclusion (null while running), run_attempt. Without its first column it is the history BLD-001 row 10 reads.
#
# EXIT  0 every check ok · 1 a check is red (red wins) · 2 not_measured (a stop: unknown is not a pass) · 3 caller
#   error (a wiring defect)
#
# USAGE
#   red_age.sh --history FILE --as-of YYYY-MM-DDTHH:MM:SSZ
#   red_age.sh --selftest    the case table
#   red_age.sh --mutants     each planted mutant of judge(), LIMIT or WINDOW must turn the case table RED
set -uo pipefail

PROG="${0##*/}"
SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/${BASH_SOURCE[0]##*/}"
# C277 item 6: "red on main for more than 24 h". A constant: never an option, never taken from the environment.
LIMIT=86400
# E2: "in the 7 days before the tag". A constant, like LIMIT.
WINDOW=604800

caller_error() { printf 'FAIL  REDAGE %s: caller error: %s\n' "$PROG" "$*"; exit 3; }

# judge history as-of -> one line per check, the E2 line and one context line on stdout; exits 0/1/2/3 as above
judge() {
    awk -v asof="$2" -v limit="$LIMIT" -v window="$WINDOW" -v prog="$PROG" '
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
    function istime(s) {    # a real UTC instant, written YYYY-MM-DDTHH:MM:SSZ
        if (s !~ /^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z$/) return 0
        if (!isdate(substr(s, 1, 10))) return 0
        return (substr(s, 12, 2) + 0 <= 23 && substr(s, 15, 2) + 0 <= 59 && substr(s, 18, 2) + 0 <= 59)
    }
    function secs(s) { return dayno(s) * 86400 + substr(s, 12, 2) * 3600 + substr(s, 15, 2) * 60 + substr(s, 18, 2) }
    function dur(x) { return sprintf("%dh%02dm%02ds", int(x / 3600), int((x % 3600) / 60), x % 60) }
    function add(list, x) { return (list == "") ? x : list ", " x }
    BEGIN {
        FS = "\t"
        want = "check\trun_id\tcreated_at\tbranch\tevent\tconclusion\tattempt"
        if (!istime(asof)) { err = "--as-of \"" asof "\" is not a real UTC time written YYYY-MM-DDTHH:MM:SSZ"; exit }
        now = secs(asof); from = now - window
    }
    NR == 1 {
        if ($0 != want) { err = "the history header is not the seven tab-separated columns this judge reads"; exit }
        next
    }
    {
        if (NF != 7) { bad = "line " NR " has " NF " fields, not 7"; exit }
        c = $1; id = $2; ts = $3; co = $6; at = $7
        if (c !~ "^[A-Za-z0-9._/-]+$") { bad = "line " NR " check \"" c "\""; exit }
        if (id !~ /^[A-Za-z0-9._-]+$/) { bad = "line " NR " run_id \"" id "\""; exit }
        if (!istime(ts)) { bad = "line " NR " created_at \"" ts "\""; exit }
        if (at !~ /^[1-9][0-9]*$/) { bad = "line " NR " attempt \"" at "\""; exit }
        if (co == "") s = "pending"
        else if (co == "success") s = (at == 1) ? "green" : "retried"
        else if (co == "failure" || co == "timed_out" || co == "startup_failure") s = "red"
        else if (co == "cancelled" || co == "skipped" || co == "neutral" || co == "stale" || co == "action_required") s = "void"
        else { bad = "line " NR " conclusion \"" co "\""; exit }
        if ($4 != "main") { other++; next }
        if ($5 ~ /^pull_request/) { other++; next }
        t = secs(ts)
        if (t > now) { later++; next }
        runs++
        if (!(c in n)) n[c] = 0
        if (s == "pending" || s == "void") { idle++; next }
        n[c]++; i = n[c]
        T[c, i] = t; K[c, i] = (s == "green") ? "g" : "r"; ID[c, i] = id; TS[c, i] = ts
        CO[c, i] = (s == "retried") ? "success only at attempt " at : co
    }
    END {
        if (err != "") { printf "FAIL  REDAGE %s: caller error: %s\n", prog, err; exit 3 }
        if (NR == 0) { print "FAIL  REDAGE not_measured: the history file is empty"; exit 2 }
        if (bad != "") { printf "FAIL  REDAGE not_measured: unreadable history, %s\n", bad; exit 2 }
        k = 0
        for (c in n) nm[++k] = c
        for (a = 2; a <= k; a++) {    # insertion sort: name order, whatever order the awk walks n in
            v = nm[a]
            for (b = a - 1; b >= 1 && nm[b] > v; b--) nm[b + 1] = nm[b]
            nm[b + 1] = v
        }
        info = sprintf("      REDAGE counted %d run(s) of %d check(s) on main up to %s, %d of them pending or void; not counted: %d other row(s), %d after --as-of", runs, k, asof, idle, other, later)
        if (k == 0) { printf "FAIL  REDAGE not_measured: no check reported on main up to %s\n", asof; print info; exit 2 }
        wd = int(window / 86400)
        nred = 0; nnm = 0; nno = 0; nnme = 0; best = -1; bestc = ""; e2no = ""; e2nm = ""
        for (a = 1; a <= k; a++) {
            c = nm[a]; m = n[c]
            hasg = 0; lg = 0; lgi = 0
            for (i = 1; i <= m; i++) if (K[c, i] == "g" && (!hasg || T[c, i] > lg)) { hasg = 1; lg = T[c, i]; lgi = i }
            hasr = 0; st = 0; si = 0; seen = 0; ni = 0
            for (i = 1; i <= m; i++) {
                if (K[c, i] == "r" && (!hasg || T[c, i] >= lg) && (!hasr || T[c, i] < st)) { hasr = 1; st = T[c, i]; si = i }
                if (T[c, i] >= from) seen++
                if (!ni || T[c, i] > T[c, ni]) ni = i
            }
            age = hasr ? now - st : 0
            mx = age; lb = (hasr && !hasg)
            for (i = 1; i <= m; i++) {    # each stretch a green in the WINDOW ended, from its true start
                if (K[c, i] != "g" || T[c, i] < from) continue
                hp = 0; pg = 0
                for (q = 1; q <= m; q++) if (K[c, q] == "g" && T[c, q] < T[c, i] && (!hp || T[c, q] > pg)) { hp = 1; pg = T[c, q] }
                hs = 0; s0 = 0
                for (q = 1; q <= m; q++) if (K[c, q] == "r" && T[c, q] < T[c, i] && (!hp || T[c, q] >= pg) && (!hs || T[c, q] < s0)) { hs = 1; s0 = T[c, q] }
                if (hs && T[c, i] - s0 > mx) mx = T[c, i] - s0
                if (hs && !hp) lb = 1
            }
            m7 = (lb ? ">=" : "=") dur(mx)
            if (seen == 0) {
                if (ni) why = "the newest is run " ID[c, ni] " at " TS[c, ni] " (" ((K[c, ni] == "g") ? "green" : CO[c, ni]) ")"
                else why = "it has none at all"
                printf "FAIL  REDAGE %s not_measured: no green or red result in the %d days to %s; %s; a check with no result is never green (max7d=not_measured)\n", c, wd, asof, why
                nnm++
            } else if (hasr && age >= limit) {
                printf "FAIL  REDAGE %s red: red for %s%s, since run %s at %s (%s); the andon fires at %s (max7d%s)\n", c, (hasg ? "" : "at least "), dur(age), ID[c, si], TS[c, si], CO[c, si], dur(limit), m7
                nred++
            } else if (hasr && !hasg) {
                printf "FAIL  REDAGE %s not_measured: red for at least %s, since run %s at %s (%s), and no green before it in the history: the stretch may be older (max7d%s)\n", c, dur(age), ID[c, si], TS[c, si], CO[c, si], m7
                nnm++
            } else if (hasr) {
                printf "ok    REDAGE %s ok: red for %s, since run %s at %s (%s); under %s (max7d%s)\n", c, dur(age), ID[c, si], TS[c, si], CO[c, si], dur(limit), m7
            } else {
                printf "ok    REDAGE %s ok: green, newest result run %s at %s (max7d%s)\n", c, ID[c, lgi], TS[c, lgi], m7
            }
            if (!seen) { nnme++; e2nm = add(e2nm, c " (no result)") }
            else if (mx > limit) { nno++; e2no = add(e2no, c " " dur(mx)) }
            else if (lb) { nnme++; e2nm = add(e2nm, c " (at least " dur(mx) ")") }
            else if (mx > best) { best = mx; bestc = c }
        }
        if (nno > 0) {
            line = sprintf("FAIL  REDAGE E2 not met: %d check(s) red for more than %s in the %d days to %s: %s", nno, dur(limit), wd, asof, e2no)
            if (nnme > 0) line = line "; not measured: " e2nm
            print line
        } else if (nnme > 0) {
            printf "FAIL  REDAGE E2 not_measured: %d check(s) not measured in the %d days to %s: %s\n", nnme, wd, asof, e2nm
        } else {
            printf "ok    REDAGE E2 met: no check was red for more than %s in the %d days to %s (%d check(s); longest %s, %s)\n", dur(limit), wd, asof, k, dur(best), bestc
        }
        print info
        if (nred > 0) exit 1
        if (nnm > 0) exit 2
        exit 0
    }' "$1"
}

main() {
    local history="" asof=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --history|--as-of)
                [ $# -ge 2 ] || caller_error "$1 needs a value"
                case "$1" in --history) history="$2" ;; *) asof="$2" ;; esac
                shift 2 ;;
            *) caller_error "unknown option $1 (usage: $PROG --history FILE --as-of YYYY-MM-DDTHH:MM:SSZ)" ;;
        esac
    done
    [ -n "$asof" ] || caller_error "--as-of is required: a judgment is made for a named instant, never an implicit now"
    { [ -n "$history" ] && [ -f "$history" ] && [ -r "$history" ]; } || caller_error "--history '$history' is not a readable file"
    judge "$history" "$asof"
}

# --------------------------------------------------------------- selftest ---
selftest_cleanup() { case "${tmp:-}" in ''|/) return 0 ;; *) rm -rf -- "${tmp:?}" ;; esac; }
selftest() {
    local tmp pass=0 fail=0 D=2026-10 A=2026-10-06T12:00:00Z
    tmp="$(mktemp -d)"
    case "$tmp" in ?*/tmp.?*) : ;; *) echo "mktemp gave '${tmp:-}', refusing"; return 2 ;; esac
    # SEC011: validate before rm -rf. $tmp is this function's own mktemp -d (checked above).
    trap selftest_cleanup RETURN
    fx() { # name, then runs "check id YYYY-MM-DDTHH:MM:SS conclusion [attempt [branch [event]]]"; conclusion - = running
        local f="$tmp/$1" r c id ts co at br ev; shift
        printf 'check\trun_id\tcreated_at\tbranch\tevent\tconclusion\tattempt\n' > "$f"
        for r in "$@"; do
            read -r c id ts co at br ev <<< "$r"
            [ "$co" = - ] && co=""
            printf '%s\t%s\t%sZ\t%s\t%s\t%s\t%s\n' "$c" "$id" "$ts" "${br:-main}" "${ev:-push}" "$co" "${at:-1}" >> "$f"
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
    j() { row "$1" "$2" "$3" -- --history "$tmp/$4" --as-of "${5:-$A}"; }

    fx red25 "ci.yml 101 $D-01T09:00:00 success" "ci.yml 102 $D-05T11:00:00 failure"
    j red_for_25h_is_red                               1 "FAIL  REDAGE ci.yml red: red for 25h00m00s, since run 102 at $D-05T11:00:00Z (failure); the andon fires at 24h00m00s (max7d=25h00m00s)" red25
    j red_for_25h_breaks_e2                            1 "FAIL  REDAGE E2 not met: 1 check(s) red for more than 24h00m00s in the 7 days to $A: ci.yml 25h00m00s" red25
    j the_context_line_counts_what_was_read            1 "      REDAGE counted 2 run(s) of 1 check(s) on main up to $A, 0 of them pending or void; not counted: 0 other row(s), 0 after --as-of" red25
    fx red23 "ci.yml 101 $D-01T09:00:00 success" "ci.yml 102 $D-05T13:00:00 failure"
    j red_for_23h_is_ok                                0 "ok    REDAGE ci.yml ok: red for 23h00m00s, since run 102 at $D-05T13:00:00Z (failure); under 24h00m00s (max7d=23h00m00s)" red23
    j red_for_23h_meets_e2                             0 "ok    REDAGE E2 met: no check was red for more than 24h00m00s in the 7 days to $A (1 check(s); longest 23h00m00s, ci.yml)" red23
    fx exact24 "ci.yml 101 $D-01T09:00:00 success" "ci.yml 102 $D-05T12:00:00 failure"
    j red_for_exactly_24h_fires_the_andon              1 "FAIL  REDAGE ci.yml red: red for 24h00m00s, since run 102" exact24
    j red_for_exactly_24h_still_meets_e2               1 "ok    REDAGE E2 met: no check was red for more than 24h00m00s in the 7 days to $A (1 check(s); longest 24h00m00s, ci.yml)" exact24
    fx under24 "ci.yml 101 $D-01T09:00:00 success" "ci.yml 102 $D-05T12:00:01 failure"
    j one_second_under_24h_is_ok                       0 "ok    REDAGE ci.yml ok: red for 23h59m59s" under24
    fx reset "ci.yml 101 $D-01T10:00:00 success" "ci.yml 102 $D-01T11:00:00 failure" "ci.yml 103 $D-03T12:00:00 success" "ci.yml 104 $D-05T09:00:00 failure" "ci.yml 105 $D-06T11:00:00 success"
    j a_green_resets_the_age                           0 "ok    REDAGE ci.yml ok: green, newest result run 105 at $D-06T11:00:00Z (max7d=49h00m00s)" reset
    j a_closed_49h_stretch_breaks_e2                   0 "FAIL  REDAGE E2 not met: 1 check(s) red for more than 24h00m00s in the 7 days to $A: ci.yml 49h00m00s" reset
    fx shuffled "ci.yml 104 $D-05T09:00:00 failure" "ci.yml 101 $D-01T10:00:00 success" "ci.yml 105 $D-06T11:00:00 success" "ci.yml 102 $D-01T11:00:00 failure" "ci.yml 103 $D-03T12:00:00 success"
    j row_order_does_not_matter                        0 "ok    REDAGE ci.yml ok: green, newest result run 105 at $D-06T11:00:00Z (max7d=49h00m00s)" shuffled
    fx closed24 "ci.yml 101 $D-01T09:00:00 success" "ci.yml 102 $D-02T10:00:00 failure" "ci.yml 103 $D-03T10:00:00 success"
    j a_closed_stretch_of_exactly_24h_meets_e2         0 "E2 met: no check was red for more than 24h00m00s in the 7 days to $A (1 check(s); longest 24h00m00s, ci.yml)" closed24
    fx closed24s "ci.yml 101 $D-01T09:00:00 success" "ci.yml 102 $D-02T10:00:00 failure" "ci.yml 103 $D-03T10:00:01 success"
    j a_closed_stretch_1s_over_24h_breaks_e2           0 "E2 not met: 1 check(s) red for more than 24h00m00s in the 7 days to $A: ci.yml 24h00m01s" closed24s
    fx retried "ci.yml 101 $D-01T09:00:00 success" "ci.yml 102 $D-05T10:00:00 failure" "ci.yml 103 $D-05T20:00:00 success 2"
    j a_retried_green_does_not_end_a_stretch           1 "red: red for 26h00m00s, since run 102 at $D-05T10:00:00Z (failure)" retried
    fx retried_first "ci.yml 101 $D-01T09:00:00 success" "ci.yml 102 $D-05T10:00:00 success 2"
    j a_retried_green_starts_a_stretch                 1 "red: red for 26h00m00s, since run 102 at $D-05T10:00:00Z (success only at attempt 2)" retried_first
    fx timed_out "ci.yml 101 $D-01T09:00:00 success" "ci.yml 102 $D-05T10:00:00 timed_out"
    j a_timed_out_run_is_red                           1 "red: red for 26h00m00s, since run 102 at $D-05T10:00:00Z (timed_out)" timed_out
    fx startup "ci.yml 101 $D-01T09:00:00 success" "ci.yml 102 $D-05T10:00:00 startup_failure"
    j a_startup_failure_is_red                         1 "red: red for 26h00m00s, since run 102 at $D-05T10:00:00Z (startup_failure)" startup
    fx void_inside "ci.yml 101 $D-01T09:00:00 success" "ci.yml 102 $D-05T11:00:00 failure" "ci.yml 103 $D-05T20:00:00 cancelled" "ci.yml 104 $D-06T10:00:00 -"
    j void_and_pending_runs_do_not_end_a_stretch       1 "red: red for 25h00m00s, since run 102" void_inside
    j they_are_counted_as_pending_or_void              1 "counted 4 run(s) of 1 check(s) on main up to $A, 2 of them pending or void" void_inside
    fx cancelled "ci.yml 101 $D-05T09:00:00 success" "ci.yml 102 $D-05T10:00:00 cancelled"
    j a_cancelled_run_does_not_start_a_stretch         0 "ok    REDAGE ci.yml ok: green, newest result run 101 at $D-05T09:00:00Z" cancelled
    fx idle "ci.yml 101 $D-05T10:00:00 cancelled" "ci.yml 102 $D-06T10:00:00 -"
    j only_void_or_pending_runs_are_not_measured       2 "FAIL  REDAGE ci.yml not_measured: no green or red result in the 7 days to $A; it has none at all; a check with no result is never green (max7d=not_measured)" idle
    fx stale "ci.yml 101 2026-09-28T10:00:00 success"
    j no_result_in_7_days_is_not_measured              2 "FAIL  REDAGE ci.yml not_measured: no green or red result in the 7 days to $A; the newest is run 101 at 2026-09-28T10:00:00Z (green); a check with no result is never green" stale
    j an_unmeasured_check_leaves_e2_not_measured       2 "FAIL  REDAGE E2 not_measured: 1 check(s) not measured in the 7 days to $A: ci.yml (no result)" stale
    fx edge_out "ci.yml 101 2026-09-29T00:00:00 success"
    j a_result_older_than_7_days_is_not_measured       2 "the newest is run 101 at 2026-09-29T00:00:00Z (green)" edge_out
    fx edge_in "ci.yml 101 2026-09-29T12:00:00 success"
    j a_result_exactly_7_days_old_is_measured          0 "ok    REDAGE ci.yml ok: green, newest result run 101 at 2026-09-29T12:00:00Z (max7d=0h00m00s)" edge_in
    fx branch "ci.yml 101 $D-01T09:00:00 success" "ci.yml 102 $D-05T11:00:00 failure 1 feature"
    j a_red_on_another_branch_does_not_count           0 "ok    REDAGE ci.yml ok: green, newest result run 101" branch
    j the_branch_row_is_reported_as_not_counted        0 "not counted: 1 other row(s), 0 after --as-of" branch
    fx forkpr "ci.yml 101 $D-01T09:00:00 success" "ci.yml 102 $D-05T11:00:00 failure 1 main pull_request"
    j a_pull_request_from_a_main_branch_is_not_main    0 "ok    REDAGE ci.yml ok: green, newest result run 101" forkpr
    fx sched "ci.yml 101 $D-01T09:00:00 success" "ci.yml 102 $D-05T11:00:00 failure 1 main schedule"
    j a_scheduled_run_on_main_counts                   1 "red: red for 25h00m00s, since run 102" sched
    fx later "ci.yml 101 $D-01T09:00:00 success" "ci.yml 102 $D-05T11:00:00 failure" "ci.yml 103 $D-06T13:00:00 success"
    j rows_after_as_of_are_not_counted                 1 "red: red for 25h00m00s, since run 102" later
    j the_later_row_is_reported_as_not_counted         1 "not counted: 0 other row(s), 1 after --as-of" later
    j a_later_as_of_sees_the_green                     0 "ok    REDAGE ci.yml ok: green, newest result run 103 at $D-06T13:00:00Z" later "$D-06T14:00:00Z"
    fx lb_under "ci.yml 101 $D-06T00:00:00 failure"
    j red_from_the_first_row_is_a_lower_bound          2 "FAIL  REDAGE ci.yml not_measured: red for at least 12h00m00s, since run 101 at $D-06T00:00:00Z (failure), and no green before it in the history: the stretch may be older (max7d>=12h00m00s)" lb_under
    fx lb_over "ci.yml 101 $D-05T00:00:00 failure"
    j a_lower_bound_over_24h_is_red                    1 "FAIL  REDAGE ci.yml red: red for at least 36h00m00s, since run 101 at $D-05T00:00:00Z (failure); the andon fires at 24h00m00s (max7d>=36h00m00s)" lb_over
    j a_lower_bound_over_24h_breaks_e2                 1 "E2 not met: 1 check(s) red for more than 24h00m00s in the 7 days to $A: ci.yml 36h00m00s" lb_over
    fx lb_e2 "ci.yml 101 $D-02T00:00:00 failure" "ci.yml 102 $D-02T10:00:00 success" "ci.yml 103 $D-06T10:00:00 success"
    j a_closed_lower_bound_leaves_e2_not_measured      0 "FAIL  REDAGE E2 not_measured: 1 check(s) not measured in the 7 days to $A: ci.yml (at least 10h00m00s)" lb_e2
    j its_check_line_marks_the_bound                   0 "ok    REDAGE ci.yml ok: green, newest result run 103 at $D-06T10:00:00Z (max7d>=10h00m00s)" lb_e2
    fx tie "ci.yml 101 $D-05T11:00:00 success" "ci.yml 102 $D-05T11:00:00 failure"
    j a_red_in_a_green_s_second_counts_after_it        1 "red: red for 25h00m00s, since run 102" tie
    fx unclipped "ci.yml 100 2026-09-26T09:00:00 success" "ci.yml 101 2026-09-27T12:00:00 failure" "ci.yml 102 2026-09-29T18:00:00 success" "ci.yml 103 $D-06T10:00:00 success"
    j a_stretch_into_the_window_counts_in_full         0 "E2 not met: 1 check(s) red for more than 24h00m00s in the 7 days to $A: ci.yml 54h00m00s" unclipped
    fx before "ci.yml 100 2026-09-25T09:00:00 success" "ci.yml 101 2026-09-26T12:00:00 failure" "ci.yml 102 2026-09-29T11:00:00 success" "ci.yml 103 $D-06T10:00:00 success"
    j a_stretch_ended_before_the_window_is_out         0 "E2 met: no check was red for more than 24h00m00s in the 7 days to $A (1 check(s); longest 0h00m00s, ci.yml)" before
    fx two_red "a.yml 100 $D-01T09:00:00 success" "a.yml 101 $D-05T11:00:00 failure" "b.yml 201 $D-06T10:00:00 success"
    j one_red_check_makes_the_exit_red                 1 "FAIL  REDAGE a.yml red: red for 25h00m00s" two_red
    j the_other_check_is_still_judged                  1 "ok    REDAGE b.yml ok: green, newest result run 201" two_red
    fx two_nm "a.yml 101 2026-09-28T10:00:00 success" "b.yml 201 $D-06T10:00:00 success"
    j one_unmeasured_check_makes_the_exit_2            2 "FAIL  REDAGE a.yml not_measured: no green or red result" two_nm
    fx two_rednm "a.yml 100 $D-01T09:00:00 success" "a.yml 101 $D-05T11:00:00 failure" "b.yml 201 2026-09-28T10:00:00 success"
    j red_beats_not_measured                           1 "E2 not met: 1 check(s) red for more than 24h00m00s in the 7 days to $A: a.yml 25h00m00s; not measured: b.yml (no result)" two_rednm
    fx order "b.yml 201 $D-06T10:00:00 success" "a.yml 101 $D-06T09:00:00 success"
    j checks_come_out_in_name_order                    0 "a.yml ok: green, newest result run 101 at $D-06T09:00:00Z (max7d=0h00m00s)"$'\n'"ok    REDAGE b.yml ok" order
    j e2_names_the_checks_it_judged                    0 "E2 met: no check was red for more than 24h00m00s in the 7 days to $A (2 check(s); longest 0h00m00s, a.yml)" order
    fx none
    j a_header_alone_is_not_measured                   2 "FAIL  REDAGE not_measured: no check reported on main up to $A" none
    fx only_branch "ci.yml 101 $D-05T10:00:00 success 1 feature"
    j no_row_on_main_is_not_measured                   2 "FAIL  REDAGE not_measured: no check reported on main up to $A" only_branch
    : > "$tmp/empty"
    j an_empty_file_is_not_measured                    2 "FAIL  REDAGE not_measured: the history file is empty" empty
    printf 'run_id\tcreated_at\tbranch\tevent\tconclusion\tattempt\n' > "$tmp/six"
    j a_six_column_history_is_a_caller_error           3 "caller error: the history header is not the seven tab-separated columns this judge reads" six
    fx bogus "ci.yml 101 $D-05T10:00:00 bogus"
    j an_unknown_conclusion_is_not_measured            2 "FAIL  REDAGE not_measured: unreadable history, line 2 conclusion \"bogus\"" bogus
    printf 'check\trun_id\tcreated_at\tbranch\tevent\tconclusion\tattempt\nci.yml\t101\t%s-05T10:00:00Z\tmain\tpush\tsuccess\n' "$D" > "$tmp/short"
    j a_short_line_is_not_measured                     2 "unreadable history, line 2 has 6 fields, not 7" short
    fx badday "ci.yml 101 2026-09-31T10:00:00 success"
    j an_impossible_run_date_is_not_measured           2 "unreadable history, line 2 created_at \"2026-09-31T10:00:00Z\"" badday
    fx hour24 "ci.yml 101 $D-05T24:00:00 success"
    j hour_24_is_not_a_time                            2 "unreadable history, line 2 created_at \"$D-05T24:00:00Z\"" hour24
    fx badcheck "ci;yml 101 $D-05T10:00:00 success"
    j a_check_name_with_a_semicolon_is_unreadable      2 "unreadable history, line 2 check \"ci;yml\"" badcheck
    fx badattempt "ci.yml 101 $D-05T10:00:00 success 0"
    j attempt_zero_is_unreadable                       2 "unreadable history, line 2 attempt \"0\"" badattempt
    row no_as_of_is_a_caller_error                     3 "caller error: --as-of is required" -- --history "$tmp/red25"
    row a_date_only_as_of_is_a_caller_error            3 "caller error: --as-of \"2026-10-06\" is not a real UTC time written YYYY-MM-DDTHH:MM:SSZ" -- --history "$tmp/red25" --as-of 2026-10-06
    row an_impossible_as_of_is_a_caller_error          3 "caller error: --as-of \"2026-02-30T00:00:00Z\" is not a real UTC time" -- --history "$tmp/red25" --as-of 2026-02-30T00:00:00Z
    row an_as_of_at_hour_24_is_a_caller_error          3 "caller error: --as-of \"$D-06T24:00:00Z\" is not a real UTC time" -- --history "$tmp/red25" --as-of "$D-06T24:00:00Z"
    row a_missing_history_is_a_caller_error            3 "caller error: --history '$tmp/nope' is not a readable file" -- --history "$tmp/nope" --as-of "$A"
    row an_option_without_a_value_is_a_caller_error    3 "caller error: --as-of needs a value" -- --history "$tmp/red25" --as-of
    row limit_is_not_an_option                         3 "caller error: unknown option --limit" -- --history "$tmp/red25" --as-of "$A" --limit 172800
    row window_is_not_an_option                        3 "caller error: unknown option --window" -- --history "$tmp/red25" --as-of "$A" --window 691200
    row limit_is_not_read_from_the_environment         1 "red: red for 25h00m00s" LIMIT=172800 -- --history "$tmp/red25" --as-of "$A"
    row window_is_not_read_from_the_environment        2 "the newest is run 101 at 2026-09-29T00:00:00Z (green)" WINDOW=691200 -- --history "$tmp/edge_out" --as-of "$A"
    row help_prints_the_header                         0 "red_age.sh --mutants" -- --help
    fx year "ci.yml 101 2026-12-30T09:00:00 success" "ci.yml 102 2026-12-31T13:00:00 failure"
    j dates_cross_a_year_end                           0 "ok: red for 23h00m00s, since run 102" year 2027-01-01T12:00:00Z
    fx leap "ci.yml 101 2028-02-27T09:00:00 success" "ci.yml 102 2028-02-28T12:00:00 failure"
    j a_leap_day_has_24_hours                          1 "red: red for 48h00m00s, since run 102" leap 2028-03-01T12:00:00Z
    fx noleap "ci.yml 101 2027-02-26T09:00:00 success" "ci.yml 102 2027-02-28T12:00:00 failure"
    j a_february_without_a_leap_day_has_28_days        1 "red: red for 24h00m00s, since run 102" noleap 2027-03-01T12:00:00Z
    if [ "$LIMIT" = 86400 ] && [ "$WINDOW" = 604800 ]; then
        printf '  ok    %-50s LIMIT=86400 WINDOW=604800\n' committed_limit_is_24h_and_window_7_days; pass=$((pass + 1))
    else
        printf '  BROKE %-50s LIMIT=%s WINDOW=%s, the rulings say 24 h and 7 days\n' committed_limit_is_24h_and_window_7_days "$LIMIT" "$WINDOW"; fail=$((fail + 1))
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
limit_is_48h                /^LIMIT=86400/s/^LIMIT=86400/LIMIT=172800/
limit_one_second_late       /^LIMIT=86400/s/^LIMIT=86400/LIMIT=86401/
limit_from_env              /^LIMIT=86400/s/^LIMIT=86400/LIMIT=${LIMIT:-86400}/
window_is_8_days            /^WINDOW=604800/s/^WINDOW=604800/WINDOW=691200/
window_from_env             /^WINDOW=604800/s/^WINDOW=604800/WINDOW=${WINDOW:-604800}/
retried_counts_as_green     /^judge() {$/,/^}$/s/? "green" : "retried"/? "green" : "green"/
failure_is_not_red          /^judge() {$/,/^}$/s/co == "failure" || co == "timed_out"/co == "never" || co == "timed_out"/
void_counts_as_green        /^judge() {$/,/^}$/s/) s = "void"/) s = "green"/
pending_counts_as_green     /^judge() {$/,/^}$/s/if (co == "") s = "pending"/if (co == "") s = "green"/
every_branch_counts         /^judge() {$/,/^}$/s/if (\$4 != "main")/if (0)/
pull_requests_count         /^judge() {$/,/^}$/s|if (\$5 ~ /^pull_request/)|if (0)|
later_rows_count            /^judge() {$/,/^}$/s/if (t > now)/if (0)/
green_does_not_reset        /^judge() {$/,/^}$/s/(!hasg || T\[c, i\] >= lg)/(1)/
tie_puts_green_after_red    /^judge() {$/,/^}$/s/T\[c, i\] >= lg/T[c, i] > lg/
newest_green_is_the_oldest  /^judge() {$/,/^}$/s/T\[c, i\] > lg)/T[c, i] < lg)/
andon_past_24h              /^judge() {$/,/^}$/s/age >= limit/age > limit/
e2_at_24h                   /^judge() {$/,/^}$/s/mx > limit/mx >= limit/
no_window                   /^judge() {$/,/^}$/s/if (seen == 0) {/if (0) {/
window_edge_exclusive       /^judge() {$/,/^}$/s/if (T\[c, i\] >= from) seen++/if (T[c, i] > from) seen++/
lower_bound_trusted         /^judge() {$/,/^}$/s/} else if (hasr && !hasg) {/} else if (0) {/
lower_bound_e2_trusted      /^judge() {$/,/^}$/s/if (hs && !hp) lb = 1/if (0) lb = 1/
e2_current_age_only         /^judge() {$/,/^}$/s/if (hs && T\[c, i\] - s0 > mx)/if (0)/
e2_clips_start_to_window    /^judge() {$/,/^}$/s/if (K\[c, q\] == "r" && /if (K[c, q] == "r" \&\& T[c, q] >= from \&\& /
e2_counts_before_window     /^judge() {$/,/^}$/s/if (K\[c, i\] != "g" || T\[c, i\] < from) continue/if (K[c, i] != "g") continue/
worst_of                    /^judge() {$/,/^}$/s/if (nred > 0) exit 1/if (0) exit 1/
seconds_ignored             /^judge() {$/,/^}$/s/ + substr(s, 18, 2) }/ + 0 }/
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
