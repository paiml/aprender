#!/usr/bin/env bash
# dogfood_timing_lane.sh -- the nightly dogfood row-time lane, report mode.
#
# WHAT IT DOES. After the night's dogfood run, it finds that night's receipt, hands the row-time sidecar
#   (receipt-<TS>.timing.tsv, written beside the receipt by scripts/dogfood.sh) to
#   check_dogfood_row_order.sh --timing, and prints ONE line:
#     DOGFOOD-TIMING GREEN|OVER|BASELINE night=<N> cheap=+<s>s of +<s>s budget=<s>s|none (<why>)
#     DOGFOOD-TIMING NOT_MEASURED night=<N>: <reason>
#   GREEN   the cheap tier closed within the budget.       OVER  it closed after the budget (reported, never blocks).
#   BASELINE fewer than MIN_NIGHTS earlier nights were measured, so there is no budget yet.
#
# THE BUDGET COMES FROM MEASURED DURATIONS, never from a constant: the largest cheap-tier close of the latest
#   WINDOW earlier nights whose own verdict was GREEN or BASELINE, plus HEADROOM_PCT, rounded up. Tonight never sets its
#   own bar, an OVER night never raises the bar, and a NOT_MEASURED night is never a duration. The latest row on a
#   night decides it. History: --history FILE, one row per run (night, receipt, cheap_s, total_s, budget, state).
#
# NOT MEASURED IS NEVER A PASS. No receipt for tonight's night (a stale receipt is not tonight), no sidecar, a sidecar
#   the checker cannot read, or no checker, is NOT_MEASURED (exit 2). night(t) = UTC date of (t - 12 h).
#
# REPORT MODE. The checker runs with DOGFOOD_ROW_ORDER_ENFORCE=0 whatever the caller's environment says. OVER exits 0.
#   Blocking is a separate, signed-off step after three green nights; this script has no switch for it.
#
# EXIT 0 a measured line (GREEN, OVER, BASELINE) · 2 NOT_MEASURED · 3 caller error
#
# USAGE
#   dogfood_timing_lane.sh --receipts DIR --history FILE [--now ISO] [--checker PATH] [--inbox FILE]
#   dogfood_timing_lane.sh --self-test     the case table (fixtures and a stub checker, no dogfood run)
#   dogfood_timing_lane.sh --mutants       each planted mutant must turn the case table RED
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$HERE/${BASH_SOURCE[0]##*/}"
WINDOW=7
MIN_NIGHTS=3
HEADROOM_PCT=25

caller_error() { printf 'DOGFOOD-TIMING NOT_MEASURED: caller error: %s\n' "$*"; exit 3; }
row() { printf '%s\n' "$1" > "$ROWF"; }
note() { printf 'DT %s\n' "$*" >&2; }

night_of() { local e; [ -n "$1" ] || return 3   # date reads an empty string as midnight today
    e="$(date -u -d "$1" +%s 2>/dev/null)" || return 3; date -u -d "@$((e - 43200))" +%F; }

# receipt_time receipt-YYYYMMDDTHHMMSSZ.json -> YYYY-MM-DDTHH:MM:SSZ (empty when the name is not a receipt stamp)
receipt_time() {
    printf '%s\n' "${1##*/}" | sed -n 's/^receipt-\([0-9]\{4\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)T\([0-9]\{2\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)Z\.json$/\1-\2-\3T\4:\5:\6Z/p'
}

# budget HISTORY NIGHT -> "<budget> <k>" from the latest WINDOW earlier nights judged GREEN or BASELINE; "none <k>" below MIN
budget() {
    [ -f "$1" ] || { printf 'none 0\n'; return 0; }
    awk -F '\t' -v T="$2" '$1 != "night" && $1 < T { s[$1] = $6; v[$1] = $3 }
        END { for (k in s) if ((s[k] == "GREEN" || s[k] == "BASELINE") && v[k] ~ /^[0-9]+$/) print k "\t" v[k] }' "$1" |
        sort -r | head -n "$WINDOW" |
        awk -F '\t' -v MIN="$MIN_NIGHTS" -v H="$HEADROOM_PCT" '{ k++; if ($2 + 0 > m) m = $2 + 0 }
            END { if (k < MIN) { printf "none %d\n", k; exit } printf "%d %d\n", int((m * (100 + H) + 99) / 100), k }'
}

# judge RECEIPTS HISTORY NOW CHECKER -> the line on stdout; the history row in the file $ROWF; returns 0 / 2
judge() {
    local night r rt out rc cheap total b k state
    night="$(night_of "$3")" || caller_error "--now '$3' is not a time"
    : > "$ROWF"
    [ -f "$4" ] || { printf 'DOGFOOD-TIMING NOT_MEASURED night=%s: no row-order checker at %s\n' "$night" "$4"
        row "$night	-	-	-	-	NOT_MEASURED"; return 2; }
    r="$(find "$1" -maxdepth 1 -regextype posix-extended -regex '.*/receipt-[0-9]{8}T[0-9]{6}Z\.json' 2>/dev/null | LC_ALL=C sort | tail -n 1)"
    [ -n "$r" ] || { printf 'DOGFOOD-TIMING NOT_MEASURED night=%s: no dogfood receipt in the receipts dir\n' "$night"
        row "$night	-	-	-	-	NOT_MEASURED"; return 2; }
    rt="$(receipt_time "$r")"
    if [ "$(night_of "$rt" 2>/dev/null)" != "$night" ]; then
        printf 'DOGFOOD-TIMING NOT_MEASURED night=%s: the newest receipt %s is not from this night\n' "$night" "${r##*/}"
        row "$night	${r##*/}	-	-	-	NOT_MEASURED"; return 2
    fi
    rc=0; out="$(DOGFOOD_ROW_ORDER_ENFORCE=0 bash "$4" --timing "${r%.json}.timing.tsv" 2>&1)" || rc=$?
    cheap="$(printf '%s\n' "$out" | sed -n 's/^row time: [0-9]* rows; cheap tier ([0-9]* rows) closed at +\([0-9]*\)s of +\([0-9]*\)s.*$/\1 \2/p' | head -n 1)"
    if [ "$rc" -ne 0 ] || [ -z "$cheap" ]; then
        printf 'DOGFOOD-TIMING NOT_MEASURED night=%s: the checker could not read the row times (rc=%s): %s\n' "$night" "$rc" "$(printf '%s' "$out" | head -n 1 | cut -c1-120)"
        row "$night	${r##*/}	-	-	-	NOT_MEASURED"; return 2
    fi
    total="${cheap#* }"; cheap="${cheap% *}"
    read -r b k <<< "$(budget "$2" "$night")"
    if [ "$b" = none ]; then
        printf 'DOGFOOD-TIMING BASELINE night=%s cheap=+%ss of +%ss budget=none (%s of %s earlier nights measured)\n' "$night" "$cheap" "$total" "$k" "$MIN_NIGHTS"
        row "$night	${r##*/}	$cheap	$total	-	BASELINE"; return 0
    fi
    rc=0; out="$(DOGFOOD_ROW_ORDER_ENFORCE=0 bash "$4" --timing "${r%.json}.timing.tsv" --cheap-budget-s "$b" 2>&1)" || rc=$?
    if [ "$rc" -ne 0 ]; then
        printf 'DOGFOOD-TIMING NOT_MEASURED night=%s: the checker failed on the budget run (rc=%s)\n' "$night" "$rc"
        row "$night	${r##*/}	-	-	-	NOT_MEASURED"; return 2
    fi
    case "$out" in
        *"REPORT:"*) state=OVER ;;
        *PASS*) state=GREEN
           [ "$cheap" -le "$b" ] || { printf 'DOGFOOD-TIMING NOT_MEASURED night=%s: the checker said PASS at +%ss over the budget %ss\n' "$night" "$cheap" "$b"
               row "$night	${r##*/}	-	-	-	NOT_MEASURED"; return 2; } ;;
        *) printf 'DOGFOOD-TIMING NOT_MEASURED night=%s: the checker printed neither PASS nor REPORT\n' "$night"
           row "$night	${r##*/}	-	-	-	NOT_MEASURED"; return 2 ;;
    esac
    printf 'DOGFOOD-TIMING %s night=%s cheap=+%ss of +%ss budget=%ss (max of %s measured nights +%s%%)\n' "$state" "$night" "$cheap" "$total" "$b" "$k" "$HEADROOM_PCT"
    row "$night	${r##*/}	$cheap	$total	$b	$state"; return 0
}

# ------------------------------------------------------------------------------------ case table ----------
CASES=0 FAILED=0
# row NAME WANT_RC MUST MUSTNOT -- CMD...  (stdout must be one line, contain MUST and not MUSTNOT)
row_case() {
    local name="$1" want="$2" must="$3" mustnot="$4" out rc
    shift 5
    CASES=$((CASES + 1))
    rc=0; out="$("$@" 2>/dev/null)" || rc=$?
    if [ "$rc" -eq "$want" ] && [ "$(printf '%s\n' "$out" | wc -l)" -eq 1 ] &&
        case "$out" in *"$must"*) true ;; *) false ;; esac &&
        { [ -z "$mustnot" ] || case "$out" in *"$mustnot"*) false ;; *) true ;; esac; }; then
        printf 'ok    %s\n' "$name"
    else
        FAILED=$((FAILED + 1)); printf 'RED   %s (rc=%s want %s): %s\n' "$name" "$rc" "$want" "${out:0:160}"
    fi
}

# a stub with the row-order checker's --timing interface and output lines
stub_checker() {
    cat > "$1" <<'STUB'
T="" B=""
while [ "$#" -gt 0 ]; do case "$1" in --timing) T="$2"; shift 2 ;; --cheap-budget-s) B="$2"; shift 2 ;; *) shift ;; esac; done
[ -s "$T" ] || { echo "not_measured  no row-time sidecar at $T"; exit 2; }
awk -F '\t' -v B="$B" '
    NR == 1 { if ($4 != "dur_s" || $5 != "closed_at_s") { print "not_measured  not a sidecar"; bad = 2; exit 2 } next }
    $4 !~ /^[0-9]+$/ || $5 !~ /^[0-9]+$/ { print "not_measured  row " $2 " has no numeric time"; bad = 2; exit 2 }
    { n++; if ($2 != "test coverage" && $2 != "clean-room") { c++; end = $5 } total = $5 }
    END { if (bad) exit bad
        if (!n) { print "not_measured  the sidecar has no rows"; exit 2 }
        printf "row time: %d rows; cheap tier (%d rows) closed at +%ds of +%ds\n", n, c, end, total
        if (B != "" && end > B + 0) { printf "OVER  the cheap tier closed at +%ds, budget %ss\n", end, B; exit 1 } }' "$T"
rc=$?
case "$rc" in
    0) echo PASS; exit 0 ;;
    2) echo "not_measured (rc=2)" >&2; exit 2 ;;
    *) if [ "${DOGFOOD_ROW_ORDER_ENFORCE:-0}" = 1 ]; then echo "FAIL: over budget" >&2; exit 1; fi
       echo "REPORT: the cheap tier is over its budget -- report mode"; exit 0 ;;
esac
STUB
}

# receipt DIR STAMP CHEAP_S TOTAL_S -> a receipt and its sidecar (CHEAP_S "-" writes no sidecar, "x" a row with no time)
receipt() {
    printf '{}\n' > "$1/receipt-$2.json"
    case "$3" in
        -) ;;
        x) printf 'idx\tgate\tresult\tdur_s\tclosed_at_s\n0\tgit-clean\tPASS\t\t\n' > "$1/receipt-$2.timing.tsv" ;;
        *) printf 'idx\tgate\tresult\tdur_s\tclosed_at_s\n0\tgit-clean\tPASS\t5\t5\n1\tbashrs\tPASS\t%s\t%s\n2\ttest coverage\tPASS\t%s\t%s\n' \
               "$(($3 - 5))" "$3" "$(($4 - $3))" "$4" > "$1/receipt-$2.timing.tsv" ;;
    esac
}

# hist FILE NIGHT CHEAP STATE... -> history rows
hist() {
    local h="$1"; shift
    [ -f "$h" ] || printf 'night\treceipt\tcheap_s\ttotal_s\tbudget_s\tstate\n' > "$h"
    while [ "$#" -ge 3 ]; do printf '%s\treceipt-x.json\t%s\t900\t-\t%s\n' "$1" "$2" "$3" >> "$h"; shift 3; done
}

self_test() {
    local tmp NOW="2026-10-12T06:00:00Z" C R H S="$SCRIPT_PATH" i
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/dt-st.XXXXXX")" || caller_error "no temp dir"
    C="$tmp/checker.sh"; stub_checker "$C"
    # tonight's receipt: the cheap tier closes at +63s of +900s
    R="$tmp/r1"; mkdir -p "$R"; receipt "$R" 20261011T230000Z 63 900
    row_case a_first_night_is_a_baseline 0 "BASELINE night=2026-10-11 cheap=+63s of +900s budget=none (0 of 3" "" -- \
        bash "$S" --receipts "$R" --history "$tmp/h1" --now "$NOW" --checker "$C"
    row_case the_baseline_night_is_recorded 0 "2026-10-11	receipt-20261011T230000Z.json	63	900	-	BASELINE" "" -- \
        bash -c 'tail -n 1 "$1"' _ "$tmp/h1"
    H="$tmp/h2"; hist "$H" 2026-10-09 60 GREEN 2026-10-10 70 GREEN
    row_case two_measured_nights_are_still_a_baseline 0 "BASELINE night=2026-10-11 cheap=+63s of +900s budget=none (2 of 3" "" -- \
        bash "$S" --receipts "$R" --history "$H" --now "$NOW" --checker "$C"
    H="$tmp/h3"; hist "$H" 2026-10-08 60 GREEN 2026-10-09 70 BASELINE 2026-10-10 80 GREEN
    R="$tmp/r3"; mkdir -p "$R"; receipt "$R" 20261011T230000Z 95 900
    row_case within_the_budget_is_green 0 "GREEN night=2026-10-11 cheap=+95s of +900s budget=100s (max of 3 measured nights +25%)" "" -- \
        bash "$S" --receipts "$R" --history "$H" --now "$NOW" --checker "$C"
    row_case the_green_night_is_recorded 0 "2026-10-11	receipt-20261011T230000Z.json	95	900	100	GREEN" "" -- \
        bash -c 'tail -n 1 "$1"' _ "$H"
    H="$tmp/h4"; hist "$H" 2026-10-08 60 GREEN 2026-10-09 70 GREEN 2026-10-10 80 GREEN
    R="$tmp/r4"; mkdir -p "$R"; receipt "$R" 20261011T230000Z 120 900
    row_case over_the_budget_is_reported_and_exits_0 0 "OVER night=2026-10-11 cheap=+120s of +900s budget=100s" "" -- \
        bash "$S" --receipts "$R" --history "$tmp/h4" --now "$NOW" --checker "$C"
    H="$tmp/h4e"; hist "$H" 2026-10-08 60 GREEN 2026-10-09 70 GREEN 2026-10-10 80 GREEN
    row_case an_enforcing_caller_still_gets_report_mode 0 "OVER night=2026-10-11" "" -- \
        env DOGFOOD_ROW_ORDER_ENFORCE=1 bash "$S" --receipts "$R" --history "$H" --now "$NOW" --checker "$C"
    H="$tmp/h5"; hist "$H" 2026-10-07 60 GREEN 2026-10-08 70 GREEN 2026-10-09 80 GREEN 2026-10-10 500 OVER
    row_case an_over_night_never_raises_the_bar 0 "OVER night=2026-10-11 cheap=+120s of +900s budget=100s" "" -- \
        bash "$S" --receipts "$R" --history "$H" --now "$NOW" --checker "$C"
    H="$tmp/h6"; hist "$H" 2026-10-07 60 GREEN 2026-10-08 70 GREEN 2026-10-09 80 GREEN 2026-10-10 500 GREEN 2026-10-10 500 OVER
    row_case the_latest_row_on_a_night_decides_it 0 "OVER night=2026-10-11 cheap=+120s of +900s budget=100s" "" -- \
        bash "$S" --receipts "$R" --history "$H" --now "$NOW" --checker "$C"
    H="$tmp/h7"; hist "$H" 2026-10-08 60 GREEN 2026-10-09 70 GREEN 2026-10-10 80 GREEN 2026-10-11 500 GREEN
    row_case tonight_never_sets_its_own_bar 0 "OVER night=2026-10-11 cheap=+120s of +900s budget=100s" "" -- \
        bash "$S" --receipts "$R" --history "$H" --now "$NOW" --checker "$C"
    H="$tmp/h8"; hist "$H" 2026-10-03 500 GREEN
    for i in 4 5 6 7 8 9 10; do hist "$H" "$(printf '2026-10-%02d' "$i")" "$((50 + i * 3))" GREEN; done
    row_case only_the_latest_7_nights_set_the_budget 0 "OVER night=2026-10-11 cheap=+120s of +900s budget=100s (max of 7 measured" "" -- \
        bash "$S" --receipts "$R" --history "$H" --now "$NOW" --checker "$C"
    H="$tmp/h9"; hist "$H" 2026-10-08 60 GREEN 2026-10-09 70 GREEN 2026-10-10 80 NOT_MEASURED
    row_case a_not_measured_night_is_never_a_duration 0 "BASELINE night=2026-10-11 cheap=+120s of +900s budget=none (2 of 3" "" -- \
        bash "$S" --receipts "$R" --history "$H" --now "$NOW" --checker "$C"
    # not measured is never a pass
    R="$tmp/r10"; mkdir -p "$R"; receipt "$R" 20261010T230000Z 63 900
    row_case a_receipt_from_another_night_is_not_measured 2 "NOT_MEASURED night=2026-10-11: the newest receipt receipt-20261010T230000Z.json is not from this night" "" -- \
        bash "$S" --receipts "$R" --history "$tmp/h10" --now "$NOW" --checker "$C"
    row_case the_not_measured_night_is_recorded 0 "2026-10-11	receipt-20261010T230000Z.json	-	-	-	NOT_MEASURED" "" -- \
        bash -c 'tail -n 1 "$1"' _ "$tmp/h10"
    row_case the_newest_receipt_is_the_one_judged 2 "is not from this night" "" -- \
        bash -c 'receipt() { :; }; cp -- "$1"/receipt-* "$2"/ && bash "$3" --receipts "$2" --history "$4" --now "$5" --checker "$6"' _ \
        "$tmp/r1" "$R" "$S" "$tmp/h10b" "2026-10-11T06:00:00Z" "$C"
    R="$tmp/r11"; mkdir -p "$R"
    row_case no_receipt_is_not_measured 2 "NOT_MEASURED night=2026-10-11: no dogfood receipt" "" -- \
        bash "$S" --receipts "$R" --history "$tmp/h11" --now "$NOW" --checker "$C"
    row_case no_receipts_dir_is_not_measured 2 "NOT_MEASURED night=2026-10-11: no dogfood receipt" "" -- \
        bash "$S" --receipts "$tmp/absent" --history "$tmp/h11" --now "$NOW" --checker "$C"
    R="$tmp/r12"; mkdir -p "$R"; receipt "$R" 20261011T230000Z - -
    row_case no_sidecar_is_not_measured 2 "NOT_MEASURED night=2026-10-11: the checker could not read the row times (rc=2)" "BASELINE" -- \
        bash "$S" --receipts "$R" --history "$tmp/h12" --now "$NOW" --checker "$C"
    R="$tmp/r13"; mkdir -p "$R"; receipt "$R" 20261011T230000Z x -
    row_case a_row_with_no_time_is_not_measured 2 "NOT_MEASURED night=2026-10-11: the checker could not read the row times" "BASELINE" -- \
        bash "$S" --receipts "$R" --history "$tmp/h13" --now "$NOW" --checker "$C"
    R="$tmp/r14"; mkdir -p "$R"; printf '{}\n' > "$R/receipt-latest.json"; receipt "$R" 20261011T230000Z 63 900
    row_case an_unstamped_receipt_is_never_judged 0 "BASELINE night=2026-10-11 cheap=+63s" "" -- \
        bash "$S" --receipts "$R" --history "$tmp/h14" --now "$NOW" --checker "$C"
    row_case no_checker_is_not_measured 2 "NOT_MEASURED night=2026-10-11: no row-order checker" "" -- \
        bash "$S" --receipts "$tmp/r1" --history "$tmp/h15" --now "$NOW" --checker "$tmp/absent.sh"
    printf 'echo "row time: 3 rows; cheap tier (2 rows) closed at +5s of +9s"; [ "$#" -gt 2 ] && echo "something else"; exit 0\n' > "$tmp/odd.sh"
    H="$tmp/h16"; hist "$H" 2026-10-08 60 GREEN 2026-10-09 70 GREEN 2026-10-10 80 GREEN
    row_case a_checker_verdict_that_is_neither_pass_nor_report_is_not_measured 2 "printed neither PASS nor REPORT" "GREEN" -- \
        bash "$S" --receipts "$tmp/r1" --history "$H" --now "$NOW" --checker "$tmp/odd.sh"
    # caller
    row_case a_night_with_no_receipt_is_recorded 0 "2026-10-11	-	-	-	-	NOT_MEASURED" "" -- \
        bash -c 'tail -n 1 "$1"' _ "$tmp/h11"
    row_case a_night_with_no_checker_is_recorded 0 "2026-10-11	-	-	-	-	NOT_MEASURED" "" -- \
        bash -c 'tail -n 1 "$1"' _ "$tmp/h15"
    H="$tmp/h19"; hist "$H" 2026-10-08 60 GREEN 2026-10-09 83 GREEN 2026-10-10 70 GREEN
    row_case the_budget_rounds_up 0 "GREEN night=2026-10-11 cheap=+95s of +900s budget=104s" "" -- \
        bash "$S" --receipts "$tmp/r3" --history "$H" --now "$NOW" --checker "$C"
    printf 'echo "row time: 3 rows; cheap tier (2 rows) closed at +120s of +900s"; echo PASS; exit 0\n' > "$tmp/lax.sh"
    H="$tmp/h20"; hist "$H" 2026-10-08 60 GREEN 2026-10-09 70 GREEN 2026-10-10 80 GREEN
    row_case a_pass_over_the_budget_is_not_measured 2 "the checker said PASS at +120s over the budget 100s" "GREEN" -- \
        bash "$S" --receipts "$tmp/r4" --history "$H" --now "$NOW" --checker "$tmp/lax.sh"
    mkdir -p "$tmp/hdir"
    row_case an_unwritable_history_is_a_caller_error 3 "DOGFOOD-TIMING" "" -- \
        bash "$S" --receipts "$tmp/r1" --history "$tmp/hdir" --now "$NOW" --checker "$C"
    : > "$tmp/inbox"
    row_case the_line_reaches_the_inbox 0 "DOGFOOD-TIMING BASELINE night=2026-10-11" "" -- \
        bash -c 'bash "$1" --receipts "$2" --history "$3" --now "$4" --checker "$5" --inbox "$6" > /dev/null; tail -n 1 "$6"' _ \
        "$S" "$tmp/r1" "$tmp/h17" "$NOW" "$C" "$tmp/inbox"
    row_case a_bad_now_is_a_caller_error 3 "caller error" "" -- \
        bash "$S" --receipts "$tmp/r1" --history "$tmp/h18" --now "not-a-time" --checker "$C"
    row_case a_missing_receipts_dir_option_is_a_caller_error 3 "caller error: --receipts is required" "" -- \
        bash "$S" --history "$tmp/h18" --now "$NOW" --checker "$C"
    row_case an_unknown_option_is_a_caller_error 3 "caller error: unknown option --enforce" "" -- \
        bash "$S" --receipts "$tmp/r1" --history "$tmp/h18" --now "$NOW" --checker "$C" --enforce
    rm -rf -- "${tmp:?}"
    if [ "$FAILED" -eq 0 ]; then printf 'SELF-TEST-GREEN %s/%s case rows green\n' "$CASES" "$CASES"; return 0; fi
    printf 'SELF-TEST-RED %s of %s case rows red\n' "$FAILED" "$CASES"; return 1
}

# ------------------------------------------------------------------------------------ planted mutants ----
MUTANTS='m01_stale_receipt_judged_as_tonight	s/^    if \[ "\$(night_of "\$rt" 2>\/dev\/null)" != "\$night" \]; then$/    if false; then/
m02_checker_not_measured_read_as_measured	s/^    if \[ "\$rc" -ne 0 \] || \[ -z "\$cheap" \]; then$/    if false; then/
m03_over_night_raises_the_bar	s/(s\[k\] == "GREEN" || s\[k\] == "BASELINE")/(s[k] != "NOT_MEASURED")/
m04_one_night_is_a_budget	s/^MIN_NIGHTS=3$/MIN_NIGHTS=2/
m05_no_headroom	s/^HEADROOM_PCT=25$/HEADROOM_PCT=0/
m06_tonight_sets_its_own_bar	s/\$1 != "night" \&\& \$1 < T/$1 != "night" \&\& $1 <= T/
m07_window_widened	s/^WINDOW=7$/WINDOW=8/
m08_oldest_nights_set_the_budget	s/        sort -r | head -n "\$WINDOW" |/        sort | head -n "$WINDOW" |/
m09_first_row_on_a_night_decides	s/\$1 < T { s\[\$1\] = \$6; v\[\$1\] = \$3 }/$1 < T \&\& !($1 in s) { s[$1] = $6; v[$1] = $3 }/
m10_not_measured_is_a_duration	s/(s\[k\] == "GREEN" || s\[k\] == "BASELINE")/(s[k] != "OVER")/
m11_enforce_inherited_from_the_caller	s/rc=0; out="\$(DOGFOOD_ROW_ORDER_ENFORCE=0 bash "\$4" --timing "\${r%.json}.timing.tsv" --cheap-budget-s/rc=0; out="$(bash "$4" --timing "${r%.json}.timing.tsv" --cheap-budget-s/
m12_history_not_written	s/^        cat -- "\$ROWF" >> "\$history" 2>\/dev\/null ||$/        : ||/
m13_neither_verdict_read_as_green	s/^        \*PASS\*) state=GREEN$/        *) state=GREEN/
m14_newest_receipt_not_taken	s/LC_ALL=C sort | tail -n 1)/LC_ALL=C sort | head -n 1)/
m15_budget_not_rounded_up	s/ + 99) \/ 100)/) \/ 100)/
m16_checker_pass_trusted_over_budget	s/^           \[ "\$cheap" -le "\$b" \] || {/           true || {/
m17_history_write_failure_ignored	/DT the history/s/exit 3/exit 0/
m18_night_without_a_receipt_not_recorded	s/^        row "\$night\t-\t/        : "$night\t-\t/
m19_unstamped_receipt_judged	s/receipt-\[0-9\]{8}T\[0-9\]{6}Z/receipt-.*/'

mutants() {
    local tmp name expr killed=0 total=0 errors=0
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/dt-mu.XXXXXX")" || caller_error "no temp dir"
    while IFS="$(printf '\t')" read -r name expr; do
        [ -n "$name" ] || continue
        total=$((total + 1))
        sed -e "$expr" "$SCRIPT_PATH" > "$tmp/m.sh"
        if cmp -s "$SCRIPT_PATH" "$tmp/m.sh"; then errors=$((errors + 1)); printf 'ERROR    %s (the patch did not apply)\n' "$name"; continue; fi
        if bash "$tmp/m.sh" --self-test > "$tmp/out" 2>&1; then
            printf 'SURVIVED %s\n' "$name"
        else
            killed=$((killed + 1)); printf 'killed   %-44s %s\n' "$name" "$(grep -c '^RED ' "$tmp/out")"
        fi
    done <<< "$MUTANTS"
    rm -rf -- "${tmp:?}"
    printf 'MUTANTS killed=%s total=%s errors=%s\n' "$killed" "$total" "$errors"
    [ "$killed" -eq "$total" ] && [ "$errors" -eq 0 ]
}

main() {
    local receipts="" history="" now="" checker="$HERE/../check_dogfood_row_order.sh" inbox="" line rc
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --receipts) receipts="${2:-}"; shift 2 || caller_error "--receipts needs a dir" ;;
            --history) history="${2:-}"; shift 2 || caller_error "--history needs a file" ;;
            --now) now="${2:-}"; shift 2 || caller_error "--now needs a time" ;;
            --checker) checker="${2:-}"; shift 2 || caller_error "--checker needs a path" ;;
            --inbox) inbox="${2:-}"; shift 2 || caller_error "--inbox needs a file" ;;
            *) caller_error "unknown option $1" ;;
        esac
    done
    [ -n "$receipts" ] || caller_error "--receipts is required"
    [ -n "$history" ] || caller_error "--history is required"
    [ -n "$now" ] || now="$(date -u +%FT%TZ)"
    ROWF="$(mktemp "${TMPDIR:-/tmp}/dt-row.XXXXXX")" || caller_error "no temp file"
    trap 'rm -f -- "${ROWF:?}"' EXIT
    rc=0; line="$(judge "$receipts" "$history" "$now" "$checker")" || rc=$?
    [ "$rc" -eq 3 ] && { printf '%s\n' "$line"; exit 3; }
    printf '%s\n' "$line"
    if [ -s "$ROWF" ]; then
        { [ -f "$history" ] || printf 'night\treceipt\tcheap_s\ttotal_s\tbudget_s\tstate\n' > "$history"; } 2>/dev/null &&
        cat -- "$ROWF" >> "$history" 2>/dev/null ||
        { printf 'DT the history %s could not be written: this night is not recorded\n' "$history" >&2; exit 3; }
    fi
    if [ -n "$inbox" ]; then
        printf '%s\n' "${line:0:299}" >> "$inbox"
        grep -qxF -- "${line:0:299}" "$inbox" || { printf 'DT the inbox line did not read back\n' >&2; exit 3; }
    fi
    exit "$rc"
}

case "${1:-}" in
    --self-test) self_test; exit $? ;;
    --mutants) mutants; exit $? ;;
esac
main "$@"
