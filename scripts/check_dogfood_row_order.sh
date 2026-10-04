#!/usr/bin/env bash
# check_dogfood_row_order.sh -- scripts/dogfood.sh runs its cheap rows first, its long
# rows last, and a red cheap row still lets every long row run (#4672).
#
# WHY. In 0.70.1 the release dogfood ran coverage for about 3.5 h before it reached the
# bashrs row, which takes about a minute. That row was RED, and nobody knew until the
# long wait was over. The fix is an ORDER, not a new gate: no row is removed, no
# threshold moves, and the verdict stays the AND of every row. This guard holds that:
#   O1  every row after the first long row is a long row (or the clean-room reminder). A
#       row is any gate/mark call, or a call to a function whose body records one
#       (mark_version_row, ...), whatever its first word -- `mark "$n"` counts;
#   O2  every long row named in DOGFOOD_LONG_ROWS exists in the runner;
#   O3  nothing between the first row and the receipt can stop the run early, or skip a
#       row because an earlier one was red: no `set -e` and no ERR trap; no exit or
#       `kill $$` (in the main flow or any helper), no top-level return, eval or exec,
#       and no break/continue after the first long row, outside the DOGFOOD_GATES_ONLY
#       partial-run block; and the main flow never reads FAILED, RESULTS or NAMES
#       before the verdict. A lint against an accidental reorder, not a proof
#       against an adversarial author;
#   O4  BEHAVIOUR: the runner's own gate()/mark() and verdict expression are lifted and
#       driven -- a red cheap row, then every long row; each long row must still run and
#       the verdict must be NO-GO. An all-green control must be GO (discrimination).
# It reads the long set from the runner's DOGFOOD_LONG_ROWS line (one source); a runner
# without that line is ENV rc=2, not a pass. DOGFOOD_ROW_ORDER_LONG overrides it, to judge
# an older runner that predates the line (origin/main before #4672).
#
# ROW TIME. `--timing <receipt>.timing.tsv` reads the sidecar dogfood.sh writes beside its
# receipt and reports when the cheap tier closed; `--cheap-budget-s N` reports a cheap
# tier over N seconds. A missing sidecar, or a row with no numeric time, is not_measured
# (rc 2), never a pass.
#
# MODE. REPORT by default (L31: a new check blocks only after three green nights): a
# finding prints a REPORT line and exits 0. DOGFOOD_ROW_ORDER_ENFORCE=1 makes a finding
# exit 1. ENV (the guard cannot judge) is rc 2 in both modes.
#
#   check_dogfood_row_order.sh [--file F]                       O1-O4 over the runner
#   check_dogfood_row_order.sh --timing T [--cheap-budget-s N]  the row-time report
#   check_dogfood_row_order.sh --self-test                      planted defects must go RED
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
SRC="$ROOT/scripts/dogfood.sh" TIMING="" BUDGET="" SELF_TEST=0
while [ $# -gt 0 ]; do
    case "$1" in
        --file) SRC="${2:?--file needs a path}"; shift 2 ;;
        --timing) TIMING="${2:?--timing needs a path}"; shift 2 ;;
        --cheap-budget-s) BUDGET="${2:?--cheap-budget-s needs seconds}"; shift 2 ;;
        --self-test) SELF_TEST=1; shift ;;
        -h | --help) sed -n '2,38p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) printf 'check_dogfood_row_order.sh: unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

# long_rows <src> -> the long set, one line
long_rows() {
    if [ -n "${DOGFOOD_ROW_ORDER_LONG:-}" ]; then printf '%s\n' "$DOGFOOD_ROW_ORDER_LONG"; return; fi
    sed -n 's/^DOGFOOD_LONG_ROWS="\([^"]*\)"$/\1/p' "$1" | head -1
}

# wrappers <src> -> functions defined at column 0 whose body records a row (mark_version_row,
# classify_declared, ...): a call to one of them IS a row call, whatever it is named
wrappers() {
    awk '/^[A-Za-z_][A-Za-z0-9_]*\(\) *\{/ { fn = $0; sub(/\(.*/, "", fn); one = /\}[[:space:]]*$/ }
         fn != "" && fn != "gate" && fn != "mark" && !/^[[:space:]]*#/ &&
             /(^|[;&|{()]|then |else |do )[[:space:]]*(gate|mark)[[:space:]]+[^[:space:]]/ { w[fn] = 1 }
         fn != "" && (one || /^\}$/) { fn = ""; one = 0 }
         END { for (k in w) print k }' "$1" | sort | paste -sd'|'
}

# row_calls <src> -> "<line> <name>" for every row call: gate/mark or a wrapper of them, at
# command position (also after `(` and `$(`), with any first word -- `mark "$n"` is a row too
row_calls() {
    local w; w=$(wrappers "$1")
    awk -v W="gate|mark${w:+|$w}" '/^[A-Za-z_][A-Za-z0-9_]*\(\) *\{/ { fn = 1; one = /\}[[:space:]]*$/ }
         fn { if (one || /^\}$/) { fn = 0; one = 0 }; next }
         /^[[:space:]]*#/ { next }
         { s = $0
           while (match(s, "(^|[;&|{()]|then |else |do )[[:space:]]*(" W ")[[:space:]]+[^[:space:];&|)]+")) {
               t = substr(s, RSTART, RLENGTH); s = substr(s, RSTART + RLENGTH)
               sub("^.*(" W ")[[:space:]]+", "", t); print NR, t } }' "$1"
}

# lift <src> <fn> -> the function body: a one-liner, or `fn() {` to the first `}` at column 0
lift() {
    awk -v F="^$2\\\\(\\\\) *\\\\{" '$0 ~ F { f = 1; one = /\}[[:space:]]*$/ } f { print } f && (one || /^\}$/) { exit }' "$1"
}

# static <src> <long set> -> prints findings, rc 0 none / 1 findings / 2 ENV
static() {
    local src="$1" long="$2" calls first n bad=0 l name
    local -a LONG
    calls=$(row_calls "$src")
    n=$(printf '%s\n' "$calls" | grep -c .)
    [ "$n" -ge 20 ] || { printf 'ENV   only %s row calls found in %s -- not a runner this guard can judge\n' "$n" "$src"; return 2; }
    read -ra LONG <<< "$long"
    for name in "${LONG[@]}"; do
        printf '%s\n' "$calls" | awk -v n="$name" '$2 == n { f = 1 } END { exit !f }' ||
            { printf 'FAIL  O2 long row "%s" is not a row of %s -- the order rule would hold vacuously\n' "$name" "$src"; bad=1; }
    done
    first=$(printf '%s\n' "$calls" | awk -v L=" $long " 'index(L, " " $2 " ") { print $1; exit }')
    [ -n "$first" ] || { printf 'ENV   no long row (%s) found in %s\n' "$long" "$src"; return 2; }
    while read -r l name; do
        [ "$l" -gt "$first" ] || continue
        case " $long clean-room " in *" $name "*) continue ;; esac
        printf 'FAIL  O1 row %s (line %s) runs after the first long row (line %s): it waits on the long rows\n' "$name" "$l" "$first"
        bad=1
    done <<< "$calls"
    if grep -nE '^[[:space:]]*set[[:space:]][^#]*(-[a-zA-Z]*e[a-zA-Z]*([[:space:]]|$)|errexit)' "$src" > /dev/null; then
        printf 'FAIL  O3 the runner sets errexit: the first red command would stop every later row\n'; bad=1
    fi
    for name in gate mark; do
        [ "$(grep -cE "^[[:space:]]*(function[[:space:]]+)?${name}[[:space:]]*\(\)" "$src")" = 1 ] ||
            { printf 'FAIL  O4 %s() is not defined exactly once: the lifted copy may not be the one the rows call\n' "$name"; bad=1; }
    done
    awk -v fl="$first" -v start="$(printf '%s\n' "$calls" | head -1 | cut -d' ' -f1)" '
        /^# ── receipt \+ verdict/ { exit }
        /^[[:space:]]*#/ { next }
        /^if \[ -n "\$\{DOGFOOD_GATES_ONLY/ { partial = 1 }
        partial && /^fi$/ { partial = 0; next }
        partial { next }
        /^[A-Za-z_][A-Za-z0-9_]*\(\) *\{/ { fn = 1; one = /\}[[:space:]]*$/ }
        NR < start && !fn { next }      # set-up before the first row may refuse to start
        { q = $0; gsub(/"[^"]*"|\047[^\047]*\047/, "\"\"", q) }   # quoted text is inert
        { P = "(^|[;{&|()]|then |else |do )[[:space:]]*" }
        q ~ (P "exit([[:space:]]|;|\\)|$)") || $0 ~ (P "kill[^#]*\\$(\\$|\\{?PPID)") { print "stop " NR; bad = 1 }
        $0 ~ /(^|[;{&|]|then )[[:space:]]*trap[[:space:]]/ && $0 ~ /(exit|kill|ERR|DEBUG|RETURN)/ { print "stop " NR; bad = 1 }
        !fn && q ~ (P "(return|eval)([[:space:]]|;|$)") { print "stop " NR; bad = 1 }
        !fn && q ~ (P "(exec[[:space:]]+[^0-9<>&[:space:]]|alias[[:space:]]|source[[:space:]]|\\.[[:space:]]+[^[:space:]])") { print "stop " NR; bad = 1 }
        !fn && NR >= fl && q ~ (P "(break|continue)([[:space:]]|;|$)") { print "stop " NR; bad = 1 }   # a loop around the long rows
        !fn && (q ~ /(^|[^A-Za-z0-9_$])(FAILED|RESULTS|NAMES)([^A-Za-z0-9_]|$)/ || q ~ /\$\{?(FAILED|RESULTS|NAMES)([^A-Za-z0-9_]|$)/ || $0 ~ /\$\{?(FAILED|RESULTS|NAMES)([^A-Za-z0-9_]|$)/) && !/^[[:space:]]*(NAMES|RESULTS|NOTES)\+?=\(\)/ { print "cond " NR; bad = 1 }
        fn && (one || /^\}$/) { fn = 0; one = 0 }
        END { exit bad }' "$src" > "${TMPDIR:-/tmp}/dro-exit.$$" || {
        grep -q '^stop ' "${TMPDIR:-/tmp}/dro-exit.$$" &&
            printf 'FAIL  O3 an exit/kill/return between the first row and the receipt (line %s) can end the run before the long rows\n' \
                "$(sed -n 's/^stop //p' "${TMPDIR:-/tmp}/dro-exit.$$" | paste -sd, )"
        grep -q '^cond ' "${TMPDIR:-/tmp}/dro-exit.$$" &&
            printf 'FAIL  O3 the main flow reads FAILED before the verdict (line %s): a later row can be made conditional on an earlier red\n' \
                "$(sed -n 's/^cond //p' "${TMPDIR:-/tmp}/dro-exit.$$" | paste -sd, )"
        bad=1
    }
    rm -f -- "${TMPDIR:-/tmp}/dro-exit.$$"
    [ "$bad" = 0 ] && printf 'ok    O1-O3 %s rows; the long rows (%s) run last from line %s; no early stop\n' "$n" "$long" "$first"
    return "$bad"
}

# behaviour <src> <long set> -> O4: lift gate/mark/verdict and drive them
behaviour() {
    local src="$1" long="$2" d fn body verdict out name bad=0
    local -a LONG; read -ra LONG <<< "$long"
    d=$(mktemp -d "${TMPDIR:-/tmp}/dro.XXXXXX") || return 2
    {
        printf 'set -uo pipefail\nDOGFOOD_PHASE=full\nWORKLOG=%q\n' "$d"
        sed -n 's/^POST_PUBLISH_OBLIGATIONS=/&/p' "$src"
        printf 'declare -a NAMES=() RESULTS=() NOTES=() DURS=() ENDS=()\nFAILED=0\nROW_T0=$SECONDS\n'
        for fn in strip_ansi row_time gate mark; do
            body=$(lift "$src" "$fn")
            [ -n "$body" ] || { [ "$fn" = row_time ] && continue; printf 'MISSING %s\n' "$fn"; }
            printf '%s\n' "$body"
        done
    } > "$d/lib.sh"
    if grep -q '^MISSING' "$d/lib.sh"; then
        printf 'ENV   O4 cannot lift %s from %s\n' "$(sed -n 's/^MISSING //p' "$d/lib.sh" | paste -sd, )" "$src"
        rm -rf -- "${d:?}"; return 2
    fi
    verdict=$(grep -m1 -oE 'VERDICT="\$\(\[ \$FAILED -eq 0 \] && echo GO \|\| echo NO-GO\)"' "$src") || {
        printf 'FAIL  O4 the verdict is no longer `GO iff FAILED == 0` -- the AND over every row is gone\n'
        rm -rf -- "${d:?}"; return 1
    }
    # $1 = PASS|FAIL: two cheap rows of that result, one per row family (gate and mark),
    # then every long row passes
    drive() {
        { cat "$d/lib.sh"
          printf 'gate planted-cheap-gate %s\nmark planted-cheap-mark %s "planted cheap row"\n' "$( [ "$1" = PASS ] && echo true || echo false)" "$1"
          for name in "${LONG[@]}"; do printf 'gate %s true\n' "$name"; done
          printf '%s\n' "$verdict"
          printf 'for i in "${!NAMES[@]}"; do printf "ROW %%s %%s\\n" "${NAMES[$i]}" "${RESULTS[$i]}"; done\n'
          printf 'echo "VERDICT=$VERDICT"\n'
        } > "$d/drive.sh"
        bash "$d/drive.sh" 2> /dev/null
    }
    out=$(drive FAIL)
    for name in "${LONG[@]}"; do
        printf '%s\n' "$out" | grep -qx "ROW $name PASS" ||
            { printf 'FAIL  O4 after a red cheap row the long row "%s" did not run\n' "$name"; bad=1; }
    done
    printf '%s\n' "$out" | grep -qx 'VERDICT=NO-GO' ||
        { printf 'FAIL  O4 a red cheap row with green long rows gave %s, not NO-GO\n' "$(printf '%s\n' "$out" | grep -m1 '^VERDICT=' || echo 'no verdict')"; bad=1; }
    printf '%s\n' "$(drive PASS)" | grep -qx 'VERDICT=GO' ||
        { printf 'FAIL  O4 the all-green control was not GO -- the proof row cannot discriminate\n'; bad=1; }
    rm -rf -- "${d:?}"
    [ "$bad" = 0 ] && printf 'ok    O4 a red cheap row: every long row (%s) still ran, verdict NO-GO; all-green control GO\n' "$long"
    return "$bad"
}

# judge <src> -> rc 0 / 1 findings / 2 ENV
judge() {
    local long rc=0 r
    [ -f "$1" ] || { printf 'ENV   %s is missing\n' "$1"; return 2; }
    long=$(long_rows "$1")
    [ -n "$long" ] || { printf 'ENV   no DOGFOOD_LONG_ROWS="..." line in %s -- cannot judge, not a pass\n' "$1"; return 2; }
    static "$1" "$long"; r=$?; [ "$r" -gt "$rc" ] && rc="$r"
    behaviour "$1" "$long"; r=$?; [ "$r" -gt "$rc" ] && rc="$r"
    return "$rc"
}

# timing <tsv> <budget> -> rc 0 / 1 over budget / 2 not_measured
timing() {
    local tsv="$1" budget="$2" long end
    [ -s "$tsv" ] || { printf 'not_measured  no row-time sidecar at %s\n' "$tsv"; return 2; }
    case "$budget" in *[!0-9]*) printf 'not_measured  --cheap-budget-s %s is not a whole number of seconds\n' "$budget"; return 2 ;; esac
    long=$(long_rows "$SRC")
    awk -F'\t' -v L=" ${long:-test coverage} clean-room " -v B="$budget" '
        NR == 1 { if ($4 != "dur_s" || $5 != "closed_at_s") { print "not_measured  not a dogfood row-time sidecar (header: " $0 ")"; bad = 2; exit 2 } next }
        $4 !~ /^[0-9]+$/ || $5 !~ /^[0-9]+$/ { print "not_measured  row " $2 " has no numeric time (dur_s=" $4 ", closed_at_s=" $5 ")"; bad = 2; exit 2 }
        { n++; if (!index(L, " " $2 " ")) { c++; end = $5; if ($3 == "FAIL") red = red " " $2 } total = $5 }
        END {
            if (bad) exit bad
            if (!n) { print "not_measured  the sidecar has no rows"; exit 2 }
            printf "row time: %d rows; cheap tier (%d rows) closed at +%ds of +%ds%s\n", n, c, end, total, (red ? "; red:" red : "")
            if (B != "" && end > B + 0) { printf "OVER  the cheap tier closed at +%ds, budget %ss\n", end, B; exit 1 }
        }' "$tsv"
}

finish() { # finish <rc> <what>
    case "$1" in
        0) echo PASS; exit 0 ;;
        2) echo "not_measured (rc=2): $2" >&2; exit 2 ;;
        *) if [ "${DOGFOOD_ROW_ORDER_ENFORCE:-0}" = 1 ]; then echo "FAIL: $2" >&2; exit 1; fi
           echo "REPORT: $2 -- report mode (L31: blocks only after three green nights; DOGFOOD_ROW_ORDER_ENFORCE=1 enforces), #4672"
           exit 0 ;;
    esac
}

self_test() {
    local d bad=0 rc m
    d=$(mktemp -d "${TMPDIR:-/tmp}/dro-self.XXXXXX") || return 2
    trap 'rm -rf -- "${d:?}"' RETURN
    [ -f "$SRC" ] || { printf 'ENV   %s is missing\n' "$SRC"; return 2; }
    expect() { # expect <want rc> <label> <file>
        rc=0; judge "$3" > "$d/out" 2>&1 || rc=$?
        if [ "$rc" = "$1" ]; then printf 'ok    rc=%s %s%s\n' "$rc" "$2" "$( [ "$rc" != 0 ] && printf ' -- %s' "$(grep -m1 -E '^(FAIL|ENV)' "$d/out" | cut -c7-110)")"
        else printf 'FAIL  wanted rc=%s, got rc=%s: %s\n' "$1" "$rc" "$2"; sed 's/^/        /' "$d/out" | head -5; bad=1; fi
    }
    planted() { # planted <name> <label> <want rc> -- judges the mutant already written to $d/<name>.sh
        if cmp -s "$SRC" "$d/$1.sh"; then printf 'FAIL  the %s mutant did not apply (its anchor is gone)\n' "$1"; bad=1; return; fi
        expect "$3" "$2" "$d/$1.sh"
    }
    expect 0 "the runner as committed" "$SRC"
    # O1: a cheap row moved back behind the long rows (the 0.70.1 shape)
    sed 's/^# ── clean-room reminder/mark bashrs-late PASS "planted"\n&/' "$SRC" > "$d/late.sh"; planted late "a cheap row after the long rows is RED" 1
    # O2: a long row renamed away -- the rule must not hold vacuously
    sed 's/^gate test /gate tests /' "$SRC" > "$d/renamed.sh"; planted renamed "a long row missing from the runner is RED" 1
    # O3: an early stop on a red row, and errexit
    sed 's/^# ── 12\. the LONG rows/[ "$FAILED" -eq 0 ] || exit 1\n&/' "$SRC" > "$d/early.sh"; planted early "an exit before the long rows is RED" 1
    sed 's/^set -uo pipefail$/set -euo pipefail/' "$SRC" > "$d/errexit.sh"; planted errexit "set -e in the runner is RED" 1
    # O4: gate() stops the run on a red row; the verdict ignores FAILED
    awk '/^gate\(\) \{/ { g = 1 } g && /FAILED=1; fi$/ { sub(/FAILED=1; fi$/, "FAILED=1; exit 1; fi"); g = 0 } { print }' "$SRC" > "$d/gateexit.sh"
    planted gateexit "gate() exiting on a red row is RED" 1
    sed 's/\[ "\$st" = FAIL \] && FAILED=1$/[ "$st" = FAIL ] && { FAILED=1; exit 1; }/' "$SRC" > "$d/markexit.sh"
    planted markexit "mark() exiting on a red row is RED" 1
    sed 's/VERDICT="\$(\[ \$FAILED -eq 0 \] \&\& echo GO || echo NO-GO)"/VERDICT="$(echo GO)"/' "$SRC" > "$d/orverdict.sh"
    planted orverdict "a verdict that is not the AND of the rows is RED" 1
    # ENV: the long set is not declared -- cannot judge, never a pass
    sed '/^DOGFOOD_LONG_ROWS=/d' "$SRC" > "$d/nolong.sh"; planted nolong "no DOGFOOD_LONG_ROWS line is ENV" 2
    # O1 through the other row spellings: a row named by a variable, and a wrapper call
    sed 's/^# ── clean-room reminder/mark "$planted_row" PASS "planted"\n&/' "$SRC" > "$d/dynrow.sh"; planted dynrow "a variable-named row after the long rows is RED" 1
    sed 's/^# ── clean-room reminder/mark_version_row planted\n&/' "$SRC" > "$d/wrapcall.sh"; planted wrapcall "a wrapper row call after the long rows is RED" 1
    # O3: the long rows made conditional on the cheap verdict, and two other ways to stop
    sed 's/^gate test /[ "$FAILED" -eq 0 ] \&\& gate test /' "$SRC" > "$d/guarded.sh"; planted guarded "a long row run only if no earlier row was red is RED" 1
    sed 's/^# ── 12\. the LONG rows/[ "$FAILED" -eq 0 ] || kill $$\n&/' "$SRC" > "$d/killself.sh"; planted killself "a kill of the runner before the long rows is RED" 1
    # round-2 spellings: a row in a case arm, an ERR trap, a quoted kill, an exit in a helper
    # defined before the first row, and an arithmetic read of FAILED
    sed 's/^# ── clean-room reminder/case x in x) mark casearm-late PASS "planted" ;; esac\n&/' "$SRC" > "$d/casearm.sh"; planted casearm "a row in a case arm after the long rows is RED" 1
    sed 's/^# ── 12\. the LONG rows/trap "exit 1" ERR\n&/' "$SRC" > "$d/errtrap.sh"; planted errtrap "an ERR trap that ends the run is RED" 1
    sed 's/^# ── 12\. the LONG rows/[ "$FAILED" -eq 0 ] || kill -9 "$$"\n&/' "$SRC" > "$d/qkill.sh"; planted qkill "a quoted kill of the runner is RED" 1
    sed 's/^strip_ansi() {/bail() { exit 1; }\n&/' "$SRC" > "$d/bail.sh"; planted bail "an exit in a helper defined before the first row is RED" 1
    sed 's/^gate test /(( FAILED == 0 )) \&\& gate test /' "$SRC" > "$d/arith.sh"; planted arith "an arithmetic read of FAILED guarding a long row is RED" 1
    # round-3 spellings: exec of a command, a quoted array read, gate() redefined so the lifted
    # copy is not the live one, errexit spelled long, a DEBUG trap, and a kill of the parent
    sed 's/^# ── 12\. the LONG rows/[ "$FAILED" -eq 0 ] || exec true\n&/' "$SRC" > "$d/exectrue.sh"; planted exectrue "exec of a command before the long rows is RED" 1
    sed 's/^gate test /case " ${RESULTS[*]} " in *FAIL*) ;; *) gate test /' "$SRC" > "$d/arrread.sh"
    sed -i 's/^\(case " \${RESULTS\[\*\]} " in .*\)$/\1 ;; esac/' "$d/arrread.sh"; planted arrread "a quoted RESULTS read guarding a long row is RED" 1
    sed 's/^# ── 12\. the LONG rows/gate() { :; }\n&/' "$SRC" > "$d/redef.sh"; planted redef "gate() redefined before the long rows is RED" 1
    sed 's/^set -uo pipefail$/set -uo pipefail -o errtrace -o errexit/' "$SRC" > "$d/errlong.sh"; planted errlong "errexit spelled as a long option is RED" 1
    sed "s/^# ── 12\\. the LONG rows/trap 'exit 3' DEBUG\\n&/" "$SRC" > "$d/dbgtrap.sh"; planted dbgtrap "a DEBUG trap that exits is RED" 1
    sed 's/^# ── 12\. the LONG rows/[ "$FAILED" -eq 0 ] || kill -TERM $PPID\n&/' "$SRC" > "$d/ppid.sh"; planted ppid "a kill of the parent is RED" 1
    # Row time: four sidecars
    printf 'idx\tgate\tresult\tdur_s\tclosed_at_s\n0\tfmt\tPASS\t3\t3\n1\tbashrs\tFAIL\t60\t63\n2\ttest\tPASS\t900\t963\n3\tcoverage\tPASS\t12600\t13563\n' > "$d/t-ok.tsv"
    sed 's/\t60\t63$/\t\t63/' "$d/t-ok.tsv" > "$d/t-hole.tsv"
    for m in "0 t-ok.tsv - a sidecar is read: cheap tier closes at +63s" "1 t-ok.tsv 30 a cheap tier over its budget (30s) is OVER" \
        "0 t-ok.tsv 63 a cheap tier at its budget is not over" "2 t-hole.tsv - a row with no time is not_measured" \
        "2 t-none.tsv - no sidecar is not_measured" "2 t-ok.tsv 15m a budget that is not whole seconds is not_measured"; do
        read -ra M <<< "$m"; set -- "${M[@]}"; rc=0
        timing "$d/$2" "$( [ "$3" = - ] || printf '%s' "$3")" > "$d/out" 2>&1 || rc=$?
        shift 3
        if [ "$rc" = "${m%% *}" ]; then printf 'ok    rc=%s %s -- %s\n' "$rc" "$*" "$(tail -1 "$d/out" | cut -c1-90)"
        else printf 'FAIL  wanted rc=%s, got rc=%s: %s\n' "${m%% *}" "$rc" "$*"; bad=1; fi
    done
    [ "$bad" = 0 ] && { echo "SELF-TEST PASSED"; return 0; }
    echo "SELF-TEST FAILED" >&2; return 1
}

if [ "$SELF_TEST" = 1 ]; then self_test; exit $?; fi
if [ -n "$TIMING" ]; then
    echo "=== dogfood row time (check_dogfood_row_order.sh --timing) ==="
    timing "$TIMING" "$BUDGET"; finish $? "the cheap tier is over its budget"
fi
echo "=== dogfood: cheap rows first, long rows last, a red cheap row stops nothing (check_dogfood_row_order.sh) ==="
judge "$SRC"; finish $? "scripts/dogfood.sh does not run its cheap rows before its long rows, or can stop before them"
