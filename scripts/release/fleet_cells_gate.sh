#!/usr/bin/env bash
# fleet_cells_gate.sh -- no rc is cut while a fleet cell is RED, unless a dated waiver names it (#4328 C4)
#
#   fleet_cells_gate.sh --cells FILE [--waivers FILE] [--now EPOCH]   judge; rc 0 cut, 1 refuse
#   fleet_cells_gate.sh --self-test                                   case table + mutants
#
# rc.2 was cut with intel on an old apr, mini unable to install anything (no darwin asset)
# and jetson unreachable: every one of those cells was RED and nothing on the cut path read
# them. rc_cut.sh now runs this gate before it writes the tag.
#
# CELLS come from infra-64's andon monitor (#4328 C3), published where the clean-room cut
# job can read it: `fleet/cells.tsv` on the `fleet-state` branch of this repo.
#   # measured 2026-09-24T17:58:00Z
#   <host>\t<binary>\t<GREEN|RED>\t<reason>
# A retired host is REMOVED from the cells, not listed RED (C7: jetson, 2026-09-24).
#
# WAIVERS are recorded in git, dated, and expire: scripts/release/fleet-waivers.tsv
#   <host|*>\t<binary|*>\t<until YYYY-MM-DD>\t<reason>
# `*\t*` is the only key that covers MISSING or STALE cells -- no data is not green data.
#
# Refuses: any unwaived RED cell · a state other than GREEN/RED · no cells · no `# measured`
# stamp · cells older than FLEET_CELLS_MAX_AGE_H (default 6) hours · a --now that is not a whole
# epoch second from 0 to the end of year 9999. EXIT 0 cut · 1 refuse · 2 usage.
set -uo pipefail
PROG=fleet_cells_gate

# python's str.splitlines() and str.strip()/split() whitespace, as bytes (awk runs under LC_ALL=C)
FLEET_CELLS_AWK_LIB='
function pyws() { return "([ \t\n\v\f\r\034\035\036\037]|\302\205|\302\240|\341\232\200|\342\200[\200\201\202\203\204\205\206\207\210\211\212\250\251\257]|\342\201\237|\343\200\200)" }
function pylines(s) {
    gsub(/\r\n/, "\n", s)
    gsub(/[\r\v\f\034\035\036]|\302\205|\342\200\250|\342\200\251/, "\n", s)
    return s
}
function pystrip(s) { sub("^" pyws() "+", "", s); sub(pyws() "+$", "", s); return s }
'

# The two conversions the verdict needs, in bash arithmetic: proleptic Gregorian, as python's
# datetime is (Hinnant's days_from_civil / civil_from_days). Not date(1): bashrs 7.4.1, the CI pin,
# reads every date(1) call as DET002, even one that only converts a time it was given. Fields are
# read 10#, because bash reads a leading 0 as octal.
fleet_cells_day() {  # <epoch seconds, 0 or more> -> YYYY-MM-DD
    local z=$(( 10#$1 / 86400 + 719468 )) era doe yoe doy mp m
    era=$(( z / 146097 )); doe=$(( z - era * 146097 ))
    yoe=$(( (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365 ))
    doy=$(( doe - (365 * yoe + yoe / 4 - yoe / 100) )); mp=$(( (5 * doy + 2) / 153 ))
    m=$(( mp < 10 ? mp + 3 : mp - 9 ))
    printf '%04d-%02d-%02d\n' "$(( era * 400 + yoe + (m <= 2) ))" "$m" "$(( doy - (153 * mp + 2) / 5 + 1 ))"
}
fleet_cells_epoch() {  # <YYYY-MM-DDTHH:MM:SSZ, fields in range> -> epoch seconds, or -1 for a day the month lacks
    local s=$1 y m d era yoe days
    local -a mdays=(0 31 28 31 30 31 30 31 31 30 31 30 31)
    y=$(( 10#${s:0:4} )); m=$(( 10#${s:5:2} )); d=$(( 10#${s:8:2} ))
    if (( d > mdays[m] + (m == 2 && !(y % 4) && (y % 100 || !(y % 400))) )); then echo -1; return; fi
    y=$(( y - (m <= 2) )); era=$(( (y >= 0 ? y : y - 399) / 400 )); yoe=$(( y - era * 400 ))
    days=$(( era * 146097 + yoe * 365 + yoe / 4 - yoe / 100 + (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1 - 719468 ))
    echo $(( days * 86400 + 10#${s:11:2} * 3600 + 10#${s:14:2} * 60 + 10#${s:17:2} ))
}

# fleet_cells_verdict <cells text> <waivers text> <now epoch> -> `ok ...` rc 0 | `refuse ...` rc 1
# bash + awk only (ARB-AUD-10: no interpreter beyond the shell in this gate; rc_cut.sh, its caller,
# still runs python3 elsewhere). The two helpers above convert
# the stamp and the clock; awk gets the texts and the max age through ENVIRON (-v expands backslashes). The judge it replaced was a
# python3 heredoc, so awk reads the texts as python did, byte for byte under LC_ALL=C: lines break
# where str.splitlines() breaks (CRLF, CR, VT, FF, FS/GS/RS, NEL, LS, PS), "blank" is str.strip()'s
# whitespace, fields compare as strings, never as numbers, and FLEET_CELLS_MAX_AGE_H must be a
# decimal number. Where the two can differ, awk refuses and python cut (#4350).
fleet_cells_verdict() {
    local stamp measured=-1 today
    # python's int() + fromtimestamp(utc) refused a clock that is not an integer, or past year 9999,
    # with rc 1; refuse both here (1.5, 1000000000000 = year 33658). The digits are a set, never a
    # range: under a UTF-8 locale bash's [0-9] also matches other scripts' digits (Arabic-Indic 0-8).
    if ! [[ $3 =~ ^[0123456789]{1,12}$ ]] || (( 10#$3 > 253402300799 )); then
        printf 'refuse clock %q is not a whole epoch second up to 9999-12-31T23:59:59Z\n' "$3"
        return 1
    fi
    # an empty today would let every dated waiver cover: no day, no verdict
    today=$(fleet_cells_day "$3") || return 2
    [[ $today =~ ^[0123456789]{4}-[0123456789]{2}-[0123456789]{2}$ ]] || return 2
    stamp=$(CELLS=$1 LC_ALL=C awk "$FLEET_CELLS_AWK_LIB"'
    BEGIN {
        nl = split(pylines(ENVIRON["CELLS"]), L, "\n")
        for (i = 1; i <= nl; i++) {
            if (index(L[i], "# measured ") != 1 || split(pystrip(L[i]), p, pyws() "+") < 3) continue
            s = L[i]; sub(/^# measured/, "", s); print pystrip(s); exit
        }
    }')
    if [[ $stamp =~ ^[0123456789]{4}-(0[123456789]|1[012])-(0[123456789]|[12][0123456789]|3[01])T([01][0123456789]|2[0123]):[012345][0123456789]:[012345][0123456789]Z$ ]]; then
        # a stamp before 1970, year 0000 included (python had no year 0), is negative: no stamp
        measured=$(fleet_cells_epoch "$stamp")
    fi
    CELLS=$1 WAIVERS=$2 STAMP=$stamp MAX_AGE_H=${FLEET_CELLS_MAX_AGE_H:-6} LC_ALL=C \
        awk -v now="$3" -v measured="$measured" -v today="$today" "$FLEET_CELLS_AWK_LIB"'
    # waived(host, binary): host "*" asks for the `*\t*` key only (missing/stale cells)
    function waived(host, binary,    nw, w, i, f, hit) {
        nw = split(pylines(ENVIRON["WAIVERS"]), w, "\n")
        for (i = 1; i <= nw; i++) {
            if (pystrip(w[i]) == "" || substr(pystrip(w[i]), 1, 1) == "#") continue
            if (split(w[i], f, "\t") < 4 || pystrip(f[4]) == "") continue
            if (host == "*") hit = (f[1] == "*" && f[2] == "*")
            else hit = (((f[1] "") == (host "") || f[1] == "*") && ((f[2] "") == (binary "") || f[2] == "*"))
            if (hit && (f[3] "") >= today) return f[1] "/" f[2] " until " f[3] ": " f[4]
        }
        return ""
    }
    function add(msg) { bad[++nb] = msg }
    # repr(): how python quoted a malformed line or an unknown state
    function pyrepr(s,    q, out, i, ch, k) {
        q = (index(s, "\x27") && !index(s, "\"")) ? "\"" : "\x27"
        out = q
        for (i = 1; i <= length(s); i++) {
            ch = substr(s, i, 1); k = ORD[ch]
            if (ch == "\\" || ch == q) out = out "\\" ch
            else if (ch == "\t") out = out "\\t"
            else if (k < 32 || k == ORD["\177"]) out = out sprintf("\\x%02x", k)
            else out = out ch
        }
        return out q
    }
    BEGIN {
        for (i = 1; i < 256; i++) ORD[sprintf("%c", i)] = i
        max_age = ENVIRON["MAX_AGE_H"]
        # python: a float() ValueError fell into the same except as a bad stamp
        if (max_age !~ /^[ \t\n\v\f\r]*[+-]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][+-]?[0-9]+)?[ \t\n\v\f\r]*$/) measured = -1
        # Data faults (stale, unstamped, empty) are the only ones `*\t*` may waive.
        if (measured < 0) { add("cells carry no \x27# measured <YYYY-MM-DDTHH:MM:SSZ>\x27 stamp"); nd++ }
        else if ((now - measured) / 3600 > max_age + 0) {
            add(sprintf("cells are stale: measured %s, %.1f h ago (max %s h)", ENVIRON["STAMP"], (now - measured) / 3600, max_age)); nd++
        }
        nl = split(pylines(ENVIRON["CELLS"]), L, "\n")
        for (l = 1; l <= nl; l++) {
            if (pystrip(L[l]) == "" || substr(L[l], 1, 1) == "#") continue
            if (split(L[l], c, "\t") < 3) { add("malformed cell: " pyrepr(L[l])); continue }
            n++
            if (c[3] == "GREEN") continue
            if (c[3] == "RED") {
                w = waived(c[1], c[2])
                if (w != "") notes[++nn] = c[1] "/" c[2] " RED waived (" w ")"
                else add(c[1] "/" c[2] " RED: " (c[4] != "" ? c[4] : "no reason given"))
                continue
            }
            add(c[1] "/" c[2] " state " pyrepr(c[3]) " is neither GREEN nor RED")
        }
        if (n == 0) { add("no fleet cells"); nd++ }
        if (nd > 0 && nd == nb) {
            w = waived("*", "*")
            if (w != "") {
                msg = bad[1]; for (i = 2; i <= nb; i++) msg = msg "; " bad[i]
                notes[++nn] = "cells unusable, waived (" w "): " msg
                nb = 0
            }
        }
        if (nb > 0) {
            msg = bad[1]; for (i = 2; i <= nb; i++) msg = msg "; " bad[i]
            print "refuse " msg
            exit 1
        }
        msg = ""; for (i = 1; i <= nn; i++) msg = msg "; " notes[i]
        print "ok " n + 0 " fleet cells, none RED unwaived" msg
    }'
}

self_test() {
    local fail=0 got rc now d mut
    now=$(date -u -d 2026-09-24T18:00:00Z +%s)  # bashrs disable-line=DET002
    local S='# measured 2026-09-24T17:58:00Z'
    row() {  # row <want rc> <label> <cells> [waivers]
        got=$(fleet_cells_verdict "$(printf '%b' "$3")" "$(printf '%b' "${4:-}")" "$now"); rc=$?
        if [ "$rc" = "$1" ]; then echo "  ok   $2 -> rc $rc"; else printf '  FAIL %s: want rc %s, got %s\n       %s\n' "$2" "$1" "$rc" "$got"; fail=1; fi
    }
    echo "$PROG self-test"
    row 0 'every cell GREEN' "$S\nlambda\tapr\tGREEN\t\nmini\tapr\tGREEN\t\n"
    row 1 'ONE RED cell refuses the cut' "$S\nlambda\tapr\tGREEN\t\nintel\tapr\tRED\tv0.69.1 on PATH\n"
    row 0 'a dated waiver covers that cell' "$S\nintel\tapr\tRED\told\n" 'intel\tapr\t2026-09-24\tintel reimage, cop 2026-09-24\n'
    row 0 'a host-wide waiver covers any binary on it' "$S\nintel\tpv\tRED\told\n" 'intel\t*\t2026-09-30\treimage\n'
    row 1 'an EXPIRED waiver covers nothing' "$S\nintel\tapr\tRED\told\n" 'intel\tapr\t2026-09-23\tx\n'
    row 1 'a waiver for another host covers nothing' "$S\nintel\tapr\tRED\told\n" 'mini\tapr\t2026-12-31\tx\n'
    row 1 'a waiver with no reason is not recorded' "$S\nintel\tapr\tRED\told\n" 'intel\tapr\t2026-12-31\t\n'
    row 1 'an unknown state refuses' "$S\nyoga\tapr\tAMBER\t\n"
    row 1 'no cells refuses: no data is not green' "$S\n"
    row 1 'no measured stamp refuses' 'lambda\tapr\tGREEN\t\n'
    row 1 'stale cells refuse' '# measured 2026-09-24T09:00:00Z\nlambda\tapr\tGREEN\t\n'
    row 0 'only *\t* waives missing cells' '' '*\t*\t2026-09-24\tC3 not publishing yet, cop 2026-09-24\n'
    row 1 'a host waiver does not waive missing cells' '' 'intel\t*\t2026-09-30\tx\n'
    row 1 'a waiver for another binary covers nothing' "$S\nintel\tapr\tRED\told\n" '*\tpv\t2026-09-30\tx\n'
    # #4350: the cases where a naive awk port cut and the python judge refused.
    row 1 'a CRLF waiver with no reason is not recorded' "$S\nh1\tapr\tRED\told\n" 'h1\tapr\t2026-12-31\t\r\n'
    row 1 'a no-break-space reason is no reason' "$S\nh1\tapr\tRED\told\n" 'h1\tapr\t2026-12-31\t\0302\0240\n'
    row 1 'hosts compare as strings: waiver 01 is not host 1' "$S\n1\tapr\tRED\told\n" '01\tapr\t2026-12-31\tx\n'
    row 1 'binaries compare as strings: waiver 01 is not binary 1' "$S\nh1\t1\tRED\told\n" 'h1\t01\t2026-12-31\tx\n'
    row 1 'the *\t* waiver does not waive an unknown state beside stale cells' '# measured 2026-09-24T09:00:00Z\nh2\tapr\tAMBER\t\n' '*\t*\t2026-09-30\tx\n'
    row 1 'a vertical tab ends a line: what follows is a malformed cell' "$S\nh2\tapr\tGREEN\tok\vjunk\n"
    FLEET_CELLS_MAX_AGE_H=abc row 1 'a max age that is not a number refuses' "$S\nh2\tapr\tGREEN\t\n"
    FLEET_CELLS_MAX_AGE_H='\066' row 1 'a max age is not unescaped' "$S\nh2\tapr\tGREEN\t\n"
    FLEET_CELLS_MAX_AGE_H=' 1e3 ' row 0 'a max age python reads as a float is read' '# measured 2026-09-01T00:00:00Z\nh2\tapr\tGREEN\t\n'
    FLEET_CELLS_MAX_AGE_H=6e row 1 'a max age with a number in front is still not a number' "$S\nh2\tapr\tGREEN\t\n"
    now=1.5 row 1 'a clock that is not an integer refuses' "$S\nh2\tapr\tGREEN\t\n"
    now=1000000000000 row 1 'a clock past year 9999 refuses, even under a *\t* waiver to 9999' '# measured 2026-09-24T09:00:00Z\nh2\tapr\tGREEN\t\n' '*\t*\t9999-12-31\tx\n'
    now=253402300799 row 0 'the last second of year 9999 is still a clock' '# measured 9999-12-31T23:58:00Z\nh2\tapr\tGREEN\t\n'
    now=253402300800 row 1 'the first second of year 10000 refuses, even under a *\t* waiver to 9999' '# measured 2026-09-24T09:00:00Z\nh2\tapr\tGREEN\t\n' '*\t*\t9999-12-31\tx\n'
    # The calendar arithmetic that replaced date(1): each row flips if a leap or month rule is dropped.
    now=1835442000 row 0 'a leap day is a stamp (2028-02-29)' '# measured 2028-02-29T12:00:00Z\nh2\tapr\tGREEN\t\n'
    now=951829200 row 0 'a leap day in a year divisible by 400 is a stamp (2000-02-29)' '# measured 2000-02-29T12:00:00Z\nh2\tapr\tGREEN\t\n'
    now=1772366400 row 1 'a day the year lacks is no stamp (2026-02-29)' '# measured 2026-02-29T12:00:00Z\nh2\tapr\tGREEN\t\n'
    now=4107585600 row 1 'a century year is not a leap year (2100-02-29)' '# measured 2100-02-29T12:00:00Z\nh2\tapr\tGREEN\t\n'
    now=1777636800 row 1 'a day a 30-day month lacks is no stamp (2026-04-31)' '# measured 2026-04-31T12:00:00Z\nh2\tapr\tGREEN\t\n'
    now=4107542400 row 0 'the clock 2100-03-01 is that day: a waiver to it covers' '# measured 2100-02-28T23:00:00Z\nh2\tapr\tRED\told\n' 'h2\tapr\t2100-03-01\tx\n'
    now=4107542400 row 1 'and is not 2100-02-29: a waiver to that day has expired' '# measured 2100-02-28T23:00:00Z\nh2\tapr\tRED\told\n' 'h2\tapr\t2100-02-29\tx\n'
    now=1800014400 row 1 'a January clock is that year: a waiver to the December before has expired' '# measured 2027-01-15T11:58:00Z\nh1\tapr\tRED\told\n' 'h1\tapr\t2026-12-31\texpired\n'
    now=1798848000 row 1 'a January stamp a day old is stale' '# measured 2027-01-01T00:00:00Z\nh1\tapr\tGREEN\t\n'
    now=01790272800 row 1 'a zero-padded clock is the same day: an expired waiver covers nothing' "$S\nh2\tapr\tRED\told\n" 'h2\tapr\t2026-09-23\tx\n'
    row 1 'six hours and one second is stale' '# measured 2026-09-24T11:59:59Z\nh2\tapr\tGREEN\t\n'
    row 0 'exactly six hours is not stale' '# measured 2026-09-24T12:00:00Z\nh2\tapr\tGREEN\t\n'
    now=1788944400 row 0 'a stamp with 08 and 09 in its day and time is a stamp' '# measured 2026-09-09T08:08:09Z\nh2\tapr\tGREEN\t\n'
    # 1788000000 in Arabic-Indic digits, no 9 among them: [0-9] matched it under en_US.UTF-8
    now=$(printf '\331\241\331\247\331\250\331\250\331\240\331\240\331\240\331\240\331\240\331\240') \
        row 1 'a clock in digits that are not ASCII refuses' '# measured 2026-08-01T00:00:00Z\nh2\tapr\tGREEN\t\n'
    # MUTANTS: the refusal made a no-op must let the RED cell through.
    d=$(mktemp -d) || return 2
    for m in 's/                else add(c\[1\] "\/" c\[2\] " RED: "/                else ("\/" " RED: "/' 's/^            exit 1$/            exit 0/'; do
        mut=$d/m.sh; sed "$m" "${BASH_SOURCE[0]}" > "$mut"
        if cmp -s "$mut" "${BASH_SOURCE[0]}"; then echo "  FAIL mutant '$m' not built: the anchor moved"; fail=1; continue; fi
        got=$(bash -c ". '$mut' --source-only; fleet_cells_verdict \"\$(printf '%b' '$S\nintel\tapr\tRED\told\n')\" '' $now" 2>&1); rc=$?
        if [ "$rc" = 0 ]; then echo "  ok   mutant lets the RED cell through: the refusal is load-bearing"
        else echo "  FAIL mutant still refuses (rc $rc): $got"; fail=1; fi
    done
    # WIRING: rc_cut.sh runs this gate before it writes the tag, and dies on a refusal.
    local cut; cut="$(dirname -- "${BASH_SOURCE[0]}")/rc_cut.sh"
    local call tagline
    call=$(grep -n 'fleet_cells_gate.sh' "$cut" | grep -v '^\s*[0-9]*:\s*#' | head -n 1 | cut -d: -f1)
    tagline=$(grep -nF 'out=$(api_post git/refs' "$cut" | head -n 1 | cut -d: -f1)
    if [ -n "$call" ] && [ -n "$tagline" ] && [ "$call" -lt "$tagline" ]; then echo "  ok   rc_cut.sh runs the gate (line $call) before it writes the tag (line $tagline)"
    else echo "  FAIL rc_cut.sh does not run fleet_cells_gate.sh before creating the tag (call=${call:-none}, tag=${tagline:-none})"; fail=1; fi
    rm -rf -- "${d:?}"
    if [ "$fail" = 0 ]; then echo "$PROG self-test: PASS"; else echo "$PROG self-test: FAIL"; fi
    return "$fail"
}

main() {
    local cells='' waivers='' now
    now=$(date -u +%s)  # bashrs disable-line=DET002
    while [ $# -gt 0 ]; do
        case "$1" in
            --self-test) self_test; return $? ;;
            --source-only) return 0 ;;
            --cells) cells=$2; shift 2 ;;
            --waivers) waivers=$2; shift 2 ;;
            --now) now=$2; shift 2 ;;
            *) echo "usage: $PROG --cells FILE [--waivers FILE] [--now EPOCH] | --self-test" >&2; return 2 ;;
        esac
    done
    [ -n "$cells" ] || { echo "usage: $PROG --cells FILE [--waivers FILE]" >&2; return 2; }
    fleet_cells_verdict "$(cat -- "$cells" 2>/dev/null)" "$([ -n "$waivers" ] && cat -- "$waivers" 2>/dev/null)" "$now"
}

if [ "${1:-}" = --source-only ]; then return 0 2>/dev/null || exit 0; fi
main "$@"
