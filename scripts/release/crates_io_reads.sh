#!/usr/bin/env bash
# crates_io_reads.sh — what crates.io answers for each crate of a release, written as the reads file the E5 timer
#   judges (BLD-001 row 5; scripts/release/tag_to_crates.sh)
#
# WHY A SECOND SCRIPT. tag_to_crates.sh is a judge: it makes no network call, so its case table is exact and a
#   replay of the same reads gives the same verdict. This script is the one that reads crates.io. It judges
#   nothing: it writes down what crates.io answered and when, and keeps "not published" apart from "not known".
#
# ONE READ per crate of the order file, in its order:
#     GET https://crates.io/api/v1/crates/<crate>/<version>
#   with a User-Agent that names this script (crates.io refuses a request without one), a 20 s timeout, https only,
#   no redirect followed, and a 1 s pause after each request (the crates.io crawler policy: one request a second).
#     200 that names this crate and this version, with version.created_at a UTC time
#           -> a row. created_at is floored to the second (16:40:00.123456Z -> 16:40:00Z; +00:00 is written Z).
#     404   -> a row with an empty created_at: crates.io has no such version, not yet or not at all.
#     anything else: no answer, a cut transfer, a timeout, a 429, a 5xx, a redirect, or a 200 that names another
#           crate or version, has no created_at, or gives it another offset
#           -> NO row, and one FAIL line. "Not published" and "not known" are different facts: the judge reads a
#              crate with no row as not_measured, never as published and never as missing.
#   read_at is the UTC second the answer came, taken after it and floored. A created_at later than its read_at
#   (clocks more than a second apart) makes the judge's whole verdict not_measured, never a pass.
#
# THE FILE. --out names a new file. One that exists is refused, so earlier reads are never overwritten. It is
#   written once, after the last read: a run that is stopped leaves no file, never a part of one.
#     crate<TAB>version<TAB>created_at<TAB>read_at       then one row per crate that has one, in order-file order
#
# EXIT  0 every crate has a row · 2 a read failed: its crate has no row, the other rows are written (read again
#   into a new file) · 3 caller error, found before any request. Never 1: reading is not judging.
#
# USAGE
#   crates_io_reads.sh --version X.Y.Z --order FILE --out FILE
#       the order file is the cascade's own: git show v<version>:scripts/release/publish-order.txt
#   crates_io_reads.sh --selftest    the case table, against a stand-in curl, date and sleep (no network)
#   crates_io_reads.sh --mutants     each planted mutant must turn the case table RED
set -uo pipefail
export LC_ALL=C

PROG="${0##*/}"
SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/${BASH_SOURCE[0]##*/}"
API=https://crates.io/api/v1/crates
UA="paiml-aprender-crates-io-reads (BLD-001 row 5)"
VER_RE='^[0-9]+[.][0-9]+[.][0-9]+(-[0-9A-Za-z.-]+)?$'
UTC_RE='^([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2})([.][0-9]+)?(Z|[+]00:00)$'

caller_error() { printf 'FAIL  CRATESREAD %s: caller error: %s\n' "$PROG" "$*"; exit 3; }

# iscrate NAME: a crates.io name, as the judge reads one. It also keeps the URL to one path segment.
iscrate() { [[ "$1" =~ ^[A-Za-z][A-Za-z0-9_-]*$ ]] && [ "${#1}" -le 64 ]; }

# created_at CRATE VERSION < BODY -> "ok <created_at>" when the answer names this crate and version, else the reason.
#   An answer jq cannot read whole (not JSON, empty, or JSON with more after it) is "is not a JSON object".
created_at() {
    local o
    o="$(jq -r --arg c "$1" --arg v "$2" '.version as $x
        | if ($x | type) != "object" then "has no version object"
          elif $x.crate != $c then "names crate \($x.crate)"
          elif $x.num != $v then "names version \($x.num)"
          elif ($x.created_at | type) != "string" then "has no created_at"
          else "ok \($x.created_at)" end' 2>/dev/null)" || o=""
    printf '%s\n' "${o:-is not a JSON object}"
}

# fetch VERSION ORDER OUT -> the reads file; a summary line and one FAIL line per failed read; 0 or 2
fetch() {
    local ver="$1" order="$2" out="$3" ln m=0 c resp code cx body r ca ra rows="" nrow=0 nnot=0 nfail=0
    local -a crates=()
    local -A seen=()
    while IFS= read -r ln || [ -n "$ln" ]; do
        m=$((m + 1))
        iscrate "$ln" || caller_error "the order file line $m \"$ln\" is not a crate name"
        [ -z "${seen[$ln]:-}" ] || caller_error "the order file lists $ln twice (lines ${seen[$ln]} and $m)"
        seen[$ln]=$m
        crates+=("$ln")
    done < "$order"
    [ "$m" -gt 0 ] || caller_error "the order file lists no crate"
    for c in "${crates[@]}"; do
        cx=0
        # stdout is the body, a newline, then the HTTP code (000 when nothing came back): no file to clean up
        resp="$(curl -sS -w '\n%{http_code}' --max-time 20 --proto =https -H "User-Agent: $UA" "$API/$c/$ver" 2>/dev/null)" || cx=$?
        ra="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        sleep 1
        code="${resp##*$'\n'}"
        body="${resp%$'\n'*}"
        r="HTTP $code"
        if [ "$cx" -ne 0 ]; then
            r="no complete answer (HTTP ${code:-000}, curl exit $cx)"
        elif [ "$code" = 404 ]; then
            rows+="$c"$'\t'"$ver"$'\t'$'\t'"$ra"$'\n'; nnot=$((nnot + 1)); continue
        elif [ "$code" = 200 ]; then
            r="$(created_at "$c" "$ver" <<< "$body")"
            case "$r" in
                "ok "*) ca="${r#ok }"
                    if [[ "$ca" =~ $UTC_RE ]]; then
                        rows+="$c"$'\t'"$ver"$'\t'"${BASH_REMATCH[1]}Z"$'\t'"$ra"$'\n'; nrow=$((nrow + 1)); continue
                    fi
                    r="created_at \"$ca\" is not a UTC time" ;;
                *) r="the answer $r" ;;
            esac
        fi
        printf 'FAIL  CRATESREAD %s %s: %s; no row\n' "$c" "$ver" "$r"
        nfail=$((nfail + 1))
    done
    printf 'crate\tversion\tcreated_at\tread_at\n%s' "$rows" > "$out" || { echo "FAIL  CRATESREAD cannot write $out"; return 2; }
    printf 'READ  CRATESREAD %s: %s crate(s): %s published, %s not on crates.io, %s failed -> %s\n' \
        "$ver" "$m" "$nrow" "$nnot" "$nfail" "$out"
    [ "$nfail" -eq 0 ] || return 2
}

main() {
    local ver="" order="" out="" d
    while [ $# -gt 0 ]; do
        case "$1" in
            --version|--order|--out)
                [ $# -ge 2 ] || caller_error "$1 needs a value"
                case "$1" in
                    --version) ver="$2" ;;
                    --order) order="$2" ;;
                    *) out="$2" ;;
                esac
                shift 2 ;;
            *) caller_error "unknown option $1 (usage: $PROG --version X.Y.Z --order FILE --out FILE)" ;;
        esac
    done
    [ -n "$ver" ] || caller_error "--version is required: every crate is read at one version"
    [[ "$ver" =~ $VER_RE ]] || caller_error "--version \"$ver\" is not MAJOR.MINOR.PATCH"
    { [ -n "$order" ] && [ -f "$order" ] && [ -r "$order" ]; } || caller_error "--order '$order' is not a readable file"
    [ -n "$out" ] || caller_error "--out is required: the reads go to a new file"
    [ ! -e "$out" ] || caller_error "--out '$out' exists: reads are never overwritten; name a new file"
    d="$(dirname -- "$out")"
    { [ -d "$d" ] && [ -w "$d" ]; } || caller_error "--out '$out': its directory is not a writable directory"
    command -v curl > /dev/null || caller_error "curl is not on PATH"
    command -v jq > /dev/null || caller_error "jq is not on PATH"
    fetch "$ver" "$order" "$out"
}

# --------------------------------------------------------------- selftest ---
selftest_cleanup() { case "${tmp:-}" in ''|/) return 0 ;; *) rm -rf -- "${tmp:?}" ;; esac; }
selftest() {
    local tmp pass=0 fail=0 V=0.70.2 D=2026-10-03 H NL=$'\n' TB=$'\t' L o rc=0
    tmp="$(mktemp -d)"
    case "$tmp" in ?*/tmp.?*) : ;; *) echo "mktemp gave '${tmp:-}', refusing"; return 2 ;; esac
    # SEC011: validate before rm -rf. $tmp is this function's own mktemp -d (checked above).
    trap selftest_cleanup RETURN
    mkdir -p "$tmp/bin" "$tmp/fx" "$tmp/out" "$tmp/t"
    # Stand-ins, found first on PATH. curl logs its arguments and answers from $FX/<crate>.*: the body, then a newline
    #   and the code, as -w '\n%{http_code}' makes the real one do;
    # date logs its arguments and gives 16:40:01Z, 16:40:02Z, ... in turn; sleep logs its arguments.
    cat > "$tmp/bin/curl" <<'STUB'
#!/usr/bin/env bash
w="" u="" p="" a
{ printf 'CURL'; for a in "$@"; do printf ' [%s]' "$a"; [ "$p" = -w ] && w="$a"; case "$a" in https://*) u="$a" ;; esac; p="$a"; done; printf '\n'; } >> "$LOG"
c="${u#https://crates.io/api/v1/crates/}"; c="${c%%/*}"
if [ -f "$FX/$c.body" ]; then cat -- "$FX/$c.body"; fi
if [ "$w" = '\n%{http_code}' ]; then printf '\n%s' "$(cat -- "$FX/$c.code" 2>/dev/null || printf 000)"; fi
exit "$(cat -- "$FX/$c.exit" 2>/dev/null || printf 0)"
STUB
    cat > "$tmp/bin/date" <<'STUB'
#!/usr/bin/env bash
printf 'DATE' >> "$LOG"; printf ' [%s]' "$@" >> "$LOG"; printf '\n' >> "$LOG"
n=$(( $(cat -- "$DN") + 1 )); printf '%s' "$n" > "$DN"; printf '2026-10-03T16:40:%02dZ\n' "$n"
STUB
    cat > "$tmp/bin/sleep" <<'STUB'
#!/usr/bin/env bash
printf 'SLEEP' >> "$LOG"; printf ' [%s]' "$@" >> "$LOG"; printf '\n' >> "$LOG"
STUB
    chmod +x "$tmp/bin/curl" "$tmp/bin/date" "$tmp/bin/sleep"
    H="crate${TB}version${TB}created_at${TB}read_at"
    printf '%s\n' apr-format aprender aprender-tsp > "$tmp/o3"
    printf '%s\n' aprender > "$tmp/o1"
    # fx crate code [body [curl-exit]]: one stand-in answer; a body of "-" means none
    fx() {
        printf '%s' "$2" > "$tmp/fx/$1.code"
        rm -f -- "${tmp:?}/fx/${1:?}.body"
        if [ "${3:--}" != - ]; then printf '%s' "$3" > "$tmp/fx/$1.body"; fi
        printf '%s' "${4:-0}" > "$tmp/fx/$1.exit"
    }
    vb() { printf '{"version":{"id":1,"crate":"%s","num":"%s","created_at":"%s","yanked":false}}' "$1" "$2" "$3"; }
    fx3() {
        fx apr-format 200 "$(vb apr-format "$V" "${D}T16:30:00.123456+00:00")"
        fx aprender 200 "$(vb aprender "$V" "${D}T16:31:00Z")"
        fx aprender-tsp 200 "$(vb aprender-tsp "$V" "${D}T16:32:00.999999Z")"
    }
    # want rows "crate created read": - is empty, HH:MM:SS is that time on $D -> the expected file
    want() {
        local r c ca ra s="$H$NL"
        for r in "$@"; do
            read -r c ca ra <<< "$r"
            if [ "$ca" = - ]; then ca=""; else ca="${D}T${ca}Z"; fi
            s+="$c$TB$V$TB$ca$TB${D}T${ra}Z$NL"
        done
        printf '%s' "$s"
    }
    # row name expect-rc needle expected-file|- -- args: a fresh call log; the output file is compared byte for byte
    # with the expected file, which is given without its last newline (a command substitution drops it) and gets it back here
    row() {
        local name="$1" expect="$2" needle="$3" wantf="$4" o rc=0 got out="$tmp/out/$1.tsv"; shift 5
        : > "$tmp/log"; printf 0 > "$tmp/dn"
        o="$(env PATH="$tmp/bin:$PATH" FX="$tmp/fx" LOG="$tmp/log" DN="$tmp/dn" TMPDIR="$tmp/t" bash "$SCRIPT_PATH" "$@" < /dev/null 2>&1)" || rc=$?
        if [ "$rc" != "$expect" ]; then printf '  BROKE %-50s expected exit %s got %s\n%s\n' "$name" "$expect" "$rc" "$o"; fail=$((fail + 1)); return 0; fi
        case "$o" in *"$needle"*) : ;; *) printf '  BROKE %-50s exit %s but never said: %s\n%s\n' "$name" "$rc" "$needle" "$o"; fail=$((fail + 1)); return 0 ;; esac
        if [ "$expect" = 3 ] && [ -s "$tmp/log" ]; then printf '  BROKE %-50s a caller error made a request\n' "$name"; fail=$((fail + 1)); return 0; fi
        if [ "$wantf" != - ]; then
            got="$(cat -- "$out" 2>/dev/null; printf x)"; got="${got%x}"
            if [ "$got" != "$wantf$NL" ]; then printf '  BROKE %-50s the reads file differs\n--- want\n%s\n--- got\n%s\n' "$name" "$wantf" "$got"; fail=$((fail + 1)); return 0; fi
        fi
        printf '  ok    %-50s exit=%s\n' "$name" "$rc"; pass=$((pass + 1))
    }
    # r name expect-rc needle expected-file|-: the three-crate order, --out a new file named after the row
    r() { row "$1" "$2" "$3" "$4" -- --version "$V" --order "$tmp/o3" --out "$tmp/out/$1.tsv"; }
    # same name expected-log: the call log of the last row, compared byte for byte
    same() {
        local got; got="$(cat -- "$tmp/log"; printf x)"; got="${got%x}"
        if [ "$got" = "$2" ]; then printf '  ok    %-50s calls as expected\n' "$1"; pass=$((pass + 1))
        else printf '  BROKE %-50s the calls differ\n--- want\n%s--- got\n%s\n' "$1" "$2" "$got"; fail=$((fail + 1)); fi
    }
    req() { printf 'CURL [-sS] [-w] [\\n%%{http_code}] [--max-time] [20] [--proto] [=https] [-H] [User-Agent: paiml-aprender-crates-io-reads (BLD-001 row 5)] [https://crates.io/api/v1/crates/%s/%s]\nDATE [-u] [+%%Y-%%m-%%dT%%H:%%M:%%SZ]\nSLEEP [1]\n' "$1" "$V"; }

    fx3
    r all_published_rows_in_order_floored 0 "READ  CRATESREAD $V: 3 crate(s): 3 published, 0 not on crates.io, 0 failed" \
        "$(want "apr-format 16:30:00 16:40:01" "aprender 16:31:00 16:40:02" "aprender-tsp 16:32:00 16:40:03")"
    L="$(req apr-format)$NL$(req aprender)$NL$(req aprender-tsp)$NL"
    same one_get_per_crate_ua_timeout_https_utc_1s_pause "$L"
    o="$(bash "$(dirname -- "$SCRIPT_PATH")/tag_to_crates.sh" --version "$V" --tag-time "${D}T16:00:00Z" --as-of "${D}T17:00:00Z" \
        --order "$tmp/o3" --reads "$tmp/out/all_published_rows_in_order_floored.tsv" < /dev/null 2>&1)" || rc=$?
    case "$rc:$o" in
        0:*"E5 met: tag to crates.io for $V took 0h32m00s"*) printf '  ok    %-50s exit=0\n' the_judge_reads_the_file; pass=$((pass + 1)) ;;
        *) printf '  BROKE %-50s exit %s\n%s\n' the_judge_reads_the_file "$rc" "$o"; fail=$((fail + 1)) ;;
    esac
    fx3; fx aprender 404 '{"errors":[{"detail":"crate `aprender` does not have a version `0.70.2`"}]}'
    r a_404_is_a_row_with_no_created_at 0 "2 published, 1 not on crates.io, 0 failed" \
        "$(want "apr-format 16:30:00 16:40:01" "aprender - 16:40:02" "aprender-tsp 16:32:00 16:40:03")"
    fx3; fx aprender 000 - 28
    r a_timeout_is_no_row 2 "FAIL  CRATESREAD aprender $V: no complete answer (HTTP 000, curl exit 28); no row" \
        "$(want "apr-format 16:30:00 16:40:01" "aprender-tsp 16:32:00 16:40:03")"
    fx3; fx aprender 200 "$(vb aprender "$V" "${D}T16:31:00Z")" 18
    r a_cut_transfer_is_no_row 2 "FAIL  CRATESREAD aprender $V: no complete answer (HTTP 200, curl exit 18); no row" \
        "$(want "apr-format 16:30:00 16:40:01" "aprender-tsp 16:32:00 16:40:03")"
    fx3; fx aprender 429 '{"errors":[{"detail":"rate limited"}]}'
    r a_429_is_no_row 2 "FAIL  CRATESREAD aprender $V: HTTP 429; no row" "$(want "apr-format 16:30:00 16:40:01" "aprender-tsp 16:32:00 16:40:03")"
    fx3; fx aprender 503 -
    r a_5xx_is_no_row 2 "FAIL  CRATESREAD aprender $V: HTTP 503; no row" -
    fx3; fx aprender 301 "$(vb aprender "$V" "${D}T16:31:00Z")"
    r a_redirect_is_no_row 2 "FAIL  CRATESREAD aprender $V: HTTP 301; no row" -
    fx3; fx aprender 200 "$(vb aprender-core "$V" "${D}T16:31:00Z")"
    r another_crate_is_no_row 2 "aprender $V: the answer names crate aprender-core; no row" -
    fx3; fx aprender 200 "$(vb aprender 0.70.1 "${D}T16:31:00Z")"
    r another_version_is_no_row 2 "aprender $V: the answer names version 0.70.1; no row" -
    fx3; fx aprender 200 '{"version":{"crate":"aprender","num":"0.70.2"}}'
    r no_created_at_is_no_row 2 "aprender $V: the answer has no created_at; no row" -
    fx3; fx aprender 200 '{"errors":[]}'
    r no_version_object_is_no_row 2 "aprender $V: the answer has no version object; no row" -
    fx3; fx aprender 200 '<html>busy</html>'
    r a_body_that_is_not_json_is_no_row 2 "aprender $V: the answer is not a JSON object; no row" -
    fx3; fx aprender 200 -
    r an_empty_answer_is_no_row 2 "aprender $V: the answer is not a JSON object; no row" -
    fx3; fx aprender 200 "$(vb aprender "$V" "${D}T16:31:00Z")<html>"
    r json_with_more_after_it_is_no_row 2 "aprender $V: the answer is not a JSON object; no row" -
    fx3; fx aprender 200 "$(vb aprender "$V" "${D}T18:31:00+02:00")"
    r an_offset_other_than_utc_is_no_row 2 "aprender $V: created_at \"${D}T18:31:00+02:00\" is not a UTC time; no row" -
    fx3; fx aprender 200 "$(vb aprender "$V" "${D} 16:31:00Z")"
    r a_created_at_without_t_is_no_row 2 "aprender $V: created_at \"${D} 16:31:00Z\" is not a UTC time; no row" -
    fx apr-format 000 - 6; fx aprender 000 - 6; fx aprender-tsp 000 - 6
    r every_read_failed_writes_only_the_header 2 "0 published, 0 not on crates.io, 3 failed" "$H"
    fx3
    printf 'old\n' > "$tmp/out/kept.tsv"
    row an_existing_out_is_refused 3 "caller error: --out '$tmp/out/kept.tsv' exists: reads are never overwritten" - -- --version "$V" --order "$tmp/o3" --out "$tmp/out/kept.tsv"
    if [ "$(cat -- "$tmp/out/kept.tsv")" = old ]; then printf '  ok    %-50s kept\n' the_refused_out_file_is_unchanged; pass=$((pass + 1))
    else printf '  BROKE %-50s it was overwritten\n' the_refused_out_file_is_unchanged; fail=$((fail + 1)); fi
    row out_dir_must_exist 3 "caller error: --out '$tmp/nodir/x.tsv': its directory is not a writable directory" - -- --version "$V" --order "$tmp/o3" --out "$tmp/nodir/x.tsv"
    row version_is_required 3 "caller error: --version is required" - -- --order "$tmp/o3" --out "$tmp/out/v.tsv"
    row a_v_prefixed_version_is_a_caller_error 3 "caller error: --version \"v$V\" is not MAJOR.MINOR.PATCH" - -- --version "v$V" --order "$tmp/o3" --out "$tmp/out/v.tsv"
    fx aprender 404 -
    row a_prerelease_version_is_read 0 "READ  CRATESREAD $V-rc.1: 1 crate(s): 0 published, 1 not on crates.io" - -- --version "$V-rc.1" --order "$tmp/o1" --out "$tmp/out/rc.tsv"
    row order_is_required 3 "caller error: --order '' is not a readable file" - -- --version "$V" --out "$tmp/out/o.tsv"
    row out_is_required 3 "caller error: --out is required" - -- --version "$V" --order "$tmp/o3"
    row an_option_needs_a_value 3 "caller error: --out needs a value" - -- --version "$V" --order "$tmp/o3" --out
    row an_unknown_option_is_a_caller_error 3 "caller error: unknown option --retries" - -- --retries 3 --version "$V" --order "$tmp/o3" --out "$tmp/out/u.tsv"
    printf '%s\n' 'aprender/../../x' > "$tmp/oslash"
    row a_slash_in_a_name_is_a_caller_error 3 "caller error: the order file line 1 \"aprender/../../x\" is not a crate name" - -- --version "$V" --order "$tmp/oslash" --out "$tmp/out/s.tsv"
    printf '%s\n' '# publish order' apr-format > "$tmp/oc"
    row an_order_comment_is_a_caller_error 3 "caller error: the order file line 1 \"# publish order\" is not a crate name" - -- --version "$V" --order "$tmp/oc" --out "$tmp/out/c.tsv"
    printf '%s\n' apr-format aprender aprender > "$tmp/od"
    row an_order_duplicate_is_a_caller_error 3 "caller error: the order file lists aprender twice (lines 2 and 3)" - -- --version "$V" --order "$tmp/od" --out "$tmp/out/d.tsv"
    : > "$tmp/oe"
    row an_empty_order_is_a_caller_error 3 "caller error: the order file lists no crate" - -- --version "$V" --order "$tmp/oe" --out "$tmp/out/e.tsv"
    printf 'a%064d\n' 0 > "$tmp/o65"
    row a_65_character_name_is_a_caller_error 3 "is not a crate name" - -- --version "$V" --order "$tmp/o65" --out "$tmp/out/l.tsv"
    printf 'a%063d' 0 > "$tmp/o64"; fx "a$(printf '%063d' 0)" 404 -
    row a_64_character_last_line_without_newline_is_read 0 "1 crate(s): 0 published, 1 not on crates.io" - -- --version "$V" --order "$tmp/o64" --out "$tmp/out/l64.tsv"
    row help_prints_the_header 0 "crates_io_reads.sh --mutants" - -- --help
    printf -- '--- %s/%s rows ---\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ]
}

# ---------------------------------------------------------------- mutants ---
# Each mutant is "name sed-script". It must change this file, still parse, and turn --selftest RED with at least
# one BROKE row. A pattern that no longer matches is reported, never skipped: a mutant that changed nothing proves
# nothing. Every copy runs in one directory beside a copy of the judge, and the unmutated copy must be green there
# first: a kill that the directory caused would prove nothing.
mutants() {
    local tmp pass=0 fail=0 name expr copy o rc
    tmp="$(mktemp -d)"
    case "$tmp" in ?*/tmp.?*) : ;; *) echo "mktemp gave '${tmp:-}', refusing"; return 2 ;; esac
    trap selftest_cleanup RETURN
    cp -- "$(dirname -- "$SCRIPT_PATH")/tag_to_crates.sh" "$tmp/tag_to_crates.sh" || { echo "  BROKE no judge beside this script"; return 2; }
    cp -- "$SCRIPT_PATH" "$tmp/unmutated.sh"
    bash "$tmp/unmutated.sh" --selftest > /dev/null 2>&1 || { echo "  BROKE the unmutated copy is red where the mutants run"; return 2; }
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
ua_dropped                      /^fetch() {$/,/^}$/s/ -H "User-Agent: \$UA"//
timeout_dropped                 /^fetch() {$/,/^}$/s/ --max-time 20//
any_protocol                    /^fetch() {$/,/^}$/s/ --proto =https//
redirects_followed              /^fetch() {$/,/^}$/s/curl -sS /curl -sSL /
no_pause                        /^fetch() {$/,/^}$/s/^        sleep 1$/        :/
read_at_local_time              /^fetch() {$/,/^}$/s/date -u +/date +/
read_at_is_a_constant           /^fetch() {$/,/^}$/s/ra="\$(date -u +%Y-%m-%dT%H:%M:%SZ)"/ra="2026-10-03T16:40:00Z"/
read_at_before_the_request      /^fetch() {$/,/^}$/{/ra="\$(date -u/d;s/^        cx=0$/        cx=0; ra="$(date -u +%Y-%m-%dT%H:%M:%SZ)"/;}
curl_exit_ignored               /^fetch() {$/,/^}$/s/if \[ "\$cx" -ne 0 ]; then/if false; then/
a_404_is_a_failure              /^fetch() {$/,/^}$/s/elif \[ "\$code" = 404 ]; then/elif false; then/
redirect_is_read                /^fetch() {$/,/^}$/s/elif \[ "\$code" = 200 ]; then/elif [ "$code" = 200 ] || [ "$code" = 301 ]; then/
another_crate_allowed           /^created_at() {$/,/^}$/s/elif \$x.crate != \$c then/elif false then/
another_version_allowed         /^created_at() {$/,/^}$/s/elif \$x.num != \$v then/elif false then/
missing_created_at_unnamed      /^created_at() {$/,/^}$/s/elif (\$x.created_at | type) != "string" then/elif false then/
any_offset_is_utc               /^UTC_RE=/s/(Z|[[]+[]]00:00)\$/(Z|[+-][0-9]{2}:[0-9]{2})$/
fraction_kept                   /^fetch() {$/,/^}$/s/"\${BASH_REMATCH\[1]}Z"/"$ca"/
a_failed_read_exits_0           /^fetch() {$/,/^}$/s/\[ "\$nfail" -eq 0 ] || return 2/true/
a_failed_read_is_a_404_row      /^fetch() {$/,/^}$/s/^        nfail=\$((nfail + 1))$/        rows+="$c"$'\\t'"$ver"$'\\t'$'\\t'"$ra"$'\\n'; nfail=$((nfail + 1))/
header_renamed                  /^fetch() {$/,/^}$/s/crate\\tversion\\tcreated_at\\tread_at/crate\\tversion\\tcreated\\tread_at/
overwrite_allowed               /^main() {$/,/^}$/s/\[ ! -e "\$out" ] || caller_error/true || caller_error/
version_unchecked               /^main() {$/,/^}$/s/\[\[ "\$ver" =~ \$VER_RE ]] || caller_error/true || caller_error/
order_names_unchecked           /^fetch() {$/,/^}$/s/iscrate "\$ln" || caller_error/true || caller_error/
name_length_unchecked           /^iscrate() {/s/ && \[ "\${#1}" -le 64 ]//
order_duplicates_allowed        /^fetch() {$/,/^}$/s/\[ -z "\${seen\[\$ln]:-}" ] || caller_error/true || caller_error/
empty_order_allowed             /^fetch() {$/,/^}$/s/\[ "\$m" -gt 0 ] || caller_error/true || caller_error/
last_line_without_newline_lost  /^fetch() {$/,/^}$/s/while IFS= read -r ln || \[ -n "\$ln" ]; do/while IFS= read -r ln; do/
body_keeps_the_code             /^fetch() {$/,/^}$/s/body="\${resp%\$'\\n'\*}"/body="$resp"/
empty_answer_unnamed            /^created_at() {$/,/^}$/s/"\${o:-is not a JSON object}"/"$o"/
partial_json_accepted           /^created_at() {$/,/^}$/s/ || o=""$//
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
