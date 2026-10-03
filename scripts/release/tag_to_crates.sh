#!/usr/bin/env bash
# tag_to_crates.sh — from the release tag to the last crate of its cascade on crates.io, judged against the 2 h bar.
#   The receipt for E5 (BLD-001 row 5)
#
# THE RULE. Operator ruling C277 item 6 (2026-10-03) lists the bars 0.70.2 ships on; one is "tag to crates.io for
#   0.70.2 itself is under 2 hours". BLD-001 row 5 asks for a timer receipt that runs from tag creation to the last
#   crate's created_at, with a case table: 2 h 01 min is red, 1 h 59 min is green, a missing crates.io timestamp is
#   not_measured. This script is that timer. It judges reads a fetcher already took; it makes no network call.
#
# THE CRATES. Every line of scripts/release/publish-order.txt at the tag (the file exists from v0.69.0), each at
#   --version. The clock stops at the newest created_at among them, not at the root crate `aprender` alone: the
#   cascade publishes two crates after it (v0.69.0 to v0.70.0). The three facade crates that
#   scripts/release/publish_strict.sh appends to the order (provable-contracts, provable-contracts-macros,
#   provable-contracts-cli) are not timed: they carry their own version (0.4.0 in crates/facades/Cargo.toml at every
#   tag from v0.66.0 to v0.70.0), never --version, so no read at --version can time them.
#
# A READ. One row per crate: what crates.io answered for <crate> at --version, and when. The fetcher GETs
#   https://crates.io/api/v1/crates/<crate>/<version>; version.created_at is the publish time, and a 404 means not
#   published, written as an empty created_at. read_at is the UTC second the answer came. The fetcher floors
#   fractional seconds (16:40:00.123456Z -> 16:40:00Z); that never moves a crate across the tag or the bar, both
#   whole seconds. A failed read (a timeout, a 5xx, a 429) is NOT a row: "not published" and "not known" are
#   different facts.
#
# THE CLOCK starts at --tag-time, the annotated tag's tagger date (a lightweight tag has none, so it gives no start):
#     TZ=UTC git for-each-ref --format='%(taggerdate:format-local:%Y-%m-%dT%H:%M:%SZ)' refs/tags/v<version>
#   and stops at the newest created_at of the crates. A crate not on crates.io at its read stops it no earlier than
#   that read: its time is a lower bound ("at least"). Nothing after --as-of counts: a created_at after it reads as
#   not yet published, and a later read counts only up to it, so a replay over newer reads gives the same answer.
#
# VERDICT, per crate, in the order of the order file:
#     not_measured  no row reads it: a crate nobody looked at is never green;
#     not_measured  published before the tag: the clock has no valid start;
#     red           published LIMIT (2 h) or more after the tag, or still not on crates.io at a read (or at --as-of)
#                   LIMIT or more after it: a lower bound past the bar is already past it;
#     not_measured  not on crates.io at a read under LIMIT after the tag, or at a read before the tag: a later read
#                   decides;
#     ok            published under LIMIT after the tag.
#   "Under 2 hours" is strict: exactly 2h00m00s is red. LIMIT IS A CONSTANT, not an option and not an environment
#   value: a threshold a caller can raise is theater. Changing it is a reviewed commit, and --selftest pins it.
#
# E5. One more line. It is met when every crate is ok, and names the last crate and the time it took. It is "not
#   met" when a crate is red: red wins over not_measured. Otherwise it is not_measured and counts the reasons. A
#   context line follows: what the root crate `aprender` alone took, the one-crate method of the earlier baseline,
#   printed for comparison and never the verdict.
#
# READS. A TSV file. Its first line is exactly these four tab-separated names; then one crate per line:
#     crate  version  created_at (YYYY-MM-DDTHH:MM:SSZ, UTC; empty when not published)  read_at (the same form)
#   A row the judge cannot read (a short line, an unknown or repeated crate, another version, a bad time, a
#   created_at after its read_at) makes the whole judgment not_measured: a row is never skipped.
#
# ORDER. A file with one crate name per line and nothing else, as the cascade's own order file is:
#     git show v<version>:scripts/release/publish-order.txt
#
# EXIT  0 E5 met · 1 a crate is red (red wins) · 2 not_measured (a stop: unknown is not a pass) · 3 caller error
#   (a wiring defect)
#
# USAGE
#   tag_to_crates.sh --version X.Y.Z --tag-time T --as-of T --order FILE --reads FILE   (T is YYYY-MM-DDTHH:MM:SSZ)
#   tag_to_crates.sh --selftest    the case table
#   tag_to_crates.sh --mutants     each planted mutant of judge(), main() or LIMIT must turn the case table RED
set -uo pipefail

PROG="${0##*/}"
SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/${BASH_SOURCE[0]##*/}"
# C277 item 6: "tag to crates.io for 0.70.2 itself is under 2 hours". A constant: never an option, never taken from
# the environment.
LIMIT=7200

caller_error() { printf 'FAIL  TAG2CRATES %s: caller error: %s\n' "$PROG" "$*"; exit 3; }

# judge version tag-time as-of order reads -> one line per crate, the E5 line and one context line on stdout;
# exits 0/1/2/3 as above
judge() {
    awk -v ver="$1" -v tagt="$2" -v asof="$3" -v limit="$LIMIT" -v prog="$PROG" '
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
    function iscrate(s) { return (s ~ /^[A-Za-z][A-Za-z0-9_-]*$/ && length(s) <= 64) }    # a crates.io name
    function why(list, n, what) { if (n == 0) return list; return (list == "") ? n " " what : list ", " n " " what }
    function stop(x, d) { if (!hasclk || x > clk) { clk = x; clkd = d; hasclk = 1 } }    # the clock: the newest stop
    BEGIN {
        FS = "\t"
        want = "crate\tversion\tcreated_at\tread_at"
        order = ARGV[1]; ARGV[1] = ""
        if (ver !~ /^[0-9]+[.][0-9]+[.][0-9]+(-[0-9A-Za-z.-]+)?$/) { err = "--version \"" ver "\" is not MAJOR.MINOR.PATCH"; exit }
        if (!istime(tagt)) { err = "--tag-time \"" tagt "\" is not a real UTC time written YYYY-MM-DDTHH:MM:SSZ"; exit }
        if (!istime(asof)) { err = "--as-of \"" asof "\" is not a real UTC time written YYYY-MM-DDTHH:MM:SSZ"; exit }
        tag = secs(tagt); now = secs(asof)
        if (now < tag) { err = "--as-of " asof " is earlier than --tag-time " tagt ": nothing is published before the clock starts"; exit }
        while ((rc = (getline ln < order)) > 0) {
            if (!iscrate(ln)) { err = "the order file line " (m + 1) " \"" ln "\" is not a crate name"; exit }
            if (ln in pos) { err = "the order file lists " ln " twice (lines " pos[ln] " and " (m + 1) ")"; exit }
            m++; pos[ln] = m; crate[m] = ln
        }
        if (rc < 0) { err = "the order file cannot be read"; exit }
        close(order)
        if (m == 0) { err = "the order file lists no crate"; exit }
    }
    NR == 1 {
        if ($0 != want) { err = "the reads header is not crate, version, created_at and read_at, tab-separated"; exit }
        next
    }
    {
        c = $1; v = $2; ca = $3; ra = $4
        if (NF != 4) { bad = "line " NR " has " NF " fields, not 4"; exit }
        if (!iscrate(c)) { bad = "line " NR " crate \"" c "\" is not a crate name"; exit }
        if (!(c in pos)) { bad = "line " NR " crate \"" c "\" is not in the order file"; exit }
        if (c in LN) { bad = "line " NR " is a second row for " c " (first at line " LN[c] ")"; exit }
        if (v != ver) { bad = "line " NR " version \"" v "\" is not --version " ver; exit }
        if (ca != "" && !istime(ca)) { bad = "line " NR " created_at \"" ca "\" is not a real UTC time written YYYY-MM-DDTHH:MM:SSZ"; exit }
        if (!istime(ra)) { bad = "line " NR " read_at \"" ra "\" is not a real UTC time written YYYY-MM-DDTHH:MM:SSZ"; exit }
        if (ca != "" && secs(ca) > secs(ra)) { bad = "line " NR " created_at " ca " is after its read_at " ra; exit }
        CA[c] = ca; RA[c] = ra; LN[c] = NR; rows++
    }
    END {
        if (err != "") { printf "FAIL  TAG2CRATES %s: caller error: %s\n", prog, err; exit 3 }
        if (NR == 0) { print "FAIL  TAG2CRATES not_measured: the reads file is empty"; exit 2 }
        if (bad != "") { print "FAIL  TAG2CRATES not_measured: unreadable reads, " bad; exit 2 }
        for (i = 1; i <= m; i++) {
            c = crate[i]
            if (!(c in LN)) {
                printf "FAIL  TAG2CRATES %s not_measured: no read of it in the reads file\n", c
                nnm++; k1++; continue
            }
            ca = CA[c]; ra = RA[c]
            if (ca != "" && secs(ca) <= now) {
                C = secs(ca)
                if (C < tag) {
                    printf "FAIL  TAG2CRATES %s not_measured: published at %s, %s before the tag at %s: the clock has no valid start\n", c, ca, dur(tag - C), tagt
                    nnm++; k3++; continue
                }
                el = C - tag; stop(el, c " published at " ca); np++
                if (c == "aprender") { fac = el; hasfac = 1 }
                if (el >= limit) {
                    printf "FAIL  TAG2CRATES %s red: published %s after the tag, at %s; the bar is under %s\n", c, dur(el), ca, dur(limit)
                    nred++
                } else {
                    printf "ok    TAG2CRATES %s ok: published %s after the tag, at %s\n", c, dur(el), ca
                }
                continue
            }
            R = secs(ra)
            if (R < tag) {
                printf "FAIL  TAG2CRATES %s not_measured: read at %s, before the tag at %s: it says nothing about this release\n", c, ra, tagt
                nnm++; k4++; continue
            }
            e = (R < now) ? R : now
            et = (R < now) ? ra : asof
            stop(e - tag, c " not on crates.io at " et)
            if (e - tag >= limit) {
                printf "FAIL  TAG2CRATES %s red: not on crates.io at %s, %s after the tag: it takes at least that long; the bar is under %s\n", c, et, dur(e - tag), dur(limit)
                nred++
            } else {
                printf "FAIL  TAG2CRATES %s not_measured: not on crates.io at %s, %s after the tag, still under %s: a later read decides\n", c, et, dur(e - tag), dur(limit)
                nnm++; k2++
            }
        }
        exact = (np == m)
        lb = exact ? "" : "at least "
        if (nred > 0) {
            printf "FAIL  TAG2CRATES E5 not met: tag to crates.io for %s took %s%s (%s); not under %s, %d crate(s) at or past it\n", ver, lb, dur(clk), clkd, dur(limit), nred
        } else if (nnm > 0) {
            r = why("", k1, "not read")
            r = why(r, k2, "not yet on crates.io at its read")
            r = why(r, k3, "published before the tag")
            r = why(r, k4, "read before the tag")
            sofar = hasclk ? ("; so far at least " dur(clk)) : ""
            printf "FAIL  TAG2CRATES E5 not_measured: %d of %d crate(s) for %s not measured: %s%s\n", nnm, m, ver, r, sofar
        } else {
            printf "ok    TAG2CRATES E5 met: tag to crates.io for %s took %s (%s, the last of %d crate(s)); under %s\n", ver, dur(clk), clkd, m, dur(limit)
        }
        if (!("aprender" in pos)) fa = "is not in the order file"
        else if (hasfac) fa = "alone took " dur(fac)
        else fa = "is not timed here"
        printf "      TAG2CRATES context: %d read row(s) for %d crate(s) of the order file; tag %s, as of %s; the root crate aprender %s (the one-crate method of the earlier baseline, never the verdict)\n", rows, m, tagt, asof, fa
        if (nred > 0) exit 1
        if (nnm > 0) exit 2
        exit 0
    }' "$4" "$5"
}

main() {
    local ver="" tagt="" asof="" order="" reads=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --version|--tag-time|--as-of|--order|--reads)
                [ $# -ge 2 ] || caller_error "$1 needs a value"
                case "$1" in
                    --version) ver="$2" ;;
                    --tag-time) tagt="$2" ;;
                    --as-of) asof="$2" ;;
                    --order) order="$2" ;;
                    *) reads="$2" ;;
                esac
                shift 2 ;;
            *) caller_error "unknown option $1 (usage: $PROG --version X.Y.Z --tag-time T --as-of T --order FILE --reads FILE)" ;;
        esac
    done
    [ -n "$ver" ] || caller_error "--version is required: the crates are judged at one version"
    [ -n "$tagt" ] || caller_error "--tag-time is required: the clock starts at the tag, never at a guess"
    [ -n "$asof" ] || caller_error "--as-of is required: a judgment is made for a named instant, never an implicit now"
    { [ -n "$order" ] && [ -f "$order" ] && [ -r "$order" ]; } || caller_error "--order '$order' is not a readable file"
    { [ -n "$reads" ] && [ -f "$reads" ] && [ -r "$reads" ]; } || caller_error "--reads '$reads' is not a readable file"
    # awk reads an operand written name=value as an assignment and "-" as stdin: a relative path gets a "./".
    case "$order" in /*) : ;; *) order="./$order" ;; esac
    case "$reads" in /*) : ;; *) reads="./$reads" ;; esac
    judge "$ver" "$tagt" "$asof" "$order" "$reads"
}

# --------------------------------------------------------------- selftest ---
selftest_cleanup() { case "${tmp:-}" in ''|/) return 0 ;; *) rm -rf -- "${tmp:?}" ;; esac; }
selftest() {
    local tmp L64 L65 pass=0 fail=0 V=0.70.2 D=2026-10-03 T=2026-10-03T16:00:00Z A=2026-10-03T19:00:00Z R=18:30:00 NL=$'\n'
    tmp="$(mktemp -d)"
    case "$tmp" in ?*/tmp.?*) : ;; *) echo "mktemp gave '${tmp:-}', refusing"; return 2 ;; esac
    # SEC011: validate before rm -rf. $tmp is this function's own mktemp -d (checked above).
    trap selftest_cleanup RETURN
    at() { case "$1" in -) : ;; [0-2][0-9]:[0-5][0-9]:[0-5][0-9]) printf '%sT%sZ' "$D" "$1" ;; *) printf '%s' "$1" ;; esac; }
    fx() { # name, then rows "crate created_at read_at [version]": - is empty, HH:MM:SS is that time on $D, else as is
        local f="$tmp/$1" r c ca ra v; shift
        printf 'crate\tversion\tcreated_at\tread_at\n' > "$f"
        for r in "$@"; do
            read -r c ca ra v <<< "$r"
            printf '%s\t%s\t%s\t%s\n' "$c" "${v:-$V}" "$(at "$ca")" "$(at "$ra")" >> "$f"
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
    # j name expect-rc needle reads [as-of [order [tag-time]]]
    j() { row "$1" "$2" "$3" -- --version "$V" --tag-time "${7:-$T}" --as-of "${5:-$A}" --order "$tmp/${6:-o3}" --reads "$tmp/$4"; }

    printf '%s\n' apr-format aprender aprender-tsp > "$tmp/o3"
    printf '%s\n' aprender > "$tmp/o1"
    printf '%s\n' apr-format aprender-tsp > "$tmp/o2"
    printf '%s\n' apr-format '' aprender > "$tmp/ob"
    printf '%s\n' '# publish order' apr-format > "$tmp/oc"
    printf '%s\n' apr-format aprender aprender > "$tmp/od"
    : > "$tmp/oe"
    L64="a$(printf '%063d' 0)"; L65="a$(printf '%064d' 0)"    # crates.io names are at most 64 characters
    printf '%s\n' "$L64" > "$tmp/o64"

    fx g159 "apr-format 16:20:00 $R" "aprender 16:40:00 $R" "aprender-tsp 17:59:00 $R"
    j green_1h59m_is_met                               0 "ok    TAG2CRATES E5 met: tag to crates.io for $V took 1h59m00s (aprender-tsp published at ${D}T17:59:00Z, the last of 3 crate(s)); under 2h00m00s" g159
    j the_root_crate_alone_is_context_only             0 "      TAG2CRATES context: 3 read row(s) for 3 crate(s) of the order file; tag $T, as of $A; the root crate aprender alone took 0h40m00s (the one-crate method of the earlier baseline, never the verdict)" g159
    fx g159s "aprender-tsp 17:59:00 $R" "apr-format 16:20:00 $R" "aprender 16:40:00 $R"
    j crate_lines_follow_the_order_file                0 "ok    TAG2CRATES apr-format ok: published 0h20m00s after the tag, at ${D}T16:20:00Z${NL}ok    TAG2CRATES aprender ok: published 0h40m00s after the tag, at ${D}T16:40:00Z${NL}ok    TAG2CRATES aprender-tsp ok: published 1h59m00s after the tag, at ${D}T17:59:00Z" g159s
    fx r201 "apr-format 16:20:00 $R" "aprender 16:40:00 $R" "aprender-tsp 18:01:00 $R"
    j red_2h01m_is_red                                 1 "FAIL  TAG2CRATES aprender-tsp red: published 2h01m00s after the tag, at ${D}T18:01:00Z; the bar is under 2h00m00s" r201
    j red_2h01m_fails_e5                               1 "FAIL  TAG2CRATES E5 not met: tag to crates.io for $V took 2h01m00s (aprender-tsp published at ${D}T18:01:00Z); not under 2h00m00s, 1 crate(s) at or past it" r201
    j the_root_crate_alone_would_pass_a_red_e5         1 "the root crate aprender alone took 0h40m00s" r201
    fx exact2h "apr-format 16:20:00 $R" "aprender 16:40:00 $R" "aprender-tsp 18:00:00 $R"
    j exactly_2h_is_red                                1 "aprender-tsp red: published 2h00m00s after the tag" exact2h
    fx under1s "apr-format 16:20:00 $R" "aprender 16:40:00 $R" "aprender-tsp 17:59:59 $R"
    j one_second_under_2h_is_met                       0 "E5 met: tag to crates.io for $V took 1h59m59s" under1s
    fx miss_late "apr-format 16:20:00 $R" "aprender 16:40:00 $R" "aprender-tsp - $R"
    j missing_past_2h_is_red                           1 "FAIL  TAG2CRATES aprender-tsp red: not on crates.io at ${D}T18:30:00Z, 2h30m00s after the tag: it takes at least that long; the bar is under 2h00m00s" miss_late
    j missing_past_2h_e5_is_a_lower_bound              1 "E5 not met: tag to crates.io for $V took at least 2h30m00s (aprender-tsp not on crates.io at ${D}T18:30:00Z)" miss_late
    fx miss_2h "apr-format 16:20:00 $R" "aprender 16:40:00 $R" "aprender-tsp - 18:00:00"
    j missing_at_exactly_2h_is_red                     1 "aprender-tsp red: not on crates.io at ${D}T18:00:00Z, 2h00m00s after the tag" miss_2h
    fx miss_early "apr-format 16:20:00 $R" "aprender 16:40:00 $R" "aprender-tsp - 17:50:00"
    j missing_under_2h_is_not_measured                 2 "FAIL  TAG2CRATES aprender-tsp not_measured: not on crates.io at ${D}T17:50:00Z, 1h50m00s after the tag, still under 2h00m00s: a later read decides" miss_early
    j e5_counts_reasons_and_the_clock_so_far           2 "FAIL  TAG2CRATES E5 not_measured: 1 of 3 crate(s) for $V not measured: 1 not yet on crates.io at its read; so far at least 1h50m00s" miss_early
    j a_read_after_as_of_counts_only_to_as_of          2 "aprender-tsp not_measured: not on crates.io at ${D}T17:50:00Z, 1h50m00s after the tag" miss_late "${D}T17:50:00Z"
    j a_created_at_after_as_of_is_not_yet_published    2 "aprender-tsp not_measured: not on crates.io at ${D}T17:59:30Z, 1h59m30s after the tag" r201 "${D}T17:59:30Z"
    j as_of_past_2h_without_the_crate_is_red           1 "aprender-tsp red: not on crates.io at ${D}T18:00:30Z, 2h00m30s after the tag: it takes at least that long" r201 "${D}T18:00:30Z"
    fx pretag "apr-format 15:59:00 $R" "aprender 16:40:00 $R" "aprender-tsp 17:30:00 $R"
    j published_before_the_tag_is_not_measured         2 "FAIL  TAG2CRATES apr-format not_measured: published at ${D}T15:59:00Z, 0h01m00s before the tag at $T: the clock has no valid start" pretag
    j e5_names_a_pre_tag_crate                         2 "1 of 3 crate(s) for $V not measured: 1 published before the tag; so far at least 1h30m00s" pretag
    fx pretag_red "apr-format 15:59:00 $R" "aprender 16:40:00 $R" "aprender-tsp 18:01:00 $R"
    j red_wins_over_not_measured                       1 "E5 not met: tag to crates.io for $V took at least 2h01m00s (aprender-tsp published at ${D}T18:01:00Z)" pretag_red
    fx at_tag "apr-format 16:00:00 $R" "aprender 16:40:00 $R" "aprender-tsp 17:30:00 $R"
    j published_at_the_tag_second_is_ok                0 "apr-format ok: published 0h00m00s after the tag" at_tag
    fx noread "apr-format 16:20:00 $R" "aprender 16:40:00 $R"
    j a_crate_with_no_read_is_not_measured             2 "FAIL  TAG2CRATES aprender-tsp not_measured: no read of it in the reads file" noread
    j no_read_is_never_red_however_late                2 "1 of 3 crate(s) for $V not measured: 1 not read; so far at least 0h40m00s" noread 2026-10-04T16:00:00Z
    fx readpre "apr-format 16:20:00 $R" "aprender 16:40:00 $R" "aprender-tsp - 15:30:00"
    j a_read_before_the_tag_says_nothing               2 "FAIL  TAG2CRATES aprender-tsp not_measured: read at ${D}T15:30:00Z, before the tag at $T: it says nothing about this release" readpre
    fx apr_missing "apr-format 16:20:00 $R" "aprender - 17:50:00" "aprender-tsp 17:30:00 $R"
    j the_root_crate_missing_is_not_timed              2 "the root crate aprender is not timed here" apr_missing
    fx g2 "apr-format 16:20:00 $R" "aprender-tsp 17:59:00 $R"
    j the_root_crate_not_in_the_order_file             0 "the root crate aprender is not in the order file" g2 "$A" o2
    fx header_only
    j only_a_header_reads_nothing                      2 "E5 not_measured: 3 of 3 crate(s) for $V not measured: 3 not read${NL}      TAG2CRATES context: 0 read row(s)" header_only
    : > "$tmp/empty"
    j an_empty_reads_file_is_not_measured              2 "FAIL  TAG2CRATES not_measured: the reads file is empty" empty
    printf 'crate\tversion\tcreated_at\n' > "$tmp/hdr3"; tail -n +2 "$tmp/g159" >> "$tmp/hdr3"
    j a_three_column_header_is_a_caller_error          3 "caller error: the reads header is not crate, version, created_at and read_at, tab-separated" hdr3
    fx dup "apr-format 16:20:00 $R" "aprender 16:40:00 $R" "aprender-tsp 17:59:00 $R" "aprender 16:50:00 $R"
    j a_second_row_is_unreadable                       2 "FAIL  TAG2CRATES not_measured: unreadable reads, line 5 is a second row for aprender (first at line 3)" dup
    fx unknown "apr-format 16:20:00 $R" "aprender 16:40:00 $R" "aprender-tsp 17:59:00 $R" "provable-contracts 2026-09-01T10:00:00Z $R 0.4.0"
    j a_facade_crate_row_is_unreadable                 2 "unreadable reads, line 5 crate \"provable-contracts\" is not in the order file" unknown
    fx wrongver "apr-format 16:20:00 $R" "aprender 16:40:00 $R 0.70.1" "aprender-tsp 17:59:00 $R"
    j another_version_is_unreadable                    2 "unreadable reads, line 3 version \"0.70.1\" is not --version $V" wrongver
    fx frac "apr-format 16:20:00 $R" "aprender ${D}T16:40:00.123456Z $R" "aprender-tsp 17:59:00 $R"
    j fractional_seconds_are_unreadable                2 "unreadable reads, line 3 created_at \"${D}T16:40:00.123456Z\" is not a real UTC time written YYYY-MM-DDTHH:MM:SSZ" frac
    fx noreadat "apr-format 16:20:00 $R" "aprender 16:40:00 -" "aprender-tsp 17:59:00 $R"
    j an_empty_read_at_is_unreadable                   2 "unreadable reads, line 3 read_at \"\" is not a real UTC time" noreadat
    fx after "apr-format 16:20:00 $R" "aprender 18:40:00 $R" "aprender-tsp 17:59:00 $R"
    j created_after_its_read_is_unreadable             2 "unreadable reads, line 3 created_at ${D}T18:40:00Z is after its read_at ${D}T18:30:00Z" after
    fx short "apr-format 16:20:00 $R"; printf 'aprender\t%s\t%sT16:40:00Z\n' "$V" "$D" >> "$tmp/short"
    j a_short_line_is_unreadable                       2 "unreadable reads, line 3 has 3 fields, not 4" short
    fx semi "apr;format 16:20:00 $R"
    j a_bad_crate_name_is_unreadable                   2 "unreadable reads, line 2 crate \"apr;format\" is not a crate name" semi
    fx long65 "apr-format 16:20:00 $R" "$L65 16:40:00 $R"
    j a_65_character_crate_name_is_unreadable          2 "unreadable reads, line 3 crate \"$L65\" is not a crate name" long65
    fx long64 "$L64 16:20:00 $R"
    j a_64_character_crate_name_is_a_crate             0 "ok    TAG2CRATES $L64 ok: published 0h20m00s after the tag" long64 "$A" o64
    fx sept31 "apr-format 16:20:00 $R" "aprender 2026-09-31T16:40:00Z $R"
    j september_31_is_unreadable                       2 "unreadable reads, line 3 created_at \"2026-09-31T16:40:00Z\"" sept31
    fx hour24 "apr-format 16:20:00 $R" "aprender 16:40:00 24:00:00"
    j hour_24_is_unreadable                            2 "unreadable reads, line 3 read_at \"${D}T24:00:00Z\"" hour24
    j a_blank_order_line_is_a_caller_error             3 "caller error: the order file line 2 \"\" is not a crate name" g159 "$A" ob
    j an_order_comment_is_a_caller_error               3 "caller error: the order file line 1 \"# publish order\" is not a crate name" g159 "$A" oc
    j an_order_duplicate_is_a_caller_error             3 "caller error: the order file lists aprender twice (lines 2 and 3)" g159 "$A" od
    j an_empty_order_is_a_caller_error                 3 "caller error: the order file lists no crate" g159 "$A" oe
    j as_of_before_the_tag_is_a_caller_error           3 "caller error: --as-of ${D}T15:00:00Z is earlier than --tag-time $T: nothing is published before the clock starts" g159 "${D}T15:00:00Z"
    j a_tag_time_with_an_offset_is_a_caller_error      3 "caller error: --tag-time \"${D}T18:00:00+02:00\" is not a real UTC time written YYYY-MM-DDTHH:MM:SSZ" g159 "$A" o3 "${D}T18:00:00+02:00"
    j an_impossible_as_of_is_a_caller_error            3 "caller error: --as-of \"2026-02-30T12:00:00Z\" is not a real UTC time" g159 2026-02-30T12:00:00Z
    row a_v_prefixed_version_is_a_caller_error         3 "caller error: --version \"v$V\" is not MAJOR.MINOR.PATCH" -- --version "v$V" --tag-time "$T" --as-of "$A" --order "$tmp/o3" --reads "$tmp/g159"
    row version_is_required                            3 "caller error: --version is required" -- --tag-time "$T" --as-of "$A" --order "$tmp/o3" --reads "$tmp/g159"
    row tag_time_is_required                           3 "caller error: --tag-time is required" -- --version "$V" --as-of "$A" --order "$tmp/o3" --reads "$tmp/g159"
    row as_of_is_required                              3 "caller error: --as-of is required: a judgment is made for a named instant, never an implicit now" -- --version "$V" --tag-time "$T" --order "$tmp/o3" --reads "$tmp/g159"
    row order_is_required                              3 "caller error: --order '' is not a readable file" -- --version "$V" --tag-time "$T" --as-of "$A" --reads "$tmp/g159"
    row reads_is_required                              3 "caller error: --reads '' is not a readable file" -- --version "$V" --tag-time "$T" --as-of "$A" --order "$tmp/o3"
    row a_missing_reads_file_is_a_caller_error         3 "caller error: --reads '$tmp/nope' is not a readable file" -- --version "$V" --tag-time "$T" --as-of "$A" --order "$tmp/o3" --reads "$tmp/nope"
    row a_missing_order_file_is_a_caller_error         3 "caller error: --order '$tmp/nope' is not a readable file" -- --version "$V" --tag-time "$T" --as-of "$A" --order "$tmp/nope" --reads "$tmp/g159"
    row an_option_needs_a_value                        3 "caller error: --reads needs a value" -- --version "$V" --tag-time "$T" --as-of "$A" --order "$tmp/o3" --reads
    row limit_is_not_an_option                         3 "caller error: unknown option --limit" -- --limit 10800 --version "$V" --tag-time "$T" --as-of "$A" --order "$tmp/o3" --reads "$tmp/r201"
    row limit_from_the_environment_is_ignored          1 "aprender-tsp red: published 2h01m00s after the tag" LIMIT=10800 -- --version "$V" --tag-time "$T" --as-of "$A" --order "$tmp/o3" --reads "$tmp/r201"
    row help_prints_the_header                         0 "tag_to_crates.sh --mutants" -- --help
    cp -- "$tmp/g159" "$tmp/x=y"
    cd -- "$tmp" || return 2
    row a_relative_path_with_an_equals_sign_is_a_file  0 "E5 met: tag to crates.io for $V took 1h59m00s" -- --version "$V" --tag-time "$T" --as-of "$A" --order o3 --reads x=y
    cd -- "$OLDPWD" || return 2
    fx year "aprender 2027-01-01T01:29:59Z 2027-01-01T02:00:00Z"
    j dates_cross_a_year_end                           0 "aprender ok: published 1h59m59s after the tag" year 2027-01-01T02:00:00Z o1 2026-12-31T23:30:00Z
    fx leap "aprender 2028-03-01T00:30:00Z 2028-03-01T01:00:00Z"
    j a_leap_day_has_24_hours                          1 "aprender red: published 25h30m00s after the tag" leap 2028-03-01T01:00:00Z o1 2028-02-28T23:00:00Z
    fx noleap "aprender 2027-03-01T00:30:00Z 2027-03-01T01:00:00Z"
    j a_february_without_a_leap_day_has_28_days        0 "aprender ok: published 1h30m00s after the tag" noleap 2027-03-01T01:00:00Z o1 2027-02-28T23:00:00Z
    if [ "$LIMIT" = 7200 ]; then
        printf '  ok    %-50s LIMIT=7200\n' committed_limit_is_2h; pass=$((pass + 1))
    else
        printf '  BROKE %-50s LIMIT=%s, the ruling says under 2 hours\n' committed_limit_is_2h "$LIMIT"; fail=$((fail + 1))
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
limit_is_3h                     /^LIMIT=7200/s/^LIMIT=7200/LIMIT=10800/
limit_one_second_late           /^LIMIT=7200/s/^LIMIT=7200/LIMIT=7201/
limit_from_env                  /^LIMIT=7200/s/^LIMIT=7200/LIMIT=${LIMIT:-7200}/
exactly_2h_is_green             /^judge() {$/,/^}$/s/if (el >= limit)/if (el > limit)/
missing_at_2h_is_not_red        /^judge() {$/,/^}$/s/if (e - tag >= limit)/if (e - tag > limit)/
missing_is_never_red            /^judge() {$/,/^}$/s/if (e - tag >= limit)/if (0)/
early_read_aged_to_as_of        /^judge() {$/,/^}$/s/e = (R < now) ? R : now/e = now/
read_after_as_of_not_clipped    /^judge() {$/,/^}$/s/e = (R < now) ? R : now/e = R/
later_created_is_published      /^judge() {$/,/^}$/s/if (ca != "" && secs(ca) <= now)/if (ca != "")/
pre_tag_is_timed                /^judge() {$/,/^}$/s/if (C < tag)/if (0)/
read_before_tag_trusted         /^judge() {$/,/^}$/s/if (R < tag)/if (0)/
no_read_is_ok                   /^judge() {$/,/^}$/s/nnm++; k1++; continue/k1++; continue/
unknown_crate_allowed           /^judge() {$/,/^}$/s/if (!(c in pos))/if (0)/
duplicate_row_allowed           /^judge() {$/,/^}$/s/if (c in LN)/if (0)/
wrong_version_allowed           /^judge() {$/,/^}$/s/if (v != ver)/if (0)/
created_after_read_allowed      /^judge() {$/,/^}$/s/if (ca != "" && secs(ca) > secs(ra))/if (0)/
short_line_allowed              /^judge() {$/,/^}$/s/if (NF != 4)/if (0)/
crate_name_unchecked            /^judge() {$/,/^}$/s/if (!iscrate(c))/if (0)/
header_not_checked              /^judge() {$/,/^}$/s/if (\$0 != want)/if (0)/
empty_reads_trusted             /^judge() {$/,/^}$/s/if (NR == 0)/if (0)/
unreadable_rows_ignored         /^judge() {$/,/^}$/s/if (bad != "")/if (0)/
order_duplicates_allowed        /^judge() {$/,/^}$/s/if (ln in pos)/if (0)/
as_of_before_tag_allowed        /^judge() {$/,/^}$/s/if (now < tag)/if (0)/
version_unchecked               /^judge() {$/,/^}$/s|if (ver !~ |if (0 \&\& ver !~ |
tag_time_unchecked              /^judge() {$/,/^}$/s/if (!istime(tagt))/if (0)/
as_of_unchecked                 /^judge() {$/,/^}$/s/if (!istime(asof))/if (0)/
as_of_optional                  /^main() {$/,/^}$/s/\[ -n "\$asof" \] || caller_error/true || caller_error/
order_line_unchecked            /^judge() {$/,/^}$/s/if (!iscrate(ln))/if (0)/
order_empty_allowed             /^judge() {$/,/^}$/s/if (m == 0)/if (0)/
created_at_unchecked            /^judge() {$/,/^}$/s/if (ca != "" && !istime(ca))/if (0)/
read_at_unchecked               /^judge() {$/,/^}$/s/if (!istime(ra))/if (0)/
crate_name_length_unchecked     /^judge() {$/,/^}$/s/ && length(s) <= 64)/)/
crate_name_max_63               /^judge() {$/,/^}$/s/length(s) <= 64)/length(s) < 64)/
hour_24_allowed                 /^judge() {$/,/^}$/s/substr(s, 12, 2) + 0 <= 23/substr(s, 12, 2) + 0 <= 24/
calendar_unchecked              /^judge() {$/,/^}$/s/return (civil(dayno(s)) == s)/return 1/
e5_not_measured_beats_red       /^judge() {$/,/^}$/s/if (nred > 0) {/if (nred > 0 \&\& nnm == 0) {/
worst_of                        /^judge() {$/,/^}$/s/if (nred > 0) exit 1/if (0) exit 1/
clock_is_the_earliest_crate     /^judge() {$/,/^}$/s/x > clk)/x < clk)/
only_the_root_crate_is_timed    /^judge() {$/,/^}$/s/el = C - tag; stop(el, /el = C - tag; if (c == "aprender") stop(el, /
clock_ignores_missing_crates    /^judge() {$/,/^}$/s/stop(e - tag, /if (0) stop(e - tag, /
lower_bound_shown_as_exact      /^judge() {$/,/^}$/s/lb = exact ? "" : "at least "/lb = ""/
relative_path_unprefixed        /^main() {$/,/^}$/s/reads="\.\/\$reads"/reads="$reads"/
seconds_ignored                 /^judge() {$/,/^}$/s/ + substr(s, 18, 2) }/ + 0 }/
leap_years_ignored              /^judge() {$/,/^}$/s/doe = yoe \* 365 + int(yoe \/ 4) - int(yoe \/ 100) + doy/doe = yoe * 365 + doy/
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
