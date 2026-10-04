#!/usr/bin/env bash
# weekly_promote.sh — the weekly cadence: say which night's candidate a release would promote. It never tags (#4703).
#
# The rule (operator): one job picks C once a night and publishes it as refs/heads/nightly/<night>; every producer
# measures that C; the nightly train prints its line for that C; a release promotes that C. On one fixed weekday this
# printer reads the train's ledger for the last seven nights and prints ONE line on stdout:
#     PROMOTE H=<sha>          the newest night whose train line is RELEASABLE for that night's pick
#     SKIP: none RELEASABLE    no night in the window was; "(not_measured: ...)" is appended when a night could not be read
# A night counts only when all three agree: its train line is "RELEASABLE H=<sha>", the bundle's "# C" is that sha, and
# refs/heads/nightly/<night> resolves to it. A RELEASABLE line for another commit (main's head after a merge) is not
# the night's C and never counts. Notes on every night go to stderr. It prints; it never tags, pushes or releases.
#
# LEDGER (the nightly train's --out DIR): DIR/<D>/line (the train's line) and DIR/<D>/bundle.tsv, whose rows
# "# C<TAB><sha>" and "# as of<TAB><ISO time>" give the commit and the run time. A run's night is the UTC date of
# (run time - 12 h), the same rule as the pick, so a train at 04:45Z judges the night picked the evening before.
# When several runs fall on one night, the latest run time decides it.
#
# Usage:
#   weekly_promote.sh --ledger DIR [--as-of ISO] [--repo DIR] [--remote NAME] [--np PATH] [--inbox FILE]
#   weekly_promote.sh --self-test | --mutants
#   --np is the nightly-pick command (default: nightly_pick.sh beside this file); its "resolve" gives the night's C.
#   --inbox appends the line (at most 300 bytes) to FILE and reads it back.
# Exit: 0 PROMOTE; 1 SKIP, every night measured; 2 SKIP with a night not measured; 3 caller error.
# Report-only: nothing reads this line yet. The timer that runs it on the fixed weekday is ticket text, not this file.
set -uo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$HERE/$(basename -- "${BASH_SOURCE[0]}")"
WINDOW=7   # nights read, ending with the night of --as-of; a constant, never an option

caller_error() { printf 'weekly_promote: caller error: %s\n' "$1" >&2; exit 3; }
note() { printf 'WP %s\n' "$*" >&2; }

# night_of ISO -> the night (UTC date of the time minus 12 h); 3 on an unreadable time
night_of() {
    local e
    [ -n "$1" ] || return 3   # date reads an empty string as midnight today
    e="$(date -u -d "$1" +%s 2>/dev/null)" || return 3
    date -u -d "@$((e - 43200))" +%F
}

# judge LEDGER ASOF NP REPO REMOTE -> prints the line; returns 0 PROMOTE, 1 SKIP, 2 SKIP not measured
judge() {
    local led="$1" end k n d t a line h c p rc nm=0 why=""
    local -A RUN=() AT=()
    end="$(night_of "$2")" || { printf 'SKIP: none RELEASABLE (not_measured: --as-of %s is not a time)\n' "$2"; return 3; }
    if [ ! -d "$led" ]; then printf 'SKIP: none RELEASABLE (not_measured: no ledger)\n'; return 2; fi
    for d in "$led"/*/; do
        [ -d "$d" ] || continue
        d="${d%/}"
        a="$(awk -F '\t' '$1 == "# as of" { print $2; exit }' "$d/bundle.tsv" 2>/dev/null)"
        n="$(night_of "$a")" || { note "ledger ${d##*/}: no run time in bundle.tsv, not counted"; continue; }
        t="$(date -u -d "$a" +%s)"
        # the latest run on a night decides it
        p="${AT[$n]:-}"
        if [ -z "$p" ] || [ "$t" -gt "$p" ]; then
            AT["$n"]="$t"
            RUN["$n"]="$d"
        fi
    done
    for k in $(seq 0 $((WINDOW - 1))); do
        n="$(date -u -d "$end -$k day" +%F)"
        d="${RUN[$n]:-}"
        if [ -z "$d" ]; then note "night $n: no train run, not_measured"; nm=$((nm + 1)); continue; fi
        line="$(head -n 1 "$d/line" 2>/dev/null)"
        if [ -z "$line" ]; then note "night $n: no train line, not_measured"; nm=$((nm + 1)); continue; fi
        h="$(printf '%s\n' "$line" | awk 'match($0, /^RELEASABLE H=[0-9a-f]{40}( |$)/) { print substr($0, RSTART + 13, 40) }')"
        if [ -z "$h" ]; then note "night $n: ${line:0:100}"; continue; fi
        c="$(awk -F '\t' '$1 == "# C" { print $2; exit }' "$d/bundle.tsv" 2>/dev/null)"
        if [ "$c" != "$h" ]; then note "night $n: line H=${h:0:10} but bundle C=${c:0:10}, not counted"; continue; fi
        if [ ! -f "$3" ]; then note "night $n: no nightly-pick command at $3, not_measured"; nm=$((nm + 1)); continue; fi
        rc=0; p="$(bash "$3" resolve --repo "$4" --remote "$5" --night "$n" 2>/dev/null)" || rc=$?
        case "$rc" in
            0) ;;
            1) note "night $n: RELEASABLE H=${h:0:10}, but the night has no pick, not counted"; continue ;;
            *) note "night $n: the night's pick could not be read (rc=$rc), not_measured"; nm=$((nm + 1)); continue ;;
        esac
        # a pick that exits 0 but prints no sha was never read: not_measured, never "another commit"
        if ! printf '%s\n' "$p" | grep -Eqx '[0-9a-f]{40}'; then note "night $n: the night's pick read as \"${p:0:20}\", not_measured"; nm=$((nm + 1)); continue; fi
        if [ "$p" != "$h" ]; then note "night $n: RELEASABLE H=${h:0:10} is not the night's pick ${p:0:10}, not counted"; continue; fi
        note "night $n: RELEASABLE H=${h:0:10}, the night's pick"
        printf 'PROMOTE H=%s\n' "$h"
        return 0
    done
    [ "$nm" -eq 0 ] || why=" (not_measured: $nm of $WINDOW nights)"
    printf 'SKIP: none RELEASABLE%s\n' "$why"
    [ "$nm" -eq 0 ] && return 1
    return 2
}

# ------------------------------------------------------------------------------------ the case table ----
CASES=0; FAILED=0
row() { # NAME WANT-RC MUST MUSTNOT -- CMD...
    local name="$1" want="$2" must="$3" mustnot="$4" out rc
    shift 5
    out="$("$@" 2>&1)"; rc=$?
    CASES=$((CASES + 1))
    if [ "$rc" != "$want" ]; then printf 'RED   %-48s rc=%s want %s\n' "$name" "$rc" "$want"; FAILED=$((FAILED + 1)); return; fi
    if [ -n "$must" ] && ! printf '%s\n' "$out" | grep -qF -e "$must"; then printf 'RED   %-48s missing: %s\n' "$name" "$must"; FAILED=$((FAILED + 1)); return; fi
    if [ -n "$mustnot" ] && printf '%s\n' "$out" | grep -qF -e "$mustnot"; then printf 'RED   %-48s forbidden: %s\n' "$name" "$mustnot"; FAILED=$((FAILED + 1)); return; fi
    printf 'ok    %s\n' "$name"
}
sha() { printf '%040x\n' "$1"; }   # a fixture commit id
# run LEDGER D ASOF LINE [C] : one train run in the ledger, as the nightly train writes it
run() {
    mkdir -p "$1/$2" || return 1
    printf '%s\n' "$4" > "$1/$2/line"
    printf 'lane\tkind\tstate\n# C\t%s\n# as of\t%s\n' "${5:-}" "$3" > "$1/$2/bundle.tsv"
}
# a stub nightly-pick: resolve reads picks.tsv beside it (night, sha | NM | EMPTY); a missing night has no pick
stub_np() {
    cat > "$1/np.sh" <<'STUB'
n=''; while [ "$#" -gt 0 ]; do case "$1" in --night) n="$2"; shift 2 ;; *) shift ;; esac; done
v="$(awk -F '\t' -v n="$n" '$1 == n { print $2; exit }' "$(dirname "$0")/picks.tsv")"
case "$v" in '') echo "has no pick" >&2; exit 1 ;; NM) exit 2 ;; EMPTY) exit 0 ;; *) echo "$v" ;; esac
STUB
    : > "$1/picks.tsv"
}
pick() { printf '%s\t%s\n' "$2" "$3" >> "$1/picks.tsv"; }

self_test() {
    local tmp L W ASOF="2026-10-12T06:00:00Z" i s
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/wp-st.XXXXXX")" || caller_error "no temp dir"
    W="$tmp/w"; mkdir -p "$W"; stub_np "$W"
    wp() { bash "$SCRIPT_PATH" --ledger "$1" --as-of "${2:-$ASOF}" --np "$W/np.sh"; }
    # a full week (nights 10-05 .. 10-11, the train at 04:45Z next morning), every night RELEASABLE on its pick
    full() {
        local i n
        for i in 5 6 7 8 9 10 11; do
            n="$(printf '2026-10-%02d' "$i")"
            run "$1" "$(date -u -d "$n +1 day" +%F)" "$(date -u -d "$n +1 day" +%F)T04:45:00Z" \
                "RELEASABLE H=$(sha "$i") [C=$(sha "$i" | cut -c1-10) pin=unpinned]" "$(sha "$i")"
        done
    }
    for i in 5 6 7 8 9 10 11; do pick "$W" "$(printf '2026-10-%02d' "$i")" "$(sha "$i")"; done

    L="$tmp/l1"; full "$L"
    row the_newest_releasable_night_is_promoted 0 "PROMOTE H=$(sha 11)" "SKIP" -- wp "$L"
    row stdout_is_one_line 0 "lines=1" "" -- eval 'wp "$L" 2> /dev/null | awk "END { print \"lines=\" NR }"'

    L="$tmp/l2"; full "$L"; run "$L" 2026-10-12 2026-10-12T04:45:00Z "NOT RELEASABLE: mutants, 123 [C=x pin=unpinned]" "$(sha 11)"
    row a_newer_red_night_promotes_the_older_one 0 "PROMOTE H=$(sha 10)" "PROMOTE H=$(sha 11)" -- wp "$L"

    L="$tmp/l3"; full "$L"
    for i in 06 07 08 09 10 11 12; do run "$L" "2026-10-$i" "2026-10-${i}T04:45:00Z" "NOT RELEASABLE: x, not_measured" "$(sha 1)"; done
    row no_releasable_night_skips 1 "SKIP: none RELEASABLE" "PROMOTE" -- wp "$L"
    row every_night_measured_says_no_not_measured 1 "" "not_measured" -- eval 'wp "$L" 2> /dev/null'

    # a RELEASABLE line for another commit (main's head after a merge) is not the night's C
    L="$tmp/l4"; full "$L"; run "$L" 2026-10-12 2026-10-12T04:45:00Z "RELEASABLE H=$(sha 99) [C=x]" "$(sha 99)"
    row a_line_for_another_commit_is_not_the_night 0 "PROMOTE H=$(sha 10)" "PROMOTE H=$(sha 99)" -- wp "$L"
    row the_other_commit_is_named 0 "is not the night's pick" "" -- wp "$L"

    L="$tmp/l5"; full "$L"; run "$L" 2026-10-12 2026-10-12T04:45:00Z "RELEASABLE H=$(sha 11) [C=x]" "$(sha 98)"
    row a_bundle_on_another_commit_is_not_counted 0 "PROMOTE H=$(sha 10)" "PROMOTE H=$(sha 11)" -- wp "$L"

    # a night with no pick, or whose pick cannot be read, is never promoted
    L="$tmp/l6"; full "$L"; s="$tmp/w6"; mkdir -p "$s"; stub_np "$s"
    for i in 5 6 7 8 9; do pick "$s" "$(printf '2026-10-%02d' "$i")" "$(sha "$i")"; done; pick "$s" 2026-10-11 NM
    row a_night_without_a_pick_is_not_promoted 0 "PROMOTE H=$(sha 9)" "PROMOTE H=$(sha 10)" -- \
        bash "$SCRIPT_PATH" --ledger "$L" --as-of "$ASOF" --np "$s/np.sh"
    row a_night_without_a_pick_is_named 0 "but the night has no pick, not counted" "" -- \
        bash "$SCRIPT_PATH" --ledger "$L" --as-of "$ASOF" --np "$s/np.sh"
    row an_unreadable_pick_is_not_measured 0 "could not be read (rc=2), not_measured" "PROMOTE H=$(sha 11)" -- \
        bash "$SCRIPT_PATH" --ledger "$L" --as-of "$ASOF" --np "$s/np.sh"
    row no_pick_command_is_not_measured 2 "SKIP: none RELEASABLE (not_measured: 7 of 7 nights)" "PROMOTE" -- \
        bash "$SCRIPT_PATH" --ledger "$L" --as-of "$ASOF" --np "$tmp/absent.sh"
    # a pick that exits 0 and prints no sha was never read: the week is not measured, not a measured SKIP
    s="$tmp/w6e"; mkdir -p "$s"; stub_np "$s"
    for i in 5 6 7 8 9 10 11; do pick "$s" "$(printf '2026-10-%02d' "$i")" EMPTY; done
    row an_empty_pick_is_not_measured 2 "SKIP: none RELEASABLE (not_measured: 7 of 7 nights)" "PROMOTE" -- \
        bash "$SCRIPT_PATH" --ledger "$L" --as-of "$ASOF" --np "$s/np.sh"

    row no_ledger_is_not_measured 2 "SKIP: none RELEASABLE (not_measured: no ledger)" "PROMOTE" -- wp "$tmp/absent"
    mkdir -p "$tmp/empty"
    row an_empty_ledger_is_not_measured 2 "(not_measured: 7 of 7 nights)" "PROMOTE" -- wp "$tmp/empty"
    L="$tmp/l7"; run "$L" 2026-10-12 2026-10-12T04:45:00Z "" "$(sha 11)"
    row a_run_without_a_line_is_not_measured 2 "no train line, not_measured" "PROMOTE" -- wp "$L"
    L="$tmp/l8"; mkdir -p "$L/2026-10-12"; printf 'RELEASABLE H=%s\n' "$(sha 11)" > "$L/2026-10-12/line"
    row a_run_without_a_run_time_is_not_counted 2 "no run time in bundle.tsv" "PROMOTE" -- wp "$L"

    # only the seven nights ending with the night of --as-of
    L="$tmp/l9"; run "$L" 2026-10-04 2026-10-04T04:45:00Z "RELEASABLE H=$(sha 3) [C=x]" "$(sha 3)"; pick "$W" 2026-10-03 "$(sha 3)"
    row a_night_before_the_window_is_not_promoted 2 "SKIP: none RELEASABLE" "PROMOTE" -- wp "$L"
    L="$tmp/l10"; full "$L"; run "$L" 2026-10-12x 2026-10-12T13:00:00Z "RELEASABLE H=$(sha 12) [C=x]" "$(sha 12)"; pick "$W" 2026-10-12 "$(sha 12)"
    row a_night_after_as_of_is_not_promoted 0 "PROMOTE H=$(sha 11)" "PROMOTE H=$(sha 12)" -- wp "$L"

    # the latest run on a night decides it, in either order
    L="$tmp/l11"; full "$L"; run "$L" 2026-10-12b 2026-10-12T09:00:00Z "NOT RELEASABLE: x, 1 [C=x]" "$(sha 11)"
    row a_later_red_rerun_overrides_the_night 0 "PROMOTE H=$(sha 10)" "PROMOTE H=$(sha 11)" -- wp "$L"
    L="$tmp/l12"; full "$L"; run "$L" 2026-10-12a 2026-10-12T01:00:00Z "NOT RELEASABLE: x, 1 [C=x]" "$(sha 11)"
    row an_earlier_red_run_is_overridden 0 "PROMOTE H=$(sha 11)" "" -- wp "$L"

    # the line must start with RELEASABLE H=<40 hex>
    L="$tmp/l13"; full "$L"; run "$L" 2026-10-12 2026-10-12T04:45:00Z "NOT RELEASABLE: x RELEASABLE H=$(sha 11)" "$(sha 11)"
    row a_not_releasable_line_never_reads_as_releasable 0 "PROMOTE H=$(sha 10)" "PROMOTE H=$(sha 11)" -- wp "$L"
    L="$tmp/l14"; full "$L"; run "$L" 2026-10-12 2026-10-12T04:45:00Z "RELEASABLE H=$(sha 11 | cut -c1-12)" "$(sha 11 | cut -c1-12)"
    row a_short_sha_is_not_releasable 0 "PROMOTE H=$(sha 10)" "" -- wp "$L"

    # it never tags, pushes or releases: git and gh on PATH record every call
    mkdir -p "$tmp/bin"
    printf '#!/bin/sh\necho "git $*" >> "%s/calls"\n' "$tmp" > "$tmp/bin/git"
    printf '#!/bin/sh\necho "gh $*" >> "%s/calls"\n' "$tmp" > "$tmp/bin/gh"
    chmod +x "$tmp/bin/git" "$tmp/bin/gh"; : > "$tmp/calls"
    row it_never_tags_pushes_or_releases 0 "calls=0" "" -- \
        eval 'PATH="$tmp/bin:$PATH" bash "$SCRIPT_PATH" --ledger "$tmp/l1" --as-of "$ASOF" --np "$W/np.sh" > /dev/null 2>&1; grep -cE "tag|push|release" "$tmp/calls" | sed "s/^/calls=/"; true'

    row the_line_reaches_the_inbox 0 "PROMOTE H=$(sha 11)" "" -- \
        eval 'bash "$SCRIPT_PATH" --ledger "$tmp/l1" --as-of "$ASOF" --np "$W/np.sh" --inbox "$tmp/ib" > /dev/null 2>&1; cat "$tmp/ib"'
    row a_bad_as_of_is_a_caller_error 3 "" "PROMOTE" -- wp "$tmp/l1" "not-a-time"
    row an_unknown_option_is_a_caller_error 3 "unknown option" "" -- bash "$SCRIPT_PATH" --ledger "$tmp/l1" --tag v1

    rm -rf -- "${tmp:?}"
    if [ "$FAILED" -eq 0 ]; then printf 'SELF-TEST-GREEN %s/%s case rows green\n' "$CASES" "$CASES"; return 0; fi
    printf 'SELF-TEST-RED %s of %s case rows red\n' "$FAILED" "$CASES"; return 1
}

# ------------------------------------------------------------------------------------ planted mutants ----
MUTANTS='m01_any_line_is_releasable	s|/\^RELEASABLE H=\[0-9a-f\]{40}( \|\$)/|/RELEASABLE H=[0-9a-f]{40}( \|$)/|
m02_bundle_commit_ignored	s/if \[ "\$c" != "\$h" \]; then/if false; then/
m03_night_pick_ignored	s/if \[ "\$p" != "\$h" \]; then/if false; then/
m04_no_pick_promoted	s/1) note "night \$n: RELEASABLE H=\${h:0:10}, but the night has no pick, not counted"; continue ;;/1) ;;/
m05_unreadable_pick_counted	s/\*) note "night \$n: the night.s pick could not be read (rc=\$rc), not_measured"; nm=\$((nm + 1)); continue ;;/*) ;;/
m06_oldest_first	s/n="\$(date -u -d "\$end -\$k day" +%F)"/n="$(date -u -d "$end -$((WINDOW - 1 - k)) day" +%F)"/
m07_first_run_decides	s/\[ "\$t" -gt "\$p" \]/[ "$t" -lt "$p" ]/
m08_night_is_the_run_date	s/date -u -d "@\$((e - 43200))" +%F/date -u -d "@$e" +%F/
m09_not_measured_reads_as_measured	s/\[ "\$nm" -eq 0 \] \&\& return 1/return 1/
m10_window_widened	s/^WINDOW=7 /WINDOW=8 /
m11_no_ledger_passes	s/if \[ ! -d "\$led" \]; then printf .SKIP: none RELEASABLE (not_measured: no ledger)\\n.; return 2; fi/:/
m12_it_tags	s/printf .PROMOTE H=%s\\n. "\$h"/git tag "v-$h" 2> \/dev\/null; printf "PROMOTE H=%s\\n" "$h"/
m13_missing_line_not_measured_dropped	s/if \[ -z "\$line" \]; then note "night \$n: no train line, not_measured"; nm=\$((nm + 1)); continue; fi/:/
m14_empty_run_time_read_as_today	s/\[ -n "\$1" \] || return 3   # date reads/: # date reads/
m15_empty_pick_counted_as_measured	s/^        if ! printf .%s\\n. "\$p" | grep -Eqx .\[0-9a-f\]{40}.; then .*$/        :/'

mutants() {
    local tmp name expr killed=0 total=0 errors=0 out
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/wp-mu.XXXXXX")" || caller_error "no temp dir"
    while IFS="$(printf '\t')" read -r name expr; do
        [ -n "$name" ] || continue
        total=$((total + 1))
        sed -e "$expr" "$SCRIPT_PATH" > "$tmp/weekly_promote.sh"
        if cmp -s "$SCRIPT_PATH" "$tmp/weekly_promote.sh"; then
            printf 'ERROR %-40s the patch did not apply (an error, never a survivor)\n' "$name"; errors=$((errors + 1)); continue
        fi
        if out="$(bash "$tmp/weekly_promote.sh" --self-test 2>&1)"; then
            printf 'SURVIVED %s\n' "$name"
        else
            killed=$((killed + 1))
            printf 'killed   %-40s %s\n' "$name" "$(printf '%s\n' "$out" | grep -c -e '^RED ')"
        fi
    done <<< "$MUTANTS"
    rm -rf -- "${tmp:?}"
    printf 'MUTANTS killed=%s total=%s errors=%s\n' "$killed" "$total" "$errors"
    [ "$killed" -eq "$total" ] && [ "$errors" -eq 0 ]
}

# ------------------------------------------------------------------------------------ main ----
case "${1:-}" in
    --self-test) self_test; exit $? ;;
    --mutants) mutants; exit $? ;;
    -h | --help) sed -n '2,25p' "$SCRIPT_PATH"; exit 0 ;;
esac
ledger=''; asof=''; repo=.; remote=origin; np="$HERE/nightly_pick.sh"; inbox=''
while [ "$#" -gt 0 ]; do
    case "$1" in
        --ledger) ledger="${2:-}"; shift 2 || caller_error "--ledger DIR" ;;
        --as-of) asof="${2:-}"; shift 2 || caller_error "--as-of ISO" ;;
        --repo) repo="${2:-}"; shift 2 || caller_error "--repo DIR" ;;
        --remote) remote="${2:-}"; shift 2 || caller_error "--remote NAME" ;;
        --np) np="${2:-}"; shift 2 || caller_error "--np PATH" ;;
        --inbox) inbox="${2:-}"; shift 2 || caller_error "--inbox FILE" ;;
        *) caller_error "unknown option '$1'" ;;
    esac
done
[ -n "$ledger" ] || caller_error "--ledger DIR"
[ -n "$asof" ] || asof="$(date -u +%FT%TZ)"
rc=0; line="$(judge "$ledger" "$asof" "$np" "$repo" "$remote")" || rc=$?
printf '%s\n' "$line"
[ "$rc" -ne 3 ] || exit 3
if [ -n "$inbox" ]; then
    printf '%s\n' "${line:0:299}" >> "$inbox" || exit 3
    [ "$(tail -n 1 "$inbox")" = "${line:0:299}" ] || { printf 'weekly_promote: the inbox read-back differs\n' >&2; exit 3; }
fi
exit "$rc"
