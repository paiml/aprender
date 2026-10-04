#!/usr/bin/env bash
# known_failure_ratchet.sh -- judge tonight's evidence against the known-failure list (#4689).
#
# THE RULE. A failure in the evidence bundle is KNOWN only when its key (model sha256, leg, check) is on
#   the list with a ticket, and a known failure is still printed RED with that ticket. A failure that
#   passed in the last release's bundle is NEW; any other unlisted failure is UNLISTED. Both are RED.
#   The list is shrink-only: an added or edited row, a listed key that passes here (STALE), a row with
#   no ticket and a key listed twice are RED. A missing or unreadable bundle is NOT_MEASURED, never a
#   pass. The logic is in scripts/lib/known_failure_ratchet.sh; this file is the command line, the case
#   table and the planted mutants. Contract: contracts/known-failure-ratchet-v1.yaml.
#
# THE SEED. scripts/release/known_failures.tsv. The 0.70.1 release had no nightly bundle, so its list
#   is the set fixed by the v0.70.1 release notes and tag: tickets #4661 to #4666, one row per
#   (model, leg) the ticket names. It only shrinks, as each ticket is fixed or re-ruled.
#
# REPORT-ONLY. Nothing reads this verdict yet; it blocks nothing.
#
# USAGE
#   known_failure_ratchet.sh --head DIR [--base DIR|none] [--list FILE] [--list-base FILE|none|REF]
#     --head       tonight's bundle directory (DIR/checks.tsv)
#     --base       the last release's bundle directory, or none (default none)
#     --list       the list (default: the committed scripts/release/known_failures.tsv)
#     --list-base  the list at the base ref: a file, none, or a git ref to read it from (default origin/main)
#   known_failure_ratchet.sh --self-test     the case table (fixtures, no network)
#   known_failure_ratchet.sh --mutants       each planted mutant must turn the case table RED
# EXIT 0 GREEN · 1 RED · 2 NOT_MEASURED · 3 caller error
set -uo pipefail
PROG="${0##*/}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$HERE/${BASH_SOURCE[0]##*/}"
LIB="$HERE/../lib/known_failure_ratchet.sh"
LIST_REL="scripts/release/known_failures.tsv"

caller_error() { printf 'FAIL  KFR %s: caller error: %s\n' "$PROG" "$*"; exit 3; }

# ------------------------------------------------------------------------------------ the case table ----
# row NAME RC MUST-MATCH MUST-NOT-MATCH -- ARGS...   (fixtures live under $FX)
CASES=0; FAILED=0
row() {
    local name="$1" want="$2" must="$3" mustnot="$4" out rc
    shift 5
    out="$(bash "$SCRIPT_PATH" "$@" 2>&1)"; rc=$?
    CASES=$((CASES + 1))
    if [ "$rc" != "$want" ]; then printf 'RED   %-44s rc=%s want %s\n' "$name" "$rc" "$want"; FAILED=$((FAILED + 1)); return; fi
    if [ -n "$must" ] && ! printf '%s\n' "$out" | grep -qF -e "$must"; then printf 'RED   %-44s missing: %s\n' "$name" "$must"; FAILED=$((FAILED + 1)); return; fi
    if [ -n "$mustnot" ] && printf '%s\n' "$out" | grep -qF -e "$mustnot"; then printf 'RED   %-44s forbidden: %s\n' "$name" "$mustnot"; FAILED=$((FAILED + 1)); return; fi
    printf 'ok    %s\n' "$name"
}
# fixture helpers: three keys a, b, c; a and b are listed (#1, #2), c is not
SA=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
SB=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
SC=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
tsv() { printf '%s\t%s\t%s\t%s\n' "$@"; }
bundle() { # DIR STATE-A STATE-B STATE-C   ('-' leaves the key out)
    mkdir -p "$1"
    { printf 'sha256\tleg\tcheck\tstate\n'
      [ "$2" = - ] || tsv "$SA" sm_89 golden_output "$2"
      [ "$3" = - ] || tsv "$SB" sm_121 serve_api_chat "$3"
      [ "$4" = - ] || tsv "$SC" sm_89 capability_match "$4"; } > "$1/checks.tsv"
}
self_test() {
    local FX
    FX="$(mktemp -d "${TMPDIR:-/tmp}/kfr-st.XXXXXX")" || caller_error "no temp dir"
    # lists
    { printf 'sha256\tleg\tcheck\tticket\tmodel\n'; tsv "$SA" sm_89 golden_output '#1'; tsv "$SB" sm_121 serve_api_chat '#2'; } > "$FX/list"
    { cat "$FX/list"; tsv "$SC" sm_89 capability_match '#3'; } > "$FX/grew"
    { printf 'sha256\tleg\tcheck\tticket\n'; tsv "$SA" sm_89 golden_output '#1'; tsv "$SB" sm_121 serve_api_chat '#9'; } > "$FX/edited"
    { printf 'sha256\tleg\tcheck\tticket\n'; tsv "$SA" sm_89 golden_output '#1'; printf '%s\tsm_121\tserve_api_chat\n' "$SB"; } > "$FX/noticket"
    { printf 'sha256\tleg\tcheck\tticket\n'; tsv "$SA" sm_89 golden_output '#1'; tsv "$SB" sm_121 serve_api_chat '2'; } > "$FX/badticket"
    { cat "$FX/list"; tsv "$SA" sm_89 golden_output '#1'; } > "$FX/dup"
    { printf 'sha256\tleg\tcheck\tticket\n'; tsv "$SA" sm_89 golden_output '#1'; } > "$FX/shrunk"
    # bundles: base = last release, head = tonight
    bundle "$FX/base" fail fail pass
    bundle "$FX/h_known" fail fail pass
    bundle "$FX/h_new" fail fail fail
    bundle "$FX/h_fixed" pass fail pass
    bundle "$FX/h_nmlisted" not_measured fail pass
    bundle "$FX/h_nolisted" - fail pass
    bundle "$FX/h_cgone" fail fail -
    bundle "$FX/h_cnm" fail fail not_measured
    bundle "$FX/h_shrunk" fail pass pass
    mkdir -p "$FX/h_nochecks"; : > "$FX/h_nochecks/bundle.tsv"
    bundle "$FX/h_badstate" fail fail pass; tsv "$SC" sm_89 capability_match maybe >> "$FX/h_badstate/checks.tsv"
    bundle "$FX/h_conflict" fail fail pass; tsv "$SC" sm_89 capability_match fail >> "$FX/h_conflict/checks.tsv"
    mkdir -p "$FX/h_empty"; printf 'sha256\tleg\tcheck\tstate\n' > "$FX/h_empty/checks.tsv"
    local L=(--list "$FX/list" --list-base "$FX/list")

    row must_accept_only_listed_failures_is_green 0 "KFR GREEN: listed=2 known_failing=2 new=0 unlisted=0" "FAIL" -- "${L[@]}" --head "$FX/h_known" --base "$FX/base"
    row a_known_failure_is_still_printed_red 0 "counted RED with its ticket, never green" "" -- "${L[@]}" --head "$FX/h_known" --base "$FX/base"
    row a_known_failure_names_its_ticket 0 "aaaaaaaaaaaa sm_89 golden_output #1" "" -- "${L[@]}" --head "$FX/h_known" --base "$FX/base"
    row planted_new_failure_is_red 1 "KFR NEW cccccccccccc sm_89 capability_match" "KFR GREEN" -- "${L[@]}" --head "$FX/h_new" --base "$FX/base"
    row planted_new_failure_counts 1 "KFR RED: listed=2 known_failing=2 new=1 unlisted=0" "" -- "${L[@]}" --head "$FX/h_new" --base "$FX/base"
    row unlisted_failure_without_base_is_red 1 "KFR UNLISTED cccccccccccc" "KFR GREEN" -- "${L[@]}" --head "$FX/h_new" --base none
    row fixed_entry_left_in_list_is_red 1 "KFR STALE aaaaaaaaaaaa sm_89 golden_output #1" "KFR GREEN" -- "${L[@]}" --head "$FX/h_fixed" --base "$FX/base"
    row list_that_grew_is_red 1 "KFR ADDED cccccccccccc sm_89 capability_match #3" "KFR GREEN" -- --list "$FX/grew" --list-base "$FX/list" --head "$FX/h_new" --base "$FX/base"
    row edited_ticket_is_red 1 "KFR ADDED bbbbbbbbbbbb sm_121 serve_api_chat #9" "KFR GREEN" -- --list "$FX/edited" --list-base "$FX/list" --head "$FX/h_known" --base "$FX/base"
    row entry_without_ticket_is_red 1 "has no ticket" "KFR GREEN" -- --list "$FX/noticket" --list-base "$FX/noticket" --head "$FX/h_known" --base "$FX/base"
    row ticket_not_a_number_is_red 1 "is not #<number>" "KFR GREEN" -- --list "$FX/badticket" --list-base "$FX/badticket" --head "$FX/h_known" --base "$FX/base"
    row key_listed_twice_is_red 1 "is listed twice" "KFR GREEN" -- --list "$FX/dup" --list-base "$FX/dup" --head "$FX/h_known" --base "$FX/base"
    row a_shrunk_list_is_green 0 "KFR GREEN: listed=1 known_failing=1" "FAIL" -- --list "$FX/shrunk" --list-base "$FX/list" --head "$FX/h_shrunk" --base "$FX/base"
    row missing_head_bundle_is_not_measured 2 "KFR NOT_MEASURED" "KFR GREEN" -- "${L[@]}" --head "$FX/absent" --base "$FX/base"
    row head_without_checks_tsv_is_not_measured 2 "no readable checks.tsv" "KFR GREEN" -- "${L[@]}" --head "$FX/h_nochecks" --base "$FX/base"
    row head_measuring_nothing_is_not_measured 2 "measured no check" "KFR GREEN" -- "${L[@]}" --head "$FX/h_empty" --base "$FX/base"
    row unknown_state_is_not_measured 2 "unknown state \"maybe\"" "KFR GREEN" -- "${L[@]}" --head "$FX/h_badstate" --base "$FX/base"
    row conflicting_rows_are_not_measured 2 "is both pass and fail" "KFR GREEN" -- "${L[@]}" --head "$FX/h_conflict" --base "$FX/base"
    row listed_key_not_measured_is_not_measured 2 "listed, not measured on the head" "KFR GREEN" -- "${L[@]}" --head "$FX/h_nmlisted" --base "$FX/base"
    row listed_key_absent_is_not_measured 2 "listed, not measured on the head" "KFR GREEN" -- "${L[@]}" --head "$FX/h_nolisted" --base "$FX/base"
    row base_pass_absent_here_is_not_measured 2 "passed in the last release bundle, not measured on the head" "KFR GREEN" -- "${L[@]}" --head "$FX/h_cgone" --base "$FX/base"
    row base_pass_not_measured_here_is_not_measured 2 "passed in the last release bundle, not measured on the head" "KFR GREEN" -- "${L[@]}" --head "$FX/h_cnm" --base "$FX/base"
    row unreadable_base_bundle_is_not_measured 2 "base bundle: not_measured" "KFR GREEN" -- "${L[@]}" --head "$FX/h_known" --base "$FX/absent"
    row red_outranks_not_measured 1 "KFR RED" "KFR NOT_MEASURED" -- --list "$FX/grew" --list-base "$FX/list" --head "$FX/absent" --base "$FX/base"
    row bootstrap_without_a_base_list_still_judges 1 "KFR BOOTSTRAP" "KFR GREEN" -- --list "$FX/list" --list-base none --head "$FX/h_fixed" --base "$FX/base"
    row missing_list_is_a_caller_error 3 "caller error" "KFR GREEN" -- --list "$FX/absent" --list-base none --head "$FX/h_known"
    row unknown_option_is_a_caller_error 3 "caller error" "" -- --head "$FX/h_known" --skip-stale
    # the committed seed: six tickets, nine (model, leg) rows, valid; tonight has no bundle yet
    row the_seed_list_is_valid_and_not_measured 2 "KFR NOT_MEASURED: listed=9 known_failing=0" "FAIL" -- --list "$HERE/known_failures.tsv" --list-base "$HERE/known_failures.tsv" --head "$FX/absent"
    row the_seed_names_tickets_4661_to_4666 0 "tickets=#4661 #4662 #4663 #4664 #4665 #4666" "" -- --print-tickets "$HERE/known_failures.tsv"
    rm -rf -- "${FX:?}"
    printf '%s %s/%s case rows green\n' "$([ "$FAILED" -eq 0 ] && echo SELF-TEST-GREEN || echo SELF-TEST-RED)" "$((CASES - FAILED))" "$CASES"
    [ "$FAILED" -eq 0 ]
}

# ------------------------------------------------------------------------------------ the mutants ----
# name<TAB>sed expression on the lib. Each must change the lib and turn the case table RED.
MUTANTS='m01_new_not_red	s/new++; red = 1/new++/
m02_unlisted_not_red	s/unl++; red = 1/unl++/
m03_stale_not_red	s/H\[k\] == "pass") { printf "FAIL  KFR STALE/H[k] == "pass_never") { printf "FAIL  KFR STALE/
m04_added_not_red	s/if (!(LT\[k\] in LB))/if (0)/
m05_no_ticket_accepted	s/if ($4 == "") { bad(/if (0) { bad(/
m06_ticket_format_accepted	s/if ($4 !~ \/^#\[0-9\]+$\/)/if (0)/
m07_duplicate_accepted	s/if (k in seen) {/if (0) {/
m08_missing_bundle_passes	s/does not exist: not_measured\\n'"'"' "$1"; return 2/does not exist: not_measured\\n'"'"' "$1"; return 0/
m09_unknown_state_accepted	s/if ($4 != "pass" \&\& $4 != "fail" \&\& $4 != "not_measured")/if (0)/
m10_empty_bundle_measured	s/if (rc == 0 \&\& n == 0)/if (0)/
m11_listed_unmeasured_ignored	s/listed, not measured on the head\\n", name(k), L\[k\]; nm = 1/listed, not measured on the head\\n", name(k), L[k]/
m12_base_pass_gone_ignored	s/not measured on the head\\n", name(k); nm = 1/not measured on the head\\n", name(k)/
m13_nm_outranks_red	s/if (red) { printf "KFR RED/if (red \&\& !nm) { printf "KFR RED/
m14_head_nm_is_green	s/\[ "\$rc" -eq 0 \] || { nm=1; printf .NM    KFR head bundle/[ "$rc" -eq 0 ] || { printf '"'"'NM    KFR head bundle/
m15_base_unreadable_ignored	s/\[ "\$rc" -eq 0 \] || { nm=1; printf .NM    KFR base bundle/[ "$rc" -eq 0 ] || { printf '"'"'NM    KFR base bundle/
m16_conflict_accepted	s/if ((k in st) \&\& st\[k\] != $4)/if (0)/
m17_known_counted_green	s/H\[k\] == "not_measured") { printf "NM    KFR %s %s: listed/H[k] == "fail") { printf "NM    KFR %s %s: listed/'
mutants() {
    local tmp name expr killed=0 total=0 errors=0 out
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/kfr-mu.XXXXXX")" || caller_error "no temp dir"
    mkdir -p "$tmp/release" "$tmp/lib"
    cp "$SCRIPT_PATH" "$HERE/known_failures.tsv" "$tmp/release/" || caller_error "cannot copy the checker"
    while IFS="$(printf '\t')" read -r name expr; do
        [ -n "$name" ] || continue
        total=$((total + 1))
        sed -e "$expr" "$LIB" > "$tmp/lib/known_failure_ratchet.sh"
        if cmp -s "$LIB" "$tmp/lib/known_failure_ratchet.sh"; then
            printf 'ERROR %-32s the patch did not apply (an error, never a survivor)\n' "$name"; errors=$((errors + 1)); continue
        fi
        if out="$(bash "$tmp/release/known_failure_ratchet.sh" --self-test 2>&1)"; then
            printf 'SURVIVED %s\n' "$name"
        else
            killed=$((killed + 1))
            printf 'killed   %-32s %s\n' "$name" "$(printf '%s\n' "$out" | grep -c -e '^RED ')"
        fi
    done <<< "$MUTANTS"
    rm -rf -- "${tmp:?}"
    printf 'MUTANTS killed=%s total=%s errors=%s\n' "$killed" "$total" "$errors"
    [ "$killed" -eq "$total" ] && [ "$errors" -eq 0 ]
}

# ------------------------------------------------------------------------------------ main ----
[ -f "$LIB" ] || caller_error "no library at $LIB"
# shellcheck source=../lib/known_failure_ratchet.sh
. "$LIB" || caller_error "cannot source $LIB"

case "${1:-}" in
    --self-test) self_test; exit $? ;;
    --mutants) mutants; exit $? ;;
    --print-tickets)
        [ -f "${2:-}" ] || caller_error "--print-tickets FILE"
        t="$(mktemp)" || caller_error "no temp file"
        kfr_list "$2" "$t" || { rm -f -- "${t:?}"; exit 1; }
        printf 'tickets=%s rows=%s\n' "$(cut -f4 "$t" | sort -u -t '#' -k2,2n | tr '\n' ' ' | sed 's/ $//')" "$(wc -l < "$t" | tr -d ' ')"
        rm -f -- "${t:?}"; exit 0 ;;
esac

head_dir=''; base_dir=none; list=''; list_base=origin/main
while [ "$#" -gt 0 ]; do
    case "$1" in
        --head) head_dir="${2:-}"; shift 2 || caller_error "--head DIR" ;;
        --base) base_dir="${2:-}"; shift 2 || caller_error "--base DIR|none" ;;
        --list) list="${2:-}"; shift 2 || caller_error "--list FILE" ;;
        --list-base) list_base="${2:-}"; shift 2 || caller_error "--list-base FILE|none|REF" ;;
        -h|--help) sed -n '2,29p' "$SCRIPT_PATH"; exit 0 ;;
        *) caller_error "unknown argument '$1'" ;;
    esac
done
[ -n "$head_dir" ] || caller_error "--head DIR is required"
[ -n "$list" ] || list="$HERE/known_failures.tsv"
[ -f "$list" ] || caller_error "no list at '$list'"
lb_tmp=''
if [ "$list_base" != none ] && [ ! -f "$list_base" ]; then
    lb_tmp="$(mktemp)" || caller_error "no temp file"
    if git -C "$HERE" show "$list_base:$LIST_REL" > "$lb_tmp" 2>/dev/null; then list_base="$lb_tmp"
    elif git -C "$HERE" rev-parse --verify -q "$list_base^{commit}" > /dev/null; then list_base=none   # the ref has no list yet
    else rm -f -- "${lb_tmp:?}"; caller_error "--list-base '$list_base' is not a file, none, or a git ref"; fi
fi
kfr_judge "$list" "$list_base" "$head_dir" "$base_dir"; rc=$?
[ -z "$lb_tmp" ] || rm -f -- "${lb_tmp:?}"
exit "$rc"
