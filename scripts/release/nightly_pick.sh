#!/usr/bin/env bash
# nightly_pick.sh — pick the night's candidate C once and publish it as refs/heads/nightly/<night>.
#
# The rule (operator): one job picks C once a night and publishes it as a ref; every producer measures that C; the
# train prints its line for that C; a release promotes that C. Planted falsifier: a merge to main after C is picked
# must not change the night's line.
#
# Why a ref and not cron spacing: scheduled events on this account fire 1h43m-5h48m late (14 nights measured), so
# workflows that each read main's head at their own late start judge different commits. The night is chained from the
# one pick instead: producers start on the pick (workflow_run or dispatch) and verify the commit they were handed.
#
# Usage:
#   nightly_pick.sh night   [--at EPOCH]
#   nightly_pick.sh pick    [--repo DIR] [--remote NAME] [--night N | --at EPOCH] --sha SHA
#   nightly_pick.sh resolve [--repo DIR] [--remote NAME] --night N
#   nightly_pick.sh verify  [--repo DIR] [--remote NAME] --night N --sha SHA
#   nightly_pick.sh --self-test | --mutants
# Exit: 0 ok; 1 RED (refused); 2 not_measured; 3 caller error. Report-only: nothing reads this yet.
set -uo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$HERE/$(basename -- "${BASH_SOURCE[0]}")"
LIB="$HERE/../lib/nightly_pick.sh"

caller_error() { printf 'nightly_pick: caller error: %s\n' "$1" >&2; exit 3; }

# ------------------------------------------------------------------------------------ the case table ----
CASES=0; FAILED=0
row() { # NAME WANT-RC MUST MUSTNOT -- CMD...
    local name="$1" want="$2" must="$3" mustnot="$4" out rc
    shift 5
    out="$("$@" 2>&1)"; rc=$?
    CASES=$((CASES + 1))
    if [ "$rc" != "$want" ]; then printf 'RED   %-44s rc=%s want %s\n' "$name" "$rc" "$want"; FAILED=$((FAILED + 1)); return; fi
    if [ -n "$must" ] && ! printf '%s\n' "$out" | grep -qF -e "$must"; then printf 'RED   %-44s missing: %s\n' "$name" "$must"; FAILED=$((FAILED + 1)); return; fi
    if [ -n "$mustnot" ] && printf '%s\n' "$out" | grep -qF -e "$mustnot"; then printf 'RED   %-44s forbidden: %s\n' "$name" "$mustnot"; FAILED=$((FAILED + 1)); return; fi
    printf 'ok    %s\n' "$name"
}
same() { # NAME A B : two outputs must be equal
    CASES=$((CASES + 1))
    if [ -n "$2" ] && [ "$2" = "$3" ]; then printf 'ok    %s\n' "$1"; else printf 'RED   %-44s %s != %s\n' "$1" "$2" "$3"; FAILED=$((FAILED + 1)); fi
}
commit() { # REPO MSG -> new head sha
    git -C "$1" -c core.hooksPath=/dev/null -c user.name=fixture -c user.email=fixture@example.invalid commit -q --allow-empty -m "$2" && git -C "$1" rev-parse HEAD
}
# the train's line for a night, as the train must print it: from the night's ref, never from main
train_line() { local c; c="$(bash "$SCRIPT_PATH" resolve --repo "$1" --night "$2")" || return $?; printf 'NOT RELEASABLE: stub [C=%s]\n' "${c:0:10}"; }
# the line as a train that reads main's head prints it: the defect this row removes
main_line() { printf 'NOT RELEASABLE: stub [C=%s]\n' "$(git -C "$1" ls-remote "$2" refs/heads/main | cut -c1-10)"; }

self_test() {
    local FX W c1 c2 c3 br n=2026-10-04 l1 l2 m1 m2
    FX="$(mktemp -d "${TMPDIR:-/tmp}/np-st.XXXXXX")" || caller_error "no temp dir"
    W="$FX/w"
    git init -q --bare -b main "$FX/origin.git" && git clone -q "$FX/origin.git" "$W" 2>/dev/null || caller_error "no fixture repo"
    git -C "$W" checkout -q -b main 2>/dev/null
    c1="$(commit "$W" one)" && git -C "$W" push -q origin main || caller_error "fixture push"

    row night_mid_evening_is_today 0 "2026-10-04" "" -- bash "$SCRIPT_PATH" night --at 1791154800      # 2026-10-04T23:00Z
    row night_late_fire_is_still_the_night 0 "2026-10-04" "" -- bash "$SCRIPT_PATH" night --at 1791179280  # 2026-10-05T05:48Z
    row night_turns_at_noon 0 "2026-10-05" "" -- bash "$SCRIPT_PATH" night --at 1791201600            # 2026-10-05T12:00Z
    row unpicked_night_is_not_measured 2 "has no pick" "" -- bash "$SCRIPT_PATH" resolve --repo "$W" --night "$n"
    row first_pick_publishes_the_ref 0 "C=$c1 PICKED night $n" "" -- bash "$SCRIPT_PATH" pick --repo "$W" --night "$n" --sha "$c1"
    row the_ref_is_on_the_remote 0 "$c1" "" -- git -C "$W" ls-remote "$FX/origin.git" "refs/heads/nightly/$n"
    l1="$(train_line "$W" "$n")"; m1="$(main_line "$W" "$FX/origin.git")"
    # THE PLANT: a merge to main after the pick
    c2="$(commit "$W" merged-after-the-pick)" && git -C "$W" push -q origin main || caller_error "fixture push"
    row a_second_pick_keeps_the_first_c 0 "C=$c1 KEPT night $n" "PICKED" -- bash "$SCRIPT_PATH" pick --repo "$W" --night "$n" --sha "$c2"
    row resolve_after_a_merge_is_still_c 0 "$c1" "$c2" -- bash "$SCRIPT_PATH" resolve --repo "$W" --night "$n"
    l2="$(train_line "$W" "$n")"; m2="$(main_line "$W" "$FX/origin.git")"
    same a_merge_after_the_pick_does_not_change_the_line "$l1" "$l2"
    CASES=$((CASES + 1))   # the counter-row: reading main's head DOES change the line, so the plant can fail
    if [ "$m1" != "$m2" ]; then printf 'ok    %s\n' main_head_reader_changes_the_line; else printf 'RED   %-44s the plant did not move main\n' main_head_reader_changes_the_line; FAILED=$((FAILED + 1)); fi
    row producer_handed_c_verifies 0 "NP OK" "" -- bash "$SCRIPT_PATH" verify --repo "$W" --night "$n" --sha "$c1"
    row producer_handed_main_head_is_red 1 "is not the C of night $n" "NP OK" -- bash "$SCRIPT_PATH" verify --repo "$W" --night "$n" --sha "$c2"
    row verify_on_an_unpicked_night_is_not_measured 2 "has no pick" "NP OK" -- bash "$SCRIPT_PATH" verify --repo "$W" --night 2026-10-03 --sha "$c1"
    row next_night_picks_its_own_c 0 "C=$c2 PICKED night 2026-10-05" "" -- bash "$SCRIPT_PATH" pick --repo "$W" --night 2026-10-05 --sha "$c2"
    row the_old_night_still_resolves_to_its_c 0 "$c1" "$c2" -- bash "$SCRIPT_PATH" resolve --repo "$W" --night "$n"
    git -C "$W" checkout -q -b side && br="$(commit "$W" branch-only)" && git -C "$W" push -q origin side && git -C "$W" checkout -q main
    row a_commit_not_on_main_is_refused 1 "is not on main" "PICKED" -- bash "$SCRIPT_PATH" pick --repo "$W" --night 2026-10-06 --sha "$br"
    row a_refused_pick_publishes_no_ref 2 "has no pick" "" -- bash "$SCRIPT_PATH" resolve --repo "$W" --night 2026-10-06
    c3="$(commit "$W" racer)" && git -C "$W" push -q origin main && git -C "$W" push -q origin "$c3:refs/heads/nightly/2026-10-07"
    row a_pick_never_overwrites_an_existing_ref 0 "C=$c3 KEPT night 2026-10-07" "PICKED" -- bash "$SCRIPT_PATH" pick --repo "$W" --night 2026-10-07 --sha "$c2"
    # a race: the ref is absent when read and created by another picker before the push lands. race_shim DIR NIGHT
    # WINNER puts a git on PATH whose first ls-remote sees nothing and whose push first creates the ref at WINNER.
    race_shim() {
        mkdir -p "$1" && printf '%s\n' '#!/usr/bin/env bash' \
            "case \" \$* \" in" \
            "  *' ls-remote '*) [ -f '$1/seen' ] || { : > '$1/seen'; exit 0; } ;;" \
            "  *' push '*) '$(command -v git)' -C '$FX/origin.git' update-ref refs/heads/nightly/$2 '$3' ;;" \
            "esac" "exec '$(command -v git)' \"\$@\"" > "$1/git" && chmod +x "$1/git"
    }
    race_shim "$FX/bin" 2026-10-10 "$c3"
    row a_lost_race_keeps_the_winner 0 "C=$c3 KEPT night 2026-10-10 (another pick won the race)" "PICKED" -- env PATH="$FX/bin:$PATH" bash "$SCRIPT_PATH" pick --repo "$W" --night 2026-10-10 --sha "$c2"
    # the winner is an ANCESTOR of the loser's commit (main only moves forward), so a plain push would fast-forward it
    race_shim "$FX/bin-ff" 2026-10-11 "$c1"
    row a_lost_race_to_an_older_commit_keeps_the_winner 0 "C=$c1 KEPT night 2026-10-11 (another pick won the race)" "PICKED" -- env PATH="$FX/bin-ff:$PATH" bash "$SCRIPT_PATH" pick --repo "$W" --night 2026-10-11 --sha "$c3"
    row a_lost_race_leaves_the_ref_at_the_winner 0 "$c1" "$c3" -- bash "$SCRIPT_PATH" resolve --repo "$W" --night 2026-10-11
    row an_unknown_commit_is_not_measured 2 "cannot tell whether" "PICKED" -- bash "$SCRIPT_PATH" pick --repo "$W" --night 2026-10-12 --sha 0123456789abcdef0123456789abcdef01234567
    # a decoy: a BRANCH named refs/heads/nightly/<night> must not be read as the night's ref (ls-remote matches by tail)
    git -C "$W" push -q origin "$br:refs/heads/refs/heads/nightly/2026-10-13" || caller_error "fixture decoy push"
    row a_decoy_branch_is_not_the_night 2 "is ambiguous" "C=" -- bash "$SCRIPT_PATH" pick --repo "$W" --night 2026-10-13 --sha "$c3"
    row a_decoy_branch_does_not_resolve 2 "is ambiguous" "$br" -- bash "$SCRIPT_PATH" resolve --repo "$W" --night 2026-10-13
    # the rolling tag refs/tags/nightly (as on the real remote) is never C: only the branch is
    git -C "$W" push -q origin "$br:refs/tags/nightly" || caller_error "fixture tag push"
    row the_rolling_tag_is_not_a_pick 2 "has no pick" "$br" -- bash "$SCRIPT_PATH" resolve --repo "$W" --night 2026-10-14
    row a_pick_beside_the_rolling_tag_picks_the_branch 0 "C=$c3 PICKED night 2026-10-14" "KEPT" -- bash "$SCRIPT_PATH" pick --repo "$W" --night 2026-10-14 --sha "$c3"
    row the_rolling_tag_never_resolves 0 "$c3" "$br" -- bash "$SCRIPT_PATH" resolve --repo "$W" --night 2026-10-14
    # a branch named nightly blocks every nightly/<night> (a directory/file clash): not_measured, never a C
    git init -q --bare -b main "$FX/df.git" && git -C "$W" push -q "$FX/df.git" main "$c3:refs/heads/nightly" || caller_error "fixture df remote"
    row a_branch_named_nightly_is_not_measured 2 "NOT_MEASURED" "C=" -- bash "$SCRIPT_PATH" pick --repo "$W" --remote "$FX/df.git" --night 2026-10-15 --sha "$c3"
    row unreachable_remote_is_not_measured 2 "NOT_MEASURED" "PICKED" -- bash "$SCRIPT_PATH" pick --repo "$W" --remote "$FX/absent.git" --night 2026-10-08 --sha "$c2"
    row unreachable_remote_resolve_is_not_measured 2 "cannot read" "" -- bash "$SCRIPT_PATH" resolve --repo "$W" --remote "$FX/absent.git" --night "$n"
    row malformed_night_is_a_caller_error 3 "not YYYY-MM-DD" "PICKED" -- bash "$SCRIPT_PATH" pick --repo "$W" --night 2026-1-4 --sha "$c1"
    row malformed_sha_is_a_caller_error 3 "not 40 hex" "PICKED" -- bash "$SCRIPT_PATH" pick --repo "$W" --night 2026-10-09 --sha "${c1:0:10}"
    # a multi-line argument: one well-formed line must not carry another past the check (grep -x matches per line)
    row multiline_night_is_a_caller_error 3 "not YYYY-MM-DD" "PICKED" -- bash "$SCRIPT_PATH" pick --repo "$W" --night "2026-10-09"$'\n'"x" --sha "$c1"
    row multiline_sha_is_a_caller_error 3 "not 40 hex" "PICKED" -- bash "$SCRIPT_PATH" pick --repo "$W" --night 2026-10-09 --sha "$c1"$'\n'"x"
    row trailing_newline_night_is_a_caller_error 3 "not YYYY-MM-DD" "PICKED" -- bash "$SCRIPT_PATH" pick --repo "$W" --night "2026-10-09"$'\n' --sha "$c1"
    row trailing_newline_sha_is_a_caller_error 3 "not 40 hex" "PICKED" -- bash "$SCRIPT_PATH" pick --repo "$W" --night 2026-10-09 --sha "$c1"$'\n'
    row pick_without_sha_is_a_caller_error 3 "caller error" "PICKED" -- bash "$SCRIPT_PATH" pick --repo "$W" --night 2026-10-09
    row unknown_command_is_a_caller_error 3 "caller error" "" -- bash "$SCRIPT_PATH" repick
    rm -rf -- "${FX:?}"
    printf '%s %s/%s case rows green\n' "$([ "$FAILED" -eq 0 ] && echo SELF-TEST-GREEN || echo SELF-TEST-RED)" "$((CASES - FAILED))" "$CASES"
    [ "$FAILED" -eq 0 ]
}

# ------------------------------------------------------------------------------------ the mutants ----
# name<TAB>sed expression on the lib. Each must change the lib and turn the case table RED.
MUTANTS='m01_pick_moves_the_ref	s/--force-with-lease="refs\/heads\/nightly\/\$3:"/-f/
m02_existing_ref_ignored	s/    if \[ "\$rc" -eq 0 \]; then$/    if false; then/
m03_branch_commit_accepted	s/merge-base --is-ancestor "\$4" FETCH_HEAD 2>\/dev\/null ||/true ||/
m04_verify_accepts_any_sha	s/if \[ "\$c" = "\$4" \]; then printf .NP OK/if true; then printf '"'"'NP OK/
m05_unpicked_night_resolves	s/1) printf .NP NOT_MEASURED: night %s has no pick\\n. "\$3" >\&2; return 2 ;;/1) printf "\\n"; return 0 ;;/
m06_unreadable_is_absent	s/\[ "\$rc" -eq 0 \] || return 2$/[ "$rc" -eq 0 ] || return 1/
m07_night_is_the_fire_date	s/\$((e - 43200))/$e/
m08_bad_night_accepted	s/\[\[ \$1 =~ .*\$ \]\] || {/true || {/
m09_bad_sha_accepted	s/\[\[ \$2 =~ .*\$ \]\] || {/true || {/
m10_race_loser_claims_picked	s/printf .C=%s KEPT night %s (another pick won the race)\\n. "\$c" "\$3"/printf '"'"'C=%s PICKED night %s\\n'"'"' "$c" "$3"/
m11_plain_push_fast_forwards	s/ --force-with-lease="refs\/heads\/nightly\/\$3:"//
m12_unknown_commit_is_red	/cannot tell whether/s/return 2 ;;/return 1 ;;/
m13_ref_name_matched_by_tail	s/NF && \$2 != r { found = 1 }/NF \&\& 0 { found = 1 }/
m14_read_widened_to_the_rolling_tag	s/ls-remote --refs "\$2" "refs\/heads\/nightly\/\$3"/ls-remote --refs "\$2" "nightly"/
m15_night_checked_per_line	s/\[\[ \$1 =~ \(.*\) \]\] ||/printf "%s\\n" "$1" | grep -qxE "\1" ||/
m16_sha_checked_per_line	s/\[\[ \$2 =~ \(.*\) \]\] ||/printf "%s\\n" "$2" | grep -qxE "\1" ||/'
mutants() {
    local tmp name expr killed=0 total=0 errors=0 out
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/np-mu.XXXXXX")" || caller_error "no temp dir"
    mkdir -p "$tmp/release" "$tmp/lib"
    cp "$SCRIPT_PATH" "$tmp/release/" || caller_error "cannot copy the checker"
    while IFS="$(printf '\t')" read -r name expr; do
        [ -n "$name" ] || continue
        total=$((total + 1))
        sed -e "$expr" "$LIB" > "$tmp/lib/nightly_pick.sh"
        if cmp -s "$LIB" "$tmp/lib/nightly_pick.sh"; then
            printf 'ERROR %-32s the patch did not apply (an error, never a survivor)\n' "$name"; errors=$((errors + 1)); continue
        fi
        if out="$(bash "$tmp/release/nightly_pick.sh" --self-test 2>&1)"; then
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
# shellcheck source=../lib/nightly_pick.sh
. "$LIB" || caller_error "cannot source $LIB"

cmd="${1:-}"; [ "$#" -gt 0 ] && shift
case "$cmd" in
    --self-test) self_test; exit $? ;;
    --mutants) mutants; exit $? ;;
    night | pick | resolve | verify) ;;
    -h | --help) sed -n '2,19p' "$SCRIPT_PATH"; exit 0 ;;
    *) caller_error "unknown command '$cmd' (night|pick|resolve|verify|--self-test|--mutants)" ;;
esac
repo=.; remote=origin; night=''; at=''; sha=''
while [ "$#" -gt 0 ]; do
    case "$1" in
        --repo) repo="${2:-}"; shift 2 || caller_error "--repo DIR" ;;
        --remote) remote="${2:-}"; shift 2 || caller_error "--remote NAME" ;;
        --night) night="${2:-}"; shift 2 || caller_error "--night YYYY-MM-DD" ;;
        --at) at="${2:-}"; shift 2 || caller_error "--at EPOCH" ;;
        --sha) sha="${2:-}"; shift 2 || caller_error "--sha SHA" ;;
        *) caller_error "unknown option '$1'" ;;
    esac
done
if [ "$cmd" = night ]; then np_night "$at"; exit $?; fi
if [ -z "$night" ]; then
    [ "$cmd" = pick ] || caller_error "--night YYYY-MM-DD"
    night="$(np_night "$at")" || exit 3
fi
case "$cmd" in
    pick) [ -n "$sha" ] || caller_error "pick needs --sha SHA"; np_pick "$repo" "$remote" "$night" "$sha" ;;
    resolve) np_resolve "$repo" "$remote" "$night" ;;
    verify) [ -n "$sha" ] || caller_error "verify needs --sha SHA"; np_verify "$repo" "$remote" "$night" "$sha" ;;
esac
exit $?
