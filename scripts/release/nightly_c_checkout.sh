#!/usr/bin/env bash
# nightly_c_checkout.sh — put a nightly producer's work tree on the night's C, the commit the pick published.
#
# A producer starts on the "Nightly pick" workflow_run, so actions/checkout hands it main's head at the moment the
# event fired, not C: a merge that lands while the pick runs, or a rerun of the pick, moves main away from the ref.
# This step reads refs/heads/nightly/<night> (scripts/lib/nightly_pick.sh), switches the tree to that commit when it
# is not already there, and then asks np_verify whether HEAD is C. A producer that measures any other commit is RED.
#
# The night comes from --at, the pick run's start (github.event.workflow_run.run_started_at), never from the clock
# here: a producer queued for hours still measures the night it was started for.
#
# Usage:
#   nightly_c_checkout.sh [checkout] [--repo DIR] [--remote NAME] --at ISO8601|EPOCH
#   nightly_c_checkout.sh --self-test | --mutants
# On success it appends NIGHTLY_C and NIGHTLY_NIGHT to $GITHUB_ENV and one line to $GITHUB_STEP_SUMMARY, when set.
# Exit: 0 HEAD is C; 1 RED (HEAD is not C, or the tree could not be switched); 2 not_measured (no pick, or the remote
# could not be read); 3 caller error.
#
# Everything runs inside functions, called on the last line: the checkout replaces this file under a running bash.
set -uo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$HERE/$(basename -- "${BASH_SOURCE[0]}")"
LIB="$HERE/../lib/nightly_pick.sh"

caller_error() { printf 'nightly_c_checkout: caller error: %s\n' "$1" >&2; exit 3; }

to_epoch() { # ISO8601|EPOCH -> epoch
    case "$1" in
        '') return 3 ;;
        *[!0-9]*) date -u -d "$1" +%s 2>/dev/null || return 3 ;;
        *) printf '%s\n' "$1" ;;
    esac
}

checkout_c() { # REPO REMOTE AT
    local repo="$1" remote="$2" epoch night c head rc=0 how
    epoch="$(to_epoch "$3")" || { printf 'NC caller error: --at %s is not a time\n' "$3" >&2; return 3; }
    night="$(np_night "$epoch")" || return 3
    c="$(np_resolve "$repo" "$remote" "$night")" || return $?
    head="$(git -C "$repo" rev-parse HEAD 2>/dev/null)" || { printf 'NC RED: %s is not a work tree\n' "$repo"; return 1; }
    if [ "$head" != "$c" ]; then
        git -C "$repo" fetch -q --no-tags "$remote" "+refs/heads/nightly/$night:refs/remotes/$remote/nightly/$night" 2>/dev/null \
            || { printf 'NC NOT_MEASURED: cannot fetch refs/heads/nightly/%s\n' "$night"; return 2; }
        git -C "$repo" -c advice.detachedHead=false checkout -q --detach "$c" 2>/dev/null \
            || { printf 'NC RED: cannot switch the tree from %s to C=%s\n' "${head:0:10}" "${c:0:10}"; return 1; }
        how="switched from ${head:0:10}"
    else
        how="already C"
    fi
    np_verify "$repo" "$remote" "$night" "$(git -C "$repo" rev-parse HEAD)" || { rc=$?; return "$rc"; }
    if [ -n "${GITHUB_ENV:-}" ]; then
        printf 'NIGHTLY_C=%s\nNIGHTLY_NIGHT=%s\n' "$c" "$night" >> "$GITHUB_ENV" || { printf 'NC NOT_MEASURED: cannot write GITHUB_ENV\n'; return 2; }
    fi
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
        printf 'Measured C=`%s` of night %s (%s).\n' "$c" "$night" "$how" >> "$GITHUB_STEP_SUMMARY" || true
    fi
    printf 'NIGHTLY C=%s night=%s (%s)\n' "$c" "$night" "$how"
}

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
same() { # NAME A B : two values must be equal
    CASES=$((CASES + 1))
    if [ -n "$2" ] && [ "$2" = "$3" ]; then printf 'ok    %s\n' "$1"; else printf 'RED   %-44s %s != %s\n' "$1" "$2" "$3"; FAILED=$((FAILED + 1)); fi
}
commit() { # REPO MSG -> new head sha; the file f holds MSG, so two commits differ in a tracked file
    printf "%s\n" "$2" > "$1/f" && git -C "$1" add f && git -C "$1" -c core.hooksPath=/dev/null -c user.name=fixture -c user.email=fixture@example.invalid commit -q -m "$2" && git -C "$1" rev-parse HEAD
}

self_test() {
    local FX A W S c1 c2 n=2026-10-04 at=2026-10-04T20:31:07Z env
    FX="$(mktemp -d "${TMPDIR:-/tmp}/nc-st.XXXXXX")" || caller_error "no temp dir"
    A="$FX/author"; W="$FX/w"; S="$FX/shallow"
    git init -q --bare -b main "$FX/origin.git" && git clone -q "$FX/origin.git" "$A" 2>/dev/null || caller_error "no fixture repo"
    git -C "$A" checkout -q -b main 2>/dev/null
    c1="$(commit "$A" one)" && git -C "$A" push -q origin main || caller_error "fixture push"
    git -C "$A" push -q origin "$c1:refs/heads/nightly/$n" || caller_error "fixture pick"
    c2="$(commit "$A" two)" && commit "$A" three >/dev/null && git -C "$A" push -q origin main || caller_error "fixture merge"
    git clone -q "$FX/origin.git" "$W" 2>/dev/null && git clone -q --depth 1 "file://$FX/origin.git" "$S" 2>/dev/null || caller_error "fixture clones"

    # the defect: a producer that checked out main's head and measured it
    row main_head_is_switched_to_c 0 "switched from" "" -- bash "$SCRIPT_PATH" checkout --repo "$W" --at "$at"
    same the_tree_is_on_c_after_a_merge "$(git -C "$W" rev-parse HEAD)" "$c1"
    row a_second_run_finds_c_already 0 "already C" "switched" -- bash "$SCRIPT_PATH" checkout --repo "$W" --at "$at"
    row a_shallow_checkout_reaches_c 0 "C=$c1" "" -- bash "$SCRIPT_PATH" checkout --repo "$S" --at "$at"
    same the_shallow_tree_is_on_c "$(git -C "$S" rev-parse HEAD)" "$c1"
    # a late start still measures the night it was started for: 05:48Z the next day is still night N
    git -C "$W" -c advice.detachedHead=false checkout -q --detach "$c2"
    row a_late_start_is_still_the_night 0 "night=$n" "" -- bash "$SCRIPT_PATH" checkout --repo "$W" --at 2026-10-05T05:48:00Z
    # the night comes from --at, not from the clock: an epoch the day after has no pick
    git -C "$W" -c advice.detachedHead=false checkout -q --detach "$c2"
    row an_unpicked_night_is_not_measured 2 "has no pick" "NIGHTLY C=" -- bash "$SCRIPT_PATH" checkout --repo "$W" --at 1791230400
    same not_measured_leaves_the_tree "$(git -C "$W" rev-parse HEAD)" "$c2"
    row epoch_at_is_accepted 0 "C=$c1" "" -- bash "$SCRIPT_PATH" checkout --repo "$W" --at 1791147600
    # a tree that cannot be switched is RED, never a measurement of the wrong commit
    git -C "$W" -c advice.detachedHead=false checkout -q --detach "$c2"
    printf 'local edit\n' > "$W/f"
    row an_unswitchable_tree_is_red 1 "cannot switch" "NIGHTLY C=" -- bash "$SCRIPT_PATH" checkout --repo "$W" --at "$at"
    same the_unswitchable_tree_is_left_alone "$(git -C "$W" rev-parse HEAD)" "$c2"
    git -C "$W" checkout -q -- f
    # GITHUB_ENV carries C to every later step; it is written only once HEAD is C
    env="$FX/github_env"; : > "$env"
    row env_is_written 0 "" "" -- env GITHUB_ENV="$env" bash "$SCRIPT_PATH" checkout --repo "$S" --at "$at"
    row env_holds_c 0 "NIGHTLY_C=$c1" "" -- cat "$env"
    row env_holds_the_night 0 "NIGHTLY_NIGHT=$n" "" -- cat "$env"
    : > "$env"
    row no_env_when_not_measured 2 "" "" -- env GITHUB_ENV="$env" bash "$SCRIPT_PATH" checkout --repo "$S" --at 1791230400
    same not_measured_writes_no_env "$(wc -c < "$env" | tr -d ' ')x" "0x"
    # a ref that moves while the step runs (a forced re-pick; the ruleset forbids it, the fixture plants it with a
    # post-checkout hook): the commit just checked out is no longer C, so the step is RED and C is never exported
    mkdir -p "$FX/hooks"
    printf '#!/bin/sh\ngit push -q -f "%s" "%s:refs/heads/nightly/%s" 2>/dev/null\n' "$FX/origin.git" "$c2" "$n" > "$FX/hooks/post-checkout"
    chmod +x "$FX/hooks/post-checkout"
    git -C "$W" -c advice.detachedHead=false checkout -q --detach "$c2"; : > "$env"
    row a_ref_moved_mid_step_is_red 1 "is not the C" "NIGHTLY C=" -- env GITHUB_ENV="$env" GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.hooksPath GIT_CONFIG_VALUE_0="$FX/hooks" bash "$SCRIPT_PATH" checkout --repo "$W" --at "$at"
    same a_moved_ref_exports_nothing "$(wc -c < "$env" | tr -d ' ')x" "0x"
    git -C "$A" push -q -f origin "$c1:refs/heads/nightly/$n" || caller_error "fixture restore"
    row unreachable_remote_is_not_measured 2 "cannot read" "NIGHTLY C=" -- bash "$SCRIPT_PATH" checkout --repo "$W" --remote "$FX/absent.git" --at "$at"
    row missing_at_is_a_caller_error 3 "--at" "" -- bash "$SCRIPT_PATH" checkout --repo "$W"
    row bad_at_is_a_caller_error 3 "not a time" "" -- bash "$SCRIPT_PATH" checkout --repo "$W" --at yesterday-ish
    row unknown_command_is_a_caller_error 3 "caller error" "" -- bash "$SCRIPT_PATH" measure
    rm -rf -- "${FX:?}"
    printf '%s %s/%s case rows green\n' "$([ "$FAILED" -eq 0 ] && echo SELF-TEST-GREEN || echo SELF-TEST-RED)" "$((CASES - FAILED))" "$CASES"
    [ "$FAILED" -eq 0 ]
}

# ------------------------------------------------------------------------------------ the mutants ----
# name<TAB>sed expression on the code above the case table (never on this table itself). Each must change that code
# and turn at least one case row RED; a mutant that breaks the script without a RED row is an error, not a kill.
MUTANTS='m01_main_head_measured	s/    if \[ "\$head" != "\$c" \]; then$/    if false; then/
m02_verify_skipped	s/    np_verify "\$repo" "\$remote" "\$night" "\$(git -C "\$repo" rev-parse HEAD)" || { rc=\$?; return "\$rc"; }/    true/
m03_env_not_written	s/printf .NIGHTLY_C=%s\\nNIGHTLY_NIGHT=%s\\n. "\$c" "\$night" >> "\$GITHUB_ENV"/true/
m04_no_pick_is_ok	s/    c="\$(np_resolve "\$repo" "\$remote" "\$night")" || return \$?/    c="$(np_resolve "$repo" "$remote" "$night")" || return 0/
m05_night_from_the_clock	s/    night="\$(np_night "\$epoch")" || return 3/    night="$(np_night)" || return 3/
m06_fetches_main	s/"+refs\/heads\/nightly\/\$night:refs\/remotes\/\$remote\/nightly\/\$night"/main/
m07_switch_failure_ignored	s/|| { printf .NC RED: cannot switch/|| true || { printf '"'"'NC RED: cannot switch/
m08_bad_at_accepted	s/date -u -d "\$1" +%s 2>\/dev\/null || return 3/date -u +%s/
m09_env_before_verify	/^    np_verify "\$repo"/{h;d};/^    printf .NIGHTLY C=%s night=%s/{x;G}'
mutants() {
    local tmp name expr killed=0 total=0 errors=0 out cut reds
    cut="$(grep -n -m1 -e "^# -* the case table" "$SCRIPT_PATH" | cut -d: -f1)"
    [ -n "$cut" ] || caller_error "no case-table marker"
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/nc-mu.XXXXXX")" || caller_error "no temp dir"
    mkdir -p "$tmp/release" "$tmp/lib"
    cp "$LIB" "$tmp/lib/nightly_pick.sh" || caller_error "cannot copy the lib"
    while IFS="$(printf '\t')" read -r name expr; do
        [ -n "$name" ] || continue
        total=$((total + 1))
        { head -n "$((cut - 1))" "$SCRIPT_PATH" | sed -e "$expr"; tail -n "+$cut" "$SCRIPT_PATH"; } > "$tmp/release/nightly_c_checkout.sh"
        if cmp -s "$SCRIPT_PATH" "$tmp/release/nightly_c_checkout.sh"; then
            printf 'ERROR %-32s the patch did not apply (an error, never a survivor)\n' "$name"; errors=$((errors + 1)); continue
        fi
        if out="$(bash "$tmp/release/nightly_c_checkout.sh" --self-test 2>&1)"; then
            printf 'SURVIVED %s\n' "$name"
        else
            reds="$(printf '%s\n' "$out" | grep -c -e '^RED ')"
            if [ "$reds" -eq 0 ]; then
                printf 'ERROR %-32s failed with no RED row (a broken script, not a kill)\n' "$name"; errors=$((errors + 1)); continue
            fi
            killed=$((killed + 1))
            printf 'killed   %-32s %s\n' "$name" "$reds"
        fi
    done <<< "$MUTANTS"
    rm -rf -- "${tmp:?}"
    printf 'MUTANTS killed=%s total=%s errors=%s\n' "$killed" "$total" "$errors"
    [ "$killed" -eq "$total" ] && [ "$errors" -eq 0 ]
}

# ------------------------------------------------------------------------------------ main ----
main() {
    local cmd repo=. remote=origin at=''
    [ -f "$LIB" ] || caller_error "no library at $LIB"
    # shellcheck source=../lib/nightly_pick.sh
    . "$LIB" || caller_error "cannot source $LIB"
    cmd="${1:-}"; [ "$#" -gt 0 ] && shift
    case "$cmd" in
        --self-test) self_test; return $? ;;
        --mutants) mutants; return $? ;;
        -h | --help) sed -n '2,20p' "$SCRIPT_PATH"; return 0 ;;
        checkout) ;;
        --repo | --remote | --at) set -- "$cmd" "$@" ;;
        *) caller_error "unknown command '$cmd' (checkout|--self-test|--mutants)" ;;
    esac
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --repo) repo="${2:-}"; shift 2 || caller_error "--repo DIR" ;;
            --remote) remote="${2:-}"; shift 2 || caller_error "--remote NAME" ;;
            --at) at="${2:-}"; shift 2 || caller_error "--at ISO8601|EPOCH" ;;
            *) caller_error "unknown option '$1'" ;;
        esac
    done
    [ -n "$at" ] || caller_error "--at ISO8601|EPOCH (the pick run's start) is required"
    checkout_c "$repo" "$remote" "$at"
}
main "$@"; exit $?
