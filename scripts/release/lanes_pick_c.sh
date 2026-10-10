#!/usr/bin/env bash
# lanes_pick_c.sh — C, the commit release-lanes-nightly measures (#5082 hit 5)
#
#   bash scripts/release/lanes_pick_c.sh c REPO REF EVENT SHA   # C on stdout
#   bash scripts/release/lanes_pick_c.sh --self-test | --mutants
#
# THE DEFECT (#5082). A scheduled release-lanes run measured `github.sha`, main's head when GitHub created the run.
# The train judges the commit tonight's scheduled nightly.yml measured, and GitHub creates the two runs minutes to
# hours late and not in cron order, so on 2026-10-07 the lanes measured 4098007133 while the train's night measured
# a719e1aa28: green lanes for a commit nobody shipped.
#
# THE PICK. REF set (a caller passed C) -> REF. EVENT schedule -> the head sha of the newest scheduled nightly.yml run
# on main, if GitHub created it in the last MAX_AGE_H hours (tonight's, not last night's). Any other event (a
# dispatch) -> SHA. A schedule with no such run, or a failed read, prints not_measured and exits 2: the schedule
# path never falls back to SHA. One API call, no poll.
#
# EXIT  c: 0 C printed · 2 not_measured · 3 a caller error. --self-test / --mutants: 0 green, 1 red.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$HERE/$(basename -- "${BASH_SOURCE[0]}")"
MAX_AGE_H=12            # the crons are 17 min apart and nightly.yml runs every 24 h: 12 h tells tonight from last night

caller_error() { printf 'lanes_pick_c: %s\n' "$1" >&2; exit 3; }
not_measured() { printf 'not_measured: %s\n' "$1" >&2; return 2; }

# tonight REPO -> the head sha of tonight's scheduled nightly.yml run on main, rc 0; else rc 2
tonight() {
    local row sha created now age
    row="$(gh api "repos/$1/actions/workflows/nightly.yml/runs?event=schedule&branch=main&per_page=1" \
        --jq '.workflow_runs[0] // empty | "\(.head_sha) \(.created_at | fromdateiso8601)"' 2>/dev/null)" \
        || { not_measured "the nightly.yml run list could not be read"; return; }
    [ -n "$row" ] || { not_measured "nightly.yml has no scheduled run on main"; return; }
    sha=${row%% *}; created=${row#* }
    [[ $sha =~ ^[0-9a-f]{40}$ ]] || { not_measured "nightly.yml's newest run names no full sha: ${sha:0:60}"; return; }
    [[ $created =~ ^[0-9]+$ ]] || { not_measured "nightly.yml's newest run has no readable created_at"; return; }
    now="${LANES_NOW:-$EPOCHSECONDS}"   # bash 5; created_at is parsed by gh's jq: no date call (bashrs DET002)
    age=$((now - created))
    [ "$age" -ge 0 ] && [ "$age" -le $((MAX_AGE_H * 3600)) ] \
        || { not_measured "nightly.yml's newest scheduled run is not tonight's (created ${age}s ago, $sha)"; return; }
    printf '%s\n' "$sha"
}

# pick_c REPO REF EVENT SHA -> C on stdout
pick_c() {
    local repo=$1 ref=$2 event=$3 sha=$4
    if [ -n "$ref" ]; then
        [[ $ref =~ ^[0-9a-f]{40}$ ]] || caller_error "REF is not a 40-hex sha: ${ref:0:60}"
        printf '%s\n' "$ref"; return 0
    fi
    if [ "$event" = schedule ]; then tonight "$repo"; return; fi
    [[ $sha =~ ^[0-9a-f]{40}$ ]] || caller_error "SHA is not a 40-hex sha: ${sha:0:60}"
    printf '%s\n' "$sha"
}

# ---------------------------------------------------------------- the case table ---------------------------------
self_test() {
    local tmp pass=0 fail=0 now=2000000000 head night old
    tmp="$(mktemp -d)" || return 1
    head=$(printf 'a%.0s' {1..40}); night=$(printf 'b%.0s' {1..40}); old=$(printf 'c%.0s' {1..40})
    mkdir -p "$tmp/bin"
    # the fixture gh: prints $FX_ROW as the --jq result, then exits FX_RC (a read that fails part way)
    printf '#!/usr/bin/env bash\nprintf "%%s" "$FX_ROW"\nexit "${FX_RC:-0}"\n' > "$tmp/bin/gh"
    chmod +x "$tmp/bin/gh"
    at() { printf '%s' "$1"; }   # created_at as gh's jq prints it: epoch seconds (fromdateiso8601)
    # case_ NAME WANT_RC WANT_STDOUT WANT_STDERR ROW FX_RC REF EVENT SHA : WANT_STDERR is a substring ("" = any)
    case_() {
        local name=$1 want_rc=$2 want=$3 err=$4 o rc=0
        o="$(PATH="$tmp/bin:$PATH" FX_ROW="$5" FX_RC="$6" LANES_NOW=$now bash "$SCRIPT_PATH" c paiml/aprender "$7" "$8" "$9" 2> "$tmp/err")" || rc=$?
        if [ "$rc" = "$want_rc" ] && [ "$o" = "$want" ] && { [ -z "$err" ] || grep -qF -- "$err" "$tmp/err"; }; then printf '  ok    %s\n' "$name"; pass=$((pass + 1))
        else printf '  BROKE %-46s rc %s stdout [%s] stderr [%s], wanted rc %s [%s] [%s]\n' "$name" "$rc" "$o" "$(head -c 120 "$tmp/err")" "$want_rc" "$want" "$err"; fail=$((fail + 1)); fi
    }
    # late start: main moved after nightly.yml was created, so github.sha is a later head; C is the train's commit
    case_ late_start_measures_the_nightlys_commit 0 "$night" "" "$night $(at $((now - 7 * 3600)))" "" "" schedule "$head"
    case_ an_on_time_night_measures_the_nightlys_commit 0 "$night" "" "$night $(at $((now - 600)))" "" "" schedule "$head"
    case_ lanes_created_before_nightly_is_not_measured 2 "" "is not tonight's" "$old $(at $((now - 24 * 3600)))" "" "" schedule "$head"
    case_ no_scheduled_nightly_is_not_measured 2 "" "has no scheduled run" "" "" "" schedule "$head"
    case_ a_failed_read_is_not_measured 2 "" "could not be read" "$night $(at $((now - 600)))" 1 "" schedule "$head"
    case_ a_short_sha_is_not_measured 2 "" "names no full sha" "${night:0:10} $(at $((now - 600)))" "" "" schedule "$head"
    case_ a_run_dated_after_now_is_not_measured 2 "" "is not tonight's" "$night $(at $((now + 600)))" "" "" schedule "$head"
    case_ an_unreadable_date_is_not_measured 2 "" "no readable created_at" "$night yesterday-ish" "" "" schedule "$head"
    case_ a_callers_ref_is_c 0 "$old" "" "$night $(at $((now - 600)))" "" "$old" schedule "$head"
    case_ a_short_ref_is_a_caller_error 3 "" "REF is not" "" "" "${old:0:10}" schedule "$head"
    case_ a_dispatch_measures_its_sha 0 "$head" "" "" "" "" workflow_dispatch "$head"
    case_ a_dispatch_with_no_sha_is_a_caller_error 3 "" "SHA is not" "" "" "" workflow_dispatch ""
    rm -rf -- "${tmp:?}"
    printf -- '--- %s/%s rows ---\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ]
}

# each row: NAME SED-EXPR applied to this script; the case table must go red
LP_MUTANTS='m1_schedule_falls_back_to_the_head s/if \[ "\$event" = schedule \]; then tonight "\$repo"; return; fi/if [ "$event" = schedule ]; then tonight "$repo" || printf "%s\\n" "$sha"; return 0; fi/
m2_schedule_takes_the_head s/if \[ "\$event" = schedule \]; then tonight "\$repo"; return; fi/:/
m3_last_nights_run_accepted s/\[ "\$age" -le \$((MAX_AGE_H \* 3600)) \]/true/
m4_future_run_accepted s/\[ "\$age" -ge 0 \] \&\& //
m5_short_sha_accepted s/{40}\$ \]\] || { not_measured "nightly/{1,40}$ ]] || { not_measured "nightly/
m6_failed_read_ignored s/2>\/dev\/null)" \\$/2>\/dev\/null || true)" \\/
m7_ref_ignored s/^    if \[ -n "\$ref" \]; then$/    if false; then/
m8_empty_run_list_passes s/\[ -n "\$row" \] || { not_measured/true || { not_measured/'

mutants() {
    local tmp pass=0 fail=0 name expr o rc m
    tmp="$(mktemp -d)" || return 1
    m="$tmp/lanes_pick_c.sh"
    cp "$SCRIPT_PATH" "$m"
    if ! bash "$m" --self-test > /dev/null 2>&1; then printf '  BROKE %-46s the unmutated script is not green in the mutant dir\n' baseline; rm -rf -- "${tmp:?}"; return 1; fi
    while read -r name expr; do
        [ -n "$name" ] || continue
        sed -e "$expr" "$SCRIPT_PATH" > "$m"
        if cmp -s "$SCRIPT_PATH" "$m"; then printf '  BROKE %-46s changed nothing: its pattern no longer matches\n' "$name"; fail=$((fail + 1)); continue; fi
        if ! bash -n "$m" 2>/dev/null; then printf '  BROKE %-46s does not parse\n' "$name"; fail=$((fail + 1)); continue; fi
        o="$(bash "$m" --self-test 2>&1)"; rc=$?
        case "$rc:$o" in
            0:*) printf '  BROKE %-46s SURVIVED: the case table stayed green\n' "$name"; fail=$((fail + 1)) ;;
            *"  BROKE "*) printf '  ok    %-46s killed\n' "$name"; pass=$((pass + 1)) ;;
            *) printf '  BROKE %-46s exit %s with no broken row: not a kill\n' "$name" "$rc"; fail=$((fail + 1)) ;;
        esac
    done < <(printf '%s\n' "$LP_MUTANTS")
    printf -- '--- %s/%s mutants killed ---\n' "$pass" "$((pass + fail))"
    rm -rf -- "${tmp:?}"
    [ "$fail" -eq 0 ]
}

case "${1:-}" in
    --self-test) self_test; exit $? ;;
    --mutants) mutants; exit $? ;;
    c) [ $# -eq 5 ] || caller_error "usage: c REPO REF EVENT SHA"; pick_c "$2" "$3" "$4" "$5"; exit $? ;;
    *) caller_error "usage: c REPO REF EVENT SHA | --self-test | --mutants" ;;
esac
