#!/usr/bin/env bash
# check_build_report.sh — the contract of `make build-report` (APR-RELEASE-001 §5 P0·Instrument).
#
# 1092 ledger records are committed under docs/build-ledger/ in the P0·Instrument schema
# (sha host host_class job queue_wait_s exec_s total_s peak_rss_mb free_disk_gb exit) and
# nothing in the tree reads them, so every number §1 and §7 derive from them is [U]:
# p95 `ci / gate`, queue p95 per host, and `max PRs per train`. This guard asserts the
# reader exists, is wired to a make target, and DECLINES rather than inventing numbers
# when the ledger is too thin to support them.
#
# Why a guard and not only the script's own --self-test: a facility with a self-test and
# no caller is not measured by anything (the class that shipped `check_package_includes.sh`
# vacuous). This runs from `make lint-scripts`, so the self-test has a caller.
#
# exit: 0 all assertions hold · 1 an assertion failed · 2 usage, or a tool is missing
set -euo pipefail

PROG=${0##*/}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
FAIL=0
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

chk() { # chk <name> <expected> <actual>
    if [ "$2" = "$3" ]; then
        printf '  ok   %-46s %s\n' "$1" "$3"
    else
        printf '  FAIL %-46s want=%s got=%s\n' "$1" "$2" "$3"
        FAIL=1
    fi
}

command -v jq >/dev/null 2>&1 || { printf '%s: jq is required\n' "$PROG" >&2; exit 2; }

RPT="$ROOT/scripts/build_report.sh"
printf '== %s ==\n' "$PROG"

# --- 1. the reader exists and is executable -------------------------------------------
[ -f "$RPT" ] || { printf '  FAIL %-46s scripts/build_report.sh is absent\n' "reader exists"; exit 1; }
chk "reader is executable" "yes" "$([ -x "$RPT" ] && echo yes || echo no)"

# --- 2. it has a caller: a make target ------------------------------------------------
chk "make target build-report declared" "yes" \
    "$(grep -qE '^build-report:' "$ROOT/Makefile" && echo yes || echo no)"
chk "build-report in .PHONY" "yes" \
    "$(grep -qE '^\.PHONY:.*\bbuild-report\b' "$ROOT/Makefile" && echo yes || echo no)"
chk "self-test has a caller (lint-scripts)" "yes" \
    "$(grep -qE 'check_build_report\.sh' "$ROOT/Makefile" && echo yes || echo no)"

# --- 3. the vacuity floor: it declines instead of reporting zeros ----------------------
# §8: "Fewer than 20 ledger records -> stop at P0". A report over an empty ledger that
# prints 0 reads exactly like a fast fleet.
mkdir -p "$TMP/empty"
set +e
"$RPT" --ledger "$TMP/empty" >"$TMP/empty.out" 2>"$TMP/empty.err"; rc_empty=$?
set -e
chk "empty ledger exits 2" "2" "$rc_empty"
chk "empty ledger says decline:" "yes" \
    "$(grep -qE '^decline:' "$TMP/empty.err" && echo yes || echo no)"

# 19 records is still a decline; 20 is the floor and passes.
mk_records() { # mk_records <dir> <n>
    mkdir -p "$1"
    _i=0
    while [ "$_i" -lt "$2" ]; do
        printf '{"spec":"APR-RELEASE-001","sha":"deadbeef%03d","host":"intel-w1","host_class":"intel","job":"ci / gate","queue_wait_s":%d,"exec_s":%d,"total_s":%d,"exit":0}\n' \
            "$_i" "$_i" "$((_i * 2))" "$((_i * 3))" > "$1/rec-$_i.json"
        _i=$((_i + 1))
    done
}
mk_records "$TMP/n19" 19
set +e
"$RPT" --ledger "$TMP/n19" >/dev/null 2>"$TMP/n19.err"; rc19=$?
set -e
chk "19 records exits 2 (floor is 20)" "2" "$rc19"

mk_records "$TMP/n20" 20
set +e
"$RPT" --ledger "$TMP/n20" >"$TMP/n20.out" 2>&1; rc20=$?
set -e
chk "20 records exits 0" "0" "$rc20"

# --- 4. it answers the questions §1 and §7 ask ----------------------------------------
chk "prints a p95 total_s" "yes" \
    "$(grep -qiE 'p95' "$TMP/n20.out" && echo yes || echo no)"
chk "prints queue_wait p95" "yes" \
    "$(grep -qiE 'queue' "$TMP/n20.out" && echo yes || echo no)"
chk "prints the §1 max PRs/train number" "yes" \
    "$(grep -qiE 'PRs/train|prs_per_train' "$TMP/n20.out" && echo yes || echo no)"
chk "emits the §7 gate: line" "yes" \
    "$(grep -qE '^gate: ' "$TMP/n20.out" && echo yes || echo no)"

# --- 4b. the bound must come from the SLOWEST required check, not from `ci / gate` -----
# Required checks on main are `gate` AND `workspace-test` (branch protection). A PR merges
# when the SLOWEST of them is green, so a throughput bound keyed on `ci / gate` alone
# overstates it — measured 5.4x on the committed ledger (p95 1205 s vs 6451 s).
mk_two() { # mk_two <dir>: 20 fast `ci / gate` + 20 slow `workspace-test`
    mkdir -p "$1"
    _i=0
    while [ "$_i" -lt 20 ]; do
        printf '{"sha":"a%03d","host":"intel-w1","host_class":"intel","job":"ci / gate","queue_wait_s":1,"exec_s":100,"total_s":100,"exit":0}\n' \
            "$_i" > "$1/fast-$_i.json"
        printf '{"sha":"b%03d","host":"intel-w1","host_class":"intel","job":"workspace-test","queue_wait_s":1,"exec_s":1000,"total_s":1000,"exit":0}\n' \
            "$_i" > "$1/slow-$_i.json"
        _i=$((_i + 1))
    done
}
mk_two "$TMP/two"
# The binding-check expectation below must not depend on what GitHub says
# today: pin the required set for this row (the derive-at-run-time path has
# its own rows in 5f, including the hidden-gh fallback).
export BUILD_REPORT_REQUIRED_CHECKS='["ci / gate","gate","workspace-test"]'
set +e
"$RPT" --ledger "$TMP/two" >"$TMP/two.out" 2>&1
set -e
# 3 x 72h / 1000 s = 777 (workspace-test); keyed on ci / gate it would be 7776.
chk "bound uses the slowest required check" "777" \
    "$(sed -n 's|.*max PRs/train \([0-9][0-9]*\).*|\1|p' "$TMP/two.out" | head -1)"
chk "names which check binds" "yes" \
    "$(grep -qiE 'binding|slowest required' "$TMP/two.out" && echo yes || echo no)"

# --- 5. a parse error is a reject, never a silent skip ---------------------------------
mk_records "$TMP/bad" 20
printf 'not json at all\n' > "$TMP/bad/broken.json"
set +e
"$RPT" --ledger "$TMP/bad" >/dev/null 2>"$TMP/bad.err"; rc_bad=$?
set -e
chk "one malformed record exits 1" "1" "$rc_bad"
chk "malformed record says reject:" "yes" \
    "$(grep -qE '^reject:' "$TMP/bad.err" && echo yes || echo no)"

# --- 5b. quorum findings (3/3 lanes, grounding=measured): a JSON value that is not an
#         object, and an empty/whitespace file, must hit the SAME contract as bad text.
#         jq crashes with exit 5 on has() over a non-object; empty files emit nothing and
#         were silently ignored. Both looked like a pass.
for _shape in '[]' '123' '"str"' 'true' 'null' '' '   '; do
    _dir="$TMP/shape-$(printf '%s' "$_shape" | tr -c 'a-z0-9' '_' | cut -c1-8)"
    mk_records "$_dir" 20
    printf '%s' "$_shape" > "$_dir/odd.json"
    set +e
    "$RPT" --ledger "$_dir" >/dev/null 2>"$_dir.err"; _rc=$?
    set -e
    chk "non-object/empty [$_shape] exits 1" "1" "$_rc"
    chk "non-object/empty [$_shape] says reject:" "yes" \
        "$(grep -qE '^reject:' "$_dir.err" && echo yes || echo no)"
done

# --- 5c. percentile precision: ceil(k/100*n) in floating point. k=7,n=100 must be 7,
#         not 8 (ceil(7.000000000000001)). The k=50/95/100 table does not reach this.
set +e
"$RPT" --self-test --percentile-probe 2>"$TMP/pp.err" >"$TMP/pp.out"; _rcpp=$?
set -e
chk "--percentile-probe exits 0" "0" "$_rcpp"
chk "p7 of 1..100 is 7, not 8" "7" \
    "$(sed -n 's/^p7=\([0-9][0-9]*\)$/\1/p' "$TMP/pp.out" | head -1)"

# --- 5d. both spellings of the gate required check are one check
#         (scripts/pr_review_quorum_arm.sh: branch protection names `ci / gate`,
#          ruleset 13878864 names a bare `gate`; both spellings are accepted there)
# Behavioural, not a grep of the script's own source (that row was vacuous by
# construction — quorum round 3 on #3271, lane 1, measured): a ledger recording
# the gate under BOTH spellings must measure both under the pinned set.
mk_spellings() {
    mkdir -p "$1"
    _i=0
    while [ "$_i" -lt 20 ]; do
        printf '{"sha":"g%02d","host":"intel-w1","host_class":"intel","job":"gate","queue_wait_s":1,"exec_s":5,"total_s":6,"exit":0}\n' "$_i" > "$1/bare-$_i.json"
        printf '{"sha":"c%02d","host":"intel-w1","host_class":"intel","job":"ci / gate","queue_wait_s":1,"exec_s":7,"total_s":8,"exit":0}\n' "$_i" > "$1/ci-$_i.json"
        _i=$((_i + 1))
    done
}
mk_spellings "$TMP/spellings"
set +e
BUILD_REPORT_REQUIRED_CHECKS='["ci / gate","gate","workspace-test"]' "$RPT" --ledger "$TMP/spellings" --format json >"$TMP/spellings.json" 2>/dev/null
set -e
chk "both gate spellings are MEASURED under the required set" "ci / gate,gate" \
    "$(jq -r '[.required_measured[].job] | sort | join(",")' "$TMP/spellings.json" 2>/dev/null)"
chk "the slower spelling binds" "ci / gate" \
    "$(jq -r '.binding_job' "$TMP/spellings.json" 2>/dev/null)"

# --- 5e. host class is DERIVED from the runner-name prefix when a record lacks it
#         (§5 amended 2026-09-21, ruling on #3271 round 2: the fleet is managed
#          per class; the ledger has no per-host field worth grouping on)
mk_hosts() { # records WITHOUT host_class: five declared prefixes, one unknown, one null host
    mkdir -p "$1"
    _i=0
    for _h in intel-clean-room-13 gx10-build yoga-build-2 lambda-vector mini-m4 mystery-host-9; do
        _j=0
        while [ "$_j" -lt 4 ]; do
            printf '{"sha":"h%s%02d","host":"%s","job":"ci / gate","queue_wait_s":1,"exec_s":10,"total_s":11,"exit":0}\n' \
                "$_i" "$_j" "$_h" > "$1/h-$_i-$_j.json"
            _j=$((_j + 1))
        done
        _i=$((_i + 1))
    done
    # a record with no host at all
    printf '{"sha":"nohost","job":"ci / gate","queue_wait_s":1,"exec_s":10,"total_s":11,"exit":0}\n' > "$1/nohost.json"
}
mk_hosts "$TMP/hosts"
set +e
"$RPT" --ledger "$TMP/hosts" --format json >"$TMP/hosts.json" 2>"$TMP/hosts.err"; rc_hosts=$?
set -e
chk "records without host_class still report (rc 0)" "0" "$rc_hosts"
_classes=$(jq -r '[.host_table[].host_class] | sort | join(",")' "$TMP/hosts.json" 2>/dev/null)
chk "prefixes intel/gx10/yoga/lambda/mini classified" "gx10,intel,lambda,mini,other,yoga" "$_classes"
_other_n=$(jq -r '.host_table[] | select(.host_class=="other") | .n' "$TMP/hosts.json" 2>/dev/null)
chk "unknown prefix AND null host both land in other" "5" "$_other_n"

# --- 5f. the required-check SET is derived at run time, and the report SAYS where it came from
#         (§5 amended 2026-09-21: two mechanisms answer "what is required"; one name is the
#          one-mechanism error)
_src=$(jq -r '.required_source' "$TMP/hosts.json")
chk "required_source is pinned/derived/fallback, never absent" "yes" \
    "$(printf '%s' "$_src" | grep -qE '^(pinned \(BUILD_REPORT_REQUIRED_CHECKS\)|derived|fallback \(.+\))$' && echo yes || echo no)"
# And the derive path itself, un-pinned: derived when gh answers, else a named fallback.
set +e
env -u BUILD_REPORT_REQUIRED_CHECKS "$RPT" --ledger "$TMP/hosts" --format json >"$TMP/derive.json" 2>/dev/null
set -e
chk "un-pinned, the source is derived or a named fallback" "yes" \
    "$(jq -r '.required_source' "$TMP/derive.json" 2>/dev/null | grep -qE '^(derived|fallback \(.+\))$' && echo yes || echo no)"
chk "text report prints the required set with its source" "yes" \
    "$(grep -qE 'required set: .+ \[(pinned \(BUILD_REPORT_REQUIRED_CHECKS\)|derived|fallback \(.+\))\]' "$TMP/two.out" && echo yes || echo no)"
# The fallback must NAME its reason. Hide gh and look for it.
_nogh=$(mktemp -d "${TMPDIR:-/tmp}/nogh.XXXXXX")
for _t in bash sh jq sed grep sort cat mktemp rm cp printf head tail wc date tr awk dirname basename uniq ls cut xargs env; do
    _p=$(command -v "$_t" 2>/dev/null) && ln -sf "$_p" "$_nogh/$_t"
done
[ -x /usr/bin/find ] && ln -sf /usr/bin/find "$_nogh/find"
set +e
env -i PATH="$_nogh" HOME="${HOME:-/tmp}" TMPDIR="$TMP" bash "$RPT" --ledger "$TMP/two" --format json >"$TMP/nogh.json" 2>"$TMP/nogh.err"; rc_nogh=$?   # env -i: no pin reaches it
set -e
chk "without gh the report still runs (rc 0)" "0" "$rc_nogh"
chk "without gh the source names the reason" "fallback (gh not on PATH)" \
    "$(jq -r '.required_source' "$TMP/nogh.json" 2>/dev/null)"
rm -rf "${_nogh:?}"

# --- 6. the script's own case table runs ----------------------------------------------
set +e
"$RPT" --self-test >"$TMP/st.out" 2>&1; rc_st=$?
set -e
chk "--self-test exits 0" "0" "$rc_st"

# --- 7. the committed ledger is above the floor and reports ---------------------------
set +e
"$RPT" --ledger "$ROOT/docs/build-ledger" >"$TMP/real.out" 2>"$TMP/real.err"; rc_real=$?
set -e
chk "committed ledger exits 0" "0" "$rc_real"

if [ "$FAIL" -eq 0 ]; then
    printf '%s: PASS\n' "$PROG"
else
    printf '%s: FAIL\n' "$PROG" >&2
fi
exit "$FAIL"
