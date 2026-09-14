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

# --- 5. a parse error is a reject, never a silent skip ---------------------------------
mk_records "$TMP/bad" 20
printf 'not json at all\n' > "$TMP/bad/broken.json"
set +e
"$RPT" --ledger "$TMP/bad" >/dev/null 2>"$TMP/bad.err"; rc_bad=$?
set -e
chk "one malformed record exits 1" "1" "$rc_bad"
chk "malformed record says reject:" "yes" \
    "$(grep -qE '^reject:' "$TMP/bad.err" && echo yes || echo no)"

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
