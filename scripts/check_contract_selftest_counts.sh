#!/usr/bin/env bash
# check_contract_selftest_counts.sh -- a self-test size quoted in contract prose equals the size
# the self-test prints today (#2331: derive the numbers; do not quote them)
#
#   bash scripts/check_contract_selftest_counts.sh                  # report: print the verdict, exit 0
#   bash scripts/check_contract_selftest_counts.sh --mode enforce   # exit 0 match · 1 drift · 2 not measured
#   bash scripts/check_contract_selftest_counts.sh --self-test      # case table over fixtures, no self-test runs
#
# The contracts quote "67-row --self-test", "41 self-test rows", "17 mutants". No check read them,
# so they drifted: ci_mg_reuse.sh printed 68 rows while its contract said 67. Each row below names
# the contract, the literal text on either side of the number, and the command that derives it.
# A derivation reads its count ONLY from a self-test that exited 0 and reported zero failures: a
# red or unparseable self-test is NOT_MEASURED (L25), never a count and never a pass.
#
# MODE: report by default. This is a new check (L31); it is enforce-ready, not enforced.
# EXIT (enforce) 0 every count matches · 1 a count drifted · 2 not measured (an anchor moved, a
# self-test was red, or its output did not parse).
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=check_contract_selftest_counts

# CONTRACT|TEXT BEFORE THE NUMBER|TEXT AFTER IT ("" = end of line)|DERIVATION
ROWS=$(cat <<'EOF'
contracts/apr-required-checks-v1.yaml|(decide, resolve, |-row --self-test)|mg_rows
contracts/apr-required-checks-v1.yaml|(B1..B4, |-row self-test)|bo_rows
contracts/compound-ship-gates-v1.yaml|tag_coverage_gate.sh (| self-test rows)|tcg_rows
contracts/compound-ship-gates-v1.yaml|(wiring rows; | mutants)|tcg_mutants
contracts/compound-ship-gates-v1.yaml|dying rc, and each of its ||tcg_mutants
EOF
)

# quoted FILE BEFORE AFTER -> prints the number between BEFORE and AFTER; rc 2 unless exactly one
# line carries BEFORE<digits>AFTER (AFTER "" = the digits end the line).
quoted() {
    local f=$1
    [ -r "$f" ] || return 2
    B=$2 A=$3 awk 'BEGIN { b = ENVIRON["B"]; a = ENVIRON["A"]; n = 0 }
        {
            s = $0
            while ((i = index(s, b)) > 0) {
                r = substr(s, i + length(b))
                if (match(r, /^[0-9]+/)) {
                    num = substr(r, 1, RLENGTH); t = substr(r, RLENGTH + 1)
                    if ((a == "" && t ~ /^[[:space:]]*$/) || (a != "" && index(t, a) == 1)) { n++; v = num }
                }
                s = r
            }
        }
        END { if (n != 1) exit 2; print v }' "$f"
}

# Parsers: OUTPUT RC -> the count, or rc 2 (not measured). Pure; the self-test feeds them fixtures.
parse_mg_rows() {
    [ "$2" = 0 ] || return 2
    sed -nE 's/^ci_mg_reuse self-test: ([0-9]+) passed, 0 failed$/\1/p' <<< "$1" | grep -E '^[1-9][0-9]*$' || return 2
}
parse_bo_rows() {
    [ "$2" = 0 ] || return 2
    local p
    p=$(sed -nE 's/^SELF-TEST PASSED \(([0-9]+)\/([0-9]+) rows\)$/\1 \2/p' <<< "$1")
    [ -n "$p" ] && [ "${p% *}" = "${p#* }" ] && [ "${p% *}" -gt 0 ] || return 2
    echo "${p% *}"
}
parse_tcg_rows() {
    [ "$2" = 0 ] || return 2
    grep -q '^  FAIL' <<< "$1" && return 2
    local n
    n=$(grep -c '^  ok ' <<< "$1")
    [ "$n" -gt 0 ] || return 2
    echo "$n"
}
parse_tcg_mutants() {
    [ "$2" = 0 ] || return 2
    local p
    p=$(sed -nE 's/^  mutants killed ([0-9]+)\/([0-9]+)$/\1 \2/p' <<< "$1")
    [ -n "$p" ] && [ "${p% *}" = "${p#* }" ] && [ "${p% *}" -gt 0 ] || return 2
    echo "${p% *}"
}

# derive NAME -> runs the self-test once per NAME (cached) and prints its count; rc 2 not measured
declare -A CACHE=()
derive() {
    local name=$1 script out rc
    if [ -n "${CACHE[$name]+x}" ]; then [ -n "${CACHE[$name]}" ] || return 2; echo "${CACHE[$name]}"; return 0; fi
    case "$name" in
        mg_rows) script=scripts/ci_mg_reuse.sh ;;
        bo_rows) script=scripts/check_receipt_gate_base_owned.sh ;;
        tcg_rows) script=scripts/release/tag_coverage_gate.sh ;;
        tcg_mutants) script=scripts/check_tag_coverage_gated.sh ;;
        *) CACHE[$name]=""; return 2 ;;
    esac
    out=$(cd "$ROOT" && bash "$script" --self-test 2>&1); rc=$?
    CACHE[$name]=$("parse_$name" "$out" "$rc") || { CACHE[$name]=""; return 2; }
    echo "${CACHE[$name]}"
}

# judge ROOT DERIVE_FN -> prints one line per row; rc 0 match, 1 drift, 2 not measured (worst wins)
judge() {
    local root=$1 dfn=$2 worst=0 file before after name want have rc
    while IFS='|' read -r file before after name; do
        [ -n "$file" ] || continue
        have=$(quoted "$root/$file" "$before" "$after"); rc=$?
        if [ "$rc" -ne 0 ]; then
            echo "NOT_MEASURED  $file: not exactly one line reads '${before}<N>${after}' -- the anchor moved"
            worst=2; continue
        fi
        want=$("$dfn" "$name"); rc=$?
        if [ "$rc" -ne 0 ]; then
            echo "NOT_MEASURED  $file quotes $have; $name could not be derived (its self-test was red or did not parse)"
            worst=2; continue
        fi
        if [ "$have" = "$want" ]; then echo "ok    $file quotes $have = $name"
        else echo "DRIFT $file quotes ${before}${have}${after}; $name derives $want"; [ "$worst" -eq 2 ] || worst=1; fi
    done <<< "$ROWS"
    return "$worst"
}

self_test() {
    local d fail=0 got rc
    d=$(mktemp -d) || return 2
    mkdir -p "$d/good/contracts" "$d/stale/contracts" "$d/moved/contracts"
    printf '%s\n' "    - 'scripts/ci_mg_reuse.sh -- key (decide, resolve, 79-row --self-test), Refs #4675'" \
        "    - 'scripts/check_receipt_gate_base_owned.sh -- the resolver (B1..B4, 17-row self-test)'" \
        > "$d/good/contracts/apr-required-checks-v1.yaml"
    printf '%s\n' "      before the dryrun and cascade steps with a dying rc, and each of its 22" \
        "      - 'scripts/release/tag_coverage_gate.sh (50 self-test rows)'" \
        "      - 'scripts/check_tag_coverage_gated.sh (wiring rows; 22 mutants)'" \
        > "$d/good/contracts/compound-ship-gates-v1.yaml"
    cp "$d/good/contracts/"* "$d/stale/contracts/"; cp "$d/good/contracts/"* "$d/moved/contracts/"
    sed -i 's/79-row/67-row/' "$d/stale/contracts/apr-required-checks-v1.yaml"     # the #4820 drift, planted
    sed -i 's/(wiring rows; /(wiring rows, /' "$d/moved/contracts/compound-ship-gates-v1.yaml"
    fx() { case "$1" in mg_rows) echo 79 ;; bo_rows) echo 17 ;; tcg_rows) echo 50 ;; tcg_mutants) echo 22 ;; *) return 2 ;; esac; }
    fx_red() { [ "$1" = mg_rows ] && return 2; fx "$1"; }
    t() { # t NAME WANT_RC CMD... ; the command's rc must be WANT_RC
        local name=$1 want=$2; shift 2
        "$@" > /dev/null 2>&1; rc=$?
        if [ "$rc" = "$want" ]; then echo "  ok   $name -> rc $rc"; else echo "  FAIL $name: wanted rc $want, got $rc"; fail=1; fi
    }
    v() { # v NAME WANT CMD... ; the command must print WANT (and rc 0), or WANT=- means rc 2
        local name=$1 want=$2; shift 2
        got=$("$@" 2>/dev/null); rc=$?
        if { [ "$want" = - ] && [ "$rc" = 2 ]; } || { [ "$rc" = 0 ] && [ "$got" = "$want" ]; }; then echo "  ok   $name"
        else echo "  FAIL $name: wanted '$want', got '$got' rc $rc"; fail=1; fi
    }
    echo "$PROG self-test: judge"
    t "every quoted count matches its derivation" 0 judge "$d/good" fx
    t "MUST-FAIL: a stale count (67 quoted, 79 derived) is DRIFT" 1 judge "$d/stale" fx
    t "a moved anchor is NOT_MEASURED, never ok" 2 judge "$d/moved" fx
    t "a red self-test is NOT_MEASURED, never ok" 2 judge "$d/good" fx_red
    t "a missing contract is NOT_MEASURED" 2 judge "$d/none" fx
    echo "$PROG self-test: quoted"
    v "number between the anchors" 79 quoted "$d/good/contracts/apr-required-checks-v1.yaml" "(decide, resolve, " "-row --self-test)"
    v "an empty AFTER means the number ends the line" 22 quoted "$d/good/contracts/compound-ship-gates-v1.yaml" "dying rc, and each of its " ""
    printf '%s\n' "x (B1..B4, 17-row self-test)" "y (B1..B4, 18-row self-test)" > "$d/twice.yaml"
    v "two lines carrying the anchor are ambiguous" - quoted "$d/twice.yaml" "(B1..B4, " "-row self-test)"
    printf '%s\n' "x (B1..B4, 17 rows)" > "$d/wrongsuffix.yaml"
    v "the number must be followed by AFTER" - quoted "$d/wrongsuffix.yaml" "(B1..B4, " "-row self-test)"
    echo "$PROG self-test: parsers"
    v "mg: green line" 79 parse_mg_rows "ci_mg_reuse self-test: 79 passed, 0 failed" 0
    v "mg: a failure is not a count" - parse_mg_rows "ci_mg_reuse self-test: 78 passed, 1 failed" 1
    v "mg: rc!=0 is not a count even with a green line" - parse_mg_rows "ci_mg_reuse self-test: 79 passed, 0 failed" 1
    v "mg: a failed row is not a count even at rc 0" - parse_mg_rows "ci_mg_reuse self-test: 78 passed, 1 failed" 0
    v "bo: green line" 17 parse_bo_rows "SELF-TEST PASSED (17/17 rows)" 0
    v "bo: partial is not a count" - parse_bo_rows "SELF-TEST PASSED (16/17 rows)" 0
    v "tcg rows: ok lines counted" 2 parse_tcg_rows "$(printf '  ok   a\n  ok   b\nsummary')" 0
    v "tcg rows: a FAIL line is not a count" - parse_tcg_rows "$(printf '  ok   a\n  FAIL b')" 0
    v "tcg rows: zero ok lines is not a count" - parse_tcg_rows "nothing ran" 0
    v "tcg mutants: all killed" 22 parse_tcg_mutants "  mutants killed 22/22" 0
    v "tcg mutants: a survivor is not a count" - parse_tcg_mutants "  mutants killed 21/22" 0
    rm -rf -- "${d:?}"
    [ "$fail" -eq 0 ] && { echo "$PROG self-test: PASS"; return 0; }
    echo "$PROG self-test: FAIL"; return 1
}

mode=report
case "${1:-}" in
    --self-test) self_test; exit $? ;;
    --mode) [ "${2:-}" = report ] || [ "${2:-}" = enforce ] || { echo "usage: $0 [--mode report|enforce | --self-test]" >&2; exit 2; }; mode=$2 ;;
    '') ;;
    *) echo "usage: $0 [--mode report|enforce | --self-test]" >&2; exit 2 ;;
esac
judge "$ROOT" derive; rc=$?
case "$rc" in 0) v=PASS ;; 1) v=DRIFT ;; *) v=NOT_MEASURED ;; esac
echo "$PROG: $v (mode $mode)"
[ "$mode" = enforce ] && exit "$rc"
exit 0
