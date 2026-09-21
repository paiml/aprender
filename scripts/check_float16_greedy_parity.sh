#!/usr/bin/env bash
# check_float16_greedy_parity.sh - the PR-time half of the #3076 greedy-parity gate.
#
# scripts/float16_greedy_parity.sh compares apr's F16/BF16 CPU matvec with the pinned
# llama.cpp. Its model-backed run needs the models and the llama.cpp build, which CI
# runners do not carry, so it runs at release time: it is declared in Cargo.toml
# [package.metadata.dogfood] and executed by `scripts/dogfood.sh --phase pre-publish`.
#
# This is what scripts/guard_tree.sh runs on every pull request: the case table of the
# comparison helpers both halves share (scripts/float16_parity_lib.sh). It needs no models
# and takes well under a second. Without it the comparison logic could rot between
# releases and still look wired. Exit 0 iff every row holds.
set -euo pipefail
# shellcheck source=float16_parity_lib.sh
. "$(dirname "$0")/float16_parity_lib.sh" || exit 2

case_table() {
    local fails=0 got
    # (a, b, expected first_diff)
    check() {
        got=$(first_diff "$1" "$2")
        if [ "$got" = "$3" ]; then
            printf 'ok    first_diff(%q, %q) = %s\n' "$1" "$2" "$got"
        else
            printf 'FAIL  first_diff(%q, %q) = %s, want %s\n' "$1" "$2" "$got" "$3"
            fails=$((fails + 1))
        fi
    }
    check "Berlin. The capital" "Berlin. The capital" -1
    check "Berlin. The capital" "Berlin. A capital" 8
    check "Berlin" "Berlin." 6
    check "" "" -1
    check "x" "" 0
    check 'print("a\\nb")' 'print("a\\nc")' 11
    local t
    t=$(trim $'  Berlin. The capital\n\n')
    if [ "$t" = "Berlin. The capital" ]; then printf 'ok    trim strips both ends\n'; else
        printf 'FAIL  trim gave %q\n' "$t"; fails=$((fails + 1)); fi
    t=$(trim $'a  b')
    if [ "$t" = $'a  b' ]; then printf 'ok    trim keeps inner whitespace\n'; else
        printf 'FAIL  trim changed inner whitespace: %q\n' "$t"; fails=$((fails + 1)); fi
    t=$(json_str $'a"b\\c\nd')
    if [ "$t" = '"a\"b\\c\nd"' ]; then printf 'ok    json_str escapes\n'; else
        printf 'FAIL  json_str gave %s\n' "$t"; fails=$((fails + 1)); fi
    printf -- '--- case table: %s failure(s) ---\n' "$fails"
    [ "$fails" -eq 0 ]
}

case_table
