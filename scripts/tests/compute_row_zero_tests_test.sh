#!/usr/bin/env bash
# compute_row_zero_tests_test.sh -- case table for the compute-tests pass rule in
# ci/sections.yml (workspace-test-shard, #5048).
#
# The row runs the compute crate's lib tests through `tee` and judges the log, not
# the test runner's status (the step tolerates a SIGSEGV at harness exit). The old
# rule took any `test result: ok.` line, so a run of 0 tests passed: a cfg or
# feature gate that removed every test left the row green with nothing tested.
#
# This script reads the rule out of the file (never a copy of it), replays it on
# planted logs, and requires each row's verdict:
#
#   zero_tests_is_red          ok. 0 passed                       -> red
#   tests_passed_is_green      ok. 1873 passed                    -> green
#   ten_passed_is_green        ok. 10 passed (a 0 after a digit)  -> green
#   failed_is_red              FAILED line                        -> red
#   crash_without_result_red   no result line at all              -> red
#   segv_after_pass_is_green   ok line, then a SIGSEGV message    -> green
#
# Usage:
#   compute_row_zero_tests_test.sh                 rows against this tree
#   compute_row_zero_tests_test.sh --rule-from F   rows against the rule in file F
#   compute_row_zero_tests_test.sh --mutants       the planted RED: the pre-#5048 rule
#                                                  must break zero_tests_is_red only
set -euo pipefail

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
LOG_PATH=/tmp/compute-test.log

# rule_of FILE -> the pass rule: the text after `tee /tmp/compute-test.log; ` on the
# compute row, without the closing quote. Exactly one row, or a non-zero return.
rule_of() {
    local n
    n=$(grep -c "tee $LOG_PATH; " "$1" || true)
    [ "$n" -eq 1 ] || { printf 'FAIL: %s compute rows in %s, not 1\n' "$n" "$1" >&2; return 1; }
    grep "tee $LOG_PATH; " "$1" | sed "s|.*tee $LOG_PATH; ||; s|'[[:space:]]*\$||"
}

# verdict RULE LOG -> green or red: the rule run on LOG in place of the CI log path
verdict() {
    local rule=${1//$LOG_PATH/$2}
    if bash -c "$rule"; then echo green; else echo red; fi
}

# table RULE -> one line per broken row; returns the count of broken rows
table() {
    local rule=$1 dir fail=0 name want body got
    dir=$(mktemp -d)
    while IFS='|' read -r name want body; do
        printf '%b' "$body" > "$dir/$name.log"
        got=$(verdict "$rule" "$dir/$name.log")
        if [ "$got" != "$want" ]; then
            printf '  BROKE %-28s want %s, got %s\n' "$name" "$want" "$got"
            fail=$((fail + 1))
        fi
    done <<'ROWS'
zero_tests_is_red|red|running 0 tests\n\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n
tests_passed_is_green|green|running 1873 tests\n\ntest result: ok. 1873 passed; 0 failed; 4 ignored; 0 measured; 0 filtered out; finished in 9.12s\n
ten_passed_is_green|green|test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n
failed_is_red|red|test a::b ... FAILED\n\ntest result: FAILED. 1872 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n
crash_without_result_red|red|running 1873 tests\nerror: test failed, to rerun pass `--lib`\n
segv_after_pass_is_green|green|test result: ok. 1873 passed; 0 failed; 4 ignored; 0 measured; 0 filtered out\nerror: process didn't exit successfully (signal: 11, SIGSEGV: invalid memory reference)\n
ROWS
    rm -rf "${dir:?}"
    return "$fail"
}

case "${1:-}" in
    --mutants)
        # The planted RED: the rule as it stood before #5048. It must break exactly
        # the zero-tests row, or this table cannot tell the two rules apart.
        old='grep -q "test result: ok\." /tmp/compute-test.log && ! grep -q "test result: FAILED" /tmp/compute-test.log'
        if out=$(table "$old"); then echo "SURVIVED: the pre-#5048 rule passes every row"; exit 1; fi
        if [ "$(grep -c BROKE <<<"$out")" -eq 1 ] && grep -q 'BROKE zero_tests_is_red' <<<"$out"; then
            echo "KILLED: the pre-#5048 rule breaks zero_tests_is_red and nothing else"
        else
            printf 'UNEXPECTED: the pre-#5048 rule broke:\n%s\n' "$out"; exit 1
        fi
        ;;
    --rule-from)
        rule=$(rule_of "${2:?--rule-from needs a file}")
        if out=$(table "$rule"); then echo "6/6 rows hold on $2"; exit 0; fi
        printf '%s\n' "$out"; exit 1
        ;;
    '')
        rule=$(rule_of "$ROOT/ci/sections.yml")
        if out=$(table "$rule"); then echo "6/6 rows hold on ci/sections.yml"; exit 0; fi
        printf '%s\n' "$out"; exit 1
        ;;
    *) printf 'usage: %s [--rule-from FILE | --mutants]\n' "$0" >&2; exit 2 ;;
esac
