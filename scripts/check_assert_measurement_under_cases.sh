#!/usr/bin/env bash
# check_assert_measurement_under_cases.sh - the must-match / must-not-match table
# for scripts/assert_measurement_under.sh, RUN rather than read.
#
# EXECUTED, NEVER SOURCED. It is invoked as a command
# (bash scripts/check_assert_measurement_under_cases.sh), so the option-neutrality
# rule that governs scripts/apr_bin.sh - a set inside a SOURCED file mutates the
# CALLER shell, which is what scripts/check_sourced_libs_option_neutral.sh exists
# to catch - does not apply here. The set -euo pipefail below governs this
# script own shell and no other.
#
# WHY A TABLE. CLAUDE.md Verification Discipline rule 7: guard patterns ship a
# table. The apr-invocation patterns in this repository were wrong five times;
# every one of those was caught by a must-match/must-not-match table and NONE by
# review. The defect this validator replaces (06-REVIEW.md IN-01) is exactly of
# that shape: the inline awk bar check coerced a non-numeric token to 0 with
# "v + 0", and 0 is under every bar this repository asserts - so a total_s token
# of abc, or an empty one, printed OK. A gate that cannot fail is theater. The
# MUST_FAIL rows below are the ones that coercion let through.
#
# MEASURED, not argued: driven against the naive coercion extracted verbatim from
# the four pre-existing bar sites, 7 of these 23 rows behaved the other way -
# empty, abc, 1.2.3, a lone dot, -1, two tokens, and a trailing carriage return.
#
# just forecast-sc1-sweep runs this table BEFORE it measures anything, so the
# validator guard is exercised on every gate invocation and not only when
# somebody remembers to run it.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VALIDATOR="$HERE/assert_measurement_under.sh"

if [ ! -f "$VALIDATOR" ]; then
    echo "FAIL: $VALIDATOR does not exist - the table has nothing to drive." >&2
    exit 1
fi

checked=0
bad=0

# A row is: <mode> <value> <bar> <expectation>
#   mode        under | atleast
#   expectation MUST_PASS | MUST_FAIL
run_row() {
    row_mode="$1"
    row_value="$2"
    row_bar="$3"
    row_want="$4"
    set +e
    row_out=$(bash "$VALIDATOR" "$row_mode" "$row_value" "$row_bar" "table-probe" 2>&1)
    row_rc=$?
    set -e
    if [ "$row_rc" -eq 0 ]; then
        row_got="MUST_PASS"
    else
        row_got="MUST_FAIL"
    fi
    checked=$((checked+1))
    if [[ "$row_got" == "$row_want" ]]; then
        printf '  ok    %-8s %-14s bar=%-6s %s\n' \
            "$row_mode" "<$row_value>" "$row_bar" "$row_want"
    else
        bad=$((bad+1))
        printf '  BAD   %-8s %-14s bar=%-6s wanted %s, got %s (rc=%s): %s\n' \
            "$row_mode" "<$row_value>" "$row_bar" "$row_want" "$row_got" \
            "$row_rc" "$row_out"
    fi
}

# A real carriage return, built without a backslash escape so the table itself
# cannot be misread by a linter or by a future editor.
CR=$(printf 'X' | tr 'X' '\015')

echo "Driving the table through $VALIDATOR"
echo
echo "MUST_PASS - a well-formed non-negative decimal strictly under the bar:"
run_row under "0" "2.0" MUST_PASS
run_row under "0.0" "2.0" MUST_PASS
run_row under "1.999" "2.0" MUST_PASS
run_row under ".5" "2.0" MUST_PASS
run_row under "1" "2.0" MUST_PASS
# The 100 ms SC4 bar (chronos-bench) and the 2 s SC1 bars share this validator,
# so a row at each scale is here: a units mix-up is a silent gate inversion.
run_row under "42.5" "100" MUST_PASS

echo
echo "MUST_FAIL - malformed, or at/over the bar. Seven of these printed OK under"
echo "            the inline awk coercion the four old bar sites used:"
run_row under "" "2.0" MUST_FAIL
run_row under "abc" "2.0" MUST_FAIL
run_row under "2.0" "2.0" MUST_FAIL
run_row under "2.5" "2.0" MUST_FAIL
run_row under "1.2.3" "2.0" MUST_FAIL
run_row under "." "2.0" MUST_FAIL
run_row under "-1" "2.0" MUST_FAIL
# An exponent notation awk would happily coerce, and no measurement line in this
# repository ever emits. The old coercion refused it only by ACCIDENT - it read
# it as 1000, which is over the bar - so at a 2000 ms bar it would have passed.
run_row under "1e3" "2.0" MUST_FAIL
run_row under "1 2" "2.0" MUST_FAIL
run_row under "1.5${CR}" "2.0" MUST_FAIL

echo
echo "The OTHER direction. The forecast-pool-ratio bar is best at-least 2.0, not"
echo "under 2.0; reusing the under mode there would INVERT a gate, so the mode is"
echo "explicit at every call site and both directions are tabled:"
run_row atleast "2.0" "2.0" MUST_PASS
run_row atleast "5.162" "2.0" MUST_PASS
run_row atleast "1.999" "2.0" MUST_FAIL
run_row atleast "0" "2.0" MUST_FAIL
run_row atleast "abc" "2.0" MUST_FAIL
run_row atleast "" "2.0" MUST_FAIL

echo
echo "An unknown mode is a REFUSAL, never a silent fall-through to one direction:"
run_row sideways "1.0" "2.0" MUST_FAIL

echo
if [ "$checked" -eq 0 ]; then
    echo "FAIL: the table drove ZERO rows, so it proved nothing." >&2
    exit 1
fi
if [ "$bad" -ne 0 ]; then
    echo "FAIL: $bad of $checked rows behaved the other way." >&2
    echo "      A MUST_FAIL row that passes is the IN-01 coercion defect still" >&2
    echo "      present in the validator; a MUST_PASS row that fails means every" >&2
    echo "      real gate now fails closed and will be re-loosened by whoever" >&2
    echo "      debugs it next." >&2
    exit 1
fi
echo "TABLE OK: $checked/$checked rows behaved as tabled ($VALIDATOR)"
