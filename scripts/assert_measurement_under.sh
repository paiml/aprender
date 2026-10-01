#!/usr/bin/env bash
# assert_measurement_under.sh - the ONE numeric bar check every wall-clock gate
# in this repository calls.
#
#   assert_measurement_under.sh under   VALUE BAR LABEL    # asserts VALUE < BAR
#   assert_measurement_under.sh atleast VALUE BAR LABEL    # asserts VALUE >= BAR
#
# EXECUTED, NEVER SOURCED. It is invoked as a command
# (bash scripts/assert_measurement_under.sh ...), so the option-neutrality rule
# that governs scripts/apr_bin.sh - a set inside a SOURCED file mutates the
# CALLER shell, which is what scripts/check_sourced_libs_option_neutral.sh
# exists to catch - does not apply here. The set -euo pipefail below governs
# this script own shell and no other.
#
# WHY IT EXISTS (06-REVIEW.md IN-01). Five bar sites read a measurement through
#
#     awk -v v="$total" 'BEGIN { exit (v + 0 < 2.0) ? 0 : 1 }'
#
# and awk coerces a non-numeric token to 0. Zero is under every bar this
# repository asserts, so a broken measurement line - a renamed field, a
# truncated log, a harness that printed nothing - made the gate print OK. That
# is a gate reporting a result it never measured, which is worse than red: red
# gets investigated.
#
# So the SHAPE is checked FIRST and the comparison only runs on a token that is
# a well-formed non-negative decimal. What counts as well-formed is fixed by
# scripts/check_assert_measurement_under_cases.sh, a 23-row must-match /
# must-not-match table that just forecast-sc1-sweep runs BEFORE it measures
# anything. CLAUDE.md rule 7: the table is what catches a wrong pattern, not
# review - and rule 4: extending a guard to a new site requires re-mutating it
# THERE, which is why each of the five converted sites was fed a garbage token
# and observed failing in its own scope.
#
# BOTH DIRECTIONS ARE EXPLICIT. forecast-pool-ratio asserts best >= 2.0 and the
# rest assert value < bar. A single silent direction would have inverted that
# gate the first time somebody reused this script there, so the mode is a
# required first argument and an unknown mode is a refusal, never a default.

set -euo pipefail

usage() {
    echo "usage: $0 MODE VALUE BAR LABEL   where MODE is under or atleast" >&2
}

if [ "$#" -ne 4 ]; then
    echo "FAIL: expected 4 arguments, got $#" >&2
    usage
    exit 2
fi

mode="$1"
value="$2"
bar="$3"
label="$4"

# THE SHAPE CHECK, before any arithmetic. A token is well formed only if it is
# non-empty, made only of digits and dots, carries at least one digit, and
# carries at most one dot. That rejects, in order: the empty token (the harness
# printed no measurement at all), abc / -1 / 1e3 / "1 2" / a trailing carriage
# return (a byte that is neither digit nor dot), a lone dot (no digit), and
# 1.2.3 (two dots).
#
# NO CHARACTER CLASS ANYWHERE, DELIBERATELY, and the reason is measured. bashrs
# 6.66.3 reads a bracket as the start of a `[ ]` test expression WHEREVER it
# appears - in the glob the review itself suggested
# (`case "$1" in '' | *!0-9.* | *.*.*)`, brackets elided here so this comment
# does not trip the same rule), in its parameter-expansion cousin, and even
# inside a SINGLE-QUOTED ERE handed to grep. All three were tried and all three
# produced SC1020 "Missing space before closing ] in test expression".
# Reproduced minimally on a five-line script in plan 06-16.
#
# Worse than the noise: the finding did NOT surface when the same construct sat
# inside this longer file, only in the minimal reproducer and in multi-file mode.
# Shipping a construct the linter rejects in the small and happens to miss in the
# large is a false green of exactly the kind this validator exists to stop, so
# the check is written character by character with explicit alternation instead.
# A measurement token is never more than a few bytes; the loop costs nothing.
#
# Accepts 0, 0.0, 1, 1.999, .5, 42.5. Rejects the empty token, abc, -1, 1e3,
# "1 2", a trailing carriage return, a lone dot, and 1.2.3. Which of those is
# which is not an argument - it is the 23-row table in
# scripts/check_assert_measurement_under_cases.sh, run on every gate invocation.
shape_ok() {
    tok="$1"
    if [ -z "$tok" ]; then
        return 1
    fi
    seen_digit=no
    rest="$tok"
    # Consume one byte per turn. The default branch is the IN-01 class: a sign,
    # an exponent, a space, a carriage return, a letter - anything awk would
    # have coerced to 0.
    while [ -n "$rest" ]; do
        ch=${rest:0:1}
        rest=${rest:1}
        case "$ch" in
            0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9) seen_digit=yes ;;
            .) : ;;
            *) return 1 ;;
        esac
    done
    # At least one digit, so a lone dot is not a number.
    if [ "$seen_digit" != yes ]; then
        return 1
    fi
    # At most one dot, so 1.2.3 is not a number. A plain glob, no bracket.
    case "$tok" in
        *.*.*) return 1 ;;
        *) return 0 ;;
    esac
}

if ! shape_ok "$value"; then
    echo "FAIL \"$label\": the measurement token is not a well-formed non-negative" >&2
    echo "     decimal number, so there is nothing to compare against the $bar bar." >&2
    echo "     token: \"$value\"" >&2
    echo "     This is the IN-01 defect exactly: the old inline awk check coerced this" >&2
    echo "     to 0 and printed OK. A bar that cannot parse its input has not" >&2
    echo "     measured anything." >&2
    exit 1
fi

# The BAR is a literal written at the call site, never a parsed measurement - but
# a typo there would silently move every gate, so it gets the same check.
if ! shape_ok "$bar"; then
    echo "FAIL \"$label\": the bar itself is not a well-formed number: \"$bar\"" >&2
    exit 1
fi

case "$mode" in
    under)
        if awk -v v="$value" -v b="$bar" 'BEGIN { exit (v < b) ? 0 : 1 }'; then
            echo "  $label OK: $value < $bar"
            exit 0
        fi
        echo "FAIL \"$label\": $value is at or above the $bar bar" >&2
        exit 1
        ;;
    atleast)
        if awk -v v="$value" -v b="$bar" 'BEGIN { exit (v >= b) ? 0 : 1 }'; then
            echo "  $label OK: $value >= $bar"
            exit 0
        fi
        echo "FAIL \"$label\": $value is below the $bar bar" >&2
        exit 1
        ;;
    *)
        echo "FAIL \"$label\": unknown mode \"$mode\"." >&2
        echo "     Pick the direction deliberately: under asserts value < bar," >&2
        echo "     atleast asserts value >= bar. Defaulting to one of them would" >&2
        echo "     invert the forecast-pool-ratio gate the first time it was reused." >&2
        usage
        exit 2
        ;;
esac
