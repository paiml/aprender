#!/usr/bin/env bash
# check_pass_grep_anchored.sh - every "zero-count" pass-grep must actually turn RED.
#
# PMAT-CI-PASSGREP-001 closed ONE instance of this defect (ci.yml's
# `grep -q "test result.*0 failed"`, which matched "10 failed" and merged a run
# with 10 failures GREEN). The pattern is a CLASS, not an incident: two more
# live instances survived that fix - `contracts/qwen-story-v1.yaml`'s
# FALSIFY-QWEN-STORY-007 (`grep -qE '0 error'`, satisfied by "10 error(s)") and
# `scripts/dogfood-book.sh` - so a point fix is not enough. This is the ratchet.
#
# METHOD: empirical, not syntactic. Regex-linting regexes is brittle; instead we
# extract each pass-grep pattern from the tree and RUN it against a synthetic
# line in which every count is non-zero. A pass-detector that matches an
# all-failing line fails open, by definition. No heuristic about anchors,
# word boundaries, or leading spaces is required - the probe settles it.
#
# Exit 0 = every pass-grep correctly rejects the failing probe.
# Exit 1 = at least one pass-grep fails open (prints file:line and the pattern).

set -euo pipefail

SELF_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

cd "$(dirname "$0")/.." || exit 1

# --self-test: prove this checker can turn RED, using the exact pattern that
# shipped in ci.yml before PMAT-CI-PASSGREP-001. A gate nobody has ever seen
# fail is indistinguishable from a gate that cannot fail.
if [ "${1:-}" = "--self-test" ]; then
    TMP=$(mktemp -d)
    cleanup_selftest() {
        if [ -n "$TMP" ] && [ "$TMP" != / ]; then
            rm -rf "$TMP"
        fi
    }
    trap cleanup_selftest EXIT
    mkdir -p "$TMP/scripts" "$TMP/contracts" "$TMP/.github/workflows"
    touch "$TMP/Makefile"
    cp "$SELF_PATH" "$TMP/scripts/"
    # The pre-PMAT-CI-PASSGREP-001 ci.yml line, verbatim.
    HISTORICAL='grep -q "test result.*0 failed" out.log'
    printf '%s\n' "$HISTORICAL" > "$TMP/scripts/historical_pattern.sh"
    if (cd "$TMP" && MIN_EXPECTED=1 bash scripts/check_pass_grep_anchored.sh >/dev/null 2>&1); then
        printf 'SELF-TEST FAILED: the checker accepted `grep -q "test result.*0 failed"`,\n' >&2
        printf 'the exact pattern that merged 10 test failures GREEN (PMAT-CI-PASSGREP-001).\n' >&2
        exit 1
    fi
    printf 'self-test OK: the checker turns RED on the PMAT-CI-PASSGREP-001 pattern.\n'
    exit 0
fi

# A line in which EVERY count is non-zero. Any pattern claiming to detect
# "zero errors / zero failures / zero warnings" must NOT match this.
PROBE='Summary: 10 error(s), 20 warning(s), 30 info(s) :: test result: FAILED. 90 passed; 10 failed; 40 ignored'

# Zero-count pass-detector keywords. A grep pattern mentioning one of these is
# asserting "none of these happened" and is therefore in scope.
KEYWORDS='0 error|0 fail|0 warn|0 issue|0 violation|0 problem'

SEARCH_PATHS=(.github/workflows contracts scripts Makefile)

# Known zero-count pass-greps in the tree at the time of writing. If the
# extractor finds fewer than this, it has gone blind - see the tail of this
# script.
MIN_EXPECTED="${MIN_EXPECTED:-2}"

VIOLATIONS=0
CHECKED=0

# Pull the quoted pattern out of a grep invocation. Handles the two forms used
# in this tree: grep -q'X' with single quotes and with double quotes. Emits one
# pattern per line.
extract_patterns() {
    # Two passes rather than one, so neither sed script needs escaped quotes:
    # the single-quote form is written inside double quotes and vice versa.
    sed -nE "s/.*grep[[:space:]]+(-[A-Za-z]+[[:space:]]+)*'([^']*)'.*/\2/p" "$1"
    # shellcheck disable=SC2016
    sed -nE 's/.*grep[[:space:]]+(-[A-Za-z]+[[:space:]]+)*"([^"]*)".*/\2/p' "$1"
}

SELF="scripts/check_pass_grep_anchored.sh"

while IFS= read -r hit; do
    file="${hit%%:*}"
    rest="${hit#*:}"
    lineno="${rest%%:*}"

    # This checker's own header quotes the historical bad patterns verbatim as
    # documentation; they are prose, not gates.
    [ "$file" = "$SELF" ] && continue

    # Re-read just this line and pull its grep pattern(s) out.
    line_text=$(sed -n "${lineno}p" "$file")

    trimmed=$(printf '%s' "$line_text" | sed 's/^[[:space:]]*//')

    # A commented-out grep is not a gate.
    case "$trimmed" in
        '#'*) continue ;;
        *) ;;
    esac

    # In a contract, only the `test:` field is executed by `pv`. Everything
    # else - five_whys, references, if_fails - is prose, and prose routinely
    # quotes the very bad patterns this checker exists to describe. Scanning it
    # would make the gate fire on its own documentation.
    case "$file" in
        contracts/*)
            case "$trimmed" in
                test:*) ;;
                *) continue ;;
            esac
            ;;
        *) ;;
    esac
    patterns=$(printf '%s\n' "$line_text" | extract_patterns /dev/stdin || true)

    [ -z "$patterns" ] && continue

    while IFS= read -r pat; do
        [ -z "$pat" ] && continue
        # Only patterns that actually assert a zero count are in scope.
        printf '%s\n' "$pat" | grep -qE "$KEYWORDS" || continue

        CHECKED=$((CHECKED + 1))

        # THE PROBE. If a "zero errors" detector matches an all-failing line,
        # it can never turn red on that failure mode.
        if printf '%s\n' "$PROBE" | grep -qE -- "$pat" 2>/dev/null; then
            VIOLATIONS=$((VIOLATIONS + 1))
            printf 'FAIL-OPEN %s:%s\n' "$file" "$lineno"
            printf '          pattern: %s\n' "$pat"
            printf '          matches: %s\n' "$PROBE"
            printf '          fix: anchor the count, e.g. (^|[^0-9])0 error\n'
        fi
    done <<EOF
$patterns
EOF
done <<EOF
$(grep -rnE "grep.*($KEYWORDS)" "${SEARCH_PATHS[@]}" 2>/dev/null || true)
EOF

if [ "$VIOLATIONS" -gt 0 ]; then
    printf '\n%s pass-grep(s) fail open (of %s checked).\n' "$VIOLATIONS" "$CHECKED" >&2
    printf 'A gate that cannot turn RED on a real regression is worth negative EV.\n' >&2
    exit 1
fi

# Fail-closed observability (the lesson of the dark beat lane): a checker that
# silently examines NOTHING reports the same "OK" as one that examined
# everything. If the extraction regex rots, or the search paths move, this
# must go red rather than congratulate itself on an empty set.
if [ "$CHECKED" -lt "$MIN_EXPECTED" ]; then
    printf 'ERROR: only %s pass-grep(s) found (expected >= %s).\n' \
        "$CHECKED" "$MIN_EXPECTED" >&2
    printf 'The extractor probably stopped matching - this check is now vacuous.\n' >&2
    exit 1
fi

printf 'OK: %s zero-count pass-grep(s) checked, all correctly reject a failing line.\n' "$CHECKED"
