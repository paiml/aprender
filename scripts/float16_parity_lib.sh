# float16_parity_lib.sh - the comparison helpers of the #3076 greedy-parity gate, shared by
# scripts/float16_greedy_parity.sh (the release-time, model-backed run) and
# scripts/check_float16_greedy_parity.sh (the PR-time case table that proves them).
#
# SOURCED, so option-neutral: no `set` here (scripts/check_sourced_libs_option_neutral.sh).
# Callers use `. scripts/float16_parity_lib.sh || exit 2`.

# first_diff A B -> prints the first differing character offset, or -1 when equal. The
# strings go in through the environment, not `awk -v`, which would interpret backslash
# escapes in generated code.
first_diff() {
    A=$1 B=$2 awk 'BEGIN {
        a = ENVIRON["A"]; b = ENVIRON["B"]
        if (a == b) { print -1; exit }
        n = length(a) < length(b) ? length(a) : length(b)
        for (i = 1; i <= n; i++) if (substr(a, i, 1) != substr(b, i, 1)) { print i - 1; exit }
        print n
    }'
}

trim() {
    local s=$1
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

# json_str S -> S as a JSON string literal (backslash, quote, control characters escaped).
json_str() {
    local s=$1
    s=${s//\\/\\\\}
    s=${s//\"/\\\"}
    s=${s//$'\n'/\\n}
    s=${s//$'\t'/\\t}
    s=${s//$'\r'/\\r}
    printf '"%s"' "$s"
}
