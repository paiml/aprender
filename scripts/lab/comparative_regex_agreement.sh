#!/usr/bin/env bash
# scripts/lab/comparative_regex_agreement.sh -- #6599, PRREV-019 (F5). LAB/nightly check, never inside a PR, merge-queue or
# release job (C324). PR-REVIEW-SKILL-002-v2.md F5: B4's comparative regex and the claim-literals guard's RATIO_RE drifted
# because each was written from its own vocabulary. The fix F5 asks for is ONE pattern: the union of the competitor lists,
# the measured five-word gap, and a shared case table. This check holds B4's side of that to account, from the files:
#   1. gap    RATIO_GAP_RE, MULT_RE and RATIO_LEFT_RE are byte-identical in check_pr_review_receipt.sh (B4) and
#             check_no_claim_literals.sh, so the five-word bound cannot drift on one side.
#   2. union  B4's competitor list (case-folded) contains every competitor the claim-literals guard names.
#   3. table  B4's COMPARATIVE_RE gives the expected verdict on EVERY row of
#             tests/fixtures/pr-review/comparative-claim-cases.tsv (MATCH / NO-MATCH).
# INFO (never RED, F5 is advisory): rows of that table the claim-literals RATIO_RE gets wrong -- the residual until the two
# are one pattern. It is printed with its count so the gap stays visible instead of being read as agreement.
# Patterns are READ from the scripts (single-quoted assignment lines), never sourced or evaluated.
# Usage: scripts/lab/comparative_regex_agreement.sh | --self-test      Exit: 0 green, 1 RED, 2 cannot measure
# Overrides (tests): B4_SCRIPT CLAIM_SCRIPT CASES
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
B4_SCRIPT="${B4_SCRIPT:-$ROOT/scripts/check_pr_review_receipt.sh}"
CLAIM_SCRIPT="${CLAIM_SCRIPT:-$ROOT/scripts/check_no_claim_literals.sh}"
CASES="${CASES:-$ROOT/tests/fixtures/pr-review/comparative-claim-cases.tsv}"

# ext FILE NAME -> the value of the first `NAME='...'` line, empty when absent.
ext() { { grep -m1 -E "^$2='" "$1" || true; } | sed -E "s/^$2='(.*)'\$/\1/"; }
# tokens STR -> one lowercase alternative per line (outer parens stripped).
tokens() { printf '%s' "$1" | sed -E 's/^\(//; s/\)$//' | tr '|' '\n' | tr 'A-Z' 'a-z' | sort -u; }

run() {
    local k a b red=0 n=0 info=0 v s _r bre cre mism
    for f in "$B4_SCRIPT" "$CLAIM_SCRIPT" "$CASES"; do
        [ -f "$f" ] || { echo "NO-MEASURE: $f does not exist" >&2; return 2; }
    done
    for k in RATIO_LEFT_RE MULT_RE RATIO_GAP_RE COMPETITOR_RE; do
        for f in "$B4_SCRIPT" "$CLAIM_SCRIPT"; do
            [ -n "$(ext "$f" "$k")" ] || { echo "NO-MEASURE: $k not found in $f" >&2; return 2; }
        done
    done
    for k in RATIO_LEFT_RE MULT_RE RATIO_GAP_RE; do
        a="$(ext "$B4_SCRIPT" "$k")"; b="$(ext "$CLAIM_SCRIPT" "$k")"
        if [ "$a" = "$b" ]; then echo "  ok    gap    $k identical"
        else echo "  RED   gap    $k differs: B4=[$a] claim-literals=[$b]"; red=1; fi
    done
    mism="$(comm -13 <(tokens "$(ext "$B4_SCRIPT" COMPETITOR_RE)") <(tokens "$(ext "$CLAIM_SCRIPT" COMPETITOR_RE)") | tr '\n' ' ')"
    if [ -z "$mism" ]; then echo "  ok    union  B4's competitor list contains the claim-literals list"
    else echo "  RED   union  B4 lacks: $mism"; red=1; fi
    bre="$(ext "$B4_SCRIPT" RATIO_LEFT_RE)[0-9]+(\\.[0-9]+)?[[:space:]]*$(ext "$B4_SCRIPT" MULT_RE)[[:space:]]*$(ext "$B4_SCRIPT" RATIO_GAP_RE)$(ext "$B4_SCRIPT" COMPETITOR_RE)"
    cre="$(ext "$CLAIM_SCRIPT" RATIO_LEFT_RE)[0-9]+(\\.[0-9]+)?$(ext "$CLAIM_SCRIPT" MULT_RE)[[:space:]]*$(ext "$CLAIM_SCRIPT" RATIO_GAP_RE)$(ext "$CLAIM_SCRIPT" COMPETITOR_RE)"
    while IFS=$'\t' read -r v s _r; do
        case "$v" in MATCH|NO-MATCH) ;; *) continue ;; esac
        n=$((n+1))
        a=NO-MATCH; grep -Eqi -- "$bre" <<<"$s" && a=MATCH
        if [ "$a" != "$v" ]; then echo "  RED   table  B4 says $a, table says $v: $s"; red=1; fi
        b=NO-MATCH; grep -Eq -- "$cre" <<<"$s" && b=MATCH
        if [ "$b" != "$v" ]; then info=$((info+1)); echo "  INFO  claim-literals says $b, table says $v: $s"; fi
    done < <(grep -vE '^[[:space:]]*(#|$)' "$CASES")
    [ "$n" -gt 0 ] || { echo "NO-MEASURE: $CASES has no MATCH/NO-MATCH rows" >&2; return 2; }
    [ "$red" -eq 0 ] && echo "  ok    table  B4 agrees on all $n rows"
    echo "  info  residual: claim-literals RATIO_RE disagrees on $info of $n rows (advisory, F5)"
    [ "$red" -eq 0 ] && echo "GREEN comparative-regex agreement" || echo "RED comparative-regex agreement"
    return "$red"
}

self_test() {
    local t pass=0 fail=0 rc out
    t="$(mktemp -d)"
    case "$t" in /tmp/*|/var/tmp/*) : ;; *) echo "NO-GO: odd mktemp path $t" >&2; return 2 ;; esac
    row() { if [ "$2" = "$3" ]; then pass=$((pass+1)); printf '  ok    %-58s %s\n' "$1" "$3"
            else fail=$((fail+1)); printf '  FAIL  %-58s want=%s got=%s\n' "$1" "$3" "$2"; fi; }
    probe() { # probe NAME WANT_RC [env assignments...]
        local name="$1" want="$2"; shift 2
        set +e; out="$(env "$@" bash "${BASH_SOURCE[0]}" 2>&1)"; rc=$?; set -e
        row "$name" "$rc" "$want"
    }
    printf 'comparative_regex_agreement self-test\n'
    probe "real tree is GREEN" 0
    # planted violations: each is a copy of the real B4 script with one pattern changed
    sed -E "s/^(RATIO_GAP_RE=.*)\{0,5\}'/\1{0,0}'/" "$B4_SCRIPT" > "$t/gap.sh"
    probe "PLANTED: gap narrowed to zero words -> RED" 1 "B4_SCRIPT=$t/gap.sh"
    sed -E "/^COMPETITOR_RE=/ s/sklearn\|//; /^COMPETITOR_RE=/ s/\|transformers//" "$B4_SCRIPT" > "$t/comp.sh"
    probe "PLANTED: competitor dropped from B4 -> RED" 1 "B4_SCRIPT=$t/comp.sh"
    sed -E "s/^(MULT_RE=).*/\1'(x)'/" "$B4_SCRIPT" > "$t/mult.sh"
    probe "PLANTED: multiplier loses the typographic x -> RED" 1 "B4_SCRIPT=$t/mult.sh"
    { cat "$CASES"; printf 'NO-MATCH\t3.2x faster than ollama\tplanted wrong expectation\n'; } > "$t/cases.tsv"
    probe "PLANTED: table row contradicting B4 -> RED" 1 "CASES=$t/cases.tsv"
    printf '# no rows\n' > "$t/empty.tsv"
    probe "empty table is NO-MEASURE (2), not green" 2 "CASES=$t/empty.tsv"
    probe "missing script is NO-MEASURE (2), not green" 2 "B4_SCRIPT=$t/none.sh"
    sed -E "/^COMPETITOR_RE=/ s/\(ollama\|/(ollama|zzzcompetitor|/" "$B4_SCRIPT" > "$t/extra.sh"
    probe "B4 with an EXTRA competitor stays GREEN (superset ok)" 0 "B4_SCRIPT=$t/extra.sh"
    printf 'self-test: %s passed, %s failed\n' "$pass" "$fail"
    [ -d "${t:?}" ] && rm -rf "${t:?}"
    [ "$fail" -eq 0 ]
}

case "${1:-}" in
    --self-test) self_test ;;
    '') run ;;
    *) echo "usage: $(basename "$0") [--self-test]" >&2; exit 2 ;;
esac
