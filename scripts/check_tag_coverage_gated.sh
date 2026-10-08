#!/usr/bin/env bash
# check_tag_coverage_gated.sh -- tag_coverage_gate.sh, the judge of coverage-nightly's receipt, holds (#3690, #4734, #4735)
#
#   bash scripts/check_tag_coverage_gated.sh              # the gate's own case table
#   bash scripts/check_tag_coverage_gated.sh --self-test  # mutants of the gate itself
#
# #3676 moved COV_FLOOR off the PR path; nothing on the release path read it, so a floor breach
# still reached the crates.io cascade. #4734: the tag push's coverage section was vacuous on
# v0.70.1 (0 tests, no %), so the gate now judges coverage-nightly's sha-keyed receipt instead.
# C333 took the gate off the release path (autopilot preflight and cut_tag --resolve no longer call it),
# so the wiring facts left with it. This guard holds two:
#   1. the gate's own case table passes (it stubs gh over a real git history; no network);
#   2. every mutant of the gate below turns its case table RED. A table that a deleted check
#      cannot turn red holds nothing up (L25).
#
# EXIT 0 green · 1 red · 2 the subject moved (a file or anchor is missing).
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="$ROOT/scripts/release/tag_coverage_gate.sh"

judge() {
    bash "$GATE" --self-test > /dev/null 2>&1 || { echo "FAIL  tag_coverage_gate.sh --self-test is red"; return 1; }
    echo "PASS  tag_coverage_gate.sh judges coverage-nightly's receipt (#3690, #4734); off the release path (C333)"
}

# Each row: NAME, then a fixed string in the gate, then what replaces its first occurrence. A row
# whose string is not found is vacuous and fails the self-test (the subject moved).
MUTANTS=$(cat <<'EOF'
schema check deleted|[ "$schema" = coverage-receipt/v1 ] |||true ||
receipt-sha check deleted|[ "$sha" = "$h" ] ||| true ||
status check deleted|[ "$st" = measured ] |||true ||
test-count check deleted|[[ $passed =~ ^[1-9][0-9]*$ ]] |||true ||
line-total check deleted|[[ $tot =~ ^[1-9][0-9]*$ ]] |||true ||
covered<=total check deleted|[[ $cov =~ ^[0-9]+$ ]] && [ "$cov" -le "$tot" ] |||true ||
pct-is-covered/total check deleted|[ "$pct" = "$want" ] |||true ||
floor compare ignores the floor|p + 0 >= f + 0|p + 0 >= 0
no COV_FLOOR passes as floor 0|if [ -z "$floor" ]; then|if false; then
ancestry check deleted|--is-ancestor "$h" "$sha" 2>/dev/null || return 1|--is-ancestor "$h" "$sha" 2>/dev/null || true
surface file may be added or deleted|[ "$st" = M ] || return 1|true || return 1
model-ladder receipt may be modified|[ "$st" = A ] || return 1|true || return 1
any path rides on a bump|else return 1; fi|else :; fi
any version's receipt dir rides on a bump|${v//./\\.}|[^/]+
Cargo diff not compared|blank_versions() { sed -E 's/version = "[^"]*"/version = ""/g' \| sort; }|blank_versions() { :; }
oldest run picked, not newest|sort_by(.createdAt) \| reverse \||sort_by(.createdAt) \|
--resolve always passes|gate "the release commit $2" "$2" "no tag, nothing carried" ;;|exit 0 ;;
EOF
)

# mutate SRC OLD NEW DST -> DST is SRC with the first OLD replaced by NEW; rc 1 if OLD is absent
mutate() {
    OLD=$2 NEW=$3 awk 'BEGIN { o = ENVIRON["OLD"]; n = ENVIRON["NEW"] }
        !done && (i = index($0, o)) { $0 = substr($0, 1, i - 1) n substr($0, i + length(o)); done = 1 }
        { print } END { exit !done }' "$1" > "$4"
}

self_test() {
    local d fail=0 name old new killed=0 total=0
    d=$(mktemp -d) || return 2
    if bash "$GATE" --self-test > /dev/null 2>&1; then echo "  ok   the real gate's case table is GREEN"
    else echo "  FAIL the real gate's case table is RED, so no mutant below can be told apart"; fail=1; fi
    # The separator is '|'; a literal '|' inside a fixed string is written '\|'.
    while IFS= read -r row; do
        [ -n "$row" ] || continue
        row=${row//\\|/$'\x1f'}
        IFS='|' read -r name old new <<< "$row"
        old=${old//$'\x1f'/|}; new=${new//$'\x1f'/|}
        total=$((total + 1))
        if ! mutate "$GATE" "$old" "$new" "$d/gate.sh"; then
            echo "  FAIL mutant '$name': its string is not in the gate -- vacuous, the subject moved"; fail=1; continue
        fi
        if bash "$d/gate.sh" --self-test > /dev/null 2>&1; then echo "  FAIL mutant '$name' SURVIVED: the case table stayed green"; fail=1
        else echo "  ok   mutant '$name' -> RED"; killed=$((killed + 1)); fi
    done <<< "$MUTANTS"
    echo "  mutants killed $killed/$total"
    rm -f -- "$d/gate.sh"; rmdir -- "$d"
    [ "$fail" -eq 0 ] && { echo "check_tag_coverage_gated self-test: PASS"; return 0; }
    echo "check_tag_coverage_gated self-test: FAIL"; return 1
}

case "${1:-}" in
    --self-test) self_test ;;
    '') judge ;;
    *) echo "usage: $0 [--self-test]" >&2; exit 2 ;;
esac
