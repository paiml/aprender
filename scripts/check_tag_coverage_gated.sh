#!/usr/bin/env bash
# check_tag_coverage_gated.sh -- the release reads coverage-nightly's receipt for the release commit (#3690, #4734, #4735)
#
#   bash scripts/check_tag_coverage_gated.sh              # judge scripts/release/autopilot.sh
#   bash scripts/check_tag_coverage_gated.sh --self-test  # wiring rows + mutants of the gate itself
#
# #3676 moved COV_FLOOR off the PR path; nothing on the release path read it, so a floor breach
# still reached the crates.io cascade. #4734: the tag push's coverage section was vacuous on
# v0.70.1 (0 tests, no %), so the gate now judges coverage-nightly's sha-keyed receipt instead.
# This guard holds four facts:
#   1. the gate's own case table passes (it stubs gh over a real git history; no network);
#   2. autopilot.sh invokes the gate on the tag and its commit ("$T" "$MC");
#   3. that call sits BEFORE the dryrun and cascade steps, and a nonzero rc dies;
#   4. every mutant of the gate below turns its case table RED. A table that a deleted check
#      cannot turn red holds nothing up (L25).
# The before-the-tag `--resolve` call in cut_tag() is held by scripts/check_tag_step_gated.sh.
#
# EXIT 0 wired · 1 not wired · 2 the subject moved (a file or anchor is missing).
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="$ROOT/scripts/release/tag_coverage_gate.sh"
CALL='bash scripts/release/tag_coverage_gate.sh "$T" "$MC"'
DIE='|| die "tag coverage on $T refused'

# wired FILE -> prints ok or the reason; rc 0 wired, 1 not, 2 anchor missing
wired() {
    local f=$1 call dry casc
    [ -f "$f" ] || { echo "no $f"; return 2; }
    dry=$(grep -nF 'if run_step dryrun; then' "$f" | head -1 | cut -d: -f1)
    casc=$(grep -nF 'if run_step cascade; then' "$f" | head -1 | cut -d: -f1)
    [ -n "$dry" ] && [ -n "$casc" ] || { echo "autopilot.sh has no dryrun/cascade step anchor -- the subject moved"; return 2; }
    call=$(grep -nF -- "$CALL" "$f" | head -1 | cut -d: -f1)
    [ -n "$call" ] || { echo "autopilot.sh never runs the tag coverage gate on \"\$T\" \"\$MC\""; return 1; }
    [ "$call" -lt "$dry" ] && [ "$call" -lt "$casc" ] || { echo "the tag coverage gate runs at line $call, after the dryrun ($dry) or cascade ($casc) step"; return 1; }
    local after
    after=$(sed -n "$call,$((call + 3))p" "$f")
    grep -qF -- "$DIE" <<< "$after" || { echo "nothing dies on the tag coverage gate's rc within 3 lines of line $call"; return 1; }
    echo ok
}

judge() {
    local ap=$1 why rc
    bash "$GATE" --self-test > /dev/null 2>&1 || { echo "FAIL  tag_coverage_gate.sh --self-test is red"; return 1; }
    why=$(wired "$ap"); rc=$?
    if [ "$rc" -eq 0 ]; then echo "PASS  autopilot judges the nightly coverage receipt for the release commit before T-4 (#3690, #4734)"; return 0; fi
    echo "FAIL  $why"; return "$rc"
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
Cargo diff not compared|    bump_in_place <<< "$d"|    true
Cargo lines compared as a set, not pairwise in place (#4819)|if (m[i] != p[i]) bad = 1|if (0) bad = 1
a hunk may add more lines than it removes (#4819)|if (nm != np) bad = 1|if (0) bad = 1
Cargo diff read with context, so a move inside one hunk pairs (#4819)|d=$("$GIT" diff -U0 "$h"|d=$("$GIT" diff "$h"
--resolve takes any ref, not a 40-hex sha (#4819)|[[ $2 =~ ^[0-9a-f]{40}$ ]] \|\| { echo "usage: $0 --resolve|true \|\| { echo "usage: $0 --resolve
TAG SHA takes any ref, not a 40-hex sha (#4819)|[[ $2 =~ ^[0-9a-f]{40}$ ]] \|\| { echo "usage: $0 TAG|true \|\| { echo "usage: $0 TAG
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
    local d fail=0 want name rc old new killed=0 total=0
    d=$(mktemp -d) || return 2
    local AP="$ROOT/scripts/release/autopilot.sh"
    cp -- "$AP" "$d/real.sh"
    grep -vF -- "$CALL" "$AP" > "$d/deleted.sh"
    grep -vF -- "$DIE" "$AP" > "$d/unread.sh"
    # the call moved after the cascade step: drop it, then re-insert it after the anchor
    grep -vF -- "$CALL" "$AP" | awk -v c="  $CALL" '{print} /if run_step cascade; then/{print c}' > "$d/late.sh"
    for row in "0 real" "1 deleted" "1 unread" "1 late"; do
        want=${row%% *}; name=${row#* }
        wired "$d/$name.sh" > /dev/null; rc=$?
        if [ "$rc" = "$want" ]; then echo "  ok   $name -> rc $rc"; else echo "  FAIL $name: wanted rc $want, got $rc"; fail=1; fi
    done
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
    rm -f -- "$d/real.sh" "$d/deleted.sh" "$d/unread.sh" "$d/late.sh" "$d/gate.sh"; rmdir -- "$d"
    [ "$fail" -eq 0 ] && { echo "check_tag_coverage_gated self-test: PASS"; return 0; }
    echo "check_tag_coverage_gated self-test: FAIL"; return 1
}

case "${1:-}" in
    --self-test) self_test ;;
    '') judge "$ROOT/scripts/release/autopilot.sh" ;;
    *) echo "usage: $0 [--self-test]" >&2; exit 2 ;;
esac
