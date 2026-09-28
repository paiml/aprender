#!/usr/bin/env bash
# check_tag_coverage_gated.sh -- the release autopilot reads the tag's own coverage before T-4 (#3690)
#
#   bash scripts/check_tag_coverage_gated.sh              # judge scripts/release/autopilot.sh
#   bash scripts/check_tag_coverage_gated.sh --self-test  # mutation rows against copies
#
# #3676 moved COV_FLOOR onto the tag push (`ci / coverage`); nothing on the release path read
# it, so a floor breach on the tag still reached the crates.io cascade. autopilot.sh now runs
# scripts/release/tag_coverage_gate.sh in its preflight step. This guard holds three facts:
#   1. the gate's own case table passes (it stubs gh; no network);
#   2. autopilot.sh invokes the gate on the tag and its commit ("$T" "$MC");
#   3. that call sits BEFORE the dryrun and cascade steps, and a nonzero rc dies.
# A call after the cascade, or one whose rc nothing reads, would keep 2 true and gate nothing.
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
    if [ "$rc" -eq 0 ]; then echo "PASS  autopilot reads the tag's ci / coverage before T-4 (#3690)"; return 0; fi
    echo "FAIL  $why"; return "$rc"
}

self_test() {
    local d fail=0 want name
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
    rm -f -- "$d/real.sh" "$d/deleted.sh" "$d/unread.sh" "$d/late.sh"; rmdir -- "$d"
    [ "$fail" -eq 0 ] && { echo "check_tag_coverage_gated self-test: PASS"; return 0; }
    echo "check_tag_coverage_gated self-test: FAIL"; return 1
}

case "${1:-}" in
    --self-test) self_test ;;
    '') judge "$ROOT/scripts/release/autopilot.sh" ;;
    *) echo "usage: $0 [--self-test]" >&2; exit 2 ;;
esac
