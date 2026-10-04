#!/usr/bin/env bash
# check_tag_step_gated.sh -- the release train's tag step must call
# scripts/check_milestone_cut.sh, and must not be reachable without it (PMAT-3459).
#
# THE DEFECT, measured 2026-09-17. v0.68.1 was tagged at 15:06:29Z by a per-train
# autopilot copy living OUTSIDE the repository, six minutes after #3455 merged the
# milestone gate. `grep -c check_milestone_cut autopilot.sh` was 0: the tag step read
# no milestone at all. The spec required the read; no code path that cuts a tag made it.
#
# WHY THIS IS BEHAVIOURAL AND NOT A grep. "The file mentions the gate" is satisfied by
# a comment. What must hold is that `git tag` is UNREACHABLE unless the gate returned
# 0, so this guard EXTRACTS cut_tag() from the autopilot and RUNS it against stubs,
# once per gate outcome, and asserts on whether a tag was cut:
#     gate rc 0 -> tag is cut
#     gate rc 1 -> no tag, no publish   (the milestone holds open items)
#     gate rc 2 -> no tag               (Unknown; never a silent pass)
# #3459 part 2 made the tag path three steps (must-carry gate, carry, STRICT gate), so the
# stubs answer each call separately and record the ORDER they ran in:
#     must-carry rc 1/2 -> no tag AND nothing carried (a blocker is never carried around)
#     carry rc 2        -> no tag
#     all clean         -> the carry ran BEFORE the strict gate, and the tag is cut
# --self-test then builds MUTANTS (gate calls removed, verdicts discarded, the carry call
# removed) and requires this guard to go RED on each. It also runs the carry script's own
# case table, which lives in scripts/release/ where guard_tree cannot discover it.
#
#   check_tag_step_gated.sh              judge scripts/release/autopilot.sh
#   check_tag_step_gated.sh --self-test  case table + the gate-removed mutant
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2

# SEC011: never let an empty or '/' value reach `rm -rf`. One validated helper, used
# by every call site in this file, rather than the check repeated at each one.
rmtree() { case "${1:-}" in ''|/) return 0 ;; *) [ -d "$1" ] && rm -rf -- "$1" ;; esac; return 0; }
SUBJECT="$ROOT/scripts/release/autopilot.sh"

# run_cut_tag <autopilot> <strict-rc> [<must-carry-rc> [<carry-rc> [<readiness>]]] -- extract cut_tag(), run
# it with stubs, print a transcript (SAY/DIE/GIT-TAG/GIT-PUSH lines, then the CALL order).
# Returns 2 if the function is missing.
run_cut_tag() {
    local ap=$1 grc=$2 mrc=${3:-0} crc=${4:-0} rdy=${5:-pass} tmr=${6:-pass} d fn
    d=$(mktemp -d) || return 2
    fn=$(awk '/^cut_tag\(\) \{/,/^\}/' "$ap")
    [ -n "$fn" ] || { rmtree "$d"; return 2; }
    mkdir -p "$d/scripts/release" "$d/ap"
    # #3715 B1: the readiness step's log, as the T-1 `readiness` step leaves it (or does not)
    case "$rdy" in
        pass)   printf 'ok    R8 release-readiness-v1 for 0.0.0: Pass\nok    R8 #3715 ENFORCE PASS version=0.0.0 commit=deadbeef pv=pv_x out_sha256=0\n' > "$d/ap/readiness-t1.log" ;;
        report) printf 'WARN  R8 REPORT-ONLY release-readiness-v1 for 0.0.0: Fail, 3 violation(s)\n' > "$d/ap/readiness-t1.log" ;;
        stale)  printf 'ok    R8 #3715 ENFORCE PASS version=0.0.0 commit=cafef00d pv=pv_x out_sha256=0\n' > "$d/ap/readiness-t1.log" ;;
        absent) : ;;
    esac
    # #4670 R10: the timers step's log, as the T-1 `timers` step leaves it (or does not)
    case "$tmr" in
        pass)     printf 'ok    R10 h1: release open (present 0.0.0); all 2 tool-installing timer(s) disarmed\nok    R10 TIMERS PASS release=0.0.0 hosts=1 timers=2\n' > "$d/ap/timers-t1.log" ;;
        other)    printf 'ok    R10 TIMERS PASS release=0.0.1 hosts=1 timers=2\n' > "$d/ap/timers-t1.log" ;;
        nomarker) printf 'ok    R10 TIMERS PASS release=none hosts=1 timers=2\n' > "$d/ap/timers-t1.log" ;;
        notmeas)  printf 'NOT_MEASURED R10 at least one release host could not be judged; not judged is not a pass\n' > "$d/ap/timers-t1.log" ;;
        fail)     printf 'FAIL  R10 a tool-installing timer is armed on a release host while the release is open\n' > "$d/ap/timers-t1.log" ;;
        absent)   : ;;
    esac
    printf '#!/usr/bin/env bash\nif [ "${2:-}" = --must-carry ]; then echo CALL-MUST-CARRY >> %q; exit %s; fi\necho CALL-STRICT >> %q; exit %s\n' \
        "$d/calls" "$mrc" "$d/calls" "$grc" > "$d/scripts/check_milestone_cut.sh"
    printf '#!/usr/bin/env bash\necho CALL-CARRY >> %q\nexit %s\n' "$d/calls" "$crc" > "$d/scripts/release/carry_milestone_items.sh"
    {
        printf 'set -uo pipefail\n'
        printf 'REPO_ROOT=%q\nLOG=%q\nAP=%q\n' "$d" "$d/log" "$d/ap"
        printf 'say() { printf "SAY %%s\\n" "$*"; }\n'
        printf 'die() { printf "DIE %%s\\n" "$*"; exit 1; }\n'
        printf 'git() { printf "GIT-%%s %%s\\n" "$(printf %%s "$1" | tr "a-z" "A-Z")" "$*"; }\n'
        printf '%s\n' "$fn"
        printf 'cut_tag 0.0.0 v0.0.0 deadbeef\n'
    } > "$d/harness.sh"
    bash "$d/harness.sh" 2>&1
    cat "$d/log" 2>/dev/null
    printf 'ORDER %s\n' "$(tr '\n' ' ' < "$d/calls" 2>/dev/null)"
    rmtree "$d"
}

# judge <autopilot> -- 0 when every gate outcome behaves; 1 otherwise. Prints rows.
judge() {
    local ap=$1 out bad=0
    if ! grep -q '^cut_tag() {' "$ap"; then
        printf 'FAIL  %s has no cut_tag() -- the tag path moved and this guard is judging nothing\n' "$ap" >&2
        return 2
    fi
    out=$(run_cut_tag "$ap" 0) || true
    if grep -q 'GIT-TAG' <<< "$out"; then printf 'ok    gate rc=0 -> the tag is cut\n'
    else printf 'FAIL  gate rc=0 -> NO tag was cut\n%s\n' "$out" >&2; bad=1; fi

    out=$(run_cut_tag "$ap" 1) || true
    if grep -q 'GIT-TAG' <<< "$out"; then
        printf 'FAIL  gate rc=1 (milestone holds open items) -> A TAG WAS CUT ANYWAY\n%s\n' "$out" >&2; bad=1
    else printf 'ok    gate rc=1 -> no tag, no publish\n'; fi

    out=$(run_cut_tag "$ap" 2) || true
    if grep -q 'GIT-TAG' <<< "$out"; then
        printf 'FAIL  gate rc=2 (Unknown) -> A TAG WAS CUT ANYWAY\n%s\n' "$out" >&2; bad=1
    else printf 'ok    gate rc=2 (Unknown) -> no tag\n'; fi
    # #3459 part 2: the must-carry gate, the carry, and their ORDER
    out=$(run_cut_tag "$ap" 0) || true
    if grep -q '^ORDER CALL-MUST-CARRY CALL-CARRY CALL-STRICT $' <<< "$out" && grep -q 'GIT-TAG' <<< "$out"; then
        printf 'ok    all clean -> must-carry, then the carry, then STRICT, then the tag\n'
    else printf 'FAIL  all clean did not run must-carry -> carry -> strict -> tag\n%s\n' "$out" >&2; bad=1; fi
    for m in 1 2; do
        out=$(run_cut_tag "$ap" 0 "$m") || true
        if grep -q 'GIT-TAG' <<< "$out" || grep -q 'CALL-CARRY' <<< "$out"; then
            printf 'FAIL  must-carry rc=%s -> a tag was cut or items were CARRIED around a blocker\n%s\n' "$m" "$out" >&2; bad=1
        else printf 'ok    must-carry rc=%s -> nothing carried, no tag\n' "$m"; fi
    done
    out=$(run_cut_tag "$ap" 0 0 2) || true
    if grep -q 'GIT-TAG' <<< "$out"; then
        printf 'FAIL  carry rc=2 -> A TAG WAS CUT over a failed carry\n%s\n' "$out" >&2; bad=1
    else printf 'ok    carry rc=2 -> no tag\n'; fi
    # #3715 B1: no ENFORCED readiness Pass for exactly this version+commit -> no tag, nothing carried
    for r in absent report stale; do
        out=$(run_cut_tag "$ap" 0 0 0 "$r") || true
        if grep -q 'GIT-TAG' <<< "$out" || grep -q 'CALL-' <<< "$out"; then
            printf 'FAIL  readiness %s -> a tag was cut or the milestone was touched without an enforced #3715 Pass\n%s\n' "$r" "$out" >&2; bad=1
        else printf 'ok    readiness %s -> no tag, nothing carried\n' "$r"; fi
    done
    # #4670 R10: no timers PASS naming exactly this release -> no tag, nothing carried
    for t in absent other nomarker notmeas fail; do
        out=$(run_cut_tag "$ap" 0 0 0 pass "$t") || true
        if grep -q 'GIT-TAG' <<< "$out" || grep -q 'CALL-' <<< "$out"; then
            printf 'FAIL  timers %s -> a tag was cut or the milestone was touched without an R10 PASS for this release\n%s\n' "$t" "$out" >&2; bad=1
        else printf 'ok    timers %s -> no tag, nothing carried\n' "$t"; fi
    done
    return "$bad"
}

# guard_tree.sh decides a guard "advertises --self-test" by looking for the literal
# substring `self-test` in its OWN `--help` output (guard_tree.sh:47, advertises_self_test).
# Without this handler the self-test below is discovered by nothing and runs nowhere --
# a facility with a self-test and no caller. With it, guard_tree gives this guard TWO
# rows, `[self-test]` and `[run]`, inside the guard-tree job.
case "${1:-}" in -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

if [ "${1:-}" = "--self-test" ]; then
    echo "=== the tag-gate guard must still turn RED (mutants) ==="
    d=$(mktemp -d) || exit 2
    trap 'rmtree "${d:-}"' EXIT
    bad=0
    ok()  { printf 'ok    %s\n' "$*"; }
    nok() { printf 'FAIL  %s\n' "$*" >&2; bad=1; }

    # M1: the gate call deleted -- the #3459 defect exactly, restored.
    sed '/check_milestone_cut\.sh/d' "$SUBJECT" > "$d/m1.sh"
    if judge "$d/m1.sh" > "$d/m1.out" 2>&1; then
        nok "MUTANT 1 (gate call deleted) PASSED -- this guard cannot see its own defect"
    else
        ok "mutant 1: gate call deleted -> RED ($(grep -c '^FAIL' "$d/m1.out") failing row(s))"
    fi

    # M2: the gate runs but its verdict is discarded (`|| true`) -- absence-as-consent.
    sed 's#\(bash "$REPO_ROOT/scripts/check_milestone_cut.sh" "$v" >> "$LOG" 2>&1\) || rc=$?#\1 || true#' "$SUBJECT" > "$d/m2.sh"
    if ! grep -q '|| true' "$d/m2.sh"; then
        nok "MUTANT 2 could not be built -- the gate-call line did not match; this self-test is vacuous"
    elif judge "$d/m2.sh" > "$d/m2.out" 2>&1; then
        nok "MUTANT 2 (verdict discarded) PASSED -- a gate whose result is thrown away reads as gated"
    else
        ok "mutant 2: gate verdict discarded -> RED"
    fi

    # M4: the MUST-CARRY verdict discarded -> items are carried around a blocker and a tag is cut.
    sed 's#\(bash "$REPO_ROOT/scripts/check_milestone_cut.sh" "$v" --must-carry >> "$LOG" 2>&1\) || rc=$?#\1 || true#' "$SUBJECT" > "$d/m4.sh"
    if cmp -s "$SUBJECT" "$d/m4.sh"; then
        nok "MUTANT 4 could not be built -- the must-carry call line did not match; vacuous"
    elif judge "$d/m4.sh" > "$d/m4.out" 2>&1; then
        nok "MUTANT 4 (must-carry verdict discarded) PASSED"
    else
        ok "mutant 4: must-carry verdict discarded -> RED"
    fi
    # M5: the carry call deleted -> a milestone is judged strict without anything having been moved.
    sed '/carry_milestone_items\.sh" "\$v"/d' "$SUBJECT" > "$d/m5.sh"
    if cmp -s "$SUBJECT" "$d/m5.sh"; then
        nok "MUTANT 5 could not be built -- the carry call line did not match; vacuous"
    elif judge "$d/m5.sh" > "$d/m5.out" 2>&1; then
        nok "MUTANT 5 (carry call deleted) PASSED"
    else
        ok "mutant 5: carry call deleted -> RED"
    fi
    # M6 (#3715 B1): the readiness requirement deleted -> a skipped or report-mode readiness step tags.
    sed '/ENFORCE PASS for\|readiness-t1\.log" 2>\/dev\/null/d; /no .#3715 ENFORCE PASS/d' "$SUBJECT" > "$d/m6.sh"
    if cmp -s "$SUBJECT" "$d/m6.sh"; then
        nok "MUTANT 6 could not be built -- the readiness check line did not match; vacuous"
    elif judge "$d/m6.sh" > "$d/m6.out" 2>&1; then
        nok "MUTANT 6 (readiness requirement deleted) PASSED"
    else
        ok "mutant 6: #3715 readiness requirement deleted -> RED"
    fi
    # M7 (#4670 R10): the timers requirement deleted -> a skipped or unjudged timers step tags.
    sed '/R10 TIMERS PASS. for\|timers-t1\.log" 2>\/dev\/null/d' "$SUBJECT" > "$d/m7.sh"
    if cmp -s "$SUBJECT" "$d/m7.sh"; then
        nok "MUTANT 7 could not be built -- the timers check line did not match; vacuous"
    elif judge "$d/m7.sh" > "$d/m7.out" 2>&1; then
        nok "MUTANT 7 (R10 timers requirement deleted) PASSED"
    else
        ok "mutant 7: #4670 R10 timers requirement deleted -> RED"
    fi
    # the carry script's own case table: it lives in scripts/release/, where guard_tree cannot see it
    if bash "$ROOT/scripts/release/carry_milestone_items.sh" --self-test > "$d/carry.out" 2>&1; then
        ok "carry_milestone_items.sh case table ($(grep -c '^ok ' "$d/carry.out") rows)"
    else
        nok "carry_milestone_items.sh case table FAILED"; cat "$d/carry.out" >&2
    fi
    # M3: cut_tag() removed entirely -> ENV (2), never a pass.
    awk '/^cut_tag\(\) \{/,/^\}/ {next} {print}' "$SUBJECT" > "$d/m3.sh"
    judge "$d/m3.sh" > "$d/m3.out" 2>&1; rc=$?
    [ "$rc" -eq 2 ] && ok "mutant 3: cut_tag() removed -> ENV rc=2, not a pass" \
        || nok "mutant 3: cut_tag() removed gave rc=$rc, wanted 2"

    # and the real subject must be GREEN, or a red subject masquerades as a killed mutant
    if judge "$SUBJECT" > "$d/real.out" 2>&1; then
        ok "the real subject is GREEN, so the REDs above are the mutants'"
    else
        nok "the real subject is itself RED -- the mutations prove nothing"; cat "$d/real.out" >&2
    fi
    [ "$bad" -eq 0 ] && { echo "SELF-TEST PASSED"; exit 0; }
    echo "SELF-TEST FAILED" >&2; exit 1
fi

echo "=== the tag step cannot be reached without the milestone gate (check_tag_step_gated.sh) ==="
judge "$SUBJECT"; rc=$?
[ "$rc" -eq 0 ] && echo "PASS" || echo "FAIL: the tag step is reachable without a clean milestone (rc=$rc)" >&2
exit "$rc"
