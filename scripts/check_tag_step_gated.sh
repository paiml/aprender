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

# run_cut_tag <autopilot> <strict-rc> [<must-carry-rc> [<carry-rc> [<readiness> [<preflight-rc>]]]] -- extract cut_tag(), run
# it with stubs, print a transcript (SAY/DIE/GIT-TAG/GIT-PUSH lines, then the CALL order).
# Returns 2 if the function is missing.
run_cut_tag() {
    local ap=$1 grc=$2 mrc=${3:-0} crc=${4:-0} rdy=${5:-pass} prc=${6:-0} d fn
    d=$(mktemp -d) || return 2
    fn=$(awk '/^cut_tag\(\) \{/,/^\}/' "$ap")
    [ -n "$fn" ] || { rmtree "$d"; return 2; }
    mkdir -p "$d/scripts/release" "$d/ap" "$d/wt/scripts"
    # #3715 B1: the readiness step's log, as the T-1 `readiness` step leaves it (or does not)
    case "$rdy" in
        pass)   printf 'ok    R8 release-readiness-v1 for 0.0.0: Pass\nok    R8 #3715 ENFORCE PASS version=0.0.0 commit=deadbeef pv=pv_x out_sha256=0\n' > "$d/ap/readiness-t1.log" ;;
        report) printf 'WARN  R8 REPORT-ONLY release-readiness-v1 for 0.0.0: Fail, 3 violation(s)\n' > "$d/ap/readiness-t1.log" ;;
        stale)  printf 'ok    R8 #3715 ENFORCE PASS version=0.0.0 commit=cafef00d pv=pv_x out_sha256=0\n' > "$d/ap/readiness-t1.log" ;;
        absent) : ;;
    esac
    printf '#!/usr/bin/env bash\nif [ "${2:-}" = --must-carry ]; then echo CALL-MUST-CARRY >> %q; exit %s; fi\necho CALL-STRICT >> %q; exit %s\n' \
        "$d/calls" "$mrc" "$d/calls" "$grc" > "$d/scripts/check_milestone_cut.sh"
    printf '#!/usr/bin/env bash\necho CALL-CARRY >> %q\nexit %s\n' "$d/calls" "$crc" > "$d/scripts/release/carry_milestone_items.sh"
    # #4805: the unchanged publish preflight, run on the release worktree with the local tag in place
    printf '#!/usr/bin/env bash\n[ "${PUBLISH_PREFLIGHT_ROOT:-}" = %q ] || { echo "preflight asked about ${PUBLISH_PREFLIGHT_ROOT:-nothing}"; exit 3; }\n[ $# = 0 ] || { echo "preflight given a mode: $*"; exit 3; }\necho CALL-PREFLIGHT >> %q\nexit %s\n' \
        "$d/wt" "$d/calls" "$prc" > "$d/wt/scripts/check_publish_preflight.sh"
    {
        printf 'set -uo pipefail\n'
        printf 'REPO_ROOT=%q\nLOG=%q\nAP=%q\nWT=%q\n' "$d" "$d/log" "$d/ap" "$d/wt"
        printf 'say() { printf "SAY %%s\\n" "$*"; }\n'
        printf 'die() { printf "DIE %%s\\n" "$*"; exit 1; }\n'
        printf 'git() { printf "GIT-%%s %%s\\n" "$(printf %%s "$1" | tr "a-z" "A-Z")" "$*"; }\n'
        printf '%s\n' "$fn"
        printf 'cut_tag 0.0.0 v0.0.0 deadbeef\n'
    } > "$d/harness.sh"
    bash "$d/harness.sh" 2>&1
    cat "$d/log" 2>/dev/null
    printf 'ORDER %s\n' "$(tr '\n' ' ' 2>/dev/null < "$d/calls")"
    rmtree "$d"
}

# run_cut_release <autopilot> <gate-rc> [<pass-sha> [<tag-sha>]] -- #4805: extract cut_release(), run it
# with a stub release_gate.sh (rc, and a PASS line naming <pass-sha>) and a git whose rev-parse
# answers <tag-sha>. Prints SAY/DIE/GH lines and the CALL order. Returns 2 if the function is missing.
run_cut_release() {
    local ap=$1 grc=$2 psha=${3:-deadbeef} tsha=${4:-deadbeef} d fn
    d=$(mktemp -d) || return 2
    fn=$(awk '/^cut_release\(\) \{/,/^\}/' "$ap")
    [ -n "$fn" ] || { rmtree "$d"; return 2; }
    mkdir -p "$d/scripts/release" "$d/ap"
    printf '#!/usr/bin/env bash\n[ "${1:-}" = v0.0.0 ] && [ "${2:-}" = %q ] || { echo "release gate asked about $*"; exit 3; }\necho CALL-RELEASE-GATE >> %q\n[ %s = 0 ] && echo "RELEASE-GATE PASS preflight=PASS clean-room=PASS tag=v0.0.0 sha=%s"\nexit %s\n' \
        "$d/wt" "$d/calls" "$grc" "$psha" "$grc" > "$d/scripts/release/release_gate.sh"
    {
        printf 'set -uo pipefail\n'
        printf 'REPO_ROOT=%q\nLOG=%q\nAP=%q\nWT=%q\nREPO=o/r\n' "$d" "$d/log" "$d/ap" "$d/wt"
        printf 'say() { printf "SAY %%s\\n" "$*"; }\n'
        printf 'die() { printf "DIE %%s\\n" "$*"; exit 1; }\n'
        printf 'git() { [ "$1" = rev-parse ] && { echo %s; return 0; }; printf "GIT %%s\\n" "$*"; }\n' "$tsha"
        printf 'gh() { printf "GH-%%s %%s\\n" "$(printf %%s "$1" | tr "a-z" "A-Z")" "$*"; echo CALL-GH-RELEASE >> %q; }\n' "$d/calls"
        printf '%s\n' "$fn"
        printf 'cut_release v0.0.0 deadbeef\n'
    } > "$d/harness.sh"
    bash "$d/harness.sh" 2>&1
    printf 'ORDER %s\n' "$(tr '\n' ' ' 2>/dev/null < "$d/calls")"
    rmtree "$d"
}

# judge <autopilot> -- 0 when every gate outcome behaves; 1 otherwise. Prints rows.
judge() {
    local ap=$1 out bad=0
    if ! grep -q '^cut_tag() {' "$ap" || ! grep -q '^cut_release() {' "$ap"; then
        printf 'FAIL  %s has no cut_tag() or no cut_release() -- the tag path moved and this guard is judging nothing\n' "$ap" >&2
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
    if grep -q '^ORDER CALL-MUST-CARRY CALL-CARRY CALL-STRICT CALL-PREFLIGHT $' <<< "$out" && grep -q 'GIT-TAG' <<< "$out" && grep -q 'GIT-PUSH' <<< "$out"; then
        printf 'ok    all clean -> must-carry, the carry, STRICT, the local tag, the preflight, then the push\n'
    else printf 'FAIL  all clean did not run must-carry -> carry -> strict -> tag -> preflight -> push\n%s\n' "$out" >&2; bad=1; fi
    # #4805: the unchanged publish preflight red or unjudged on the local tag -> local tag removed, nothing pushed
    for p in 1 2; do
        out=$(run_cut_tag "$ap" 0 0 0 pass "$p") || true
        if grep -q 'GIT-PUSH' <<< "$out" || ! grep -q '^GIT-TAG tag -d v0.0.0' <<< "$out"; then
            printf 'FAIL  preflight rc=%s on the local tag -> the tag was PUSHED, or the local tag was left behind\n%s\n' "$p" "$out" >&2; bad=1
        else printf 'ok    preflight rc=%s on the local tag -> local tag removed, nothing pushed\n' "$p"; fi
    done
    # #4805: the GitHub release waits for the release gate (publish preflight + clean-room on the tag)
    out=$(run_cut_release "$ap" 0) || true
    if grep -q '^ORDER CALL-RELEASE-GATE CALL-GH-RELEASE $' <<< "$out"; then printf 'ok    release gate rc=0 -> gate, then the release\n'
    else printf 'FAIL  release gate rc=0 -> no release, or not gate-first\n%s\n' "$out" >&2; bad=1; fi
    for g in 1 2; do
        out=$(run_cut_release "$ap" "$g") || true
        if grep -q 'CALL-GH-RELEASE' <<< "$out"; then printf 'FAIL  release gate rc=%s -> A RELEASE WAS MADE\n%s\n' "$g" "$out" >&2; bad=1
        else printf 'ok    release gate rc=%s -> no release\n' "$g"; fi
    done
    out=$(run_cut_release "$ap" 0 cafef00d) || true
    if grep -q 'CALL-GH-RELEASE' <<< "$out"; then printf 'FAIL  release gate PASS for another sha -> A RELEASE WAS MADE\n%s\n' "$out" >&2; bad=1
    else printf 'ok    release gate PASS names another sha -> no release\n'; fi
    out=$(run_cut_release "$ap" 0 deadbeef cafef00d) || true
    if grep -q 'CALL-' <<< "$out"; then printf 'FAIL  tag names another commit -> the gate was asked or a release made\n%s\n' "$out" >&2; bad=1
    else printf 'ok    tag names another commit -> no gate, no release\n'; fi
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
    sed '/ENFORCE PASS for\|index(\$0, n) == 1/d; /no .#3715 ENFORCE PASS/d' "$SUBJECT" > "$d/m6.sh"
    if cmp -s "$SUBJECT" "$d/m6.sh"; then
        nok "MUTANT 6 could not be built -- the readiness check line did not match; vacuous"
    elif judge "$d/m6.sh" > "$d/m6.out" 2>&1; then
        nok "MUTANT 6 (readiness requirement deleted) PASSED"
    else
        ok "mutant 6: #3715 readiness requirement deleted -> RED"
    fi
    # #4805 mutants. Each must turn the judge RED, or the rows above cannot see the line they guard.
    mutant() { # mutant <n> <what> <sed-script>
        sed "$3" "$SUBJECT" > "$d/m$1.sh"
        if cmp -s "$SUBJECT" "$d/m$1.sh"; then nok "MUTANT $1 could not be built -- its line did not match; vacuous"
        elif judge "$d/m$1.sh" > "$d/m$1.out" 2>&1; then nok "MUTANT $1 ($2) PASSED"
        else ok "mutant $1: $2 -> RED"; fi
    }
    mutant 7 "preflight-at-tag call deleted" '/PUBLISH_PREFLIGHT_ROOT="\$WT" bash "\$WT\/scripts\/check_publish_preflight.sh"/d'
    mutant 8 "preflight-at-tag verdict discarded" 's#\(check_publish_preflight\.sh" >> "$LOG" 2>&1\) || rc=$?#\1 || true#'
    mutant 9 "release gate call deleted" '/scripts\/release\/release_gate\.sh" "\$t" "\$WT"/d'
    mutant 10 "release gate verdict + PASS-line check discarded" 's#\(release_gate\.sh" "$t" "$WT" >> "$LOG" 2>&1\) || rc=$?#\1 || true#; /grep -qx "RELEASE-GATE PASS/,/|| die "release gate passed/d'
    mutant 11 "release tag-identity check deleted" '/refs\/tags\/\${t}^{commit}/,/|| die "tag \$t does not name/d'
    # the release gate's own case table, same reason as the carry table below
    if bash "$ROOT/scripts/release/check_release_gate.sh" --self-test > "$d/rgate.out" 2>&1; then
        ok "check_release_gate.sh case table ($(grep -c '^ok ' "$d/rgate.out") rows)"
    else
        nok "check_release_gate.sh case table FAILED"; cat "$d/rgate.out" >&2
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
