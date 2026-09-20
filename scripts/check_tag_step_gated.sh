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
# --self-test then removes the gate call to build a MUTANT and requires this guard to
# go RED on it. A guard that cannot fail on the defect it names is theater.
#
#   check_tag_step_gated.sh              judge scripts/release/autopilot.sh
#   check_tag_step_gated.sh --self-test  case table + the gate-removed mutant
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2

# SEC011: never let an empty or '/' value reach `rm -rf`. One validated helper, used
# by every call site in this file, rather than the check repeated at each one.
rmtree() { case "${1:-}" in ''|/) return 0 ;; *) [ -d "$1" ] && rm -rf -- "$1" ;; esac; return 0; }
SUBJECT="$ROOT/scripts/release/autopilot.sh"

# run_cut_tag <autopilot> <gate-rc> -- extract cut_tag(), run it with stubs, print a
# transcript (SAY/DIE/GIT-TAG/GIT-PUSH lines). Returns 2 if the function is missing.
run_cut_tag() {
    local ap=$1 grc=$2 d fn
    d=$(mktemp -d) || return 2
    fn=$(awk '/^cut_tag\(\) \{/,/^\}/' "$ap")
    [ -n "$fn" ] || { rmtree "$d"; return 2; }
    mkdir -p "$d/scripts"
    printf '#!/usr/bin/env bash\nexit %s\n' "$grc" > "$d/scripts/check_milestone_cut.sh"
    {
        printf 'set -uo pipefail\n'
        printf 'REPO_ROOT=%q\nLOG=%q\n' "$d" "$d/log"
        printf 'say() { printf "SAY %%s\\n" "$*"; }\n'
        printf 'die() { printf "DIE %%s\\n" "$*"; exit 1; }\n'
        printf 'git() { printf "GIT-%%s %%s\\n" "$(printf %%s "$1" | tr "a-z" "A-Z")" "$*"; }\n'
        printf '%s\n' "$fn"
        printf 'cut_tag 0.0.0 v0.0.0 deadbeef\n'
    } > "$d/harness.sh"
    bash "$d/harness.sh" 2>&1
    cat "$d/log" 2>/dev/null
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
    return "$bad"
}

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
