#!/usr/bin/env bash
# check_tag_step_gated.sh -- the release train's tag step must not be reachable without its
# gate (PMAT-3459, which first placed the milestone gate inside cut_tag).
# #4688: the milestone cut and the coverage receipt (#4691) left the release path; what this guard
# still runs is cut_tag's release-policy / #3715 readiness gate ahead of `git tag`.
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
#     all green -> tag is cut; readiness or policy refused -> no tag
# --self-test then builds MUTANTS (gate requirements deleted, verdicts discarded)
# and requires this guard to go RED on each. It also runs the carry script's own
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

# run_cut_tag <autopilot> <strict-rc> [<must-carry-rc> [<carry-rc> [<readiness> [<covjob-rc> [<policy> [<models>]]]]]] -- extract cut_tag(), run
# it with stubs, print a transcript (SAY/DIE/GIT-TAG/GIT-PUSH lines, then the CALL order).
# Returns 2 if the function is missing.
run_cut_tag() {
    local ap=$1 grc=$2 mrc=${3:-0} crc=${4:-0} rdy=${5:-pass} jrc=${6:-0} pol=${7:-none} mdl=${8:-crux} d fn pfn
    d=$(mktemp -d) || return 2
    fn=$(awk '/^cut_tag\(\) \{/,/^\}/' "$ap")
    [ -n "$fn" ] || { rmtree "$d"; return 2; }
    mkdir -p "$d/scripts/release" "$d/ap" "$d/scripts/lib" "$d/contracts"
    pfn=$(awk '/^ap_policy_applies\(\) \{/,/^\}/' "$ap")
    # the standing release policy: the release worktree's own reader (this checkout's copy) and a ladder
    # with no policy (none), one covering 0.0.0 (covers) or one the reader refuses (bad)
    cp -- "$ROOT/scripts/lib/release_policy.sh" "$ROOT"/scripts/lib/release_policy_*.awk "$d/scripts/lib/" || { rmtree "$d"; return 2; }
    {   printf 'ladder:\n'
        case "$pol" in
            covers|bad) printf '  release_policy:\n    name: crux-smoke\n    since: "0.0.0"\n    date: "d"\n    quote: "q"\n'
                printf '    hosts: [lambda, gx10]\n    thinking: ["off"]\n    larger_rows: nightly\n    red_row_needs: ticket\n    ticket_owner: "#1"\n'
                [ "$pol" = bad ] || printf '    release_notes: known_failures\n' ;;
        esac
        printf '  emergency_scopes:\n'
    } > "$d/contracts/model-capability-ladder-v1.yaml"
    # the models lane's log: a CRUX-smoke GO for this commit (crux), for another (stale), or none
    case "$mdl" in
        crux)  printf 'MODELS GO (CRUX smoke) on lambda and gx10 at deadbeef: the judge passed both receipts (apr 0.0.0 (deadbeef))\n' > "$d/ap/models-t1.log" ;;
        stale) printf 'MODELS GO (CRUX smoke) on lambda and gx10 at cafef00d: the judge passed both receipts (apr 0.0.0 (cafef00d))\n' > "$d/ap/models-t1.log" ;;
        ladder) printf 'MODELS GO on lambda and gx10 at deadbeef: the judge passed both receipts (apr 0.0.0 (deadbeef))\n' > "$d/ap/models-t1.log" ;;
        absent) : ;;
    esac
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
    printf '#!/usr/bin/env bash\necho CALL-COVJOB >> %q\nexit %s\n' "$d/calls" "$jrc" > "$d/scripts/release/tag_coverage_gate.sh"
    {
        printf 'set -uo pipefail\n'
        printf 'REPO_ROOT=%q\nLOG=%q\nAP=%q\n' "$d" "$d/log" "$d/ap"
        printf 'say() { printf "SAY %%s\\n" "$*"; }\n'
        printf 'die() { printf "DIE %%s\\n" "$*"; exit 1; }\n'
        printf 'git() { printf "GIT-%%s %%s\\n" "$(printf %%s "$1" | tr "a-z" "A-Z")" "$*"; }\n'
        printf '%s\n' "$pfn"
        printf '%s\n' "$fn"
        printf 'cut_tag 0.0.0 v0.0.0 deadbeef\n'
    } > "$d/harness.sh"
    (cd "$d" && bash "$d/harness.sh" 2>&1)
    cat "$d/log" 2>/dev/null
    printf 'ORDER %s\n' "$(tr '\n' ' ' 2>/dev/null < "$d/calls")"
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

    # #3715 B1: no ENFORCED readiness Pass for exactly this version+commit -> no tag, nothing carried
    for r in absent report stale; do
        out=$(run_cut_tag "$ap" 0 0 0 "$r") || true
        if grep -q 'GIT-TAG' <<< "$out" || grep -q 'CALL-' <<< "$out"; then
            printf 'FAIL  readiness %s -> a tag was cut or the milestone was touched without an enforced #3715 Pass\n%s\n' "$r" "$out" >&2; bad=1
        else printf 'ok    readiness %s -> no tag, nothing carried\n' "$r"; fi
    done
    # the standing release policy (contracts/model-capability-ladder-v1.yaml `ladder.release_policy`):
    # a covered version tags on the models lane's CRUX-smoke GO for exactly this commit, with no readiness run
    out=$(run_cut_tag "$ap" 0 0 0 absent 0 covers crux) || true
    if grep -q 'GIT-TAG' <<< "$out" && grep -q '^SAY POLICY-GATE ' <<< "$out"; then
        printf 'ok    policy covers, CRUX-smoke GO at this commit, readiness not run -> the tag is cut\n'
    else printf 'FAIL  policy covers + CRUX-smoke GO -> NO tag (or no POLICY-GATE line)\n%s\n' "$out" >&2; bad=1; fi
    for m in stale ladder absent; do
        out=$(run_cut_tag "$ap" 0 0 0 pass 0 covers "$m") || true
        if grep -q 'GIT-TAG' <<< "$out" || grep -q 'CALL-' <<< "$out"; then
            printf 'FAIL  policy covers, models log %s -> a tag was cut or the milestone was touched without a CRUX-smoke GO\n%s\n' "$m" "$out" >&2; bad=1
        else printf 'ok    policy covers, models log %s -> no tag, nothing carried (a readiness Pass does not stand in)\n' "$m"; fi
    done
    out=$(run_cut_tag "$ap" 0 0 0 pass 0 bad crux) || true
    if grep -q 'GIT-TAG' <<< "$out" || grep -q 'CALL-' <<< "$out" || ! grep -q '^DIE the standing release policy cannot be judged' <<< "$out"; then
        printf 'FAIL  unreadable policy block -> a tag was cut, the milestone was touched, or the refusal named another cause\n%s\n' "$out" >&2; bad=1
    else printf 'ok    unreadable policy block -> no tag, nothing carried (not measured is not a pass)\n'; fi
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

    # M6 (#3715 B1): the readiness requirement deleted -> a skipped or report-mode readiness step tags.
    sed '/ENFORCE PASS for\|index(\$0, n) == 1/d; /no .#3715 ENFORCE PASS/d' "$SUBJECT" > "$d/m6.sh"
    if cmp -s "$SUBJECT" "$d/m6.sh"; then
        nok "MUTANT 6 could not be built -- the readiness check line did not match; vacuous"
    elif judge "$d/m6.sh" > "$d/m6.out" 2>&1; then
        nok "MUTANT 6 (readiness requirement deleted) PASSED"
    else
        ok "mutant 6: #3715 readiness requirement deleted -> RED"
    fi
    # M9: under the policy, the CRUX-smoke GO requirement discarded -> a stale or absent models GO tags.
    sed 's/|| die "the standing release policy covers \$v but/|| true; : "/' "$SUBJECT" > "$d/m9.sh"
    if cmp -s "$SUBJECT" "$d/m9.sh"; then
        nok "MUTANT 9 could not be built -- the CRUX-smoke GO die line did not match; vacuous"
    elif judge "$d/m9.sh" > "$d/m9.out" 2>&1; then
        nok "MUTANT 9 (CRUX-smoke GO requirement discarded) PASSED"
    else
        ok "mutant 9: CRUX-smoke GO requirement discarded -> RED"
    fi
    # M10: the policy verdict discarded -> an unreadable policy block reads as "no policy".
    sed 's/|| die "the standing release policy cannot be judged/|| true; : "/' "$SUBJECT" > "$d/m10.sh"
    if cmp -s "$SUBJECT" "$d/m10.sh"; then
        nok "MUTANT 10 could not be built -- the policy-judge die line did not match; vacuous"
    elif judge "$d/m10.sh" > "$d/m10.out" 2>&1; then
        nok "MUTANT 10 (policy verdict discarded) PASSED"
    else
        ok "mutant 10: policy verdict discarded -> RED"
    fi
    # M11: the policy branch never taken -> a covered release still demands the readiness run it replaced.
    sed 's/^    if \[ "\$pol" = 1 \]; then$/    if false; then/' "$SUBJECT" > "$d/m11.sh"
    if cmp -s "$SUBJECT" "$d/m11.sh"; then
        nok "MUTANT 11 could not be built -- the policy branch line did not match; vacuous"
    elif judge "$d/m11.sh" > "$d/m11.out" 2>&1; then
        nok "MUTANT 11 (policy branch never taken) PASSED"
    else
        ok "mutant 11: policy branch never taken -> RED"
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

echo "=== the tag step cannot be reached without its readiness or policy gate (check_tag_step_gated.sh) ==="
judge "$SUBJECT"; rc=$?
[ "$rc" -eq 0 ] && echo "PASS" || echo "FAIL: the tag step is reachable without its gate (rc=$rc)" >&2
exit "$rc"
