#!/usr/bin/env bash
# t2_preflight.sh — APR-RELEASE-001 §4 T-0 (operator 2026-09-17): the pre-publish dogfood runs on origin/main HEAD
# BEFORE the bump PR opens; writes preflight-<sha>.verdict = "GO <sha>" or "NO-GO <sha>". prepare_bump.sh --ship refuses without GO.
#   t2_preflight.sh <version>     the train's state dir AP is derived from it (#3618)
#   t2_preflight.sh --self-test   decide()'s case table; needs no version
set -uo pipefail
# D1/D2/D3 (PMAT-3459): $0-derived root and CARGO_HOME-relative bin. Resolved BEFORE any cd.
# NOT `git rev-parse --show-toplevel` — git refuses a container bind-mounted tree (#3586).
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" || exit 2
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"

# ---------------------------------------------------------------------------
# decide <sha> <log> <dogfood_rc> [ladder_log]
#   Prints ONE verdict line and returns 0 GO / 1 NO-GO / 2 UNKNOWN.
#   GO requires POSITIVE EVIDENCE the gate ran: dogfood's terminal VERDICT line, AND a
#   row count at or above the declared gate set, AND zero failures outside the rows that
#   cannot be green before the bump (step 3 names them). Any one missing is UNKNOWN with
#   reason=<why> with exit 2 (#3561).
# ---------------------------------------------------------------------------

# ladder_unmeasurable_pre_bump <ladder_log> -> rc 0 when the model-ladder gate's OWN run (t2 re-runs it in
# the same worktree) is red for exactly one reason: the CRUX scope found no receipt from the release binary
# on a named host. That receipt is written by T-1's models lane for the release commit, after the bump, so
# on main's head it cannot exist. Any other red in that run (wrong-sha receipt, declined, unreadable, an
# unscoped ladder) is not this, and keeps NO-GO.
ladder_unmeasurable_pre_bump() {
    local l=$1
    grep -qE '^SCOPED: crux-smoke( |$)' "$l" 2> /dev/null || return 1
    [ "$(grep -cE '^RED ' "$l")" -eq 1 ] || return 1
    grep -qE '^RED   OPERATOR EMERGENCY SCOPE: CRUX smoke only -- NOT satisfied' "$l" || return 1
    grep -qE '^FAIL ' "$l" || return 1
    # Count, never `! a | grep -q`: under pipefail, grep -q exits at the first stray FAIL row, the
    # writer dies of SIGPIPE (141) on a log longer than the pipe, and `!` turns 141 into a pass.
    [ "$(grep -E '^FAIL ' "$l" | grep -cvE '^FAIL  host [A-Za-z0-9_.-]+ has no CRUX receipt from the release binary -- ')" -eq 0 ]
}

decide() {
    local sha=$1 log=$2 rc=$3 lad=${4:-} rows fails declared name red dn vrow line deferred=''

    # Doctrine 4: crash / missing input / never-asked are UNKNOWN with reason=<why>, exit 2 --
    # never GO, never a fabricated FAIL. GO requires POSITIVE EVIDENCE that the gate
    # ran, not merely the absence of complaints (#3561).
    unknown() { printf 'UNKNOWN %s dogfood_rc=%s reason=%s\n' "$sha" "$rc" "$1"; return 2; }

    [ -r "$log" ] || { unknown no-log; return; }

    # 1. dogfood's own terminal VERDICT line. dogfood.sh defines mark() at line 203 and
    #    can exit before it (the "cannot resolve this crate's identity" path and friends),
    #    which yields a log with zero rows AND no verdict line.
    grep -qE '^VERDICT:' "$log" || { unknown no-verdict-line; return; }

    # 2. a row floor DERIVED from the declared gate set, which dogfood prints inside a
    #    mark row -- so its absence is itself the crash signal, not a missing feature.
    declared=$(grep -oE '[0-9]+ declared gate\(s\) discovered' "$log" | head -1 | cut -d' ' -f1)
    [ -n "$declared" ] || { unknown no-declared-gate-count; return; }
    # dogfood.sh mark() prints a PASS row as `[ OK ]` (padded), every other status as `[FAIL]`,
    # `[SKIP]` ... -- a bare [A-Z]+ never matched the OK rows, so a green run sat under its own floor.
    rows=$(grep -cE '^[[:space:]]*\[( OK |[A-Z]+)\]' "$log")
    [ "$rows" -ge "$declared" ] || { unknown "rows=$rows<declared=$declared"; return; }

    # 3. only then does the absence of failures mean anything. A [FAIL] row is judged by its NAME
    #    (the field after the status), never by a word anywhere on the line. Pre-bump reds:
    #    - version-unpublished: the version on main's head IS published.
    #    - declared:check_model_ladder, D5 (E1 #3998): ONLY with the CRUX scope on the row, the version
    #      row red beside it (proof this is a pre-bump head), and the gate's own run red solely for
    #      missing release-binary receipts. T-1 judges the same row on the release commit; the verdict
    #      names it (deferred_to_T1=) so no reader takes this GO for a ladder pass.
    #    - dogfood-gates: the roll-up of the declared rows, each judged on its own row above; it is
    #      excused only when its RED count is at least 1 and equals the declared [FAIL] rows it
    #      summarises: a roll-up that is red with 0 RED is red for a reason no row above names.
    red=$(grep -E '^[[:space:]]*\[FAIL\]' "$log")
    dn=$(grep -cE '^[[:space:]]*\[FAIL\] declared:' <<< "$red")
    vrow=0
    if grep -qE '^[[:space:]]*\[FAIL\] version-unpublished ' <<< "$red"; then vrow=1; fi
    fails=0
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        read -r _ name _ <<< "$line"
        case "$name" in
            version-unpublished) continue ;;
            declared:check_model_ladder)
                if [[ "$line" == *' -- SCOPED: crux-smoke '* ]] && [ "$vrow" -eq 1 ] \
                   && ladder_unmeasurable_pre_bump "$lad"; then
                    deferred=check_model_ladder; continue
                fi ;;
            dogfood-gates)
                [[ "$line" =~ discovered,\ ([0-9]+)\ RED ]] && [ "${BASH_REMATCH[1]}" -ge 1 ] && [ "${BASH_REMATCH[1]}" -eq "$dn" ] && continue ;;
        esac
        fails=$((fails + 1))
    done <<< "$red"
    if [ "$fails" -eq 0 ]; then
        printf 'GO %s dogfood_rc=%s rows=%s declared=%s fails_excluding_version_row=0%s\n' "$sha" "$rc" "$rows" "$declared" "${deferred:+ deferred_to_T1=$deferred}"; return 0
    fi
    printf 'NO-GO %s dogfood_rc=%s fails=%s\n' "$sha" "$rc" "$fails"; return 1
}

# ---------------------------------------------------------------------------
# --self-test: the case table. Doctrine 4 -- crash / missing input / never-asked
# are UNKNOWN with reason=<why> with exit 2, never GO and never a fabricated FAIL.
# Row and VERDICT formats below are copied from a REAL dogfood log
# (rel-0682-autopilot/preflight-daad9f7c5....log), not invented.
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    d=$(mktemp -d) || exit 2
    # SEC011: validate before rm -rf. An empty or '/' value must never reach it.
    cleanup() { case "${d:-}" in ''|/) return 0 ;; *) [ -d "$d" ] && rm -rf -- "$d" ;; esac; }
    trap cleanup EXIT
    bad=0
    ok()  { printf 'ok    %s\n' "$*"; }
    nok() { printf 'FAIL  %s\n' "$*"; bad=$((bad + 1)); }

    # mklog <file> <declared-gate-count> [extra row ...]
    #   Builds a log in dogfood's real shape: the dogfood-gates row carrying the declared
    #   count, one row per declared gate, any extra rows, and the terminal VERDICT line LAST.
    mklog() {
        local f=$1 n=$2 i=1 r; shift 2
        printf '  [ OK ] dogfood-gates              %s declared gate(s) discovered and all green\n' "$n" > "$f"
        while [ "$i" -le "$n" ]; do printf '  [ OK ] declared:gate%s  scripts/check_gate%s.sh\n' "$i" "$i" >> "$f"; i=$((i + 1)); done
        for r in "$@"; do printf '%s\n' "$r" >> "$f"; done
        printf 'VERDICT: GO (phase pre-publish)\n' >> "$f"
    }

    # 1. THE DEFECT (#3561): dogfood died before mark() at line 203, so the log has ZERO rows.
    : > "$d/empty.log"
    v=$(decide abc "$d/empty.log" 2); rc=$?
    case "$v" in GO*) nok "CRASH, zero rows, rc=2 -> '$v' (must not be GO)";; *) ok "crash with zero rows is not GO";; esac
    [ "$rc" -eq 2 ] && ok "crash returns 2 (UNKNOWN)" || nok "crash returned $rc, doctrine 4 wants 2"

    # 2. missing input: the log file does not exist at all
    v=$(decide abc "$d/nosuch.log" 2 2>/dev/null); rc=$?
    case "$v" in GO*) nok "MISSING log -> '$v' (must not be GO)";; *) ok "missing log is not GO";; esac
    [ "$rc" -eq 2 ] && ok "missing log returns 2 (UNKNOWN)" || nok "missing log returned $rc, wanted 2"

    # 3. truncated: rows present, but the run died before its terminal VERDICT line
    mklog "$d/full.log" 8; grep -v '^VERDICT:' "$d/full.log" > "$d/trunc.log"
    v=$(decide abc "$d/trunc.log" 2); rc=$?
    case "$v" in GO*) nok "TRUNCATED (no VERDICT line) -> '$v' (must not be GO)";; *) ok "no VERDICT line is not GO";; esac
    [ "$rc" -eq 2 ] && ok "truncated returns 2 (UNKNOWN)" || nok "truncated returned $rc, wanted 2"

    # 4. too few rows: a VERDICT line, but far fewer rows than the declared gate set
    { printf '  [ OK ] dogfood-gates              8 declared gate(s) discovered and all green\n'
      printf 'VERDICT: GO (phase pre-publish)\n'; } > "$d/thin.log"
    v=$(decide abc "$d/thin.log" 0); rc=$?
    case "$v" in GO*) nok "1 row vs 8 declared gates -> '$v' (must not be GO)";; *) ok "row count below the declared floor is not GO";; esac
    [ "$rc" -eq 2 ] && ok "below-floor returns 2 (UNKNOWN)" || nok "below-floor returned $rc, wanted 2"

    # 5. FIRST-GREEN PROOF: a complete run whose only FAIL is the documented version row.
    #    dogfood_rc is 1 here on purpose -- the version row makes dogfood exit non-zero, and
    #    that single row must NOT be allowed to block the bump.
    mklog "$d/go.log" 8 '  [FAIL] version-unpublished        aprender 0.68.2 is ALREADY in the crates.io index'
    v=$(decide abc "$d/go.log" 1); rc=$?
    case "$v" in GO*) ok "complete run, only version-unpublished red -> GO";; *) nok "wanted GO, got '$v'";; esac
    [ "$rc" -eq 0 ] && ok "GO returns 0" || nok "GO returned $rc"

    # 6. a real failure must still be NO-GO, not UNKNOWN -- the fix must not turn reds into UNKNOWNs
    mklog "$d/nogo.log" 8 '  [FAIL] declared:check_model_ladder  exit=1 capability_match FAIL'
    v=$(decide abc "$d/nogo.log" 1); rc=$?
    case "$v" in NO-GO*) ok "a genuine FAIL row -> NO-GO";; *) nok "wanted NO-GO, got '$v'";; esac
    [ "$rc" -eq 1 ] && ok "NO-GO returns 1" || nok "NO-GO returned $rc"

    # D5 (E1 #3998): the pre-bump ladder row. On main's head the gate reads the released version's CRUX
    # scope, bound to HEAD, and only T-1's models lane writes receipts for the release commit -- so its
    # red is as certain before the bump as version-unpublished's. Fixtures copied from a REAL run of
    # scripts/check_model_ladder.sh at origin/main ad8fafc794 and from the 0.70.2 T-2 log.
    LROW='  [FAIL] declared:check_model_ladder scripts/check_model_ladder.sh exit=1 -- SCOPED: crux-smoke (a recorded scope, not the full gate) — RED   OPERATOR EMERGENCY SCOPE'
    VROW='  [FAIL] version-unpublished        aprender 0.70.2 is ALREADY in the crates.io index — bump the version'
    # mkred <file> <declared> <red-count> [row ...]: dogfood's shape when declared gates are red
    mkred() {
        local f=$1 n=$2 m=$3 i=1 r; shift 3
        while [ "$i" -le "$n" ]; do printf '  [ OK ] declared:gate%s  scripts/check_gate%s.sh\n' "$i" "$i" >> "$f"; i=$((i + 1)); done
        for r in "$@"; do printf '%s\n' "$r" >> "$f"; done
        printf '  [FAIL] dogfood-gates              %s declared gate(s) discovered, %s RED (each named in its own row above)\n' "$((n + m))" "$m" >> "$f"
        printf 'VERDICT: ❌ NO-GO — fix the ROOT CAUSE (Toyota way), never bypass. Re-run dogfood.\n' >> "$f"
    }
    { printf -- '--- model capability ladder receipts for 0.70.2 (evidence/dogfood/models/0.70.2) ---\n'
      printf 'SCOPED: crux-smoke -- the emergency scope recorded for release 0.70.2 in contracts/model-capability-ladder-v1.yaml applies\n'
      printf 'FAIL  host lambda has no CRUX receipt from the release binary -- the smoke gate needs every named host\n'
      printf 'FAIL  host gx10 has no CRUX receipt from the release binary -- the smoke gate needs every named host\n'
      printf 'RED   OPERATOR EMERGENCY SCOPE: CRUX smoke only -- NOT satisfied (see FAIL rows)\n'
    } > "$d/lad.log"
    dec() { local l=$1 g=$2; shift 2; decide abc "$l" 1 "$g"; }

    # 7. THE DEFECT: only the version row and the pre-bump ladder row are red -> GO, and it says what it deferred
    mkred "$d/d5.log" 8 1 "$LROW" "$VROW"
    v=$(dec "$d/d5.log" "$d/lad.log"); rc=$?
    [ "$v" = 'GO abc dogfood_rc=1 rows=11 declared=9 fails_excluding_version_row=0 deferred_to_T1=check_model_ladder' ] && [ "$rc" -eq 0 ] \
        && ok "pre-bump ladder row + version row -> GO, deferred_to_T1 named" || nok "D5: wanted the exact deferred GO line, got '$v' rc=$rc"
    # 8. a ladder red that is not the scoped one stays NO-GO
    mkred "$d/c8.log" 8 1 '  [FAIL] declared:check_model_ladder scripts/check_model_ladder.sh exit=1 — capability_match FAIL' "$VROW"
    v=$(dec "$d/c8.log" "$d/lad.log"); case "$v" in NO-GO*) ok "unscoped ladder red -> NO-GO";; *) nok "unscoped ladder red -> '$v'";; esac
    # 9. the deferral is one row: any other declared red keeps NO-GO
    mkred "$d/c9.log" 7 2 "$LROW" '  [FAIL] declared:check_perf041_marker scripts/check_perf041_marker.sh exit=1 — stale' "$VROW"
    v=$(dec "$d/c9.log" "$d/lad.log"); case "$v" in NO-GO*) ok "scoped ladder + another declared red -> NO-GO";; *) nok "case 9 -> '$v'";; esac
    # 10. SCOPED text in a row that is not the ladder gate is not deferred
    mkred "$d/c10.log" 8 1 '  [FAIL] declared:check_ladder_provenance scripts/check_ladder_provenance.sh exit=1 -- SCOPED: crux-smoke (a recorded scope, not the full gate)' "$VROW"
    v=$(dec "$d/c10.log" "$d/lad.log"); case "$v" in NO-GO*) ok "SCOPED on another row -> NO-GO";; *) nok "case 10 -> '$v'";; esac
    # 11. the gate's own run shows a red that is not a missing receipt (a wrong-sha receipt) -> NO-GO
    sed 's/^FAIL  host gx10 .*/FAIL  host gx10 receipt apr sha 1234 is not the cut abc/' "$d/lad.log" > "$d/lad11.log"
    v=$(dec "$d/d5.log" "$d/lad11.log"); case "$v" in NO-GO*) ok "a CRUX red other than no-receipt -> NO-GO";; *) nok "case 11 -> '$v'";; esac
    # 12. no version-unpublished red = not provably pre-bump -> NO-GO
    mkred "$d/c12.log" 8 1 "$LROW" '  [ OK ] version-unpublished        0.70.3 absent'
    v=$(dec "$d/c12.log" "$d/lad.log"); case "$v" in NO-GO*) ok "scoped ladder red with the version unpublished -> NO-GO";; *) nok "case 12 -> '$v'";; esac
    # 13. the gate's own run is missing, or has no FAIL row at all -> NO-GO
    v=$(dec "$d/d5.log" "$d/nosuch.log"); case "$v" in NO-GO*) ok "no ladder re-run log -> NO-GO";; *) nok "case 13a -> '$v'";; esac
    grep -v '^FAIL' "$d/lad.log" > "$d/lad13.log"
    v=$(dec "$d/d5.log" "$d/lad13.log"); case "$v" in NO-GO*) ok "ladder re-run with zero FAIL rows -> NO-GO";; *) nok "case 13b -> '$v'";; esac
    # 13c. the re-run printed a second RED (a policy that could not be applied) -> NO-GO
    { cat "$d/lad.log"; printf 'RED   the standing release policy cannot be applied to 0.70.2 -- nothing was judged\n'; } > "$d/lad13c.log"
    v=$(dec "$d/d5.log" "$d/lad13c.log"); case "$v" in NO-GO*) ok "ladder re-run with a second RED -> NO-GO";; *) nok "case 13c -> '$v'";; esac
    # 13d. the re-run did not judge the CRUX scope (no SCOPED line) -> NO-GO
    grep -v '^SCOPED' "$d/lad.log" > "$d/lad13d.log"
    v=$(dec "$d/d5.log" "$d/lad13d.log"); case "$v" in NO-GO*) ok "ladder re-run without the SCOPED line -> NO-GO";; *) nok "case 13d -> '$v'";; esac
    # 13e. the re-run's one RED is not the CRUX scope's verdict -> NO-GO
    sed 's/^RED .*/RED   the recorded emergency scope for 0.70.2 is unusable/' "$d/lad.log" > "$d/lad13e.log"
    v=$(dec "$d/d5.log" "$d/lad13e.log"); case "$v" in NO-GO*) ok "ladder re-run whose RED is another reason -> NO-GO";; *) nok "case 13e -> '$v'";; esac
    # 14. the roll-up's RED count must equal the declared red rows it summarises
    mkred "$d/c14.log" 8 2 "$LROW" "$VROW"
    v=$(dec "$d/c14.log" "$d/lad.log"); case "$v" in NO-GO*) ok "roll-up 2 RED over 1 declared red row -> NO-GO";; *) nok "case 14 -> '$v'";; esac
    # 15. a second ladder row that is red and unscoped -> NO-GO
    mkred "$d/c15.log" 7 2 "$LROW" '  [FAIL] declared:check_model_ladder scripts/check_model_ladder.sh exit=1 — capability_match FAIL' "$VROW"
    v=$(dec "$d/c15.log" "$d/lad.log"); case "$v" in NO-GO*) ok "scoped + unscoped ladder rows -> NO-GO";; *) nok "case 15 -> '$v'";; esac
    # 16. version-unpublished inside another row's note is not that row
    mklog "$d/c16.log" 8 '  [FAIL] fmt                        see version-unpublished'
    v=$(decide abc "$d/c16.log" 1); case "$v" in NO-GO*) ok "'version-unpublished' in a note does not excuse the row -> NO-GO";; *) nok "case 16 -> '$v'";; esac
    # 17. a roll-up red with 0 RED summarises no declared row: it is red for its own reason -> NO-GO
    mkred "$d/c17.log" 8 0
    v=$(decide abc "$d/c17.log" 1); case "$v" in NO-GO*) ok "roll-up [FAIL] with 0 RED -> NO-GO";; *) nok "case 17 -> '$v'";; esac
    # 18. a host row without the release-binary clause is not the pre-bump reason -> NO-GO
    sed 's/^FAIL  host gx10 .*/FAIL  host gx10 has no CRUX receipt/' "$d/lad.log" > "$d/lad18.log"
    v=$(dec "$d/d5.log" "$d/lad18.log"); case "$v" in NO-GO*) ok "a host row without 'from the release binary --' -> NO-GO";; *) nok "case 18 -> '$v'";; esac
    # 19. the re-run judged another scope, one whose name only starts with crux-smoke -> NO-GO
    sed 's/^SCOPED: crux-smoke /SCOPED: crux-smoke-v2 /' "$d/lad.log" > "$d/lad19.log"
    v=$(dec "$d/d5.log" "$d/lad19.log"); case "$v" in NO-GO*) ok "ladder re-run SCOPED to crux-smoke-v2 -> NO-GO";; *) nok "case 19 -> '$v'";; esac
    # 20. one stray FAIL row, then more host rows than a pipe holds. `! a | grep -q` read grep -q's early
    #     exit, the writer's SIGPIPE (141) under pipefail, as "no stray row" -> GO. It must be NO-GO.
    { sed -n '1,2p' "$d/lad.log"
      printf 'FAIL  CRUX receipt gx10.json is from apr sha %s, not the release binary abc\n' "'1234'"
      yes 'FAIL  host lambda has no CRUX receipt from the release binary -- the smoke gate needs every named host' | head -n 20000
      grep '^RED ' "$d/lad.log"; } > "$d/lad20.log"
    v=$(dec "$d/d5.log" "$d/lad20.log"); case "$v" in NO-GO*) ok "a stray FAIL row ahead of 20000 host rows -> NO-GO";; *) nok "case 20 (SIGPIPE) -> '$v'";; esac

    [ "$bad" -eq 0 ] && { echo "self-test OK"; exit 0; }
    echo "self-test FAILED: $bad case(s)"; exit 1
fi

# shellcheck source=scripts/release/lib_release_params.sh
. "$REPO_ROOT/scripts/release/lib_release_params.sh" || exit 2
release_params "${1:-}" "$REPO_ROOT" || { echo "usage: t2_preflight.sh <version> | --self-test" >&2; exit 2; }
cd "$REPO_ROOT" || exit 2; git fetch -q origin main || exit 2
sha=$(git rev-parse origin/main); wt="$AP/preflight-wt"
[ -d "$wt" ] && git worktree remove --force "$wt" > /dev/null 2>&1
git worktree add --detach "$wt" "$sha" > /dev/null 2>&1 || exit 2
cd "$wt" || exit 2
export CARGO_TARGET_DIR="$REPO_ROOT/target"
bash scripts/dogfood.sh --phase pre-publish > "$AP/preflight-$sha.log" 2>&1; rc=$?
# D5: the model-ladder gate's own output, from the same worktree and HEAD. dogfood's row keeps only 96
# characters of it, too few to tell a missing receipt from a wrong one; decide() reads every line.
bash scripts/check_model_ladder.sh > "$AP/preflight-$sha.ladder.log" 2>&1
# The verdict is decided by decide() above -- the same function the --self-test case table
# exercises, so the fixtures judge the code this path actually runs.
decide "$sha" "$AP/preflight-$sha.log" "$rc" "$AP/preflight-$sha.ladder.log" > "$AP/preflight-$sha.verdict"
cat "$AP/preflight-$sha.verdict"; grep -E '^\s*\[FAIL\]' "$AP/preflight-$sha.log" | cut -c1-160
