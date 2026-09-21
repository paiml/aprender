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
# decide <sha> <log> <dogfood_rc>
#   Prints ONE verdict line and returns 0 GO / 1 NO-GO / 2 UNKNOWN.
#   GO requires POSITIVE EVIDENCE the gate ran: dogfood's terminal VERDICT line, AND a
#   row count at or above the declared gate set, AND zero failures outside the documented
#   version-unpublished row. Any one missing is UNKNOWN with reason=<why> with exit 2 (#3561).
# ---------------------------------------------------------------------------
decide() {
    local sha=$1 log=$2 rc=$3 rows fails declared

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
    rows=$(grep -cE '^[[:space:]]*\[[A-Z]+\]' "$log")
    [ "$rows" -ge "$declared" ] || { unknown "rows=$rows<declared=$declared"; return; }

    # 3. only then does the absence of failures mean anything. version-unpublished is the
    #    one row that legitimately differs pre-bump (the version IS published).
    fails=$(grep -E '^[[:space:]]*\[FAIL\]' "$log" | grep -vcE 'version-unpublished' || true)
    if [ "$fails" -eq 0 ]; then
        printf 'GO %s dogfood_rc=%s rows=%s declared=%s fails_excluding_version_row=0\n' "$sha" "$rc" "$rows" "$declared"; return 0
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
        printf '  [PASS] dogfood-gates              %s declared gate(s) discovered and all green\n' "$n" > "$f"
        while [ "$i" -le "$n" ]; do printf '  [PASS] declared:gate%s  scripts/check_gate%s.sh\n' "$i" "$i" >> "$f"; i=$((i + 1)); done
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
    { printf '  [PASS] dogfood-gates              8 declared gate(s) discovered and all green\n'
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
# The verdict is decided by decide() above -- the same function the --self-test case table
# exercises, so the fixtures judge the code this path actually runs.
decide "$sha" "$AP/preflight-$sha.log" "$rc" > "$AP/preflight-$sha.verdict"
cat "$AP/preflight-$sha.verdict"; grep -E '^\s*\[FAIL\]' "$AP/preflight-$sha.log" | cut -c1-160
