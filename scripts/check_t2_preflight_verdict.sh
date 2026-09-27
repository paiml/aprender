#!/usr/bin/env bash
# check_t2_preflight_verdict.sh — the T-0 preflight's verdict must require positive
# evidence that the gate RAN, never merely the absence of complaints (#3561).
#
# THE DEFECT, measured 2026-09-20. scripts/release/t2_preflight.sh decided
#     if [ "$rc" -eq 0 ] || [ "$fails" -eq 0 ]; then GO
# where `fails` counts [FAIL] ROWS. dogfood.sh defines its row emitter mark() at line
# 203 and can exit before it, so a crash produces a log with ZERO rows. Zero rows =>
# fails=0 => the `||` is satisfied => GO, on the gate that guards publishing to
# crates.io, with dogfood_rc=2 printed in the very line that says GO.
#
# WHY THIS GUARD IS A SEPARATE FILE. guard_tree.sh's universe is exactly
# `git ls-files 'scripts/check_*.sh'` — a FLAT glob on scripts/, so a self-test living
# in scripts/release/ is discovered by nothing and runs nowhere. A facility with a
# self-test and no caller proves nothing. This wrapper gives it the caller, inside the
# guard-tree job, without a .github/workflows edit.
#
#   check_t2_preflight_verdict.sh              run the verdict case table
#   check_t2_preflight_verdict.sh --self-test  prove the case table can still turn RED
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
SUBJECT="$ROOT/scripts/release/t2_preflight.sh"

[ -r "$SUBJECT" ] || { echo "check_t2_preflight_verdict: ENV no $SUBJECT — the guard did not run" >&2; exit 2; }
grep -q '^decide() {' "$SUBJECT" || { echo "check_t2_preflight_verdict: ENV $SUBJECT has no decide() — the subject moved, this guard is judging nothing" >&2; exit 2; }

if [ "${1:-}" = "--self-test" ]; then
    echo "=== the verdict case table must still turn RED (mutation) ==="
    d=$(mktemp -d) || exit 2
    # SEC011: validate before rm -rf. An empty or '/' value must never reach it.
    cleanup() { case "${d:-}" in ''|/) return 0 ;; *) [ -d "$d" ] && rm -rf -- "$d" ;; esac; }
    trap cleanup EXIT
    mkdir -p "$d/scripts/release"
    # MUTANT: restore the pre-#3561 condition — absence of [FAIL] rows is read as consent.
    cat > "$d/mutant-decide" <<'MUTANT'
decide() {
    local sha=$1 log=$2 rc=$3 fails
    fails=$(grep -E '^[[:space:]]*\[FAIL\]' "$log" | grep -vcE 'version-unpublished' || true)
    if [ "$rc" -eq 0 ] || [ "$fails" -eq 0 ]; then
        printf 'GO %s dogfood_rc=%s fails_excluding_version_row=%s\n' "$sha" "$rc" "$fails"; return 0
    fi
    printf 'NO-GO %s dogfood_rc=%s fails=%s\n' "$sha" "$rc" "$fails"; return 1
}
MUTANT
    # Spliced over the subject's decide(): its `decide() {` line through the first line that is
    # exactly `}`. awk, not python (#3697): a subject with no closing line is ENV, never a mutant.
    awk -v mf="$d/mutant-decide" '
        !done && /^decide\(\) \{$/ { while ((getline l < mf) > 0) print l; skip = 1; next }
        skip { if ($0 == "}") { skip = 0; done = 1 }; next }
        { print }
        END { if (!done) exit 3 }' "$SUBJECT" > "$d/scripts/release/t2_preflight.sh" || exit 2
    bash "$d/scripts/release/t2_preflight.sh" --self-test > "$d/mut.out" 2>&1; mrc=$?
    if [ "$mrc" -eq 0 ]; then
        echo "FAIL  the pre-#3561 mutant PASSED the case table — the table does not discriminate" >&2
        sed -n '1,20p' "$d/mut.out" >&2; exit 1
    fi
    n=$(grep -c '^FAIL' "$d/mut.out" || true)
    echo "ok    mutant (absence-as-consent) turns the table RED: rc=$mrc, $n failing case(s)"
    grep -m1 'CRASH, zero rows' "$d/mut.out" | sed 's/^/      /'
    # and the real subject must be GREEN, or a red subject would masquerade as a red mutant
    bash "$SUBJECT" --self-test > "$d/real.out" 2>&1; rrc=$?
    [ "$rrc" -eq 0 ] && echo "ok    the real subject is GREEN (rc=0), so the RED above is the mutant's" \
        || { echo "FAIL  the real subject is itself RED (rc=$rrc) — the mutation proves nothing" >&2; exit 1; }
    echo "SELF-TEST PASSED"
    exit 0
fi

echo "=== T-0 preflight verdict: GO needs positive evidence (check_t2_preflight_verdict.sh) ==="
bash "$SUBJECT" --self-test; rc=$?
[ "$rc" -eq 0 ] && echo "PASS" || echo "FAIL: the verdict case table is RED (rc=$rc)" >&2
exit "$rc"
