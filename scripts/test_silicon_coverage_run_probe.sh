#!/usr/bin/env bash
# The run probe in scripts/check_silicon_coverage.sh, in both polarities and
# under the mutation that removes it (R-5, docs/specifications/yoga-nightly-job.md
# §9.2 — the file lives in paiml/infra).
#
# WHAT IS BEING PROVEN, AND WHY IT NEEDS A MUTATION
# -------------------------------------------------
# R-5 changed the guard'"'"'s question from "could a runner serve this axis?" to
# "did a job carrying it conclude?". Six committed fixtures show it answers the
# six distinguishable inputs correctly. Fixtures alone cannot show that the RUN
# PROBE is what produces those answers — a guard that failed for some unrelated
# reason would satisfy every red case and look identical.
#
# So the probe is deleted and the fixtures are re-run. `axis_evidence` is
# replaced, between its committed seam markers, by a stub returning a fixed
# fresh positive — the pre-R-5 world, where an axis is covered because a machine
# exists. The `uncovered` fixture MUST then go GREEN. If it stays red, something
# other than the probe is deciding and the change is decoration.
#
# This runs offline. No gh, no network, no runner: everything the guard reads
# from GitHub is a committed file under tests/fixtures/silicon-coverage/.
#
#   bash scripts/test_silicon_coverage_run_probe.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GUARD="$REPO_ROOT/scripts/check_silicon_coverage.sh"
FIXTURES="$REPO_ROOT/tests/fixtures/silicon-coverage"

[ -f "$GUARD" ] || { printf 'NO-GO: %s is missing.\n' "$GUARD"; exit 2; }
[ -d "$FIXTURES" ] || { printf 'NO-GO: %s is missing.\n' "$FIXTURES"; exit 2; }

TMP="$(mktemp -d)" || exit 2
trap 'rm -rf "${TMP:?}"' EXIT

# Freshness is measured against now, so the timestamps are stamped in HERE. A
# committed literal would turn `covered` into `stale` three days after it was
# written, and a fixture that changes its own verdict with the calendar proves
# whatever the calendar says.
FRESH="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STALE="$(date -u -d '-10 days' +%Y-%m-%dT%H:%M:%SZ)"

materialise() { # $1 = case name -> prints the scratch dir
    _m_src="$FIXTURES/$1"; _m_dst="$TMP/$1"
    mkdir -p "$_m_dst" || return 1
    cp "$_m_src/policy.txt" "$_m_src/runners.tsv" "$_m_dst/" || return 1
    sed -e "s|@FRESH@|$FRESH|g" -e "s|@STALE@|$STALE|g" \
        "$_m_src/jobs.tsv.in" > "$_m_dst/jobs.tsv" || return 1
    printf '%s' "$_m_dst"
}

# run_case <guard> <case> <want rc> <want verdict token>
run_case() {
    _rc_guard="$1"; _rc_case="$2"; _rc_want_rc="$3"; _rc_want="$4"
    _rc_dir="$(materialise "$_rc_case")" || { printf 'NO-GO: cannot materialise %s\n' "$_rc_case"; return 2; }
    _rc_out="$TMP/$_rc_case.out"
    GITHUB_STEP_SUMMARY=/dev/null bash "$_rc_guard" --fixture "$_rc_dir" > "$_rc_out" 2>&1
    _rc_got=$?
    _rc_line="$(grep -E "^  ${_rc_want} +x86_64-cuda-sm89" "$_rc_out")"
    if [ "$_rc_got" != "$_rc_want_rc" ] || [ -z "$_rc_line" ]; then
        printf '  FAIL %-10s want rc=%s verdict=%-9s got rc=%s\n' \
            "$_rc_case" "$_rc_want_rc" "$_rc_want" "$_rc_got"
        sed 's/^/       | /' "$_rc_out"
        return 1
    fi
    printf '  ok   %-10s rc=%s  %s\n' "$_rc_case" "$_rc_got" "$(printf '%s' "$_rc_line" | sed 's/^  *//')"
    return 0
}

printf '== the run probe decides coverage (R-5 §9.2) ==\n'
printf 'guard:    %s\n' "$GUARD"
printf 'fixtures: %s\n' "$FIXTURES"
printf 'stamped:  FRESH=%s  STALE=%s\n\n' "$FRESH" "$STALE"

# ── part 1: the shipped guard answers all five inputs ───────────────────────
printf -- '-- shipped guard --\n'
cases=0; failed=0
# case      rc  verdict      what it distinguishes
for row in \
    "covered   0 ok" \
    "uncovered 1 UNCOVERED" \
    "stale     1 STALE" \
    "ready     0 ready" \
    "promote   1 PROMOTE" \
    "wrongjob  1 UNCOVERED" \
    ; do
    set -- $row
    cases=$((cases + 1))
    run_case "$GUARD" "$1" "$2" "$3" || failed=$((failed + 1))
done

# ── part 2: delete the probe; `uncovered` must go green ─────────────────────
printf -- '\n-- mutation: axis_evidence replaced by a fixed positive --\n'
MUTANT="$TMP/mutant.sh"
cat > "$TMP/mutate.awk" <<'AWK'
BEGIN { skip = 0 }
$0 == "# MUTATION-SEAM-BEGIN axis_evidence" {
    print
    print "axis_evidence() {"
    print "    printf '%s\\tsuccess\\tMUTANT\\tno-probe\\n' \"" FRESH "\""
    print "}"
    skip = 1
    next
}
$0 == "# MUTATION-SEAM-END axis_evidence" { skip = 0 }
skip == 0 { print }
AWK
awk -v FRESH="$FRESH" -f "$TMP/mutate.awk" "$GUARD" > "$MUTANT" || exit 2

# A mutator that matched nothing produces a byte-identical copy and would
# "prove" the probe discriminates by re-running the very code it claims to have
# removed. Assert the edit happened, and that the result is still a script.
if cmp -s "$GUARD" "$MUTANT"; then
    printf 'NO-GO: the mutation changed nothing — the seam markers moved.\n'
    printf 'Restore `# MUTATION-SEAM-BEGIN axis_evidence` / `...-END ...` in %s.\n' "$GUARD"
    exit 2
fi
if ! bash -n "$MUTANT"; then
    printf 'NO-GO: the mutant is not valid shell; the mutation, not the guard, is broken.\n'
    exit 2
fi
printf '  mutant built: %s line(s) vs %s, syntax ok\n' \
    "$(wc -l < "$MUTANT" | tr -d ' ')" "$(wc -l < "$GUARD" | tr -d ' ')"


# The load-bearing assertion of this whole file: WITHOUT the probe, a runner
# that merely exists is scored as coverage and both red cases turn green.
# `wrongjob` is in here too, because the job-name filter lives INSIDE the probe:
# if it survived the mutation it would be some other code deciding.
mut_failed=0
for mcase in uncovered wrongjob; do
    if run_case "$MUTANT" "$mcase" 0 ok; then
        printf '  -> the probe is what turns `%s` red. Discriminating.\n' "$mcase"
    else
        printf '  -> MUTATION SURVIVED on `%s`: still red with the run probe deleted.\n' "$mcase"
        printf '     Something other than the probe is producing that verdict, so the\n'
        printf '     fixture proves nothing about R-5. Stop and find out what.\n'
        mut_failed=$((mut_failed + 1))
    fi
done

printf -- '\n-- denominators --\n'
printf '%s fixture case(s) checked, %s failed; 1 mutation applied over 2 case(s), %s survived\n' \
    "$cases" "$failed" "$mut_failed"

if [ "$failed" -ne 0 ] || [ "$mut_failed" -ne 0 ]; then
    printf '\nFAIL: the coverage guard does not classify its own committed cases, or the\n'
    printf 'run probe is not what decides them.\n'
    exit 1
fi
printf '\nOK: coverage is a RUN. Six cases classified, and deleting the probe makes\n'
printf 'both UNCOVERED cases go green.\n'
