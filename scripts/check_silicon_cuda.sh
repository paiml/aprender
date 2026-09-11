#!/usr/bin/env bash
# An axis whose id says `cuda` must actually invoke CUDA.
#
# WHY THIS EXISTS (paiml/infra PMAT-272, YOGA-NIGHTLY-001 §9.3 / R-6).
#
# `.github/silicon-coverage.txt` scores an axis covered when a RUNNER can serve
# its selector. `scripts/check_silicon_coverage.sh` asks exactly that question
# and answers it correctly. Neither asks what the job then RUNS.
#
# Measured 2026-09-09: `--features cuda` appears **zero** times in the whole of
# .github/workflows/silicon-nightly.yml, while the file declares an axis named
# `aarch64-cuda-sm121` that holds gx10 — the fleet's only Blackwell box — for a
# 90-minute timeout to run `cargo test -p aprender-compute --lib`. That is a CPU
# test suite. The ledger reads the axis as covered; the coverage is of the CPU.
#
# It is the fleet's signature defect wearing the exact vocabulary of the tool
# built to prevent it: a name that asserts a property nothing measures. And it
# is about to be duplicated — yoga (x86_64 + Ada sm_89) is being added, and the
# obvious move is to copy the sm_121 leg's shape, which would produce a second
# `*-cuda-*` axis that runs no CUDA and reads green.
#
# THE RULE. For every job in silicon-nightly.yml whose axis id contains `cuda`,
# its `run:` block must invoke cargo with `--features cuda` (or a feature list
# containing `cuda`). Zero such invocations is a violation, named by axis.
#
# NOT A YAML PARSER, same posture as check_silicon_coverage.sh: this reads the
# workflow textually and answers exactly one question per cuda-named job.
#
# THE DENOMINATOR IS PRINTED. "0 violations" over 0 axes examined is the failure
# this repo has already had once; a run that cannot find any cuda-named axis is
# a NO-GO (exit 2), never a pass.
#
# SELF-TEST FIRST, against committed fixtures, and it exits 2 if the matcher
# cannot tell them apart. One fixture per FORM VARIANT, not per form:
# `--features cuda`, `--features "cuda,foo"`, `--all-features`, and a cuda-named
# axis with no cuda invocation at all.
set -uo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
WF="${SILICON_WF:-$HERE/.github/workflows/silicon-nightly.yml}"
FIXTURES="${SILICON_CUDA_FIXTURES:-$HERE/.github/fixtures/silicon-cuda-guard}"

# Does this job block invoke CUDA? Reads the block, answers yes/no.
#   $1 = file, $2 = job id
job_invokes_cuda() {
    _f="$1"; _job="$2"
    # The block runs from `  <job>:` to the next job at the same indent.
    awk -v job="  $_job:" '
        $0 == job { inblock = 1; next }
        inblock && /^  [A-Za-z0-9_-]+:$/ { exit }
        inblock { print }
    ' "$_f" | grep -cE -- '--features[= ]"?[^"]*cuda|--all-features'
}

# Every job id declared under `jobs:` — one per line.
job_ids() {
    awk '
        /^jobs:[[:space:]]*$/ { injobs = 1; next }
        injobs && /^[A-Za-z_]/ { exit }
        injobs && /^  [A-Za-z0-9_-]+:$/ { id = $1; sub(/:$/, "", id); print id }
    ' "$1"
}

selftest() {
    _bad=0
    for _spec in "cuda-features-ok.yml 0" \
                 "cuda-features-list-ok.yml 0" \
                 "cuda-all-features-ok.yml 0" \
                 "cuda-named-no-cuda.yml 1"; do
        # shellcheck disable=SC2086
        set -- $_spec
        _f="$FIXTURES/$1"; _want="$2"
        [ -f "$_f" ] || { printf '  %s: fixture missing\n' "$1"; _bad=$((_bad + 1)); continue; }
        _got=0
        for _j in $(job_ids "$_f"); do
            case "$_j" in *cuda*) ;; *) continue ;; esac
            [ "$(job_invokes_cuda "$_f" "$_j")" -ge 1 ] || _got=1
        done
        if [ "$_got" != "$_want" ]; then
            printf '  %s: expected violations=%s, got %s\n' "$1" "$_want" "$_got"
            _bad=$((_bad + 1))
        fi
    done
    if [ "$_bad" -gt 0 ]; then
        printf 'INSTRUMENT BROKEN — the matcher cannot classify its own fixtures.\n' >&2
        printf 'Refusing to judge the workflow with a checker that failed its own tests.\n' >&2
        return 2
    fi
    printf 'self-test: 4 fixtures, matcher classifies all of them as specified\n'
    return 0
}

printf -- '-- instrument self-test --\n'
selftest || exit 2
[ "${1:-}" = "--self-test" ] && exit 0

printf -- '\n-- cuda-named axes in %s --\n' "${WF#"$HERE"/}"
[ -f "$WF" ] || { printf 'NO-GO: %s does not exist.\n' "$WF" >&2; exit 2; }

examined=0
violations=0
for job in $(job_ids "$WF"); do
    case "$job" in *cuda*) ;; *) continue ;; esac
    examined=$((examined + 1))
    if [ "$(job_invokes_cuda "$WF" "$job")" -ge 1 ]; then
        printf '  ok        %-24s invokes cargo with the cuda feature\n' "$job"
    else
        printf '  VIOLATION %-24s axis is named `cuda` and invokes CUDA NOWHERE\n' "$job"
        violations=$((violations + 1))
    fi
done

printf '\n%s cuda-named axis/axes examined, %s violation(s)\n' "$examined" "$violations"

if [ "$examined" -eq 0 ]; then
    printf '\nNO-GO: no cuda-named axis found in %s.\n' "$WF" >&2
    printf 'Nothing was measured. "0 violations" over 0 axes is the shape this\n' >&2
    printf 'guard exists to refuse — a renamed axis must not read as a clean bill.\n' >&2
    exit 2
fi

if [ "$violations" -gt 0 ]; then
    printf '\nFAIL: an axis named `cuda` that runs no CUDA is a coverage claim with\n' >&2
    printf 'nothing behind it. The ledger scores it covered because a runner can\n' >&2
    printf 'serve the selector; this asks what the job actually runs.\n' >&2
    exit 1
fi

printf '\nOK: every cuda-named axis invokes CUDA.\n'
