#!/usr/bin/env bash
# cuda_nightly_yield_not_green.sh — two properties of .github/workflows/cuda-nightly.yml (#5030).
#
#   P1  NO PR PATH. cuda-nightly is a LAB lane (it cannot block a merge or a release), and a LAB
#       lane never runs inside a PR job (C324 rule 5). The `on:` block carries no `pull_request`,
#       `pull_request_target` or `push` trigger, and no live line names the `cuda-check` label.
#   P2  A YIELDED NIGHT IS NOT GREEN. Yield-to-training lives in its own `decide` job, and
#       `falsifiers` needs it and runs only on `needs.decide.outputs.proceed == 'true'`. A yield
#       therefore SKIPS the job that measures, and scripts/release/nightly_train.sh reads a skipped
#       job as not_measured. When the yield was a step inside `falsifiers`, the job concluded
#       SUCCESS on a night that measured nothing. Also refused: a `falsifiers` step that still reads
#       `steps.decide` (that step no longer exists there, so the expression is empty and every step
#       it gates is silently skipped — green again), and a second matrix leg in `decide` (its
#       `proceed` output would be last-writer-wins across legs).
#
# Text checks over the workflow, comment lines excluded. --self-test runs the real file (must be
# GREEN) and one planted mutant per rule (each must be RED).
#
# EXIT  0 green · 1 red · 64 usage
#
#   bash scripts/ci/cuda_nightly_yield_not_green.sh [WORKFLOW]
#   bash scripts/ci/cuda_nightly_yield_not_green.sh --self-test
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_WF="$ROOT/.github/workflows/cuda-nightly.yml"

# job_header WF JOB -> the job's own keys, from `  JOB:` to its `    steps:`, comments dropped.
job_header() {
    awk -v J="  $2:" '
        $0 == J { inj = 1; next }
        inj && /^  [A-Za-z0-9_-]+:[[:space:]]*$/ { exit }
        inj && /^    steps:/ { exit }
        inj && $0 !~ /^[[:space:]]*#/ { print }
    ' "$1"
}

# job_body WF JOB -> every live line of the job, comments dropped.
job_body() {
    awk -v J="  $2:" '
        $0 == J { inj = 1; next }
        inj && /^  [A-Za-z0-9_-]+:[[:space:]]*$/ { exit }
        inj && $0 !~ /^[[:space:]]*#/ { print }
    ' "$1"
}

# on_block WF -> the live lines of the top-level `on:` block.
on_block() {
    awk '
        /^on:/ { ino = 1; next }
        ino && /^[A-Za-z]/ { exit }
        ino && $0 !~ /^[[:space:]]*#/ { print }
    ' "$1"
}

check() {
    local wf="$1" red=0 hdr body
    [ -s "$wf" ] || { printf 'RED   %s: no such workflow, or empty\n' "$wf"; return 1; }

    # Here-strings and plain files, never `producer | grep -q`: under pipefail grep -q exits at
    # its first match, the producer takes SIGPIPE, and a FOUND match reads as not found.
    if command grep -qE '^  (pull_request|pull_request_target|push)[[:space:]]*:' <<<"$(on_block "$wf")"; then
        printf 'RED   P1 the on: block carries a pull_request/pull_request_target/push trigger\n'; red=1
    fi
    if awk '!/^[[:space:]]*#/ && /cuda-check/ { f = 1 } END { exit !f }' "$wf"; then
        printf 'RED   P1 a live line still names the cuda-check label\n'; red=1
    fi

    hdr="$(job_header "$wf" decide)"
    if [ -z "$hdr" ]; then
        printf 'RED   P2 no `decide` job\n'; red=1
    else
        command grep -qE '^      proceed:[[:space:]]*\$\{\{ *steps\.decide\.outputs\.proceed *\}\}' <<<"$hdr" \
            || { printf 'RED   P2 `decide` does not export outputs.proceed from its decide step\n'; red=1; }
        [ "$(command grep -cE '^          - name:' <<<"$hdr")" = 1 ] \
            || { printf 'RED   P2 `decide` must have exactly one matrix leg (its proceed output is per run, not per leg)\n'; red=1; }
    fi

    hdr="$(job_header "$wf" falsifiers)"
    body="$(job_body "$wf" falsifiers)"
    if [ -z "$hdr" ]; then
        printf 'RED   P2 no `falsifiers` job\n'; red=1
    else
        command grep -qE '^    needs:[[:space:]]*(decide|\[[^]]*\bdecide\b[^]]*\])[[:space:]]*$' <<<"$hdr" \
            || { printf 'RED   P2 `falsifiers` does not need `decide`\n'; red=1; }
        command grep -qF "needs.decide.outputs.proceed == 'true'" <<<"$hdr" \
            || { printf "RED   P2 the falsifiers job if: does not require needs.decide.outputs.proceed == 'true'\n"; red=1; }
        if command grep -q 'steps\.decide' <<<"$body"; then
            printf 'RED   P2 a `falsifiers` line reads steps.decide, which is empty there (every step it gates skips)\n'; red=1
        fi
    fi

    [ "$red" = 0 ] && printf 'GREEN %s: no PR path; a yielded night skips falsifiers\n' "$wf"
    return "$red"
}

self_test() {
    local td fails=0 rc m name want prog
    td="$(mktemp -d)"
    trap 'rm -rf "${td:?}"' RETURN
    cp "$DEFAULT_WF" "$td/real.yml"

    # name;want;sed program applied to the real file. Each mutant must turn the check RED, and
    # its log must carry `want` — the rule it plants against. rc=1 alone is not enough: the
    # first version's label-path mutant went RED through a P2 rule while the P1 cuda-check rule
    # it was aimed at could not fire at all (a piped grep -q under pipefail).
    local -a mutants=(
        'pr-trigger;pull_request/pull_request_target/push trigger;s/^  workflow_dispatch:$/  pull_request:\n    types: [labeled]\n  workflow_dispatch:/'
        'push-trigger;pull_request/pull_request_target/push trigger;s/^  workflow_dispatch:$/  push:\n  workflow_dispatch:/'
        'live-label;names the cuda-check label;s/^name: \(.*\)$/name: \1 cuda-check/'
        'label-path;names the cuda-check label;0,/^      needs\.decide\.outputs\.proceed == .true. }}$/s//      github.event.label.name == '"'"'cuda-check'"'"' }}/'
        'no-needs;does not need `decide`;s/^    needs: decide$//'
        'no-proceed-gate;job if: does not require;s/^      needs\.decide\.outputs\.proceed == .true. }}$/      true }}/'
        'stale-step-ref;reads steps.decide;0,/needs\.decide\.outputs\.proceed == .true.$/s//steps.decide.outputs.proceed == '"'"'true'"'"'/'
        'second-leg;exactly one matrix leg;0,/^            select: blackwell$/s//            select: blackwell\n          - name: other\n            select: other/'
        'no-output;does not export outputs.proceed;s/^      proceed: \${{ steps\.decide\.outputs\.proceed }}$//'
        'no-decide-job;no `decide` job;s/^  decide:$/  decidex:/'
    )

    if check "$td/real.yml" > "$td/real.log" 2>&1; then
        printf 'ok    real workflow is GREEN\n'
    else
        printf 'FAIL  real workflow is RED:\n'; sed 's/^/        /' "$td/real.log"; fails=$((fails + 1))
    fi

    for m in "${mutants[@]}"; do
        name="${m%%;*}"; m="${m#*;}"; want="${m%%;*}"; prog="${m#*;}"
        sed "$prog" "$td/real.yml" > "$td/$name.yml"
        if cmp -s "$td/real.yml" "$td/$name.yml"; then
            printf 'FAIL  mutant %s did not change the file (its anchor is gone)\n' "$name"; fails=$((fails + 1)); continue
        fi
        check "$td/$name.yml" > "$td/$name.log" 2>&1; rc=$?
        if [ "$rc" != 1 ]; then
            printf 'FAIL  mutant %s exited %s, want 1\n' "$name" "$rc"; fails=$((fails + 1))
        elif ! command grep -qF -- "$want" "$td/$name.log"; then
            printf 'FAIL  mutant %s is RED, but not by its rule (%s):\n' "$name" "$want"
            sed 's/^/        /' "$td/$name.log"; fails=$((fails + 1))
        else
            printf 'ok    mutant %s is RED by its rule (%s)\n' "$name" "$want"
        fi
    done

    : > "$td/empty.yml"
    check "$td/empty.yml" > /dev/null 2>&1; rc=$?
    if [ "$rc" = 1 ]; then printf 'ok    an empty workflow is RED\n'
    else printf 'FAIL  an empty workflow exited %s, want 1\n' "$rc"; fails=$((fails + 1)); fi

    if [ "$fails" = 0 ]; then printf 'self-test: PASS (%s mutants)\n' "${#mutants[@]}"; return 0; fi
    printf 'self-test: FAIL (%s)\n' "$fails"; return 1
}

case "${1:-}" in
    --self-test) self_test ;;
    -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}" ;;
    -*) printf 'usage: %s [WORKFLOW] | --self-test\n' "$0" >&2; exit 64 ;;
    *) check "${1:-$DEFAULT_WF}" ;;
esac
