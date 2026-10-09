#!/usr/bin/env bash
# ci_gpu_touched_test.sh — the case table for the two advisory GPU PR jobs
# (PMAT-1098, spec docs/specifications/06x-release-schedule.md §2 rows 67-C1
# and 67-D1).
#
# WHAT IS PINNED HERE, AND WHY IT IS NOT ci_gpu_touched.sh --self-test ALONE
#   The decision script owns its own case table (`--self-test`); this file runs
#   it AND pins the two things a self-test cannot see:
#     (a) the MUTATION the spec registers — drop `aprender-serve` from the GPU
#         crate set and the row that expects `gpu_touched=1` for a serve diff
#         must go RED. A case table that survives its own mutation is theater.
#     (b) the WORKFLOW shape. The decision is only worth anything if ci.yml
#         actually gates the two GPU jobs on it, on the runners the operator
#         named, off the merge queue, and OUT of `gate.needs` (advisory in
#         0.67; rows 68-C3 / 68-D2 promote them to required after five
#         consecutive green nightlies). Every one of those is a property of
#         the YAML, so it is read from the YAML.
#     (c) Q6 quorum, C324 (DELETE-TO-LAB 3/3): the two jobs are LAB. They left
#         ci.yml (no PR, merge-queue or release job runs them) and run only in
#         .github/workflows/gpu-lab-nightly.yml, on schedule and dispatch.
#
#   bash scripts/tests/ci_gpu_touched_test.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 2
SCRIPT="scripts/ci_gpu_touched.sh"
WF=".github/workflows/ci.yml"
NW=".github/workflows/gpu-lab-nightly.yml"   # Q6 quorum, C324: the LAB home
# #4433: the job BODIES moved verbatim to ci/sections.yml; gx10/yoga/gpu-touched
# in ci.yml are the fat jobs that run them and must keep the same decision shape.
SEC="ci/sections.yml"

n=0
red=0
row() { # row <want-rc> <label> <grep -E pattern> <cmd...>
    local want=$1 label=$2 pat=$3 rc=0 out
    shift 3
    n=$((n + 1))
    out=$("$@" 2>&1) || rc=$?
    if [ "$rc" = "$want" ] && grep -qE -- "$pat" <<< "$out"; then
        printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
    else
        printf 'FAIL  row %-2s rc=%s (wanted %s, must match /%s/)  %s\n' \
            "$n" "$rc" "$want" "$pat" "$label"
        printf '%s\n' "$out" | sed 's/^/        /'
        red=1
    fi
}

td=$(mktemp -d "${TMPDIR:-/tmp}/ci-gpu-touched.XXXXXX")
trap 'rm -rf "${td:?}"' EXIT

# --- 1. the decision script exists and its own case table is green -----------
row 0 "scripts/ci_gpu_touched.sh exists (the decision has a script, not a comment)" \
    '^PRESENT$' bash -c "[ -f '$SCRIPT' ] && echo PRESENT || echo ABSENT"
row 0 "its own case table is green (bash $SCRIPT --self-test)" \
    'checks, 0 failed' bash "$SCRIPT" --self-test

# --- 2. the registered mutation: drop aprender-serve from the GPU set --------
# The spec names this one explicitly. `aprender-serve` is the crate that holds
# every CUDA inference path, so a GPU selection that cannot see it is the exact
# shape of a gate that never fires.
printf 'crates/aprender-serve/src/gguf/cuda/matmul.rs\n' > "$td/serve.txt"
row 0 "a diff under crates/aprender-serve/ -> gpu_touched=1 naming the crate" \
    '^gpu_crates=.*aprender-serve' bash "$SCRIPT" --diff-from "$td/serve.txt"
sed 's/aprender-serve //' "$SCRIPT" > "$td/mutant.sh"
row 0 "MUTANT (crate list without aprender-serve) answers 0 — the row discriminates" \
    '^MUTANT-BLIND$' bash -c "
        if bash '$td/mutant.sh' --diff-from '$td/serve.txt' 2>/dev/null | grep -q '^gpu_touched=1'; then
            echo MUTANT-STILL-SEES-IT
        else
            echo MUTANT-BLIND
        fi"

# --- 3. the workflow shape (rows 67-C1 / 67-D1) ------------------------------
jobq() { # jobq <job> <python expr over `job`>
    jobq_in "$SEC" "$@"
}
jobq_wf() { # jobq_wf <job> <python expr over `job`> -- a job in the workflow itself
    jobq_in "$WF" "$@"
}
jobq_nw() { # jobq_nw <job> <python expr over `job`> -- a job in the LAB nightly
    jobq_in "$NW" "$@"
}
jobq_in() {
    python3 - "$1" "$2" "$3" <<'PY'
import sys, yaml
wf, name, expr = sys.argv[1], sys.argv[2], sys.argv[3]
doc = yaml.safe_load(open(wf, encoding="utf-8")) or {}
job = (doc.get("jobs") or {}).get(name)
if job is None:
    print("NO-SUCH-JOB %s" % name); sys.exit(1)
print(eval(expr, {"job": job, "doc": doc}))
PY
}

row 0 "gpu-quick runs on the disposable gx10 runner, literally labelled" \
    "^\['self-hosted', 'Linux', 'ARM64', 'cuda', 'gx10', 'ephemeral', 'docker'\]$" \
    jobq gpu-quick "job['runs-on']"
row 0 "cuda-unit runs on the disposable yoga runner, literally labelled" \
    "^\['self-hosted', 'Linux', 'X64', 'cuda', 'yoga', 'ephemeral', 'docker'\]$" \
    jobq cuda-unit "job['runs-on']"
row 0 "gpu-quick is capped at 15 minutes (row 67-C1)" \
    '^15$' jobq gpu-quick "job['timeout-minutes']"
# 15 (row 67-D1) -> 35 (#3810, measured) -> 50 (#3636, the clippy step). Both
# raises left this row at 15; it is not wired into CI, so nothing went red.
row 0 "cuda-unit is capped at 50 minutes (67-D1, raised by #3810 and #3636)" \
    '^50$' jobq cuda-unit "job['timeout-minutes']"
# #4336: an apr-cli-only diff (cuda_lint=1, gpu_touched=0) starts cuda-unit for
# the clippy step alone, so every OTHER cargo step must carry a gpu_touched if:.
row 0 "cuda-unit also starts on cuda_lint=1 (apr-cli-only diff, #4336)" \
    "^True$" jobq cuda-unit "'cuda_lint' in job.get('if','')"
row 0 "every cuda-unit cargo TEST step is gated on gpu_touched — cuda_lint alone never runs them" \
    "^True$" jobq cuda-unit "all('gpu_touched' in (s.get('if') or '') for s in job['steps'] if 'cargo test' in (s.get('run') or '') or 'cuda unit tests' in (s.get('name') or ''))"
row 0 "the clippy --features cuda step has NO step if: — it runs whenever the job does" \
    "^True$" jobq cuda-unit "any('check_clippy_cuda.sh' in (s.get('run') or '') and not s.get('if') for s in job['steps'])"

for j in gpu-quick cuda-unit; do
    row 0 "$j runs ONLY on schedule / workflow_dispatch — LAB, never pull_request, merge_group or push (Q6, C324)" \
        "^True$" jobq "$j" "\"(github.event_name == 'schedule' || github.event_name == 'workflow_dispatch') &&\" in job.get('if','') and not any(e in job.get('if','') for e in ('pull_request', 'merge_group', \"'push'\"))"
    row 0 "$j is gated on the decision job's gpu_touched (cuda-unit ALSO on cuda_lint, #4336; skipped, not queued, otherwise)" \
        "^True$" jobq "$j" "\"gpu_touched\" in job.get('if','') and 'gpu-touched' in job.get('needs',[])"
    row 0 "$j is NOT continue-on-error — an honest red on the PR is the point" \
        "^False$" jobq "$j" "bool(job.get('continue-on-error', False))"
    row 0 "$j preflights the self-hosted toolset before it touches the GPU (row 67-C2)" \
        "^True$" jobq "$j" "any('ci_self_hosted_preflight.sh' in (s.get('run') or '') for s in job['steps'])"
done

row 0 "the decision section is documented on the clean-room pool" \
    "clean-room" jobq gpu-touched "job['runs-on']"
row 0 "the decision section publishes gpu_touched as a job output" \
    "gpu_touched" jobq gpu-touched "sorted(job.get('outputs',{}))"
row 0 "the decision section is LAB too: schedule / workflow_dispatch only (Q6, C324)" \
    "^True$" jobq gpu-touched "job.get('if','') == \"github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'\""
row 0 "the decision section's own case table still runs before it answers" \
    "^True$" jobq gpu-touched "'ci_gpu_touched.sh --self-test' in (job['steps'][1].get('run') or '')"

# Q6 quorum, C324: no job of ci.yml runs a GPU LAB section, and none is named gx10/yoga/gpu-touched.
row 0 "ci.yml has no gpu-touched / gx10 / yoga job (they left the PR path)" \
    "^NONE$" jobq_in "$WF" gate \
    "'NONE' if not ({'gpu-touched','gx10','yoga','gpu-quick','cuda-unit'} & set(doc['jobs'])) else sorted({'gpu-touched','gx10','yoga','gpu-quick','cuda-unit'} & set(doc['jobs']))"
row 0 "no ci.yml step runs a GPU LAB section through fat_driver" \
    "^NONE$" jobq_in "$WF" gate \
    "'NONE' if not __import__('re').search(r\"--sections '[^']*\\b(gpu-touched|gpu-quick|cuda-unit)\\b\", __import__('json').dumps(doc['jobs'])) else 'FOUND'"

# The LAB home: schedule + workflow_dispatch only, the fat jobs' runners and perf-* groups.
row 0 "gpu-lab-nightly.yml triggers on schedule and workflow_dispatch ONLY" \
    "^\['schedule', 'workflow_dispatch'\]$" jobq_nw gx10 "sorted(doc[True])"
row 0 "gx10 runs gpu-touched,gpu-quick on the disposable gx10 runner in perf-gx10" \
    "^True$" jobq_nw gx10 \
    "job['runs-on'] == ['self-hosted','Linux','ARM64','cuda','gx10','ephemeral','docker'] and job['concurrency']['group'] == 'perf-gx10' and any(\"--sections 'gpu-touched,gpu-quick'\" in (s.get('run') or '') for s in job['steps'])"
row 0 "yoga runs gpu-touched,cuda-unit on the disposable yoga runner in perf-yoga" \
    "^True$" jobq_nw yoga \
    "job['runs-on'] == ['self-hosted','Linux','X64','cuda','yoga','ephemeral','docker'] and job['concurrency']['group'] == 'perf-yoga' and any(\"--sections 'gpu-touched,cuda-unit'\" in (s.get('run') or '') for s in job['steps'])"

# ADVISORY in 0.67. This is the row that has to be DELETED, not edited, when
# 68-C3 / 68-D2 promote the jobs — which is the point of pinning it.
# The pin reads what the gate JUDGES, not only its needs: no gpu-quick or cuda-unit section result is read.
row 0 "no gpu-quick / cuda-unit result is read by gate — advisory in 0.67 (68-C3 / 68-D2 promote them)" \
    "^ADVISORY$" jobq_wf gate \
    "'PROMOTED' if (({'gpu-quick','cuda-unit','gx10'} & set(job.get('needs',[]))) or __import__('re').search(r'(:|\"\\$\\w+\" |[.]\")(gpu-quick|cuda-unit)\\b', ''.join(s.get('run') or '' for s in job['steps']))) else 'ADVISORY'"

row 0 "gpu-quick reuses cuda-nightly's yield-to-training step (a running training job wins)" \
    "^True$" jobq gpu-quick \
    "any('yielding to training' in (s.get('run') or '') for s in job['steps'])"

printf '\n%s checks, %s failed\n' "$n" "$red"
[ "$red" -eq 0 ]
