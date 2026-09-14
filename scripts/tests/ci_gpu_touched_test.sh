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
#
#   bash scripts/tests/ci_gpu_touched_test.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 2
SCRIPT="scripts/ci_gpu_touched.sh"
WF=".github/workflows/ci.yml"

n=0
red=0
row() { # row <want-rc> <label> <grep -E pattern> <cmd...>
    local want=$1 label=$2 pat=$3 rc=0 out
    shift 3
    n=$((n + 1))
    out=$("$@" 2>&1) || rc=$?
    if [ "$rc" = "$want" ] && printf '%s\n' "$out" | grep -qE -- "$pat"; then
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
    python3 - "$WF" "$1" "$2" <<'PY'
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
row 0 "cuda-unit is capped at 15 minutes (row 67-D1)" \
    '^15$' jobq cuda-unit "job['timeout-minutes']"

for j in gpu-quick cuda-unit; do
    row 0 "$j runs ONLY on pull_request — never merge_group, never push (the queue stays under 20 min)" \
        "^True$" jobq "$j" "\"github.event_name == 'pull_request'\" in job.get('if','')"
    row 0 "$j fires only when the decision job said gpu_touched=1 (skipped, not queued, otherwise)" \
        "^True$" jobq "$j" "\"gpu_touched\" in job.get('if','') and 'gpu-touched' in job.get('needs',[])"
    row 0 "$j is NOT continue-on-error — an honest red on the PR is the point" \
        "^False$" jobq "$j" "bool(job.get('continue-on-error', False))"
    row 0 "$j preflights the self-hosted toolset before it touches the GPU (row 67-C2)" \
        "^True$" jobq "$j" "any('ci_self_hosted_preflight.sh' in (s.get('run') or '') for s in job['steps'])"
done

row 0 "the decision job itself runs on the clean-room pool, so a non-GPU PR never holds a GPU runner" \
    "clean-room" jobq gpu-touched "job['runs-on']"
row 0 "the decision job publishes gpu_touched as a job output" \
    "gpu_touched" jobq gpu-touched "sorted(job.get('outputs',{}))"

# ADVISORY in 0.67. This is the row that has to be DELETED, not edited, when
# 68-C3 / 68-D2 promote the jobs — which is the point of pinning it.
row 0 "neither GPU job is in gate.needs — advisory in 0.67 (68-C3 / 68-D2 promote them)" \
    "^ADVISORY$" jobq gate \
    "'PROMOTED' if ({'gpu-quick','cuda-unit'} & set(job.get('needs',[]))) else 'ADVISORY'"

row 0 "gpu-quick reuses cuda-nightly's yield-to-training step (a running training job wins)" \
    "^True$" jobq gpu-quick \
    "any('yielding to training' in (s.get('run') or '') for s in job['steps'])"

printf '\n%s checks, %s failed\n' "$n" "$red"
[ "$red" -eq 0 ]
