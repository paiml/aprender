#!/usr/bin/env bash
# ci_gpu_touched.sh — does this PR touch the GPU set? (PMAT-1098, spec
# docs/specifications/06x-release-schedule.md §2 rows 67-C1 and 67-D1)
#
# WHY A SEPARATE SCRIPT AND A SEPARATE JOB
#   The two advisory GPU jobs (`gpu-quick` on gx10, `cuda-unit` on yoga) must be
#   SKIPPED on a PR that cannot affect the GPU paths — not queued and then
#   no-op'd. A job that is queued has already claimed the runner; on a fleet with
#   one GB10 and one RTX 4060 that is the same as holding the GPU. So the
#   decision is taken FIRST, on the clean-room pool, and published as a job
#   output the GPU jobs put in their `if:`.
#
# THE GPU SET
#   The five crates that own a CUDA/GPU code path, plus the workflows that DRIVE
#   the GPU hosts — a change to cuda-nightly.yml or to ci.yml's own GPU jobs is a
#   change to what runs on gx10 and yoga, and the cheapest place to find out that
#   it is wrong is the PR that made it.
#
# OUTPUT (KEY=VALUE on stdout, so a workflow can `>> "$GITHUB_OUTPUT"` it)
#   gpu_touched=1 | 0     — the decision
#   gpu_crates=<space list>  — only when gpu_touched=1; may be EMPTY when the
#                             trigger was a workflow rather than a crate
#   reason=<one line>     — always, so the verdict is auditable in the log
#
# EXIT
#   0  a decision was reached (either polarity)
#   2  ENV — no diff could be derived, or the derived diff is empty. NEVER guess.
#      An empty diff is not "no GPU crates touched": on a merge ref with no
#      parent, or a shallow clone with no origin/main, the diff is empty for a
#      reason that has nothing to do with the PR. Answering 0 there is how a
#      gate goes permanently dark while staying green.
#
#   bash scripts/ci_gpu_touched.sh                      # derive the diff from git
#   bash scripts/ci_gpu_touched.sh --diff-from FILE     # the seam the tests use
#   bash scripts/ci_gpu_touched.sh --self-test          # the case table
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The GPU set, spec row 67-C1 verbatim. ONE declaration: the registered mutation
# for this script is deleting `aprender-serve ` from this line, and
# scripts/tests/ci_gpu_touched_test.sh replays exactly that edit.
GPU_CRATES="aprender-gpu aprender-cuda-edge aprender-serve aprender-train aprender-compute"

# Workflows that drive the GPU hosts. cuda-nightly.yml and silicon-nightly.yml
# ARE the gx10/yoga lanes; ci.yml is where `gpu-quick` and `cuda-unit` are
# defined, so editing it changes what those runners execute.
GPU_WORKFLOWS=".github/workflows/cuda-nightly.yml .github/workflows/silicon-nightly.yml .github/workflows/ci.yml"

DIFF_FROM=""
SELF_TEST=0
while [ $# -gt 0 ]; do
    case "$1" in
        --diff-from) DIFF_FROM=$2; shift 2 ;;
        --self-test | --selftest) SELF_TEST=1; shift ;;
        *)
            printf 'usage: %s [--diff-from FILE] | --self-test\n' "$0" >&2
            exit 2
            ;;
    esac
done

derive_diff() {
    # A merge ref (the PR's merge commit, which is what actions/checkout leaves
    # on a `pull_request` run) has a first parent: HEAD^1..HEAD IS the PR's diff.
    # Otherwise fall back to the merge base with main. Neither -> ENV.
    if git -C "$REPO_ROOT" rev-parse --verify --quiet HEAD^1 > /dev/null 2>&1; then
        git -C "$REPO_ROOT" diff --name-only HEAD^1..HEAD 2> /dev/null && return 0
    fi
    if git -C "$REPO_ROOT" rev-parse --verify --quiet origin/main > /dev/null 2>&1; then
        git -C "$REPO_ROOT" diff --name-only origin/main...HEAD 2> /dev/null && return 0
    fi
    return 1
}

decide() {
    local diff crate hit_crates="" hit_wf="" f
    if [ -n "$DIFF_FROM" ]; then
        if [ ! -f "$DIFF_FROM" ]; then
            printf 'reason=ENV: the diff-from path names no file: %s\n' "$DIFF_FROM" >&2
            return 2
        fi
        diff=$(cat "$DIFF_FROM")
    else
        diff=$(derive_diff) || {
            printf 'reason=ENV: no diff derivable (no HEAD^1 and no origin/main) — refusing to guess\n' >&2
            return 2
        }
    fi
    # Strip blank lines; an all-blank file is an empty diff.
    diff=$(printf '%s\n' "$diff" | grep -v '^[[:space:]]*$' || true)
    if [ -z "$diff" ]; then
        printf 'reason=ENV: the derived diff is EMPTY — that is undecidable, not "no GPU crates touched"\n' >&2
        return 2
    fi

    for crate in $GPU_CRATES; do
        if printf '%s\n' "$diff" | grep -q "^crates/${crate}/"; then
            hit_crates="${hit_crates}${crate} "
        fi
    done
    for f in $GPU_WORKFLOWS; do
        if printf '%s\n' "$diff" | grep -qx "$f"; then
            hit_wf="${hit_wf}${f} "
        fi
    done

    if [ -n "$hit_crates" ] || [ -n "$hit_wf" ]; then
        printf 'gpu_touched=1\n'
        printf 'gpu_crates=%s\n' "${hit_crates% }"
        if [ -n "$hit_crates" ] && [ -n "$hit_wf" ]; then
            printf 'reason=GPU crate(s) %s and GPU-host workflow(s) %s touched\n' "${hit_crates% }" "${hit_wf% }"
        elif [ -n "$hit_crates" ]; then
            printf 'reason=GPU crate(s) %s touched\n' "${hit_crates% }"
        else
            printf 'reason=GPU-host workflow(s) %s touched — the lane that drives gx10/yoga changed\n' "${hit_wf% }"
        fi
        return 0
    fi
    printf 'gpu_touched=0\n'
    printf 'reason=no path under crates/{%s} and no GPU-host workflow in the diff\n' "$(printf '%s' "$GPU_CRATES" | tr ' ' ',')"
    return 0
}

self_test() {
    local td n=0 red=0 out rc T="$0"
    td=$(mktemp -d "${TMPDIR:-/tmp}/ci-gpu.XXXXXX")
    # shellcheck disable=SC2064
    trap "rm -rf '${td:?}'" RETURN
    row() { # row <want-rc> <label> <pattern> <cmd...>
        local want=$1 label=$2 pat=$3
        shift 3
        n=$((n + 1))
        rc=0
        out=$("$@" 2>&1) || rc=$?
        if [ "$rc" = "$want" ] && printf '%s\n' "$out" | grep -qE -- "$pat"; then
            printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
        else
            printf 'FAIL  row %-2s rc=%s (wanted %s, must match /%s/)  %s\n' "$n" "$rc" "$want" "$pat" "$label"
            printf '%s\n' "$out" | sed 's/^/        /'
            red=1
        fi
    }

    printf 'crates/aprender-serve/src/gguf/cuda/matmul.rs\n' > "$td/serve.txt"
    row 0 "a diff under crates/aprender-serve/ -> gpu_touched=1" \
        '^gpu_touched=1$' bash "$T" --diff-from "$td/serve.txt"
    row 0 "  ...naming the crate, so the log says WHY the GPU runner was claimed" \
        '^gpu_crates=aprender-serve$' bash "$T" --diff-from "$td/serve.txt"

    printf 'crates/aprender-core/src/lib.rs\ncrates/aprender-core/src/traits.rs\n' > "$td/core.txt"
    row 0 "a diff under crates/aprender-core/ ONLY -> gpu_touched=0 (no GPU path)" \
        '^gpu_touched=0$' bash "$T" --diff-from "$td/core.txt"

    printf '.github/workflows/cuda-nightly.yml\n' > "$td/wf.txt"
    row 0 "cuda-nightly.yml touched -> 1: the workflow that DRIVES the GPU hosts counts" \
        '^gpu_touched=1$' bash "$T" --diff-from "$td/wf.txt"
    row 0 "  ...with an empty crate list and a reason that names the workflow" \
        'GPU-host workflow' bash "$T" --diff-from "$td/wf.txt"

    printf 'scripts/ci_test_tier.sh\nscripts/tests/guard_tree_test.sh\n' > "$td/scripts.txt"
    row 0 "a scripts-only diff -> gpu_touched=0" \
        '^gpu_touched=0$' bash "$T" --diff-from "$td/scripts.txt"

    : > "$td/empty.txt"
    row 2 "an EMPTY diff -> ENV (exit 2), never a green 0" \
        'undecidable' bash "$T" --diff-from "$td/empty.txt"
    printf '\n   \n' > "$td/blank.txt"
    row 2 "a diff of blank lines -> ENV (exit 2) too — same undecidability, different shape" \
        'undecidable' bash "$T" --diff-from "$td/blank.txt"
    row 2 "a --diff-from that names no file -> ENV (exit 2)" \
        'names no file' bash "$T" --diff-from "$td/does-not-exist.txt"

    # Both polarities of the crate match: a crate whose NAME contains a GPU
    # crate's name as a prefix must not be mistaken for it.
    printf 'crates/aprender-gpu-docs-that-do-not-exist/src/lib.rs\n' > "$td/prefix.txt"
    row 0 "crates/aprender-gpu-<something-else>/ is NOT crates/aprender-gpu/ (prefix, not membership)" \
        '^gpu_touched=0$' bash "$T" --diff-from "$td/prefix.txt"
    printf 'docs/aprender-serve.md\n' > "$td/doc.txt"
    row 0 "a doc merely NAMING a GPU crate is not a touch of it" \
        '^gpu_touched=0$' bash "$T" --diff-from "$td/doc.txt"

    # MUTANT (registered by the spec): a crate set that has lost aprender-serve
    # must answer 0 on the serve fixture, or row 1 above proves nothing.
    sed 's/aprender-serve //' "$T" > "$td/mutant.sh"
    row 0 "MUTANT without aprender-serve answers 0 on the serve diff — row 1 discriminates" \
        '^MUTANT-BLIND$' bash -c "
            if bash '$td/mutant.sh' --diff-from '$td/serve.txt' 2>/dev/null | grep -q '^gpu_touched=1'; then
                echo MUTANT-STILL-SEES-IT
            else
                echo MUTANT-BLIND
            fi"
    # MUTANT: a copy that answers 0 instead of exiting 2 on an empty diff is the
    # dark-gate shape. The ENV rows must lose it.
    row 0 "MUTANT that answers 0 on an empty diff is caught by the ENV rows" \
        '^MUTANT-CAUGHT$' bash -c "
            sed 's/^    if \[ -z \"\$diff\" \]; then/    if false; then/' '$T' > '$td/mutant2.sh'
            if bash '$td/mutant2.sh' --diff-from '$td/empty.txt' 2>/dev/null | grep -q '^gpu_touched=0'; then
                echo MUTANT-CAUGHT
            else
                echo MUTANT-NOT-REPRODUCED
            fi"

    printf '\n%s checks, %s failed\n' "$n" "$red"
    [ "$red" -eq 0 ]
}

if [ "$SELF_TEST" = 1 ]; then
    self_test
    exit $?
fi
decide
