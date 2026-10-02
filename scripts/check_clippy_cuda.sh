#!/usr/bin/env bash
# check_clippy_cuda.sh — clippy with `--features cuda` is clean (#3636).
#
# `ci / lint` runs clippy without `cuda`; `cuda-unit` runs tests. Nothing ran
# the combination, so findings under #[cfg(feature = "cuda")] accumulated
# invisibly (61 in aprender-train + 1 in aprender-compute by 2026-09-24) —
# the #2370 shape, one feature flag over.
#
# `apr-cli --features cuda` enables realizar/cuda, entrenar/cuda and
# trueno/cuda, and cargo clippy lints every workspace member it builds, so
# one invocation covers compute, gpu, serve and train.
#
# SCOPE: CI runs this in `cuda-unit` whenever scripts/ci_gpu_touched.sh answers
# cuda_lint=1: a GPU-set diff, or an apr-cli diff (apr-cli is cuda-gated but not
# in the GPU set, so it gets this step without the cuda test steps; #4336).
#
#   bash scripts/check_clippy_cuda.sh              # the gate
#   bash scripts/check_clippy_cuda.sh --self-test  # planted unused import -> RED
#
# Exit: 0 clean, 1 findings, 2 cannot measure (no nvcc: refuses to pass vacuously).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD=(cargo clippy -p apr-cli --features cuda -- -D warnings)

preflight() {
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "check_clippy_cuda: nvcc not found — cannot build the cuda feature; refusing to report clean" >&2
        exit 2
    fi
}

run_gate() {
    local log="$1" rc=0
    (cd "$ROOT" && "${CMD[@]}") >"$log" 2>&1 || rc=$?
    return "$rc"
}

gate() {
    preflight
    local log
    log="$(mktemp)"
    if run_gate "$log"; then
        echo "check_clippy_cuda: clean (${CMD[*]})"
        rm -f "$log"
        return 0
    fi
    grep -E '^(error|warning)(\[|:)' "$log" | sort | uniq -c | sort -rn | head -40 >&2 || true
    echo "check_clippy_cuda: RED — full log: $log" >&2
    return 1
}

self_test() {
    preflight
    local target="$ROOT/crates/aprender-train/src/lib.rs" backup log
    backup="$(mktemp)"
    log="$(mktemp)"
    cp "$target" "$backup"
    # shellcheck disable=SC2064  # expand now: absolute paths, restore survives any cd
    # The backup is deleted ONLY after a successful restore; a failed cp keeps it
    # and names it, so the original is never lost with the tree left mutated.
    trap "if cp '$backup' '$target'; then rm -f '$backup'; else echo \"check_clippy_cuda: RESTORE FAILED — original kept at $backup\" >&2; exit 3; fi" EXIT
    printf '\n#[cfg(feature = "cuda")]\nuse std::collections::BinaryHeap; // check_clippy_cuda planted\n' >>"$target"
    if run_gate "$log"; then
        echo "SELF-TEST FAILED: the planted cuda-only unused import passed the gate" >&2
        return 1
    fi
    if ! grep -q 'unused import: `std::collections::BinaryHeap`' "$log"; then
        echo "SELF-TEST FAILED: the gate went RED but not on the planted import — log: $log" >&2
        return 1
    fi
    rm -f "$log"
    echo "SELF-TEST PASSED: planted #[cfg(feature = \"cuda\")] unused import -> RED"
}

case "${1:-}" in
    --self-test) self_test ;;
    "") gate ;;
    *) echo "usage: $0 [--self-test]" >&2; exit 2 ;;
esac
