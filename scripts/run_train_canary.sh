#!/usr/bin/env bash
# run_train_canary.sh - build and run crates/aprender-train-canary (trueno WGPU
# vs Burn WGPU matmul) and fail if trueno's lead over Burn collapses (#3174).
#
# usage: scripts/run_train_canary.sh [--out FILE] [--iters N]
#        scripts/run_train_canary.sh --compare RESULT.json [BASELINES_DIR|BASELINE.json]
#
# The canary is excluded from the workspace (burn pulls a libsqlite3-sys that
# conflicts with aprender-rag), so no workspace job builds it. This script is
# the one place that does: its own target dir, the binary run under the GPU
# lock (never the build), evidence JSON written to --out.
#
# THE FLOOR. crates/aprender-train-canary/baselines/*.json each hold the measured
# per-size ratio (burn_ms / trueno_ms, median of N) for one named wgpu adapter
# (the canary pins both backends to the single discrete GPU and records it). A run
# FAILS when any size's ratio falls below baseline * (1 - tolerance), when a
# baseline size is missing from the run, or when a ratio is not a positive
# finite number. A run on an adapter the baseline was not measured on exits 3: no
# baseline is not a pass.
#
# THE COMPARATOR is scripts/check_train_canary_comparator.sh, a guard CI runs
# without a GPU: its case table runs before every compare, so a comparator that
# stops catching a collapse fails loudly instead of passing vacuously.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE="crates/aprender-train-canary"
BASELINE="$REPO_ROOT/$CRATE/baselines"
TMP="${TMPDIR:-/tmp}"
OUT="$TMP/train-canary-$$.json"
ITERS=10
MODE=run
RESULT=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --out) OUT="${2:?--out needs a path}"; shift 2 ;;
        --iters) ITERS="${2:?--iters needs N}"; shift 2 ;;
        --compare)
            MODE=compare
            RESULT="${2:?--compare needs a result file}"
            shift 2
            case "${1:-}" in
                "" | -*) ;;
                *) BASELINE="$1"; shift ;;
            esac
            ;;
        -h | --help) sed -n '2,23p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "run_train_canary: unknown argument $1" >&2; exit 2 ;;
    esac
done

COMPARATOR="$REPO_ROOT/scripts/check_train_canary_comparator.sh"
case "$MODE" in
    compare) bash "$COMPARATOR" --compare "$RESULT" "$BASELINE" ;;
    run)
        bash "$COMPARATOR"
        TARGET="${CANARY_TARGET_DIR:-$REPO_ROOT/target/train-canary}"
        (cd "$REPO_ROOT/$CRATE" && CARGO_TARGET_DIR="$TARGET" cargo build --release --locked)
        BIN="$TARGET/release/aprender-train-canary"
        flock -w 3600 /tmp/apr-gpu.lock "$BIN" --iters "$ITERS" --json "$OUT"
        echo "evidence: $OUT"
        bash "$COMPARATOR" --compare "$OUT" "$BASELINE"
        ;;
esac
