#!/usr/bin/env bash
# run_train_canary.sh - build and run crates/aprender-train-canary (trueno WGPU
# vs Burn WGPU matmul) and fail if trueno's lead over Burn collapses (#3174).
#
# usage: scripts/run_train_canary.sh [--out FILE] [--iters N]
#        scripts/run_train_canary.sh --compare RESULT.json [BASELINE.json]
#        scripts/run_train_canary.sh --self-test
#
# The canary is excluded from the workspace (burn pulls a libsqlite3-sys that
# conflicts with aprender-rag), so no workspace job builds it. This script is
# the one place that does: its own target dir, the binary run under the GPU
# lock (never the build), evidence JSON written to --out.
#
# THE FLOOR. crates/aprender-train-canary/baseline.json holds the measured
# per-size ratio (burn_ms / trueno_ms, median of N) for one named GPU. A run
# FAILS when any size's ratio falls below baseline * (1 - tolerance), when a
# baseline size is missing from the run, or when a ratio is not a positive
# finite number. A run on a GPU the baseline was not measured on exits 3: no
# baseline is not a pass.
#
# THE CASE TABLE. --self-test runs the comparator over fixtures that must go
# RED or stay GREEN. Every run executes it first, so a comparator that stops
# catching a collapse fails loudly instead of passing vacuously.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE="crates/aprender-train-canary"
BASELINE="$REPO_ROOT/$CRATE/baseline.json"
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
            if [ -n "${3:-}" ]; then BASELINE="$3"; shift; fi
            shift 2
            ;;
        --self-test) MODE=self-test; shift ;;
        -h | --help) sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "run_train_canary: unknown argument $1" >&2; exit 2 ;;
    esac
done

compare() {
    # compare MODE RESULT BASELINE GPU
    python3 - "$@" <<'PY'
import json, math, os, sys, tempfile

def verdict(result, baseline, gpu):
    """Return (code, lines). 0 pass, 1 collapse/defect, 3 no baseline for this GPU."""
    if baseline.get("gpu") != gpu:
        return 3, [f"NO-BASELINE  baseline is for {baseline.get('gpu')!r}, this run is {gpu!r}"]
    tol = float(baseline.get("tolerance", 0.5))
    got = {r["size"]: r.get("ratio") for r in result.get("rows", [])}
    lines, bad = [], 0
    for row in baseline.get("rows", []):
        size, base = row["size"], float(row["ratio"])
        floor = base * (1.0 - tol)
        r = got.get(size)
        if not isinstance(r, (int, float)) or not math.isfinite(r) or r <= 0:
            lines.append(f"FAIL  {size}: ratio {r!r} missing or not a positive finite number")
            bad += 1
        elif r < floor:
            lines.append(f"FAIL  {size}: ratio {r:.2f}x < floor {floor:.2f}x (baseline {base:.2f}x, tolerance {tol:.0%})")
            bad += 1
        else:
            lines.append(f"ok    {size}: ratio {r:.2f}x >= floor {floor:.2f}x (baseline {base:.2f}x)")
    if not baseline.get("rows"):
        lines.append("FAIL  baseline has no rows")
        bad += 1
    return (1 if bad else 0), lines

G = "RTX-X"
BASE = {"gpu": G, "tolerance": 0.5, "rows": [{"size": "a", "ratio": 10.0}, {"size": "b", "ratio": 4.0}]}
def res(**r):
    return {"rows": [{"size": s, "ratio": v} for s, v in r.items()]}
CASES = [
    # (name, result, baseline, gpu, expected code)
    ("same as baseline", res(a=10.0, b=4.0), BASE, G, 0),
    ("faster than baseline", res(a=30.0, b=9.0), BASE, G, 0),
    ("exactly at floor", res(a=5.0, b=2.0), BASE, G, 0),
    ("one size collapsed", res(a=10.0, b=1.9), BASE, G, 1),
    ("burn now faster", res(a=0.8, b=4.0), BASE, G, 1),
    ("size missing from run", res(a=10.0), BASE, G, 1),
    ("ratio NaN", res(a=float("nan"), b=4.0), BASE, G, 1),
    ("ratio zero", res(a=0.0, b=4.0), BASE, G, 1),
    ("ratio a string", res(a="10", b=4.0), BASE, G, 1),
    ("empty baseline", res(a=10.0), {"gpu": G, "rows": []}, G, 1),
    ("other GPU", res(a=10.0, b=4.0), BASE, "RTX-Y", 3),
]

def self_test():
    bad = 0
    for name, r, b, g, want in CASES:
        got, _ = verdict(r, b, g)
        ok = got == want
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} want={want} got={got} {name}")
    print(f"case table: {len(CASES) - bad}/{len(CASES)}")
    return bad == 0

mode = sys.argv[1]
if not self_test():
    print("FAIL  run_train_canary: the comparator case table does not hold")
    sys.exit(1)
if mode == "self-test":
    sys.exit(0)
with open(sys.argv[2]) as f:
    result = json.load(f)
with open(sys.argv[3]) as f:
    baseline = json.load(f)
code, lines = verdict(result, baseline, sys.argv[4])
print("\n".join(lines))
print({0: "PASS  train canary: no size collapsed below its floor",
       1: "FAIL  train canary: trueno's lead over Burn collapsed (#3174)",
       3: "NO-BASELINE  train canary: not measured against a baseline for this GPU"}[code])
sys.exit(code)
PY
}

gpu_name() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name --format=csv,noheader | head -1
    else
        echo "unknown"
    fi
}

case "$MODE" in
    self-test) compare self-test ;;
    compare) compare compare "$RESULT" "$BASELINE" "$(gpu_name)" ;;
    run)
        compare self-test
        TARGET="${CANARY_TARGET_DIR:-$REPO_ROOT/target/train-canary}"
        (cd "$REPO_ROOT/$CRATE" && CARGO_TARGET_DIR="$TARGET" cargo build --release --locked)
        BIN="$TARGET/release/aprender-train-canary"
        flock /tmp/apr-gpu.lock "$BIN" --iters "$ITERS" --json "$OUT"
        echo "evidence: $OUT"
        compare compare "$OUT" "$BASELINE" "$(gpu_name)"
        ;;
esac
