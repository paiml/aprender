#!/usr/bin/env bash
# check_train_canary_comparator.sh - the collapse comparator for the
# trueno-vs-Burn train canary, and its case table (#3174).
#
# usage: scripts/check_train_canary_comparator.sh
#            case table, then validate the committed baseline (what CI runs)
#        scripts/check_train_canary_comparator.sh --compare RESULT BASELINE GPU
#            case table, then judge one canary run (scripts/run_train_canary.sh)
#
# A run FAILS (1) when any baseline size's ratio (burn_ms / trueno_ms) is below
# baseline * (1 - tolerance), is missing, or is not a positive finite number.
# A run on a GPU the baseline was not measured on exits 3: no baseline is not
# a pass. The case table runs first in every mode, so a comparator that stops
# catching a collapse goes RED here, in CI, without a GPU.
#
# Cargo-free (python over JSON only), so guard_tree.sh runs it everywhere: it
# must never write that build tool's name followed by a space.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-}" in
    "") set -- baseline "$REPO_ROOT/crates/aprender-train-canary/baseline.json" ;;
    --compare)
        [ "$#" -eq 4 ] || { echo "usage: $0 --compare RESULT BASELINE GPU" >&2; exit 2; }
        set -- compare "$2" "$3" "$4"
        ;;
    -h | --help) sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "check_train_canary_comparator: unknown argument $1" >&2; exit 2 ;;
esac

exec python3 - "$@" <<'PY'
import json, math, sys

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

def baseline_errors(b):
    errs = []
    if not isinstance(b.get("gpu"), str) or not b["gpu"].strip():
        errs.append("`gpu` must name the GPU the baseline was measured on")
    tol = b.get("tolerance", 0.5)
    if not isinstance(tol, (int, float)) or not 0 < tol < 1:
        errs.append(f"`tolerance` {tol!r} must be in (0, 1)")
    rows = b.get("rows")
    if not isinstance(rows, list) or not rows:
        errs.append("`rows` must be a non-empty list")
    else:
        for r in rows:
            v = r.get("ratio") if isinstance(r, dict) else None
            if not isinstance(r, dict) or not r.get("size") or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                errs.append(f"row {r!r}: needs a `size` and a positive finite `ratio`")
    return errs


mode = sys.argv[1]
if not self_test():
    print("FAIL  check_train_canary_comparator: the case table does not hold; the comparator cannot be trusted")
    sys.exit(1)
if mode == "baseline":
    with open(sys.argv[2]) as f:
        errs = baseline_errors(json.load(f))
    for e in errs:
        print(f"FAIL  {sys.argv[2]}: {e}")
    if errs:
        sys.exit(1)
    print(f"PASS  comparator case table holds; {sys.argv[2]} is a usable baseline")
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
