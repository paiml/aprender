#!/usr/bin/env bash
# check_train_canary_comparator.sh - the collapse comparator for the
# trueno-vs-Burn train canary, and its case table (#3174).
#
# usage: scripts/check_train_canary_comparator.sh
#            case table, then validate the committed baseline (what CI runs)
#        scripts/check_train_canary_comparator.sh --compare RESULT BASELINE
#            case table, then judge one canary run (scripts/run_train_canary.sh)
#
# A run FAILS (1) when any baseline size's ratio (burn_ms / trueno_ms) is below
# baseline * (1 - tolerance), is missing, or is not a positive finite number,
# when the run does not name the adapter it ran on, or when the baseline itself
# is unusable (no GPU name, tolerance outside (0, 1), a non-positive ratio: any
# of those makes the floor unable to fail). A run on an adapter the baseline was
# not measured on exits 3: no baseline is not a pass.
#
# The CI mode also checks that the baseline's sizes are exactly the canary's
# SIZES, so a size added to the canary cannot go ungated.
#
# The case table runs first in every mode, so a comparator that stops catching
# a collapse goes RED here, in CI, without a GPU. Cargo-free (python over text
# only), so guard_tree.sh runs it everywhere: it must never write that build
# tool's name followed by a space.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE="$REPO_ROOT/crates/aprender-train-canary"
case "${1:-}" in
    "") set -- baseline "$CRATE/baseline.json" "$CRATE/src/main.rs" ;;
    --compare)
        [ "$#" -eq 3 ] || { echo "usage: $0 --compare RESULT BASELINE" >&2; exit 2; }
        set -- compare "$2" "$3"
        ;;
    -h | --help) sed -n '2,23p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "check_train_canary_comparator: unknown argument $1" >&2; exit 2 ;;
esac

exec python3 - "$@" <<'PY'
import json, math, re, sys


def positive(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def baseline_errors(b):
    errs = []
    if not isinstance(b.get("gpu"), str) or not b["gpu"].strip():
        errs.append("`gpu` must name the adapter the baseline was measured on")
    tol = b.get("tolerance")
    if not positive(tol) or not tol < 1:
        errs.append(f"`tolerance` {tol!r} must be in (0, 1): at >= 1 the floor is <= 0 and nothing can fail")
    rows = b.get("rows")
    if not isinstance(rows, list) or not rows:
        errs.append("`rows` must be a non-empty list")
        return errs
    for r in rows:
        if not isinstance(r, dict) or not r.get("size") or not positive(r.get("ratio")):
            errs.append(f"row {r!r}: needs a `size` and a positive finite `ratio`")
    return errs


def verdict(result, baseline):
    """Return (code, lines). 0 pass, 1 collapse/defect, 3 no baseline for this adapter."""
    errs = baseline_errors(baseline)
    if errs:
        return 1, [f"FAIL  baseline: {e}" for e in errs]
    gpu = result.get("gpu")
    if not isinstance(gpu, str) or not gpu.strip():
        return 1, ["FAIL  the run does not name the adapter it ran on"]
    if baseline["gpu"] != gpu:
        return 3, [f"NO-BASELINE  baseline is for {baseline['gpu']!r}, this run is {gpu!r}"]
    tol = baseline["tolerance"]
    got = {r.get("size"): r.get("ratio") for r in result.get("rows", []) if isinstance(r, dict)}
    lines, bad = [], 0
    for row in baseline["rows"]:
        size, base = row["size"], row["ratio"]
        floor = base * (1.0 - tol)
        r = got.get(size)
        if not positive(r):
            lines.append(f"FAIL  {size}: ratio {r!r} missing or not a positive finite number")
            bad += 1
        elif r < floor:
            lines.append(f"FAIL  {size}: ratio {r:.2f}x < floor {floor:.2f}x (baseline {base:.2f}x, tolerance {tol:.0%})")
            bad += 1
        else:
            lines.append(f"ok    {size}: ratio {r:.2f}x >= floor {floor:.2f}x (baseline {base:.2f}x)")
    return (1 if bad else 0), lines


def canary_sizes(src):
    """The `m x k x n` sizes in the canary's `const SIZES` array."""
    m = re.search(r"const SIZES[^=]*=\s*\[(.*?)\];", src, re.S)
    if not m:
        return None
    body = re.sub(r"//[^\n]*", "", m.group(1))
    return [f"{a}x{b}x{c}" for a, b, c in re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", body)]


G = "RTX-X"
BASE = {"gpu": G, "tolerance": 0.5, "rows": [{"size": "a", "ratio": 10.0}, {"size": "b", "ratio": 4.0}]}


def res(gpu=G, **r):
    return {"gpu": gpu, "rows": [{"size": s, "ratio": v} for s, v in r.items()]}


def base(**over):
    b = dict(BASE)
    b.update(over)
    return b


SRC = "const SIZES: [(usize, usize, usize); 2] = [\n    (4, 2560, 9728), // seq=4\n    (32, 2560, 4096),\n];\n"
CASES = [
    # (name, result, baseline, expected code)
    ("same as baseline", res(a=10.0, b=4.0), BASE, 0),
    ("faster than baseline", res(a=30.0, b=9.0), BASE, 0),
    ("exactly at floor", res(a=5.0, b=2.0), BASE, 0),
    ("one size collapsed", res(a=10.0, b=1.9), BASE, 1),
    ("burn now faster", res(a=0.8, b=4.0), BASE, 1),
    ("size missing from run", res(a=10.0), BASE, 1),
    ("ratio NaN", res(a=float("nan"), b=4.0), BASE, 1),
    ("ratio negative", res(a=-10.0, b=4.0), BASE, 1),
    ("ratio a string", res(a="10", b=4.0), BASE, 1),
    ("run names no adapter", res(gpu=None, a=10.0, b=4.0), BASE, 1),
    ("run on another adapter", res(gpu="RTX-Y", a=10.0, b=4.0), BASE, 3),
    ("baseline with no rows", res(a=10.0), base(rows=[]), 1),
    ("baseline tolerance 1.5 (floor < 0)", res(a=0.1, b=0.1), base(tolerance=1.5), 1),
    ("baseline tolerance missing", res(a=10.0, b=4.0), {k: v for k, v in BASE.items() if k != "tolerance"}, 1),
    ("baseline ratio zero (floor 0)", res(a=0.1, b=4.0), base(rows=[{"size": "a", "ratio": 0}, {"size": "b", "ratio": 4.0}]), 1),
    ("baseline names no adapter", res(a=10.0, b=4.0), base(gpu=""), 1),
]
SIZE_CASES = [
    # (name, source, expected)
    ("two sizes, comment ignored", SRC, ["4x2560x9728", "32x2560x4096"]),
    ("commented-out size ignored", SRC.replace("    (32", "    // (1, 1, 1),\n    (32"), ["4x2560x9728", "32x2560x4096"]),
    ("no SIZES const", "fn main() {}\n", None),
]


def self_test():
    bad = 0
    for name, r, b, want in CASES:
        got, _ = verdict(r, b)
        ok = got == want
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} want={want} got={got} {name}")
    for name, src, want in SIZE_CASES:
        got = canary_sizes(src)
        ok = got == want
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} sizes {name}" + ("" if ok else f" -> {got}"))
    n = len(CASES) + len(SIZE_CASES)
    print(f"case table: {n - bad}/{n}")
    return bad == 0


mode = sys.argv[1]
if not self_test():
    print("FAIL  check_train_canary_comparator: the case table does not hold; the comparator cannot be trusted")
    sys.exit(1)
with open(sys.argv[2]) as f:
    first = json.load(f)
if mode == "baseline":
    errs = baseline_errors(first)
    with open(sys.argv[3]) as f:
        sizes = canary_sizes(f.read())
    rows = [r.get("size") for r in first.get("rows", []) if isinstance(r, dict)]
    if sizes is None:
        errs.append(f"no `const SIZES` array found in {sys.argv[3]}")
    elif sorted(sizes) != sorted(rows):
        errs.append(f"baseline sizes {sorted(rows)} != canary SIZES {sorted(sizes)}: a size would go ungated")
    for e in errs:
        print(f"FAIL  {sys.argv[2]}: {e}")
    if errs:
        sys.exit(1)
    print(f"PASS  comparator case table holds; baseline gates all {len(sizes)} canary sizes on {first['gpu']}")
    sys.exit(0)
with open(sys.argv[3]) as f:
    baseline = json.load(f)
code, lines = verdict(first, baseline)
print("\n".join(lines))
print({0: "PASS  train canary: no size collapsed below its floor",
       1: "FAIL  train canary: collapse or unusable input (#3174)",
       3: "NO-BASELINE  train canary: not measured against a baseline for this adapter"}[code])
sys.exit(code)
PY
