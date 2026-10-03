#!/usr/bin/env python3
"""Check one night of an apr-perf-ledger-v1 ledger (0.71 V1/V2 baselines, APR-OBS-001 OBS-05).

The subject (`apr bench --emit raw-samples-v1`, #4551) hands over raw samples only; the
recorder derives every statistic and appends one row per host x backend. This checks that row.
  L1 an empty ledger is RED.
  L2 every row carries the §2.1 identity; a GPU row without gpu_proof.on_gpu is backend_unproven
     and is excluded from the GPU series.
  L3 every expected host x backend has an admissible row (missing host row is RED).
  L4 both arms (apr, llamacpp) ran the same identity in the same run, at the pinned comparator
     from scripts/llama_pin.toml, on the §2.4 workload, each arm referencing its raw-samples sha.
  L5 the ratio is derived, never asserted: ratio[m] == apr.median[m] / llamacpp.median[m].

Exit codes: 0 GREEN, 2 RED/REFUSED.
"""
import argparse
import json
import re
import sys
from pathlib import Path

SCHEMA = "apr-perf-ledger-v1"
IDENTITY = ("ts", "host", "apr_version", "binary_sha256", "build_identity",
            "model_id", "model_sha256", "backend", "request_id")
SHARED = ("host", "model_sha256", "backend", "prompt_sha256", "run_id")
METRICS = ("load_ms", "ttft_ms", "pp512_tok_s", "tg128_tok_s", "wall_ms")
WORKLOAD = {"max_tokens": 128, "context": 4096, "seed": 0, "greedy": True, "ignore_eos": True}
MIN_PROMPTS, REPS, PROMPT_TOKENS, PROMPT_TOL = 8, 5, 512, 8
DEFAULT_EXPECT = "gx10:cuda,lambda:cpu"   # APR-OBS-001 §7 OBS-05
REL_EPS = 1e-9
HERE = Path(__file__).resolve().parent
PIN_FILE = HERE.parent / "llama_pin.toml"


class Red(Exception):
    pass


def require(cond, message):
    if not cond:
        raise Red(message)


def pinned_commit(path=PIN_FILE):
    m = re.search(r'^build_commit\s*=\s*"([0-9a-f]+)"', Path(path).read_text(), re.M)
    require(m, f"no build_commit in {path}")
    return m.group(1)


def sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def gpu_proven(row):
    proof = row.get("gpu_proof") or {}
    return row.get("backend") == "cpu" or (proof.get("on_gpu") is True and bool(proof.get("source")))


def check_identity(row):
    missing = [k for k in IDENTITY if row.get(k) in (None, "", "unknown")]
    require(not missing, f"L2: {row.get('host')}/{row.get('backend')}: identity missing {missing}")


def check_workload(row):
    w = row.get("workload") or {}
    off = [k for k, v in WORKLOAD.items() if w.get(k) != v]
    require(not off, f"L4: workload differs from §2.4 on {off}")
    require(abs((w.get("prompt_tokens") or 0) - PROMPT_TOKENS) <= PROMPT_TOL, "L4: prompt_tokens not 512±8")
    require((w.get("prompts") or 0) >= MIN_PROMPTS and w.get("reps") == REPS, "L4: need ≥8 prompts × 5 reps")


def check_arm(name, arm, row, pin):
    require(isinstance(arm, dict), f"L4: arm {name} missing")
    differ = [k for k in SHARED if arm.get(k) != row.get(k)]
    require(not differ, f"L4: REFUSED: arm {name} identity differs from the row on {differ}")
    require(sha(arm.get("raw_samples_sha256")), f"L4: arm {name} has no raw-samples sha256")
    require(name != "llamacpp" or arm.get("build_commit") == pin,
            f"L4: llamacpp at {arm.get('build_commit')!r}, pin is {pin}")
    n = (row["workload"]["prompts"]) * REPS
    for m in METRICS:
        s = (arm.get("stats") or {}).get(m) or {}
        require(isinstance(s.get("median"), (int, float)) and s["median"] > 0, f"L4: {name}.{m} median missing")
        require(s.get("n") == n, f"L4: {name}.{m} n={s.get('n')} != prompts×reps={n}")


def check_ratio(row):
    apr, ref = row["arms"]["apr"]["stats"], row["arms"]["llamacpp"]["stats"]
    for m in METRICS:
        want = apr[m]["median"] / ref[m]["median"]
        got = (row.get("ratio") or {}).get(m)
        require(isinstance(got, (int, float)) and abs(got - want) <= REL_EPS * want,
                f"L5: ratio.{m}={got} is not apr/llamacpp median = {want:.6g}")


def check_row(row, pin):
    require(isinstance(row, dict) and row.get("schema") == SCHEMA, f"row schema is not {SCHEMA}")
    check_identity(row)
    check_workload(row)
    arms = row.get("arms") or {}
    for name in ("apr", "llamacpp"):
        check_arm(name, arms.get(name), row, pin)
    check_ratio(row)


def evaluate(rows, expect, pin):
    require(rows, "L1: empty ledger (R-2: empty is RED)")
    cells = {}
    for row in rows:
        check_row(row, pin)
        if gpu_proven(row):
            cells[(row["host"], row["backend"])] = row
    missing = [f"{h}:{b}" for h, b in expect if (h, b) not in cells]
    require(not missing, f"L3: missing host row (or backend_unproven) for {missing}")
    return len(cells)


def parse_expect(text):
    return [tuple(cell.split(":", 1)) for cell in text.split(",") if cell]


def load(path):
    lines = Path(path).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def outcome(path, expect, pin):
    try:
        evaluate(load(path), expect, pin)
        return 0
    except (Red, AttributeError, TypeError, KeyError, ValueError):
        return 2


EXPECT = {"green.jsonl": 0, "empty.jsonl": 2, "missing-host.jsonl": 2, "missing-version.jsonl": 2,
          "gpu-no-proof.jsonl": 2, "arm-identity-mismatch.jsonl": 2, "asserted-ratio.jsonl": 2,
          "wrong-pin.jsonl": 2, "short-workload.jsonl": 2, "no-raw-sha.jsonl": 2, "n-mismatch.jsonl": 2,
          "wrong-seed.jsonl": 2, "zero-median.jsonl": 2}


def self_test():
    root, expect = HERE / "fixtures" / "ledger", parse_expect(DEFAULT_EXPECT)
    pin = pinned_commit(root / "llama_pin.toml")
    bad = [(n, e, g) for n, e in EXPECT.items() if (g := outcome(root / n, expect, pin)) != e]
    # L1 on its own: an empty ledger is RED even when no cell is expected.
    bad += [("empty.jsonl --expect ''", 2, g) for g in [outcome(root / "empty.jsonl", [], pin)] if g != 2]
    for name, want, got in bad:
        print(f"SELF-TEST FAIL {name}: expected {want}, got {got}")
    print(f"self-test: {len(EXPECT) + 1 - len(bad)}/{len(EXPECT) + 1} cases as planted")
    return 2 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ledger", nargs="?", help="one night, JSONL (one row per line)")
    ap.add_argument("--expect", default=DEFAULT_EXPECT, help="host:backend,... cells that must be present")
    ap.add_argument("--pin-file", default=str(PIN_FILE))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test or not args.ledger:
        return self_test()
    try:
        n = evaluate(load(args.ledger), parse_expect(args.expect), pinned_commit(args.pin_file))
    except (Red, AttributeError, TypeError, KeyError, ValueError) as err:
        print(f"RED: {err}")
        return 2
    print(f"GREEN: {n} host×backend cells admissible")
    return 0


if __name__ == "__main__":
    sys.exit(main())
