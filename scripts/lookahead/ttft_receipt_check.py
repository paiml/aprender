#!/usr/bin/env python3
"""Check an apr-serve-ttft-v1 receipt (0.71 exit criteria V1 + V2).

V2: the apr arm reports load_ms, ttft_ms, pp512 and tg128 together.
V1: apr TTFT <= 2.0x llama.cpp at pin d1d3c3396, on the same identity.
A ratio across mismatched identity is refused (APR-OBS-001 §2.1, S-4).

Exit codes: 0 PASS, 1 FAIL (ratio above bound), 2 RED/REFUSED.
"""
import argparse
import json
import sys
from pathlib import Path

SCHEMA = "apr-serve-ttft-v1"
PIN = "d1d3c3396"
BOUND = 2.0
MIN_SAMPLES = 5
SHARED = ("host", "model_sha256", "backend", "prompt_sha256", "prompt_n", "completion_n")
IDENTITY = ("ts", "apr_version", "binary_sha256", "build_identity", "model_id")
V2_FIELDS = ("load_ms", "ttft_ms_p50", "pp512_tok_s", "tg128_tok_s")
HERE = Path(__file__).resolve().parent


class Red(Exception):
    pass


def require(cond, message):
    if not cond:
        raise Red(message)


def check_identity(doc):
    arms = doc.get("arms") or {}
    apr, ref = arms.get("apr") or {}, arms.get("llamacpp") or {}
    require(apr and ref, "receipt needs both arms: apr and llamacpp (R-2: empty is RED)")
    missing = [k for k in IDENTITY + SHARED if apr.get(k) in (None, "", "unknown")]
    require(not missing, f"apr identity fields missing: {missing}")
    differ = [k for k in SHARED if apr.get(k) != ref.get(k)]
    require(not differ, f"REFUSED: ratio across mismatched identity on {differ}")
    require(ref.get("pin") == PIN, f"llamacpp pin {ref.get('pin')!r} != {PIN}")
    return apr, ref


def check_v2(apr):
    missing = [k for k in V2_FIELDS if not isinstance(apr.get(k), (int, float)) or apr[k] <= 0]
    require(not missing, f"V2: apr arm must report {list(V2_FIELDS)} together; missing {missing}")


def check_samples(apr, ref):
    for name, arm in (("apr", apr), ("llamacpp", ref)):
        require((arm.get("n") or 0) >= MIN_SAMPLES, f"{name}: n < {MIN_SAMPLES} samples")
        require(isinstance(arm.get("ttft_ms_p50"), (int, float)) and arm["ttft_ms_p50"] > 0,
                f"{name}: ttft_ms_p50 missing")


def evaluate(doc):
    require(isinstance(doc, dict) and doc.get("schema") == SCHEMA, f"schema is not {SCHEMA}")
    apr, ref = check_identity(doc)
    check_v2(apr)
    check_samples(apr, ref)
    ratio = apr["ttft_ms_p50"] / ref["ttft_ms_p50"]
    return ratio, ratio <= BOUND


def run(path):
    ratio, ok = evaluate(json.loads(Path(path).read_text() or "null"))
    print(f"V1 ttft ratio apr/llamacpp = {ratio:.3f} (bound {BOUND}): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


EXPECT = {"pass.json": 0, "ratio-fail.json": 1, "identity-mismatch.json": 2,
          "missing-load.json": 2, "wrong-pin.json": 2, "few-samples.json": 2, "empty.json": 2}


def outcome(path):
    try:
        return 0 if evaluate(json.loads(path.read_text() or "null"))[1] else 1
    except Red:
        return 2


def self_test():
    """Planted falsifiers: each fixture must land on its expected exit code."""
    fx = HERE / "fixtures" / "ttft"
    bad = [f"{n}: got {outcome(fx / n)}, want {w}" for n, w in EXPECT.items() if outcome(fx / n) != w]
    for b in bad:
        print("FALSIFIER SURVIVED:", b)
    print("self-test:", "RED" if bad else f"ok ({len(EXPECT)} fixtures)")
    return 2 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("receipt", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test or not a.receipt:
        return self_test()
    try:
        return run(a.receipt)
    except Red as e:
        print("RED:", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
