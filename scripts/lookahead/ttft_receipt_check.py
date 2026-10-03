#!/usr/bin/env python3
"""Check an apr-serve-ttft-v1 receipt (0.71 exit criteria V1 + V2).

V2: the apr arm reports load_ms, TTFT, pp512 and tg128 together.
V1: apr TTFT <= 2.0x llama.cpp at pin d1d3c3396, on the same identity.

Not a second schema. Each arm's `provenance` is validated by
scripts/lib/bench_receipt.py (binary sha, compute_class, join key), and the
TTFT median is recomputed from the arm's raw `ttft_ms` samples with the same
helper the parity lanes use. A stated `ttft_ms_p50` that is not derivable from
the samples (beyond derivation.ratio_tolerance in scripts/perf-matrix.yaml) is
a fabricated measurement. A ratio across mismatched identity or compute class
is refused (APR-OBS-001 §2.1, S-4; EXT R-17).

Exit codes: 0 PASS, 1 FAIL (ratio above bound), 2 RED/REFUSED.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import bench_receipt as br  # noqa: E402

SCHEMA = "apr-serve-ttft-v1"
PIN = "d1d3c3396"
BOUND = 2.0
MIN_SAMPLES = 5
SHARED = ("model_sha256", "prompt_sha256", "prompt_n", "completion_n")
JOIN = br.JOIN_KEY_REQUIRED + ("compute_class",)
APR_IDENTITY = ("ts", "apr_version", "build_identity", "model_id")
V2_FIELDS = ("load_ms", "load_resolution_ms", "pp512_tok_s", "tg128_tok_s")
# load_ms := runs[*].cold_start_ms from `apr test llm bench --start` (la-71/bench-ready-ms):
# an upper bound, the first passing probe. Its bracket is --health-poll-ms (default 50).
# The old 2 s poll gives a +/-2 s number, which is not a V2 load time.
LOAD_RES_MAX_MS = 100.0


class Red(Exception):
    pass


def require(cond, message):
    if not cond:
        raise Red(message)


def require_clean(errors, prefix=""):
    """bench_receipt rules append to an error list; the first one is RED."""
    if errors:
        raise Red(prefix + errors[0])


def check_provenance(name, arm):
    """bench_receipt's provenance rules, plus: the backend is proven, not 'unknown'."""
    prov = arm.get("provenance")
    require(isinstance(prov, dict), f"{name}.provenance: missing (bench_receipt rules)")
    errors = []
    br._check_provenance(prov, errors)
    require_clean(errors, f"{name}: ")
    gaps = [k for k in JOIN if not prov.get(k) or prov.get(k) == "unknown"]
    require(not gaps, f"{name}.provenance: no backend/join-key proof for {gaps}")
    return prov


def check_identity(doc):
    arms = doc.get("arms") or {}
    apr, ref = arms.get("apr") or {}, arms.get("llamacpp") or {}
    require(apr and ref, "receipt needs both arms: apr and llamacpp (R-2: empty is RED)")
    missing = [k for k in APR_IDENTITY + SHARED if apr.get(k) in (None, "", "unknown")]
    require(not missing, f"apr identity fields missing: {missing}")
    p_apr, p_ref = check_provenance("apr", apr), check_provenance("llamacpp", ref)
    differ = [k for k in SHARED if apr.get(k) != ref.get(k)]
    differ += [k for k in JOIN if p_apr.get(k) != p_ref.get(k)]
    require(not differ, f"REFUSED: ratio across mismatched identity on {differ}")
    # Stricter than bench_receipt's RULE 3 (any pin): V1 names this one.
    require(str(ref.get("build_commit") or "").startswith(PIN),
            f"llamacpp build_commit {ref.get('build_commit')!r} is not pin {PIN}")
    return apr, ref


def check_v2(apr):
    missing = [k for k in V2_FIELDS if not isinstance(apr.get(k), (int, float)) or apr[k] <= 0]
    require(not missing, f"V2: apr arm must report load_ms, TTFT, pp512, tg128 together; missing {missing}")
    require(apr["load_resolution_ms"] <= LOAD_RES_MAX_MS,
            f"V2: load_ms bracket {apr['load_resolution_ms']} ms > {LOAD_RES_MAX_MS} ms is not a load time")


def median_ttft(name, arm):
    """The median from raw samples; a stated p50 must agree with it."""
    require(arm.get("failed") == 0,
            f"{name}.failed={arm.get('failed')!r}: a median over the requests that survived is not a measurement")
    errors = []
    med = br._median_of(arm, "ttft_ms", name, errors)
    require_clean(errors)
    require(len(arm["ttft_ms"]) >= MIN_SAMPLES, f"{name}: n < {MIN_SAMPLES} samples")
    stated = arm.get("ttft_ms_p50")
    if stated is not None:
        tol = br.ratio_tolerance() * max(med, 1e-9)
        require(isinstance(stated, (int, float)) and abs(stated - med) <= tol,
                f"{name}.ttft_ms_p50={stated!r} does not follow from its samples (median {med:.3f})")
    return med


def evaluate(doc):
    require(isinstance(doc, dict) and doc.get("schema") == SCHEMA, f"schema is not {SCHEMA}")
    apr, ref = check_identity(doc)
    check_v2(apr)
    ratio = median_ttft("apr", apr) / median_ttft("llamacpp", ref)
    return ratio, ratio <= BOUND


def run(path):
    ratio, ok = evaluate(json.loads(Path(path).read_text() or "null"))
    print(f"V1 ttft ratio apr/llamacpp = {ratio:.3f} (bound {BOUND}): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


EXPECT = {"pass.json": 0, "ratio-fail.json": 1, "identity-mismatch.json": 2,
          "missing-load.json": 2, "wrong-pin.json": 2, "few-samples.json": 2, "empty.json": 2,
          "fabricated-p50.json": 2, "failed-requests.json": 2, "no-backend-proof.json": 2,
          "class-mismatch.json": 2, "summary-only.json": 2, "bad-binary-sha.json": 2,
          "unpinned.json": 2, "coarse-load.json": 2}


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
