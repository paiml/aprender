#!/usr/bin/env python3
"""Check an apr-trace-v1 receipt (0.71 exit criterion V6, APR-OBS-001 OBS-09/OBS-11).

A trace is what `apr serve` returns for a request sent with `X-Trace-Level: step|layer`.
  T1 the §2.1 identity block is complete; backend != cpu needs gpu_proof.
  T2 provenance "Measured" only when the tracer ran and emitted events.
  T3 the per-layer sum is <= the request wall; so is the sum of top-level steps
     (nested steps such as attention/ffn run inside transformer_block and are not summed).
  T4 layer traces cover layers 0..n_layers-1 exactly once; step names are TraceStep names.
  T5 (--diff) a before/after pair (OBS-11) must share host, model, backend, prompt, level.

Exit codes: 0 ADMISSIBLE, 2 RED/REFUSED.
"""
import argparse
import json
import sys
from pathlib import Path

SCHEMA = "apr-trace-v1"
IDENTITY = ("ts", "host", "apr_version", "binary_sha256", "build_identity",
            "model_id", "model_sha256", "backend", "request_id")
SHARED = ("host", "model_sha256", "backend", "prompt_sha256", "trace_level")
TOP_STEPS = {"tokenize", "embed", "transformer_block", "lm_head", "sample", "decode"}
NESTED_STEPS = {"layer_norm", "attention", "ffn", "kernel_launch", "brick_profile"}
LEVELS = ("step", "layer")
EPS_MS = 1e-6
HERE = Path(__file__).resolve().parent


class Red(Exception):
    pass


def require(cond, message):
    if not cond:
        raise Red(message)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def check_identity(doc):
    missing = [k for k in IDENTITY if doc.get(k) in (None, "", "unknown")]
    require(not missing, f"T1: identity fields missing: {missing}")
    require(doc["backend"] == "cpu" or doc.get("gpu_proof"), "T1: backend != cpu without gpu_proof")


def check_provenance(doc):
    tracer = doc.get("tracer") or {}
    require(doc.get("provenance") == "Measured", "T2: timings without provenance Measured are RED")
    require(tracer.get("ran") is True and (tracer.get("events") or 0) > 0,
            "T2: provenance Measured but the tracer did not run (or emitted 0 events)")


def check_layers(doc):
    layers = doc.get("layers") or []
    n = doc.get("n_layers")
    require(isinstance(n, int) and n > 0, "T4: layer trace needs n_layers > 0")
    idx = sorted(row.get("layer") for row in layers if isinstance(row.get("layer"), int))
    require(idx == list(range(n)), f"T4: layers must be 0..{n - 1} once each, got {len(idx)} rows")
    require(all(number(row.get("ms")) for row in layers), "T4: every layer needs ms >= 0")
    return sum(row["ms"] for row in layers)


def check_steps(doc):
    steps = doc.get("steps") or []
    names = [row.get("step") for row in steps]
    unknown = sorted({s for s in names if s not in TOP_STEPS | NESTED_STEPS}, key=str)
    require(not unknown, f"T4: unknown TraceStep names: {unknown}")
    require(steps and all(number(row.get("ms")) for row in steps), "T4: steps need ms >= 0")
    return sum(row["ms"] for row in steps if row["step"] in TOP_STEPS)


def check_wall(doc):
    wall = doc.get("wall_ms")
    require(number(wall) and wall > 0, "T3: wall_ms missing")
    summed = check_layers(doc) if doc["trace_level"] == "layer" else check_steps(doc)
    require(summed <= wall + EPS_MS, f"T3: {doc['trace_level']} sum {summed} ms > wall {wall} ms")
    return summed, wall


def evaluate(doc):
    require(isinstance(doc, dict) and doc.get("schema") == SCHEMA, f"schema is not {SCHEMA}")
    check_identity(doc)
    require(doc.get("trace_level") in LEVELS, f"trace_level must be one of {LEVELS}")
    check_provenance(doc)
    return check_wall(doc)


def evaluate_diff(doc):
    require(isinstance(doc, dict), "diff needs {before, after}")
    before, after = doc.get("before"), doc.get("after")
    evaluate(before)
    evaluate(after)
    differ = [k for k in SHARED if before.get(k) != after.get(k)]
    require(not differ, f"T5: REFUSED: before/after identity differs on {differ}")
    require(before["binary_sha256"] != after["binary_sha256"], "T5: before and after are the same binary")
    return before["wall_ms"], after["wall_ms"]


def load(path):
    return json.loads(Path(path).read_text() or "null")


EXPECT = {"pass-layer.json": 0, "pass-step.json": 0, "measured-not-run.json": 2, "measured-zero-events.json": 2,
          "layer-sum-over-wall.json": 2, "step-sum-over-wall.json": 2,
          "missing-identity.json": 2, "gpu-no-proof.json": 2, "sparse-layers.json": 2,
          "unknown-step.json": 2, "unmeasured-timings.json": 2, "empty.json": 2}
EXPECT_DIFF = {"diff-pass.json": 0, "diff-identity-mismatch.json": 2, "diff-same-binary.json": 2}


def outcome(fn, path):
    try:
        fn(load(path))
        return 0
    except (Red, AttributeError, TypeError, KeyError):
        return 2


def self_test():
    root = HERE / "fixtures" / "trace"
    cases = [(evaluate, n, e) for n, e in EXPECT.items()]
    cases += [(evaluate_diff, n, e) for n, e in EXPECT_DIFF.items()]
    bad = [(n, e, got) for fn, n, e in cases if (got := outcome(fn, root / n)) != e]
    for name, want, got in bad:
        print(f"SELF-TEST FAIL {name}: expected {want}, got {got}")
    print(f"self-test: {len(cases) - len(bad)}/{len(cases)} fixtures as planted")
    return 2 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("receipt", nargs="?")
    ap.add_argument("--diff", action="store_true", help="receipt is {before, after} (OBS-11)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test or not args.receipt:
        return self_test()
    try:
        result = (evaluate_diff if args.diff else evaluate)(load(args.receipt))
    except (Red, AttributeError, TypeError, KeyError) as err:
        print(f"RED: {err}")
        return 2
    print(f"ADMISSIBLE: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
