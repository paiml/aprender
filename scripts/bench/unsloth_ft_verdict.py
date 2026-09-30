#!/usr/bin/env python3
"""Verdict for beat-unsloth-finetune-throughput-v1 (0.72 T2, row R5).

This is the part of the R5 harness that decides. It reads the per-run
receipts the GPU half writes (one JSON per run per side) and prints ONE
verdict. It never runs a model, so every rule below is tested on a CPU box
by scripts/tests/unsloth_ft_verdict_test.py.

Exit codes (0/1/2, the same shape as the serve witness):
  0  PASS          ratio >= threshold, same work, incumbent on its fast path
  1  FAIL          SAME-WORK FAIL (names the field), or ratio < threshold
  2  NOT_MEASURED  too few runs, a missing field, the incumbent on its slow
                   path (INCUMBENT_SLOW_PATH), or the two sides on different
                   GPUs. NOT_MEASURED is never a pass (L25) and has no ratio.

Usage:
  unsloth_ft_verdict.py --apr r1.json r2.json r3.json \
                        --incumbent u1.json u2.json u3.json [--threshold 0.8]
"""
import argparse
import json
import statistics
import sys

THRESHOLD = 0.80
MIN_RUNS = 3
PARAM_TOLERANCE = 0.01

# Every receipt carries these. A missing one is NOT_MEASURED, never a default.
COMMON_FIELDS = (
    "side", "gpu_name", "gpu_uuid", "device_trace_line", "model", "data_sha256",
    "rank", "alpha", "targets", "trainable_params", "optimizer",
    "grad_checkpointing", "packing", "seq_len", "batch", "grad_accum",
    "warmup_steps", "timed_steps", "timed_after_compile",
    "label_tokens_timed", "timed_seconds",
)
# The same key names as train-run-receipt-v1 (TRR-002).
APR_FIELDS = ("apr_version", "apr_git_sha")
INCUMBENT_VERSIONS = ("unsloth", "unsloth_zoo", "torch", "transformers", "fla", "causal_conv1d")

# contract 1.1.0 canonical_task pins these on BOTH sides.
PINNED = dict(
    optimizer="adamw_fp32",
    grad_checkpointing=False,
    packing=False,
    timed_after_compile=True,
)
# Equal on every run of both sides, or the two sides did different work.
SAME_WORK_FIELDS = (
    "model", "data_sha256", "rank", "alpha", "seq_len", "batch", "grad_accum",
    "warmup_steps", "timed_steps",
)


class Verdict(Exception):
    def __init__(self, code, verdict, reason):
        super().__init__(reason)
        self.code = code
        self.verdict = verdict
        self.reason = reason


def not_measured(reason):
    return Verdict(2, "NOT_MEASURED", reason)


def same_work_fail(reason):
    return Verdict(1, "FAIL", "SAME-WORK FAIL: " + reason)


def check_fields(r, where):
    need = COMMON_FIELDS + (APR_FIELDS if r.get("side") == "apr" else ("versions",))
    missing = [f for f in need if f not in r or r[f] is None]
    if missing:
        raise not_measured("%s: missing %s" % (where, ",".join(missing)))
    if r["side"] == "incumbent":
        v = r["versions"]
        if not isinstance(v, dict):
            raise not_measured("%s: versions is not a map" % where)
        if not v.get("fla"):
            raise Verdict(2, "INCUMBENT_SLOW_PATH",
                          "%s: fla absent, HF GDN runs its fp32 torch fallback" % where)
        absent = [k for k in INCUMBENT_VERSIONS if not v.get(k)]
        if absent:
            raise not_measured("%s: versions missing %s" % (where, ",".join(absent)))
    if r["timed_seconds"] <= 0 or r["label_tokens_timed"] <= 0:
        raise not_measured("%s: empty timed window" % where)


def check_pins(r, where):
    for k, want in PINNED.items():
        if r[k] != want:
            raise same_work_fail("%s: %s=%r, canonical task pins %r" % (where, k, r[k], want))


def tok_s(r):
    return r["label_tokens_timed"] / r["timed_seconds"]


def decide(apr, inc, threshold=THRESHOLD):
    """Return (exit code, verdict, reason, ratio or None)."""
    try:
        for side, runs in (("apr", apr), ("incumbent", inc)):
            if len(runs) < MIN_RUNS:
                raise not_measured("%s: %d runs, need %d" % (side, len(runs), MIN_RUNS))
            for i, r in enumerate(runs):
                where = "%s run %d" % (side, i + 1)
                if r.get("side") != side:
                    raise not_measured("%s: side=%r" % (where, r.get("side")))
                check_fields(r, where)
        every = apr + inc
        gpus = set(r["gpu_uuid"] for r in every)
        if len(gpus) != 1:
            raise not_measured("runs span %d GPUs: %s" % (len(gpus), ",".join(sorted(gpus))))
        for side, runs in (("apr", apr), ("incumbent", inc)):
            for i, r in enumerate(runs):
                check_pins(r, "%s run %d" % (side, i + 1))
        a0, u0 = apr[0], inc[0]
        for k in SAME_WORK_FIELDS:
            for r in every:
                if r[k] != a0[k]:
                    raise same_work_fail("%s differs (%r vs %r)" % (k, a0[k], r[k]))
        for r in every:
            if sorted(r["targets"]) != sorted(a0["targets"]):
                raise same_work_fail("targets differ (%s vs %s)" % (
                    ",".join(sorted(a0["targets"])), ",".join(sorted(r["targets"]))))
        pa, pu = a0["trainable_params"], u0["trainable_params"]
        if pa <= 0 or pu <= 0 or abs(pa - pu) / max(pa, pu) >= PARAM_TOLERANCE:
            raise same_work_fail("trainable_params %d vs %d (tolerance %.0f%%)" % (
                pa, pu, PARAM_TOLERANCE * 100))
        ratio = statistics.median(tok_s(r) for r in apr) / statistics.median(tok_s(r) for r in inc)
    except Verdict as v:
        return v.code, v.verdict, v.reason, None
    if ratio >= threshold:
        return 0, "PASS", "ratio %.4f >= %.4f" % (ratio, threshold), ratio
    return 1, "FAIL", "ratio %.4f < %.4f" % (ratio, threshold), ratio


def load(paths):
    runs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apr", nargs="*", default=[])
    ap.add_argument("--incumbent", nargs="*", default=[])
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    a = ap.parse_args(argv)
    out = dict()
    try:
        apr, inc = load(a.apr), load(a.incumbent)
    except (OSError, ValueError) as err:
        code, out["verdict"], out["reason"] = 2, "NOT_MEASURED", "unreadable receipt: " + str(err)
    else:
        code, out["verdict"], out["reason"], ratio = decide(apr, inc, a.threshold)
        if ratio is not None:
            out["ratio"] = round(ratio, 4)
    print(json.dumps(out))
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
