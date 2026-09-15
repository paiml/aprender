#!/usr/bin/env python3
"""Convert what this repo MEASURES into what scripts/perf_gate.sh READS.

THE DEFECT THIS CLOSES (PERF-004). perf_gate.sh accepted no artifact any
in-tree producer emitted. Four real inputs were tried and all four returned
rc=1; only the gate's own synthetic fixture passed. Two incompatible band
schemas had grown side by side:

    parity_block.py   parity.lanes[].bands[].ratio_aggregate_tok_per_sec
    perf_gate.sh                      bands[].agg_ratio

THE RULE THIS OBEYS. Five fields the gate requires -- `requested`, `completed`,
`timeouts`, `drain_ms`, `tokenization.method` -- looked like the same kind of
gap. They are not. Two are DERIVABLE from counts the harness already records.
Three are UNMEASURED: nothing in this tree measures them, and teaching this
converter to emit a plausible value for them would hand-assign three numbers
that no instrument produced, which is the exact defect APR-PERF-GATE-001 exists
to remove.

So this file derives the first two and REFUSES the last three. In their place it
copies the UNMEASURED entries out of scripts/perf-receipt-fields.yaml into the
receipt's `unmeasured` block, with each field's owning ticket and spec section.
The gate then fails with "drain_ms absent -- the drain rule (§4.4.7) is not
implemented in any client (owner PERF-004)" instead of "ArmC schema". A finding
names the thing that has to be built; a schema error names nothing.

USAGE

  # from band artifacts: DIR/{prefix}-c{N}.json, each an `apr test llm bench
  # --output` BenchmarkReport
  perf_receipt.py --from-bands DIR --subject apr --comparator llamacpp \\
                  --host lambda --workload W1 --out receipt.json \\
                  --binary PATH --binary-sha256 HEX --compute-class cuda \\
                  --model NAME --quantization Q4_K_M

  # from a parity block, which already carries provenance for both sides
  perf_receipt.py --from-parity parity.json --host lambda --workload W1 \\
                  --out receipt.json

  # re-derive the eight committed JOIN digits from evidence/parity-http/bands
  perf_receipt.py --fixture-check

  # derive and print, write nothing. The only mode that needs no provenance,
  # and therefore the only one that can read a corpus whose binary digest was
  # never recorded.
  perf_receipt.py --from-bands DIR --subject apr --comparator llamacpp \\
                  --derive-only

Exit: 0 wrote (or printed) a receipt - 1 refused - 2 usage/read error

REPRODUCING THE END-TO-END RUN, AND WHAT IT DOES NOT LICENSE
------------------------------------------------------------
The commit that added this file quotes a live run: band artifacts produced by
`apr test llm bench`, converted here, and given a verdict by
scripts/perf_gate.sh. State the instrument exactly, because a benchmark
artifact is only as identified as the binary that made it.

  subject     the cargo-installed `apr` on that box, `apr 0.64.0 (ce712eae0)`
              sha256 964503625a69462e24964c5b8118b1b78a1c0cae8e4dbdf845b2da25281d01c9
  comparator  a locally built llama-server,
              `version: 7746 (39173bcac)` == scripts/llama_pin.toml's pin

  Neither is named by its absolute path, and losing the path loses nothing: a
  binary is identified by its digest and its version stamp, which is the whole
  argument three lines below. The path is machine-specific provenance that
  check_hardcoded_paths.sh --full counts as a shipped finding, and #2733's
  PERF-032 armed that ratchet -- it had never gated anything before, which is
  how twenty of them landed.
  compute     cpu on BOTH sides, read from the runtimes' own logs rather than
              from the flags: llama printed "offloaded 0/25 layers to GPU", and
              apr printed no CUDA banner, which is what
              parity_host_receipt.sh's apr_class_from_log() calls cpu.

`. scripts/apr_bin.sh` REFUSES on that box -- rc=1, "STALE apr BINARY ...
reports apr 0.64.0 (ce712eae0), HEAD a468eac4e". There is no apr built from
HEAD anywhere on it, so no step could have used one, and saying so is the
honest form. The binary that ran is three commits behind, and the other apr on
PATH is worse: ~/.local/bin/apr is stamped `v0.64.0+no-git`, carrying no commit
at all, so nothing could establish what source built it. It won a bare `apr`.

WHY THAT RUN STILL PROVES WHAT IT CLAIMS, and only that. The subject under test
is this converter and perf_gate.sh AT HEAD -- Python and Bash, run from this
checkout. apr is the INSTRUMENT that produced the input artifact, and the
receipt records which one by digest rather than by name. The Rust delta between
ce712eae0 and HEAD is five files, all of them `apr profile` (PERF-016):
commands/profile.rs plus the four it `include!`s, reached only through
`ExtendedCommands::Profile`. Neither executed step -- `apr serve run` and
`apr test llm bench` -- is in that path; the one cross-module consumer,
serve_plan.rs, imports `detect_gpu_hardware` and `query_gpu_vram_mb`, and the
delta touches neither.

WHAT IT DOES NOT LICENSE: any statement about apr's speed. Those ratios are a
CPU-only published binary against llama.cpp on a box at load 16-74, ten-second
runs, two runs per band. They are not a parity result, they are not cited as
one anywhere, and the receipt is deliberately left in scratch rather than
committed under evidence/. Rebuilding to satisfy apr_bin.sh was declined
rather than skipped: `cargo install --path crates/apr-cli --force` overwrites
~/.cargo/bin/apr while other agents on this box may be running it, and it would
not change a single thing the run establishes.
"""
import argparse
import json
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_receipt  # noqa: E402

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(SCRIPTS, "perf-receipt-fields.yaml")
ROOT = os.path.dirname(SCRIPTS)

# THE LADDER IS READ, NOT RESTATED (PP-24/PP-33). This file carried the tuple
# (1, 4, 8, 16) TWICE, and three other files carried it as well; a matrix edit
# changed the gate's expected cell set and left every producer emitting the old
# ladder. `bench_receipt.declared_bands()` is the one reader.
def bands_default():
    return tuple(bench_receipt.declared_bands())


# THE SPEC GENERATION THIS CONVERTER WRITES.
#
# `schema_version` is the WIRE generation, and this converter's inputs are
# HISTORICAL band artifacts: they carry no server-reported prefill timings, no
# batch-invariance witness, no comparator run_id and no per-request rows beyond
# latency. It therefore writes a v2 wire under the v3 spec and says so in both
# fields, rather than stamping `3` on a document that cannot satisfy the v3
# rules. perf_gate.sh's L1 arm reads exactly this distinction and reports such a
# receipt as historical instead of failing it for not having been written in the
# future.
SCHEMA = "perf-receipt/v3.0"
SPEC_VERSION = "PP-LLAMA-001 v3.0"
WIRE_SCHEMA_VERSION = 2
# The wire generation a P1 EXECUTOR block reaches. The restructured executor
# writes the interleaved replicates, the comparator's declared admission, the
# subject's own resolved configuration and the device record, which is the set
# of facts the v3 rules are stated over -- so a receipt derived from one is a
# v3 document and says so. A historical single-report-per-lane corpus is not,
# and still converts to 2.
WIRE_SCHEMA_VERSION_V3 = 3


# ===========================================================================
# THE TWO ESTIMATORS (PP-LLAMA-001 §4.3; the table is quoted verbatim from the
# implementation contract's Statistics section).
#
# WINDOW-UNIT metrics -- aggregate and prefill throughput -- are ONE number per
# replicate: there is no per-request aggregate. Their estimator is therefore
# over replicates: the mean of the per-replicate ln(subject/comparator), a
# one-sided t lower bound at 95% with df = n-1, exponentiated. Below five
# replicates it bounds nothing and reports only.
#
# REQUEST-UNIT metrics -- decode rate, ttft, itl -- have one value per request,
# so the bound comes from resampling WHOLE REQUESTS.
#
# THE RUST PRODUCER IS THE CONFORMANT ONE. crates/aprender-test-lib/src/
# perf_gate/ draws its resamples from a SplitMix64 stream; this is Python's
# Mersenne Twister through `random.Random(2026)`, so the two are NOT
# bit-identical and were never meant to be. This is the derivation for the
# legacy shell harness, whose inputs are BenchmarkReport files rather than the
# producer's own sample rows; where the two disagree the Rust one governs.
# ===========================================================================
#
# One-sided t, lower tail, 95%, by degrees of freedom. Key 31 is the >30 row:
# the table is looked up at min(df, 31) so a large sample takes the normal
# quantile rather than falling off the end.
T_LOWER_ONE_SIDED_95 = {
    1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895,
    8: 1.860, 9: 1.833, 10: 1.812, 11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761,
    15: 1.753, 16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729, 20: 1.725,
    21: 1.721, 22: 1.717, 23: 1.714, 24: 1.711, 25: 1.708, 26: 1.706,
    27: 1.703, 28: 1.701, 29: 1.699, 30: 1.697, 31: 1.645,
}
T_TABLE_MAX_DF = 31
# Resamples, seed and lower-bound percentile for the request-unit bootstrap.
# All three are INTEGERS and definitional -- they say how the interval is
# constructed, not what value a verdict is compared against -- so they are not
# matrix thresholds (PP-33 exempts definitional comparisons).
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 2026
LCB_PERCENTILE = 5
REPLICATE_METHOD = "replicate_t_lower"
BOOTSTRAP_METHOD = "paired_percentile_bootstrap"


def t_lower_one_sided_95(df):
    """The one-sided 95% t multiplier for `df` degrees of freedom."""
    return T_LOWER_ONE_SIDED_95[min(df, T_TABLE_MAX_DF)]


def _positive_pairs(subject, comparator):
    """The k-th subject value beside the k-th comparator value, both positive.

    A non-positive rate is not a measurement, and a pair missing on one side is
    not a pair: the replicate estimator divides the k-th by the k-th, so
    silently shortening one list would pair replicate 3 with replicate 4.
    """
    pairs = []
    for s, c in zip(subject or [], comparator or []):
        if isinstance(s, (int, float)) and isinstance(c, (int, float)) \
                and s > 0 and c > 0:
            pairs.append((float(s), float(c)))
    return pairs


def replicate_t_lower(subject, comparator, n_min):
    """§4.3's window-unit estimator, or None when there is nothing to divide."""
    pairs = _positive_pairs(subject, comparator)
    n = len(pairs)
    if n == 0:
        return None
    logs = [math.log(s / c) for s, c in pairs]
    mean = statistics.fmean(logs)
    out = {"point": math.exp(mean), "lcb95": None, "method": REPLICATE_METHOD,
           "n": n}
    if n < n_min:
        # A point estimate with no bound, said out loud. The alternative --
        # reporting a bound computed from two replicates -- is a number whose
        # width is an artefact of the sample size.
        out["note"] = "n<%d: reporting only" % n_min
        return out
    sd = statistics.stdev(logs)
    out["lcb95"] = math.exp(mean - t_lower_one_sided_95(n - 1) * sd / math.sqrt(n))
    return out


def paired_percentile_bootstrap(subject, comparator):
    """§4.3's request-unit estimator over whole requests.

    ONE PRNG STREAM, drawn subject-then-comparator inside each resample, so the
    two lanes of a resample are as paired as the interleaved invocation that
    produced them. `lcb95` is the LCB_PERCENTILE-th percentile of the ratio
    draws by lower-index rank, which is the convention the Rust producer uses.
    """
    subject = [float(v) for v in (subject or []) if isinstance(v, (int, float)) and v > 0]
    comparator = [float(v) for v in (comparator or []) if isinstance(v, (int, float)) and v > 0]
    if not subject or not comparator:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        s = statistics.median(rng.choices(subject, k=len(subject)))
        c = statistics.median(rng.choices(comparator, k=len(comparator)))
        if c > 0:
            draws.append(s / c)
    if not draws:
        return None
    draws.sort()
    index = (LCB_PERCENTILE * (len(draws) - 1)) // 100
    return {"point": statistics.median(subject) / statistics.median(comparator),
            "lcb95": draws[index], "method": BOOTSTRAP_METHOD,
            "n": min(len(subject), len(comparator))}


# --------------------------------------------------------------- the ledger --
def unmeasured_from_ledger(path=LEDGER):
    """The UNMEASURED bucket, keyed by field, straight out of the map.

    Read rather than restated so the receipt cannot drift from the ledger the
    guard checks. A hand-copied list is a second definition, and a second
    definition is how the two band schemas happened.
    """
    import yaml
    with open(path, encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    out = {}
    for name, spec in (doc.get("fields") or {}).items():
        if spec.get("class") != "UNMEASURED":
            continue
        out[name] = {"owner": spec.get("owner"), "spec": spec.get("spec"),
                     "needs": " ".join((spec.get("needs") or "").split())}
    return out


# ------------------------------------------------------------ P1: bands dir --
def _runs(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["runs"]


def _median(runs, key):
    return statistics.median([r[key] for r in runs])


def _band_paths(work, prefix, c):
    return os.path.join(work, "%s-c%d.json" % (prefix, c))


def _one_concurrency(runs):
    """The concurrency this artifact was run at, or None if it disagrees.

    A band file whose runs were taken at different concurrencies is not one
    band, and averaging across it silently reports a mixture as a measurement.
    """
    levels = {r.get("concurrency") for r in runs}
    return levels.pop() if len(levels) == 1 else None


def _samples_ms(runs):
    """Per-request latencies, retained per §4.4.5 so a bootstrap stays possible."""
    out = []
    for run in runs:
        out.extend(d["latency_ms"] for d in run.get("request_details", []))
    return out


def _sides_for(work, subject, comparator, c, errors):
    """Both sides of one band, or None with the reason recorded."""
    s_path, c_path = _band_paths(work, subject, c), _band_paths(work, comparator, c)
    if not os.path.exists(s_path):
        return None
    if not os.path.exists(c_path):
        errors.append("band c=%d has a subject and no comparator; refusing to "
                      "report half a comparison" % c)
        return None
    s_runs, c_runs = _runs(s_path), _runs(c_path)
    if _one_concurrency(s_runs) != c or _one_concurrency(c_runs) != c:
        errors.append("band c=%d: the artifact's own runs do not all report "
                      "concurrency=%d" % (c, c))
        return None
    return s_runs, c_runs


def _band_from_runs(c, s_runs, c_runs):
    """One band, every value derived from the two artifacts' own samples."""
    band = {
        "concurrency": c,
        "aggregate_tok_per_sec": _median(s_runs, "tokens_per_sec"),
        "decode_tok_per_sec": _median(s_runs, "decode_tok_per_sec"),
        "tokens_total": sum(r["completion_tokens_total"] for r in s_runs),
        "agg_ratio": _median(s_runs, "tokens_per_sec") / _median(c_runs, "tokens_per_sec"),
        "decode_ratio": _median(s_runs, "decode_tok_per_sec") / _median(c_runs, "decode_tok_per_sec"),
        # THE COMPARATOR'S OWN COMPLETION, carried rather than dropped.
        # llama.cpp lost 3/80, 3/223, 6/302 and 10/522 requests on the
        # 2026-08-25 corpus. Its tok/s counts tokens from SUCCESSFUL requests
        # only over the same wall clock, so every lost request lowers the
        # denominator of agg_ratio and flatters the subject. Arm C asserts
        # completed == requested on the subject and never looked at the other
        # side of its own ratio.
        "comparator_requested": sum(r["total_requests"] for r in c_runs),
        "comparator_completed": sum(r["successful"] for r in c_runs),
    }
    # PP-4: prefill was DROPPED at this boundary while the input carried it, so
    # a receipt could not state one third of the metrics the spec gates on.
    #
    # `prefill_source` is the load-bearing half. The harness derives this from
    # the client's own first-token timing, which at c>1 measures queueing rather
    # than prefill -- on the 2026-08-25 corpus the client-derived figure falls
    # to single digits at c=16 while the server reports thousands. Carrying it
    # unlabelled under the server-timed name would publish a number whose
    # semantics nobody could recover, so it is labelled at the point of
    # creation and perf_gate.sh refuses the label on a v3 receipt.
    if all("prefill_tok_per_sec" in r for r in s_runs):
        band["prefill_tok_per_sec"] = _median(s_runs, "prefill_tok_per_sec")
        band["prefill_source"] = "client-derived"
        if all("prefill_tok_per_sec" in r for r in c_runs):
            band["prefill_ratio"] = (_median(s_runs, "prefill_tok_per_sec")
                                     / _median(c_runs, "prefill_tok_per_sec"))
    if all("ttft_p50_ms" in r for r in s_runs):
        band["ttft_p50_ms"] = _median(s_runs, "ttft_p50_ms")
    return band


def bands_from_dir(work, subject, comparator, errors):
    """One band per concurrency for which BOTH sides exist."""
    bands, samples, requested, completed = [], [], 0, 0
    for c in bands_default():
        sides = _sides_for(work, subject, comparator, c, errors)
        if sides is None:
            continue
        s_runs, c_runs = sides
        bands.append(_band_from_runs(c, s_runs, c_runs))
        samples.extend(_samples_ms(s_runs))
        requested += sum(r["total_requests"] for r in s_runs)
        completed += sum(r["successful"] for r in s_runs)
    if not bands:
        errors.append("no band had both sides present under %s" % work)
    return bands, samples, requested, completed


# --------------------------------------------------------- P2: parity block --
def _lane_of(block, want, errors):
    lanes = block.get("lanes") or []
    if want:
        lanes = [l for l in lanes if l.get("lane") == want]
        if not lanes:
            errors.append("no lane named %r in the parity block" % want)
            return None
    if len(lanes) != 1:
        errors.append("the parity block carries %d lanes; name one with --lane"
                      % len(lanes))
        return None
    return lanes[0]


def _convert_parity_band(raw):
    """THE RENAME AND THE RE-NESTING, in one place.

        parity.lanes[].bands[].ratio_aggregate_tok_per_sec  ->  bands[].agg_ratio

    Ratios are RE-DERIVED from the samples beside them and never copied: a
    stated ratio is the one field that can be wrong while every field around it
    is right, which is parity_block.py's own rule applied to its own output.
    """
    subj, comp = raw.get("subject") or {}, raw.get("comparator") or {}
    band = {"concurrency": raw.get("concurrency"),
            "aggregate_tok_per_sec": statistics.median(subj["aggregate_tok_per_sec"]),
            "decode_tok_per_sec": statistics.median(subj["decode_tok_per_sec"])}
    if subj.get("prefill_tok_per_sec") and comp.get("prefill_tok_per_sec"):
        band["prefill_tok_per_sec"] = statistics.median(subj["prefill_tok_per_sec"])
        band["prefill_source"] = "client-derived"
        band["prefill_ratio"] = (statistics.median(subj["prefill_tok_per_sec"])
                                 / statistics.median(comp["prefill_tok_per_sec"]))
    for src, dst in (("aggregate_tok_per_sec", "agg_ratio"),
                     ("decode_tok_per_sec", "decode_ratio")):
        band[dst] = statistics.median(subj[src]) / statistics.median(comp[src])
    # Omitted rather than guessed when the block predates PERF-004: Arm C's
    # zero-token check reads an absent count as "not zero", which is the honest
    # reading of "this document never carried it".
    for src, dst in (("tokens_total", "tokens_total"),):
        if subj.get(src) is not None:
            band[dst] = subj[src]
    for src, dst in (("requested", "comparator_requested"),
                     ("completed", "comparator_completed")):
        if comp.get(src) is not None:
            band[dst] = comp[src]
    return band


def bands_from_parity(path, lane_name, errors):
    """Rename and re-nest P2's bands into the shape the gate reads.

    Ratios are RE-DERIVED from the samples beside them rather than copied:
    parity_block.py's own doctrine is that nothing accepts a ratio as input,
    because a stated ratio is the one field that can be wrong while every field
    around it is right.
    """
    with open(path, encoding="utf-8") as handle:
        doc = json.load(handle)
    block = doc.get("parity") if isinstance(doc.get("parity"), dict) else doc
    lane = _lane_of(block, lane_name, errors)
    if lane is None:
        return None, None, None
    if lane.get("comparability") == "cross-class-existence-only":
        errors.append("lane %r is cross-class-existence-only. A cross-class row "
                      "cannot arm Arm B's floor, and a receipt that merely MARKS "
                      "it still walks into Arm A, which reads no comparator at "
                      "all and would gate it happily. Refusing." % lane.get("lane"))
        return None, None, None
    bands = [_convert_parity_band(raw) for raw in (lane.get("bands") or [])]
    if not bands:
        errors.append("lane %r carries no bands" % lane.get("lane"))
        return None, None, None
    subject = lane.get("subject") or {}
    return bands, subject, lane


# ============================================================================
# P2 -> P3, EXECUTOR LAYOUT: the v3 wire.
#
# A parity block written from the restructured executor carries what the v2
# conversion above could not state: the replicates were INTERLEAVED inside one
# invocation, so the k-th subject value and the k-th comparator value are a
# PAIR; the comparator declared its own admission before the first request; the
# subject declared its own resolved configuration; and the device was recorded
# either side of the band. Those are exactly the facts §7.4's status vocabulary
# is stated over, so this branch emits schema_version 3 -- and, per PP-3, never
# a bare scalar ratio: a ratio lives in `ratios` beside the `baseline` it was
# taken against, sharing a run_id, or it does not exist.
# ============================================================================
def _matrix_protocol():
    return bench_receipt.matrix_number("protocol")


def _wire_samples(rows):
    """PP-7's raw rows in the wire shape.

    `issued_ms`, `settled_ms` and `in_flight_at_start` are positions on the
    window's clock and this harness records a DURATION per request instead.
    They are null here and named in unproduced_fields; putting `latency_ms`
    under `settled_ms` would publish a duration as an instant.
    """
    out = []
    for row in rows or []:
        out.append({"index": row.get("index"),
                    "issued_ms": None,
                    "settled_ms": None,
                    "latency_ms": row.get("latency_ms"),
                    "outcome": "completed" if row.get("finish_reason") else None,
                    "generated_tokens": row.get("completion_tokens"),
                    "prompt_tokens": row.get("prompt_tokens"),
                    "ttft_ms": row.get("ttft_ms"),
                    "in_flight_at_start": None})
    return out


def _median_or_none(values):
    values = [v for v in (values or []) if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def _lane_band(side, c, ctx):
    """The metric half of one band, for one lane. Shared by band and baseline."""
    span = _median_or_none(side.get("elapsed_secs"))
    return {
        "concurrency": c,
        "aggregate_tok_per_sec": _median_or_none(side.get("aggregate_tok_per_sec")),
        "decode_tok_per_sec": _median_or_none(side.get("decode_tok_per_sec")),
        "prefill_tok_per_sec": _median_or_none(side.get("prefill_tok_per_sec")),
        # CLIENT-DERIVED, said at the point of creation. §3 wants a
        # SERVER-reported prefill; this harness times the first token at the
        # client, which at c>1 measures queueing. perf_gate.sh refuses the
        # label on a v3 receipt, which is the correct outcome: the field is
        # reported and cannot arm anything.
        "prefill_source": "client-derived",
        "tokens_total": side.get("tokens_total"),
        "requested": side.get("requested"),
        "completed": side.get("completed"),
        "timeouts": side.get("timeouts"),
        "drain_ms": side.get("drain_ms"),
        "short_of_n_predict": side.get("short_of_n_predict"),
        "window_ms": ctx["window_ms"],
        "span_ms": span * 1000.0 if span is not None else None,
        "client_concurrency": side.get("client_concurrency"),
        "stream_mode": side.get("stream_mode"),
        "witness": side.get("witness"),
        "samples": _wire_samples(side.get("request_rows")),
        "replicate_files": side.get("replicate_files"),
    }


def _stream_witness(side, ctx):
    """PP-27's client half: how much of a request elapsed before token one."""
    median = _median_or_none(side.get("request_ttft_over_e2e"))
    if median is None:
        return None
    ceiling = ctx["live_ttft_over_e2e_max"]
    verdict = "undeclared" if side.get("stream_mode") is None else "live"
    if median >= ceiling:
        verdict = "replayed"
    return {"client_ttft_over_e2e_median": median, "verdict": verdict}


def _join_key(raw, c, ctx):
    """PP-22. Two receipts whose join keys differ may not be quotiented.

    `window_ms`, `replicates` and `interleaved` are in the key because a ratio
    taken over a different window, or over a sweep rather than an interleave,
    is a different experiment; `n_ctx_slot`, `kv_type`, `fa` and `n_batch` are
    in it because a `-b 1` comparator lane is a different comparator.
    """
    admission = raw.get("comparator_admission") or {}
    return {"host": ctx["host"], "workload": ctx["workload"], "band": c,
            "model": ctx["model"], "quant": ctx["quantization"],
            "tokenization": ctx["tokenization"],
            "window_ms": ctx["window_ms"],
            "replicates": raw.get("replicates"),
            "interleaved": raw.get("interleaved"),
            "n_ctx_slot": admission.get("n_ctx_slot"),
            "kv_type": admission.get("kv_type"),
            "fa": admission.get("fa"),
            "n_batch": admission.get("n_batch"),
            "n_predict": ctx["n_predict"]}


def _nonconformance(raw, band, ctx):
    """Every §7.4 reason this band may be cited but may not arm a threshold."""
    out = []
    if raw.get("interleaved") is not True:
        out.append("protocol.interleaved=%r -- a sweep is not a paired "
                   "measurement (§4.3)" % (raw.get("interleaved"),))
    n = raw.get("replicates")
    if not isinstance(n, int) or n < ctx["replicates_min"]:
        out.append("replicates=%r is below perf-matrix.yaml protocol."
                   "replicates_min=%d, which bounds no variance"
                   % (n, ctx["replicates_min"]))
    if band.get("stream_mode") != "live":
        out.append("stream_mode=%r -- ttft, itl and decode are undefined "
                   "without a live stream (PP-27)" % (band.get("stream_mode"),))
    short = band.get("short_of_n_predict")
    if short is None:
        out.append("short_of_n_predict is unreported, so nothing witnesses a "
                   "completion that stopped early (PP-28)")
    elif short:
        out.append("short_of_n_predict=%d -- the sampler did not hold, so the "
                   "throughput denominators are not comparable" % short)
    timeouts = band.get("timeouts")
    if timeouts is None:
        out.append("timeouts are unreported -- a hung request produces no "
                   "sample at all, so it cannot be discarded, it simply never "
                   "appears")
    elif timeouts:
        out.append("timeouts=%d" % timeouts)
    isolation = raw.get("isolation") or {}
    if isolation.get("contended"):
        out.append("the band shared the device with foreign compute pid(s) %s "
                   "(§5.4, PP-19)" % (isolation.get("foreign_pids"),))
    return out


def _band_status(raw, band, baseline, ratios, ctx):
    """§7.4, in the order the vocabulary is defined."""
    c = band.get("concurrency")
    witness = band.get("witness")
    if isinstance(c, int) and c > 1:
        ok = isinstance(witness, dict) and witness.get("batch_invariance") == "PASS"
        if not ok:
            return "INVALID-CORRECTNESS", [
                "batch-invariance witness is %s at c=%d -- a batch that is not "
                "invariant across its own slots, or froze on one token, is "
                "measuring garbage at full speed (PP-26)"
                % ("absent" if not isinstance(witness, dict)
                   else repr(witness.get("batch_invariance")), c)]
    if ctx["comparator_stale"]:
        return "COMPARATOR_STALE", [
            "the comparator pin expired on %s, before this run started at %s "
            "(PP-20)" % (ctx["pin_expiry"], ctx["started_utc"])]
    reasons = _nonconformance(raw, band, ctx)
    if reasons:
        return "NONCONFORMANT-VALID", reasons
    if baseline is None or ratios is None:
        return "UNMEASURED", ["no comparator lane was joined to this band"]
    return "MEASURED", []


# The throughput keys an INVALID-CORRECTNESS band must NOT carry: a band whose
# tokens are wrong has no throughput to report.
INVALID_STRIPS = ("aggregate_tok_per_sec", "decode_tok_per_sec",
                  "prefill_tok_per_sec", "prefill_source")


def _ratios_for(raw, ctx):
    """agg and prefill by the replicate estimator, dec by the bootstrap."""
    subj, comp = raw.get("subject") or {}, raw.get("comparator") or {}
    n_min = ctx["replicates_min"]
    out = {
        "agg": replicate_t_lower(subj.get("aggregate_tok_per_sec"),
                                 comp.get("aggregate_tok_per_sec"), n_min),
        "dec": paired_percentile_bootstrap(subj.get("request_decode_tok_per_sec"),
                                           comp.get("request_decode_tok_per_sec")),
        "prefill": replicate_t_lower(subj.get("prefill_tok_per_sec"),
                                     comp.get("prefill_tok_per_sec"), n_min),
    }
    return out if out["agg"] is not None else None


def _v3_band(raw, ctx, unproduced):
    """One band of the v3 wire, with its baseline, its ratios and its status."""
    c = raw.get("concurrency")
    subj, comp = raw.get("subject") or {}, raw.get("comparator") or {}
    band = _lane_band(subj, c, ctx)
    baseline = _lane_band(comp, c, ctx)
    baseline["run_id"] = ctx["run_id"]
    ratios = _ratios_for(raw, ctx)
    band.update({
        "replicates": raw.get("replicates"),
        "interleaved": raw.get("interleaved"),
        "stream_witness": _stream_witness(subj, ctx),
        "join_key": _join_key(raw, c, ctx),
        "comparator_admission": raw.get("comparator_admission"),
        "isolation": raw.get("isolation"),
        "gpu_layers": raw.get("gpu_layers"),
        "baseline": baseline,
        "ratios": ratios,
        "samples_file": None,
        "roofline_tok_per_sec": None,
        "suspect": [],
        "errors": 0,
        "truncated": 0,
    })
    status, reasons = _band_status(raw, band, baseline, ratios, ctx)
    band["status"] = status
    band["status_reasons"] = reasons
    band["comparator_status"] = "MEASURED" if (baseline and ratios) else "UNMEASURED"
    if status == "INVALID-CORRECTNESS":
        for key in INVALID_STRIPS:
            band.pop(key, None)
            unproduced.append("bands[c=%s].%s: withheld -- the band is "
                              "INVALID-CORRECTNESS" % (c, key))
        band["ratios"] = None
        band["baseline"] = None
        band["comparator_status"] = "UNMEASURED"
    elif status != "MEASURED":
        for reason in reasons:
            unproduced.append("bands[c=%s].status: %s -- %s" % (c, status, reason))
    if (raw.get("isolation") or {}).get("contended") is None:
        unproduced.append("bands[c=%s].isolation: the device probe was absent, "
                          "so nobody asked who else was on the device (§5.4)" % (c,))
    return band


def _scaling(bands):
    """scaling_efficiency and overhead_share -- REPORTED, never gated."""
    base = next((b.get("aggregate_tok_per_sec") for b in bands
                 if b.get("concurrency") == 1), None)
    base_dec = next((b.get("decode_tok_per_sec") for b in bands
                     if b.get("concurrency") == 1), None)
    for band in bands:
        c = band.get("concurrency")
        agg = band.get("aggregate_tok_per_sec")
        band["scaling_efficiency"] = (
            (agg / base) / c if base and agg and isinstance(c, int) and c > 1 else None)
        band["overhead_share"] = (
            agg / base_dec if c == 1 and agg and base_dec else None)


def _executor_context(block, lane, args):
    """Everything a band needs that is stated once for the whole invocation."""
    protocol = _matrix_protocol()
    prov = (lane.get("subject") or {}).get("provenance") or {}
    started = None
    for raw in lane.get("bands") or []:
        started = (raw.get("subject_effective_config") or {}).get("started_utc")
        if started:
            break
    expiry = block.get("comparator_pin_expiry")
    # PP-20: a pin that expired BEFORE the run started makes every ratio in it
    # COMPARATOR_STALE. Both are ISO-8601, so the date compares as text.
    stale = bool(expiry and started and expiry < started[:len(expiry)])
    return {
        "run_id": block.get("run_id"),
        "host": args.host, "workload": args.workload,
        "model": prov.get("model") or args.model,
        "quantization": prov.get("quantization") or args.quantization,
        "tokenization": None,
        "window_ms": protocol["window_ms"],
        "n_predict": protocol["n_predict"],
        "replicates_min": protocol["replicates_min"],
        "live_ttft_over_e2e_max": bench_receipt.matrix_number(
            "stream", "live_ttft_over_e2e_max"),
        "pin_expiry": expiry, "started_utc": started,
        "comparator_stale": stale,
        "protocol": protocol,
    }


def bands_from_parity_v3(block, lane, args, errors):
    """The v3 bands of one executor lane, plus what could not be produced."""
    ctx = _executor_context(block, lane, args)
    if not ctx["run_id"]:
        errors.append("the parity block declares layout=executor and carries no "
                      "run_id. Without one the two lanes of a band cannot be "
                      "shown to come from the same invocation, and a ratio "
                      "across two invocations is a quotient of two afternoons "
                      "(PP-3).")
        return None, None, None
    unproduced = []
    bands = [_v3_band(raw, ctx, unproduced) for raw in (lane.get("bands") or [])]
    if not bands:
        errors.append("lane %r carries no bands" % lane.get("lane"))
        return None, None, None
    _scaling(bands)
    return bands, ctx, unproduced


# --------------------------------------------------------------- assembling --
def _provenance_from_args(args, errors):
    prov = {"binary_path": args.binary, "binary_sha256": args.binary_sha256,
            "resolution": args.resolution, "compute_class": args.compute_class}
    for key, value in list(prov.items()):
        if not value:
            errors.append("provenance.%s is required and was not given. It is "
                          "not derivable from a benchmark artifact: the harness "
                          "never sees the binary it drove." % key)
    for key, value in (("host", args.host), ("accelerator", args.accelerator),
                       ("model", args.model), ("quantization", args.quantization)):
        if value:
            prov[key] = value
    return prov


def _print_table(bands, requested, completed):
    base = next((b["aggregate_tok_per_sec"] for b in bands
                 if b["concurrency"] == 1), None)
    print("  c   agg_tok_s  scaling_eff   agg_ratio  decode_ratio   cmp completed")
    for b in bands:
        c = b["concurrency"]
        agg = b.get("aggregate_tok_per_sec")
        eff = (agg / base) / c if base and agg else float("nan")
        # v2 spells the comparator's counts on the band; v3 puts the whole
        # comparator band under `baseline`. Read whichever is there.
        baseline = b.get("baseline") or {}
        cr = b.get("comparator_requested", baseline.get("requested"))
        cc = b.get("comparator_completed", baseline.get("completed"))
        comp = "%s/%s" % (cc, cr) if cr is not None else "-"
        # A v3 band carries NO bare scalar ratio (PP-3): the ratio lives in
        # `ratios` beside the baseline it was taken against. The table reads
        # whichever spelling the receipt actually has rather than requiring the
        # one the rules forbid.
        ratios = b.get("ratios") or {}
        agg_r = b.get("agg_ratio", (ratios.get("agg") or {}).get("point"))
        dec_r = b.get("decode_ratio", (ratios.get("dec") or {}).get("point"))
        print("%3d  %10s  %11.4f  %10s  %12s  %14s"
              % (c, "-" if agg is None else "%.2f" % agg, eff,
                 "-" if agg_r is None else "%.4f" % agg_r,
                 "-" if dec_r is None else "%.4f" % dec_r, comp))
    if requested is not None:
        print("  subject completed/requested: %d/%d" % (completed, requested))


def _ladder_v3(lane, unproduced):
    """PP-24: what was REQUESTED, and what the servers admitted.

    `slots_admitted.llama` is null BY CONSTRUCTION here and says so: the
    executor relaunches the comparator per band with `-np c`, so its admission
    equals the request at every band and imposes no ceiling. Reporting `min` of
    those per-band values would derive a one-rung ladder out of a fact about the
    launcher. The per-band figure is carried on each band's
    comparator_admission.
    """
    declared = list(bands_default())
    apr = None
    for raw in lane.get("bands") or []:
        apr = (raw.get("subject_effective_config") or {}).get("slots_admitted")
        if apr is not None:
            break
    unproduced.append("ladder.slots_admitted.llama: the comparator is "
                      "relaunched per band with -np c, so it admits exactly "
                      "what was asked and states no ceiling (PP-24)")
    derived = [c for c in declared if apr is None or c <= apr]
    return {"declared": declared, "derived": derived,
            "slots_admitted": {"apr": apr, "llama": None}}


def _reconcile_join_key(prov):
    """Drop a `_join_key_incomplete` note the receipt has since satisfied.

    bench_receipt._check_join_key WRITES that list into the provenance object it
    is handed, and parity_block.py self-validates its block before emitting it.
    A parity SIDE carries no host/accelerator/model/quantization -- those come
    from the receipt's own flags -- so the note travels into every receipt built
    from a block and names four fields that are present two lines above it. A
    note that lists a field the document supplies is a rotting claim, and this
    file exists to remove that class rather than to add one.
    """
    stale = [k for k in prov.get("_join_key_incomplete") or [] if not prov.get(k)]
    if stale:
        prov["_join_key_incomplete"] = stale
    else:
        prov.pop("_join_key_incomplete", None)
    return prov


def _every(bands, key):
    """The sum of a per-band counter, or None when any band omits it."""
    total = 0
    for band in bands:
        if band.get(key) is None:
            return None
        total += band[key]
    return total


def _fill_from_parity_v3(args, receipt, block, errors):
    """A v3 receipt from an executor-layout parity block."""
    lane = _lane_of(block, args.lane, errors)
    if lane is None:
        return None
    if lane.get("comparability") == "cross-class-existence-only":
        errors.append("lane %r is cross-class-existence-only; a cross-class row "
                      "cannot arm a threshold and must not be converted into a "
                      "receipt that Arm A would gate happily." % lane.get("lane"))
        return None
    bands, ctx, unproduced = bands_from_parity_v3(block, lane, args, errors)
    if bands is None:
        return None

    subject = lane.get("subject") or {}
    comparator = lane.get("comparator") or {}
    prov = dict(subject.get("provenance") or {})
    for key, value in (("host", args.host), ("accelerator", args.accelerator),
                       ("model", args.model), ("quantization", args.quantization)):
        if value and not prov.get(key):
            prov[key] = value
    first = (lane.get("bands") or [{}])[0]
    config = (first.get("subject_effective_config") or {})
    body = config.get("body") or {}
    _reconcile_join_key(prov)
    prov["started_utc"] = ctx["started_utc"]
    prov["clock_source"] = ((body.get("server") or {}).get("clock_source"))
    prov["subject"] = {"path": prov.get("binary_path"),
                       "sha256": prov.get("binary_sha256"),
                       "commit": (body.get("server") or {}).get("build_commit"),
                       "feature_set": prov.get("feature_set")}
    # PP-25: ONE client drove both lanes, and here that is a structural fact
    # rather than a claim -- the executor runs `$APR test llm bench` against
    # both ports inside one band. Its commit is not recorded anywhere in the
    # artifact, so it is null and named below.
    prov["client"] = {"path": prov.get("binary_path"),
                      "sha256": prov.get("binary_sha256"), "commit": None}
    prov["comparator"] = {
        "commit": comparator.get("build_commit"),
        "cmake": None,
        "sha256": (comparator.get("provenance") or {}).get("binary_sha256"),
        "pin_expiry": ctx["pin_expiry"],
        "props": (first.get("comparator_admission") or {}).get("props")}
    prov["server_config"] = body or None
    prov["model_file"] = None
    for name, why in (
            ("provenance.client.commit", "the executor records the client "
             "binary's digest but not the commit it was built from (PP-25)"),
            ("provenance.comparator.cmake", "the CMakeCache witness is checked "
             "by scripts/llama_bin.sh and not carried into the block"),
            ("provenance.model_file", "no producer hashes the GGUF the servers "
             "loaded, so PP-23's roofline has no input"),
            ("bands[].samples[].issued_ms", "the harness records a per-request "
             "DURATION, not a position on the window clock (PP-10)"),
            ("bands[].roofline_tok_per_sec", "no bandwidth file is committed "
             "for this host (PP-23)")):
        unproduced.append("%s: %s" % (name, why))
    if not prov.get("server_config"):
        unproduced.append("provenance.server_config: GET /v1/effective-config "
                          "returned %r, so the subject never stated its own "
                          "resolved configuration (PP-2)" % (config.get("state"),))

    samples, requested, completed = [], 0, 0
    for raw in lane.get("bands") or []:
        side = raw.get("subject") or {}
        samples.extend(side.get("samples_ms") or [])
        requested += side.get("requested") or 0
        completed += side.get("completed") or 0

    receipt.update({
        "spec": SPEC_VERSION,
        "schema_version": WIRE_SCHEMA_VERSION_V3,
        "run_id": ctx["run_id"],
        "client_model": "closed_loop",
        "protocol": {
            "window_ms": ctx["protocol"]["window_ms"],
            "warmup_requests_per_worker": ctx["protocol"]["warmup_requests_per_worker"],
            "quiesce_ms": ctx["protocol"]["quiesce_ms"],
            "cooldown_ms": ctx["protocol"]["cooldown_ms"],
            "n_predict": ctx["protocol"]["n_predict"],
            # OBSERVED, not declared: these two are what the run did, and §7.4
            # reads them to decide whether a band may arm anything.
            "replicates": min(b.get("replicates") or 0 for b in bands),
            "interleaved": all(b.get("interleaved") is True for b in bands),
            "sampler": dict(ctx["protocol"]["sampler"]),
        },
        "ladder": _ladder_v3(lane, unproduced),
        "bands": bands,
        "provenance": prov,
        "samples_ms": samples,
        "requested": requested,
        "completed": completed,
        "short_of_n_predict": _every(bands, "short_of_n_predict"),
        "timeouts": _every(bands, "timeouts"),
        "unproduced_fields": unproduced,
    })
    return receipt


def _fill_from_parity(args, receipt, errors):
    with open(args.from_parity, encoding="utf-8") as handle:
        doc = json.load(handle)
    block = doc.get("parity") if isinstance(doc.get("parity"), dict) else doc
    # THE LAYOUT IS READ OFF THE BLOCK, never selected by a flag: the producer
    # states which of its two shapes it wrote, and a flag would let a caller
    # declare the shape its file is not in.
    if isinstance(block, dict) and block.get("layout") == "executor":
        return _fill_from_parity_v3(args, receipt, block, errors)
    bands, subject, _lane = bands_from_parity(args.from_parity, args.lane, errors)
    if bands is None:
        return None
    prov = dict(subject.get("provenance") or {})
    if args.host and not prov.get("host"):
        prov["host"] = args.host
    _reconcile_join_key(prov)
    # §4.4.5's raw per-request latencies. A parity block written before
    # PERF-004 does not carry them -- P2 read request_details and kept only
    # per-RUN summaries -- and there is nothing on a parity side that can stand
    # in for them. ttft_p50_ms is per-run and is a different quantity;
    # substituting it would put a plausible list of numbers under a label that
    # does not describe them.
    if subject.get("samples_ms") is None:
        errors.append("the subject side carries no samples_ms. A parity block "
                      "written before PERF-004 kept only per-run summaries, and "
                      "§4.4.5's raw per-request latencies cannot be recovered "
                      "from it. Re-emit the block with parity_block.py, or use "
                      "--from-bands.")
        return None
    receipt.update({"bands": bands, "provenance": prov,
                    "samples_ms": list(subject["samples_ms"])})
    if subject.get("requested") is not None:
        receipt["requested"] = subject["requested"]
        receipt["completed"] = subject.get("completed")
    return receipt


def _fill_from_bands(args, receipt, errors):
    bands, samples, requested, completed = bands_from_dir(
        args.from_bands, args.subject, args.comparator, errors)
    if errors:
        return None
    receipt.update({"bands": bands, "requested": requested,
                    "completed": completed, "samples_ms": samples})
    if not args.derive_only:
        receipt["provenance"] = _provenance_from_args(args, errors)
    return receipt


def build(args, errors):
    receipt = {
        "schema": SCHEMA,
        "spec_version": SPEC_VERSION,
        "schema_version": WIRE_SCHEMA_VERSION,
        "host": args.host,
        "workload": args.workload,
        # PP-24: the ladder this receipt was REQUESTED at, read from the matrix
        # rather than restated. `derived` stays null until a producer can report
        # what each server admitted -- an unmeasured derivation is null, never a
        # copy of the declared list wearing the derived name.
        "ladder": {"declared": list(bands_default()), "derived": None,
                   "slots_admitted": {"apr": None, "llama": None}},
        # WHAT THIS RECEIPT CANNOT SAY, said out loud. Every entry here is a
        # field perf_gate.sh requires that no instrument in this tree produces.
        # It is copied from scripts/perf-receipt-fields.yaml, never restated.
        "unmeasured": unmeasured_from_ledger(),
    }
    fill = _fill_from_parity if args.from_parity else _fill_from_bands
    if fill(args, receipt, errors) is None:
        return None
    if not receipt.get("samples_ms"):
        errors.append("samples_ms came out empty. §4.4.5 keeps raw per-request "
                      "samples so a threshold can be re-derived later; a receipt "
                      "without them forecloses that permanently.")
    return receipt


# ----------------------------------------------------------------- selftest --
# END TO END, not unit: the claim being proved is that a BenchmarkReport this
# repo can actually produce reaches a verdict from scripts/perf_gate.sh, and
# that what remains failing is the UNMEASURED bucket by name rather than a
# schema error. A converter tested only against its own output would prove the
# same thing the gate's synthetic fixture proved -- nothing.
def _fake_run(concurrency, agg, decode, n_req, ok, tokens, latency0):
    return {
        "concurrency": concurrency, "tokens_per_sec": agg,
        "decode_tok_per_sec": decode, "total_requests": n_req,
        "successful": ok, "failed": n_req - ok,
        "completion_tokens_total": tokens,
        # Read by parity_block._side for the lane-level samples. Present on
        # every real BenchmarkReport; absent here until the P2 chain rows
        # below started exercising that reader.
        "prefill_tok_per_sec": agg * 0.9, "ttft_p50_ms": 30.0 + latency0 * 0.1,
        # A real timing distribution is not a constant; bench_receipt.py rejects
        # a flat one as the fabricated-measurement shape, and it is right to.
        "request_details": [{"latency_ms": latency0 + i * 1.7, "ttft_ms": 12.0,
                             "completion_tokens": 128, "prompt_tokens": 100,
                             "itl_ms": 9.0} for i in range(5)],
    }


def _write_report(path, runs):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"runs": runs, "aggregate": {}, "regressions": []}, handle)


def _fixture(work, agg_c4=360.0):
    for c, agg, dec in ((1, 100.0, 110.0), (4, agg_c4, 112.0)):
        _write_report(_band_paths(work, "apr", c),
                      [_fake_run(c, agg, dec, 20, 20, 2560, 100.0),
                       _fake_run(c, agg * 1.01, dec, 20, 20, 2560, 101.0)])
        _write_report(_band_paths(work, "llamacpp", c),
                      [_fake_run(c, agg / 0.9, dec / 1.1, 20, 20, 2560, 90.0),
                       _fake_run(c, agg / 0.9, dec / 1.1, 20, 20, 2560, 91.0)])


PROV_ARGS = ["--binary", "/opt/pinned", "--binary-sha256", "0" * 64,
             "--compute-class", "cuda", "--resolution", "scripts/apr_bin.sh"]


def _convert(work, out, extra=()):
    return main(["--from-bands", work, "--host", "lambda", "--workload", "W1",
                 "--out", out] + list(extra))


def _edit_runs(work, mutate):
    """Apply `mutate` to each band artifact's runs, in place."""
    for c in (1, 4):
        for side in ("apr", "llamacpp"):
            path = _band_paths(work, side, c)
            with open(path, encoding="utf-8") as handle:
                doc = json.load(handle)
            mutate(doc["runs"])
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(doc, handle)


def _mutant(work, name, mutate):
    """A fresh fixture with one thing wrong, converted; returns the rc."""
    where = os.path.join(work, name)
    os.makedirs(where, exist_ok=True)
    _fixture(where)
    _edit_runs(where, mutate)
    return _convert(where, os.devnull, PROV_ARGS)


def _selftest_verdict_rows(work, gate):
    """Convert a fixture, run the real gate on it, and read the verdict."""
    out = os.path.join(work, "receipt.json")
    rc = _convert(work, out, PROV_ARGS + [
        "--accelerator", "rtx-4090", "--model", "qwen2.5-coder-1.5b-instruct",
        "--quantization", "Q4_K_M"])
    import subprocess
    proc = subprocess.run(["bash", gate, "--host", "lambda", "--phase", "merge",
                           "--workload", "W1", "--receipt", out],
                          capture_output=True, text=True, check=False)
    text = proc.stdout + proc.stderr
    return [
        ("converts a real report shape", rc == 0),
        # Arms A and B RUN. Before this converter they could not: the gate saw
        # no bands at all and said so twice.
        ("ArmA scores the converted receipt", "ArmA c=4" in text),
        # Arm B1/B2 became Arm L3 (§7.5): the floors were replaced by one
        # non-inferiority arm with an asymmetric gated set. A v2 receipt's bare
        # scalar ratios are READ there and reported as historical, never gated.
        ("ArmL3 scores the converted receipt", "ArmL3 c=4" in text),
        # The delegate ACCEPTS it -- provenance and raw samples are present, so
        # the residual failure is not a schema rejection any more.
        ("no schema rejection remains", "ArmC schema" not in text),
        # ... and what does remain is named, with an owner.
        ("drain_ms failure names the gap",
         "drain_ms absent -- The drain rule is not implemented" in text),
        ("drain_ms failure names its owner", "owner PERF-004)" in text),
        ("verdict is still FAIL", proc.returncode == 1),
    ]


def _drop_samples(runs):
    for run in runs:
        run["request_details"] = []


def _mix_concurrency(runs):
    runs[-1]["concurrency"] = runs[-1]["concurrency"] + 4


def _selftest_refusal_rows(work):
    return [
        # §4.4.5's raw samples are what a later threshold would be re-derived
        # from; a receipt without them forecloses that permanently.
        ("refuses a receipt with no raw samples",
         _mutant(work, "stripped", _drop_samples) == 1),
        # A band file whose runs disagree on concurrency is a mixture, and
        # averaging across it reports the mixture as one band.
        ("refuses a band that mixes concurrencies",
         _mutant(work, "mixed", _mix_concurrency) == 1),
        # The harness never sees the binary it drove, so provenance is not
        # derivable and must not be defaulted. This is exactly why the
        # 2026-08-25 corpus cannot produce a conformant receipt.
        ("refuses a receipt with no provenance",
         _convert(work, os.devnull) == 1),
        # `--resolution` USED TO DEFAULT to "scripts/apr_bin.sh". That string
        # is the provenance claim that matters -- never PATH, never a hardcoded
        # absolute path -- so defaulting it wrote an assertion nothing checked
        # into every receipt whose caller forgot the flag. It is now required,
        # and this row is the proof that omitting it REFUSES rather than
        # asserting on the caller's behalf.
        ("refuses a receipt whose resolution is not stated",
         _convert(work, os.devnull, [a for a in PROV_ARGS
                                     if a not in ("--resolution",
                                                  "scripts/apr_bin.sh")]) == 1),
        # DISCRIMINATION -- the same tree, unmutated, still converts. Without
        # this row every refusal above could come from a converter that refuses
        # everything.
        ("still converts an unmutated tree",
         _convert(work, os.devnull, PROV_ARGS) == 0),
    ]


# ------------------------------------------ P2: the HISTORICAL layout, end to end
# a pre-restructure $WORK -> lib/parity_block.py -> --from-parity ->
# scripts/perf_gate.sh.
#
# THIS BRANCH HAD NEVER RUN. `--from-parity` was reachable from no test, and
# the join underneath it was broken in a way no unit test could see:
# parity_block.py read `$WORK/apr-<lane>.json` and `$WORK/llama-<lane>.json`,
# and parity_host_receipt.sh -- its only producer, which invokes it directly --
# wrote neither. So every complete host run ended at
#
#     FAIL  lane cpu is missing a side; refusing to report half a comparison
#
# which reads as "the benchmark did not run". The chain had never produced a
# parity block, and a receipt-schema epic had no test that ran it.
#
# WHAT THIS ROW SET COVERS NOW. The executor has since been restructured and no
# longer writes this layout at all -- `_selftest_p1_chain_rows` covers what it
# does write. These rows keep the HISTORICAL reading alive, because the
# committed 2026-08-25 corpus is in it and still has to derive its eight JOIN
# digits. A three-field lanes.txt line is that layout's spelling.
PARITY_LANE = "cpu"
# A SYNTHETIC RATIO USED TO BUILD A FIXTURE, never compared against anything.
# It sits above Arm L3's non-inferiority bound and below the sanity ceiling, so
# the lane verdict the fixture produces is PASS; scripts/check_thresholds_in_matrix.sh
# allowlists it by name for exactly that reason.
FIXTURE_RATIO = 1.05


def _parity_work(root, bands=None):
    """A $WORK in the HISTORICAL layout: one report per lane per band."""
    bands = bands_default() if bands is None else bands
    work = os.path.join(root, "p2")
    if os.path.isdir(work):
        import shutil
        shutil.rmtree(work)
    os.makedirs(work)
    for c in bands:
        agg, dec = 100.0 + c, 110.0 + c
        _write_report(os.path.join(work, "apr-%s-c%d.json" % (PARITY_LANE, c)),
                      [_fake_run(c, agg, dec, 20, 20, 2560, 100.0),
                       _fake_run(c, agg * 1.01, dec * 1.01, 20, 20, 2560, 103.0)])
        _write_report(os.path.join(work, "llama-%s-c%d.json" % (PARITY_LANE, c)),
                      [_fake_run(c, agg / FIXTURE_RATIO, dec / FIXTURE_RATIO,
                                 20, 20, 2560, 90.0),
                       _fake_run(c, agg * 1.01 / FIXTURE_RATIO,
                                 dec * 1.01 / FIXTURE_RATIO, 20, 20, 2560, 93.0)])
    with open(os.path.join(work, "lanes.txt"), "w", encoding="utf-8") as fh:
        # THE HISTORICAL SPELLING: <name> <subject class> <comparator class>.
        # The restructured executor writes two fields instead, and a band
        # metadata file beside them; parity_block.py reads both and refuses a
        # two-field line with no metadata rather than guessing which it is.
        fh.write("%s cpu cpu\n" % PARITY_LANE)
    return work


def _run_parity_block(work, out, extra=()):
    """parity_block.py with the argument set parity_host_receipt.sh passes."""
    import subprocess
    return subprocess.run(
        [sys.executable, os.path.join(SCRIPTS, "lib", "parity_block.py"),
         "--work", work, "--apr", "/opt/apr", "--apr-sha", "1" * 64,
         "--llama", "/opt/llama-server", "--llama-sha", "2" * 64,
         "--llama-build", "39173bcac", "--model", "/models/q.gguf",
         "--install-source", "crates.io", "--out", out] + list(extra),
        capture_output=True, text=True, check=False)


def _selftest_parity_chain_rows(root, gate):
    import subprocess
    work = _parity_work(root)
    block = os.path.join(work, "parity.json")
    emitted = _run_parity_block(work, block)

    rows = [("P2 producer layout emits a parity block", emitted.returncode == 0)]
    if emitted.returncode != 0:
        # Say WHY here rather than leaving six rows failing with no reason.
        rows.append(("P2 parity_block stderr: " + emitted.stderr.strip()
                     .replace("\n", " ")[:160], False))
        return rows

    receipt = os.path.join(work, "receipt.json")
    rc = main(["--from-parity", block, "--lane", PARITY_LANE,
               "--host", "lambda", "--workload", "W1", "--out", receipt])
    rows.append(("--from-parity converts that block", rc == 0))
    if rc != 0:
        return rows

    proc = subprocess.run(["bash", gate, "--host", "lambda", "--phase", "merge",
                           "--workload", "W1", "--receipt", receipt],
                          capture_output=True, text=True, check=False)
    text = proc.stdout + proc.stderr
    rows += [
        ("P2 chain reaches ArmA", "ArmA c=4" in text),
        ("P2 chain reaches ArmL3", "ArmL3 c=4" in text),
        ("P2 chain hits no schema rejection", "ArmC schema" not in text),
        ("P2 chain still FAILs on the unmeasured bucket",
         proc.returncode == 1 and "drain_ms absent" in text),
    ]

    # RED, and NO FALLBACK. The lane-level side is the c=1 band; with c=1 gone
    # the block must refuse by name rather than report the lane from c=4 under
    # the same label.
    starved = _parity_work(os.path.join(root, "starved"), bands=(4, 8, 16))
    missing = _run_parity_block(starved, os.path.join(starved, "parity.json"))
    rows.append(("refuses a lane whose c=1 band is absent",
                 missing.returncode == 1 and "c=1 band" in missing.stderr))
    return rows


# ------------------------------- P1: the RESTRUCTURED executor, end to end ---
# scripts/parity_host_receipt.sh (restructured) -> lib/parity_block.py ->
# --from-parity -> a v3 receipt.
#
# THE BREAK THESE ROWS CLOSE. The executor was restructured to relaunch the
# comparator per band, read /props and /v1/effective-config per band, and run
# the replicates INTERLEAVED, writing a band metadata file, N per-replicate
# reports per lane, the two isolation records and a two-field lanes.txt. Its
# consumers still read one report per lane per band and unpacked THREE names
# out of every lanes.txt line, so the P1 -> P2 -> P3 chain could not complete:
# the block died on a missing side, and before that on a ValueError.
#
# The fixture under tests/fixtures/perf-gate/parity-host-work is a synthetic
# $WORK in the executor's layout -- see its README for the snippet that wrote
# it. It is deliberately CONFORMANT, so the three mutations below are the only
# thing separating MEASURED from every other §7.4 verdict.
P1_FIXTURE_DIR = os.path.join(ROOT, "tests", "fixtures", "perf-gate",
                              "parity-host-work")
P1_LANE = "cpu"
P1_PIN_EXPIRY = "2026-12-01"


def _p1_work(root, name, mutate=None):
    """A private copy of the committed $WORK fixture, optionally mutated."""
    import shutil
    dest = os.path.join(root, name)
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(P1_FIXTURE_DIR, dest)
    if mutate is not None:
        mutate(dest)
    return dest


def _p1_edit(work, name, mutate):
    path = os.path.join(work, name)
    with open(path, encoding="utf-8") as handle:
        doc = json.load(handle)
    mutate(doc)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle)


def _p1_set(key, value):
    """A mutation that sets one key, for `_p1_edit`."""
    def mutate(doc):
        doc[key] = value
    return mutate


def _p1_receipt(work, extra=()):
    """The whole chain: P2 over $WORK, then P3 over the block it wrote."""
    block = os.path.join(work, "parity.json")
    emitted = _run_parity_block(work, block, ["--pin-expiry", P1_PIN_EXPIRY] + list(extra))
    if emitted.returncode != 0:
        return None, emitted.stderr.strip().replace("\n", " ")[:200]
    out = os.path.join(work, "receipt.json")
    rc = main(["--from-parity", block, "--lane", P1_LANE, "--host", "lambda",
               "--workload", "W1", "--accelerator", "cpu",
               "--model", "qwen2.5-coder-7b-instruct",
               "--quantization", "q4_k_m", "--out", out])
    if rc != 0:
        return None, "--from-parity refused the executor block"
    with open(out, encoding="utf-8") as handle:
        return json.load(handle), ""


def _p1_statuses(receipt):
    return [b.get("status") for b in receipt.get("bands") or []]


def _p1_strip_witness(work):
    """The real harness's shape: no per-replicate report carries a witness."""
    for name in os.listdir(work):
        if not name.endswith(".json") or name.startswith("band-"):
            continue
        path = os.path.join(work, name)
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        if isinstance(doc, dict) and isinstance(doc.get("runs"), list):
            for run in doc["runs"]:
                run.pop("witness", None)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(doc, handle)


def _p1_witness_file(work, bands):
    """A scripts/perf041_batched_parity_probe.py witness with the given bands."""
    path = os.path.join(work, "perf041-witness.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"witness_version": 2, "probe": "perf041", "commit": "0" * 40,
                   "bands": [{"c": c, "m_formed": c, "result": "PASS",
                              "divergence_at": 3, "intra_agree_to": 128,
                              "max_constant_run": 1, "declared_min": 128,
                              "reason": None} for c in bands]}, handle)
    return path


def _selftest_p1_witness_rows(root):
    """PP-26 reaches an executor block only through --witness-json (the bench
    reports carry none). Both polarities: attached for the band -> the band is
    not INVALID-CORRECTNESS; the band missing from the witness -> it is."""
    rows = []
    work = _p1_work(root, "p1-witness-attached", _p1_strip_witness)
    wit = _p1_witness_file(work, [1, 4])
    receipt, why = _p1_receipt(work, ["--witness-json", wit])
    ok = receipt is not None and "INVALID-CORRECTNESS" not in _p1_statuses(receipt) \
        and all(isinstance(b.get("witness"), dict) and b["witness"].get("batch_invariance") == "PASS"
                for b in receipt.get("bands") or [] if (b.get("concurrency") or 0) > 1)
    rows.append(("witness_attached_from_perf041" + ("" if ok else ": " + (why or str(_p1_statuses(receipt)))), ok))
    work = _p1_work(root, "p1-witness-absent-band", _p1_strip_witness)
    wit = _p1_witness_file(work, [1])
    receipt, why = _p1_receipt(work, ["--witness-json", wit])
    statuses = _p1_statuses(receipt) if receipt else []
    ok = receipt is not None and "INVALID-CORRECTNESS" in statuses
    rows.append(("witness_absent_band_is_invalid_correctness" + ("" if ok else ": " + (why or str(statuses))), ok))
    return rows


def _selftest_p1_chain_rows(root):
    rows = []

    # ---- must-not-fire: the tree as it stands reaches MEASURED --------------
    receipt, why = _p1_receipt(_p1_work(root, "p1-ok"))
    if receipt is None:
        rows.append(("p1_chain_reads_the_restructured_executor: " + why, False))
        return rows
    bands = receipt["bands"]
    lcb = [(b.get("ratios") or {}).get(m, {}).get("lcb95")
           for b in bands for m in ("agg", "prefill")]
    ns = [(b.get("ratios") or {}).get("agg", {}).get("n") for b in bands]
    rows.append(("p1_chain_reads_the_restructured_executor",
                 receipt.get("schema_version") == WIRE_SCHEMA_VERSION_V3
                 and len(bands) == 2
                 and _p1_statuses(receipt) == ["MEASURED", "MEASURED"]
                 and all(v is not None for v in lcb)
                 and ns == [5, 5]))
    # A LOWER BOUND EQUAL TO ITS POINT ESTIMATE is not a bound; it is what a
    # zero-variance sample produces, and it would pass the row above.
    rows.append(("p1_chain_lcb95_sits_below_the_point_estimate",
                 all(b["ratios"][m]["lcb95"] < b["ratios"][m]["point"]
                     for b in bands for m in ("agg", "dec", "prefill"))))

    # ---- must-fire: a sweep is not a paired measurement (§4.3) --------------
    def _sweep(work):
        for name in ("band-cpu-c1.json", "band-cpu-c4.json"):
            _p1_edit(work, name, _p1_set("interleaved", False))

    receipt, why = _p1_receipt(_p1_work(root, "p1-sweep", _sweep))
    rows.append(("p1_chain_non_interleaved_is_nonconformant",
                 receipt is not None
                 and set(_p1_statuses(receipt)) == {"NONCONFORMANT-VALID"}
                 and any("interleaved" in r for b in receipt["bands"]
                         for r in b["status_reasons"])))

    # ---- must-fire: somebody else was on the device (§5.4, PP-19) ----------
    def _contend(work):
        _p1_edit(work, "iso-cpu-c1-before.json", lambda d: d.update({
            "compute_pids": [{"pid": 4242, "used_memory_mib": 5120},
                             {"pid": 9999, "used_memory_mib": 8192}],
            "foreign_pids": [9999]}))

    receipt, why = _p1_receipt(_p1_work(root, "p1-contended", _contend))
    named = ""
    if receipt is not None:
        named = " ".join([r for b in receipt["bands"] for r in b["status_reasons"]]
                         + receipt.get("unproduced_fields", []))
    rows.append(("p1_chain_contended_band_is_named",
                 receipt is not None
                 and _p1_statuses(receipt)[0] == "NONCONFORMANT-VALID"
                 and "9999" in named))

    # ---- must-fire: three replicates bound nothing (§4.3) ------------------
    def _short(work):
        for name in ("band-cpu-c1.json", "band-cpu-c4.json"):
            _p1_edit(work, name, _p1_set("replicates", 3))

    receipt, why = _p1_receipt(_p1_work(root, "p1-short", _short))
    rows.append(("p1_chain_short_replicates_have_no_bound",
                 receipt is not None
                 and "MEASURED" not in _p1_statuses(receipt)
                 and all(b["ratios"]["agg"]["lcb95"] is None
                         and b["ratios"]["agg"]["n"] == 3
                         for b in receipt["bands"])))
    return rows


# ------------------------------------------------------- the JOIN fixture ---
# §12 row 7 / §10's registered prediction: the derivation over the committed
# 2026-08-25 band artifacts reproduces eight ratios to four decimals with ZERO
# GPU. That is the only part of the parity chain this repository can prove on a
# laptop, and it pins the ARITHMETIC -- median over run-level values, subject
# over comparator, per band -- against a corpus nobody can quietly re-measure.
#
# WHAT IT DOES NOT LICENSE. The corpus is NON-CONFORMANT on five keys the master
# makes fatal (n=2 replicates, unequal windows between the lanes, not
# interleaved, comparator completion_tokens 112 against n_predict 128, no
# provenance at all), so these eight numbers are FIXTURE VALUES and never a
# parity claim. The statistic is the HISTORICAL run-level median, not the
# per-request estimator the v3 spec defines.
JOIN_FIXTURE_DIR = os.path.join(ROOT, "evidence", "parity-http", "bands")
JOIN_FIXTURE_BANDS = (1, 4, 8, 16)
JOIN_FIXTURE_AGG = ("0.5341", "0.2308", "0.1685", "0.0967")
JOIN_FIXTURE_DEC = ("0.5873", "0.9231", "1.3525", "1.5540")


def fixture_check(work=None, quiet=False):
    """Re-derive the eight committed JOIN digits. 0 when every one reproduces."""
    work = work or JOIN_FIXTURE_DIR
    errors = []
    bands, _samples, _req, _comp = bands_from_dir(work, "apr", "llamacpp", errors)
    if errors:
        for e in errors:
            sys.stderr.write("fixture-check: %s\n" % e)
        return 1
    got_bands = tuple(b["concurrency"] for b in bands)
    if got_bands != JOIN_FIXTURE_BANDS:
        sys.stderr.write("fixture-check: bands %s, expected %s -- the fixture lost a "
                         "band, so the four-decimal assertions below would be checking "
                         "the wrong quotients\n" % (got_bands, JOIN_FIXTURE_BANDS))
        return 1
    bad = 0
    for i, band in enumerate(bands):
        for metric, key, want in (("agg", "agg_ratio", JOIN_FIXTURE_AGG[i]),
                                  ("dec", "decode_ratio", JOIN_FIXTURE_DEC[i])):
            got = "%.4f" % band[key]
            mark = "ok  " if got == want else "BAD "
            if got != want:
                bad += 1
            if not quiet:
                print("  %s c=%-2d %-3s %s (expected %s)"
                      % (mark, band["concurrency"], metric, got, want))
    if bad and quiet:
        sys.stderr.write("fixture-check: %d of 8 committed digits did not reproduce\n" % bad)
    return 1 if bad else 0


def _selftest_fixture_rows(work):
    """The fixture reproduces -- AND a one-number perturbation breaks it.

    Without the second row this proves only that some numbers were compared;
    the assertion could be against a value re-derived from the same edit.
    """
    import shutil
    rows = [("the JOIN fixture reproduces its eight committed digits",
             fixture_check(quiet=True) == 0)]
    perturbed = os.path.join(work, "join-perturbed")
    if os.path.isdir(perturbed):
        shutil.rmtree(perturbed)
    shutil.copytree(JOIN_FIXTURE_DIR, perturbed)
    victim = os.path.join(perturbed, "apr-c4.json")
    with open(victim, encoding="utf-8") as handle:
        doc = json.load(handle)
    for run in doc["runs"]:
        run["tokens_per_sec"] = run["tokens_per_sec"] * 1.10
    with open(victim, "w", encoding="utf-8") as handle:
        json.dump(doc, handle)
    rows.append(("a 10% perturbation of ONE band artifact breaks the fixture check",
                 fixture_check(perturbed, quiet=True) == 1))
    return rows


def selftest():
    import shutil
    import tempfile
    gate = os.path.join(SCRIPTS, "perf_gate.sh")
    work = tempfile.mkdtemp(prefix="perf-receipt-selftest-")
    try:
        _fixture(work)
        rows = (_selftest_verdict_rows(work, gate)
                + _selftest_refusal_rows(work)
                + _selftest_fixture_rows(work)
                + _selftest_parity_chain_rows(work, gate)
                + _selftest_p1_chain_rows(work)
                + _selftest_p1_witness_rows(work))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    bad = sum(1 for _name, ok in rows if not ok)
    for name, ok in rows:
        print("  %-5s %s" % ("ok" if ok else "BROKE", name))
    print("  %d passed, %d broken" % (len(rows) - bad, bad))
    return 1 if bad else 0


def _parser():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-bands", help="directory of per-concurrency BenchmarkReport artifacts")
    src.add_argument("--from-parity", help="a parity block emitted by parity_block.py")
    ap.add_argument("--subject", default="apr")
    ap.add_argument("--comparator", default="llamacpp")
    ap.add_argument("--lane", help="which lane of a multi-lane parity block")
    ap.add_argument("--host")
    ap.add_argument("--workload", default="W1")
    ap.add_argument("--binary")
    ap.add_argument("--binary-sha256")
    # NO DEFAULT. This field asserts HOW the binary was resolved, and
    # "scripts/apr_bin.sh" is the one claim that matters -- never PATH,
    # never a hardcoded absolute path. A caller that omits it gets that
    # assertion written into the receipt on its behalf, which is a
    # provenance claim nothing checked: exactly this epic's defect at
    # the scale of one argparse default. It is now required, and
    # _provenance_from_args refuses an empty one by the same rule that
    # refuses a missing digest.
    ap.add_argument("--resolution")
    ap.add_argument("--compute-class")
    ap.add_argument("--accelerator")
    ap.add_argument("--model")
    ap.add_argument("--quantization")
    ap.add_argument("--derive-only", action="store_true",
                    help="print the derived table and write nothing")
    ap.add_argument("--out", default="-")
    return ap


def _report_derived(receipt):
    _print_table(receipt["bands"], receipt.get("requested"),
                 receipt.get("completed"))
    print("  wrote nothing: --derive-only")
    for name in sorted(receipt["unmeasured"]):
        entry = receipt["unmeasured"][name]
        print("  UNMEASURED %-28s owner=%s %s"
              % (name, entry["owner"], entry["spec"]))
    return 0


def _emit(receipt, out):
    # SELF-VALIDATION, with the same code Arm C delegates to. A producer that
    # writes something the gate rejects has moved the failure to release day,
    # when the only cheap option is to weaken the gate.
    schema_errors = bench_receipt.validate(receipt)
    if schema_errors:
        sys.stderr.write("perf_receipt: refusing to emit a receipt this repo's "
                         "own gate would reject:\n")
        for e in schema_errors:
            sys.stderr.write("  %s\n" % e)
        return 1
    text = json.dumps(receipt, indent=2, sort_keys=True)
    if out == "-":
        print(text)
    else:
        with open(out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        sys.stderr.write("wrote %s\n" % out)
    return 0


def main(argv=None):
    head = (argv if argv is not None else sys.argv[1:])[:1]
    if head == ["--selftest"]:
        return selftest()
    if head == ["--fixture-check"]:
        return fixture_check()
    args = _parser().parse_args(argv)

    errors = []
    receipt = build(args, errors)
    if receipt is None or errors:
        sys.stderr.write("perf_receipt: refusing to emit a receipt:\n")
        for e in errors:
            sys.stderr.write("  %s\n" % e)
        return 1

    if args.derive_only:
        return _report_derived(receipt)
    return _emit(receipt, args.out)


if __name__ == "__main__":
    sys.exit(main())
