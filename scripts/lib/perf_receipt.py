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

  subject     /home/noah/.cargo/bin/apr, `apr 0.64.0 (ce712eae0)`
              sha256 964503625a69462e24964c5b8118b1b78a1c0cae8e4dbdf845b2da25281d01c9
  comparator  /home/noah/src/llama.cpp/build/bin/llama-server
              `version: 7746 (39173bcac)` == scripts/llama_pin.toml's pin
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
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_receipt  # noqa: E402

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(SCRIPTS, "perf-receipt-fields.yaml")
BANDS_DEFAULT = (1, 4, 8, 16)


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
    return {
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


def bands_from_dir(work, subject, comparator, errors):
    """One band per concurrency for which BOTH sides exist."""
    bands, samples, requested, completed = [], [], 0, 0
    for c in BANDS_DEFAULT:
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
        eff = (b["aggregate_tok_per_sec"] / base) / c if base else float("nan")
        cr, cc = b.get("comparator_requested"), b.get("comparator_completed")
        comp = "%d/%d" % (cc, cr) if cr is not None else "-"
        print("%3d  %10.2f  %11.4f  %10.4f  %12.4f  %14s"
              % (c, b["aggregate_tok_per_sec"], eff, b["agg_ratio"],
                 b["decode_ratio"], comp))
    if requested is not None:
        print("  subject completed/requested: %d/%d" % (completed, requested))


def _fill_from_parity(args, receipt, errors):
    bands, subject, _lane = bands_from_parity(args.from_parity, args.lane, errors)
    if bands is None:
        return None
    prov = dict(subject.get("provenance") or {})
    if args.host and not prov.get("host"):
        prov["host"] = args.host
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
        "schema": "perf-receipt/v2.2-partial",
        "host": args.host,
        "workload": args.workload,
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
        ("ArmB scores the converted receipt", "ArmB1 c=4" in text),
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


# ------------------------------------------- P2: the parity chain, end to end
# scripts/parity_host_receipt.sh -> lib/parity_block.py -> --from-parity ->
# scripts/perf_gate.sh.
#
# THIS BRANCH HAD NEVER RUN. `--from-parity` was reachable from no test, and
# the join underneath it was broken in a way no unit test could see:
# parity_block.py read `$WORK/apr-<lane>.json` and `$WORK/llama-<lane>.json`,
# and parity_host_receipt.sh -- its only producer, which invokes it directly --
# writes neither. run_lane() writes `apr-<lane>-c<N>.json`,
# `llama-<lane>-c<N>.json`, two `.log` files and `lanes.txt`, and nothing else.
# So every complete host run ended at
#
#     FAIL  lane cpu is missing a side; refusing to report half a comparison
#
# which reads as "the benchmark did not run". The chain had never produced a
# parity block, and a receipt-schema epic had no test that ran it.
#
# The layout below is written from run_lane()'s own output targets, so a future
# divergence between producer and consumer fails HERE rather than on a host at
# 3am.
PARITY_LANE = "cpu"
PARITY_BANDS = (1, 4, 8, 16)   # llama_pin.toml [protocol.http] concurrency bands
PARITY_RATIO = 1.05            # inside [0.80, 1.50], so the lane verdict is PASS


def _parity_work(root, bands=PARITY_BANDS):
    """A $WORK laid out exactly as parity_host_receipt.sh run_lane() writes it."""
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
                      [_fake_run(c, agg / PARITY_RATIO, dec / PARITY_RATIO,
                                 20, 20, 2560, 90.0),
                       _fake_run(c, agg * 1.01 / PARITY_RATIO,
                                 dec * 1.01 / PARITY_RATIO, 20, 20, 2560, 93.0)])
    with open(os.path.join(work, "lanes.txt"), "w", encoding="utf-8") as fh:
        # run_lane: printf '%s %s %s\n' <klass> <apr class taken> <cmp taken>
        fh.write("%s cpu cpu\n" % PARITY_LANE)
    return work


def _run_parity_block(work, out):
    """parity_block.py with the argument set parity_host_receipt.sh passes."""
    import subprocess
    return subprocess.run(
        [sys.executable, os.path.join(SCRIPTS, "lib", "parity_block.py"),
         "--work", work, "--apr", "/opt/apr", "--apr-sha", "1" * 64,
         "--llama", "/opt/llama-server", "--llama-sha", "2" * 64,
         "--llama-build", "39173bcac", "--model", "/models/q.gguf",
         "--install-source", "crates.io", "--out", out],
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
        ("P2 chain reaches ArmB", "ArmB1 c=4" in text),
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


def selftest():
    import shutil
    import tempfile
    gate = os.path.join(SCRIPTS, "perf_gate.sh")
    work = tempfile.mkdtemp(prefix="perf-receipt-selftest-")
    try:
        _fixture(work)
        rows = (_selftest_verdict_rows(work, gate)
                + _selftest_refusal_rows(work)
                + _selftest_parity_chain_rows(work, gate))
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
    if (argv if argv is not None else sys.argv[1:])[:1] == ["--selftest"]:
        return selftest()
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
