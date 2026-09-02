#!/usr/bin/env python3
"""Assemble a parity block from harness reports, and REFUSE to emit an invalid one.

A producer that writes something the gate rejects has not saved anyone work; it
has moved the failure to release day, when the only cheap option is to weaken
the gate. So this validates its own output with the same code the gate runs and
exits non-zero rather than printing a block that will be rejected later.

It also derives every ratio from the samples it just collected. Nothing here
accepts a ratio as input, because a stated ratio is the one field that can be
wrong while every field around it is right (F12).

TWO INPUT LAYOUTS, ONE OUTPUT SHAPE.

  EXECUTOR   the layout scripts/parity_host_receipt.sh writes today: a band
             metadata file per band, N per-replicate reports per lane, the
             comparator's `GET /props`, the subject's `GET
             /v1/effective-config`, and a device record before and after
             (§4.3, §5.2, §5.3, §5.4). Detected by the presence of
             `band-<class>-c<c>.json` and read by `_executor_lane`.
  HISTORICAL one report per lane per band, `{apr,llama}-<class>-c<N>.json`,
             with no metadata beside it. The committed 2026-08-25 corpus is in
             this layout and still has to derive its eight JOIN digits, so the
             reading is KEPT -- but it can carry no replicate estimator, no
             admission record and no isolation record, because none of those
             were ever written down.

The mode is DETECTED, never selected by a flag. A flag would let a caller
declare the layout its files are not in, which is the same class of defect as
labelling a lane by intent.
"""
import argparse
import glob
import json
import os
import statistics
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_receipt  # noqa: E402

# THE THRESHOLDS ARE NOT HERE (PP-33). This file carried FLOOR = 0.80,
# STRETCH = 1.50 and CEILING = 1.50 -- a second, independent encoding of the
# release gate's numbers that no matrix edit could reach, and a STRETCH that
# gated nothing at all while sitting beside a CEILING with the same value. The
# floor is now Arm L3's own non-inferiority bound (1 - delta.agg_ratio) and the
# ceiling is derivation.sanity_ceiling, both read through the shared
# bench_receipt._matrix() helper so producer and gate cannot disagree.


def _samples(path, key):
    """Per-run values from a BenchmarkReport, in run order."""
    with open(path, encoding="utf-8") as handle:
        runs = json.load(handle)["runs"]
    return [r[key] for r in runs]


def _runs(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["runs"]


def _request_latencies(path):
    """Raw per-REQUEST latencies, in run order (§4.4.5).

    This block used to read the BenchmarkReport's request_details and keep only
    per-run summaries, so the raw samples died at this boundary and no
    downstream consumer could re-derive a threshold from a parity block. The
    quantities beside them are per-RUN and none of them is a substitute.
    """
    out = []
    for run in _runs(path):
        out.extend(d["latency_ms"] for d in run.get("request_details", []))
    return out


def _side(binary, sha, klass, path, install_source=None, feature_set=None):
    prov = {"binary_path": binary, "binary_sha256": sha,
            "resolution": "scripts/apr_bin.sh" if install_source else "scripts/llama_bin.sh",
            "compute_class": klass}
    if feature_set is not None:
        prov["feature_set"] = feature_set
    runs = _runs(path)
    side = {"provenance": prov,
            "decode_tok_per_sec": _samples(path, "decode_tok_per_sec"),
            "prefill_tok_per_sec": _samples(path, "prefill_tok_per_sec"),
            "ttft_p50_ms": _samples(path, "ttft_p50_ms"),
            "samples_ms": _request_latencies(path),
            # SGLang asserts completed == requested before it reads a
            # throughput at all, and this block dropped both counts while
            # reading the runs that hold them -- so a lane could report a ratio
            # whose denominator had lost requests, and nothing downstream could
            # tell. Carried on BOTH sides: a comparator that lost requests
            # flatters the subject.
            "requested": sum(r["total_requests"] for r in runs),
            "completed": sum(r["successful"] for r in runs)}
    if install_source:
        side["install_source"] = install_source
    return side


# THE LADDER IS NOT HERE EITHER. `BANDS_DEFAULT = (1, 4, 8, 16)` was one of
# five static copies of a ladder PP-24 says is DERIVED at run time; the producer
# and the gate could silently disagree about which bands were even requested.
def bands_default():
    return tuple(bench_receipt.declared_bands())


# THE LANE-LEVEL SIDE IS THE c=1 BAND. It is not a separate measurement.
#
# THE DEFECT THIS REPLACES (PERF-004). This block used to read
# `$WORK/apr-<lane>.json` and `$WORK/llama-<lane>.json` for the lane-level
# samples. scripts/parity_host_receipt.sh -- its ONLY producer, which invokes
# it directly at line 144 -- writes neither. Its run_lane() writes exactly
# `apr-<lane>-c<N>.json`, `llama-<lane>-c<N>.json`, the two `.log` files and
# `lanes.txt` (lines 96-124), and its own header comment claims a third
# spelling, `$WORK/<class>.json`. So every complete run of the host receipt
# script reached this file and died on
#
#     FAIL  lane cpu is missing a side; refusing to report half a comparison
#
# which reads as "the benchmark did not run" and is in fact "the consumer
# requires an artifact no producer has ever written". The P2 chain has never
# emitted a parity block.
#
# WHY THE CONSUMER MOVED AND NOT THE PRODUCER. Adding an unbanded run to
# run_lane() would have made the producer emit it -- and that run would BE a
# c=1 run: `apr test llm bench --concurrency` defaults to 1, and the
# lane-level artifacts in the 2026-08-25 corpus (evidence/parity-http/
# lambda-apr.json, lambda-llamacpp.json) are concurrency=[1] on every run. So
# option (a) pays for a second, independently-measured copy of a number the
# band sweep already has, and two independent copies of "the same" number is
# how a receipt comes to disagree with itself. It also stops
# llama_pin.toml's `http_concurrency_bands` from being the single statement of
# what was measured, and nothing in the artifact would record that the
# unbanded run was taken at c=1 rather than at some other concurrency.
#
# NO FALLBACK. If band c=1 is absent this refuses by name. Accepting either
# layout would report a lane from whichever happened to be present and prove
# neither contract.
LANE_BAND = 1


def _band_side(path):
    """One side of one band: the two metrics, plus the counts SGLang asserts on.

    The counts were dropped here while the runs holding them were being read, so
    a band could report a ratio whose denominator had lost requests and nothing
    downstream could tell. Carried on BOTH sides: a comparator that loses
    requests flatters the subject.
    """
    runs = _runs(path)
    # PP-8: the concurrency THIS SIDE's client actually drove. The band states
    # one concurrency at the top and both sides used to inherit it by
    # assumption, so a c=4 subject over a c=1 comparator was expressible and
    # invisible. `None` when the runs disagree -- a mixture is not a band, and
    # bench_receipt.validate_parity refuses the pair rather than averaging it.
    levels = {r.get("concurrency") for r in runs}
    side = {"aggregate_tok_per_sec": _samples(path, "tokens_per_sec"),
            "decode_tok_per_sec": _samples(path, "decode_tok_per_sec"),
            "prefill_tok_per_sec": _samples(path, "prefill_tok_per_sec"),
            "client_concurrency": levels.pop() if len(levels) == 1 else None,
            "tokens_total": sum(r["completion_tokens_total"] for r in runs),
            "requested": sum(r["total_requests"] for r in runs),
            "completed": sum(r["successful"] for r in runs)}
    return side


# ===========================================================================
# THE P1 EXECUTOR LAYOUT (§4.3, §5.2, §5.3, §5.4).
#
# scripts/parity_host_receipt.sh was restructured: it relaunches the comparator
# PER BAND (`-np c`), reads `GET /props` and `GET /v1/effective-config` per
# band, and runs the replicates INTERLEAVED -- subject, comparator, subject,
# comparator -- inside ONE invocation, because thermal state, graph-capture
# warm state and free VRAM all drift across a sweep and alternation is the only
# design that cancels the drift. It therefore writes, per band tag
# `<class>-c<c>`:
#
#   band-<tag>.json                  what the band was (every key below)
#   <lane>-<tag>-r<k>.json           one report per replicate, per lane
#   <comparator>-<tag>.props.json    the admission the comparator declared
#   <subject>-<tag>.config.json      the subject's resolved configuration, or
#                                    the literal text `absent`
#   iso-<tag>-{before,after}.json    the device record (§5.4)
#   lanes.txt                        `<class> <gpu-layers>`, one line per lane
#
# THIS FILE READ NONE OF IT. It read one report per lane per band and nothing
# else, so every complete run of the executor reached this file and died on a
# missing side -- which reads as "the benchmark did not run" and is in fact
# "the consumer requires an artifact the producer stopped writing".
#
# NOTHING BELOW GUESSES A FILE NAME. The replicate templates, the props file and
# the two isolation files are all NAMED BY THE PRODUCER inside band-<tag>.json,
# and this reader expands those names rather than restating the convention. A
# restated convention is a second definition, and a second definition is how the
# two band schemas this repository already reconciled came to exist.
# ===========================================================================
BAND_META_GLOB = "band-*.json"
# The subject's config route is recorded as the literal text `absent` when the
# server has no such route, so "did not parse" is a RECORDED STATE here.
CONFIG_ABSENT = "absent"


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _maybe_json(path):
    """A JSON document, or None when the file is absent or is not JSON."""
    try:
        return _read_json(path)
    except (OSError, ValueError):
        return None


# The comparator invocation is a QUANTITY per flag (PP-15), and PP-22 carries
# n_ctx_slot, kv_type, fa and n_batch in the join key precisely so a `-b 1` lane
# can never be joined to a default one. The flags are recorded verbatim by the
# producer; this splits them so the join key can be built from them.
_COMPARATOR_FLAG_KEYS = {
    "-ngl": "gpu_layers", "-c": "n_ctx", "-t": "threads", "-np": "parallel",
    "-fa": "fa", "-ub": "n_ubatch", "-b": "n_batch_flag",
    "-ctk": "kv_type_k", "-ctv": "kv_type_v",
}


def _comparator_flags(text):
    """The comparator's flag QUANTITIES, split out of the recorded line."""
    out, tokens = {}, (text or "").split()
    index = 0
    while index < len(tokens):
        key = _COMPARATOR_FLAG_KEYS.get(tokens[index])
        nxt = tokens[index + 1] if index + 1 < len(tokens) else None
        if key and nxt is not None and not nxt.startswith("-"):
            out[key] = nxt
            index += 2
            continue
        index += 1
    return out


def _kv_type(flags):
    """One kv type, or None when K and V disagree.

    A band whose K cache is f16 and whose V cache is q8_0 has no single
    `kv_type` to put in a join key, and picking one of the two would let two
    unlike runs join.
    """
    key, value = flags.get("kv_type_k"), flags.get("kv_type_v")
    return key if key is not None and key == value else None


def _isolation(work, meta):
    """The device record either side of the band (§5.4, PP-19).

    `contended` is True when EITHER record names a foreign compute pid, False
    when both were asked and named none, and None when the probe was absent --
    an unasked probe is not an idle device, and collapsing the two is how an
    unmeasured isolation reads as isolation.
    """
    out = {"before": None, "after": None, "contended": None, "foreign_pids": []}
    asked = False
    for when in ("before", "after"):
        name = meta.get("isolation_%s_file" % when)
        record = _maybe_json(os.path.join(work, name)) if name else None
        out[when] = record
        if not isinstance(record, dict):
            continue
        foreign = record.get("foreign_pids")
        if foreign is None:
            continue
        asked = True
        out["foreign_pids"].extend(foreign)
    if asked:
        # DEDUPED: the same intruder seen before AND after the band is one
        # intruder, and reporting it twice would make a contended band look
        # worse the longer it sat there.
        out["foreign_pids"] = sorted(set(out["foreign_pids"]))
        out["contended"] = bool(out["foreign_pids"])
    return out


def _effective_config(work, meta):
    """The subject's own resolved configuration (§5.2, PP-2), or its absence."""
    name = meta.get("subject_effective_config_file")
    body = _maybe_json(os.path.join(work, name)) if name else None
    out = {"state": meta.get("subject_effective_config"), "file": name,
           "compute_class": None, "slots_admitted": None, "started_utc": None,
           "body": body if isinstance(body, dict) else None}
    if not isinstance(body, dict):
        return out
    out["compute_class"] = body.get("compute_class")
    scheduler = body.get("scheduler")
    if isinstance(scheduler, dict):
        out["slots_admitted"] = scheduler.get("slots_admitted")
    server = body.get("server")
    if isinstance(server, dict):
        out["started_utc"] = server.get("started_utc")
    return out


def _replicate_paths(work, meta, side):
    """The per-replicate report paths the producer NAMED, k ascending."""
    template = (meta.get("replicate_files") or {}).get(side)
    count = meta.get("replicates")
    if not template or not isinstance(count, int) or count < 1:
        return []
    return [os.path.join(work, template.replace("{k}", str(k)))
            for k in range(1, count + 1)]


def _request_rows(paths):
    """Every raw per-request row across the replicates, in replicate order."""
    rows = []
    for path in paths:
        for run in _runs(path):
            rows.extend(run.get("request_details") or [])
    return rows


# The raw row fields §4.4.5 / PP-7 keeps. `token_times` is deliberately NOT
# among them: it belongs in the samples file beside the receipt, not inline.
_ROW_FIELDS = ("latency_ms", "ttft_ms", "completion_tokens", "prompt_tokens",
               "finish_reason")


def _carried_rows(rows):
    """The raw rows, indexed, with only the fields the wire keeps."""
    out = []
    for index, row in enumerate(rows):
        carried = {"index": index}
        for key in _ROW_FIELDS:
            carried[key] = row.get(key)
        out.append(carried)
    return out


def _short_of_n_predict(rows, n_predict):
    """PP-28: rows that completed BELOW the pinned generation length.

    `truncated` counts drain-abandoned requests only, so without this nothing
    in the artifact witnesses a completion that simply stopped early -- and a
    lane that generated 112 tokens where the other generated 128 has a
    throughput denominator the other one does not.
    """
    return sum(1 for row in rows
               if row.get("completion_tokens") is not None
               and row.get("completion_tokens") != n_predict)


WITNESS_RESULTS = ("PASS", "FAIL", "UNMEASURABLE")


def load_perf041_witness(path):
    """`bands[]` of a scripts/perf041_batched_parity_probe.py witness, keyed by c.

    The per-replicate bench reports never carry a witness (the non-band bench
    has no such field), so the PP-26 result reaches an executor block only
    through this file. A result outside PASS/FAIL/UNMEASURABLE is refused
    rather than mapped to the failing side, the same rule apr-cli's band
    producer applies (test_llm_band.rs `witness_of`).
    """
    if not path:
        return {}
    with open(path, encoding="utf-8") as handle:
        doc = json.load(handle)
    out = {}
    for band in doc.get("bands") or []:
        c = band.get("c")
        result = band.get("result")
        if c is None:
            continue
        if result not in WITNESS_RESULTS:
            raise ValueError("witness band c=%r result %r is not one of %s"
                             % (c, result, list(WITNESS_RESULTS)))
        out[int(c)] = {
            "batch_invariance": result,
            "divergence_at": band.get("divergence_at"),
            "declared_min": band.get("declared_min"),
            "m_formed": band.get("m_formed"),
            "intra_agree_to": band.get("intra_agree_to"),
            "max_constant_run": band.get("max_constant_run"),
            "source": "scripts/perf041_batched_parity_probe.py (%s, commit %s)"
                      % (os.path.basename(path), doc.get("commit")),
        }
    return out


def _witness_for(paths, c, witnesses):
    """The band's witness: from the reports if every replicate agrees, else
    from the perf041 file for this c, else None (which §7.4 reads as absent)."""
    agreed = _agreed(paths, "witness")
    if agreed is not None:
        return agreed
    return witnesses.get(int(c)) if c is not None else None


def _agreed(paths, key):
    """A per-run value all the replicates agree on, or None.

    Disagreement is None rather than a majority: a band whose replicates
    disagree about whether the stream was live has not measured one thing.
    """
    seen, missing = [], False
    for path in paths:
        for run in _runs(path):
            if key not in run:
                missing = True
                continue
            seen.append(json.dumps(run[key], sort_keys=True))
    if missing or not seen or len(set(seen)) != 1:
        return None
    return json.loads(seen[0])


def _summed(paths, key):
    """A per-run counter summed across the replicates, or None if unreported."""
    total, seen = 0, False
    for path in paths:
        for run in _runs(path):
            if run.get(key) is None:
                continue
            seen = True
            total += run[key]
    return total if seen else None


def _per_replicate(paths, key):
    """ONE value per REPLICATE, in replicate order (§4.3's estimator input).

    The executor runs `--runs 1`, so each file normally holds a single run; the
    median collapses a multi-run file rather than letting the list length stop
    matching the replicate count, which is what makes the k-th subject value
    and the k-th comparator value a PAIR.
    """
    return [statistics.median([r[key] for r in _runs(path)]) for path in paths]


def _per_replicate_opt(paths, key):
    """`_per_replicate`, or None when a replicate does not report the key."""
    out = []
    for path in paths:
        values = [r[key] for r in _runs(path) if r.get(key) is not None]
        if not values:
            return None
        out.append(statistics.median(values))
    return out


def _decode_rate_rows(rows):
    """Per-REQUEST decode rate, tok/s, for the request-unit bootstrap.

    generated tokens over the time after the first one arrived. A row missing
    either timing, or whose decode span is not positive, contributes nothing:
    an infinite rate is not a fast request.
    """
    out = []
    for row in rows:
        tokens = row.get("completion_tokens")
        latency, ttft = row.get("latency_ms"), row.get("ttft_ms")
        if tokens is None or latency is None or ttft is None:
            continue
        span = latency - ttft
        if span <= 0 or tokens <= 1:
            continue
        out.append((tokens - 1) * 1000.0 / span)
    return out


def _ttft_over_e2e_rows(rows):
    """Per-request client-side ttft/e2e (PP-27's client half)."""
    out = []
    for row in rows:
        latency, ttft = row.get("latency_ms"), row.get("ttft_ms")
        if latency is None or ttft is None or latency <= 0:
            continue
        out.append(ttft / latency)
    return out


def _executor_side(work, meta, side, n_predict, witnesses=None):
    witnesses = witnesses or {}
    meta_c = meta.get("concurrency")
    """One lane of one band, read from the replicate files the producer named."""
    paths = [p for p in _replicate_paths(work, meta, side) if os.path.exists(p)]
    if not paths:
        return None, []
    runs = [r for path in paths for r in _runs(path)]
    rows = _request_rows(paths)
    levels = {r.get("concurrency") for r in runs}
    out = {
        # PER-REPLICATE, in replicate order: the k-th entry here and the k-th
        # entry on the other side are the PAIR §4.3's estimator divides.
        "aggregate_tok_per_sec": _per_replicate(paths, "tokens_per_sec"),
        "decode_tok_per_sec": _per_replicate(paths, "decode_tok_per_sec"),
        "prefill_tok_per_sec": _per_replicate(paths, "prefill_tok_per_sec"),
        "ttft_p50_ms": _per_replicate(paths, "ttft_p50_ms"),
        "client_concurrency": levels.pop() if len(levels) == 1 else None,
        "replicate_files": [os.path.basename(p) for p in paths],
        "tokens_total": sum(r["completion_tokens_total"] for r in runs),
        "requested": sum(r["total_requests"] for r in runs),
        "completed": sum(r["successful"] for r in runs),
        "samples_ms": [row["latency_ms"] for row in rows if "latency_ms" in row],
        # PP-7: the raw rows survive the boundary. This block used to keep only
        # per-RUN summaries, so a threshold could never be re-derived from a
        # parity block afterwards.
        "request_rows": _carried_rows(rows),
        "elapsed_secs": _per_replicate_opt(paths, "elapsed_secs"),
        "request_decode_tok_per_sec": _decode_rate_rows(rows),
        "request_ttft_over_e2e": _ttft_over_e2e_rows(rows),
        "short_of_n_predict": _short_of_n_predict(rows, n_predict),
        # NOT PRODUCED BY TODAY'S HARNESS, and carried as None rather than
        # omitted so a consumer can tell "the run had no live stream" from "no
        # instrument here reports one". §7.4 treats an absent conformance
        # input exactly as it treats a failing one.
        "stream_mode": _agreed(paths, "stream_mode"),
        "witness": _witness_for(paths, meta_c, witnesses),
        "timeouts": _summed(paths, "timeouts"),
        "drain_ms": _summed(paths, "drain_ms"),
    }
    return out, paths


def _executor_band(work, meta, floor, ceiling, n_predict, witnesses=None):
    """One band of the executor layout: both lanes, every record beside them."""
    c = meta.get("concurrency")
    subject, s_paths = _executor_side(work, meta, "subject", n_predict, witnesses)
    comparator, c_paths = _executor_side(work, meta, "comparator", n_predict, witnesses)
    if subject is None or comparator is None:
        return None
    flags = _comparator_flags(meta.get("comparator_flags"))
    band = {
        "concurrency": c,
        "interleaved": meta.get("interleaved"),
        # DECLARED vs PRESENT. A replicate whose bench invocation failed leaves
        # no file at all, so the pair count is what was READ, never what was
        # asked for; §7.4 sizes its bound on the smaller of the two.
        "replicates_declared": meta.get("replicates"),
        "replicates": min(len(s_paths), len(c_paths)),
        "client_concurrency_declared": meta.get("client_concurrency"),
        "subject": subject,
        "comparator": comparator,
        "comparator_admission": {
            "slots_admitted": meta.get("comparator_slots_admitted"),
            "n_ctx_slot": meta.get("comparator_n_ctx_slot"),
            "n_batch": meta.get("comparator_n_batch"),
            "kv_type": _kv_type(flags),
            "kv_type_k": flags.get("kv_type_k"),
            "kv_type_v": flags.get("kv_type_v"),
            "fa": flags.get("fa"),
            "gpu_layers": flags.get("gpu_layers"),
            "parallel": flags.get("parallel"),
            "flags": meta.get("comparator_flags"),
            "props_file": meta.get("comparator_props_file"),
            "props": _maybe_json(os.path.join(
                work, meta.get("comparator_props_file") or "")),
        },
        "subject_compute_class": meta.get("subject_compute_class"),
        "subject_effective_config": _effective_config(work, meta),
        "gpu_layers": {"requested": meta.get("gpu_layers_requested"),
                       "resolved": meta.get("gpu_layers_resolved"),
                       "total": meta.get("gpu_layers_total")},
        "isolation": _isolation(work, meta),
    }
    ok = True
    for metric in ("aggregate_tok_per_sec", "decode_tok_per_sec"):
        ratio = (statistics.median(band["subject"][metric])
                 / statistics.median(band["comparator"][metric]))
        band["ratio_" + metric] = round(ratio, 4)
        if ratio < floor or ratio > ceiling:
            ok = False
    band["verdict"] = "PASS" if ok else "FAIL"
    return band


def _executor_bands(work, klass, floor, ceiling, witnesses=None):
    """Every band of one lane, ascending, and the ladder the lane REQUESTED.

    The requested ladder is the set of band METADATA files, not the set of
    bands that produced a pair, and not the matrix's declared ladder: the
    executor reads its bands from scripts/llama_pin.toml and writes one
    metadata file per band it ran. Deriving the ladder from the bands that
    survived would let a band whose reports vanished disappear from its own
    denominator, which is the unmeasured-band-reads-as-passing shape; deriving
    it from the matrix would fail every lane the pin declares differently.
    """
    n_predict = bench_receipt.matrix_number("protocol", "n_predict")
    out, requested = [], []
    for path in sorted(glob.glob(os.path.join(work, BAND_META_GLOB))):
        meta = _maybe_json(path)
        if not isinstance(meta, dict) or meta.get("class") != klass:
            continue
        if isinstance(meta.get("concurrency"), int):
            requested.append(meta["concurrency"])
        band = _executor_band(work, meta, floor, ceiling, n_predict, witnesses)
        if band is not None:
            out.append(band)
    return (sorted(out, key=lambda b: b["concurrency"]), sorted(requested))


def _executor_lane(klass, args, work):
    """One lane of the executor layout, or None with the reason on stderr."""
    floor, ceiling = bench_receipt.lane_bounds()
    witnesses = load_perf041_witness(getattr(args, "witness_json", None))
    bands, requested = _executor_bands(work, klass, floor, ceiling, witnesses)
    if not bands:
        sys.stderr.write(
            "FAIL  lane %s: no band metadata file under %s carries class %r.\n"
            "      The executor writes one band-<class>-c<c>.json per band; "
            "without it nothing states what the band was.\n" % (klass, work, klass))
        return None
    base = next((b for b in bands if b["concurrency"] == LANE_BAND), None)
    if base is None:
        sys.stderr.write(
            "FAIL  lane %s: the c=%d band is the lane-level measurement and it "
            "is absent.\n      Refusing to report half a comparison. There is no "
            "fallback to another band: a lane reported from c=4 while claiming "
            "the lane ratio would be a different measurement under the same "
            "name.\n" % (klass, LANE_BAND))
        return None
    # THE CLASS COMES FROM THE SERVER'S OWN OUTPUT, never from the flag: the
    # executor reads it out of the loader's line about itself and writes it into
    # the band metadata. The comparator's class is the OFFLOAD QUANTITY it was
    # given -- zero layers is the cpu path -- and `comparator_class_source`
    # records which of the two statements the value came from.
    apr_class = base["subject_compute_class"] or "unknown"
    ngl = base["comparator_admission"].get("gpu_layers")
    comp_class, source = "unknown", "unresolved"
    if ngl is not None and ngl.isdigit():
        comp_class = "cpu" if int(ngl) == 0 else apr_class
        source = "declared-gpu-layers"
    feats = ["cli", "inference"] + ([apr_class] if apr_class != "cpu" else [])
    subject = dict(base["subject"])
    subject["provenance"] = {
        "binary_path": args.apr, "binary_sha256": args.apr_sha,
        "resolution": "scripts/apr_bin.sh", "compute_class": apr_class,
        "feature_set": feats}
    subject["install_source"] = args.install_source
    comparator = dict(base["comparator"])
    comparator["provenance"] = {
        "binary_path": args.llama, "binary_sha256": args.llama_sha,
        "resolution": "scripts/llama_bin.sh", "compute_class": comp_class}
    comparator["name"] = "llama.cpp"
    comparator["build_commit"] = args.llama_build
    if args.pin_expiry:
        comparator["pin_expiry"] = args.pin_expiry
    comparator["compute_class_source"] = source
    ratio = (statistics.median(subject["decode_tok_per_sec"])
             / statistics.median(comparator["decode_tok_per_sec"]))
    lane = {"lane": apr_class, "layout": "executor",
            "subject": subject, "comparator": comparator,
            "ratio_decode": round(ratio, 4),
            "ratio_prefill": round(
                statistics.median(subject["prefill_tok_per_sec"])
                / statistics.median(comparator["prefill_tok_per_sec"]), 4),
            "declared_bands": requested,
            "ladder_declared": list(bands_default()),
            "bands": bands, "ceiling": ceiling}
    _apply_verdict(lane, ratio, apr_class, comp_class)
    if any(b["verdict"] == "FAIL" for b in bands):
        lane["verdict"] = "FAIL"
    return lane


def _band_from(name, c, work, floor, ceiling):
    """One concurrency band, both metrics, ratios DERIVED from the samples."""
    a = os.path.join(work, "apr-%s-c%d.json" % (name, c))
    l = os.path.join(work, "llama-%s-c%d.json" % (name, c))
    if not (os.path.exists(a) and os.path.exists(l)):
        return None
    band = {"concurrency": c,
            "subject": _band_side(a),
            "comparator": _band_side(l)}
    ok = True
    for metric in ("aggregate_tok_per_sec", "decode_tok_per_sec"):
        ratio = (statistics.median(band["subject"][metric])
                 / statistics.median(band["comparator"][metric]))
        band["ratio_" + metric] = round(ratio, 4)
        if ratio < floor or ratio > ceiling:
            ok = False
    band["verdict"] = "PASS" if ok else "FAIL"
    return band


def _lane_from(name, apr_class, comp_class, args, work):
    """One lane, or None if a side is missing.

    The lane-level samples come from the c=1 band -- see LANE_BAND. The paths
    named in the failure below are the ones run_lane() writes, so a missing
    side names a file the producer was supposed to have produced rather than a
    file nothing has ever produced.
    """
    apr_json = os.path.join(work, "apr-%s-c%d.json" % (name, LANE_BAND))
    cmp_json = os.path.join(work, "llama-%s-c%d.json" % (name, LANE_BAND))
    missing = [p for p in (apr_json, cmp_json) if not os.path.exists(p)]
    if missing:
        sys.stderr.write(
            "FAIL  lane %s: the c=%d band is the lane-level measurement and "
            "these are absent:\n" % (name, LANE_BAND))
        for path in missing:
            sys.stderr.write("        %s\n" % path)
        sys.stderr.write("      Refusing to report half a comparison. There is "
                         "no fallback to another band: a lane reported from "
                         "c=4 while claiming the lane ratio would be a "
                         "different measurement under the same name.\n")
        return None
    # feature_set is DERIVED from the class actually taken, so a cuda lane
    # cannot be claimed by a build that never took the cuda path.
    feats = ["cli", "inference"] + ([apr_class] if apr_class != "cpu" else [])
    subject = _side(args.apr, args.apr_sha, apr_class, apr_json,
                    install_source=args.install_source, feature_set=feats)
    comparator = _side(args.llama, args.llama_sha, comp_class, cmp_json)
    comparator["name"] = "llama.cpp"
    comparator["build_commit"] = args.llama_build

    ratio = (statistics.median(subject["decode_tok_per_sec"])
             / statistics.median(comparator["decode_tok_per_sec"]))
    lane = {"lane": apr_class, "subject": subject, "comparator": comparator,
            "ratio_decode": round(ratio, 4),
            "ratio_prefill": round(
                statistics.median(subject["prefill_tok_per_sec"])
                / statistics.median(comparator["prefill_tok_per_sec"]), 4)}
    floor, ceiling = bench_receipt.lane_bounds()
    declared = bands_default()
    bands = [b for b in (_band_from(name, c, work, floor, ceiling) for c in declared) if b]
    if bands:
        lane["declared_bands"] = list(declared)
        lane["bands"] = bands
        lane["ceiling"] = ceiling
    _apply_verdict(lane, ratio, apr_class, comp_class)
    if bands and any(b["verdict"] == "FAIL" for b in bands):
        # A lane cannot be greener than its worst band.
        lane["verdict"] = "FAIL"
    return lane


def _apply_verdict(lane, ratio, apr_class, comp_class):
    """A same-class lane gets a floor and a verdict; a cross-class one gets
    neither. #2696's honest shape is to say it is not a comparison rather than
    invent a number for it."""
    if apr_class != comp_class:
        lane["comparability"] = "cross-class-existence-only"
        lane["verdict"] = "EXISTENCE-ONLY"
        lane["note"] = ("apr took the %s path while the comparator took %s; "
                        "this is not a comparison and arms no floor"
                        % (apr_class, comp_class))
        return
    floor, ceiling = bench_receipt.lane_bounds()
    lane["floor"] = floor
    lane["ceiling"] = ceiling
    # WHERE THE NUMBER CAME FROM, carried in the artifact. Without it a reader
    # of a block cannot tell a matrix-read floor from a literal, which is the
    # state this file was in.
    lane["threshold_source"] = "scripts/perf-matrix.yaml#arms.L3.delta,derivation.sanity_ceiling"
    lane["verdict"] = "PASS" if ratio >= floor else "FAIL"


def _lane_rows(lanes_txt):
    """The lanes that ran, from the file the executor appends to.

    TWO SPELLINGS, because the producer changed and the corpus did not. The
    executor writes `<class> <gpu-layers>` -- two fields, the second a QUANTITY
    handed to the loader, NOT a class. This reader used to unpack three names
    out of every line and therefore raised ValueError on the only line its own
    producer has ever written. A three-field line is the historical spelling
    (`<name> <subject class> <comparator class>`) and is still read.
    """
    with open(lanes_txt, encoding="utf-8") as handle:
        return [line.split() for line in handle if line.strip()]


def _executor_layout(work):
    """Did the executor write a band metadata file? That is the whole test."""
    return bool(glob.glob(os.path.join(work, BAND_META_GLOB)))


def build(args):
    lanes_txt = os.path.join(args.work, "lanes.txt")
    if not os.path.exists(lanes_txt):
        sys.stderr.write("FAIL  no lane ran; nothing to report\n")
        return None
    rows = _lane_rows(lanes_txt)
    executor = _executor_layout(args.work)

    lanes = []
    for row in rows:
        if executor:
            lane = _executor_lane(row[0], args, args.work)
        elif len(row) == 3:
            lane = _lane_from(row[0], row[1], row[2], args, args.work)
        else:
            sys.stderr.write(
                "FAIL  lanes.txt line %r has %d field(s) and no band metadata "
                "file sits beside it. Two fields are the executor spelling and "
                "need band-<class>-c<c>.json; three are the historical one.\n"
                % (" ".join(row), len(row)))
            return None
        if lane is None:
            return None
        lanes.append(lane)

    floor, ceiling = bench_receipt.lane_bounds()
    block = {"instrument": "apr test llm bench",
             "protocol_ref": "scripts/llama_pin.toml#protocol.http",
             "model": os.path.basename(args.model),
             "floor": floor, "ceiling": ceiling,
             "threshold_source": "scripts/perf-matrix.yaml#arms.L3.delta,derivation.sanity_ceiling",
             "lanes": lanes}
    if executor:
        # ONE INVOCATION, ONE IDENTITY (PP-3). The two lanes of an interleaved
        # band are only a pair because they were driven inside the same
        # invocation, and nothing in the artifact said so. A receipt derived
        # from this block copies the id onto every band and onto the baseline
        # beside it, so a cross-run pairing stops being expressible.
        block["run_id"] = args.run_id or uuid.uuid4().hex
        block["layout"] = "executor"
        if args.pin_expiry:
            block["comparator_pin_expiry"] = args.pin_expiry
    return block


def main():
    ap = argparse.ArgumentParser()
    for flag in ("--work", "--apr", "--apr-sha", "--llama", "--llama-sha",
                 "--llama-build", "--model", "--install-source", "--out"):
        ap.add_argument(flag, required=True)
    # OPTIONAL, so the historical corpus and every existing caller still run.
    # `--pin-expiry` is what makes COMPARATOR_STALE (§7.4, PP-20) decidable
    # downstream: the executor refuses to start against an expired pin, but a
    # block read months later cannot tell that from silence.
    ap.add_argument("--pin-expiry")
    ap.add_argument("--witness-json", help="scripts/perf041_batched_parity_probe.py "
                    "witness for this host; its bands[] supply each band's PP-26 "
                    "block, since the per-replicate bench reports carry none")
    ap.add_argument("--run-id", help="the invocation id both lanes share; "
                                     "minted here when not given")
    args = ap.parse_args()

    block = build(args)
    if block is None:
        return 1

    # SELF-VALIDATION. The same function the release gate calls.
    errors = bench_receipt.validate_parity(block)
    if errors:
        sys.stderr.write("FAIL  refusing to emit a block this repo's own gate "
                         "would reject:\n")
        for e in errors:
            sys.stderr.write("        %s\n" % e)
        return 1

    text = json.dumps({"parity": block}, indent=2)
    if args.out == "-":
        print(text)
    else:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        sys.stderr.write("wrote %s\n" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
