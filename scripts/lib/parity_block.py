#!/usr/bin/env python3
"""Assemble a parity block from harness reports, and REFUSE to emit an invalid one.

A producer that writes something the gate rejects has not saved anyone work; it
has moved the failure to release day, when the only cheap option is to weaken
the gate. So this validates its own output with the same code the gate runs and
exits non-zero rather than printing a block that will be rejected later.

It also derives every ratio from the samples it just collected. Nothing here
accepts a ratio as input, because a stated ratio is the one field that can be
wrong while every field around it is right (F12).
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_receipt  # noqa: E402

FLOOR = 0.80          # the release gate: below this, a lane FAILs
STRETCH = 1.50        # the stated goal, recorded so the distance is visible
CEILING = 1.50        # a ratio above this is likelier a measurement error than a win


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


BANDS_DEFAULT = (1, 4, 8, 16)


def _band_side(path):
    """One side of one band: the two metrics, plus the counts SGLang asserts on.

    The counts were dropped here while the runs holding them were being read, so
    a band could report a ratio whose denominator had lost requests and nothing
    downstream could tell. Carried on BOTH sides: a comparator that loses
    requests flatters the subject.
    """
    runs = _runs(path)
    return {"aggregate_tok_per_sec": _samples(path, "tokens_per_sec"),
            "decode_tok_per_sec": _samples(path, "decode_tok_per_sec"),
            "tokens_total": sum(r["completion_tokens_total"] for r in runs),
            "requested": sum(r["total_requests"] for r in runs),
            "completed": sum(r["successful"] for r in runs)}


def _band_from(name, c, work):
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
        if ratio < FLOOR or ratio > CEILING:
            ok = False
    band["verdict"] = "PASS" if ok else "FAIL"
    return band


def _lane_from(name, apr_class, comp_class, args, work):
    """One lane, or None if a side is missing."""
    apr_json = os.path.join(work, "apr-%s.json" % name)
    cmp_json = os.path.join(work, "llama-%s.json" % name)
    if not (os.path.exists(apr_json) and os.path.exists(cmp_json)):
        sys.stderr.write("FAIL  lane %s is missing a side; refusing to report "
                         "half a comparison\n" % name)
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
    bands = [b for b in (_band_from(name, c, work) for c in BANDS_DEFAULT) if b]
    if bands:
        lane["declared_bands"] = list(BANDS_DEFAULT)
        lane["bands"] = bands
        lane["ceiling"] = CEILING
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
    lane["floor"] = FLOOR
    lane["stretch"] = STRETCH
    lane["verdict"] = "PASS" if ratio >= FLOOR else "FAIL"


def build(args):
    lanes_txt = os.path.join(args.work, "lanes.txt")
    if not os.path.exists(lanes_txt):
        sys.stderr.write("FAIL  no lane ran; nothing to report\n")
        return None
    with open(lanes_txt, encoding="utf-8") as handle:
        rows = [line.split() for line in handle if line.strip()]

    lanes = []
    for name, apr_class, comp_class in rows:
        lane = _lane_from(name, apr_class, comp_class, args, args.work)
        if lane is None:
            return None
        lanes.append(lane)

    return {"instrument": "apr test llm bench",
            "protocol_ref": "scripts/llama_pin.toml#protocol.http",
            "model": os.path.basename(args.model),
            "floor": FLOOR, "stretch": STRETCH, "lanes": lanes}


def main():
    ap = argparse.ArgumentParser()
    for flag in ("--work", "--apr", "--apr-sha", "--llama", "--llama-sha",
                 "--llama-build", "--model", "--install-source", "--out"):
        ap.add_argument(flag, required=True)
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
