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


def _side(binary, sha, klass, path, install_source=None, feature_set=None):
    prov = {"binary_path": binary, "binary_sha256": sha,
            "resolution": "scripts/apr_bin.sh" if install_source else "scripts/llama_bin.sh",
            "compute_class": klass}
    if feature_set is not None:
        prov["feature_set"] = feature_set
    side = {"provenance": prov,
            "decode_tok_per_sec": _samples(path, "decode_tok_per_sec"),
            "prefill_tok_per_sec": _samples(path, "prefill_tok_per_sec"),
            "ttft_p50_ms": _samples(path, "ttft_p50_ms")}
    if install_source:
        side["install_source"] = install_source
    return side


class Refusal(Exception):
    """A ratio that cannot honestly be derived. Carries the sentence the gate
    would have used, naming WHICH SIDE was not a measurement."""


def _ratio(subject, comparator, label, metric, guard_subject=False):
    """median(subject) / median(comparator), or a NAMED refusal (#2735).

    A non-positive COMPARATOR median used to raise ZeroDivisionError straight
    out of here. That exits non-zero, so it was never a false green -- but a
    traceback names no band and no side, and reads as a tooling defect rather
    than as a measurement that must not be published. It is reachable, not
    theoretical: `-b 1` aborts llama.cpp at load (GGML_ASSERT, rc=134), so the
    comparator is exactly the side that produces nothing on gx10 today.

    NOT try/except -> return None. A dropped band vanishes from lane["bands"]
    and surfaces downstream as _check_bands' declared-vs-seen mismatch, which
    reports the wrong cause. A missing measurement must be RED, by name.

    guard_subject stays False where bench_receipt._median_of already inspects
    BOTH sides: there the producer only has to stop crashing before its own
    gate gets to speak, and the refusal is left verbatim to the gate. It is
    True for prefill, which no validator reads at all -- without it a zero
    subject prefill is published as `ratio_prefill: 0.0`, a ratio derived from
    a non-measurement, with rc=0.
    """
    sides = [("comparator", comparator)]
    if guard_subject:
        sides.append(("subject", subject))
    for side, samples in sides:
        median = statistics.median(samples)
        if median <= 0:
            raise Refusal(
                "%s.%s.%s: median of %d sample(s) is %s -- a throughput of "
                "zero or less is not a measurement, and a ratio derived from "
                "it would be fabricated"
                % (label, side, metric, len(samples), median))
    return statistics.median(subject) / statistics.median(comparator)


BANDS_DEFAULT = (1, 4, 8, 16)


def _band_from(name, c, work):
    """One concurrency band, both metrics, ratios DERIVED from the samples."""
    a = os.path.join(work, "apr-%s-c%d.json" % (name, c))
    l = os.path.join(work, "llama-%s-c%d.json" % (name, c))
    if not (os.path.exists(a) and os.path.exists(l)):
        return None
    band = {"concurrency": c,
            "subject": {"aggregate_tok_per_sec": _samples(a, "tokens_per_sec"),
                        "decode_tok_per_sec": _samples(a, "decode_tok_per_sec")},
            "comparator": {"aggregate_tok_per_sec": _samples(l, "tokens_per_sec"),
                           "decode_tok_per_sec": _samples(l, "decode_tok_per_sec")}}
    ok = True
    for metric in ("aggregate_tok_per_sec", "decode_tok_per_sec"):
        ratio = _ratio(band["subject"][metric], band["comparator"][metric],
                       "band[%d]" % c, metric)
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

    label = "lane[%s]" % apr_class
    ratio = _ratio(subject["decode_tok_per_sec"],
                   comparator["decode_tok_per_sec"], label,
                   "decode_tok_per_sec")
    prefill = _ratio(subject["prefill_tok_per_sec"],
                     comparator["prefill_tok_per_sec"], label,
                     "prefill_tok_per_sec", guard_subject=True)
    lane = {"lane": apr_class, "subject": subject, "comparator": comparator,
            "ratio_decode": round(ratio, 4),
            "ratio_prefill": round(prefill, 4)}
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

    try:
        block = build(args)
    except Refusal as refusal:
        sys.stderr.write("FAIL  refusing to emit a block whose ratio would be "
                         "derived from a non-measurement:\n")
        sys.stderr.write("        %s\n" % refusal)
        return 1
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
