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
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_receipt  # noqa: E402

# THE THRESHOLDS COME FROM THE DECLARATION, NOT FROM THIS FILE (#2743).
#
# These were literals -- FLOOR = 0.80, CEILING = 1.50 -- while
# scripts/llama_pin.toml declared `band_floor = 0.80` and `band_ceiling = 1.50`
# and NOTHING read either one. Two unjoined copies of the numbers that decide
# PASS and FAIL for this entire epic, and the block emitted by this very file
# carries `protocol_ref: scripts/llama_pin.toml#protocol.http` -- it cited the
# pin as its authority for thresholds it had never read. Editing the pin's floor
# would have changed no verdict, and nothing would have said so.
#
# Same defect as `-b 1` (#2737) and `flash_attention = false` (#2743), one level
# up: not a knob that mis-describes how a number was MEASURED, but a knob that
# mis-describes how it is JUDGED.
#
# FAIL CLOSED. An unreadable declaration raises here rather than falling back to
# a built-in default: a receipt built from thresholds of unknown provenance is
# worse than no receipt, because it looks authoritative. `band_floor` and
# `band_ceiling` are required keys of the fairness protocol
# (scripts/check_bench_protocol.sh), so a missing one is a broken tree.
def _pin_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "llama_pin.toml")


def pin_get(key, path=None):
    """The raw value of a top-level scalar key in the pin declaration.

    Deliberately the same shape as llama_pin_get_raw in scripts/llama_bin.sh:
    first match wins, trailing comment stripped, surrounding quotes stripped.
    Two readers of one file that disagree about how to read it are two files.
    """
    path = path or _pin_path()
    pattern = re.compile(r"^\s*" + re.escape(key) + r"\s*=\s*(.*?)\s*$")
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            found = pattern.match(line)
            if not found:
                continue
            value = found.group(1)
            # A `#` inside a quoted string is not a comment; strip only a
            # trailing one that follows the value.
            if not value.startswith(('"', "'", "[")):
                value = value.split("#", 1)[0].strip()
            return value.strip().strip('"')
    raise SystemExit("FAIL  scripts/llama_pin.toml declares no %s; the parity "
                     "block cannot be judged against thresholds it cannot read "
                     "(#2743)" % key)


def _pin_float(key):
    raw = pin_get(key)
    try:
        return float(raw)
    except ValueError:
        raise SystemExit("FAIL  llama_pin.toml %s = %r is not a number (#2743)"
                         % (key, raw))


def _pin_bands():
    raw = pin_get("http_concurrency_bands")
    try:
        bands = tuple(int(x) for x in raw.strip("[]").split(",") if x.strip())
    except ValueError:
        raise SystemExit("FAIL  llama_pin.toml http_concurrency_bands = %r is "
                         "not a list of integers (#2743)" % raw)
    if not bands:
        raise SystemExit("FAIL  llama_pin.toml declares no concurrency band; a "
                         "lane with no band arms nothing (#2743)")
    return bands


FLOOR = _pin_float("band_floor")        # the release gate: below this, a lane FAILs
CEILING = _pin_float("band_ceiling")    # above this is likelier an error than a win
STRETCH = CEILING     # the stated goal, recorded so the distance is visible


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


# Declared in the pin as http_concurrency_bands, and read from it. This was
# a fourth hardcoded copy of a declared value (#2743): a band added to the
# pin would never have been measured, and the receipt's `declared_bands`
# would have named the bands this tuple happened to hold.
BANDS_DEFAULT = _pin_bands()


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
