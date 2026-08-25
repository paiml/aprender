#!/usr/bin/env python3
"""The ONE validator for a bench receipt (PARITY-002).

Requires Python 3.6+ (json, sys only).

A receipt that proves WHICH BINARY ran but not WHICH DISPATCH PATH it took
catches the wrong-binary class -- five independent in-tree rediscoveries:
PMAT_BIN, pv_bin.sh, apr_bin.sh, aprender#2384, the unpinned llama.cpp
comparator -- and misses the wrong-compute-class one entirely. That miss is
how a CPU-only apr side measured against a CUDA comparator validates cleanly
and reports the fabricated 14x regression documented at
crates/apr-cli/src/dispatch.rs:165 (`ratio_median=0.070x ... with nothing
wrong in apr's decode path`).

Two rules carry the weight:

  1. compute_class is REQUIRED, and describes the path TAKEN, not the
     hardware present.
  2. A cross-class run CANNOT carry a threshold. This is the born-disarmed
     failure made unwriteable rather than discouraged: the shape where a row
     goes EXISTENCE-ONLY and the threshold never arms for any release, which
     is exactly how check_multiplatform_dogfood never passed for any release.

Usage:  bench_receipt.py <receipt.json> [...]
Exit:   0 all valid - 1 a receipt is invalid - 2 usage/read error
"""
import json
import statistics
import sys

COMPUTE_CLASSES = ("cpu", "cuda", "metal", "wgpu", "unknown")
PROVENANCE_REQUIRED = ("binary_path", "binary_sha256", "resolution", "compute_class")
DISCARD_REASONS = ("early_eos", "negative_delta", "nonzero_exit")


def _err(errors, msg):
    errors.append(msg)


def _check_provenance(prov, errors):
    """Which binary ran, and which path it took."""
    for key in PROVENANCE_REQUIRED:
        if key not in prov:
            _err(errors, "provenance.%s: missing (required)" % key)

    klass = prov.get("compute_class")
    if klass is not None and klass not in COMPUTE_CLASSES:
        _err(errors, "provenance.compute_class: %r not in %s"
                     % (klass, list(COMPUTE_CLASSES)))

    # A class the build cannot reach is a fabricated claim, not a measurement.
    features = prov.get("feature_set")
    if isinstance(features, list) and klass in ("cuda", "wgpu") and klass not in features:
        _err(errors, "provenance.compute_class=%s but feature_set=%s does not "
                     "contain it -- a build without the feature cannot take "
                     "that path" % (klass, features))

    sha = prov.get("binary_sha256")
    if sha is not None and not _is_sha256(sha):
        _err(errors, "provenance.binary_sha256: not a 64-char lowercase hex digest")


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _check_samples(receipt, errors):
    """Raw samples survive, and are a distribution rather than a constant."""
    samples = receipt.get("samples_ms")
    if samples is None:
        _err(errors, "samples_ms: missing -- summary statistics cannot be "
                     "resampled, so a receipt without raw samples permanently "
                     "forecloses bootstrap threshold derivation")
        return
    if not isinstance(samples, list) or not samples:
        _err(errors, "samples_ms: present but empty -- a measurement over zero "
                     "samples is a vacuous pass")
        return
    if any(not isinstance(x, (int, float)) for x in samples):
        _err(errors, "samples_ms: contains a non-numeric entry")
    elif len(samples) > 1 and len(set(samples)) == 1:
        # F12: a value with the form of a measurement and nothing behind it.
        _err(errors, "samples_ms: all %d samples identical (%r) -- a real "
                     "timing distribution is not constant; this is the "
                     "fabricated-measurement shape (F12)"
                     % (len(samples), samples[0]))

    n = receipt.get("n")
    if n is not None and n != len(samples):
        _err(errors, "n=%r disagrees with len(samples_ms)=%d" % (n, len(samples)))


def _check_discards(receipt, errors):
    """runs_discarded is legal, but never unexplained."""
    discarded = receipt.get("runs_discarded")
    if discarded is None:
        return
    if not isinstance(discarded, int) or discarded < 0:
        _err(errors, "runs_discarded: must be an integer >= 0")
    elif discarded > 0 and receipt.get("discard_reason") not in DISCARD_REASONS:
        _err(errors, "runs_discarded=%d but discard_reason=%r is not one of %s"
                     % (discarded, receipt.get("discard_reason"),
                        list(DISCARD_REASONS)))


def _check_cross_class(receipt, errors):
    """THE STRUCTURAL RULE: a cross-class run cannot arm a threshold."""
    if receipt.get("comparability") != "cross-class-existence-only":
        return
    if "threshold" in receipt:
        _err(errors, "comparability=cross-class-existence-only carries a "
                     "threshold object -- a cross-class run cannot arm a "
                     "threshold. This is the born-disarmed failure: the row "
                     "goes EXISTENCE-ONLY and the threshold never arms.")


def validate(receipt):
    """Return a list of human-readable errors; empty means valid."""
    # `_expect` is the case-table annotation, not receipt content.
    receipt = {k: v for k, v in receipt.items() if k != "_expect"}
    errors = []

    prov = receipt.get("provenance")
    if not isinstance(prov, dict):
        _err(errors, "provenance: missing -- a receipt with no provenance is an "
                     "anonymous number, not evidence")
        return errors

    _check_provenance(prov, errors)
    _check_samples(receipt, errors)
    _check_discards(receipt, errors)
    _check_cross_class(receipt, errors)
    return errors


def _bench_of(receipt):
    """The bench block, or None. A HOST receipt nests it under `bench`; a bare
    bench receipt IS the block."""
    if isinstance(receipt.get("bench"), dict):
        return receipt["bench"]
    if "samples_ms" in receipt:
        return receipt
    return None


def _validate_one(path):
    """Validate a single receipt file. Returns (rc, lines_to_print)."""
    try:
        with open(path, encoding="utf-8") as handle:
            receipt = json.load(handle)
    except (OSError, ValueError) as exc:
        return 2, ["%s: cannot read: %s" % (path, exc)]
    errors = validate(receipt)
    if errors:
        return 1, ["FAIL %s: %s" % (path, e) for e in errors]
    return 0, ["ok   %s" % path]


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _mode_has_bench(path):
    """Exit 0 iff a bench block is PRESENT. Says nothing about validity."""
    try:
        return 0 if _bench_of(_load(path)) is not None else 1
    except (OSError, ValueError):
        return 2


def _mode_bench(path):
    """Validate the bench block of a host receipt."""
    try:
        bench = _bench_of(_load(path))
    except (OSError, ValueError) as exc:
        sys.stderr.write("%s: cannot read: %s\n" % (path, exc))
        return 2
    if bench is None:
        sys.stderr.write("%s: no bench block\n" % path)
        return 1
    errors = validate(bench)
    for e in errors:
        print("FAIL %s: %s" % (path, e))
    return 1 if errors else 0


def _mode_bench_median(path):
    """Print the median of the bench block's raw samples."""
    try:
        bench = _bench_of(_load(path))
    except (OSError, ValueError):
        return 2
    if not bench or not bench.get("samples_ms"):
        return 1
    print(round(statistics.median(bench["samples_ms"]), 3))
    return 0



# ===========================================================================
# PARITY LANES (#2696). A bench block says how fast one runtime is. A parity
# lane says how fast it is RELATIVE TO A COMPARATOR, which is the claim a
# release actually makes -- and the claim that has never once been checked
# against the artifact users receive.
#
# On 2026-08-24 every performance figure in this repo came from a local
# `--features cuda` build. The published `cargo install aprender` binary has no
# CUDA linked at all, accepts `--gpu` in silence, and decodes at 15.7 tok/s
# against llama.cpp's 158.9 -- 0.099x, with 7.5 SECONDS to first token. Nothing
# was wrong with the kernels. Nothing had looked at the artifact.
#
# Four rules, each closing one way that number could have been reported as fine.
# ===========================================================================

PARITY_LANE_REQUIRED = ("lane", "subject", "comparator", "ratio_decode", "verdict")
PARITY_SIDE_REQUIRED = ("provenance", "decode_tok_per_sec")
INSTALL_SOURCES = ("crates.io", "local-build", "release-artifact")
RATIO_TOLERANCE = 0.01


def _median_of(side, key, label, errors):
    """Raw samples or nothing. A side that ships only a summary has already
    discarded what a bootstrap would need, and cannot be re-derived."""
    values = side.get(key)
    if values is None:
        _err(errors, "%s.%s: missing -- a parity lane carries RAW SAMPLES on "
                     "both sides, never a pre-computed summary" % (label, key))
        return None
    if not isinstance(values, list) or not values:
        _err(errors, "%s.%s: must be a non-empty list of samples" % (label, key))
        return None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        _err(errors, "%s.%s: contains a non-numeric entry" % (label, key))
        return None
    if any(v <= 0 for v in values):
        _err(errors, "%s.%s: contains a non-positive rate -- a throughput of "
                     "zero or less is not a measurement" % (label, key))
        return None
    return statistics.median(values)


def _check_install_source(side, label, errors):
    """RULE 4 -- WHICH ARTIFACT. #2696 is exactly the case where a local build
    and the published binary are different runtimes by a factor of 6.6x. A lane
    that does not say which one it measured cannot be read."""
    src = side.get("install_source")
    if src is None:
        _err(errors, "%s.install_source: missing -- a lane that does not say "
                     "whether it measured the PUBLISHED artifact or a local "
                     "build is unreadable (#2696)" % label)
    elif src not in INSTALL_SOURCES:
        _err(errors, "%s.install_source: %r not in %s"
                     % (label, src, list(INSTALL_SOURCES)))


def _side_provenance(side, label, errors):
    """The provenance object, validated, or None."""
    prov = side.get("provenance")
    if isinstance(prov, dict):
        _check_provenance(prov, errors)
        return prov
    if prov is not None:
        _err(errors, "%s.provenance: must be an object" % label)
    return None


def _check_parity_side(side, label, errors, require_install_source):
    """Each side names its binary AND the dispatch path that binary took."""
    if not isinstance(side, dict):
        _err(errors, "%s: missing" % label)
        return None
    for key in PARITY_SIDE_REQUIRED:
        if key not in side:
            _err(errors, "%s.%s: missing (required)" % (label, key))
    if require_install_source:
        _check_install_source(side, label, errors)
    return _side_provenance(side, label, errors)


def _check_comparator_pin(comp, label, errors):
    """RULE 3 -- THE COMPARATOR IS PINNED. An unpinned denominator makes the
    ratio meaningless across time: it moves silently between releases while the
    receipt claims a fixed baseline."""
    build = comp.get("build_commit")
    if not build:
        _err(errors, "%s.comparator.build_commit: missing -- an unpinned "
                     "comparator makes the ratio meaningless across time" % label)
    elif build == "UNPINNED":
        _err(errors, "%s.comparator.build_commit=UNPINNED -- usable for an "
                     "existence-only row, never for a ratio" % label)


def _check_class_pair(lane, subj_class, comp_class, label, errors):
    """RULE 1 -- SAME CLASS, OR NO VERDICT.

    Comparing a cpu-class apr against a cuda-class comparator is the
    fabricated-14x-regression shape, and it is ALSO how #2696's 0.099x would
    look if reported as a kernel defect. The published binary takes the cpu
    path even when handed --gpu, so its lane must either use a cpu-class
    comparator or decline to render a verdict. Returns True if cross-class.
    """
    if subj_class is None or comp_class is None or subj_class == comp_class:
        return False
    if lane.get("comparability") != "cross-class-existence-only":
        _err(errors, "%s: subject compute_class=%s vs comparator=%s is a "
                     "CROSS-CLASS comparison and must be marked "
                     "comparability=cross-class-existence-only"
                     % (label, subj_class, comp_class))
    if lane.get("verdict") == "PASS":
        _err(errors, "%s: a cross-class lane cannot render verdict=PASS -- it "
                     "is not a comparison" % label)
    if "floor" in lane:
        _err(errors, "%s: a cross-class lane carries a floor -- this is the "
                     "born-disarmed shape, made unwriteable" % label)
    return True


def _check_stated_ratio(lane, derived, label, errors):
    """RULE 2 -- THE RATIO IS DERIVED, NOT ASSERTED. A stated ratio that does
    not follow from the samples beside it is a fabricated measurement (F12)
    wearing the shape of a computed one."""
    stated = lane.get("ratio_decode")
    if stated is None:
        return
    if isinstance(stated, bool) or not isinstance(stated, (int, float)):
        _err(errors, "%s.ratio_decode: must be a number" % label)
        return
    if abs(derived - stated) > RATIO_TOLERANCE * max(derived, 1e-9):
        _err(errors, "%s.ratio_decode=%r does not follow from the samples "
                     "(derived %.4f) -- a stated ratio that its own samples do "
                     "not produce is a fabricated measurement"
                     % (label, stated, derived))


def _check_verdict(lane, derived, cross, label, errors):
    """RULE 5 -- THE VERDICT FOLLOWS FROM THE FLOOR. A PASS below the declared
    floor is the gate lying about its own rule."""
    floor = lane.get("floor")
    if isinstance(floor, bool) or not isinstance(floor, (int, float)):
        if not cross and floor is None:
            _err(errors, "%s.floor: missing -- a same-class lane with no floor "
                         "records a number nothing can fail" % label)
        return
    expected = "PASS" if derived >= floor else "FAIL"
    verdict = lane.get("verdict")
    if verdict in ("PASS", "FAIL") and verdict != expected:
        _err(errors, "%s.verdict=%s but ratio %.4f against floor %.4f requires "
                     "%s" % (label, verdict, derived, floor, expected))


def _check_parity_lane(lane, index, errors):
    label = "parity.lanes[%d]" % index
    if not isinstance(lane, dict):
        _err(errors, "%s: must be an object" % label)
        return
    for key in PARITY_LANE_REQUIRED:
        if key not in lane:
            _err(errors, "%s.%s: missing (required)" % (label, key))

    subj_prov = _check_parity_side(lane.get("subject"), label + ".subject",
                                   errors, require_install_source=True)
    comp_prov = _check_parity_side(lane.get("comparator"), label + ".comparator",
                                   errors, require_install_source=False)
    _check_comparator_pin(lane.get("comparator") or {}, label, errors)
    cross = _check_class_pair(lane, (subj_prov or {}).get("compute_class"),
                              (comp_prov or {}).get("compute_class"), label, errors)

    subj_med = _median_of(lane.get("subject") or {}, "decode_tok_per_sec",
                          label + ".subject", errors)
    comp_med = _median_of(lane.get("comparator") or {}, "decode_tok_per_sec",
                          label + ".comparator", errors)
    if subj_med is None or comp_med is None or comp_med <= 0:
        return
    derived = subj_med / comp_med
    _check_stated_ratio(lane, derived, label, errors)
    if not lane.get("bands"):
        _check_verdict(lane, derived, cross, label, errors)
    if not cross:
        _check_bands(lane, lane.get("floor", 0.80), lane.get("ceiling", 1.50),
                     lane.get("declared_bands") or [], errors)


# ===========================================================================
# BANDS. A parity claim at one concurrency is not a parity claim.
#
# Measured 2026-08-25 on lambda (RTX 4090, comparator 39173bcac), the two
# metrics move in OPPOSITE directions as concurrency rises:
#
#   band    aggregate ratio    per-user decode ratio
#   c=1          0.534x               0.587x
#   c=4          0.231x               0.923x
#   c=8          0.169x               1.352x
#   c=16         0.097x               1.554x
#
# apr's aggregate is flat at ~110 tok/s at every band -- it serialises rather
# than batching -- so per-user decode RISES simply because each request gets the
# whole GPU in turn. A gate reading only per-user decode would score c=16 a
# comfortable PASS while a sixteen-user deployment ran at a tenth of llama.cpp.
# That is the cannot-fail shape, and it is why both metrics are required here.
# ===========================================================================

BAND_METRICS = ("aggregate_tok_per_sec", "decode_tok_per_sec")


def _band_ratio(band, metric, errors, label):
    """Ratio DERIVED from this band's own samples, or None."""
    subj = _median_of(band.get("subject") or {}, metric, label + ".subject", errors)
    comp = _median_of(band.get("comparator") or {}, metric, label + ".comparator", errors)
    if subj is None or comp is None or comp <= 0:
        return None
    return subj / comp


def _band_metric(band, metric, floor, ceiling, label, failed, errors):
    """One metric of one band: derive the ratio, check any stated one, and
    record whether it sits inside the band."""
    ratio = _band_ratio(band, metric, errors, label + "." + metric)
    if ratio is None:
        return None
    stated = band.get("ratio_" + metric)
    if isinstance(stated, (int, float)) and not isinstance(stated, bool):
        if abs(ratio - stated) > RATIO_TOLERANCE * max(ratio, 1e-9):
            _err(errors, "%s.ratio_%s=%r does not follow from its samples "
                         "(derived %.4f)" % (label, metric, stated, ratio))
    if ratio < floor or ratio > ceiling:
        failed.append("%s %.4fx" % (metric, ratio))
    return ratio


def _check_one_band(band, index, floor, ceiling, errors):
    """One concurrency level, both metrics, verdict derived not asserted."""
    label = "band[%s]" % band.get("concurrency", "?")
    if not isinstance(band, dict):
        _err(errors, "parity.bands[%d]: must be an object" % index)
        return None
    if not isinstance(band.get("concurrency"), int) or band["concurrency"] < 1:
        _err(errors, "%s.concurrency: must be a positive integer" % label)
        return None

    failed = []
    for metric in BAND_METRICS:
        ratio = _band_metric(band, metric, floor, ceiling, label, failed, errors)
        if ratio is None:
            return None

    expected = "PASS" if not failed else "FAIL"
    verdict = band.get("verdict")
    if verdict in ("PASS", "FAIL") and verdict != expected:
        _err(errors, "%s.verdict=%s but %s requires %s"
                     % (label, verdict, failed or "both metrics in band", expected))
    return expected


def _check_bands(lane, floor, ceiling, declared, errors):
    """Every declared band measured, both metrics in band, or the lane FAILS."""
    bands = lane.get("bands")
    if not isinstance(bands, list) or not bands:
        _err(errors, "parity lane carries no `bands` -- a parity claim at one "
                     "concurrency is not a parity claim (see the table above)")
        return
    seen, outcomes = _walk_bands(bands, floor, ceiling, errors)
    missing = [c for c in declared if c not in seen]
    if missing:
        _err(errors, "bands %s are declared in the protocol and absent from the "
                     "receipt -- an unmeasured band is not a passing band" % missing)
    if "FAIL" in outcomes and lane.get("verdict") == "PASS":
        _err(errors, "a lane cannot render verdict=PASS while a band FAILS")


def _walk_bands(bands, floor, ceiling, errors):
    """Check each band; return (concurrencies seen, verdicts)."""
    seen, outcomes = [], []
    for i, band in enumerate(bands):
        outcome = _check_one_band(band, i, floor, ceiling, errors)
        if isinstance(band, dict) and isinstance(band.get("concurrency"), int):
            seen.append(band["concurrency"])
        if outcome:
            outcomes.append(outcome)
    return seen, outcomes


def validate_parity(block):
    """Validate a parity block. Returns a list of errors; empty means valid."""
    errors = []
    if not isinstance(block, dict):
        return ["parity: must be an object"]
    for key in ("instrument", "protocol_ref", "model"):
        if not block.get(key):
            _err(errors, "parity.%s: missing (required)" % key)
    lanes = block.get("lanes")
    if not isinstance(lanes, list) or not lanes:
        # VACUITY: a parity block with no lanes passes every rule above by
        # having nothing to check, which is how a green gate covers nothing.
        _err(errors, "parity.lanes: missing or empty -- a parity block with no "
                     "lanes is vacuously clean")
        return errors
    for i, lane in enumerate(lanes):
        _check_parity_lane(lane, i, errors)
    return errors


def _parity_of(receipt):
    if isinstance(receipt.get("parity"), dict):
        return receipt["parity"]
    if "lanes" in receipt:
        return receipt
    return None


def _mode_has_parity(path):
    """Exit 0 iff a parity block is PRESENT. Says nothing about validity."""
    try:
        return 0 if _parity_of(_load(path)) is not None else 1
    except (OSError, ValueError):
        return 2


def _mode_parity(path):
    try:
        block = _parity_of(_load(path))
    except (OSError, ValueError) as exc:
        sys.stderr.write("%s: cannot read: %s\n" % (path, exc))
        return 2
    if block is None:
        sys.stderr.write("%s: no parity block\n" % path)
        return 1
    errors = validate_parity({k: v for k, v in block.items() if k != "_expect"})
    for e in errors:
        print("FAIL %s: %s" % (path, e))
    return 1 if errors else 0


def _mode_parity_ratio(path):
    """Print `lane ratio verdict` per lane, ratio DERIVED from the samples."""
    try:
        block = _parity_of(_load(path))
    except (OSError, ValueError):
        return 2
    if not block or not isinstance(block.get("lanes"), list):
        return 1
    for lane in block["lanes"]:
        try:
            s = statistics.median(lane["subject"]["decode_tok_per_sec"])
            c = statistics.median(lane["comparator"]["decode_tok_per_sec"])
            print("%s %.4f %s" % (lane.get("lane", "?"), s / c,
                                  lane.get("verdict", "?")))
        except (KeyError, TypeError, ZeroDivisionError, statistics.StatisticsError):
            print("%s ERROR ERROR" % lane.get("lane", "?"))
            return 1
    return 0

MODES = {
    "--has-bench": _mode_has_bench,
    "--bench": _mode_bench,
    "--bench-median": _mode_bench_median,
    "--has-parity": _mode_has_parity,
    "--parity": _mode_parity,
    "--parity-ratio": _mode_parity_ratio,
}


def main(argv):
    if len(argv) >= 3 and argv[1] in MODES:
        return MODES[argv[1]](argv[2])
    if len(argv) < 2:
        sys.stderr.write("usage: bench_receipt.py [--bench|--has-bench|"
                         "--bench-median|--parity|--has-parity|"
                         "--parity-ratio] <receipt.json> [...]\n")
        return 2
    rc = 0
    for path in argv[1:]:
        one_rc, lines = _validate_one(path)
        for line in lines:
            print(line)
        if one_rc == 2:
            return 2
        rc = rc or one_rc
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
