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


def main(argv):
    if len(argv) < 2:
        sys.stderr.write("usage: bench_receipt.py <receipt.json> [...]\n")
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
