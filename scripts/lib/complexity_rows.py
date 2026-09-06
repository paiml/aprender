#!/usr/bin/env python3
"""Turn `$PMAT analyze complexity --format json` output into ratchet rows.

Reads one or more analyser JSON documents (named as positional arguments) and
writes one line per Rust function that is over EITHER threshold:

    <path>::<function> <cyclomatic> <cognitive>

sorted, one row per key. The thresholds arrive in the environment
(CX_MAX_CYCLOMATIC / CX_MAX_COGNITIVE) rather than as flags so that this file
parses no argv of its own -- hand-rolled argument parsing is banned in this
repository, and a positional file list needs none.

WHY A KEY CAN COLLIDE, AND WHY THE MAX WINS.  `<path>::<function>` is not
unique: four keys in this repository name two functions each (two `impl` blocks
in one file exposing the same method name, e.g.
crates/aprender-core/src/classification/svc_rbf.rs::fit). The obvious fix --
putting the line number in the key -- is the one this repository has already
paid for: a file:line baseline DRIFTS the moment anything above it is edited,
and CI reads the drift as growth. So a colliding key carries the MAX of each
metric. The ratchet then reads "no function of this name in this file exceeds
the recorded number", which is strictly conservative: it can never hide a
violation, it can only decline to attribute one to the right twin.

WHY THIS FILE IS SEVEN SMALL FUNCTIONS AND NOT ONE LOOP.  The first draft was
one `main` at cognitive 45, and the repository's own pre-commit hook -- the
gate this whole guard exists to move into CI -- refused to commit it. Fixing
the code rather than the checker is the rule; this is that fix.
"""

import json
import os
import sys


def _threshold(name):
    raw = os.environ.get(name, "")
    if raw.strip().lstrip("-").isdigit():
        return int(raw)
    sys.stderr.write(
        "complexity_rows.py: %s must be set to an integer (got %r)\n" % (name, raw)
    )
    raise SystemExit(2)


def _rel_path(entry):
    rel = entry.get("path") or ""
    if rel.startswith("./"):
        return rel[2:]
    return rel


def _metrics(func):
    metrics = func.get("metrics") or {}
    return (
        int(metrics.get("cyclomatic") or 0),
        int(metrics.get("cognitive") or 0),
    )


def _offenders(entry, max_cyclomatic, max_cognitive):
    """Yield (key, cyclomatic, cognitive) for each over-threshold function."""
    rel = _rel_path(entry)
    if not rel.endswith(".rs"):
        return
    for func in entry.get("functions") or []:
        cyclomatic, cognitive = _metrics(func)
        if cyclomatic > max_cyclomatic or cognitive > max_cognitive:
            yield ("%s::%s" % (rel, func.get("name") or "?"), cyclomatic, cognitive)


def _record(worst, key, cyclomatic, cognitive):
    previous = worst.get(key)
    if previous is None:
        worst[key] = (cyclomatic, cognitive)
        return
    worst[key] = (max(previous[0], cyclomatic), max(previous[1], cognitive))


def _scan(doc, max_cyclomatic, max_cognitive, worst):
    """Fold one analyser document into `worst`; return (files, functions) seen."""
    functions = 0
    for entry in doc.get("files") or []:
        functions += len(entry.get("functions") or [])
        for row in _offenders(entry, max_cyclomatic, max_cognitive):
            _record(worst, row[0], row[1], row[2])
    return int(doc.get("files_analyzed") or 0), functions


def _load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main(paths):
    max_cyclomatic = _threshold("CX_MAX_CYCLOMATIC")
    max_cognitive = _threshold("CX_MAX_COGNITIVE")

    # A reader handed no documents would print nothing and look exactly like a
    # clean tree. That is the vacuity shape this repository keeps closing, so
    # it is a hard failure rather than an empty report.
    if not paths:
        sys.stderr.write(
            "complexity_rows.py: no analyser JSON document given. An empty scan is "
            "not a clean scan.\n"
        )
        return 2

    worst = {}
    analyzed = 0
    functions = 0
    for path in paths:
        files, seen = _scan(_load(path), max_cyclomatic, max_cognitive, worst)
        analyzed += files
        functions += seen

    for key in sorted(worst):
        sys.stdout.write("%s %d %d\n" % (key, worst[key][0], worst[key][1]))

    sys.stderr.write(
        "complexity_rows.py: %d file(s), %d function(s), %d over cyclomatic>%d "
        "or cognitive>%d\n"
        % (analyzed, functions, len(worst), max_cyclomatic, max_cognitive)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
