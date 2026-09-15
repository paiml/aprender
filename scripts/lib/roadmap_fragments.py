#!/usr/bin/env python3
"""roadmap_fragments.py — docs/roadmaps/roadmap.yaml as a GENERATED artifact.

THE DEFECT THIS EXISTS FOR (#3296, five-whys #3294). Every PR in this repo
writes docs/roadmaps/roadmap.yaml, so the Amdahl serial fraction on the merge
path is 1: N pull requests contend on one file however disjoint their code is.
`.gitattributes` declares `merge=roadmap`, but that driver is CUSTOM and is
registered per-invocation by scripts/ci_resolve_dirty.sh, so GitHub's
merge-queue server -- which has no such config -- never runs it and falls back
to a plain text merge. Measured on the nine-PR batch #3295: the textual merge
placed PMAT-3226 after PMAT-3229 and check_roadmap_sorted.sh went RED, while
all nine inputs were individually sorted and green.

`union` is built-in and therefore DOES apply server-side, but it is correct
only for line-delimited order-insensitive data (impl-estimates.jsonl, #3256).
roadmap.yaml needs SORTED insertion. Two files, two policies.

THE SHAPE. A ticket writes exactly one file:

    docs/roadmaps/entries/<ID>.yaml     <- one entry block, verbatim
    docs/roadmaps/roadmap.yaml          <- base + entries, GENERATED

Unique filename by construction => pull requests are pairwise disjoint on the
roadmap => the conflict rate is 0 by proof rather than by luck.

ORDERING IS THE AGGREGATOR'S JOB, not the merge's. An entry is inserted at the
slot check_roadmap_sorted.sh would demand: within its own id-prefix, numerals
ascending, in the slots that prefix already occupies. Legacy free-form ids are
exempt and keep their position -- the same rule roadmap_merge.py enforces, and
it is imported from there rather than restated, so the two cannot drift.

    python3 scripts/lib/roadmap_fragments.py aggregate [--write]
    python3 scripts/lib/roadmap_fragments.py --selftest

Exit: 0 ok - 1 a violation (or a failing selftest row) - 2 usage/vacuity.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from scripts.lib.roadmap_diff import split_entries  # noqa: E402
from scripts.lib.roadmap_merge import parse_id  # noqa: E402

ROADMAP = os.path.join(REPO_ROOT, "docs", "roadmaps", "roadmap.yaml")
ENTRIES = os.path.join(REPO_ROOT, "docs", "roadmaps", "entries")


def fragment_path(eid, entries_dir=ENTRIES):
    return os.path.join(entries_dir, eid + ".yaml")


def read_fragments(entries_dir=ENTRIES):
    """-> [(id, block)] sorted by filename. Absent dir is EMPTY, not an error:
    before the first fragment lands the aggregate is just the base."""
    if not os.path.isdir(entries_dir):
        return []
    out = []
    for name in sorted(os.listdir(entries_dir)):
        if not name.endswith(".yaml"):
            continue
        eid = name[: -len(".yaml")]
        with open(os.path.join(entries_dir, name), encoding="utf-8") as fh:
            out.append((eid, fh.read()))
    return out


def insertion_index(base_entries, eid):
    """The slot check_roadmap_sorted.sh would demand for `eid`.

    Within the id's own prefix: before the first entry of that prefix whose
    numeral is greater. A prefix that does not occur yet, or a legacy id with
    no (prefix, numeral) shape, appends -- appending is always sorted when
    nothing of that prefix precedes it."""
    parsed = parse_id(eid)
    if parsed is None:
        return len(base_entries)
    prefix, numeral = parsed
    for i, (other, _) in enumerate(base_entries):
        p = parse_id(other)
        if p is None or p[0] != prefix:
            continue
        if p[1] > numeral:
            return i
    return len(base_entries)


def aggregate(base_text, fragments):
    """base + fragments, each at its sorted slot. Duplicate id is exit 1."""
    preamble, entries = split_entries(base_text)
    seen = {eid for eid, _ in entries}
    for eid, block in fragments:
        if eid in seen:
            raise ValueError(
                "duplicate id: %s is in the base AND in entries/%s.yaml -- a "
                "fragment adds an entry, it never redefines one" % (eid, eid)
            )
        seen.add(eid)
        entries.insert(insertion_index(entries, eid), (eid, block))
    return preamble + "".join(block for _, block in entries)


# ---------------------------------------------------------------- self-test

BASE = (
    "roadmap_version: '1.0'\nroadmap:\n"
    "- id: PMAT-100\n  title: a\n"
    "- id: PMAT-300\n  title: c\n"
    "- id: LEGACY-THING\n  title: legacy\n"
)
ROWS = []


def row(name, ok, detail=""):
    ROWS.append((name, ok, detail))
    print("%-4s %-52s %s" % ("ok" if ok else "FAIL", name, detail))


def _ids(text):
    return [eid for eid, _ in split_entries(text)[1]]


def selftest():
    # 1. THE LOSSLESS ROW: no fragments must reproduce the base byte for byte.
    out = aggregate(BASE, [])
    row("no fragments -> base is byte-identical", out == BASE,
        "" if out == BASE else "differs")

    # 2-4. placement
    out = aggregate(BASE, [("PMAT-200", "- id: PMAT-200\n  title: b\n")])
    row("numeral between two -> sorted slot",
        _ids(out) == ["PMAT-100", "PMAT-200", "PMAT-300", "LEGACY-THING"], str(_ids(out)))
    out = aggregate(BASE, [("PMAT-400", "- id: PMAT-400\n  title: d\n")])
    row("numeral after all of its prefix -> appended",
        _ids(out)[-1] == "PMAT-400", str(_ids(out)))
    out = aggregate(BASE, [("APEX-1", "- id: APEX-1\n  title: new prefix\n")])
    row("unseen prefix -> appended", _ids(out)[-1] == "APEX-1", str(_ids(out)))
    out = aggregate(BASE, [("FREEFORM", "- id: FREEFORM\n  title: legacy\n")])
    row("legacy id (no numeral) -> appended", _ids(out)[-1] == "FREEFORM", str(_ids(out)))

    # 5. numeric, not lexical: 100 < 90 as strings, 90 < 100 as numbers.
    out = aggregate(BASE, [("PMAT-90", "- id: PMAT-90\n  title: ninety\n")])
    row("numeral compared NUMERICALLY (PMAT-90 before PMAT-100)",
        _ids(out)[0] == "PMAT-90", str(_ids(out)))

    # 6. a fragment never redefines a base entry
    try:
        aggregate(BASE, [("PMAT-100", "- id: PMAT-100\n  title: clash\n")])
        row("duplicate id refused", False, "no error raised")
    except ValueError as e:
        row("duplicate id refused", "duplicate id" in str(e), str(e)[:40])

    # 7. several fragments at once stay mutually sorted
    out = aggregate(BASE, [
        ("PMAT-150", "- id: PMAT-150\n  title: x\n"),
        ("PMAT-250", "- id: PMAT-250\n  title: y\n"),
    ])
    row("two fragments -> both at their own slots",
        _ids(out) == ["PMAT-100", "PMAT-150", "PMAT-250", "PMAT-300", "LEGACY-THING"],
        str(_ids(out)))

    # 8. MUTATION: if placement were append-only, row 2 must go RED. A guard
    #    whose failure mode is untested is decoration (#3294).
    saved = globals()["insertion_index"]
    globals()["insertion_index"] = lambda entries, eid: len(entries)
    try:
        mutated = aggregate(BASE, [("PMAT-200", "- id: PMAT-200\n  title: b\n")])
        caught = _ids(mutated) != ["PMAT-100", "PMAT-200", "PMAT-300", "LEGACY-THING"]
    finally:
        globals()["insertion_index"] = saved
    row("mutation: append-only placement is CAUGHT", caught,
        "" if caught else "mutant survived -- row 2 proves nothing")

    bad = sum(1 for _, ok, _ in ROWS if not ok)
    print("\n%d row(s), %d red" % (len(ROWS), bad))
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="roadmap_fragments.py")
    ap.add_argument("cmd", nargs="?", choices=["aggregate"])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd != "aggregate":
        ap.print_usage(sys.stderr)
        return 2
    with open(ROADMAP, encoding="utf-8") as fh:
        base = fh.read()
    frags = read_fragments()
    try:
        out = aggregate(base, frags)
    except ValueError as e:
        sys.stderr.write("FAIL %s\n" % e)
        return 1
    if a.write:
        with open(ROADMAP, "w", encoding="utf-8") as fh:
            fh.write(out)
        sys.stderr.write("aggregate: %d base + %d fragment(s) -> %s\n"
                         % (len(split_entries(base)[1]), len(frags), ROADMAP))
    else:
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
