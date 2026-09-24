#!/usr/bin/env python3
"""The candidate set for #3904's silent-truncation guard — ONE definition, used by both
the enforce path and the self-test, so a fixture is never judged by a different scan
than the tree is.

KEYING. The first cut of this keyed rows by `file:line`. Inserting a six-line helper at
the top of a file then renumbered every row below it, and the guard reported one NEW
violation and one STALE fix for a line nobody had touched. That is not a cosmetic
annoyance: a reviewer facing that churn regenerates the baseline, and a regeneration
launders any genuine new truncation sitting in the same commit.

So the key is `<file>|<sha1-8 of the normalized line>|<n-th identical line in that file>`.
It survives line shifts, changes the moment the line's content changes, and the
occurrence index keeps two identical lines in one file from collapsing into a single row
(the collapse that hid two claims in #3867). The line NUMBER is reported, never keyed.

A consequence worth having: because the key hashes the text, the text recorded in the
baseline is tamper-evident. Editing a row's text to make it look innocent breaks its key.
"""
import hashlib
import os
import re
import subprocess
import sys

# Head AND tail, literal bound OR variable bound. The first cut matched only a DIGIT
# bound, so `msg[:limit]` -- the obvious way to write the same defect -- was invisible.
# Widening cost exactly three rows across the whole tree, all of them accounted for in
# the baseline, which is what made it an easy call rather than a judgement one.
SLICE = re.compile(
    r"\[:\s*[A-Za-z0-9_.]{1,24}\]"           # head:  x[:80]  x[:limit]
    r"|\[\s*-[A-Za-z0-9_.]{1,24}\s*:\s*\]"   # tail:  x[-5:]  x[-n:]
)
# The "a human reads this later" net. Deliberately wide: a numeric or id-prefix slice
# caught here is cheap to classify once, and a display slice missed here is invisible
# forever.
STRINGY = re.compile(
    r"""(f"|f'|print\(|format!|str\(|\.stderr|\.stdout|reason|message|msg|why|detail|%s|\{\})"""
)
SKIP = re.compile(r"coverage_report|/target/|_baseline\.txt|truncation_scan\.py")
# THE IDENTIFIER-PREFIX CLASS (#4046, cop ruling A'). `sha[:12]` is not a message cut short:
# it is the conventional short form of an identifier, it loses nothing a reader needs, and
# the guard's own FAIL text always said it belongs in the baseline "WITH its class". But the
# baseline ratchet (`set`, and even `set-aperture`) refuses any line the branch WROTE, so
# for new code the class has to live here, argued once, not appended row by row.
# A line is exempt ONLY if EVERY slice on it is a head slice with a LITERAL width of 1-16
# applied to a NAME that says it is an identifier. Everything else stays a finding:
#   - a display slice of any other name              (msg[:80], got[:300])
#   - a variable-width id slice                      (sha[:n])
#   - an id slice wider than 16                      (sha[:40] is the whole id, or prose)
#   - a tail slice, even of an id                    (sha[-8:])
#   - a line mixing an id slice with a display slice (the display slice still counts)
ID_NAME = re.compile(r"(sha|hash|digest|revision)[a-z0-9_]*$|_id$", re.I)
ANY_SLICE = re.compile(r"\[\s*-?[A-Za-z0-9_.]*\s*:\s*-?[A-Za-z0-9_.]*\s*\]")


def id_prefix_only(line):
    """True iff every slice on the line is an identifier prefix of literal width 1-16."""
    slices = list(ANY_SLICE.finditer(line))
    if not slices:
        return False
    for m in slices:
        head = line[:m.start()]
        # the name is the last identifier (or string key) before the slice
        nm = re.search(r"""([A-Za-z_][A-Za-z0-9_]*)["']?\]?\s*$""", head)
        w = re.fullmatch(r"\[:\s*(\d{1,2})\s*\]", m.group(0).replace(" ", ""))
        if not nm or not w or not ID_NAME.search(nm.group(1)) or not 1 <= int(w.group(1)) <= 16:
            return False
    return True
EXTS = (".py", ".sh", ".rs")


def normalize(line):
    return " ".join(line.split())


def scan(root):
    rows = []
    seen = {}
    for sub in ("scripts", "crates"):
        base = os.path.join(root, sub)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d != "target"]
            for fn in sorted(filenames):
                if not fn.endswith(EXTS):
                    continue
                path = os.path.join(dirpath, fn)
                rel = os.path.relpath(path, root)
                if SKIP.search(rel):
                    continue
                try:
                    with open(path, "r", errors="replace") as fh:
                        lines = fh.readlines()
                except OSError:
                    continue
                for n, line in enumerate(lines, 1):
                    if not SLICE.search(line) or not STRINGY.search(line):
                        continue
                    if id_prefix_only(line):
                        continue
                    text = normalize(line)
                    h = hashlib.sha1(text.encode()).hexdigest()[:8]
                    idx = seen.get((rel, h), 0)
                    seen[(rel, h)] = idx + 1
                    rows.append(("%s|%s|%d" % (rel, h, idx), "%s:%d" % (rel, n), text))
    rows.sort()
    return rows


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    for key, loc, text in scan(root):
        sys.stdout.write("%s\t%s\t%s\n" % (key, loc, text))
