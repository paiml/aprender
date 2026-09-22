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
