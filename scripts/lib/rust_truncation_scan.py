#!/usr/bin/env python3
"""#3916: the Rust half of #3904's surface.

#3904 refuses any NEW silent truncation of a value a human reads later. Its scan matches
`[:N]` slice syntax, which is Python. Rust truncates with `.chars().take(N)`, and #3904 is
blind to all of it -- a blindness found the hard way, when a `.chars().take(100)` in the
golden gate's own failure reason cost a diagnosis during #3914.

SCOPE IS MEASURED, NOT PREFERRED. Of the four candidate idioms in this tree:

    .chars().take(N)   IN   -- 64 sites, and nearly every one truncates a string into a
                              preview/snippet/display. High precision.
    .get(..N)          IN   -- 22 sites, mixed: `commit.get(..12)` is an id prefix a human
                              reads, `all_logits.get(..20)` is a tensor slice. Needs the
                              stringy filter to be worth anything.
    .truncate(N)       OUT  -- 26 sites, and the receivers are change_points, concepts,
                              gaps, examples, indexed, interfaces, guards, bytes, data,
                              results, top5. All collections. `String::truncate` and
                              `Vec::truncate` are syntactically identical and only the TYPE
                              separates them, so including this would be almost entirely
                              false positives. Recorded so the next person does not close
                              the "gap" without re-measuring it.
    .iter().take(N)    OUT  -- the bulk of the 584 raw `.take(N)` hits. An iterator limit
                              walking a top-N list does not decapitate anything.

The key scheme and the amnesty shape are #3904's, deliberately: a third bespoke baseline
format would be its own debt. Key = <file>|<sha1-8 of the normalized line>|<n-th identical
line>, which survives line shifts (keying by file:line produces churn, and churn trains a
reviewer to regenerate the baseline, laundering whatever real violation shares the commit).
"""
import hashlib
import os
import re
import sys

IDIOM = re.compile(r"\.chars\(\)\.take\(\s*[0-9]{1,4}\s*\)|\.get\(\s*\.\.\s*[0-9]{1,4}\s*\)")
GET_ONLY = re.compile(r"\.get\(\s*\.\.\s*[0-9]{1,4}\s*\)")
CHARS = re.compile(r"\.chars\(\)\.take\(")
# `.get(..N)` is only interesting on something a human reads. `format!` carries its bang
# deliberately: `let format = detect_format(data.get(..8)...)` must NOT match on the word.
STRINGY_GET = re.compile(
    r"format!|println!|eprintln!|write!|"
    r"\b(commit|sha\w*|hash|\w*key|text|\w*name|version|signature|prefix|snippet|preview|message|reason|label)\b"
)
SKIP_FILE = re.compile(r"/target/|rust_truncation_scan\.py|rust_truncation_cases\.txt")
EXTS = (".rs",)


def is_comment(line):
    t = line.lstrip()
    return t.startswith(("//", "/*", "*")) and not t.startswith("*/")


# Two shapes the first cut selected wrongly, both found by auditing the tree rather than
# by imagining cases. A PREDICATE consumes the chars and displays nothing; CONSTRUCTION
# builds the string from a repeat, so the cap is the content's size and not a cut.
PREDICATE = re.compile(r"\.take\(\s*[0-9]{1,4}\s*\)\s*\.(all|any|position|find|count|filter)\b")
CONSTRUCTED = re.compile(r"\.repeat\(|std::iter::repeat\b")


def selects(line):
    """Does this line truncate a string a human reads later?"""
    if is_comment(line):
        return False
    if not IDIOM.search(line):
        return False
    if PREDICATE.search(line) or CONSTRUCTED.search(line):
        return False
    if CHARS.search(line):
        return True
    return bool(STRINGY_GET.search(line))


def normalize(line):
    return " ".join(line.split())


def scan(root):
    rows, seen = [], {}
    for sub in ("crates", "scripts"):
        base = os.path.join(root, sub)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d != "target"]
            for fn in sorted(filenames):
                if not fn.endswith(EXTS):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                if SKIP_FILE.search(rel):
                    continue
                try:
                    lines = open(os.path.join(root, rel), errors="replace").readlines()
                except OSError:
                    continue
                for n, line in enumerate(lines, 1):
                    if not selects(line):
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
