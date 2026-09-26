#!/usr/bin/env python3
"""x86-main's first step: the cheap reds, before the fat sections start.

car#4429 went red four times on classes a two-minute check sees: rustfmt, and a
FALSIFY-VERSION-005 placed before 004 (contract_data_integrity's test-ID gap, only
seen after the contracts crate compiles). This runs the ID-order rule without cargo.

    fail_fast.py ids [--base REF]   changed contracts must not gain a test-ID gap
    fail_fast.py self-test          the rule's case table (must go RED on each plant)

The rule is validate_contracts.rs::check_falsification_ids, verbatim: the prefix is
the first id less its last `-NNN`, and test i must be `<prefix>-<i:03>`. It is
differential: a contract judged gapped on the base stays the base's debt (the Rust
test ratchets those under its CEILING); a contract this change touches may not
become gapped.
"""
import subprocess
import sys

import yaml


def gap(text):
    """The first test-ID gap in a contract's YAML text, or None."""
    try:
        doc = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return None  # unparseable is pv's and the Rust test's to report
    fts = doc.get("falsification_tests") if isinstance(doc, dict) else None
    if not isinstance(fts, list) or not fts:
        return None
    ids = [str(ft.get("id", "")) if isinstance(ft, dict) else "" for ft in fts]
    parts = ids[0].rsplit("-", 1)
    if len(parts) != 2:
        return None
    for i, got in enumerate(ids):
        want = f"{parts[0]}-{i + 1:03}"
        if got != want:
            return f"expected {want}, found {got}"
    return None


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)


def ids(base):
    mb = git("merge-base", "HEAD", base)
    if mb.returncode != 0:
        print(f"fail_fast ids: no merge-base with {base}: {mb.stderr.strip()}")
        return 1
    mb = mb.stdout.strip()
    changed = git("diff", "--name-only", "--diff-filter=AM", mb, "HEAD", "--", "contracts/")
    paths = [p for p in changed.stdout.split() if p.endswith(".yaml")]
    red = 0
    for p in paths:
        head = gap(open(p, encoding="utf-8").read())
        if head is None:
            continue
        old = git("show", f"{mb}:{p}")
        if old.returncode == 0 and gap(old.stdout) is not None:
            continue
        print(f"RED {p}: test ID gap: {head} (contract_data_integrity)")
        red += 1
    print(f"fail_fast ids: {len(paths)} changed contract(s) vs {mb}, {red} new gap(s)")
    return 1 if red else 0


def self_test():
    def c(*i):
        return yaml.safe_dump({"falsification_tests": [{"id": x} for x in i]})

    rows = [
        ("in order", c("FALSIFY-VERSION-001", "FALSIFY-VERSION-002"), False),
        ("MUTANT 005 before 004", c("FALSIFY-VERSION-001", "FALSIFY-VERSION-002",
                                    "FALSIFY-VERSION-003", "FALSIFY-VERSION-005",
                                    "FALSIFY-VERSION-004"), True),
        ("MUTANT skipped id", c("FALSIFY-X-001", "FALSIFY-X-003"), True),
        ("MUTANT starts at 002", c("FALSIFY-X-002"), True),
        ("MUTANT other prefix", c("FALSIFY-X-001", "FALSIFY-Y-002"), True),
        ("no tests", yaml.safe_dump({"equations": {}}), False),
        ("no dash in first id", c("X"), False),
        ("unparseable", "a: [", False),
    ]
    bad = 0
    for name, text, want in rows:
        got = gap(text) is not None
        ok = got == want
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: gap={got} want={want}")
    print(f"fail_fast self-test: {len(rows) - bad}/{len(rows)}")
    return 1 if bad else 0


def main(argv):
    if argv[:1] == ["self-test"]:
        return self_test()
    if argv[:1] == ["ids"]:
        base = argv[argv.index("--base") + 1] if "--base" in argv else "origin/main"
        return ids(base)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
