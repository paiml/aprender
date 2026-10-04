#!/usr/bin/env python3
"""Which moves of a dark copy never reached the compiled copy: for each dark file and each commit that moved one of
its units, the code lines that the commit added, that the dark file still holds at the pin, and that no compiled
holder of the file's changed units holds.

unit_history.py --tsv gives, per changed unit, the commits that moved the dark copy after the two copies last agreed
(dark_moves) and the compiled files that hold the unit's name (live). This reads each such commit's diff of the dark
file against its first parent, in the --source clone when the commit is tagged with one (`realizar:<sha>`), and
keeps the added lines that are code: not blank, not a comment, not bare punctuation. Lines match as text with their
whitespace collapsed, so a fix ported in other words reads as missing and a line that a holder has anywhere reads
as carried. The output is a reading list, not a verdict: each pair with missing lines needs its diff read against
the compiled copy.

Usage: dark_fix_lines.py --tsv PATH --pin REV [--source PREFIX=DIR ...] | --self-test
"""
import argparse
import csv
import os.path as pp
import subprocess
import sys

PUNCT = "{}()[];, "


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout


def norm(line):
    return " ".join(line.split())


def is_code(line):
    s = norm(line)
    return bool(s) and not s.startswith(("//", "/*", "*")) and s.strip(PUNCT) != ""


def added(diff):
    """The code lines that a unified diff adds, whitespace collapsed, in order, each once."""
    return list(dict.fromkeys(norm(x[1:]) for x in diff.splitlines()
                              if x.startswith("+") and not x.startswith("+++") and is_code(x[1:])))


def missing(adds, dark, live):
    """The added lines that the dark file still holds, and those of them that no compiled holder holds."""
    kept = [a for a in adds if a in dark]
    return kept, [a for a in kept if a not in live]


def locate(token, path, sources):
    """A dark_moves token as (repo, commit, path in that repo); a `tag:sha` token names a --source clone."""
    tag, sep, sha = token.rpartition(":")
    if not sep:
        return ".", token, path
    prefix, repo = sources[tag]
    if not path.startswith(prefix + "/"):
        raise ValueError(f"{path} is not under {prefix}")
    return repo, sha, path[len(prefix) + 1:]


def pairs(rows):
    """{(dark path, token): compiled holders} over the rows whose dark copy moved."""
    out = {}
    for r in rows:
        if r["dark_moves"] == "-":
            continue
        for t in r["dark_moves"].split(","):
            out.setdefault((r["path"], t), set()).update(p for p in r["live"].split(",") if p.endswith(".rs"))
    return out


def lines(pin, path):
    return {norm(x) for x in git(".", "show", f"{pin}:{path}").splitlines()}


def cases():
    diff = "+++ b/x.rs\n+let a = 1;\n+// c\n+}\n-let b = 2;\n+let a = 1;\n+  let  c =  3;\n"
    src = {"realizar": ("crates/aprender-serve", "/r")}
    rows = [{"path": "d.rs", "dark_moves": "-", "live": "x.rs"},
            {"path": "d.rs", "dark_moves": "a,realizar:b", "live": "p.rs,crates/q"},
            {"path": "d.rs", "dark_moves": "a", "live": "s.rs"}]
    try:
        locate("realizar:b", "crates/aprender-orchestrate/src/x.rs", src)
        outside = False
    except ValueError:
        outside = True
    return [
        ("1 not code: blank, comments, doc comments, block comments, bare punctuation",
         not any(map(is_code, ["", "   ", "// x", "/// d", "/* a */", " * b", "}", "});", "],", "{"]))),
        ("2 code: statements, a macro call, a one-line unsafe block",
         all(map(is_code, ["let x = 1;", "x?;", "contract_pre_x!();", "unsafe { a.b(&c)?; }", "} else {"]))),
        ("3 added: no header, no removed lines, no comments or punctuation, whitespace collapsed, each once",
         added(diff) == ["let a = 1;", "let c = 3;"]),
        ("4 missing: a line gone from the dark file is dropped, a line a holder has is carried",
         missing(["a", "b", "c"], {"a", "b"}, {"a"}) == (["a", "b"], ["b"])),
        ("5 locate: a bare sha is main's, a tagged one the clone's with the prefix cut",
         locate("5fa13fe8c6", "crates/aprender-serve/src/x.rs", src) == (".", "5fa13fe8c6",
                                                                        "crates/aprender-serve/src/x.rs")
         and locate("realizar:b", "crates/aprender-serve/src/x.rs", src) == ("/r", "b", "src/x.rs")),
        ("6 locate: a path outside the clone's prefix fails", outside),
        ("7 pairs: '-' rows skipped, tokens split, holders unioned, non-.rs holders dropped",
         pairs(rows) == {("d.rs", "a"): {"p.rs", "s.rs"}, ("d.rs", "realizar:b"): {"p.rs"}}),
    ]


def self_test():
    """Rule 7: the line, diff and token cases."""
    got = cases()
    for c, ok in got:
        print(("ok   " if ok else "FAIL ") + c)
    bad = sum(not ok for _, ok in got)
    print(f"{len(got) - bad}/{len(got)} cases pass")
    return 1 if bad else 0


def parse_sources(specs):
    """{clone tag: (prefix, clone dir)} from the PREFIX=DIR specs; a clone's tag is its directory's name."""
    sources = {}
    for spec in specs:
        prefix, repo = spec.split("=", 1)
        sources[pp.basename(pp.abspath(repo))] = (prefix.rstrip("/"), repo)
    return sources


def read_pairs(tsv):
    with open(tsv, newline="") as f:
        return pairs(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def check(pin, pair, live, sources):
    """(kept, missing, report lines) of one (dark path, token) pair whose dark file's units live holds."""
    d, t = pair
    repo, sha, path = locate(t, d, sources)
    kept, miss = missing(added(git(repo, "diff", "-U0", f"{sha}^", sha, "--", path)), lines(pin, d),
                         set().union(*(lines(pin, p) for p in sorted(live))))
    subj = git(repo, "log", "-1", "--format=%s", sha).strip()[:56]
    text = [f"  {d.split('/src/', 1)[-1]} {t} ({subj}): {len(kept)} kept, {len(miss)} in no compiled holder"]
    return kept, miss, text + [f"      - {m[:110]}" for m in miss[:3]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tsv")
    ap.add_argument("--pin")
    ap.add_argument("--source", action="append", default=[], metavar="PREFIX=DIR")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not (a.tsv and a.pin):
        ap.error("--tsv and --pin, or --self-test")
    pin = git(".", "rev-parse", "--short=10", a.pin).strip()
    sources, todo = parse_sources(a.source), read_pairs(a.tsv)
    out, gone, held, open_lines = [], 0, 0, 0
    for pair, live in sorted(todo.items()):
        kept, miss, text = check(pin, pair, live, sources)
        gone += not kept
        held += bool(kept) and not miss
        open_lines += len(miss)
        out += text
    print(f"dark fix lines at {pin}: {len(todo)} (dark file, commit) pairs; {gone} left no added line in the dark "
          f"file, {held} are carried in full by a compiled holder, {len(todo) - gone - held} have {open_lines} "
          f"lines that no compiled holder has")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
