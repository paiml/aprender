#!/usr/bin/env python3
"""Callers: whether reached code calls the absent fns of the no-test dark files, by name (#4700).

A no-test file is an orphan that fn_census() calls unique and none of whose absent units is a test. For each of
its absent fns (fn_census role "fns"), the sheet says whether some reached .rs file calls that name, as `name(`,
`a::name(`, `T::name(` or `.name(` (test_triage.calls() over the masked text, with
the name after each `fn` blanked so that a def is no call). By name only, so it
errs toward called: a call to a live fn of the same name counts.

Verdict: drop when the file has absent fns and reached code calls none of them (no compiled code can reach the
units only it holds: a drop candidate, batch 7); read when one is called (a live fn of that name exists, which the
dark one may predate or extend); items when it has no absent fn (only types, consts or the rest).

Usage: python3 callers.py --pin REV [--tsv OUT] | --self-test
  rc 3 when the pin lists no crates/*/src file (a vacuous answer is refused).
"""
import argparse
import re
import sys
from collections import Counter

import fn_census as fc
import orphan_census as oc
import test_triage as tt

VERDICTS = ("drop", "read", "items")
DEF_RE = re.compile(r"\bfn(\s+)(\w+)")
COLUMNS = ("path", "lines", "absent_fns", "called", "verdict", "called_names", "uncalled_names")


def no_test(res):
    """The orphans that fn_census() calls unique and none of whose absent units is a test."""
    return sorted(f for f, (v, rows, _, _) in res.items()
                  if v == "unique" and not any(fc.role(r) == "tests" for r in rows if r[0] == "absent"))


def called_names(texts):
    """Every name that a call in one of texts names."""
    out = set()
    for t in texts:
        mask = DEF_RE.sub(lambda m: "fn" + m.group(1) + " " * len(m.group(2)), fc.lex(t)[1])
        out |= {c[2] for c in tt.calls(mask, fc.pairs(mask), 0, len(mask))}
    return out


def verdict(fns, hit):
    return "items" if not fns else "read" if hit else "drop"


def sheet(files, reach, res):
    """{no-test orphan: (its absent fn names, those that reached code calls)}."""
    names = called_names(files[f] for f in reach if f.endswith(".rs"))
    out = {}
    for f in no_test(res):
        fns = sorted({r[2] for r in res[f][1] if r[0] == "absent" and fc.role(r) == "fns"})
        out[f] = (fns, [n for n in fns if n in names])
    return out


def cells(f, lines, fns, hit):
    return (f, lines, len(fns), len(hit), verdict(fns, hit), " ".join(hit),
            " ".join(n for n in fns if n not in hit))


def summary(pin, rows):
    """The report lines: absent fns called and not, files and lines per verdict, and each drop row's fns."""
    v, k, c = COLUMNS.index("verdict"), sum(r[2] for r in rows), sum(r[3] for r in rows)
    n, ln = Counter(r[v] for r in rows), Counter()
    for r in rows:
        ln[r[v]] += r[1]
    out = [f"callers at {pin}: {len(rows)} no-test orphans, {k} absent fns: called {c}, uncalled {k - c}",
           "  files: " + ", ".join(f"{x} {n[x]} ({ln[x]:,} lines)" for x in VERDICTS)]
    return out + [f"  drop {r[0]} ({r[1]}): {r[-1][:100]}" for r in rows if r[v] == "drop"]


def report(pin, out, orph, tsv):
    rows = [cells(f, orph[f]["lines"], *out[f]) for f in sorted(out)]
    print("\n".join(summary(pin, rows)))
    if tsv:
        with open(tsv, "w") as fh:
            fh.write("\t".join(COLUMNS) + "\n")
            fh.writelines("\t".join(map(str, r)) + "\n" for r in rows)


def fn_row(name, test=False, st="absent"):
    return (st, "fn", name, test, True)


RES = {"a.rs": ("unique", [fn_row("f"), fn_row("u"), fn_row("t", True, "present")], 0, None),
       "b.rs": ("unique", [fn_row("t", True)], 0, None),
       "c.rs": ("unique", [fn_row("g"), fn_row("t", True)], 0, None),
       "d.rs": ("stale", [fn_row("h")], 0, None),
       "e.rs": ("unique", [("absent", "struct", "S", False, True)], 0, None)}
CALLS = 'fn x() { foo(1); a::bar(); T::baz(2); y.qux(); let s = "nope(1)"; } // gone(2)\n'


def self_test():
    names = called_names([CALLS])
    cases = [("no_test", no_test(RES), ["a.rs", "e.rs"])]
    cases += [(f"called {n}", n in names, True) for n in ("foo", "bar", "baz", "qux")]
    cases += [(f"not called {n}", n in names, False) for n in ("x", "nope", "gone")]
    cases += [("sheet", sheet({"r.rs": "fn r() { f(); }", "a.rs": "", "e.rs": ""}, {"r.rs"}, RES),
               {"a.rs": (["f", "u"], ["f"]), "e.rs": ([], [])})]
    cases += [(f"verdict {w}", verdict(*a), w) for a, w in ((([], []), "items"), ((["f"], []), "drop"),
                                                             ((["f", "g"], ["g"]), "read"))]
    bad = [(n, got, want) for n, got, want in cases if got != want]
    for n, got, want in bad:
        print(f"FAIL {n}: got {got!r}, want {want!r}")
    print(f"callers self-test: {len(cases) - len(bad)}/{len(cases)} cases pass")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pin")
    ap.add_argument("--tsv")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.pin:
        ap.error("--pin or --self-test")
    pin = oc.git("rev-parse", "--short=10", a.pin).decode().strip()
    try:
        files, bl, tree = oc.load(pin)
    except oc.Vacuous as e:
        print(f"callers: {e}", file=sys.stderr)
        return 3
    rows, _, _, reach = oc.census(files, bl, tree)
    orph = {r["path"]: r for r in rows if r["status"] == "orphan"}
    report(pin, sheet(files, reach, fc.fn_census(files, reach, orph)), orph, a.tsv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
