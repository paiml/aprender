#!/usr/bin/env python3
"""Mixed: port-or-drop for the unique dark files whose absent units are both tests and fns (#4700).

A mixed file is an orphan that fn_census() calls unique, with at least one absent test (fn_census role "tests")
and at least one absent non-test fn (role "fns"). It needs both sheets: test_triage.triage_file() for its tests
(copy/gone/stale/live/inert, against its own and its callees' bodies) and callers.called_names() for its fns
(is each called from reached code, by name).

File verdict: `port` when the test side is port (some absent test is live: it needs porting, which needs cargo);
else `drop` when the test side is drop (every absent test is copy or gone) and the fn side is drop (no absent fn
is called); else `read`.

Usage: python3 mixed.py --pin REV [--tsv OUT] | --self-test
  rc 3 when the pin lists no crates/*/src file (a vacuous answer is refused).
"""
import argparse
import sys
from collections import Counter

import callers as cl
import fn_census as fc
import orphan_census as oc
import test_triage as tt

VERDICTS = ("drop", "port", "read")
COLUMNS = ("path", "lines", "absent_tests", "absent_fns", "test_verdict", "fn_verdict", "verdict",
           "dead_refs", "live_refs", "uncalled_fns")


def mixed(res):
    """The orphans that fn_census() calls unique, with >=1 absent test and >=1 absent non-test fn."""
    def absent_roles(rows):
        return {fc.role(r) for r in rows if r[0] == "absent"}
    return sorted(f for f, (v, rows, _, _) in res.items()
                  if v == "unique" and {"tests", "fns"} <= absent_roles(rows))


def file_verdict(test_v, fn_v):
    return "port" if test_v == "port" else "drop" if test_v == "drop" and fn_v == "drop" else "read"


def sheet(files, reach, orph, res):
    """{mixed orphan: (test rows, changed, fn names, fn names reached code calls)}."""
    cix = tt.build({f: files[f] for f in reach if f.endswith(".rs")}, said=True)
    dix = tt.build({f: files[f] for f in orph})
    mods, crates = tt.repo_mods(files)
    names = cl.called_names(files[f] for f in reach if f.endswith(".rs"))
    out = {}
    for f in mixed(res):
        rows = res[f][1]
        absent = {r[2] for r in rows if r[0] == "absent"}
        ctx = {"mods": mods, "crates": crates, "own": tt.crate_of(f)}
        test_rows = tt.triage_file(files[f], absent, cix, dix, ctx)
        fns = sorted({r[2] for r in rows if r[0] == "absent" and fc.role(r) == "fns"})
        out[f] = (test_rows, sum(r[0] == "changed" for r in rows), fns, [n for n in fns if n in names])
    return out


def cells(f, lines, test_rows, changed, fns, hit):
    test_v = tt.verdict(test_rows)
    fn_v = cl.verdict(fns, hit)
    return [f, lines, len(test_rows), len(fns), test_v, fn_v, file_verdict(test_v, fn_v),
            tt.reasons(test_rows, ("gone", "arity")), tt.reasons(test_rows, ("live",), 8),
            " ".join(n for n in fns if n not in hit)]


def tally(rows):
    v, out = COLUMNS.index("verdict"), {k: [0, 0] for k in VERDICTS}
    for r in rows:
        out[r[v]][0] += 1
        out[r[v]][1] += r[1]
    return out


def summary(pin, rows):
    v, t = COLUMNS.index("verdict"), tally(rows)
    at, af = sum(r[2] for r in rows), sum(r[3] for r in rows)
    lines = [f"mixed at {pin}: {len(rows)} mixed orphans, {at} absent tests, {af} absent fns",
             "  files: " + ", ".join(f"{k} {t[k][0]} ({t[k][1]:,} lines)" for k in VERDICTS)]
    return lines + [f"  drop {r[0]} ({r[1]}): tests {r[7]}, fns {r[9]}" for r in rows if r[v] == "drop"]


def report(pin, out, orph, tsv):
    rows = [cells(f, orph[f]["lines"], *out[f]) for f in sorted(out)]
    print("\n".join(summary(pin, rows)))
    if tsv:
        with open(tsv, "w") as fh:
            fh.write("\t".join(COLUMNS) + "\n")
            fh.writelines("\t".join(map(str, r)) + "\n" for r in rows)


RES = {
    "m.rs": ("unique", [cl.fn_row("f"), cl.fn_row("u"), cl.fn_row("t", True)], 0, None),  # mixed: fn + test
    "a.rs": ("unique", [cl.fn_row("t", True)], 0, None),  # tests-only, not mixed
    "b.rs": ("unique", [cl.fn_row("f")], 0, None),  # no-test, not mixed
    "c.rs": ("stale", [cl.fn_row("t", True), cl.fn_row("f")], 0, None),  # not unique
}
MIXED_TEXT = "fn f() { g() } fn u() {} #[test]\nfn t() { f() }\n"
REACH = {"r.rs": "fn g() {}\n"}


def self_test():
    cases = [("mixed", mixed(RES), ["m.rs"])]
    out = sheet({**REACH, "m.rs": MIXED_TEXT}, {"r.rs"}, {"m.rs": MIXED_TEXT}, {"m.rs": RES["m.rs"]})
    test_rows, changed, fns, hit = out["m.rs"]
    cases += [("mixed test_rows", [r[0] for r in test_rows], ["t"]),
              ("mixed test state", test_rows[0][1], "live"),  # t calls f, which calls reached g: live
              ("mixed fns", fns, ["f", "u"]),
              ("mixed hit", hit, [])]  # f is called only from m.rs itself (dark), not from reach
    cases += [(f"verdict {w}", file_verdict(*a), w) for a, w in
              ((("drop", "drop"), "drop"), (("drop", "read"), "read"), (("port", "drop"), "port"),
               (("read", "drop"), "read"))]
    bad = [(n, got, want) for n, got, want in cases if got != want]
    for n, got, want in bad:
        print(f"FAIL {n}: got {got!r}, want {want!r}")
    print(f"mixed self-test: {len(cases) - len(bad)}/{len(cases)} cases pass")
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
        print(f"mixed: {e}", file=sys.stderr)
        return 3
    rows, _, _, reach = oc.census(files, bl, tree)
    orph = {r["path"]: r for r in rows if r["status"] == "orphan"}
    res = fc.fn_census(files, reach, orph)
    report(pin, sheet(files, reach, orph, res), orph, a.tsv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
