#!/usr/bin/env python3
"""The order a ladder sweep measures its cells in (#4039, a #4033 lever).

    python3 scripts/lib/ladder_order.py <certification.json|-> < plan > ordered-plan
    python3 scripts/lib/ladder_order.py --selftest

A plan line is one cell of scripts/model_ladder.sh, '|'-separated:
    rung|<id>|<file>|<sha256>|<backends>|<required>|<hosts>|<bytes>
    inv|<file>|<path>|<sha256>|<bytes>

The order: CRUX-certified models first, so the CRUX sweep can start as soon as their cells are measured; then
largest first, to cut the tail; then kind and id, so the order is total and the same on every run. The
certified set is the keys of the prompt certification's per-model admissions (model_ladder_crux.load_certified
reads the same two fields). With no certification the order is largest first only, and stderr says so.

ORDER IS A SCHEDULE, NEVER A VERDICT. A reorder may not drop, add or duplicate a cell, since a dropped cell is
an unmeasured model that reads as a pass. So the output must be a permutation of the input, and anything else
exits 2 before a single cell runs.
"""
import json
import re
import sys
from collections import Counter

HEX64 = re.compile(r"[0-9a-f]{64}")


def certified(cert_path):
    """-> (set of sha256, note). An absent or unreadable certification orders by size alone; it never fails
    the sweep, because order carries no verdict."""
    if cert_path in ("", "-"):
        return set(), "no certification given: largest first only"
    try:
        with open(cert_path, encoding="utf-8") as fh:
            c = json.load(fh)
    except (OSError, ValueError) as exc:
        return set(), f"certification {cert_path} unreadable ({exc.__class__.__name__}): largest first only"
    keys = set()
    for field in ("admitted_by_sha", "admitted_by_sha_thinking"):
        v = c.get(field) if isinstance(c, dict) else None
        if isinstance(v, dict):
            keys |= {k for k in v if HEX64.fullmatch(str(k))}
    if not keys:
        return set(), f"certification {cert_path} certifies no model: largest first only"
    return keys, f"certification {cert_path}: {len(keys)} certified model(s) first"


def parse(line):
    f = line.split("|")
    if f[0] == "rung" and len(f) == 8:
        return {"kind": "rung", "id": f[1], "sha": f[3], "bytes": f[7]}
    if f[0] == "inv" and len(f) == 5:
        return {"kind": "inv", "id": "inv:" + f[1], "sha": f[3], "bytes": f[4]}
    raise ValueError(f"not a plan line: {line!r}")


def order(lines, cert):
    """-> the lines, ordered. Pure: same input, same output."""
    def key(line):
        c = parse(line)
        size = int(c["bytes"]) if c["bytes"].isdigit() else 0
        return (0 if c["sha"] in cert else 1, -size, 0 if c["kind"] == "rung" else 1, c["id"])
    return sorted(lines, key=key)


def is_permutation(before, after):
    return Counter(before) == Counter(after)


def main(argv):
    if argv[1:] == ["--selftest"]:
        return selftest()
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    lines = [ln for ln in sys.stdin.read().splitlines() if ln.strip()]
    try:
        for ln in lines:
            parse(ln)
    except ValueError as exc:
        print(f"ladder_order: {exc}", file=sys.stderr)
        return 2
    cert, note = certified(argv[1])
    out = order(lines, cert)
    if not is_permutation(lines, out):
        print("ladder_order: REFUSED -- the ordered plan is not a permutation of the plan (a cell was dropped, "
              "added or duplicated); order is a schedule and may never change what is measured", file=sys.stderr)
        return 2
    print(f"order: {note}", file=sys.stderr)
    sys.stdout.write("".join(ln + "\n" for ln in out))
    return 0


# ---- case table ---------------------------------------------------------------------------------------------
def selftest():
    A, B, C = "a" * 64, "b" * 64, "c" * 64
    small_cert = f"rung|r-small|s.gguf|{A}|cpu,cuda|1||100"
    big = f"rung|r-big|b.gguf|{B}|cpu,cuda|1||9000"
    inv_cert = f"inv|i.gguf|/m/i.gguf|{C}|500"
    inv_huge = f"inv|h.gguf|/m/h.gguf|{'d' * 64}|99999"
    absent = f"rung|r-absent|x.gguf|{'e' * 64}|cpu|1||0"
    plan = [big, absent, inv_huge, small_cert, inv_cert]
    fails = 0

    def case(name, ok):
        nonlocal fails
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
        fails += not ok

    got = order(plan, {A, C})
    case("certified first, and among them largest first (an inventory model outranks a smaller rung)",
         got[:2] == [inv_cert, small_cert])
    case("then the rest largest first; a cell with no bytes (absent) last", got[2:] == [inv_huge, big, absent])
    case("with no certification: largest first only", order(plan, set()) == [inv_huge, big, inv_cert, small_cert,
                                                                               absent])
    case("the order is total: shuffling the input does not change it", order(plan[::-1], {A, C}) == got)
    tie = [f"inv|z.gguf|/m/z.gguf|{'f' * 64}|7", f"rung|y|y.gguf|{'0' * 64}|cpu|1||7"]
    case("a size tie puts the rung before the inventory model", order(tie, set())[0].startswith("rung|y|"))
    same = [f"rung|m-b|b2.gguf|{'1' * 64}|cpu|1||50", f"rung|m-a|a2.gguf|{'2' * 64}|cpu|1||50"]
    case("a full tie (kind, size, certification) orders by id, whatever the input order",
         order(same, set()) == order(same[::-1], set()) == [same[1], same[0]])
    bad_size = f"rung|r-nosize|n.gguf|{'3' * 64}|cpu|1||unknown"
    case("a non-numeric size sorts as 0 (last), never first", order([bad_size, absent, big], set())[0] == big
         and order([bad_size, big], set())[-1] == bad_size)
    case("the output is a permutation of the input", is_permutation(plan, got))
    # must-RED: the guard that makes a reorder unable to turn a RED green
    case("must-RED: a dropped cell is not a permutation", not is_permutation(plan, got[:-1]))
    case("must-RED: a duplicated cell is not a permutation", not is_permutation(plan, got[:-1] + [got[0]]))
    case("must-RED: a substituted cell is not a permutation", not is_permutation(plan, got[:-1] + [tie[0]]))
    case("must-RED: a pure duplicate (nothing dropped) is not a permutation", not is_permutation(plan, plan + [plan[0]]))
    try:
        parse("rung|too|few")
        case("a malformed plan line is refused", False)
    except ValueError:
        case("a malformed plan line is refused", True)
    import os
    import tempfile
    d = tempfile.mkdtemp()
    p = os.path.join(d, "cert.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"admitted_by_sha": {A: ["x"]}, "admitted_by_sha_thinking": {C: {"on": ["y"]}, "junk": {}}}, fh)
    case("the certified set is the admission keys (both fields; a non-sha key ignored)", certified(p)[0] == {A, C})
    case("an absent certification orders by size, never fails", certified(os.path.join(d, "nope.json"))[0] == set())
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"admitted_by_sha": {"x" + A: ["p"], A + "0": ["p"]}}, fh)
    case("a key that only CONTAINS a sha is not a certified sha", certified(p)[0] == set())
    with open(p, "w", encoding="utf-8") as fh:
        fh.write('{"admitted_by_sha": {"' + A + '": [')   # truncated mid-write
    got_c = certified(p)
    case("a corrupt certification orders by size and says so, never crashes", got_c[0] == set() and "unreadable" in got_c[1])
    try:
        parse(big + "|extra")
        case("a rung line with an extra field is refused", False)
    except ValueError:
        case("a rung line with an extra field is refused", True)

    # The CLI path, as model_ladder.sh calls it: the refusals must fire through main(), not only in the helpers.
    import subprocess
    me = os.path.abspath(__file__)

    def cli(stdin, patch="pass"):
        code = ("import sys; sys.path.insert(0, %r); import ladder_order as m; %s; sys.exit(m.main(['ladder_order', '-']))"
                % (os.path.dirname(me), patch))
        r = subprocess.run([sys.executable, "-c", code], input=stdin, capture_output=True, text=True)
        return r.returncode, r.stdout, r.stderr
    rc, out, _ = cli("\n".join(plan) + "\n")
    case("CLI: a good plan exits 0 and prints every cell", rc == 0 and sorted(out.splitlines()) == sorted(plan))
    rc, out, err = cli("\n".join(plan + ["rung|too|few"]) + "\n")
    case("CLI must-RED: a malformed line exits 2 and prints NO plan", rc == 2 and out == "" and "not a plan line" in err)
    r = subprocess.run([sys.executable, me, "-", "extra"], input="\n".join(plan) + "\n", capture_output=True, text=True)
    case("CLI must-RED: an extra argument is refused (exit 2, NO plan)", r.returncode == 2 and r.stdout == "")
    rc, out, err = cli("\n".join(plan) + "\n", "m.order = lambda lines, cert: lines[:-1]")
    case("CLI must-RED: an order that drops a cell exits 2 and prints NO plan", rc == 2 and out == "" and "REFUSED" in err)
    rc, out, err = cli("\n".join(plan) + "\n", "m.order = lambda lines, cert: lines + lines[:1]")
    case("CLI must-RED: an order that duplicates a cell exits 2", rc == 2 and out == "" and "REFUSED" in err)
    total = 23
    print(f"{total - fails}/{total} cases")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
