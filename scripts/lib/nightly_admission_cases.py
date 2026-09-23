"""nightly_admission_cases.py -- case table + mutants for nightly_admission.py (#4040).

    python3 scripts/lib/nightly_admission_cases.py            # every row must hold
    python3 scripts/lib/nightly_admission_cases.py --mutants  # every mutant must break its NAMED row
"""

import importlib.util
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
NOW = 1_000_000.0
H = 3600.0
ANC = {"a" * 40, "b" * 40}          # ancestors of the cut; "c"*40 is a sibling


def v(host, sha, age_h, green=True):
    return {"schema": "apr-nightly-certification/v1", "host": host, "sha": sha * 40, "t_end": NOW - age_h * H,
            "green": green, "why": [] if green else ["the ladder is RED: executed=3 red=1"], "_path": "%s/%s" % (sha, host)}


def run(mod):
    anc = lambda s: s in ANC
    res = {}
    def pick(vs, hosts=("lambda",), max_age=24.0):
        return mod.select(vs, list(hosts), NOW, anc, max_age)
    c, r = pick([v("lambda", "a", 3)])
    res["green-fresh-ancestor"] = ("lambda" in c and not r, (c, r))
    c, r = pick([v("lambda", "a", 30)])
    res["stale-refused"] = ("lambda" in r and "h old" in r["lambda"], (c, r))
    c, r = pick([v("lambda", "a", 3, green=False)])
    res["red-refused"] = ("lambda" in r and "is RED" in r["lambda"], (c, r))
    c, r = pick([v("lambda", "c", 3)])
    res["sibling-refused"] = ("lambda" in r and "not the cut or an ancestor" in r["lambda"], (c, r))
    c, r = pick([v("lambda", "a", 10), v("lambda", "b", 2)])
    res["newest-green-chosen"] = (c.get("lambda", {}).get("sha") == "b" * 40, (c, r))
    c, r = pick([v("lambda", "a", 3)], hosts=("lambda", "gx10"))
    res["every-host-required"] = ("gx10" in r and "no nightly at all" in r["gx10"] and "lambda" in c, (c, r))
    c, r = pick([v("lambda", "b", 2, green=False), v("lambda", "a", 5)])
    res["older-green-when-newest-red"] = (c.get("lambda", {}).get("sha") == "a" * 40, (c, r))
    c, r = pick([v("gx10", "a", 3)])
    res["other-host-does-not-count"] = ("lambda" in r, (c, r))
    # assemble: the chosen night's files are linked where the judge reads them; a vanished receipt refuses
    d = tempfile.mkdtemp(prefix="nightly-adm-")
    lad, crux, cert = (os.path.join(d, n) for n in ("l.json", "c.json", "cert.json"))
    for p in (lad, crux, cert):
        open(p, "w").write("{}")
    good = dict(v("lambda", "a", 3), ladder={"receipt": lad}, crux={"receipt": crux}, certification={"path": cert})
    out = os.path.join(d, "out")
    bad = mod.assemble({"lambda": good}, out)
    res["assemble-links"] = (not bad and os.path.realpath(os.path.join(out, "receipts", "lambda.json")) == lad
                             and os.path.realpath(os.path.join(out, "crux", "lambda-gpu.json")) == crux
                             and os.path.exists(os.path.join(out, "crux", "prompt-certification.json")), bad)
    gone = dict(good, crux={"receipt": os.path.join(d, "missing.json")})
    bad = mod.assemble({"lambda": gone}, os.path.join(d, "out2"))
    res["assemble-refuses-vanished"] = (bool(bad) and "receipts are gone" in bad[0], bad)
    return res


MUTANTS = [
    ("age-ignored", "            elif age > max_age_h:", "            elif False:", "stale-refused"),
    ("red-accepted", '            if v.get("green") is not True:', "            if False:", "red-refused"),
    ("ancestry-ignored", "            elif not is_ancestor(sha):", "            elif False:", "sibling-refused"),
    ("oldest-chosen", 'chosen[h] = max(ok, key=lambda v: float(v.get("t_end") or 0))',
     'chosen[h] = min(ok, key=lambda v: float(v.get("t_end") or 0))', "newest-green-chosen"),
    ("any-host", '        mine = [v for v in verdicts if v.get("host") == h]', "        mine = list(verdicts)", "other-host-does-not-count"),
    ("vanished-ok", "        if not (lad and os.path.isfile(lad) and crux and os.path.isfile(crux)):", "        if not (lad and crux):",
     "assemble-refuses-vanished"),
]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    src = os.path.join(HERE, "nightly_admission.py")
    res = run(load(src, "nightly_admission"))
    bad = [r for r, (ok, _) in res.items() if not ok]
    for r, (ok, got) in res.items():
        print("%s %-30s %s" % ("PASS" if ok else "FAIL", r, "" if ok else str(got)[:200]))
    print("rows: %d pass, %d fail" % (len(res) - len(bad), len(bad)))
    if bad:
        return 1
    if "--mutants" not in sys.argv:
        return 0
    text, survived, tmp = open(src).read(), 0, tempfile.mkdtemp(prefix="nightly-adm-mut-")
    for name, old, new, row in MUTANTS:
        if text.count(old) != 1:
            print("MUTANT %-18s ANCHOR %d != 1" % (name, text.count(old)))
            survived += 1
            continue
        p = os.path.join(tmp, name + ".py")
        open(p, "w").write(text.replace(old, new))
        killed = not run(load(p, "m_" + name.replace("-", "_")))[row][0]
        print("MUTANT %-18s %s by %s" % (name, "KILLED" if killed else "SURVIVED", row))
        survived += not killed
    print("mutants: %d/%d killed by their named row" % (len(MUTANTS) - survived, len(MUTANTS)))
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
