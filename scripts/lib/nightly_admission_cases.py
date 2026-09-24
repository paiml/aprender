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


def v(host, sha, age_h, green=True, incoherent=None):
    return {"schema": "apr-nightly-certification/v1", "host": host, "sha": sha * 40, "t_end": NOW - age_h * H,
            "green": green, "why": [] if green else ["the ladder is RED: executed=3 red=1"], "_path": "%s/%s" % (sha, host),
            "_incoherent": incoherent}


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
    c, r = pick([v("lambda", "a", -2)])
    res["future-t_end-refused"] = ("lambda" in r and "FUTURE" in r["lambda"], (c, r))
    c, r = pick([v("lambda", "a", 3, incoherent="its CRUX cpu receipt is missing or not PASS")])
    res["incoherent-green-refused"] = ("lambda" in r and "says green but" in r["lambda"], (c, r))
    # assemble: the chosen night's files are linked where the judge reads them; a vanished receipt refuses
    d = tempfile.mkdtemp(prefix="nightly-adm-")
    lad, cg, cc, cert = (os.path.join(d, n) for n in ("l.json", "cg.json", "cc.json", "cert.json"))
    import json as _j
    _j.dump({"apr_sha": "a" * 40, "executed": 3, "red": 0}, open(lad, "w"))
    PASS = {"schema": "crux-inference-receipt/v1", "apr": {"sha": "a" * 40}, "summary": {"verdict": "PASS"}}
    for p in (cg, cc):
        _j.dump(PASS, open(p, "w"))
    open(cert, "w").write("{}")
    lanes = {"gpu": {"receipt": cg}, "cpu": {"receipt": cc}}
    good = dict(v("lambda", "a", 3), ladder={"receipt": lad}, crux={"lanes": lanes}, certification={"path": cert})
    out = os.path.join(d, "out")
    bad = mod.assemble({"lambda": good}, out)
    res["assemble-links"] = (not bad and os.path.realpath(os.path.join(out, "receipts", "lambda.json")) == lad
                             and os.path.realpath(os.path.join(out, "crux", "lambda-gpu.json")) == cg
                             and os.path.realpath(os.path.join(out, "crux", "lambda-cpu.json")) == cc
                             and os.path.exists(os.path.join(out, "crux", "prompt-certification.json")), bad)
    res["coherent-rederived-green"] = (mod.coherent(good) is None, mod.coherent(good))
    _j.dump(dict(PASS, summary={"verdict": "RED"}), open(cc, "w"))
    res["coherent-sees-red-lane"] = (mod.coherent(good) is not None and "cpu" in mod.coherent(good), mod.coherent(good))
    _j.dump(dict(PASS, apr={"sha": "c" * 40}), open(cc, "w"))
    res["coherent-sees-crux-sha"] = (mod.coherent(good) is not None and "measured apr" in mod.coherent(good), mod.coherent(good))
    _j.dump(PASS, open(cc, "w"))
    _j.dump({"apr_sha": "b" * 40, "executed": 3, "red": 0}, open(lad, "w"))
    res["coherent-sees-ladder-sha"] = (mod.coherent(good) is not None and "not the nightly" in mod.coherent(good), mod.coherent(good))
    _j.dump({"apr_sha": "a" * 40, "executed": 3, "red": 0}, open(lad, "w"))
    gone = dict(good, crux={"lanes": dict(lanes, cpu={"receipt": os.path.join(d, "missing.json")})})
    bad = mod.assemble({"lambda": gone}, os.path.join(d, "out2"))
    res["assemble-refuses-vanished"] = (bool(bad) and "receipts are gone" in bad[0], bad)
    # #4117: a night recorded with RELATIVE paths, then MOVED under another root (the two-host release gate judges
    # on one host, so the other host's night is copied in): admitted, and linked from where it now lives.
    import shutil as _sh
    src = os.path.join(d, "hostA-root", "a" * 40, "lambda")
    os.makedirs(os.path.join(src, "ladder")); os.makedirs(os.path.join(src, "crux"))
    _j.dump({"apr_sha": "a" * 40, "executed": 3, "red": 0}, open(os.path.join(src, "ladder", "lambda.json"), "w"))
    for lane in ("gpu", "cpu"):
        _j.dump(PASS, open(os.path.join(src, "crux", "lambda-%s.json" % lane), "w"))
    open(os.path.join(src, "prompt-certification.json"), "w").write("{}")
    rel = dict(v("lambda", "a", 3), ladder={"receipt": "ladder/lambda.json"},
               crux={"lanes": {k: {"receipt": "crux/lambda-%s.json" % k} for k in ("gpu", "cpu")}},
               certification={"path": "prompt-certification.json"})
    _j.dump(dict(rel, _path=None), open(os.path.join(src, "verdict.json"), "w"))
    moved = os.path.join(d, "hostB-root", "a" * 40, "lambda")
    _sh.copytree(src, moved); _sh.rmtree(os.path.join(d, "hostA-root"))
    loaded = [x for x in mod.load_verdicts(os.path.join(d, "hostB-root"))]
    ok_load = len(loaded) == 1 and loaded[0]["_incoherent"] is None
    bad = mod.assemble({"lambda": loaded[0]}, os.path.join(d, "out3")) if loaded else ["not loaded"]
    res["moved-root-admitted-by-relative-paths"] = (
        ok_load and not bad and os.path.realpath(os.path.join(d, "out3", "crux", "lambda-cpu.json"))
        == os.path.join(moved, "crux", "lambda-cpu.json"), (loaded[:1], bad))
    # ...and a relative path that climbs OUT of the verdict's directory is refused, never followed -- even when a
    # valid GREEN receipt sits at the place it points to (so only the escape rule can refuse it)
    _j.dump(PASS, open(os.path.join(d, "outside-cpu.json"), "w"))
    _j.dump({"apr_sha": "a" * 40, "executed": 3, "red": 0}, open(os.path.join(d, "outside-ladder.json"), "w"))
    esc = dict(rel, _path=os.path.join(moved, "verdict.json"),
               crux={"lanes": dict(rel["crux"]["lanes"], cpu={"receipt": "../../../outside-cpu.json"})})
    why = mod.coherent(esc)
    res["relative-path-escaping-refused"] = (why is not None and "escapes the verdict's directory" in why, why)
    esc_l = dict(rel, _path=os.path.join(moved, "verdict.json"), ladder={"receipt": "../../../outside-ladder.json"})
    why_l = mod.coherent(esc_l)
    res["relative-ladder-escaping-refused"] = (why_l is not None and "escapes" in why_l, why_l)
    return res


MUTANTS = [
    ("age-ignored", "            elif age > max_age_h:", "            elif False:", "stale-refused"),
    ("red-accepted", '            if v.get("green") is not True:', "            if False:", "red-refused"),
    ("ancestry-ignored", "            elif not is_ancestor(sha):", "            elif False:", "sibling-refused"),
    ("oldest-chosen", 'chosen[h] = max(ok, key=lambda v: float(v.get("t_end") or 0))',
     'chosen[h] = min(ok, key=lambda v: float(v.get("t_end") or 0))', "newest-green-chosen"),
    ("any-host", '        mine = [v for v in verdicts if v.get("host") == h]', "        mine = list(verdicts)", "other-host-does-not-count"),
    ("vanished-ok", "        gone = [p for p in [lad] + [lanes.get(k) for k in REQUIRED_LANES] if not (p and os.path.isfile(p))]",
     "        gone = []", "assemble-refuses-vanished"),
    ("future-ok", "            elif age < -SKEW_S / 3600.0:", "            elif False:", "future-t_end-refused"),
    ("green-trusted", '            elif v.get("_incoherent"):', "            elif False:", "incoherent-green-refused"),
    ("lanes-gpu-only", 'REQUIRED_LANES = ("gpu", "cpu")', 'REQUIRED_LANES = ("gpu",)', "coherent-sees-red-lane"),
    ("crux-sha-unread", '        if got != v.get("sha"):', "        if False:", "coherent-sees-crux-sha"),
    ("ladder-sha-unread", '    if lad.get("apr_sha") != v.get("sha"):', "    if False:", "coherent-sees-ladder-sha"),
    ("relative-from-cwd", '    base = os.path.dirname(os.path.abspath(v.get("_path") or ""))', '    base = os.getcwd()',
     "moved-root-admitted-by-relative-paths"),
    ("escape-followed", "    if not q.startswith(base + os.sep):", "    if False:", "relative-path-escaping-refused"),
    ("escape-followed-ladder", '    if why and "escapes" in why:\n        return "its ladder receipt: %s" % why',
     '    if False:\n        return "its ladder receipt: %s" % why', "relative-ladder-escaping-refused"),
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
