"""release_gate_classes_cases.py -- case table + mutants for release_gate_classes.py (#4045 M8).

    python3 scripts/lib/release_gate_classes_cases.py [--mutants]
A scratch publish path (autopilot STEPS, dogfood marks + version rows, preflight rule headers, a declared gate in
cargo metadata) is derived; every mutant must break the row that NAMES its rule.
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def scratch():
    d = tempfile.mkdtemp(prefix="gate-classes-")
    os.makedirs(os.path.join(d, "scripts/release"))
    os.makedirs(os.path.join(d, "scripts/lib"))
    shutil.copy(os.path.join(HERE, "dogfood_gates.py"), os.path.join(d, "scripts/lib/dogfood_gates.py"))
    open(os.path.join(d, "scripts/release/autopilot.sh"), "w").write("#!/bin/bash\nSTEPS=(wait tag cascade)\n")
    open(os.path.join(d, "scripts/dogfood.sh"), "w").write(
        "mark() { :; }\n  mark bashrs FAIL \"x\"\n    mark model-parity PASS \"y\"\n"
        "version_row() {\n  printf 'version-unpublished PASS x\\n'\n}\n")
    open(os.path.join(d, "scripts/check_publish_preflight.sh"), "w").write("#   R1  clean tree\n#   R4  on main\n")
    md = {"packages": [{"name": "aprender", "metadata": {"dogfood": {"gates": ["scripts/check_model_ladder.sh"]}}}]}
    return d, json.dumps(md)


WANT = {"step:wait", "step:tag", "step:cascade", "dogfood:bashrs", "dogfood:model-parity", "dogfood:version-unpublished",
        "dogfood:declared:check_model_ladder", "preflight:R1", "preflight:R4"}


def run(mod):
    d, md = scratch()
    u, errs = mod.derive(d, md)
    res = {}
    res["derive-steps"] = ({"step:wait", "step:cascade"} <= u, sorted(u))
    res["derive-marks"] = ({"dogfood:bashrs", "dogfood:model-parity"} <= u, sorted(u))
    res["derive-version-rows"] = ("dogfood:version-unpublished" in u, sorted(u))
    res["derive-declared"] = ("dogfood:declared:check_model_ladder" in u, sorted(u))
    res["derive-preflight"] = ({"preflight:R1", "preflight:R4"} <= u, sorted(u))
    res["derive-exact"] = (u == WANT and not errs, (sorted(u ^ WANT), errs))
    full = {g: {"class": "real", "why": "w"} for g in WANT}
    res["all-classified-clean"] = (mod.check(WANT, full) == [], mod.check(WANT, full))
    part = {g: c for g, c in full.items() if g != "dogfood:bashrs"}
    p = mod.check(WANT, part)
    res["unclassified-refused"] = (any(x.startswith("UNCLASSIFIED dogfood:bashrs") for x in p), p)
    p = mod.check(WANT, dict(full, **{"dogfood:gone": {"class": "real", "why": "w"}}))
    res["stale-refused"] = (any(x.startswith("STALE dogfood:gone") for x in p), p)
    p = mod.check(WANT, dict(full, **{"preflight:R4": {"class": "maybe", "why": "w"}}))
    res["malformed-class-refused"] = (any(x.startswith("MALFORMED preflight:R4") for x in p), p)
    p = mod.check(WANT, dict(full, **{"preflight:R4": {"class": "real", "why": " "}}))
    res["no-reason-refused"] = (any(x.startswith("MALFORMED preflight:R4") for x in p), p)
    os.remove(os.path.join(d, "scripts/release/autopilot.sh"))
    u2, errs2 = mod.derive(d, md)
    res["missing-source-declines"] = (any("autopilot.sh" in e for e in errs2), errs2)
    shutil.rmtree(d, ignore_errors=True)
    return res


MUTANTS = [
    ("steps-unread", 'm = re.search(r"^STEPS=\\(([^)]*)\\)", ap or "", re.M)', "m = None", "derive-steps"),
    ("marks-unread", 'ids |= {"dogfood:" + r for r in re.findall(r"^\\s*mark ([a-z0-9][a-z0-9_-]*) ", df, re.M)}', "pass", "derive-marks"),
    ("version-rows-unread", "vr = re.search(", "vr = None and re.search(", "derive-version-rows"),
    ("declared-unread", 'ids.add("dogfood:declared:"', 'ids.discard("dogfood:declared:"', "derive-declared"),
    ("preflight-unread", 'ids |= {"preflight:" + r for r in re.findall(r"^#\\s+(R[0-9]+)\\s", pf, re.M)}', "pass", "derive-preflight"),
    ("unclassified-ok", "    for g in sorted(universe - set(classes)):", "    for g in []:", "unclassified-refused"),
    ("stale-ok", "    for g in sorted(set(classes) - universe):", "    for g in []:", "stale-refused"),
    ("any-class-ok", 'c.get("class") not in CLASSES or ', "", "malformed-class-refused"),
    ("no-why-ok", ' or not str(c.get("why") or "").strip()', "", "no-reason-refused"),
    ("missing-source-ok", '        errs.append("scripts/release/autopilot.sh: no STEPS=(...)")', "        pass", "missing-source-declines"),
]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    src = os.path.join(HERE, "release_gate_classes.py")
    res = run(load(src, "release_gate_classes"))
    bad = [r for r, (ok, _) in res.items() if not ok]
    for r, (ok, got) in res.items():
        print("%s %-28s %s" % ("PASS" if ok else "FAIL", r, "" if ok else str(got)[:200]))
    print("rows: %d pass, %d fail" % (len(res) - len(bad), len(bad)))
    if bad:
        return 1
    if "--mutants" not in sys.argv:
        return 0
    text, survived, tmp = open(src).read(), 0, tempfile.mkdtemp(prefix="gate-classes-mut-")
    for name, old, new, row in MUTANTS:
        if text.count(old) != 1:
            print("MUTANT %-20s ANCHOR %d != 1" % (name, text.count(old)))
            survived += 1
            continue
        p = os.path.join(tmp, name + ".py")
        open(p, "w").write(text.replace(old, new))
        try:
            killed = not run(load(p, "m_" + name.replace("-", "_")))[row][0]
            how = ""
        except Exception as exc:   # a crash is not a clean kill: it is reported, and counts as SURVIVED
            killed, how = False, " (CRASHED: %r)" % exc
        print("MUTANT %-20s %s by %s%s" % (name, "KILLED" if killed else "SURVIVED", row, how))
        survived += not killed
    print("mutants: %d/%d killed by their named row" % (len(MUTANTS) - survived, len(MUTANTS)))
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
