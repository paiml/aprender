"""release_gate_classes.py -- every gate that can block a publish, DERIVED from the publish path, and its class (#4045 M8).

Operator 2026-09-23 (via the release cop): "all of these trivial issues should be caught hours early and are "fake"
gates". From 0.70 every publish-blocking gate is classified:
  real         it protects users (a wrong model, a broken binary, a bad tag): red stops the train (andon);
  bookkeeping  it protects the repo's own accounting: it auto-fixes or reports + tickets, and NEVER blocks the publish.

THE UNIVERSE IS DERIVED, never hand-kept (a hand list is how a new gate slips in unclassified):
  step:<name>              scripts/release/autopilot.sh's STEPS=(...)
  dogfood:<row>            every literal `mark <row>` in scripts/dogfood.sh, and version_row's printed rows
  dogfood:declared:<gate>  [package.metadata.dogfood] gates (scripts/lib/dogfood_gates.py over the root manifest)
  preflight:<Rn>           the `#   Rn` rule headers of scripts/check_publish_preflight.sh
The classes live in scripts/release/gate_classes.yaml. A derived gate with no class is UNCLASSIFIED (refused); a
class for a gate that no longer exists is STALE (refused): the registry and the publish path cannot drift apart.

    python3 scripts/lib/release_gate_classes.py check <repo root> [<cargo metadata json>]
"""

import json
import os
import re
import sys

CLASSES = ("real", "bookkeeping")


def _read(p):
    try:
        return open(p, encoding="utf-8").read()
    except OSError:
        return None


def derive(root, metadata=None):
    """-> (set of gate ids, [why a source could not be read]). A missing source is reported, never skipped."""
    ids, errs = set(), []
    ap = _read(os.path.join(root, "scripts/release/autopilot.sh"))
    m = re.search(r"^STEPS=\(([^)]*)\)", ap or "", re.M)
    if m:
        ids |= {"step:" + s for s in m.group(1).split()}
    else:
        errs.append("scripts/release/autopilot.sh: no STEPS=(...)")
    df = _read(os.path.join(root, "scripts/dogfood.sh"))
    if df is None:
        errs.append("scripts/dogfood.sh unreadable")
    else:
        ids |= {"dogfood:" + r for r in re.findall(r"^\s*mark ([a-z0-9][a-z0-9_-]*) ", df, re.M)}
        vr = re.search(r"^version_row\(\) \{(.*?)^\}", df, re.M | re.S)
        ids |= {"dogfood:" + r for r in re.findall(r"printf '([a-z][a-z0-9-]*) ", vr.group(1) if vr else "")}
    pf = _read(os.path.join(root, "scripts/check_publish_preflight.sh"))
    if pf is None:
        errs.append("scripts/check_publish_preflight.sh unreadable")
    else:
        ids |= {"preflight:" + r for r in re.findall(r"^#\s+(R[0-9]+)\s", pf, re.M)}
    if metadata is None:
        # No cargo needed (guard_tree's cargo-free CI job runs this): the root manifest's own [package] table in the
        # shape `cargo metadata` gives it -- `metadata` there IS the TOML table verbatim -- so the declaration is
        # still parsed by dogfood_gates.plan, its ONE parser (#2644).
        try:
            import tomllib
            man = tomllib.load(open(os.path.join(root, "Cargo.toml"), "rb")).get("package") or {}
            metadata = {"packages": [{"name": man.get("name"), "metadata": man.get("metadata") or {}}]}
        except (OSError, ValueError) as exc:
            errs.append("the root Cargo.toml is unreadable (%s): the declared dogfood gates are unknown" % exc)
    if metadata is not None:
        sys.path.insert(0, os.path.join(root, "scripts/lib"))
        import dogfood_gates
        md = json.loads(metadata) if isinstance(metadata, str) else metadata
        for ln in dogfood_gates.plan(md, "aprender"):
            if ln.startswith("GATE "):
                ids.add("dogfood:declared:" + os.path.basename(ln[5:].strip()).rsplit(".sh", 1)[0])
    return ids, errs


def load_classes(path):
    import yaml
    doc = yaml.safe_load(open(path)) or {}
    return doc.get("gates") or {}


def check(universe, classes):
    """-> [problem]. Every derived gate has a class of the two, with a reason; no class names a gate that is gone."""
    out = []
    for g in sorted(universe - set(classes)):
        out.append("UNCLASSIFIED %s -- a publish-blocking gate with no class (real | bookkeeping)" % g)
    for g in sorted(set(classes) - universe):
        out.append("STALE %s -- classified, but the publish path no longer runs it" % g)
    for g, c in sorted(classes.items()):
        if not isinstance(c, dict) or c.get("class") not in CLASSES or not str(c.get("why") or "").strip():
            out.append("MALFORMED %s -- needs class: real|bookkeeping and a why" % g)
    return out


def main(argv):
    if len(argv) < 2 or argv[0] != "check":
        sys.stderr.write("usage: release_gate_classes.py check <repo root> [<cargo metadata json file>]\n")
        return 2
    root = argv[1]
    md = open(argv[2]).read() if len(argv) > 2 else None
    universe, errs = derive(root, md)
    for e in errs:
        print("DECLINE %s" % e)
    if errs or not universe:
        return 2
    probs = check(universe, load_classes(os.path.join(root, "scripts/release/gate_classes.yaml")))
    for p in probs:
        print("FAIL  %s" % p)
    real = sum(1 for c in load_classes(os.path.join(root, "scripts/release/gate_classes.yaml")).values()
               if isinstance(c, dict) and c.get("class") == "real")
    print("%s  %d publish-blocking gate(s) derived, %d real, %d bookkeeping" % (
        "ok   " if not probs else "RED  ", len(universe), real, len(universe) - real))
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
