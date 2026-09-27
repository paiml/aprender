#!/usr/bin/env python3
"""FLOW-003 QM-11 (#4529): the tier router. It decides which tier a diff owes.

Spec: docs/specifications/FLOW-003-queue-and-release-cycle-model.md §T.2
(Def. 14, Thms 14 and 15); contract ci-tier-router-v1. It reads the input set
I = I_dep ∪ F ∪ C that QM-10 writes (scripts/ci/input_set.sh, ci-input-set-v1)
and never writes one.

  T0         D ∩ I = ∅ and I is valid at the base (Thm 14): lint only.
  selective  D ∩ C = ∅: build the whole workspace, run the tests of R' (Thm 15).
  T3         everything else, and every case the router cannot prove.

The router FAILS CLOSED. Each case it cannot decide is T3 with a reason, never
T0 or a narrower R':
  - no input set, or one `input_set.sh check` rejects;
  - a changed path in C (manifests, lockfile, toolchain, .cargo, workflows,
    proc-macro and rerun-less build-script packages);
  - a changed path in F (read, enumerated or probed at run time). ci-input-set-v1
    records F for the whole test run, not per package, so F_X of Thm 15 is not
    known and no R' can be proved smaller than the workspace;
  - a changed path in I_dep that no package owns.

Cargo target AUTODISCOVERY is an input Def. 14 does not name. Cargo lists
src/bin/, tests/, examples/ and benches/ to find targets, and neither dep-info
nor the test trace records that listing. So a PR that only ADDS
crates/x/tests/new.rs is in no term of I, and without the rule below it would
route to T0 while it adds a test binary. A path under a package's src/, tests/,
examples/, benches/ or its build.rs is therefore owned by that package whether
or not I names it.

usage:
  tier_router.py route --input-set I.json --metadata M.json --changed FILE
                       [--owners O.json] [--base-valid] [--stale FILE]
      FILE: changed repo paths, one per line (both sides of a rename).
      M.json: `cargo metadata --format-version 1 --no-deps`.
      O.json: {path: [package, ...]} from `owners`; without it a path in I_dep
              is owned only by the package whose src/ etc. contains it.
      --base-valid: `input_set.sh base` exited 0 for this base (INV-BASE).
              Without it T0 is never returned (R2-8).
      --stale FILE: paths changed between the nightly and the base, when
              INV-BASE did not hold; their owners' closure joins R'.
      Prints {"tier", "packages", "reasons"}; exit 0.
  tier_router.py owners <root> <target-dir> --metadata M.json
      {path: [package, ...]} from rustc dep-info under <target-dir>.
  tier_router.py --self-test
      The §QM-11 falsifiers and the controls, then planted mutants that must
      each turn a row RED.

exit: 0 ok · 1 a fault · 64 usage
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

AUTODISCOVERY = ("src/", "tests/", "examples/", "benches/")
AUTODISCOVERY_FILES = ("build.rs",)
INPUT_SET_KEYS = {"schema", "baseSha", "dep", "read", "dirPrefix", "absent", "config",
                  "traceSyscalls", "nightlies", "nightlyShas"}
# ROUTER_MUTATE plants one weakened rule for the self-test; unset in real runs.
MUTATE = os.environ.get("ROUTER_MUTATE", "")


def prefix_match(entries, path):
    """ci-input-set-v1 inI for pattern entries: `dir/**` covers the prefix, `**` all."""
    for e in entries:
        if e.endswith("**"):
            pre = e[:-2]
            if pre == "" or path.startswith(pre):
                return True
        elif e == path:
            return True
    return False


class InputSet:
    def __init__(self, doc):
        missing = INPUT_SET_KEYS - set(doc)
        if missing:
            raise ValueError(f"input set lacks {sorted(missing)}")
        self.dep = set(doc["dep"])
        self.read = set(doc["read"]) | set(doc["absent"])
        self.dir_prefix = list(doc["dirPrefix"])
        self.config = list(doc["config"])

    def in_c(self, p):
        return prefix_match(self.config, p)

    def in_f(self, p):
        if MUTATE == "ignore-f":
            return False
        return p in self.read or prefix_match(self.dir_prefix, p)

    def in_dep(self, p):
        return p in self.dep

    def in_i(self, p):
        return self.in_c(p) or self.in_f(p) or self.in_dep(p)


class Workspace:
    """Package dirs and the reverse-dependency graph of the workspace members."""

    def __init__(self, meta, root=None):
        root = Path(root or meta["workspace_root"])
        self.dirs = {}
        by_name = {}
        for p in meta["packages"]:
            d = Path(p["manifest_path"]).parent
            rel = "" if d == root else d.relative_to(root).as_posix() + "/"
            self.dirs[p["name"]] = rel
            by_name[p["name"]] = p
        self.rdeps = {n: set() for n in by_name}
        for n, p in by_name.items():
            for dep in p.get("dependencies", []):
                # Normal, dev and build edges all count (Thm 15); only members matter.
                if dep["name"] in by_name and (MUTATE != "drop-dev-edges" or dep.get("kind") != "dev"):
                    self.rdeps[dep["name"]].add(n)
        self.targets = {}
        for n, p in by_name.items():
            for t in p.get("targets", []):
                self.targets.setdefault(t["name"].replace("-", "_"), n)

    def autodiscovery_owner(self, path):
        """The package whose src/, tests/, examples/, benches/ or build.rs holds path."""
        best = None
        for name, d in self.dirs.items():
            if not path.startswith(d):
                continue
            tail = path[len(d):]
            if tail.startswith(AUTODISCOVERY) or tail in AUTODISCOVERY_FILES:
                if best is None or len(d) > len(self.dirs[best]):
                    best = name
        return best

    def closure(self, seeds):
        out, todo = set(seeds), list(seeds)
        while todo:
            for r in self.rdeps.get(todo.pop(), ()):
                if r not in out:
                    out.add(r)
                    todo.append(r)
        return out


def owners_of(path, ws, owners_map):
    got = set(owners_map.get(path, ()))
    a = ws.autodiscovery_owner(path)
    if a:
        got.add(a)
    return got


def route(iset, ws, changed, owners_map, base_valid, stale):
    if iset is None:
        return {"tier": "T3", "packages": sorted(ws.dirs), "reasons": ["no valid input set: fail closed"]}
    reasons, t3, seeds, touches_i = [], False, set(), False
    for p in changed:
        owners = owners_of(p, ws, owners_map)
        if iset.in_c(p):
            t3 = True
            reasons.append(f"T3: {p} is in C")
        elif iset.in_f(p):
            t3 = True
            reasons.append(f"T3: {p} is in F (read at run time; F is not per package)")
        elif iset.in_dep(p):
            touches_i = True
            if owners:
                seeds |= owners
                reasons.append(f"selective: {p} in I_dep, owned by {sorted(owners)}")
            else:
                t3 = True
                reasons.append(f"T3: {p} is in I_dep and no package owns it")
        elif owners and MUTATE != "drop-autodiscovery":
            touches_i = True
            seeds |= owners
            reasons.append(f"selective: {p} is a cargo target path of {sorted(owners)} (autodiscovery)")
        else:
            reasons.append(f"T0: {p} is not in I")
    if t3:
        return {"tier": "T3", "packages": sorted(ws.dirs), "reasons": reasons}
    if not touches_i:
        if base_valid or MUTATE == "ignore-base":
            return {"tier": "T0", "packages": [], "reasons": reasons}
        reasons.append("selective: D ∩ I = ∅ but INV-BASE did not hold; the stale paths decide R'")
    for p in stale:
        if iset.in_c(p) or iset.in_f(p):
            return {"tier": "T3", "packages": sorted(ws.dirs),
                    "reasons": reasons + [f"T3: stale {p} (nightly..base) is in C or F"]}
        o = owners_of(p, ws, owners_map)
        if iset.in_dep(p) and not o:
            return {"tier": "T3", "packages": sorted(ws.dirs),
                    "reasons": reasons + [f"T3: stale {p} is in I_dep and no package owns it"]}
        seeds |= o
    packages = ws.closure(seeds) if MUTATE != "no-closure" else set(seeds)
    return {"tier": "selective", "packages": sorted(packages), "reasons": reasons}


def owners_from_depinfo(root, target_dir, ws):
    """{repo path: [package]} from every rustc .d file under target_dir."""
    root = Path(root).resolve()
    out = {}
    for d in Path(target_dir).rglob("*.d"):
        pkg = None
        if d.parent.name == "deps":
            pkg = ws.targets.get(d.stem.rsplit("-", 1)[0])
        elif d.parent.parent.name == "build":
            name = d.parent.name.rsplit("-", 1)[0]
            pkg = name if name in ws.dirs else None
        if pkg is None:
            continue
        for line in d.read_text(errors="replace").splitlines():
            if ":" not in line or line.startswith("#"):
                continue
            for dep in line.split(":", 1)[1].replace("\\ ", "\0").split():
                p = Path(dep.replace("\0", " "))
                if not p.is_absolute():
                    continue
                try:
                    rel = p.resolve().relative_to(root).as_posix()
                except ValueError:
                    continue
                if not rel.startswith("target/"):
                    out.setdefault(rel, set()).add(pkg)
    return {k: sorted(v) for k, v in sorted(out.items())}


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _fixture():
    root = "/w"
    def pkg(name, d, deps=(), dev=()):
        return {"name": name, "manifest_path": f"{root}/{d}Cargo.toml",
                "targets": [{"name": name}],
                "dependencies": [{"name": x, "kind": None} for x in deps]
                + [{"name": x, "kind": "dev"} for x in dev]}
    meta = {"workspace_root": root, "packages": [
        pkg("facade", "", deps=["core"]),
        pkg("core", "crates/core/"),
        pkg("serve", "crates/serve/", deps=["core"]),
        pkg("cli", "crates/cli/", deps=["serve"]),
        pkg("bench", "crates/bench/", dev=["serve"]),
        pkg("lonely", "crates/lonely/"),
    ]}
    iset = {"schema": "ci-input-set-v1", "baseSha": "0" * 40, "nightlies": 7, "nightlyShas": [],
            "traceSyscalls": [],
            "dep": ["crates/core/src/lib.rs", "crates/core/README.md", "crates/serve/src/lib.rs",
                    "crates/lonely/src/lib.rs", "docs/embedded.md", "vendor/blob.bin"],
            "read": ["crates/cli/tests/data/golden.txt"],
            "absent": ["docs/probe.md"],
            "dirPrefix": ["docs/specifications/**"],
            "config": ["Cargo.toml", "Cargo.lock", "crates/core/Cargo.toml", ".github/workflows/**"]}
    owners = {"crates/core/README.md": ["core"], "docs/embedded.md": ["serve"]}
    return meta, iset, owners


def self_test():
    meta, iset_doc, owners = _fixture()
    rows = [
        # (name, changed, base_valid, stale, want tier, want packages or None)
        ("docs-only PR runs T0 only", ["docs/guide.md"], True, [], "T0", []),
        ("docs-only but INV-BASE failed is never T0", ["docs/guide.md"], False, [], "selective", []),
        ("include_str'd README runs selective", ["crates/core/README.md"], True, [], "selective",
         ["bench", "cli", "core", "facade", "serve"]),
        ("Cargo.toml change runs T3", ["crates/core/Cargo.toml"], True, [], "T3", None),
        ("workflow change runs T3", [".github/workflows/ci.yml"], True, [], "T3", None),
        ("change to crate X runs exactly R'(X)", ["crates/serve/src/lib.rs"], True, [], "selective",
         ["bench", "cli", "serve"]),
        ("a leaf crate runs only itself", ["crates/lonely/src/lib.rs"], True, [], "selective", ["lonely"]),
        ("a docs file include_str'd by another crate is owned by it", ["docs/embedded.md"], True, [],
         "selective", ["bench", "cli", "serve"]),
        ("a file a test reads is T3", ["crates/cli/tests/data/golden.txt"], True, [], "T3", None),
        ("a file a test probes for is T3", ["docs/probe.md"], True, [], "T3", None),
        ("a file added to an enumerated dir is T3", ["docs/specifications/new.md"], True, [], "T3", None),
        ("an added test target (autodiscovery) is not T0", ["crates/lonely/tests/new.rs"], True, [],
         "selective", ["lonely"]),
        ("an added bin target (autodiscovery) is not T0", ["crates/cli/src/bin/x.rs"], True, [],
         "selective", ["cli"]),
        ("a crate README nothing includes is T0", ["crates/lonely/README.md"], True, [], "T0", []),
        ("stale nightly..base code joins R'", ["docs/guide.md"], False, ["crates/lonely/src/lib.rs"],
         "selective", ["lonely"]),
        ("stale nightly..base manifest is T3", ["docs/guide.md"], False, ["Cargo.lock"], "T3", None),
        ("an unowned I_dep path is T3", ["vendor/blob.bin"], True, [], "T3", None),
    ]
    ws = Workspace(meta)
    iset = InputSet(iset_doc)
    failed = 0
    for name, changed, bv, stale, tier, pkgs in rows:
        got = route(iset, ws, changed, owners, bv, stale)
        ok = got["tier"] == tier and (pkgs is None or got["packages"] == pkgs)
        if tier == "T3":
            ok = ok and got["packages"] == sorted(ws.dirs)
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  got {got['tier']} {got['packages']}"))
    got = route(None, ws, ["docs/guide.md"], owners, True, [])
    ok = got["tier"] == "T3"
    failed += not ok
    print(f"{'PASS' if ok else 'FAIL'}  no input set is T3 (fail closed)")
    try:
        InputSet({"dep": []})
        ok = False
    except ValueError:
        ok = True
    failed += not ok
    print(f"{'PASS' if ok else 'FAIL'}  an input set missing keys is refused")
    return failed, len(rows) + 2


def self_test_mutants():
    """Each planted weakening must turn at least one row RED."""
    import subprocess
    failed = 0
    for m in ("ignore-f", "drop-dev-edges", "drop-autodiscovery", "ignore-base", "no-closure"):
        r = subprocess.run([sys.executable, __file__, "--self-test-rows"], env={**os.environ, "ROUTER_MUTATE": m},
                           capture_output=True, text=True)
        red = r.returncode != 0
        failed += not red
        print(f"{'PASS' if red else 'FAIL'}  mutant {m} turns the case table RED")
    return failed


def main(argv):
    if argv[:1] == ["--self-test-rows"]:
        failed, total = self_test()
        return 1 if failed else 0
    if argv[:1] == ["--self-test"]:
        failed, total = self_test()
        mf = self_test_mutants()
        print(f"tier_router self-test: {total - failed}/{total} rows, {5 - mf}/5 mutants RED")
        return 1 if failed or mf else 0
    if argv[:1] == ["route"]:
        a = _opts(argv[1:], {"--input-set", "--metadata", "--changed", "--owners", "--stale"}, {"--base-valid"})
        if not {"--input-set", "--metadata", "--changed"} <= set(a):
            print("tier_router: route needs --input-set, --metadata and --changed", file=sys.stderr)
            return 64
        meta = json.loads(Path(a["--metadata"]).read_text())
        ws = Workspace(meta)
        try:
            iset = InputSet(json.loads(Path(a["--input-set"]).read_text()))
        except (OSError, ValueError, KeyError) as e:
            print(f"tier_router: input set unusable ({e}); routing T3", file=sys.stderr)
            iset = None
        owners = json.loads(Path(a["--owners"]).read_text()) if "--owners" in a else {}
        changed = _lines(a["--changed"])
        stale = _lines(a["--stale"]) if "--stale" in a else []
        print(json.dumps(route(iset, ws, changed, owners, "--base-valid" in a, stale), indent=1))
        return 0
    if argv[:1] == ["owners"] and len(argv) >= 3:
        a = _opts(argv[3:], {"--metadata"}, set())
        if "--metadata" not in a:
            print("tier_router: owners needs --metadata", file=sys.stderr)
            return 64
        ws = Workspace(json.loads(Path(a["--metadata"]).read_text()), root=Path(argv[1]).resolve())
        print(json.dumps(owners_from_depinfo(argv[1], argv[2], ws), indent=1))
        return 0
    print(__doc__, file=sys.stderr)
    return 64


def _lines(f):
    return [l.strip() for l in Path(f).read_text().splitlines() if l.strip()]


def _opts(argv, valued, flags):
    out, i = {}, 0
    while i < len(argv):
        k = argv[i]
        if k in flags:
            out[k] = True
            i += 1
        elif k in valued and i + 1 < len(argv):
            out[k] = argv[i + 1]
            i += 2
        else:
            print(f"tier_router: unknown or incomplete option {k}", file=sys.stderr)
            sys.exit(64)
    return out


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
