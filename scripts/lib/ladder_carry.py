"""ladder_carry.py -- may a model-ladder / CRUX receipt measured at commit A be carried forward to B? (#4037)

Part of #4033 (faster certification), lever (d). A receipt measures what the `apr` binary did. It can
stand for a later commit only when NOTHING in the diff A..B can change that binary's inference. The
#4022/#4032 hotfix scope answered that with a hand-listed allow-list for one cut; this answers it for
every diff, from the CRATE GRAPH, never from a list.

THE INFERENCE CLOSURE. From `cargo metadata` (workspace packages): every package that declares an
`apr` binary target is a root, and the closure is every workspace package reachable from a root
through normal or build dependencies -- dev-dependencies excluded (they do not link into the binary),
OPTIONAL dependencies included (a feature can turn them on, and the gate cannot assume it is off).
Computed at BOTH endpoints and unioned, so a crate that joins the closure at B still counts.

THE PATH-IMPACT RULE, per changed path, conservative by construction:
  no impact   docs/**, evidence/**                          (never compiled, never measured)
              a crate OUTSIDE the closure                   (not linked into apr)
              tests/, benches/, examples/ of a closure crate (not linked into the binary)
  IMPACT      every other path in a closure crate (src/, build.rs, Cargo.toml, contracts
              include_str!'d, ...)
              Cargo.lock and the root Cargo.toml            (external dependencies, workspace graph)
              ANY path that belongs to no known crate and is not docs/evidence -- scripts/,
              contracts/, rust-toolchain.toml, .github/ ... : nothing proves it cannot change
              what is built or what is measured, so it is treated as if it can.
A receipt carries forward iff NO path impacts. The proof names every path and the reason it was
cleared, and the refusal names the first impacting paths.

Pure: the caller hands over the metadata JSON of each endpoint and the changed paths.
"""

import posixpath

NO_IMPACT_PREFIXES = ("docs/", "evidence/")
TEST_ONLY_DIRS = ("tests", "benches", "examples")
ROOT_IMPACT = ("Cargo.lock", "Cargo.toml")


def _rel(path, root):
    rel = posixpath.relpath(path, root) if path.startswith("/") else path
    return rel.replace("\\", "/")


def closure(metadata):
    """-> {package name: crate dir relative to the workspace root} for the `apr` binary's closure,
    plus {name: dir} of ALL workspace packages (the second tells a known crate from an unknown path)."""
    root = metadata.get("workspace_root") or ""
    pk = {p["name"]: p for p in metadata.get("packages") or []}
    dirs = {n: posixpath.dirname(_rel(p["manifest_path"], root)) for n, p in pk.items()}
    roots = [n for n, p in pk.items() for t in p.get("targets") or []
             if "bin" in (t.get("kind") or []) and t.get("name") == "apr"]
    seen, stack = set(), list(roots)
    while stack:
        n = stack.pop()
        if n in seen or n not in pk:
            continue
        seen.add(n)
        for d in pk[n].get("dependencies") or []:
            if d.get("kind") == "dev":
                continue
            if d.get("path") and d["name"] in pk:
                stack.append(d["name"])
    return {n: dirs[n] for n in seen}, dirs


def _owner(path, dirs):
    """The workspace crate whose directory contains `path` (the deepest match), or None."""
    best = None
    for n, d in dirs.items():
        if d in ("", "."):
            continue
        if path == d or path.startswith(d + "/"):
            if best is None or len(d) > len(dirs[best]):
                best = n
    return best


def impact(path, clos, dirs):
    """-> (impacts: bool, why)."""
    if path.startswith(NO_IMPACT_PREFIXES):
        return False, "docs/evidence: never compiled, never measured"
    if path in ROOT_IMPACT:
        return True, "%s: the workspace/external dependency graph" % path
    owner = _owner(path, dirs)
    if owner is None:
        return True, "belongs to no crate: nothing proves it cannot change what is built or measured"
    if owner not in clos:
        return False, "crate %s is outside the apr binary's closure" % owner
    rest = path[len(dirs[owner]) + 1:]
    if rest.split("/", 1)[0] in TEST_ONLY_DIRS:
        return False, "%s/%s is test-only in closure crate %s" % (dirs[owner], rest.split("/", 1)[0], owner)
    return True, "crate %s is in the apr binary's inference closure" % owner


def carry(paths, meta_a, meta_b):
    """-> (carries: bool, proof). Both endpoints' closures are unioned; a path any endpoint says
    impacts, impacts."""
    ca, da = closure(meta_a)
    cb, db = closure(meta_b)
    if not ca and not cb:
        return False, "no package declares an `apr` binary at either endpoint -- the closure is unknown"
    clos = dict(ca)
    clos.update(cb)
    dirs = dict(da)
    dirs.update(db)
    hits, cleared = [], []
    for p in sorted(set(paths)):
        imp, why = impact(p, clos, dirs)
        (hits if imp else cleared).append("%s (%s)" % (p, why))
    if hits:
        return False, "the diff touches the inference path: %s" % "; ".join(hits[:5]) + (
            " (+%d more)" % (len(hits) - 5) if len(hits) > 5 else "")
    return True, "carried forward: %d path(s), none on the inference path -- %s" % (len(cleared), "; ".join(cleared))
