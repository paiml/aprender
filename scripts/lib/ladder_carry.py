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
              a path in no crate that the apr build does not read and no receipt producer uses
  IMPACT      every other path in a closure crate (src/, build.rs, Cargo.toml); the root package
              (the `apr` facade) owns src/, build.rs and its test dirs at the workspace root
              a test-dir path that a closure source include_str!s
              Cargo.lock, the root Cargo.toml, rust-toolchain*, .cargo/  (cargo's own build inputs)
              a no-crate path the closure EMBEDS -- derived from include_str!/include_bytes! literals
              and build.rs literals (contracts/, configs/ ...)
              a no-crate path that MEASURES -- tracked scripts writing a ladder/CRUX receipt schema,
              the scripts that invoke them, and everything those reference
              ANY no-crate path when those two sets could not be derived (unknown = impact).
A receipt carries forward iff NO path impacts. The proof names every path and the reason it was
cleared, and the refusal names the first impacting paths.

Pure: the caller hands over the metadata JSON of each endpoint and the changed paths.
"""

import copy
import glob
import os
import posixpath
import re
import subprocess

NO_IMPACT_PREFIXES = ("docs/", "evidence/")
TEST_ONLY_DIRS = ("tests", "benches", "examples")
ROOT_IMPACT = ("Cargo.lock", "Cargo.toml")
#: cargo's OWN build inputs (the toolchain and cargo config), not a project list
CARGO_BUILD_INPUTS = ("rust-toolchain", "rust-toolchain.toml", ".cargo/")
#: the schemas a model-ladder / CRUX receipt is written in; a file naming one PRODUCES measurements
RECEIPT_SCHEMAS = ("apr-model-ladder-receipt/v2", "crux-inference-receipt/v1")
_INCLUDE = re.compile(r'include_(?:str|bytes)!\s*\((.*?)\)\s*[;,)\]]', re.S)
_STRLIT = re.compile(r'"((?:[^"\\]|\\.)*)"')


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


def _norm_repo(root, base_dir, lit):
    """A path literal relative to base_dir -> repo-relative path, or None if it leaves the repo."""
    full = posixpath.normpath(posixpath.join(base_dir, lit))
    return None if full.startswith("..") else full


def embedded_inputs(root, clos, dirs):
    """Repo paths (files or directory PREFIXES ending in /) the apr closure reads at BUILD time, derived
    from the sources: include_str!/include_bytes! literals (and concat! bases) in closure crates, and
    every string literal in a closure crate's build.rs that resolves outside the crate or names a
    top-level repo directory. -> set of repo-relative paths."""
    top = {d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")}
    out = set()
    for n, d in clos.items():
        cdir = d or "."
        for f in glob.glob(os.path.join(root, cdir, "**", "*.rs"), recursive=True):
            rel_dir = posixpath.dirname(posixpath.relpath(f, root))
            try:
                text = open(f, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            is_build = posixpath.basename(f) == "build.rs"
            for m in _INCLUDE.finditer(text):
                arg = m.group(1)
                lits = [x for x in _STRLIT.findall(arg) if "\n" not in x and len(x) < 300]
                if not lits or "\n" in arg.strip().split(",")[0] and len(arg) > 400:
                    continue
                base = cdir if "CARGO_MANIFEST_DIR" in arg else rel_dir
                p = _norm_repo(root, base, lits[0].lstrip("/") if "CARGO_MANIFEST_DIR" in arg else lits[0])
                if p:
                    out.add(p + "/" if (lits[0].endswith("/") or len(lits) > 1 and "$" in arg) else p)
            if is_build:
                for lit in _STRLIT.findall(text):
                    if lit in top:
                        out.add(lit + "/")
                        continue
                    if "/" in lit and ".." in lit:
                        p = _norm_repo(root, cdir, lit.split("{")[0])
                        if p and not p.startswith(cdir + "/"):
                            out.add(p.rstrip("/") + "/" if os.path.isdir(os.path.join(root, p)) else p)
    return {p for p in out if re.fullmatch(r"[A-Za-z0-9_.][A-Za-z0-9_./-]*", p)}


def _tracked(root):
    try:
        out = subprocess.run(["git", "-C", root, "ls-files", "scripts"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [ln for ln in out.splitlines() if ln]


def measurement_inputs(root):
    """TRACKED files that PRODUCE ladder/CRUX receipts -- they WRITE `"schema": "<receipt schema>"` and are
    not a check_* judge or a case fixture -- plus every scripts/... path they reference, plus every
    non-judge script that invokes one of them (a driver), to a fixpoint. -> set of repo paths, or None
    when the tracked set cannot be read (then every no-crate path impacts)."""
    files = _tracked(root)
    if files is None:
        return None
    files = [f for f in files if not posixpath.basename(f).startswith("check_") and "_cases" not in f]
    cache = {}
    def text(f):
        if f not in cache:
            try:
                cache[f] = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
            except OSError:
                cache[f] = ""
        return cache[f]
    writes = re.compile(r'"schema"\s*:\s*"(?:%s)"' % "|".join(re.escape(x) for x in RECEIPT_SCHEMAS))
    writers = {f for f in files if writes.search(text(f))}
    # DRIVERS: a non-judge script that invokes a WRITER directly (one reverse hop, from writers only --
    # a reverse hop from a shared helper would sweep in every script that sources it).
    wnames = {posixpath.basename(x) for x in writers}
    seen = writers | {f for f in files if any(n in text(f) for n in wnames)}
    todo = list(seen)
    while todo:                  # FORWARD only: what a writer or driver references, transitively
        f = todo.pop()
        for m in re.finditer(r"scripts/[A-Za-z0-9_./-]+\.(?:sh|py|json|yaml)", text(f)):
            if m.group(0) in files and m.group(0) not in seen:
                seen.add(m.group(0)); todo.append(m.group(0))
    return seen


#: what the ROOT package (manifest at the workspace root, dir "") owns -- cargo's own target layout
ROOT_PACKAGE_PATHS = ("src", "build.rs") + TEST_ONLY_DIRS


def _owner(path, dirs):
    """The workspace crate whose directory contains `path` (the deepest match), or None. A package at
    the workspace root (the `aprender` facade that builds `apr`) owns only cargo's target layout there."""
    best = None
    for n, d in dirs.items():
        if d in ("", "."):
            continue
        if path == d or path.startswith(d + "/"):
            if best is None or len(d) > len(dirs[best]):
                best = n
    if best is None and path.split("/", 1)[0] in ROOT_PACKAGE_PATHS:
        best = next((n for n, d in dirs.items() if d in ("", ".")), None)
    return best


def _strip_versions(doc):
    """A manifest with every WORKSPACE version removed: [package].version, [workspace.package].version, and
    the `version` of any dependency that is a `path` (a sibling crate's publish version)."""
    doc = copy.deepcopy(doc)
    for t in (doc.get("package"), (doc.get("workspace") or {}).get("package")):
        if isinstance(t, dict):
            t.pop("version", None)
    def walk(x):
        if isinstance(x, dict):
            if "path" in x:
                x.pop("version", None)
            for v in x.values():
                walk(v)
    walk(doc)
    return doc


def version_only(path, before, after):
    """True when `path` (a Cargo.toml or Cargo.lock) changes NOTHING but workspace version numbers between the
    two texts -- a release's version bump. Anything else, or an unparseable/absent side, is False."""
    import tomllib
    if before is None or after is None:
        return False
    if posixpath.basename(path) == "Cargo.lock":
        import ladder_equiv
        return ladder_equiv.lock_dep_change(before, after) is None
    try:
        return _strip_versions(tomllib.loads(before)) == _strip_versions(tomllib.loads(after))
    except (tomllib.TOMLDecodeError, TypeError):
        return False


def impact(path, clos, dirs, embedded=None, measured=None, bumped=()):
    """-> (impacts: bool, why). `embedded`/`measured` None = unknown -> any no-crate path impacts.
    `bumped`: manifests/lockfiles proven to change only workspace version numbers (version_only)."""
    if path in bumped:
        return False, "%s: a workspace version bump only -- no dependency, feature or source changed" % path
    if path.startswith(NO_IMPACT_PREFIXES):
        return False, "docs/evidence: never compiled, never measured"
    if path in ROOT_IMPACT:
        return True, "%s: the workspace/external dependency graph" % path
    if path.startswith(CARGO_BUILD_INPUTS):
        return True, "%s: a cargo build input (toolchain / cargo config)" % path
    owner = _owner(path, dirs)
    if owner is None:
        if embedded is None or measured is None:
            return True, "belongs to no crate, and the build/measurement inputs were not derived"
        hit = next((e for e in embedded if path == e or (e.endswith("/") and path.startswith(e))), None)
        if hit:
            return True, "embedded or read at build time by the apr closure (%s)" % hit
        if path in measured:
            return True, "produces the ladder/CRUX receipts (a measurement input)"
        return False, "belongs to no crate, is not embedded or read by the apr build, and produces no receipt"
    if owner not in clos:
        return False, "crate %s is outside the apr binary's closure" % owner
    rest = path if dirs[owner] in ("", ".") else path[len(dirs[owner]) + 1:]
    hit = next((e for e in embedded or () if path == e or (e.endswith("/") and path.startswith(e))), None)
    if hit and rest.split("/", 1)[0] in TEST_ONLY_DIRS:
        return True, "test-dir path embedded by the apr closure's sources (%s)" % hit
    if rest.split("/", 1)[0] in TEST_ONLY_DIRS:
        return False, "%s/%s is test-only in closure crate %s" % (dirs[owner], rest.split("/", 1)[0], owner)
    return True, "crate %s is in the apr binary's inference closure" % owner


def carry(paths, meta_a, meta_b, roots=None, bumped=()):
    """-> (carries: bool, proof). Both endpoints' closures (and, given checkout `roots`, their build and
    measurement inputs) are unioned; a path any endpoint says impacts, impacts."""
    ca, da = closure(meta_a)
    cb, db = closure(meta_b)
    if not ca and not cb:
        return False, "no package declares an `apr` binary at either endpoint -- the closure is unknown"
    clos = dict(ca)
    clos.update(cb)
    dirs = dict(da)
    dirs.update(db)
    embedded = measured = None
    if roots:
        embedded, measured = set(), set()
        for r in roots:
            embedded |= embedded_inputs(r, clos, dirs)
            mi = measurement_inputs(r)
            if mi is None:
                embedded = measured = None
                break
            measured |= mi
    hits, cleared = [], []
    for p in sorted(set(paths)):
        imp, why = impact(p, clos, dirs, embedded, measured, bumped)
        (hits if imp else cleared).append("%s (%s)" % (p, why))
    if hits:
        return False, "the diff touches the inference path: %s" % "; ".join(hits[:5]) + (
            " (+%d more)" % (len(hits) - 5) if len(hits) > 5 else "")
    return True, "carried forward: %d path(s), none on the inference path -- %s" % (len(cleared), "; ".join(cleared))


def _git(repo, *args):
    return subprocess.run(["git", "-c", "safe.directory=*", "-C", repo, *args], capture_output=True, text=True)


def _checkout_meta(tree):
    r = subprocess.run(["cargo", "metadata", "--no-deps", "--offline", "--format-version", "1",
                        "--manifest-path", os.path.join(tree, "Cargo.toml")], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    import json
    return json.loads(r.stdout)


def carry_between(repo, sha_a, sha_b):
    """Run `carry` for two commits of `repo`: each endpoint is checked out DETACHED in a throwaway worktree
    (its metadata and its sources are read there, never from whatever the caller has checked out), and the
    changed paths are `git diff --name-only A B`. Any failure -> (False, why): unknown never carries."""
    import shutil
    import tempfile
    trees, tmp = [], tempfile.mkdtemp(prefix="ladder-carry-")
    try:
        for i, sha in enumerate((sha_a, sha_b)):
            t = os.path.join(tmp, "t%d" % i)
            r = _git(repo, "worktree", "add", "--detach", "--quiet", t, sha + "^{commit}")
            if r.returncode != 0:
                return False, "cannot check out %s: %s" % (sha[:12], r.stderr.strip()[:200])
            trees.append(t)
        d = _git(repo, "diff", "--name-only", sha_a, sha_b)
        if d.returncode != 0:
            return False, "git diff %s %s failed" % (sha_a[:12], sha_b[:12])
        metas = [_checkout_meta(t) for t in trees]
        if None in metas:
            return False, "cargo metadata failed at an endpoint -- the closure is unknown"
        paths = [p for p in d.stdout.splitlines() if p]
        def text(i, p):
            f = os.path.join(trees[i], p)
            return open(f, encoding="utf-8").read() if os.path.isfile(f) else None
        bumped = {p for p in paths if posixpath.basename(p) in ("Cargo.toml", "Cargo.lock")
                  and version_only(p, text(0, p), text(1, p))}
        # workspace_root differs per worktree; each closure is relative to its own root, so both compare
        return carry(paths, metas[0], metas[1], roots=trees, bumped=bumped)
    finally:
        for t in trees:
            _git(repo, "worktree", "remove", "--force", t)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        sys.exit("usage: ladder_carry.py <repo> <receipt sha> <cut sha>")
    ok, proof = carry_between(sys.argv[1], sys.argv[2], sys.argv[3])
    print(proof)
    sys.exit(0 if ok else 1)
