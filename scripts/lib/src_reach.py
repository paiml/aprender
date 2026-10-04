#!/usr/bin/env python3
"""The crates/*/src .rs files that no package compiles (#3809, #4700).

Called by scripts/check_src_test_files_wired.sh, which owns the baseline and the case table. It is a file of its own,
not a heredoc in the guard, because bashrs parses an embedded heredoc as shell (as resolve_includes.py notes).

Static: it reads the working tree and never builds. The files are what `git ls-files --cached --others
--exclude-standard` lists under the root (tracked or new, not ignored), less any that is gone from disk.

Model. A package is any Cargo.toml with a [package] table: the root one, crates/*, and nested ones such as
crates/X/fuzz. Its roots are src/lib.rs, src/main.rs, build.rs, src/bin/*.rs, src/bin/*/main.rs,
tests|benches|examples/*.rs and */main.rs, every `path = "*.rs"` and `build = "*.rs"` in the manifest, and every
"*.rs" literal in its build script (joined to the package dir and to src/). A file's edges are `mod x;` (also
`mod r#x;`), `include!("p")` and `path = "p"`. They are joined to the file's dir and its non-mod-rs dir (dir/<stem>),
to each of those joined to a non-.rs `#[path = "d"]` literal in the file (the dir of a #[path] inline module), and
to each of all these extended by up to two names of inline `mod n {` blocks. An include!()d file is spliced into its
includer, so its lines are also joined to every such dir of the includer, through chains. A dir carried that way is
kept only if some listed path lies under it: rustc resolves a path through real dirs, and the rule ends an
include!() cycle among dark files. Reached = the closure of every package's roots. Dark = a crates/*/src .rs file
that is not reached, whether or not its crate dir holds a package.

Every approximation errs toward "reached", so a file listed dark is never compiled: only // comments are stripped
(`/* mod x; */` and a `mod x;` inside a string count), #[cfg] and `autotests = false` are ignored, and joins try
more dirs than rustc does. Two gaps err the other way: three or more levels of inline modules, and a `mod` line that
a macro builds (`mod $name;`). A compiled file that only such a line reaches is listed dark. That is a defect of this
model: fix the model, never baseline the file.

Usage: python3 scripts/lib/src_reach.py REPO_ROOT
  prints the dark paths, sorted, one per line. rc 2 usage, rc 3 when no crates/*/src file is listed (a vacuous
  answer is refused), rc 4 when the files cannot be listed or read.
"""
import os
import posixpath as pp
import re
import subprocess
import sys

INC_RE = re.compile(r'include!\s*\(\s*"([^"]+)"\s*\)')
# MOD_RE and PATH_RE must match at a word boundary, which at_word() checks: a leading \b costs re its
# literal-prefix search, about 35x over this tree.
MOD_RE = re.compile(r'mod\s+(?:r#)?(\w+)\s*([;{])')  # `mod x;`, or an inline `mod x {`
PATH_RE = re.compile(r'path\s*=\s*"([^"]+\.rs)"')
PATHDIR_RE = re.compile(r'#\[\s*path\s*=\s*"([^"\n]*)"\s*\]')
BUILD_RE = re.compile(r'\bbuild\s*=\s*"([^"]+\.rs)"')
WORD_RE = re.compile(r'\w')
RS_LIT_RE = re.compile(r'"([^"]+\.rs)"')
PKG_RE = re.compile(r'^\[package\]', re.M)
AUTO_ROOT = r'(src/(lib|main)\.rs|build\.rs|(tests|benches|examples|src/bin)/[^/]+(/main)?\.rs)'
UNIVERSE_RE = re.compile(r'crates/[^/]+/src/.+\.rs')
# A git hook exports these for the repo it runs in; `git -C ROOT` must find ROOT's own repo.
GIT_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX", "GIT_COMMON_DIR")


def listed(root):
    """Every path git lists under root, tracked or new and not ignored, that is still a file."""
    env = {k: v for k, v in os.environ.items() if k not in GIT_ENV}
    out = subprocess.run(["git", "-C", root, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                         capture_output=True, check=True, env=env).stdout
    names = set(os.fsdecode(out).split("\0")) - {""}
    return sorted(f for f in names if os.path.isfile(os.path.join(root, f)))


def read(root, f):
    with open(os.path.join(root, f), encoding="utf-8", errors="replace") as fh:
        return fh.read()


def strip(text):
    return re.sub(r'//[^\n]*', '', text)


def at_word(rx, text):
    """rx's matches in text that begin at a word boundary."""
    return [m for m in rx.finditer(text) if not m.start() or not WORD_RE.match(text, m.start() - 1)]


def scan(text):
    """The edges and dirs a (comment-stripped) file names: `mod x;` names, inline `mod n {` names, include!()
    literals, `path = "*.rs"` literals, and non-.rs #[path] literals."""
    mods = [m.groups() for m in at_word(MOD_RE, text)]
    return {"mods": [n for n, k in mods if k == ";"], "inline": sorted({n for n, k in mods if k == "{"}),
            "incs": INC_RE.findall(text), "paths": [m.group(1) for m in at_word(PATH_RE, text)],
            "pdirs": sorted({p for p in PATHDIR_RE.findall(text) if not p.endswith(".rs")})}


def widen(bases, pdirs, names):
    """bases, and each joined to a #[path] dir literal; each of those extended by one or two inline names."""
    out = set(bases) | {pp.normpath(pp.join(b, p)) for b in bases for p in pdirs}
    one = {pp.join(b, x) for b in out for x in names}
    return out | one | {pp.join(b, y) for b in one for y in names}


def targets(s, dirs, fs):
    """The listed .rs files that the edges of scan s name from any of dirs."""
    out = {pp.normpath(pp.join(d, p)) for p in s["incs"] + s["paths"] for d in dirs}
    out |= {x for m in s["mods"] for d in dirs for x in (pp.join(d, m + ".rs"), pp.join(d, m, "mod.rs"))}
    return out & fs


def find_roots(txt, tomls, pkgs):
    """The root files of every package (see Model above)."""
    roots = set()
    for d in pkgs:
        pre = d + "/" if d else ""
        auto = re.compile(re.escape(pre) + AUTO_ROOT)
        roots |= {f for f in txt if f.startswith(pre) and auto.fullmatch(f)}
        builds = {pre + "build.rs"} | {pp.normpath(pp.join(d, x)) for x in BUILD_RE.findall(tomls[d])}
        roots |= builds | {pp.normpath(pp.join(d, m.group(1))) for m in at_word(PATH_RE, tomls[d])}
        lits = {x for b in builds for x in RS_LIT_RE.findall(txt.get(b, ""))}
        roots |= {pp.normpath(pp.join(d, s, x)) for x in lits for s in ("", "src")}
    return roots & set(txt)


def all_dirs(paths):
    """The repo root and every dir that holds a listed path."""
    out = {""}
    for f in paths:
        d = pp.dirname(f)
        while d not in out:
            out.add(d)
            d = pp.dirname(d)
    return out


def includes(scans):
    """{file: the other files its include!()s name}."""
    return {f: {g for g in (pp.normpath(pp.join(pp.dirname(f), p)) for p in s["incs"]) if g in scans and g != f}
            for f, s in scans.items()}


def splice(scans, alldirs):
    """{file: the dirs its edges are joined to, before widen()}: its dir and its non-mod-rs dir, and every widened dir
    of an includer that lies in alldirs."""
    ctx = {f: {pp.dirname(f), pp.join(pp.dirname(f), pp.splitext(pp.basename(f))[0])} for f in scans}
    incs = {f: gs for f, gs in includes(scans).items() if gs}
    changed = True
    while changed:  # an include!()d file is spliced into its includer's module (inline blocks too), through chains
        changed = False
        for f, gs in incs.items():
            for g in gs:
                add = (widen(ctx[f], scans[f]["pdirs"], scans[f]["inline"]) & alldirs) - ctx[g]
                if add:
                    ctx[g] |= add
                    changed = True
    return ctx


def closure(roots, edges):
    reach, todo = set(roots), list(roots)
    while todo:
        for g in edges[todo.pop()] - reach:
            reach.add(g)
            todo.append(g)
    return reach


def reached(files, paths):
    """The .rs files of files ({path: text} for every listed .rs file and Cargo.toml) that some package compiles."""
    txt = {f: strip(t) for f, t in files.items() if f.endswith(".rs")}
    fs = set(txt)
    tomls = {pp.dirname(f): t for f, t in files.items() if pp.basename(f) == "Cargo.toml"}
    pkgs = {d for d, t in tomls.items() if PKG_RE.search(t)}
    scans = {f: scan(t) for f, t in txt.items()}
    ctx = splice(scans, all_dirs(paths))
    edges = {f: targets(s, widen(ctx[f], s["pdirs"], s["inline"]), fs) - {f} for f, s in scans.items()}
    return closure(find_roots(txt, tomls, pkgs), edges)


def dark(root):
    """(the crates/*/src .rs files listed under root, the sorted ones of them that no package compiles)."""
    paths = listed(root)
    files = {f: read(root, f) for f in paths if f.endswith(".rs") or pp.basename(f) == "Cargo.toml"}
    universe = [f for f in files if UNIVERSE_RE.fullmatch(f)]
    reach = reached(files, paths)
    return universe, sorted(f for f in universe if f not in reach)


def main(argv):
    if len(argv) != 2:
        print("usage: src_reach.py REPO_ROOT", file=sys.stderr)
        return 2
    try:
        universe, out = dark(argv[1])
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"src_reach: cannot list or read the files under {argv[1]}: {e}", file=sys.stderr)
        return 4
    if not universe:
        print(f"src_reach: no crates/*/src .rs file is listed under {argv[1]}; refusing a vacuous answer",
              file=sys.stderr)
        return 3
    sys.stdout.write("".join(f + "\n" for f in out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
