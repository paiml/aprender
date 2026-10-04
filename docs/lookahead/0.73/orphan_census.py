#!/usr/bin/env python3
"""Orphan census for docs/lookahead/0.73 (#3999): the .rs files under crates/*/src that no package compiles.

Static: it reads git objects at a pin and never builds. It is the evidence for R5 F-R5-6 and for the S5 body in
ticket-bodies-side-fixes.md.

Model. A package is any Cargo.toml with a [package] table: the root one, crates/*, and nested ones such as
crates/X/fuzz. Its roots are src/lib.rs, src/main.rs, build.rs, src/bin/*.rs, src/bin/*/main.rs,
tests|benches|examples/*.rs and */main.rs, every `path = "*.rs"` in the manifest, and every "*.rs" literal in its
build.rs (joined to the package dir and to src/). A file's edges are `mod x;` (also `mod r#x;`), `include!("p")`
and `path = "p"`. They are joined to the file's dir and its non-mod-rs dir (dir/<stem>), each extended by up to two
names of inline `mod n {` blocks in the file. An include!()d file is spliced into its includer, so its lines are
also joined to every dir of the includer, through chains. A dir carried in that way is kept only if some tracked
file lies under it: rustc resolves a path through real dirs, and the rule ends an include!() cycle among dark
files. Reached = the closure of every package's roots. An
orphan is an unreached crates/*/src file of a package. The src files of a crates/* dir that has no package (a
virtual manifest, or no manifest) are reported apart.

Each approximation errs toward "reached", so a listed orphan never compiles. Only // comments are stripped, so
`/* mod x; */` and a `mod x;` inside a string count. #[cfg] and autotests = false are ignored. Joins try more dirs
than rustc does. Three gaps err the other way: three or more inline levels, a macro-built `mod` line, and a
`#[path]` dir on an inline module. The report lists every gap site of the last two kinds in a reached file (a
`mod $name`, and a `#[path]` that names no .rs file). The suspects column names every reached file that declares
`mod <stem>;` from a dir whose path down to the orphan is made of that file's inline module names (gap 1), and
every reached file with an include!/path literal that can name the orphan from a dir at or below one of the
file's dirs (`can_name`; gap 3 when the inner item has a .rs literal). Read each suspect and each gap site.

Guard classes follow scripts/check_src_test_files_wired.sh (#3809):
- baseline: listed in scripts/src_test_files_unwired_baseline.txt.
- test-dark-parent: has a test attribute by the guard's regex, is not listed, and only dark files declare it. That
  is the guard's one-level limit.
- test-unlisted: the guard would fail on it; 0 is expected.
- no-test: outside the guard's universe.
Declarer class: declared-by-dark when some file has an edge to it (none of them is reached), else undeclared.
Twin: a reached file that holds the dark file's code. `=path` has the same text; `~path` has the same text once
// comments and all whitespace are removed (`flat`). A file with no code left after that has no twin. Deleting a
`=` twin's dark copy loses nothing, and deleting a `~` one loses only its // comments.

Usage:
  python3 docs/lookahead/0.73/orphan_census.py --pin SHA [--crate crates/NAME ...] [--tsv OUT]
  python3 docs/lookahead/0.73/cite_drift.py --list | python3 docs/lookahead/0.73/orphan_census.py --cites -
  python3 docs/lookahead/0.73/orphan_census.py --self-test
Any directory of the checkout works as the cwd. rc 3 when the pin lists no crates/*/src .rs file (with --cites,
when a cited pin does): a census of nothing is refused, never printed as 0 orphans.
"""
import argparse
import contextlib
import io
import os
import posixpath as pp
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

TEST_RE = re.compile(r'#\[(tokio::)?test(\]|\()|#\[test_case|#\[rstest|proptest! *\{')  # the #3809 guard's regex
INC_RE = re.compile(r'include!\(\s*"([^"]+)"\s*\)')
PATH_RE = re.compile(r'\bpath\s*=\s*"([^"]+\.rs)"')
MOD_RE = re.compile(r'\bmod\s+(?:r#)?(\w+)\s*;')
INLINE_RE = re.compile(r'\bmod\s+(?:r#)?(\w+)\s*\{')
RS_LIT_RE = re.compile(r'"([^"]+\.rs)"')
PATHATTR_RE = re.compile(r'#\[path\s*=\s*"([^"\n]*)"\s*\]')
MACMOD_RE = re.compile(r'\bmod\s+\$\w+')
PKG_RE = re.compile(r'^\[package\]', re.M)
AUTO_ROOT = r'(src/(lib|main)\.rs|build\.rs|(tests|benches|examples|src/bin)/[^/]+(/main)?\.rs)'
UNIVERSE_RE = re.compile(r'(crates/[^/]+)/src/.+\.rs')
CITE_RE = re.compile(r'^\s+(\S+:\d+) .*? = ([^\s:]+)(?::\S+)? @([0-9a-f]{7,40})(?: -> \S+)?\s*$')
BASELINE = "scripts/src_test_files_unwired_baseline.txt"


def strip(text):
    return re.sub(r'//[^\n]*', '', text)


def modname(f):
    """(module name, the dir that a `mod name;` is resolved from)."""
    if pp.basename(f) == "mod.rs":
        return pp.basename(pp.dirname(f)), pp.dirname(pp.dirname(f))
    return pp.splitext(pp.basename(f))[0], pp.dirname(f)


def expand(bases, names):
    """bases, each extended by one or two names of inline `mod n {` blocks."""
    return set(bases) | {pp.join(b, x) for b in bases for x in names} | \
        {pp.join(b, x, y) for b in bases for x in names for y in names}


def depth(d):
    return d.count("/") + 1 if d else 0


def under(x, b):
    """x is b or lies below it ("" is the repo root)."""
    return not b or x == b or x.startswith(b + "/")


def can_name(lit, bases, target):
    """Can the include!/path literal `lit` name `target` when joined to a dir at or below one of `bases`? Its path
    past its k leading `..`s must end `target`, and the rest of `target` (the anchor) must lie under a base, or a
    base under the anchor at most k levels down."""
    parts = pp.normpath(lit).split("/")
    k = next((i for i, x in enumerate(parts) if x != ".."), len(parts))
    tail = "/".join(parts[k:])
    if not tail or not ("/" + target).endswith("/" + tail):
        return False
    anchor = target[:len(target) - len(tail)].rstrip("/")
    return any(under(anchor, b) or (under(b, anchor) and depth(b) - depth(anchor) <= k) for b in bases)


def targets(text, dirs, fs):
    out = {pp.normpath(pp.join(d, p)) for p in INC_RE.findall(text) + PATH_RE.findall(text) for d in dirs}
    for m in MOD_RE.findall(text):
        out |= {x for d in dirs for x in (pp.join(d, m + ".rs"), pp.join(d, m, "mod.rs"))}
    return out & fs


def find_roots(txt, tomls, pkgs):
    """The root files of every package (see Model above)."""
    roots = set()
    for d in pkgs:
        pre = d + "/" if d else ""
        auto = re.compile(re.escape(pre) + AUTO_ROOT)
        roots |= {f for f in txt if auto.fullmatch(f)}
        roots |= {pp.normpath(pp.join(d, x)) for x in PATH_RE.findall(tomls[d])}
        build = txt.get(pre + "build.rs", "")
        roots |= {pp.normpath(pp.join(d, s, x)) for x in RS_LIT_RE.findall(build) for s in ("", "src")}
    return roots & set(txt)


def all_dirs(paths):
    """The repo root and every dir that holds a tracked path."""
    out = {""}
    for f in paths:
        d = pp.dirname(f)
        while d not in out:
            out.add(d)
            d = pp.dirname(d)
    return out


def includes(txt):
    """{file: the other files its include!()s name}."""
    return {f: {g for g in (pp.normpath(pp.join(pp.dirname(f), p)) for p in INC_RE.findall(t)) if g in txt and g != f}
            for f, t in txt.items()}


def splice(txt, inline, alldirs):
    """{file: the dirs its module lines are joined to, before the inline names}: its dir and its non-mod-rs dir, and
    every such dir of an includer that lies in alldirs."""
    ctx = {f: {pp.dirname(f), pp.join(pp.dirname(f), pp.splitext(pp.basename(f))[0])} for f in txt}
    incs = includes(txt)
    changed = True
    while changed:  # an include!()d file is spliced into its includer's module (inline blocks too), through chains
        changed = False
        for f, gs in incs.items():
            for g in gs:
                add = (expand(ctx[f], inline[f]) & alldirs) - ctx[g]
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


def index(txt, edges, reach):
    """declarers {file: files with an edge to it}, by_mod {name: reached files with `mod name;`} and by_base
    {basename: (reached file, include!/path literal) pairs}."""
    declarers, by_mod, by_base = defaultdict(set), defaultdict(set), defaultdict(set)
    for f, out in edges.items():
        for g in out:
            declarers[g].add(f)
    for f in reach:
        for m in MOD_RE.findall(txt[f]):
            by_mod[m].add(f)
        for p in INC_RE.findall(txt[f]) + PATH_RE.findall(txt[f]):
            by_base[pp.basename(p)].add((f, p))
    return declarers, by_mod, by_base


def suspects_of(f, ctx, inline, by_mod, by_base):
    """The suspects column of the orphan f (see the docstring)."""
    name, mdir = modname(f)
    sus = {g for g, p in by_base[pp.basename(f)] if can_name(p, ctx[g], f)}
    for g in by_mod[name]:
        for b in ctx[g]:
            rel = pp.relpath(mdir, b) if b else mdir
            if not rel.startswith("..") and rel != "." and set(rel.split("/")) <= set(inline[g]):
                sus.add(g)
    return sorted(sus)


def orphan_columns(f, text, baseline, dec):
    """The guard and declarer classes of the orphan f, which the files dec declare."""
    if f in baseline:
        guard = "baseline"
    elif not TEST_RE.search(text):
        guard = "no-test"
    else:
        guard = "test-dark-parent" if dec else "test-unlisted"
    return dict(guard=guard, declarer="declared-by-dark" if dec else "undeclared", declarers=dec)


def dark_rows(files, dark, tomls, pkgs):
    """A row per crates/*/src file in dark: crate, path, lines and status."""
    rows = []
    for f in sorted(dark):
        u = UNIVERSE_RE.fullmatch(f)
        if u:
            c = u.group(1)
            rows.append(dict(crate=c, path=f, lines=len(files[f].splitlines()),
                             status="no-manifest" if c not in tomls else "virtual" if c not in pkgs else "orphan"))
    return rows


def gap_sites(txt, reach):
    """Gaps 2 and 3 in reached files: a macro-built `mod $name`, and a `#[path]` that names no .rs file."""
    return sorted({f'{f}: #[path = "{p}"]' for f in reach for p in PATHATTR_RE.findall(txt[f]) if not p.endswith(".rs")}
                  | {f"{f}: {m}" for f in reach for m in MACMOD_RE.findall(txt[f])})


def flat(text):
    """The text without // comments and whitespace: two files that differ only there have the same flat()."""
    return re.sub(r"\s+", "", strip(text))


def twin_index(files, reach):
    """({text: reached files}, {flat text: reached files}), each list sorted."""
    same, flats = defaultdict(list), defaultdict(list)
    for f in sorted(reach):
        same[files[f]].append(f)
        flats[flat(files[f])].append(f)
    return same, flats


def twin_of(text, same, flats):
    """'=' + the first reached file with the same text, else '~' + the first with the same flat text, else ''."""
    key = flat(text)
    if not key:
        return ""
    return "=" + same[text][0] if same.get(text) else "~" + flats[key][0] if flats.get(key) else ""


def census(files, baseline=frozenset(), tree=None):
    """files: {repo path: text} with every .rs file and Cargo.toml in scope. Returns one row per crates/*/src
    file that is not reached: status orphan, virtual or no-manifest, and twin, plus the classes and suspects of
    orphans. Also returns the number of crates/*/src .rs files, the gap sites, and the set of reached .rs files."""
    txt = {f: strip(t) for f, t in files.items() if f.endswith(".rs")}
    fs = set(txt)
    tomls = {pp.dirname(f): t for f, t in files.items() if pp.basename(f) == "Cargo.toml"}
    pkgs = {d for d, t in tomls.items() if PKG_RE.search(t)}
    inline = {f: sorted(set(INLINE_RE.findall(t))) for f, t in txt.items()}
    ctx = splice(txt, inline, all_dirs(files if tree is None else tree))
    dirs = {f: expand(ctx[f], inline[f]) for f in txt}
    edges = {f: targets(t, dirs[f], fs) - {f} for f, t in txt.items()}
    reach = closure(find_roots(txt, tomls, pkgs), edges)
    declarers, by_mod, by_base = index(txt, edges, reach)
    same, flats = twin_index(files, reach)
    rows = dark_rows(files, fs - reach, tomls, pkgs)
    for row in rows:
        row["twin"] = twin_of(files[row["path"]], same, flats)
        if row["status"] == "orphan":
            f = row["path"]
            row.update(orphan_columns(f, txt[f], baseline, sorted(declarers[f])),
                       suspects=suspects_of(f, ctx, inline, by_mod, by_base))
    return rows, len([f for f in fs if UNIVERSE_RE.fullmatch(f)]), gap_sites(txt, reach), reach


def git(*args, inp=None):
    return subprocess.run(["git", *args], input=inp, capture_output=True, check=True).stdout


class Vacuous(Exception):
    """A pin that lists no crates/*/src .rs file: its census would answer 0 of 0, so it is refused (rc 3)."""


def load(pin):
    # --full-tree: without it ls-tree lists only the cwd's subtree, so a run from docs/lookahead/0.73 read no file
    # and printed "0 orphans" with rc 0.
    names = git("ls-tree", "-r", "--full-tree", "--name-only", "-z", pin).decode().split("\0")
    if not any(UNIVERSE_RE.fullmatch(f) for f in names):
        raise Vacuous(f"no crates/*/src .rs file is listed at {pin}; refusing a vacuous answer")
    want = [f for f in names if f.endswith(".rs") or pp.basename(f) == "Cargo.toml" or f == BASELINE]
    blob = git("cat-file", "--batch", inp="".join(f"{pin}:{f}\n" for f in want).encode())
    files, i = {}, 0
    for f in want:
        nl = blob.index(b"\n", i)
        size = int(blob[i:nl].split()[2])
        files[f] = blob[nl + 1:nl + 1 + size].decode("utf-8", "replace")
        i = nl + 2 + size
    base = files.pop(BASELINE, "")
    return files, {ln.split()[0] for ln in base.splitlines() if ln.strip() and not ln.startswith("#")}, names


def tally(rows, key):
    """{key(row): [rows, lines]}."""
    out = defaultdict(lambda: [0, 0])
    for r in rows:
        out[key(r)][0] += 1
        out[key(r)][1] += r["lines"]
    return out


def print_classes(orph, baseline, crates):
    g, d = Counter(r["guard"] for r in orph), Counter(r["declarer"] for r in orph)
    print("  guard: " + ", ".join(f"{k} {g[k]}" for k in ("baseline", "test-dark-parent", "test-unlisted", "no-test")))
    print("  declarer: " + ", ".join(f"{k} {d[k]}" for k in ("undeclared", "declared-by-dark")))
    bl = {b for b in baseline if not crates or any(b.startswith(c + "/") for c in crates)}
    off = sorted(bl - {r["path"] for r in orph})
    print(f"  baseline: {len(bl)} entries, {len(bl) - len(off)} of them orphans" + (f"; not: {' '.join(off)}" if off else ""))


def print_sites(orph, gaps, crates):
    sus = [r for r in orph if r["suspects"]]
    print(f"  suspects: {len(sus)} orphans have one" + "".join(f"\n    {r['path']} <- {' '.join(r['suspects'])}" for r in sus))
    gaps = [x for x in gaps if not crates or any(x.startswith(c + "/") for c in crates)]
    print(f"  gap sites in reached files: {len(gaps)}" + "".join(f"\n    {x}" for x in gaps))


def print_twins(sel):
    orph = [r for r in sel if r["status"] == "orphan" and r["twin"]]
    same = sum(r["twin"].startswith("=") for r in orph)
    print(f"  twins: {len(orph)} orphans, {sum(r['lines'] for r in orph):,} lines, have their code in a reached file: "
          f"{same} with the same text, {len(orph) - same} the same without // comments and whitespace"
          + "".join(f"\n    apart {r['path']} {r['twin']}" for r in sel if r["status"] != "orphan" and r["twin"]))


def write_tsv(path, orph):
    with open(path, "w") as fh:
        fh.write("crate\tpath\tlines\tguard\tdeclarer\tdeclarers\tsuspects\ttwin\n")
        for r in orph:
            fh.write("\t".join([r["crate"], r["path"], str(r["lines"]), r["guard"], r["declarer"],
                                ",".join(r["declarers"]) or "-", ",".join(r["suspects"]) or "-", r["twin"] or "-"]) + "\n")


def report(pin, rows, nsrc, gaps, baseline, crates, tsv):
    sel = [r for r in rows if not crates or r["crate"] in crates]
    orph = [r for r in sel if r["status"] == "orphan"]
    pk = tally(orph, lambda r: r["crate"])
    print(f"orphan_census @{pin}: {nsrc:,} crates/*/src .rs files; {len(orph):,} orphans, "
          f"{sum(r['lines'] for r in orph):,} lines, in {len(pk)} packages")
    print_classes(orph, baseline, crates)
    print_sites(orph, gaps, crates)
    print_twins(sel)
    apart = tally([r for r in sel if r["status"] != "orphan"], lambda r: (r["crate"], r["status"]))
    for (c, s), (n, ln) in sorted(apart.items()):
        print(f"  apart ({s}, no package): {c}/src {n} files, {ln:,} lines")
    for c, (n, ln) in sorted(pk.items(), key=lambda kv: (-kv[1][1], kv[0])):
        print(f"  {c}: {n} orphans, {ln:,} lines")
    if tsv:
        write_tsv(tsv, orph)


def parse_cites(stream):
    """{pin: [(loc, path)]}: each distinct cite into crates/*/src in `cite_drift.py --list` output."""
    seen = {}
    for line in stream:
        m = CITE_RE.match(line)
        if m:
            seen[m.groups()] = None
    by_pin = defaultdict(list)
    for loc, path, pin in seen:
        if UNIVERSE_RE.fullmatch(path):
            by_pin[pin].append((loc, path))
    return by_pin


def dark_at(pin):
    """{path: status} of the crates/*/src files that do not compile at pin; None when pin lists none (Vacuous)."""
    try:
        files, bl, tree = load(pin)
    except Vacuous:
        return None
    return {r["path"]: r["status"] for r in census(files, bl, tree)[0]}


def cites_line(nhits, by_pin, vacuous):
    """The --cites summary. The cites at a vacuous pin are not measured: the line counts them apart."""
    n = sum(len(v) for v in by_pin.values())
    nm = sum(len(by_pin[p]) for p in vacuous)
    tail = f"; {nm} not measured, at pins that list no crates/*/src file: {' '.join(vacuous)}" if nm else ""
    return f"orphan_census --cites: {nhits} of {n - nm} cites into crates/*/src land in a file that does not compile at its pin{tail}"


def cites(stream):
    """Print the cites into crates/*/src that land in a file that does not compile at their pin. 3 if a cited pin
    lists no crates/*/src file."""
    by_pin = parse_cites(stream)
    darks = {pin: dark_at(pin) for pin in sorted(by_pin)}
    vacuous = [pin for pin, dark in darks.items() if dark is None]
    hits = []
    for pin, dark in darks.items():
        hits += [(loc, path, pin, dark[path]) for loc, path in by_pin[pin] if path in (dark or {})]
    print(cites_line(len(hits), by_pin, vacuous))
    for loc, path, pin, s in sorted(hits):
        print(f"  {loc} {path} @{pin} ({s})")
    return 3 if vacuous else 0


def fixture():
    """The --self-test tree: {repo path: text}."""
    t = "#[test]\nfn t() {}\n"
    return {
        "Cargo.toml": '[workspace]\nmembers = ["crates/*"]\n\n[package]\nname = "root"\n',
        "src/lib.rs": '#[path = "../crates/p/src/rp.rs"] mod rp;\n',
        "crates/p/Cargo.toml": '[package]\nname = "p"\n',
        "crates/p/build.rs": 'fn main() { let _ = "src/gen_root.rs"; }\n',
        "crates/p/src/lib.rs": ("mod a;\nmod c;\ninclude!(\"e.rs\");\n#[path = \"x/f.rs\"] mod f;\n"
                                "mod outer { mod g; }\n// mod h;\nmod i;\n#[cfg(test)] mod tests;\nmod r#type;\n"
                                "/* mod j; */\nmod deep { mod mid { mod low { mod s; } } }\n"
                                'fn has(c: &str) -> bool { c.contains("#[path =") }\n'
                                '#[path = "pdir"]\nmod pin { #[path = "pt.rs"] mod pt; }\n'
                                "macro_rules! mk { ($n:ident) => { mod $n; } }\nmod zz;\n"),
        "crates/p/src/pdir/pt.rs": "",
        "crates/p/src/cyc_a.rs": 'mod ma { include!("cyc_b.rs"); }\n',
        "crates/p/src/cyc_b.rs": 'mod mb { include!("cyc_a.rs"); }\n',
        "crates/p/src/a.rs": "mod b;\n", "crates/p/src/a/b.rs": "",
        # case 16b: the chain zz/mod.rs -> aa/mid.rs -> aa/low.rs is listed, and sorts, link 2 first
        "crates/p/src/aa/mid.rs": 'include!("low.rs");\n', "crates/p/src/aa/low.rs": "mod deepy;\n",
        "crates/p/src/c/mod.rs": 'mod d;\ninclude!("../gen/part.rs");\nmod inl { include!("inl_body.rs"); }\n',
        "crates/p/src/c/d.rs": "",
        "crates/p/src/gen/part.rs": 'include!("chain.rs");\n', "crates/p/src/gen/chain.rs": "mod y;\n",
        "crates/p/src/c/y.rs": "", "crates/p/src/c/inl_body.rs": "mod v;\n", "crates/p/src/c/inl/v.rs": "",
        "crates/p/src/e.rs": "", "crates/p/src/x/f.rs": "", "crates/p/src/outer/g.rs": "",
        "crates/p/src/h.rs": t, "crates/p/src/i.rs": "", "crates/p/src/sub/i.rs": "",
        "crates/p/src/tests.rs": t, "crates/p/src/type.rs": "", "crates/p/src/j.rs": "",
        "crates/p/src/deep/mid/low/s.rs": "", "crates/p/src/gen_root.rs": "",
        "crates/p/src/k.rs": "mod l;\n", "crates/p/src/k/l.rs": t, "crates/p/src/m.rs": t,
        "crates/p/src/bin/x.rs": "", "crates/p/src/bin/y/main.rs": '#[path = "../../z.rs"] mod z;\n',
        "crates/p/src/z.rs": "", "crates/p/tests/it.rs": '#[path = "../src/w.rs"] mod w;\n', "crates/p/src/w.rs": "",
        "crates/p/src/rp.rs": "",
        "crates/p/fuzz/Cargo.toml": '[package]\nname = "p-fuzz"\n\n[[bin]]\nname = "t"\npath = "fuzz_targets/t.rs"\n',
        "crates/p/fuzz/fuzz_targets/t.rs": 'include!("../../src/n.rs");\n', "crates/p/src/n.rs": "",
        "crates/q/Cargo.toml": '[package]\nname = "q"\n\n[lib]\npath = "src/q_root.rs"\n',
        "crates/q/src/q_root.rs": 'mod qa;\n#[path = "s.rs"] mod qs;\n', "crates/q/src/qa.rs": "",
        "crates/q/src/s.rs": "",
        "crates/v/Cargo.toml": "[workspace]\nmembers = []\n", "crates/v/src/generated.rs": "",
        "crates/nm/src/x.rs": "",
        "crates/p/src/zz/mod.rs": 'include!("../aa/mid.rs");\n', "crates/p/src/zz/deepy.rs": "",
        # cases 30-31: twins of the reached tests.rs (h.rs has its text), and files with none
        "crates/p/src/tc.rs": "// a copy of tests.rs\n#[test]\nfn  t()  {}\n", "crates/p/src/note.rs": "// a comment only\n",
        "crates/p/src/dup1.rs": "fn d() {}\n", "crates/p/src/dup2.rs": "fn d() {}\n",
    }


CAN_NAME_CASES = (  # (literal, bases, target, whether the literal can name the target)
    ("s.rs", {"crates/q/src"}, "crates/p/src/deep/mid/low/s.rs", False),  # another crate
    ("pt.rs", {"crates/p/src"}, "crates/p/src/pdir/pt.rs", True),  # the anchor lies under a base
    ("x/s.rs", {"crates/p/src"}, "crates/p/src/deep/mid/low/s.rs", False),  # no suffix of the target
    ("w/s.rs", {"crates/p/src"}, "crates/p/src/neww/s.rs", False),  # a suffix only at a "/"
    ("../../pw.rs", {"crates/p/src/c/d"}, "crates/p/src/pw.rs", True),  # a base 2 below the anchor, k = 2
    ("../pw.rs", {"crates/p/src/c/d"}, "crates/p/src/pw.rs", False),  # a base 2 below, k = 1
)


def classifier(rows):
    """(rows by path, cls): cls("p/src/x.rs") is "reached", the apart status, or (status, guard, declarer)."""
    r = {x["path"]: x for x in rows}

    def cls(f):
        x = r.get("crates/" + f)
        return "reached" if x is None else x["status"] if x["status"] != "orphan" else \
            (x["status"], x["guard"], x["declarer"])
    return r, cls


def scratch_repo(root, commits):
    """Make root a git repo with one commit per {path: text, or None to delete it}; return the commits' shas. No
    GIT_* variable of the caller and no global or system config reach it, so no GIT_DIR, hook or signing applies."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")

    def run(*args):
        return subprocess.run(["git", "-C", root, *args], capture_output=True, check=True, env=env).stdout

    run("init", "-q")
    shas = []
    for i, tree in enumerate(commits):
        for path, text in tree.items():
            f = os.path.join(root, path)
            if text is None:
                os.remove(f)
                continue
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, "w") as fh:
                fh.write(text)
        run("add", "-A")
        run("commit", "-q", "-m", f"c{i}")
        shas.append(run("rev-parse", "HEAD").decode().strip())
    return shas


def at(cwd, fn, *args):
    """fn(*args) called with cwd as the working directory: its value, or the name of the exception it raised."""
    here = os.getcwd()
    os.chdir(cwd)
    try:
        return fn(*args)
    except Exception as e:  # a case reads the failure; the other cases still run
        return type(e).__name__
    finally:
        os.chdir(here)


def captured(fn, *args):
    """(fn(*args), what it printed to stdout)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        val = fn(*args)
    return val, buf.getvalue()


def subdir_cases():
    """Cases 32-34: a scratch repo read from its docs/sub, as a run from docs/lookahead/0.73 reads this one."""
    src = {"crates/p/Cargo.toml": '[package]\nname = "p"\n', "crates/p/src/lib.rs": "mod a;\n",
           "crates/p/src/a.rs": "fn a() {}\n", "crates/p/src/dark.rs": "fn d() {}\n"}
    with tempfile.TemporaryDirectory() as tmp:
        full, bare = scratch_repo(tmp, [{**src, "docs/sub/x.md": "x\n"}, dict.fromkeys(src)])
        sub = os.path.join(tmp, "docs", "sub")
        got = at(sub, lambda: census(*load(full))[:2])
        refused = at(sub, load, bare)
        stream = [f"  x.md:1 :1 = crates/p/src/dark.rs:1 @{full[:10]}\n", f"  x.md:2 :1 = crates/p/src/a.rs:1 @{bare[:10]}\n"]
        cited = at(sub, captured, cites, stream)
    want = (f"orphan_census --cites: 1 of 1 cites into crates/*/src land in a file that does not compile at its pin; "
            f"1 not measured, at pins that list no crates/*/src file: {bare[:10]}\n"
            f"  x.md:1 crates/p/src/dark.rs @{full[:10]} (orphan)\n")
    return [
        ("32 from a subdirectory the pin's full tree is read: 3 src files, crates/p/src/dark.rs the one orphan",
         isinstance(got, tuple) and got[1] == 3
         and [r["path"] for r in got[0] if r["status"] == "orphan"] == ["crates/p/src/dark.rs"]),
        ("33 a pin that lists no crates/*/src file is refused, never read as 0 orphans", refused == "Vacuous"),
        ("34 --cites counts a refused pin's cites apart, as not measured, and returns 3; the other pin still hits",
         cited == (3, want)),
    ]


def self_test():
    fx = fixture()
    rows, nsrc, gaps, _ = census(fx, {"crates/p/src/m.rs"})
    r, cls = classifier(rows)
    cases = [
        ("1 lib.rs `mod a;` reaches src/a.rs", cls("p/src/a.rs") == "reached"),
        ("2 non-mod-rs: src/a.rs `mod b;` reaches src/a/b.rs", cls("p/src/a/b.rs") == "reached"),
        ("3 mod-rs: src/c/mod.rs `mod d;` reaches src/c/d.rs", cls("p/src/c/d.rs") == "reached"),
        ("4 include!(\"e.rs\")", cls("p/src/e.rs") == "reached"),
        ("5 #[path = \"x/f.rs\"] mod f;", cls("p/src/x/f.rs") == "reached"),
        ("6 inline `mod outer { mod g; }` reaches src/outer/g.rs", cls("p/src/outer/g.rs") == "reached"),
        ("7 `// mod h;` leaves src/h.rs dark: test-unlisted, undeclared",
         cls("p/src/h.rs") == ("orphan", "test-unlisted", "undeclared")),
        ("8 `mod i;` reaches src/i.rs, not src/sub/i.rs",
         cls("p/src/i.rs") == "reached" and cls("p/src/sub/i.rs") == ("orphan", "no-test", "undeclared")),
        ("9 Cargo.toml [lib] path root", cls("q/src/q_root.rs") == "reached" and cls("q/src/qa.rs") == "reached"),
        ("10 a build.rs literal is a root", cls("p/src/gen_root.rs") == "reached"),
        ("11 dark chain: src/k.rs undeclared, src/k/l.rs test-dark-parent",
         cls("p/src/k.rs") == ("orphan", "no-test", "undeclared")
         and cls("p/src/k/l.rs") == ("orphan", "test-dark-parent", "declared-by-dark")),
        ("12 #[cfg(test)] mod tests;", cls("p/src/tests.rs") == "reached"),
        ("13 mod r#type;", cls("p/src/type.rs") == "reached"),
        ("14 /* mod j; */ counts (errs toward reached)", cls("p/src/j.rs") == "reached"),
        ("15 bin and test roots, with #[path] from them",
         all(cls(f) == "reached" for f in ("p/src/bin/x.rs", "p/src/bin/y/main.rs", "p/src/z.rs", "p/src/w.rs"))),
        ("16 include!() chain: `mod y;` in chain.rs resolves from src/c/", cls("p/src/c/y.rs") == "reached"),
        ("16b a chain met link 2 first takes a second pass: `mod deepy;` in aa/low.rs resolves from src/zz/",
         cls("p/src/zz/deepy.rs") == "reached"),
        ("17 a nested package (fuzz) include!s src/n.rs", cls("p/src/n.rs") == "reached"),
        ("17b include!() inside inline `mod inl {`: `mod v;` resolves from src/c/inl/",
         cls("p/src/c/inl/v.rs") == "reached"),
        ("18 the root package reaches crates/p/src/rp.rs by #[path]", cls("p/src/rp.rs") == "reached"),
        ("19 baseline entry", cls("p/src/m.rs") == ("orphan", "baseline", "undeclared")),
        ("20 three inline levels: dark, with lib.rs as its suspect",
         cls("p/src/deep/mid/low/s.rs")[0] == "orphan"
         and r["crates/p/src/deep/mid/low/s.rs"]["suspects"] == ["crates/p/src/lib.rs"]),
        ("21 sub/i.rs has no suspect (`sub` is no inline module of lib.rs)",
         r["crates/p/src/sub/i.rs"]["suspects"] == []),
        ("22 virtual manifest and no manifest are apart, not orphans",
         cls("v/src/generated.rs") == "virtual" and cls("nm/src/x.rs") == "no-manifest"),
        ("23 universe = crates/*/src .rs only", nsrc == sum(1 for f in fx if UNIVERSE_RE.fullmatch(f))),
        ("24 an include!() cycle among dark files ends; each is declared-by-dark",
         cls("p/src/cyc_a.rs") == cls("p/src/cyc_b.rs") == ("orphan", "no-test", "declared-by-dark")),
        ("25 a literal in another crate is no suspect (q's `#[path = \"s.rs\"]`)",
         "crates/q/src/q_root.rs" not in r["crates/p/src/deep/mid/low/s.rs"]["suspects"]),
        ("26 gap 3, `#[path = \"pdir\"] mod pin { #[path = \"pt.rs\"] mod pt; }`: dark, lib.rs its suspect, a gap site",
         cls("p/src/pdir/pt.rs") == ("orphan", "no-test", "undeclared")
         and r["crates/p/src/pdir/pt.rs"]["suspects"] == ["crates/p/src/lib.rs"]
         and 'crates/p/src/lib.rs: #[path = "pdir"]' in gaps),
        ("27 gap 2, a macro-built `mod $n;`, is a gap site", "crates/p/src/lib.rs: mod $n" in gaps),
        ("29 `\"#[path =\"` inside a string is no gap site",
         [x for x in gaps if x.startswith("crates/p/src/lib.rs")]
         == ['crates/p/src/lib.rs: #[path = "pdir"]', "crates/p/src/lib.rs: mod $n"]),
        ("30 twins: h.rs has the text of the reached tests.rs (=), tc.rs its flat text (~)",
         r["crates/p/src/h.rs"]["twin"] == "=crates/p/src/tests.rs"
         and r["crates/p/src/tc.rs"]["twin"] == "~crates/p/src/tests.rs"),
        ("31 no twin: no code (empty sub/i.rs, comment-only note.rs), or a copy of a dark file only (dup1, dup2)",
         all(r["crates/p/src/" + f]["twin"] == "" for f in ("sub/i.rs", "note.rs", "dup1.rs", "dup2.rs"))),
        ("28 can_name table", all(can_name(lit, bases, tgt) == want for lit, bases, tgt, want in CAN_NAME_CASES)),
    ] + subdir_cases()
    bad = 0
    for label, ok in cases:
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {label}")
    print(f"self-test: {len(cases) - bad}/{len(cases)} cases pass")
    return 1 if bad else 0


def census_pin(rev, crates, tsv):
    """Print rev's census, and write it to tsv when given: 0, or 3 when rev lists no crates/*/src file."""
    pin = git("rev-parse", "--short=10", rev).decode().strip()
    try:
        files, bl, tree = load(pin)
    except Vacuous as e:
        print(f"orphan_census: {e}", file=sys.stderr)
        return 3
    rows, nsrc, gaps, _ = census(files, bl, tree)
    report(pin, rows, nsrc, gaps, bl, crates, tsv)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pin")
    ap.add_argument("--crate", action="append", default=[])
    ap.add_argument("--tsv")
    ap.add_argument("--cites", metavar="LIST", help="saved `cite_drift.py --list` output, or - for stdin")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.cites:
        with (sys.stdin if a.cites == "-" else open(a.cites)) as fh:
            return cites(fh)
    if not a.pin:
        ap.error("--pin, --cites or --self-test")
    return census_pin(a.pin, set(c.rstrip("/") for c in a.crate), a.tsv)


if __name__ == "__main__":
    sys.exit(main())
