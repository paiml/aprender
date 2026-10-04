#!/usr/bin/env python3
"""Port or drop: whether the absent tests of the tests-only dark files still name live code (#4700).

A tests-only file is an orphan that fn_census.py calls unique and whose absent units are all tests. Porting a test
needs cargo; dropping one needs a reason. For each absent test this reads its text past its name, and that of the
file's own fns it calls (through chains), and places each reference in it:

  call    `f(args)`, `a::f(args)` or `T::f(args)`, where f is not a macro, not CamelCase (a tuple struct or a
          variant), and, bare, not a name the fn binds (a param, a `let` or `for` pattern, a closure param, a
          nested fn)
  method  `.f(args)`, by name only: the receiver's type is unknown, so a method is gone or left out, never live
  type    a CamelCase word that is not a std or prelude name (Result, Error, Vec, ...)

A call is `live` when a reached fn of its name (in an impl or trait of T for `T::f`; else a free fn in the crate that
its path starts with, or in the calling file's crate) takes as many params as the call passes args (self counts, for
`T::f`), and `arity` when such reached fns exist but none does. A type is `live` when a reached file (of any crate)
defines it. With no reached def, a reference is `gone` when an orphan defines
its name (or T) and no reached code holds the word: a word that reached code holds may name a derive, a trait's
provided method, a macro's output or a dependency. It is a `fixture` when only orphans with a test in them define
it: it would port with the file. Every other reference is left out: a path whose first segment is a std module or
no repo dir, file or crate; a name no file defines; and the arity of a call with `|` or `<` in an arg.

A test is `copy` when its text past its name is a reached fn's; else `stale` when it holds an arity reference, or a
gone one beside a live one (its subject lives on, in another form); else `gone` when it holds a gone one (all it
names of the repo is gone); else `live` when it holds a live one; else `inert`. A file is `drop` when every absent
test is copy or gone, `port` when one is live, else `read`. `changed` counts the file's units that a reached unit
of the same name holds in another form: read those with unit_history.py before a drop.

Errs toward live: a reference the model cannot place is left out, never called gone. Errors the other way: a bare
call to a fn that a `use` brings in from another crate reads as arity when the calling crate has a reached free fn
of that name and another arity, and a fn that a macro builds from pasted words, which no reached code spells, reads
as gone. Read the reasons of a drop or stale row.

Usage: test_triage.py --pin REV [--tsv PATH] | --self-test
rc 3 when the pin lists no crates/*/src .rs file: orphan_census.load() refuses a vacuous answer.
"""
import argparse
import os.path as pp
import re
import sys
from collections import Counter

sys.path.insert(0, pp.dirname(pp.abspath(__file__)))
import fn_census as fc  # noqa: E402
import orphan_census as oc  # noqa: E402

TYPE_KINDS = ("struct", "enum", "union", "type", "trait")
SP_RE = re.compile(r"\s*")
OPEN_RE = re.compile(r"\s*\(")
WHERE_RE = re.compile(r"\bwhere\b")
FOR_RE = re.compile(r"\bfor\b")
SELF_TY_RE = re.compile(r"\s*(?:&\s*(?:'\w+\s*)?(?:mut\s+)?)?(?:dyn\s+)?(?:\w+\s*::\s*)*(\w+)")
TURBO = r"(?:::\s*<[^()]*?>\s*)?"
CALL_RE = re.compile(r"(?P<dot>\.\s*)?(?P<path>(?:\b\w+\s*" + TURBO + r"::\s*)*)\b(?P<name>[A-Za-z_]\w*)\s*"
                     + TURBO + r"\(")
SEG_RE = re.compile(r"(\w+)\s*" + TURBO + "::")
PRE_RE = re.compile(r"::\s*\Z")
BIND_RE = re.compile(r"\b(?:let|for)\s+([^=;{}]*?)(?:=|\bin\b|;)|\bfn\s+(\w+)|\|([^|]*)\|")
WORD_RE = re.compile(r"[A-Za-z_]\w*")
TYPE_RE = re.compile(r"\b[A-Z]\w*")
TESTISH_RE = re.compile(r"(?:^|/|_)tests?(?:/|_|\.rs$)")
NAME_RE = re.compile(r'^\s*name\s*=\s*"([\w-]+)"', re.M)
STD_MODS = frozenset("""std core alloc fs io mem env ptr thread time path process fmt iter cmp ops collections sync
    hash str string vec slice char num f32 f64 i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 usize isize convert any cell
    rc borrow boxed array ffi net os hint panic pin future task marker default error mpsc atomic""".split())
STD_TYPES = frozenset("""Self Some None Ok Err Option Result Error Vec String Box Rc Arc Weak Cell RefCell Mutex
    RwLock HashMap HashSet BTreeMap BTreeSet VecDeque BinaryHeap Cow Path PathBuf OsStr OsString File Duration
    Instant SystemTime Ordering PhantomData NonNull Default Clone Copy Debug Display Iterator IntoIterator From Into
    TryFrom TryInto AsRef AsMut Send Sync Sized PartialEq Eq PartialOrd Ord Hash Fn FnMut FnOnce ToString Read
    Write""".split())
STATES = ("copy", "gone", "stale", "live", "inert")
VERDICTS = ("drop", "port", "read")
COLUMNS = ("path", "lines", "absent", *STATES, "changed", "verdict", "dead_refs", "fixtures", "live_refs")


def split_top(s, angles):
    """The parts of s between its commas outside brackets (and outside <> when angles is set; `->` is no bracket)."""
    out, depth, p = [], 0, 0
    for i, c in enumerate(s):
        if c in "([{" or (angles and c == "<"):
            depth += 1
        elif c in ")]}" or (angles and c == ">" and s[i - 1:i] != "-"):
            depth -= 1
        elif c == "," and depth == 0:
            out.append(s[p:i])
            p = i + 1
    out.append(s[p:])
    return [x.strip() for x in out if x.strip()]


def skip_generics(mask, p):
    """The index past the <...> that starts at p, after spaces; p when none does."""
    q = SP_RE.match(mask, p).end()
    if q >= len(mask) or mask[q] != "<":
        return p
    depth = 0
    for i in range(q, len(mask)):
        if mask[i] == "<":
            depth += 1
        elif mask[i] == ">" and mask[i - 1] != "-":
            depth -= 1
            if not depth:
                return i + 1
    return len(mask)


def params(mask, pr, p):
    """The params of the fn whose name ends at p, or None when no ( follows its generics."""
    m = OPEN_RE.match(mask, skip_generics(mask, p))
    if not m:
        return None
    i = m.end() - 1
    return split_top(mask[i + 1:pr.get(i, len(mask))], True)


def self_type(hdr):
    """The type that an impl header (from `impl` on) gives its fns, by its last path segment; '?' when not a path."""
    t = FOR_RE.split(WHERE_RE.split(hdr[skip_generics(hdr, 4):])[0])[-1]
    m = SELF_TY_RE.match(t)
    return m.group(1) if m else "?"


def fn_row(code, mask, pr, m, start, end, box):
    ps = params(mask, pr, m.end())
    return {"name": m.group("n1"), "box": box, "n": None if ps is None else len(ps),
            "pnames": {w for p in ps or () for w in WORD_RE.findall(p.split(":")[0])},
            "test": fc.is_test(mask, start), "span": (m.end(), end), "key": hash(fc.flat(code, mask, m.end(), end))}


def item_span(mask, pr, m, skip):
    """(kind, start, (i, end)) for the item that the ITEM_RE match m opens, or None when fn_census.scan() skips
    it (inside the span of the item before, or not at an item start)."""
    kind = m.group("k1") or m.group("k2") or m.group("k3") or m.group("k4")
    _, start, ok = fc.head(mask, m.start())
    if not ok or m.start() < skip:
        return None
    span = fc.item_end(mask, pr, m.end(), kind not in fc.SEMI_ONLY)
    return None if span is None else (kind, start, span)


def defs(text):
    """{mask, pr, fns, types} for one file. A fn row per fn that fn_census.scan() takes as a unit: its name, its box
    (its impl's self type, or its trait's name; '' outside both), its param count (None when unread) and param
    names, whether it is a test, its span past its name and the hash of its text there. types: the names of its
    struct, enum, union, type and trait items."""
    code, mask = fc.lex(text)
    pr, boxes, fns, types, skip = fc.pairs(mask), [], [], set(), 0
    for m in fc.ITEM_RE.finditer(mask):
        item = item_span(mask, pr, m, skip)
        if item is None:
            continue
        kind, start, (i, end) = item
        if kind in TYPE_KINDS:
            types.add(m.group("n1"))
        if kind in ("impl", "trait") and mask[i] == "{":
            boxes.append((i, end, m.group("n1") or self_type(mask[m.start():i])))
            continue
        if kind == "fn":
            fns.append(fn_row(code, mask, pr, m, start, end, fc.box_of(boxes, m.start())))
        skip = end
    return {"mask": mask, "pr": pr, "fns": fns, "types": types}


def crate_of(f):
    """X for a path under crates/X/, else '' (the root package)."""
    parts = f.split("/")
    return parts[1] if len(parts) > 2 and parts[0] == "crates" else ""


def add_fn(ix, f, r):
    if r["box"]:
        ix["meth"].setdefault((r["box"], r["name"]), []).append(r["n"])
    else:
        ix["free"].setdefault(r["name"], []).append(r["n"])
        ix["cfree"].setdefault((crate_of(f), r["name"]), []).append(r["n"])
    ix["fn"].add(r["name"])
    ix["by"].setdefault(r["name"], set()).add(f)
    ix["body"].add(r["key"])
    if r["test"]:
        ix["tested"].add(f)


def build(files, said=False):
    """The defs in {path: text}: free fn param counts by name and by (crate, name), method param counts by
    (box, name), fn and type names, the files that define each name, the files with a test, the hashes of the fn texts past their names, and
    (when said is set) every word the code holds."""
    ix = {"free": {}, "cfree": {}, "meth": {}, "fn": set(), "type": set(), "by": {}, "tested": set(), "body": set(),
          "said": set()}
    for f in sorted(files):
        d = defs(files[f])
        for r in d["fns"]:
            add_fn(ix, f, r)
        ix["type"] |= d["types"]
        for t in d["types"]:
            ix["by"].setdefault(t, set()).add(f)
        if said:
            ix["said"].update(WORD_RE.findall(d["mask"]))
    return ix


def repo_mods(files):
    """(mods, crates). mods: the words a repo path can start with, being each dir and file stem of a listed .rs
    path and each name in a Cargo.toml (with - as _), less the std modules. crates: {such a name: crate_of() its
    Cargo.toml}, a crates/ one over the root's."""
    out, crates = set(), {}
    for f in sorted(files):
        if f.endswith(".rs"):
            out.update(pp.splitext(x)[0].replace("-", "_") for x in f.split("/"))
        elif pp.basename(f) == "Cargo.toml":
            names = [x.replace("-", "_") for x in NAME_RE.findall(files[f])]
            out.update(names)
            crates.update((x, crate_of(f)) for x in names)
    return out - STD_MODS, crates


def calls(mask, pr, b0, b1):
    """(dot, path, name, args) per call in mask[b0:b1]: args is None when an arg holds | or <."""
    out = []
    for m in CALL_RE.finditer(mask, b0, b1):
        path = tuple(SEG_RE.findall(m.group("path")))
        if not m.group("dot") and not path and PRE_RE.search(mask, max(b0, m.start() - 64), m.start()):
            continue  # the tail of a path CALL_RE could not read, such as `Vec::<(u8, u8)>::new`
        i = m.end() - 1
        args = split_top(mask[i + 1:pr.get(i, b1)], False)
        out.append((bool(m.group("dot")), path, m.group("name"),
                    None if any("|" in a or "<" in a for a in args) else len(args)))
    return out


def testish(f, dix):
    return f in dix["tested"] or TESTISH_RE.search(f) is not None


def fits(n, ns):
    return n is None or any(k is None or k == n for k in ns)


def place_name(label, name, n, reached, in_dark, cix, dix):
    """(label, state): live or arity against the reached param counts; with none, gone or fixture when orphans
    define name and no reached code holds it; else None."""
    if reached:
        return label, "live" if fits(n, reached) else "arity"
    if name in cix["said"] or not in_dark:
        return None
    return label, "fixture" if all(testish(f, dix) for f in dix["by"].get(name, ())) else "gone"


def place_assoc(t, name, n, cix, dix):
    """(label, state) or None for a call `T::name(args)`: by T's methods when a reached file defines T, else by T."""
    label = f"{t}::{name}()"
    if t in STD_TYPES:
        return None
    if t in cix["type"]:
        return place_name(label, name, n, cix["meth"].get((t, name)), (t, name) in dix["meth"], cix, dix)
    return place_name(label, t, None, None, t in dix["type"], cix, dix)


def place_free(path, name, n, cix, dix, ctx):
    """(label, state) or None for a call `a::name(args)` or `name(args)`, by the free fns of the crate that path
    names, else of the calling file's crate; None when path starts with no crate, repo dir or file, crate, super or
    self."""
    own = ctx["crates"].get(path[0], ctx["own"]) if path else ctx["own"]
    if path and path[0] not in ("crate", "super", "self") and path[0] not in ctx["mods"]:
        return None
    return place_name(name + "()", name, n, cix["cfree"].get((own, name)), name in dix["free"], cix, dix)


def place_call(c, cix, dix, ctx):
    """(label, state) or None for one call; ctx: repo_mods() as mods and crates, and the calling file's crate."""
    dot, path, name, n = c
    if dot:
        return place_name("." + name, name, None, None, name in dix["fn"], cix, dix)
    if path and path[-1][:1].isupper():
        return place_assoc(path[-1], name, n, cix, dix)
    return place_free(path, name, n, cix, dix, ctx)


def place_type(t, cix, dix):
    return (t, "live") if t in cix["type"] else place_name(t, t, None, None, t in dix["type"], cix, dix)


def body_refs(d, r, local, cix, dix, ctx):
    """(refs, local fns called) for the fn row r of defs() d: its placed calls, methods and types."""
    mask, (b0, b1) = d["mask"], r["span"]
    body = mask[b0:b1]
    bound = set(r["pnames"])
    for m in BIND_RE.finditer(body):
        bound.update(WORD_RE.findall(m.group(m.lastindex)))
    refs, sub = set(), set()
    for c in calls(mask, d["pr"], b0, b1):
        dot, path, name, _ = c
        if name[:1].isupper() or (not dot and not path and name in bound) or (path and path[-1] in local["types"]):
            continue
        if name in local["fns"] and (dot or not path):
            sub.add(name)
            continue
        refs.add(place_call(c, cix, dix, ctx))
    refs |= {place_type(t, cix, dix) for t in set(TYPE_RE.findall(body)) - local["types"] - STD_TYPES}
    return refs - {None}, sub


def expand(name, own, sub):
    """The refs of name and of the local fns it calls, through chains."""
    seen, todo, out = {name}, [name], set()
    while todo:
        f = todo.pop()
        out |= own.get(f, set())
        for g in sub.get(f, set()) - seen:
            seen.add(g)
            todo.append(g)
    return out


def test_state(refs, copy):
    states = {s for _, s in refs}
    if copy:
        return "copy"
    if "arity" in states or {"gone", "live"} <= states:
        return "stale"
    return "gone" if "gone" in states else "live" if "live" in states else "inert"


def triage_file(text, absent, cix, dix, ctx):
    """[(test, state, refs)] for the tests of one file whose names are in absent."""
    d = defs(text)
    local = {"fns": {r["name"] for r in d["fns"]}, "types": d["types"]}
    own, sub = {}, {}
    for r in d["fns"]:
        refs, called = body_refs(d, r, local, cix, dix, ctx)
        own.setdefault(r["name"], set()).update(refs)
        sub.setdefault(r["name"], set()).update(called)
    out = []
    for r in d["fns"]:
        if r["test"] and r["name"] in absent:
            refs = expand(r["name"], own, sub)
            out.append((r["name"], test_state(refs, r["key"] in cix["body"]), refs))
    return out


def verdict(rows):
    states = {s for _, s, _ in rows}
    if states and states <= {"copy", "gone"}:
        return "drop"
    return "port" if "live" in states else "read"


def tests_only(res):
    """The orphans that fn_census() calls unique and whose absent units are all tests."""
    return sorted(f for f, (v, rows, _, _) in res.items()
                  if v == "unique" and all(fc.role(r) == "tests" for r in rows if r[0] == "absent"))


def triage(files, reach, orph, res):
    """{tests-only orphan: (rows, changed)}, res being fn_census() over the orphans."""
    cix = build({f: files[f] for f in reach if f.endswith(".rs")}, said=True)
    dix = build({f: files[f] for f in orph})
    mods, crates = repo_mods(files)
    out = {}
    for f in tests_only(res):
        rows = res[f][1]
        absent = {r[2] for r in rows if r[0] == "absent"}
        ctx = {"mods": mods, "crates": crates, "own": crate_of(f)}
        out[f] = (triage_file(files[f], absent, cix, dix, ctx), sum(r[0] == "changed" for r in rows))
    return out


def reasons(rows, states, top=None):
    """label:state×n per reference in states (label×n for one state), n the absent tests that hold it, most first."""
    c = Counter(f"{label}:{s}" if len(states) > 1 else label for _, _, refs in rows for label, s in refs if s in states)
    return ",".join(f"{k}×{n}" for k, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:top]) or "-"


def cells(f, lines, rows, changed):
    st = Counter(s for _, s, _ in rows)
    return [f, lines, len(rows), *(st[s] for s in STATES), changed, verdict(rows), reasons(rows, ("gone", "arity")),
            reasons(rows, ("fixture",)), reasons(rows, ("live",), 8)]


def tally(rows):
    """{verdict: [files, lines]} over the sheet rows."""
    v, out = COLUMNS.index("verdict"), {k: [0, 0] for k in VERDICTS}
    for r in rows:
        out[r[v]][0] += 1
        out[r[v]][1] += r[1]
    return out


def summary(pin, out, rows):
    """The report lines: the test states, the files and lines per verdict, and the reasons of each drop row."""
    st = Counter(s for f in out for _, s, _ in out[f][0])
    v, d, t = COLUMNS.index("verdict"), COLUMNS.index("dead_refs"), tally(rows)
    lines = [f"test triage at {pin}: {len(rows)} tests-only orphans, {sum(st.values())} absent tests: "
             + ", ".join(f"{s} {st[s]}" for s in STATES),
             "  files: " + ", ".join(f"{k} {t[k][0]} ({t[k][1]:,} lines)" for k in VERDICTS)]
    return lines + [f"  drop {r[0]} ({r[1]}): {r[d][:100]}" for r in rows if r[v] == "drop"]


def report(pin, out, orph, tsv):
    rows = [cells(f, orph[f]["lines"], *out[f]) for f in sorted(out)]
    print("\n".join(summary(pin, out, rows)))
    if tsv:
        with open(tsv, "w") as fh:
            fh.write("\t".join(COLUMNS) + "\n")
            fh.writelines("\t".join(map(str, r)) + "\n" for r in rows)


FIX_C = {"c/lib.rs": "pub fn live2(a: u8, b: u8) -> u8 { a + b }\npub struct Ty;\n"
                     "impl Ty { pub fn make() -> Self { Ty } pub fn m(&self) {} }\npub struct Loc;\n"
                     "impl Loc { pub fn mk(a: u8) -> Self { Loc } }\nfn twin_body() { assert_eq!(1, 1); }\n"
                     "fn uses() { said_only(); Ty::default(); }\npub type Result<T> = std::result::Result<T, ()>;\n"
                     "pub trait Tr { fn tm(&self); }\n",
         "c/fs.rs": "pub fn read_all() {}\n", "crates/other/src/lib.rs": "pub fn helper2(a: u8) {}\n"}
FIX_TOML = {"Cargo.toml": '[package]\nname = "oth"\n',
            "crates/other/Cargo.toml": '[package]\nname = "other-crate"\n[lib]\nname = "oth"\n'}
FIX_D = {"d/old.rs": "pub fn gone1() {}\npub fn said_only() {}\npub struct Old;\n"
                     "impl Old { pub fn new() -> Self { Old } }\n"
                     "impl Ty { pub fn removed(&self) {} pub fn default() -> Self { Ty } }\npub enum Error { A }\n",
         "d/tests/fixtures.rs": "pub fn fixture_make() -> u8 { 1 }\n"}
FIX_T = {"d/foo_tests.rs": """use super::*;
struct Loc;
impl Loc { fn mk() -> Self { Loc } }
fn helper() { gone1(); }
fn apply(live2: fn(u8) -> u8) -> u8 { live2(1) }
#[test] fn t_live() { live2(1, 2); }
#[test] fn t_arity() { live2(1); }
#[test] fn t_gone() { gone1(); }
#[test] fn t_type_gone() { let _o = Old::new(); }
#[test] fn t_copy() { assert_eq!(1, 1); }
#[test] fn t_inert() { let v = vec![1]; assert_eq!(v.len(), 1); }
#[test] fn t_method_gone() { let t = Ty; t.removed(); }
#[test] fn t_ufcs() { Ty::m(&Ty); Ty::make(); }
#[test] fn t_ufcs_bad() { Ty::make(1); }
#[test] fn t_uncertain() { live2(|a, b| a, 1); }
#[test] fn t_local() { helper(); }
#[test] fn t_ext() { std::fs::read("x").ok(); let _ = Vec::<u8>::new(); }
#[test] fn t_macro() { gone1!(); }
#[test] fn t_said() { said_only(); }
#[test] fn t_fixture() { let _ = fixture_make(); }
#[test] fn t_derive() { let _ = Ty::default(); }
#[test] fn t_shadow() { let live2 = |a: u8| a; live2(1); }
#[test] fn t_type_live() { let _t: Ty = Ty; }
#[test] fn t_modpath() { old::gone1(); }
#[test] fn t_stdpath() { fs::gone1(); }
#[test] fn t_loc() { Loc::mk(); }
#[test] fn t_dotbound() { let removed = 0; let t = Ty; t.removed(); }
#[test] fn t_param() { apply(|x| x); }
#[test] fn t_for() { for (i, gone1) in [(1, 2)].iter().enumerate() { gone1(i); } }
#[test] fn t_std_type() { let r: Result<u8> = Ok(1); gone1(); }
#[test] fn t_std_call() { let _ = Error::new(1); }
#[test] fn t_trait() { let _f: Option<&dyn Tr> = None; Tr::tm(&0); }
#[test] fn t_xcrate() { helper2(1, 2); }
#[test] fn t_xcrate_path() { oth::helper2(1); }
""", "d/bar_tests.rs": "#[test] fn b1() { gone1(); }\n#[test] fn b2() { assert_eq!(1, 1); }\n",
         "d/baz_tests.rs": "#[test] fn z1() { let v = 1; assert_eq!(v, 1); }\n#[test] fn z2() { gone1(); }\n"}
FIX_WANT = {"d/foo_tests.rs": {"t_live": "live", "t_arity": "stale", "t_gone": "gone", "t_type_gone": "gone",
                               "t_copy": "copy", "t_inert": "inert", "t_method_gone": "stale", "t_ufcs": "live",
                               "t_ufcs_bad": "stale", "t_uncertain": "live", "t_local": "gone", "t_ext": "inert",
                               "t_macro": "inert", "t_said": "inert", "t_fixture": "inert", "t_derive": "live",
                               "t_shadow": "inert", "t_type_live": "live", "t_modpath": "gone",
                               "t_stdpath": "inert", "t_loc": "inert", "t_dotbound": "stale", "t_param": "inert",
                               "t_for": "inert", "t_std_type": "gone", "t_std_call": "inert",
                               "t_trait": "live", "t_xcrate": "inert", "t_xcrate_path": "live",
                               "verdict": "port"},
            "d/bar_tests.rs": {"b1": "gone", "b2": "copy", "verdict": "drop"},
            "d/baz_tests.rs": {"z1": "inert", "z2": "gone", "verdict": "read"}}


def param_cases():
    cases = [("fn f(a: u8, b: HashMap<K, V>) {}", 2), ("fn f<T: Fn(u8) -> u8>(&self, t: T) {}", 2),
             ("fn f(&'a mut self) {}", 1), ("fn f() {}", 0), ("fn f(self: Box<Self>, g: fn(u8) -> u8,) {}", 2),
             ("fn f(mut self, (a, b): (u8, u8)) {}", 2), ("fn f<T>(x: Vec<Vec<T>>, y: impl Fn(u8, u8) -> u8) {}", 2),
             ("fn f(&self, cb: impl FnMut(&mut Vec<u8>) -> Result<(), E>) {}", 2)]
    return [(f"params {t!r}", defs(t)["fns"][0]["n"], n) for t, n in cases]


def box_cases():
    cases = [("impl Foo { fn m() {} }", "Foo"), ("impl<T> Bar<T> { fn m() {} }", "Bar"),
             ("impl Foo for Bar { fn m() {} }", "Bar"),
             ("impl<T: Clone> fmt::Display for crate::x::Baz<T> where T: X { fn m() {} }", "Baz"),
             ("unsafe impl<'a> Tr for &'a W { fn m() {} }", "W"), ("impl<T> Tr for [T] { fn m() {} }", "?"),
             ("trait Tr { fn m(&self); }", "Tr"), ("fn m() {}", "")]
    return [(f"box {t!r}", defs(t)["fns"][0]["box"], b) for t, b in cases]


def call_cases():
    cases = [("foo(1, 2)", [(False, (), "foo", 2)]), ("x.bar(3)", [(True, (), "bar", 1)]),
             ("Ty::make()", [(False, ("Ty",), "make", 0)]), ("Vec::<u8>::new()", [(False, ("Vec",), "new", 0)]),
             ("a::b::baz(vec![1, 2], (3, 4))", [(False, ("a", "b"), "baz", 2)]),
             ("crate::a::B::c(1)", [(False, ("crate", "a", "B"), "c", 1)]),
             ("f(|x, y| x, 2)", [(False, (), "f", None)]), ("g(x < y)", [(False, (), "g", None)]),
             ("h::<u8>(1,)", [(False, (), "h", 1)]), ("x.a::<u8>(1)", [(True, (), "a", 1)]),
             ("Vec::<(u8, u8)>::\n            new()", []), ("m!(1)", [])]
    out = []
    for t, want in cases:
        d = defs("fn t() { " + t + "; }")
        out.append((f"calls {t!r}", calls(d["mask"], d["pr"], *d["fns"][0]["span"]), want))
    return out


def triage_cases():
    cix, dix = build(FIX_C, said=True), build({**FIX_D, **FIX_T})
    mods, crates = repo_mods({**FIX_C, **FIX_D, **FIX_T, **FIX_TOML})
    out = [("crates", crates, {"oth": "other", "other_crate": "other"})]
    for f, want in FIX_WANT.items():
        ctx = {"mods": mods, "crates": crates, "own": crate_of(f)}
        rows = triage_file(FIX_T[f], set(want) - {"verdict"}, cix, dix, ctx)
        out += [(f"{f} {t}", s, want[t]) for t, s, _ in rows]
        out += [(f"{f} tests", len(rows), len(want) - 1), (f"{f} verdict", verdict(rows), want["verdict"])]
    return out


def self_test():
    cases = param_cases() + box_cases() + call_cases() + triage_cases()
    bad = [c for c in cases if c[1] != c[2]]
    for name, got, want in bad:
        print(f"FAIL {name}: got {got!r}, want {want!r}")
    print(f"test_triage self-test: {len(cases) - len(bad)}/{len(cases)} cases pass")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pin")
    ap.add_argument("--tsv")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.pin:
        ap.error("--pin or --self-test")
    pin = oc.git("rev-parse", "--short=10", a.pin).decode().strip()
    try:
        files, bl, tree = oc.load(pin)
    except oc.Vacuous as e:
        print(f"test_triage: {e}", file=sys.stderr)
        return 3
    rows, _, _, reach = oc.census(files, bl, tree)
    orph = {r["path"]: r for r in rows if r["status"] == "orphan"}
    report(pin, triage(files, reach, orph, fc.fn_census(files, reach, orph)), orph, a.tsv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
