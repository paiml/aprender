#!/usr/bin/env python3
"""Which code in the dark files has a compiled copy, unit by unit.

A unit is a fn (the outermost one, through its body or its `;`), a struct, enum, union, type, const, static or
macro_rules item, or an impl or trait header. Its key is its text from the item keyword on, with its modifiers but
not `pub`, without comments and without whitespace outside literals, plus the header of the impl or trait around
it. The code outside every unit is one more unit, `rest`, with no name, once attributes, `use`, `extern crate`,
`mod` and `include!` lines, braces, `;` and whitespace outside literals are dropped; it holds item-level macro calls
other than macro_rules, and the statements of a file that is included into a fn body. A dark unit is `present` when
a reached file has a unit of the same kind and key, `changed` when a reached unit has the same header, kind and name
but other text, and `absent` otherwise; a rest unit is never `changed`. A file is covered, stale or unique by its
worst unit. Not compared: attributes and the dropped lines (an included file is a row of its own).

The compiled side is every .rs file that orphan_census.py reaches from a package root. A unit counts as `named` when
a compiled unit of its kind and name exists under any header. The TSV gives each orphan's verdict, its home (the
reached file that holds most of its present units), the absent and changed units, and the code left outside every
unit (`rest`, with its state).

Usage: fn_census.py --pin REV [--tsv PATH] | --self-test
rc 3 when the pin lists no crates/*/src .rs file: orphan_census.load() refuses a vacuous answer.
"""
import argparse
import os.path as pp
import re
import sys
from collections import Counter

sys.path.insert(0, pp.dirname(pp.abspath(__file__)))
import orphan_census as oc  # noqa: E402

LIT = "\x01"
TOK_RE = re.compile(r"""//[^\n]*|/\*
    |(?<!\w)(?:br|cr|r)(\#*)"[\s\S]*?"\1
    |(?<!\w)[bc]"(?:[^"\\]|\\[\s\S])*"|"(?:[^"\\]|\\[\s\S])*"
    |(?<!\w)b?'(?:\\(?:u\{[0-9a-fA-F_]*\}|x[0-9a-fA-F]{2}|[^\n])|[^'\\\n])'""", re.X)
BLOCK_RE = re.compile(r"/\*|\*/")
BRK_RE = re.compile(r"[()\[\]{}]")
STOP_RE = re.compile(r"[()\[\]{};]")
LITRUN_RE = re.compile(LIT + "+")
WS_RE = re.compile(r"\s+")
MODS_RE = re.compile(r"(?:\b(?:pub(?:\s*\([^()]*\))?|const|async|unsafe|default|auto|extern(?:\s*\x01+)?)\s*)*\Z")
PUB_RE = re.compile(r"pub(?:\s*\([^()]*\))?\s*")
ITEM_RE = re.compile(r"""\b(?:(?P<k1>fn|struct|enum|union|type|trait)\s+(?:r\#)?(?P<n1>\w+)
    |(?P<k2>const|static)\s+(?:mut\s+)?(?P<n2>\w+)\s*:
    |(?P<k3>macro_rules)!\s*(?P<n3>\w+)
    |(?P<k4>impl)\b)""", re.X)
TEST_RE = re.compile(r"#\s*\[\s*(?:\w+::)*(?:test|rstest|test_case|wasm_bindgen_test)\b")
SEMI_ONLY = {"const", "static", "type"}
VERDICTS, STATES = ("covered", "stale", "unique"), ("present", "changed", "absent")
ROLES = ("tests", "fns", "items", "rest")
LABELS = {"unnamed": "absent and not named"}
COLUMNS = ("path", "lines", "twin", "verdict", "home", "units", "present", "changed", "absent", "absent_fns",
           "absent_tests", "absent_items", "changed_units", "rest")
ATTR_RE = re.compile(r"#!?\[[^\[\]]*\]")
DECL_RE = re.compile(r"\b(?:pub(?:\s*\([^()]*\))?\s*)?(?:use\s[^;]*;|extern\s+crate\s[^;]*;|mod\s+\w+\s*[;{])")
INCL_RE = re.compile(r"\binclude!\s*\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)")
NOISE_RE = re.compile(r"[\s{};]+")


def block_end(text, j):
    """End of the nested block comment whose `/*` ends at j."""
    d = 1
    while d:
        m = BLOCK_RE.search(text, j)
        if not m:
            return len(text)
        d, j = d + (1 if m.group() == "/*" else -1), m.end()
    return j


def lex(text):
    """(code, mask): text with each comment replaced by a space, and code with each literal's chars set to LIT."""
    code, mask, pos = [], [], 0
    m = TOK_RE.search(text)
    while m:
        s, e = m.span()
        code.append(text[pos:s])
        mask.append(text[pos:s])
        if m.group().startswith("/"):
            e = block_end(text, e) if m.group() == "/*" else e
            code.append(" ")
            mask.append(" ")
        else:
            code.append(m.group())
            mask.append(LIT * (e - s))
        pos = e
        m = TOK_RE.search(text, pos)
    code.append(text[pos:])
    mask.append(text[pos:])
    return "".join(code), "".join(mask)


def pairs(mask):
    """{index of an open bracket: index of its close}; an unclosed one maps to len(mask)."""
    out, stack = {}, []
    for m in BRK_RE.finditer(mask):
        if m.group() in "([{":
            stack.append(m.start())
        elif stack:
            out[stack.pop()] = m.start()
    out.update((i, len(mask)) for i in stack)
    return out


def item_end(mask, pr, p, brace):
    """(i, end) for the item whose lead ends at p: i is its `;`, or its `{` when brace is set, and end is just past
    the item. None when the enclosing block closes first."""
    m = STOP_RE.search(mask, p)
    while m:
        c, i = m.group(), m.start()
        if c == ";":
            return i, i + 1
        if c in ")]}":
            return None
        if c == "{" and brace:
            return i, pr[i] + 1
        m = STOP_RE.search(mask, pr[i] + 1)
    return None


def head(mask, p):
    """(s, start, ok) for the item whose keyword is at p: s is where its modifiers begin, start the same past any
    `pub`, and ok whether it stands where an item can: at the start of the file or after `{`, `}`, `;` or an
    attribute's `]`."""
    s = MODS_RE.search(mask, max(0, p - 200), p).start()
    pub = PUB_RE.match(mask, s, p)
    j = s - 1
    while j >= 0 and mask[j] in " \t\r\n":
        j -= 1
    return s, (pub.end() if pub else s), j < 0 or mask[j] in "{};]"


def flat(code, mask, s, e):
    """code[s:e] without whitespace outside literals."""
    out, p = [], s
    for m in LITRUN_RE.finditer(mask, s, e):
        out += [WS_RE.sub("", code[p:m.start()]), code[m.start():m.end()]]
        p = m.end()
    out.append(WS_RE.sub("", code[p:e]))
    return "".join(out)


def is_test(mask, start):
    """Whether a test attribute stands between the previous `{`, `}` or `;` and start."""
    j = max(mask.rfind(c, 0, start) for c in "{};")
    return bool(TEST_RE.search(mask, j + 1, start))


def box_of(boxes, p):
    """The header of the innermost impl or trait block around p, else ''."""
    return next((h for i, end, h in reversed(boxes) if i < p < end), "")


def blank(m):
    return " " * (m.end() - m.start())


def rest(code, mask, spans):
    """(key, shown) for the code outside spans, without attributes, `use`, `extern crate`, `mod` and `include!`
    lines, braces, `;` and whitespace outside literals. key keeps each literal's text; shown writes it as "…"."""
    out, p = [], 0
    for s, e in sorted(spans):
        s = max(s, p)
        if e > s:
            out += [mask[p:s], " " * (e - s)]
            p = e
    out.append(mask[p:])
    b, prev = "".join(out), None
    while b != prev:
        b, prev = ATTR_RE.sub(blank, b), b
    b = INCL_RE.sub(blank, DECL_RE.sub(blank, b))
    key, p = [], 0
    for m in LITRUN_RE.finditer(b):
        key += [NOISE_RE.sub("", b[p:m.start()]), code[m.start():m.end()]]
        p = m.end()
    key.append(NOISE_RE.sub("", b[p:]))
    return "".join(key), LITRUN_RE.sub('"…"', NOISE_RE.sub("", b))


def scan(text):
    """(units, shown) for one file: a (header, kind, name, key, test) row per unit, the last of them the rest unit
    when rest() leaves any code, and rest()'s shown form."""
    code, mask = lex(text)
    pr, boxes, out, spans, skip = pairs(mask), [], [], [], 0
    for m in ITEM_RE.finditer(mask):
        kind = m.group("k1") or m.group("k2") or m.group("k3") or m.group("k4")
        s, start, ok = head(mask, m.start())
        span = item_end(mask, pr, m.end(), kind not in SEMI_ONLY) if ok and m.start() >= skip else None
        if span is None:
            continue
        hdr, (i, end) = box_of(boxes, m.start()), span
        if kind in ("impl", "trait") and mask[i] == "{":
            key = flat(code, mask, start, i)
            boxes.append((i, end, key))
            out.append((hdr, kind, m.group("n1") or key, key, False))
            spans.append((s, i + 1))
            continue
        name = m.group("n1") or m.group("n2") or m.group("n3") or ""
        out.append((hdr, kind, name, flat(code, mask, start, end), kind == "fn" and is_test(mask, start)))
        spans.append((s, end))
        skip = end
    key, shown = rest(code, mask, spans)
    if key:
        out.append(("", "rest", "", key, False))
    return out, shown


def units(text):
    return scan(text)[0]


def index(files):
    """(where, names, anyname) over {path: text}: where maps each (header, kind, key) to the files that hold it,
    names holds each (header, kind, name) and anyname each (kind, name), both without rest units."""
    where, names, anyname = {}, set(), set()
    for f in sorted(files):
        for hdr, kind, name, key, _ in units(files[f]):
            where.setdefault((hdr, kind, key), set()).add(f)
            if kind != "rest":
                names.add((hdr, kind, name))
                anyname.add((kind, name))
    return where, names, anyname


def judge(text, ix):
    """(verdict, rows, shown, home) for one dark file against index(): a (state, kind, name, test, named) row per
    unit, rest()'s shown form, and home, the reached file that holds most of its present units, else ''."""
    where, names, anyname = ix
    us, left = scan(text)
    rows, homes = [], Counter()
    for h, k, n, key, t in us:
        homes.update(where.get((h, k, key), ()))
        st = "present" if (h, k, key) in where else "changed" if (h, k, n) in names else "absent"
        rows.append((st, k, n, t, (k, n) in anyname))
    states = {r[0] for r in rows}
    verdict = "unique" if "absent" in states else "stale" if "changed" in states else "covered"
    return verdict, rows, left, min(homes, key=lambda g: (-homes[g], g)) if homes else ""


def fn_census(files, reach, dark):
    """{dark path: judge()} against the units of the reached .rs files."""
    ix = index({f: files[f] for f in reach if f.endswith(".rs")})
    return {f: judge(files[f], ix) for f in sorted(dark)}


def role(row):
    return "rest" if row[1] == "rest" else "tests" if row[3] else "fns" if row[1] == "fn" else "items"


def tally(label, sel, orph, res):
    """Two summary lines for the orphans in sel: files and lines per verdict, and units per state and role."""
    n, ln, c = Counter(), Counter(), Counter()
    for f in sel:
        n[res[f][0]] += 1
        ln[res[f][0]] += orph[f]["lines"]
        c.update((r[0], role(r)) for r in res[f][1])
        c.update(("unnamed", role(r)) for r in res[f][1] if r[0] == "absent" and not r[4])
    files = ", ".join(f"{v} {n[v]} ({ln[v]:,} lines)" for v in VERDICTS)
    units_ = "; ".join(f"{LABELS.get(s, s)}: " + ", ".join(f"{c[(s, x)]} {x}" for x in ROLES)
                       for s in STATES + ("unnamed",))
    return f"  {label}: {len(sel)} files: {files}", f"    units {units_}"


def summary(orph, res):
    """Verdicts and unit states for the twins and the other orphans, and the forms of rest."""
    for label, twin in (("twins", True), ("others", False)):
        print(*tally(label, [f for f in sorted(orph) if bool(orph[f]["twin"]) == twin], orph, res), sep="\n")
    forms = Counter(res[f][2][:60] for f in orph if res[f][2])
    print(f"  rest in {sum(forms.values())} files: " + ", ".join(f"{c}× {r}" for r, c in forms.most_common()))


def tsv_row(f, orph, res):
    """The TSV cells of one orphan: fns and tests by name, other units as kind:name."""
    v, rows, left, home = res[f]
    st = Counter(r[0] for r in rows)
    pick = [[r[2] if r[1] == "fn" else f"{r[1]}:{r[2]}" for r in rows if (r[0], role(r)) == want]
            for want in (("absent", "fns"), ("absent", "tests"), ("absent", "items"))]
    pick.append([f"{r[1]}:{r[2]}" for r in rows if r[0] == "changed"])
    state = next((r[0] for r in rows if r[1] == "rest"), "")
    return [f, orph[f]["lines"], orph[f]["twin"] or "-", v, home or "-", len(rows), st["present"], st["changed"],
            st["absent"], *(",".join(p) or "-" for p in pick), f"{state}: {left}" if left else "-"]


def write_tsv(path, orph, res):
    with open(path, "w") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for f in sorted(orph):
            fh.write("\t".join(map(str, tsv_row(f, orph, res))) + "\n")


def keys_of(t):
    return [u[3] for u in units(t)]


def kinds_of(t):
    return [u[1] for u in units(t)]


def fixture_index():
    return index({"a.rs": "fn a() { 1 }\nfn b() { 2 }\n", "r.rs": "impl R { fn n() {} }\nfn a() { 1 }\n",
                "m.rs": 'y!("a");\n#[cfg(x)]\nmod q;\n'})


def lex_cases():
    return [
        ("1 a } in a string does not end the fn",
         keys_of('fn a() { let s = "}"; let t = 1; }') == ['fna(){lets="}";lett=1;}']),
        ("2 a } in a char does not end the fn",
         keys_of("fn a() { let c = '}'; let t = 1; }") == ["fna(){letc='}';lett=1;}"]),
        ("3 lifetimes are not chars",
         [u[2] for u in units("fn a<'b>(x: &'b str) -> &'b str { x }\nfn c() {}")] == ["a", "c"]),
        ("4 a raw string holding \" and }",
         keys_of('fn a() { let r = r#"a"b}"#; let t = 1; }') == ['fna(){letr=r#"a"b}"#;lett=1;}']),
        ("5 nested block comment, trailing // comment",
         keys_of("fn a() { /* } /* } */ } */ let t = 1; } // }") == ["fna(){lett=1;}"]),
        ("6 // inside a string is no comment", keys_of('fn a() { let u = "http://x"; }') == ['fna(){letu="http://x";}']),
        ("15 byte literals",
         keys_of("fn a() { let x = (b'{', b\"{\"); let t = 1; }") == ["fna(){letx=(b'{',b\"{\");lett=1;}"]),
        ("16 an escaped quote char", keys_of("fn a() { let q = '\\''; let t = '{'; }") == ["fna(){letq='\\'';lett='{';}"]),
        ("17 whitespace inside a literal counts", keys_of('fn a() { "a b" }') != keys_of('fn a() { "ab" }')),
    ]


def unit_cases():
    return [
        ("7 a fn pointer type is no fn, nor is a macro call named like an item",
         kinds_of("struct S { f: fn(u8) -> u8 }") == ["struct"] and kinds_of("fn_like!(u8);\ntype_list!{ u8 }") == ["rest"]),
        ("8 trait header, then a decl and a default method under it",
         [u[:3] for u in units("trait T { fn d(&self); fn e(&self) {} }")]
         == [("", "trait", "T"), ("traitT", "fn", "d"), ("traitT", "fn", "e")]),
        ("10 pub dropped, unsafe kept", keys_of("pub fn a() {}") == keys_of("fn a() {}") != keys_of("unsafe fn a() {}")),
        ("11 const fn is a fn; const K an item through its ;",
         [(u[1], u[3]) for u in units("const fn k() -> u8 { 1 }\nconst K: u8 = { 1 };")]
         == [("fn", "constfnk()->u8{1}"), ("const", "constK:u8={1};")]),
        ("12 an item in a macro call or an impl in a type is no unit; the call is rest",
         kinds_of("m!(struct S;);\nfn f() -> impl Iterator<Item = u8> { std::iter::empty() }") == ["fn", "rest"]),
        ("13 #[test] marks a test, #[cfg(test)] does not",
         [u[4] for u in units("#[test]\nfn t() {}\n#[cfg(test)]\nfn h() {}")] == [True, False]),
        ("14 a nested fn is part of the outer one", kinds_of("fn a() { fn inner() {} }") == ["fn"]),
        ("19 macro_rules is one unit", kinds_of("macro_rules! m { () => { fn x() {} }; }") == ["macro_rules"]),
        ("20 tuple and unit structs, static, type",
         kinds_of('struct A(u32);\nstruct B;\nstatic S: &str = "}";\ntype T = u8;') == ["struct", "struct", "static", "type"]),
        ("21 rest drops attrs, use, mod lines, pub and units, keeps a macro call",
         scan('#![allow(x)]\npub use a::b;\nmod m;\n#[cfg(test)]\nmod t { pub fn f() {} }\nimpl X { }\n'
              'y!{ static ref Z: u8 = 1; }')[1]
         == "y!staticrefZ:u8=1"),
    ]


def census_cases(ix):
    return [
        ("9 the impl header is part of the key",
         judge("impl S { fn n() {} }", ix)[1][1][0] == "absent" and judge("impl R { fn n() {} }", ix)[0] == "covered"),
        ("18 present, changed, absent; covered, stale, unique",
         [judge(t, ix)[0] for t in ("fn a() {1}", "fn a() {1} fn b() { 3 }", "fn a() {1} fn c() {}")]
         == ["covered", "stale", "unique"]
         and [r[0] for r in judge("fn a() {1} fn b() { 3 } fn c() {}", ix)[1]] == ["present", "changed", "absent"]),
        ("22 named: n is absent under S but named under R; z is named nowhere",
         [r[4] for r in judge("impl S { fn n() {} }\nfn z() {}", ix)[1]] == [False, True, False]),
        ("23 home is the reached file that holds most present units, ties by path",
         judge("fn a() { 1 }\nfn b() { 2 }", ix)[3] == "a.rs" and judge("impl R { fn n() {} }", ix)[3] == "r.rs"
         and judge("fn a() { 1 }", ix)[3] == "a.rs" and judge("fn q() {}", ix)[3] == ""
         and judge("impl R { fn n() {} }\nfn a() { 1 }", ix)[3] == "r.rs"),
        ("24 rest is a unit: the same text is present, other text absent; include! lines are dropped",
         [r[:2] for r in judge('include!("p.rs");\ny!("a");', ix)[1]] == [("present", "rest")]
         and judge('fn a() { 1 }\ny!("b");', ix)[0] == "unique" and scan('y!("a b");')[1] == 'y!("…")'
         and judge('fn a() { 1 }\ninclude!(concat!(env!("OUT_DIR"), "/g.rs"));', ix)[0] == "covered"),
    ]


def self_test():
    """Rule 7: the lexer, unit and verdict cases, in number order."""
    cases = sorted(lex_cases() + unit_cases() + census_cases(fixture_index()), key=lambda c: int(c[0].split()[0]))
    for c, ok in cases:
        print(("ok   " if ok else "FAIL ") + c)
    bad = sum(not ok for _, ok in cases)
    print(f"{len(cases) - bad}/{len(cases)} cases pass")
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
        print(f"fn_census: {e}", file=sys.stderr)
        return 3
    rows, _, _, reach = oc.census(files, bl, tree)
    orph = {r["path"]: r for r in rows if r["status"] == "orphan"}
    res = fn_census(files, reach, orph)
    print(f"fn census at {pin}: {len(orph)} orphans against {len(reach)} reached files")
    summary(orph, res)
    if a.tsv:
        write_tsv(a.tsv, orph, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
