#!/usr/bin/env python3
"""Cite drift for the 0.73 drafts: map every `path:line` cite from the commit it was read at to a head.

Each cite was read at a commit, its pin. The pin is the first commit SHA after the cite in its own
clause (clauses end at ". " and ";"), else the last SHA before it in that clause, else the SHA in
the nearest heading above it, else the first SHA in the doc's preamble, else 316dee2cd4. A bare
`:N` after a cite, or after a file named without a line, cites line N of that file. In a table the
file must be in the same cell; a cell that names none takes the file the table's header row names.

A cited path with a "/" that exists from the repo root is that file. Otherwise, when several files at
the pin end with the cited name, the cite means the one longer path that the same doc names (in prose
or on a legend line; a named path from the root is that file), else the one candidate in a crate its
clause names. A bare name such as Cargo.toml is never taken as the root file on its own.
A doc defines short names on a line that starts "Abbreviations" or "Paths" (in YAML, a comment):
`wf` = `path/file.rs` for a file, `q/` = `path/dir/` for a directory prefix, and
`results.rs` = `path/to/results.rs` for a file name.

The cite is then mapped from its pin to --head (default origin/main) through `git diff -U0`:

  same       unchanged, at the same line numbers at head
  moved      unchanged, at new line numbers (printed)
  edited     a hunk touches the cited lines: re-read them by hand
  gone       the path is not at head, but it was on main (deleted or renamed)
  off-main   the pin is a branch commit, and the path has not reached head
  external   the path is outside the repo (cop-inbox/): listed, not checked
  orphan     a bare :N with no file before it in its paragraph or cell: listed, not checked
  absent     no file at the pin fits the cite                         (defect)
  ambiguous  several files fit, and neither the doc nor the clause picks one (defect)
  eof        the cited line is past the end of the file at the pin    (defect)

A cite stays true at its pin, so drift is information; only the three defects exit 1.
`unanchored` is a hint, not a verdict: the clause names identifiers in backticks and none of
them occurs within 2 lines of the cited lines at the pin.

Usage: cite_drift.py [--head REF] [--list] [--self-test]
"""
import re
import subprocess
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"],
                     capture_output=True, text=True).stdout.strip() or str(HERE)
DEFAULT_PIN = "316dee2cd4"
EXTERNAL = ("cop-inbox/",)
EXT = r"(?:rs|wgsl|yaml|yml|toml|py|sh|json|cu|ptx|cmd|md|txt)"
NAME = r"(?<![\w./-])((?:[\w.-]+/)*[\w.-]+\." + EXT + r")"
FILE_RE = re.compile(NAME + r":(\d+)(?:-(\d+))?((?:/\d+(?:-\d+)?)*)(?![\w-])")
BARE_RE = re.compile(NAME + r"(?![\w-]|:\d)")
PATH_RE = re.compile(r"((?:[\w.-]+/)+[\w.-]+\." + EXT + r")\b")
CONT_RE = re.compile(r"(?<![\w.:'\"-]):(\d+)(?:-(\d+))?(?![\d:\w])")
EXTRA_RE = re.compile(r"/(\d+)(?:-(\d+))?")
SHA_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{10}(?![0-9a-f])")
LEGEND_LINE = re.compile(r"(?:#+\s*)?(?:Abbreviations|Paths)")
LEGEND_RE = re.compile(r"`([\w./-]+)` = `([\w./-]+)`")
FILE_NAME = re.compile(r"[\w./-]+\." + EXT)
ABBR = re.compile(r"[a-z]{1,3}")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
HEADING_RE = re.compile(r"(#{1,6})\s")
MD_BREAK = re.compile(r"\s*(?:$|#{1,6}\s|[-*+] |\d+[.)] |\||```|>)")
YAML_BREAK = re.compile(r"\s*(?:$|#|- |[\w-]+:(?:\s|$))")
CLAUSE_END = re.compile(r"(?<=[.;])\s+")
PIPE = re.compile(r"\|")
DEFECTS = ("absent", "ambiguous", "eof")
ORDER = ("same", "moved", "edited", "gone", "off-main", "external", "orphan") + DEFECTS + ("self",)


def git(*args):
    r = subprocess.run(["git", "-C", TOP, *args], capture_output=True, text=True)  # pathspecs from the top
    return r.stdout if r.returncode == 0 else None


@lru_cache(None)
def is_commit(sha):
    return git("cat-file", "-e", sha + "^{commit}") is not None


@lru_cache(None)
def is_ancestor(a, b):
    return subprocess.run(["git", "-C", TOP, "merge-base", "--is-ancestor", a, b],
                          capture_output=True).returncode == 0


@lru_cache(None)
def tree(ref):
    out = git("ls-tree", "-r", "--full-tree", "--name-only", ref) or ""
    return frozenset(p for p in out.split("\n") if p)


@lru_cache(None)
def by_suffix(ref, suffix):
    return tuple(sorted(p for p in tree(ref) if p == suffix or p.endswith("/" + suffix)))


def at(ref, path):
    """The files a path can mean at ref: itself when it is a path from the root, else every file
    that ends with it. The crates carry their own .github/workflows/ci.yml, so a suffix alone
    would make the root workflow ambiguous."""
    if "/" in path and path in tree(ref):
        return (path,)
    return by_suffix(ref, path)


@lru_cache(None)
def lines_at(ref, path):
    out = git("show", f"{ref}:{path}")
    return None if out is None else out.split("\n")


@lru_cache(None)
def hunks(pin, head, path):
    """(old_start, old_len, new_start, new_len) for each hunk of path from pin to head."""
    out = git("diff", "-U0", "--no-color", "--no-ext-diff", "--no-renames", pin, head, "--", path) or ""
    found = []
    for m in re.finditer(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", out, re.M):
        os_, ol, ns, nl = m.groups()
        found.append((int(os_), 1 if ol is None else int(ol), int(ns), 1 if nl is None else int(nl)))
    return tuple(found)


def hunk_shift(a, b, hunk):
    """How far one hunk moves old lines a..b, or None when it touches them."""
    os_, ol, _, nl = hunk
    if ol == 0:                      # nl lines inserted after old line os_
        if os_ < a:
            return nl
        return None if os_ < b else 0
    if os_ + ol - 1 < a:
        return nl - ol
    return None if os_ <= b else 0


def map_range(a, b, hs):
    """Map old lines a..b through the hunks: ('same', a, b), ('moved', a2, b2) or ('edited', a, b)."""
    shift = 0
    for h in hs:
        d = hunk_shift(a, b, h)
        if d is None:
            return ("edited", a, b)
        shift += d
    return ("same" if shift == 0 else "moved", a + shift, b + shift)


def heading_level(md, fence, line):
    """A Markdown heading's level outside a code fence, else 0."""
    m = md and not fence and HEADING_RE.match(line)
    return len(m.group(1)) if m else 0


def line_role(md, fence, line):
    """What a line is to the unit before it: 'fence' (a ``` line), 'blank', 'alone' (a line in a
    fence, or a heading: a unit by itself), 'start' (a line that cannot continue a unit) or 'text'."""
    if md and line.lstrip().startswith("```"):
        return "fence"
    if not line.strip():
        return "blank"
    if fence or heading_level(md, fence, line):
        return "alone"
    return "start" if (MD_BREAK if md else YAML_BREAK).match(line) else "text"


def units(name, text):
    """Split a doc into logical units: a paragraph, a list item, a table row, a heading, a code line.
    Returns [(text, [(line_no, offset)], heading_level)]; a unit's lines keep their numbers."""
    md = name.endswith(".md")
    out, cur, fence = [], None, False
    for no, line in enumerate(text.split("\n"), 1):
        role = line_role(md, fence, line)
        if role == "fence":
            fence = not fence
        cur = add_line(out, cur, (no, line), role, heading_level(md, fence, line))
    return [tuple(u) for u in out]


def add_line(out, cur, numbered, role, level):
    """Add a line to the unit it continues, or start a unit with it in out. Returns the unit the
    next line may continue, or None."""
    no, line = numbered
    if role == "text" and cur is not None:
        cur[1].append((no, len(cur[0]) + 1))
        cur[0] += "\n" + line
        return cur
    if role == "blank":
        return None
    out.append([line, [(no, 0)], level])
    return None if role in ("fence", "alone") else out[-1]


def line_of(spans, off):
    no = spans[0][0]
    for n, o in spans:
        if o <= off:
            no = n
    return no


def pins_in(text):
    return [(m.start(), m.group(0)) for m in SHA_RE.finditer(text) if is_commit(m.group(0))]


def legend_kind(key):
    """Which table a legend key goes in: a directory prefix, a file name or an abbreviation."""
    if key.endswith("/"):
        return "dirs"
    if FILE_NAME.fullmatch(key):
        return "names"
    return "abbrs" if ABBR.fullmatch(key) else None


class Legend:
    """The short names a doc defines on its legend lines: abbreviations, directory prefixes and
    file names."""

    def __init__(self, text):
        self.abbrs, self.dirs, self.names = {}, {}, {}
        tables = {"abbrs": self.abbrs, "dirs": self.dirs, "names": self.names}
        pairs = [kv for line in text.split("\n") if LEGEND_LINE.match(line) for kv in LEGEND_RE.findall(line)]
        for k, v in pairs:
            kind = legend_kind(k)
            if kind:
                tables[kind][k] = v
        alts = "|".join(map(re.escape, self.abbrs))
        self.abbr_re = re.compile(r"(?<![\w./-])(" + alts + r"):(\d+)(?:-(\d+))?(?![\w-])") if alts else None

    def expand(self, tok):
        """A file token with its directory prefix and its short file name spelled out."""
        for k, v in self.dirs.items():
            if tok.startswith(k):
                tok = v + tok[len(k):]
        return self.names.get(tok, tok)


def preamble_pin(us):
    """The first SHA in a unit that starts on lines 1-8, before the first ## heading."""
    for utext, spans, level in us:
        if level == 2 or spans[0][0] > 8:
            return None
        ps = [] if level else pins_in(utext)
        if ps:
            return ps[0][1]
    return None


def push_heading(section, level, utext):
    """Close the open headings at this level or deeper, then open this one with its SHA or None."""
    while section and section[-1][0] >= level:
        section.pop()
    ps = pins_in(utext)
    section.append((level, ps[0][1] if ps else None))


def inside(off, taken):
    return any(s <= off < e for s, e in taken)


def ranges_of(m):
    """The line ranges of a FILE_RE match: N or N-M, then each /K or /K-L after it."""
    first = (int(m.group(2)), int(m.group(3) or m.group(2)))
    return [first] + [(int(c), int(d or c)) for c, d in EXTRA_RE.findall(m.group(4) or "")]


def line_cites(utext, lg):
    """The path:N cites, then the abbreviation:N cites outside them, with the spans both take.
    A found row is (offset, raw, file token, [(a, b)])."""
    ms = list(FILE_RE.finditer(utext))
    found = [(m.start(), m.group(0), lg.expand(m.group(1)), ranges_of(m)) for m in ms]
    taken = [m.span() for m in ms]
    for m in lg.abbr_re.finditer(utext) if lg.abbr_re else ():
        if not inside(m.start(), taken):
            a = int(m.group(2))
            found.append((m.start(), m.group(0), lg.abbrs[m.group(1)], [(a, int(m.group(3) or a))]))
            taken.append(m.span())
    return found, taken


def bare_names(utext, lg, taken):
    """Files named without a line: not cites, but a later bare :N belongs to them."""
    return [(m.start(), m.group(0), lg.expand(m.group(1)), []) for m in BARE_RE.finditer(utext)
            if not inside(m.start(), taken)]


def table_of(name, utext, table, found):
    """A table's first row is its header, and the last file it names scopes the table's bare :N.
    Returns (that file or None,) for a table row, else None."""
    if not (name.endswith(".md") and utext.lstrip().startswith("|")):
        return None
    return table or (max(found)[2] if found else None,)


def bare_lines(utext, found, taken, table):
    """Add each bare :N to found. It cites the latest file before it in its unit (in a table, in
    its cell, else the file the header row names); with none it is an orphan (None)."""
    cells = [m.start() for m in PIPE.finditer(utext)] if table else []
    for m in CONT_RE.finditer(utext):
        if inside(m.start(), taken):
            continue
        lo = max((p for p in cells if p < m.start()), default=0)
        prev = [f for f in found if lo <= f[0] < m.start()]
        tok = max(prev)[2] if prev else table[0] if table else None
        a = int(m.group(1))
        found.append((m.start(), m.group(0), tok, [(a, int(m.group(2) or a))]))


def clause_pin(unit_pins, off, cs, ce):
    """The first SHA after off in its clause cs..ce, else the last SHA before it, else None."""
    after = [s for o, s in unit_pins if off < o < ce]
    before = [s for o, s in unit_pins if cs <= o < off]
    return after[0] if after else before[-1] if before else None


def unit_cites(name, unit, found, fallback, doc_paths):
    """Yield one dict per cited range in a unit, pinned by the SHAs in its clause, else fallback."""
    utext, spans, _ = unit
    bounds = [0] + [m.end() for m in CLAUSE_END.finditer(utext)] + [len(utext) + 1]
    unit_pins = pins_in(utext)
    for off, raw, tok, ranges in sorted(found):
        cs = max(b for b in bounds if b <= off)
        ce = min(b for b in bounds if b > off)
        pin = clause_pin(unit_pins, off, cs, ce) or fallback
        for a, b in ranges:
            yield dict(doc=name, line=line_of(spans, off), raw=raw, tok=tok, a=a, b=b, pin=pin,
                       clause=utext[cs:ce], doc_paths=doc_paths)


def scan(name, text):
    """Yield one dict per cite in the doc: token, file token, lines, pin, clause, line_no."""
    lg = Legend(text)
    named = (lg.expand(m.group(1)) for m in PATH_RE.finditer(text))
    doc_paths = frozenset(named) | frozenset(lg.names.values())
    us = units(name, text)
    doc_pin = preamble_pin(us) or DEFAULT_PIN
    section, table = [], None               # section: stack of (level, pin or None)
    for unit in us:
        utext, _, level = unit
        if level:
            push_heading(section, level, utext)
        found, taken = line_cites(utext, lg)
        found += bare_names(utext, lg, taken)
        table = table_of(name, utext, table, found)
        bare_lines(utext, found, taken, table)
        fallback = next((p for _, p in reversed(section) if p), doc_pin)
        yield from unit_cites(name, unit, found, fallback, doc_paths)


def resolve(c):
    """The files at the pin the cite can mean: the path itself from the root, else those that end
    with its name, narrowed to the longer paths the same doc names, then to the crates its clause
    names."""
    tok, pin = c["tok"], c["pin"]
    cands = at(pin, tok)
    if len(cands) > 1:
        named = sorted({x for p in c["doc_paths"] if p != tok and p.endswith("/" + tok) for x in at(pin, p)})
        pool = named or list(cands)
        if len(pool) == 1:
            return pool
        words = set(re.findall(r"[\w-]+", c["clause"]))
        in_crate = [x for x in pool if x.startswith("crates/") and x.split("/")[1] in words]
        if len(in_crate) == 1:
            return in_crate
    return list(cands)


def unchecked(tok):
    """The status of a cite that is not checked against the repo, else None."""
    if tok is None:
        return "orphan"
    if (HERE / Path(tok).name).exists() and tok.endswith(".md"):
        return "self"
    return "external" if tok.startswith(EXTERNAL) else None


def n_lines(src):
    """A file's line count: the text git show prints ends in a newline, which split leaves as ''."""
    return len(src) - (1 if src and src[-1] == "" else 0)


def drift(c, path, head):
    """Map a cite that resolved to path at its pin to head."""
    pin = c["pin"]
    if path not in tree(head):
        on_main = is_ancestor(pin, head) or path in tree((git("merge-base", pin, head) or "").strip())
        return "gone" if on_main else "off-main"
    if git("rev-parse", pin) == git("rev-parse", head):
        return "same"
    status, a2, b2 = map_range(c["a"], c["b"], hunks(pin, head, path))
    c["new"] = (a2, b2)
    return status


def classify(c, head):
    """(status, info): info is a defect's detail, or the unanchored hint."""
    status = unchecked(c["tok"])
    if status:
        return status, None
    cands = resolve(c)
    if len(cands) != 1:
        return ("ambiguous", f"{len(cands)} files") if cands else ("absent", None)
    path = cands[0]
    src = lines_at(c["pin"], path)
    if src is None or c["b"] > n_lines(src):
        return "eof", path
    c["path"] = path
    return drift(c, path, head), anchor(c, src)


def anchor(c, src):
    names = set()
    for span in re.findall(r"`([^`]+)`", c["clause"]):
        if FILE_RE.search(span) or span.startswith(":"):
            continue
        names.update(n for n in IDENT_RE.findall(span) if not n.isdigit())
    if not names:
        return None
    window = "\n".join(src[max(0, c["a"] - 3):c["b"] + 2])
    return None if any(n in window for n in names) else "unanchored"


def docs():
    return sorted(list(HERE.glob("*.md")) + list(HERE.glob("contracts-draft/*.yaml")))


def lines(a, b):
    return f"{a}" + (f"-{b}" if b != a else "")


def verdict(c, head):
    """(status, unanchored, the cite's output row)."""
    status, info = classify(c, head)
    to = f" -> :{lines(*c['new'])}" if status == "moved" else ""
    detail = f" ({info})" if info and status in ("ambiguous", "eof") else ""
    tag = " [unanchored]" if info == "unanchored" else ""
    target = f"{c.get('path', c['tok'] or '?')}:{lines(c['a'], c['b'])}"
    return status, bool(tag), f"  {c['doc']}:{c['line']} {c['raw']} = {target} @{c['pin']}{to}{detail}{tag}"


def show(title, rows):
    if rows:
        print(f"{title}:")
        print("\n".join(rows))


def report(rows, listing):
    """Print the rows by status. The unanchored same rows print when there are 40 or fewer, or with
    --list; the other same rows print only with --list."""
    for status in ORDER[1:-1]:
        show(status, [r for s, _, r in rows if s == status])
    loose = [r for s, un, r in rows if s == "same" and un]
    if listing or len(loose) <= 40:
        show("unanchored", loose)
    if listing:
        print("same:")
        print("\n".join(r for s, un, r in rows if s == "same" and not un))


def run(head, listing):
    head_sha = (git("rev-parse", "--short=10", head) or "").strip()
    if not head_sha:
        print(f"cite_drift: head {head} is not a commit here (git fetch first)")
        return 2
    texts = {d.relative_to(HERE).as_posix(): d.read_text() for d in docs()}
    cites = [c for n, t in texts.items() for c in scan(n, t)]
    rows = [verdict(c, head) for c in cites]
    print(f"cite_drift: {len(cites)} cites in {len(texts)} files; head {head} = {head_sha}")
    counts = summary(cites, rows)
    report(rows, listing)
    return 1 if any(counts[k] for k in DEFECTS) else 0


def summary(cites, rows):
    """Print the pins and the count of each status; return the counts."""
    pins = Counter(c["pin"] for c in cites)
    print("pins: " + ", ".join(f"{p} {n}" for p, n in pins.most_common()))
    counts = Counter(s for s, _, _ in rows)
    unanchored = sum(un for _, un, _ in rows)
    print(", ".join(f"{k} {counts[k]}" for k in ORDER) + f"; unanchored {unanchored}")
    return counts


class Table:
    """The self-test's case table: a row holds when what it got is what it wants."""

    def __init__(self):
        self.n, self.fails = 0, []

    def check(self, label, got, want):
        self.n += 1
        if got != want:
            self.fails.append(f"{label}: got {got!r}, want {want!r}")

    def fail(self, why):
        self.n += 1
        self.fails.append(why)


def files_in(s):
    return [m.groups() for m in FILE_RE.finditer(s)]


def conts_in(s):
    return [m.group(0) for m in CONT_RE.finditer(s)]


def picks(name, text, *keys):
    """The named fields of each cite that scan finds in text."""
    return [tuple(c[k] for k in keys) for c in scan(name, text)]


def rows_tokens(check):
    check("file plain", files_in("(qwen35_session.rs:252-262)"), [("qwen35_session.rs", "252", "262", "")])
    check("file path", files_in("`crates/x/src/a_b.rs:376`"), [("crates/x/src/a_b.rs", "376", None, "")])
    check("file slash lines", files_in("fused_k_tests_q4k.rs:330/367"), [("fused_k_tests_q4k.rs", "330", None, "/367")])
    check("file yaml", files_in("at `contracts/kernel-registry-v1.yaml:86` on"), [("contracts/kernel-registry-v1.yaml", "86", None, "")])
    check("file txt", files_in("(`evidence/x/run-logs.txt:133`, added"), [("evidence/x/run-logs.txt", "133", None, "")])
    check("file dir prefix", files_in("| q/rope.rs:62 AVX2"), [("q/rope.rs", "62", None, "")])
    check("file no line", files_in("see quantize/mod.rs and x.rs"), [])
    check("file in url", files_in("https://example.org/a.rs:443"), [])
    check("file version", files_in("v0.70.1:3 and 0.70.1:4"), [])
    check("cont paren", conts_in("from_host (:380) to"), [":380"])
    check("cont pair", conts_in("`:229/:265` at"), [":229", ":265"])
    check("cont range", conts_in("and :140-146 (the"), [":140-146"])
    check("cont time", conts_in("01:17Z, 2026-10-04T01:19Z, C277: ok"), [])
    check("cont label", conts_in("**F-R5-2 [V]:** wgpu, RQ-6: x, a:b"), [])
    check("cont json", conts_in('--weights \'{"spec_depth":0,"falsification":1}\''), [])


def rows_map(check):
    check("map before", map_range(5, 5, [(10, 2, 10, 3)]), ("same", 5, 5))
    check("map after", map_range(20, 21, [(10, 2, 10, 3)]), ("moved", 21, 22))
    check("map overlap", map_range(9, 10, [(10, 2, 10, 3)]), ("edited", 9, 10))
    check("map insert at", map_range(15, 15, [(15, 0, 16, 4)]), ("same", 15, 15))
    check("map insert in", map_range(14, 16, [(15, 0, 16, 4)]), ("edited", 14, 16))
    check("map insert before", map_range(16, 16, [(15, 0, 16, 4)]), ("moved", 20, 20))
    check("map delete", map_range(33, 33, [(30, 3, 29, 0)]), ("moved", 30, 30))
    check("map delete in", map_range(31, 31, [(30, 3, 29, 0)]), ("edited", 31, 31))


def rows_scan(check):
    doc = ("# T\n\nA (`x.rs:10` and `:20` [V at 316dee2cd4; :30 at 00052c0128]).\n"
           "Then y.rs:5 @aca6f2d7f6.\n\n## S at c115c5ed02\n\n- z.rs:7 here\n  wrapped z.rs:8\n\n## U\n\nw.rs:9\n")
    check("pins", picks("t.md", doc, "tok", "a", "pin", "line"),
          [("x.rs", 10, "316dee2cd4", 3), ("x.rs", 20, "316dee2cd4", 3), ("x.rs", 30, "00052c0128", 3),
           ("y.rs", 5, "aca6f2d7f6", 4), ("z.rs", 7, "c115c5ed02", 8), ("z.rs", 8, "c115c5ed02", 9),
           ("w.rs", 9, DEFAULT_PIN, 13)])
    wrap = "## H\n\nSee x.rs:1 and\n:2 @aca6f2d7f6.\n"
    check("wrapped line", picks("w.md", wrap, "tok", "a", "pin", "line"),
          [("x.rs", 1, "aca6f2d7f6", 3), ("x.rs", 2, "aca6f2d7f6", 4)])
    leg ="Abbreviations: `wf` = `src/wgsl_forward.rs`, `q/` = `serve/src/quantize/`.\n\n| a | wf:943 | q/rope.rs:62 |\n"
    check("legend", picks("l.md", leg, "tok", "a"), [("src/wgsl_forward.rs", 943), ("serve/src/quantize/rope.rs", 62)])
    code = "```rust\n#[test]\nfn t() {} // q.rs:3 at aca6f2d7f6\n```\n# H\n"
    check("fence", picks("f.md", code, "tok", "pin"), [("q.rs", "aca6f2d7f6")])
    bare = "Every site writes 0.0 (g.rs:126, :143; q5.rs, 12 sites from :249 to :448; b.rs:296, :312).\n"
    check("bare name", picks("b.md", bare, "tok", "a"),
          [("g.rs", 126), ("g.rs", 143), ("q5.rs", 249), ("q5.rs", 448), ("b.rs", 296), ("b.rs", 312)])
    check("bare call", picks("b.md", "`f_tests.rs::t_x` and h.rs:3\n", "tok", "a"), [("h.rs", 3)])
    names = "Paths: `results.rs` = `crates/a/results.rs`, `q/` = `crates/q/`.\n\nsee results.rs:5 and q/m.rs:6\n"
    check("paths legend", picks("p.md", names, "tok", "a"), [("crates/a/results.rs", 5), ("crates/q/m.rs", 6)])
    tab = ("| Job (`.github/workflows/ci.yml`) | In needs (:617)? |\n|---|---|\n"
           "| `gx10` (:208) | a PR (:212); it is set (`s/t.sh:49`), so | sections (:231) |\n")
    check("table head", picks("t.md", tab, "tok", "a"),
          [(".github/workflows/ci.yml", 617), (".github/workflows/ci.yml", 208), (".github/workflows/ci.yml", 212),
           ("s/t.sh", 49), (".github/workflows/ci.yml", 231)])
    two = "| A (`x.rs`) |\n|---|\n| (:5) |\n\nThe fallback (:7) runs.\n\n| B |\n|---|\n| (:9) |\n"
    check("orphan", picks("o.md", two, "tok", "a"), [("x.rs", 5), (None, 7), (None, 9)])
    yml = "# Paths: `results.rs` = `crates/a/results.rs`\nk: results.rs:5 @316dee2cd4\n"
    check("yaml legend", picks("c.yaml", yml, "tok", "pin"), [("crates/a/results.rs", "316dee2cd4")])


def rows_real(check):
    """The mechanism on real history: R2 records the WGSL RoPE shader lines as :229/:265 at
    00052c0128 and :242/:278 at 316dee2cd4, both read by hand."""
    wf = "crates/aprender-compute/src/backends/gpu/device/linalg/wgsl_forward.rs"
    hs = hunks("00052c0128", "316dee2cd4", wf)
    check("real map 229", map_range(229, 229, hs)[0:2], ("moved", 242))
    check("real map 265", map_range(265, 265, hs)[0:2], ("moved", 278))
    serve = "crates/aprender-serve/src/quantize/"
    for label, text, want in [("real absent", "no_such_file_zz.rs:5", "absent"),
                              ("real eof", "qwen35_session.rs:999999", "eof"),
                              ("real ambiguous", "mod.rs:5", "ambiguous"),
                              ("real same", "qwen35_session.rs:472", "same"),
                              ("real txt", "evidence/pmat919-postmerge-gx10-blackwell/run-logs.txt:133", "same"),
                              ("real txt eof", "scripts/include_fmt_baseline.txt:999999", "eof"),
                              ("real external", "(cop-inbox/processed/x.md:1)", "external"),
                              ("real crate word", "aprender-serve sets it (Cargo.toml:31-32)",
                               ("same", "crates/aprender-serve/Cargo.toml")),
                              ("real root path", "aprender-serve gates it (.github/workflows/ci.yml:617)",
                               ("same", ".github/workflows/ci.yml")),
                              ("real named root", "`.github/workflows/ci.yml` runs it; see ci.yml:617",
                               ("same", ".github/workflows/ci.yml")),
                              ("real two named", f"`{serve}fused_q5k_q6k.rs`, `{serve}tests/fused_q5k_q6k.rs`; "
                               "fused_q5k_q6k.rs:118", "ambiguous")]:
        c = next(scan("r.md", text + " @316dee2cd4\n"), None)
        got = classify(c, "316dee2cd4")[0] if c else "not a cite"
        check(label, (got, c and c.get("path")) if isinstance(want, tuple) else got, want)
    c = next(scan("r.md", f"`{serve}fused_q5k_q6k.rs` has fused_q6k_dot_simd (fused_q5k_q6k.rs:118 @316dee2cd4)\n"))
    check("real one named", (classify(c, "316dee2cd4")[0], c.get("path")), ("same", f"{serve}fused_q5k_q6k.rs"))


def self_test():
    """The case table. Each row must hold; a row that does not is printed and the run exits 2."""
    t = Table()
    for part in (rows_tokens, rows_map, rows_scan):
        part(t.check)
    if is_commit("00052c0128") and is_commit("316dee2cd4"):
        rows_real(t.check)
    else:
        t.fail("real-history rows need 00052c0128 and 316dee2cd4 (git fetch)")
    print(f"cite_drift self-test: {t.n - len(t.fails)}/{t.n} rows hold")
    for f in t.fails:
        print("  FAIL " + f)
    return 2 if t.fails else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--self-test" in args:
        sys.exit(self_test())
    head = args[args.index("--head") + 1] if "--head" in args else "origin/main"
    sys.exit(run(head, "--list" in args))
