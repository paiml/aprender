#!/usr/bin/env python3
"""Which side moved: for each changed unit of a dark file, the last commit after which it matched its compiled copy,
and which side has changed since.

fn_census.py calls a dark unit changed when a compiled file holds a unit of the same header, kind and name with other
text. This replays the pin's first-parent history (a merge counts by its diff to its first parent) over the dark file
and the compiled files that hold that name at the pin, and reads the unit's keys on both sides after each commit that
touched one of them. With --source PREFIX=DIR, a unit whose dark file and live holders all lie under PREFIX
first replays DIR, a clone of the repository that the pin's squashed import of PREFIX came from, up to the imported
commit. The unit is synced after a commit when the dark file holds it and every dark key is also a
compiled key. After its last synced commit it is `dark` when only the dark copy has changed since, `live` when only
the compiled copy has, and `both` when both have. It is `never` when no commit left it synced, and its changes then
count from the first commit after which both sides held it. A `live` unit is only out of date; a `dark`, `both` or
`never` unit can hold code that the compiled copy lacks.

A dark file's dark-only commits touched it but none of the compiled files that hold one of its names (impl headers
aside), after the first of those files existed. They catch what the keys leave out, such as a doc comment. A dark
file under a --source PREFIX has them in DIR's history too, judged against its namesakes under PREFIX and named
<DIR's basename>:<commit>; DIR is replayed over every such dark file and namesake, not only over changed units. The
replay checks itself: before each commit a file must hold the text the previous commit left, and after the last one
the pin's text; a source's last text must be the text that its import left on main. A gap fails the run.

Usage: unit_history.py --pin REV [--verdict V ...] [--source PREFIX=DIR ...] [--tsv PATH] | --self-test
rc 3 when the pin lists no crates/*/src .rs file: orphan_census.load() refuses a vacuous answer.
"""
import argparse
import os
import os.path as pp
import subprocess
import sys
import tempfile
from collections import Counter

sys.path.insert(0, pp.dirname(pp.abspath(__file__)))
import fn_census as fc  # noqa: E402
import orphan_census as oc  # noqa: E402

MARK = "\x02"
STATES = ("dark", "both", "never", "live", "synced")
COLUMNS = ("path", "kind", "name", "header", "state", "synced", "dark_moves", "live_moves", "live")
NONE = frozenset()


def who_moved(seq):
    """(state, last synced index or None, dark move indices, live move indices) for one unit, given seq, its (dark
    keys, live keys) pair of frozensets after each commit, oldest first."""
    ok = [bool(d) and d <= lv for d, lv in seq]
    last = max((i for i, s in enumerate(ok) if s), default=None)
    if last == len(seq) - 1:
        return "synced", last, [], []
    start = last if last is not None else next((i for i, (d, lv) in enumerate(seq) if d and lv), len(seq))
    dark = [i for i in range(start + 1, len(seq)) if seq[i][0] != seq[i - 1][0]]
    live = [i for i in range(start + 1, len(seq)) if seq[i][1] != seq[i - 1][1]]
    return ("never" if last is None else "both" if dark and live else "dark" if dark else "live"), last, dark, live


def git(repo, *args, inp=None):
    return subprocess.run(["git", "-C", repo, *args], input=inp, capture_output=True, check=True).stdout


def parse_log(text):
    """[(sha, short, date, subject, paths)] from git log --format=MARK%H %h %cs %s --name-only."""
    out = []
    for line in text.splitlines():
        if line.startswith(MARK):
            sha, short, date, subj = (line[1:].split(" ", 3) + [""])[:4]
            out.append((sha, short, date, subj, set()))
        elif line.strip() and out:
            out[-1][4].add(line.strip())
    return out


def parse_batch(blob, n):
    """The n texts of a git cat-file --batch reply, None for a missing object."""
    out, i = [], 0
    for _ in range(n):
        nl = blob.index(b"\n", i)
        head = blob[i:nl].split()
        if head[-1] == b"missing":
            out.append(None)
            i = nl + 1
            continue
        size = int(head[2])
        out.append(blob[nl + 1:nl + 1 + size].decode("utf-8", "replace"))
        i = nl + 2 + size
    return out


def gaps(steps, final):
    """Where one path's replay breaks: steps is its (commit, text before, text after) per commit that touched it, in
    order. The text before each must be the text after the one before (None before the first), and the last text
    must be final, the pin's."""
    bad, prev = [], None
    for c, before, after in steps:
        if before != prev:
            bad.append(c)
        prev = after
    return bad + (["pin"] if prev != final else [])


def read(repo, specs):
    """The texts of rev:path specs in repo, None for a missing one."""
    return parse_batch(git(repo, "cat-file", "--batch", inp="".join(s + "\n" for s in specs).encode()), len(specs))


def replay(repo, rev, paths, final):
    """(log, after, broken): rev's first-parent log over paths in repo, oldest first; {(log index, path): text}
    after each commit that touched the path, None once deleted; and {path: gaps} for each path whose replay breaks,
    final holding the text each path must end on. The paths go to git as :(top,literal) pathspecs, from repo's root
    and as plain text whatever the cwd: from docs/lookahead/0.73 a bare path matched nothing."""
    log = parse_log(git(repo, "log", "--first-parent", "--diff-merges=first-parent", "--reverse",
                        f"--format={MARK}%H %h %cs %s", "--name-only", rev, "--",
                        *(f":(top,literal){p}" for p in sorted(paths))).decode())
    asks = [(i, p) for i, c in enumerate(log) for p in sorted(c[4] & paths)]
    texts = read(repo, [f"{log[i][0]}{up}:{p}" for i, p in asks for up in ("", "^")])
    after, steps = {}, {p: [] for p in paths}
    for (i, p), now, before in zip(asks, texts[0::2], texts[1::2]):
        after[(i, p)] = now
        steps[p].append((log[i][1], before, now))
    broken = {p: g for p in sorted(paths) if (g := gaps(steps[p], final.get(p)))}
    return log, after, broken


def squash_rev(subjects, prefix):
    """The commit that the one squashed import of prefix came from, out of the squash commits' subjects."""
    head = f"Squashed '{prefix}/' content from commit "
    revs = [s[len(head):].strip() for s in subjects.splitlines() if s.startswith(head)]
    if len(revs) != 1:
        raise SystemExit(f"{prefix}: {len(revs)} squashed imports, want 1")
    return revs[0]


def seam(at_rev, first, prefix):
    """The paths whose text at the imported commit differs from the text that the import left on main."""
    return [f"{prefix}/{p}" for p, t in sorted(at_rev.items()) if t is not None and first.get(f"{prefix}/{p}") != t]


def holders(files):
    """{(header, kind, name): the files that hold such a unit}, rest units aside."""
    out = {}
    for f in sorted(files):
        for h, k, n, _, _ in fc.units(files[f]):
            if k != "rest":
                out.setdefault((h, k, n), set()).add(f)
    return out


def changed_units(text, ix, holds):
    """{(header, kind, name): compiled holders} for the changed units of one dark file."""
    where, names, _ = ix
    return {(h, k, n): holds[(h, k, n)] for h, k, n, key, _ in fc.units(text)
            if k != "rest" and (h, k, key) not in where and (h, k, n) in names}


def namesakes(text, ix, holds):
    """The compiled files that hold a name of one dark file, rest units and impl headers aside."""
    names = ix[1]
    return set().union(*(holds[(h, k, n)] for h, k, n, _, _ in fc.units(text)
                         if k not in ("rest", "impl") and (h, k, n) in names))


def key_map(text, cache):
    """{(header, kind, name): frozenset of keys} for one text; {} for no file."""
    if text is None:
        return {}
    if text not in cache:
        m = {}
        for h, k, n, key, _ in fc.units(text):
            m.setdefault((h, k, n), set()).add(key)
        cache[text] = {u: frozenset(v) for u, v in m.items()}
    return cache[text]


def watchers(units):
    """{path: units}: the units whose dark file or one of whose live holders is path."""
    watch = {}
    for u, live in units.items():
        for p in {u[0], *live}:
            watch.setdefault(p, []).append(u)
    return watch


def advance(i, c, after, cur, watch):
    """Set cur to the text of each path that log commit i (c) changed, and return the units those paths touch."""
    hit = set()
    for p in c[4]:
        if (i, p) in after:
            cur[p] = after[(i, p)]
            hit.update(watch.get(p, ()))
    return hit


def keys_at(u, units, cur, cache):
    """(dark keys, live keys) of unit u in the texts cur."""
    dk = key_map(cur.get(u[0]), cache).get(u[1:], NONE)
    return dk, NONE.union(*(key_map(cur.get(q), cache).get(u[1:], NONE) for q in units[u]))


def sweep(log, after, units):
    """{unit: [(commit, (dark keys, live keys))]} after each commit that touched the unit's dark file or one of its
    live holders. A unit is (dark path, header, kind, name); units maps it to its live holders."""
    watch, cur, cache = watchers(units), {}, {}
    seqs = {u: [] for u in units}
    for i, c in enumerate(log):
        hit = advance(i, c, after, cur, watch)
        for u in sorted(hit):
            seqs[u].append((c, keys_at(u, units, cur, cache)))
    return seqs


def dark_only(log, d, ns):
    """The log indices of the commits that touched d but no file of ns, after the first commit that touched ns."""
    born = next((i for i, c in enumerate(log) if c[4] & ns), None)
    return [] if born is None else [i for i, c in enumerate(log) if i > born and d in c[4] and not c[4] & ns]


def dark_only_all(dark, ns, srcs, log):
    """{dark path: its dark-only commits} over the log of the source whose prefix holds it, if one does, and then
    main's log; srcs holds each source's (prefix, log)."""
    out = {}
    for d in dark:
        hist = [c for p, sl in srcs if d.startswith(p + "/") for c in sl] + log
        out[d] = [hist[i] for i in dark_only(hist, d, ns[d])]
    return out


def source_paths(prefix, units, ns):
    """(the units whose dark file and live holders all lie under prefix, the paths below prefix that a source replays
    for them and for the dark-only commits of each dark file under prefix: its own path and its namesakes there)."""
    head = prefix + "/"
    mine = {u: live for u, live in units.items() if all(p.startswith(head) for p in (u[0], *live))}
    held = {p for u, live in mine.items() for p in (u[0], *live)}
    held |= {p for d, names in ns.items() if d.startswith(head) for p in (d, *names)}
    return mine, sorted(p[len(head):] for p in held if p.startswith(head))


def source_states(prefix, repo, rev, units, ns, first):
    """({unit: states}, broken, log) from repo, whose tree at rev is what prefix imported: the states of the units
    whose dark file and live holders all lie under prefix, over repo's first-parent history up to rev, and that
    history over the paths source_paths() names, each commit named <repo's basename>:<commit> and its paths put under
    prefix. first holds each path's text after its first commit on main."""
    mine, paths = source_paths(prefix, units, ns)
    if not paths:
        return {}, {}, []
    at_rev = dict(zip(paths, read(repo, [f"{rev}:{p}" for p in paths])))
    log, after, broken = replay(repo, rev, set(paths), at_rev)
    tag = pp.basename(pp.abspath(repo))
    log = [(c[0], f"{tag}:{c[1]}", c[2], c[3], {f"{prefix}/{p}" for p in c[4]}) for c in log]
    after = {(i, f"{prefix}/{p}"): t for (i, p), t in after.items()}
    bad = {f"{prefix}/{p}": g for p, g in broken.items()}
    bad.update({p: ["import"] for p in seam(at_rev, first, prefix)})
    return sweep(log, after, mine), bad, log


def before(pin, prefix, repo, units, ns, first):
    """source_states() of repo, the source of the pin's squashed import of prefix, up to the imported commit."""
    rev = squash_rev(git(".", "log", "--format=%s", f"--grep=^Squashed '{prefix}/' content from commit ", pin)
                     .decode(), prefix)
    return source_states(prefix, repo, rev, units, ns, first)


def results(seqs):
    """{unit: (state, last synced commit or None, dark move commits, live move commits)}."""
    out = {}
    for u, seq in seqs.items():
        cs = [c for c, _ in seq]
        st, last, dm, lm = who_moved([s for _, s in seq])
        out[u] = (st, None if last is None else cs[last], [cs[j] for j in dm], [cs[j] for j in lm])
    return out


def select(pin, want):
    """(files, dark, units, ns) at pin: the dark files that are no twin and whose verdict is in want, their changed
    units with the compiled files that hold each, and each dark file's namesake files."""
    files, bl, tree = oc.load(pin)
    rows, _, _, reach = oc.census(files, bl, tree)
    rs = {f: files[f] for f in reach if f.endswith(".rs")}
    ix, holds = fc.index(rs), holders(rs)
    dark = sorted(r["path"] for r in rows if r["status"] == "orphan" and not r["twin"])
    dark = [d for d in dark if fc.judge(files[d], ix)[0] in want]
    units = {(d, *u): live for d in dark for u, live in changed_units(files[d], ix, holds).items()}
    return files, dark, units, {d: namesakes(files[d], ix, holds) for d in dark}


def short(p):
    return p.split("/src/", 1)[-1]


def open_units(res):
    """{dark path: descriptions} of the units whose state is dark, both or never."""
    keep = {}
    for u, (s, _, dm, lm) in sorted(res.items()):
        if s in ("dark", "both", "never"):
            moves = ",".join(c[1] for c in dm) or "none"
            keep.setdefault(u[0], []).append(f"{u[3] or u[2]} {s} (dark moves {moves}; live moves {len(lm)})")
    return keep


def report_dark_only(dark, donly):
    """Print the dark-only commits of each dark file that has one."""
    hit = {d: cs for d, cs in donly.items() if cs}
    print(f"  dark-only commits: {len(hit)} of {len(dark)} files")
    for d, cs in sorted(hit.items()):
        for c in cs:
            print(f"    {short(d)}: {c[1]} {c[2]} {c[3][:80]}")


def report(dark, res, donly):
    """Print the unit states, the files that hold a dark, both or never unit, and the dark-only commits."""
    st = Counter(r[0] for r in res.values())
    print("  units: " + ", ".join(f"{s} {st[s]}" for s in STATES))
    keep = open_units(res)
    print(f"  files: {len(keep)} of {len(dark)} hold a dark, both or never unit")
    for d, us in sorted(keep.items()):
        print(f"    {short(d)}: " + "; ".join(us))
    report_dark_only(dark, donly)


def write_tsv(path, units, res):
    with open(path, "w") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for u in sorted(res):
            s, last, dm, lm = res[u]
            cells = [u[0], u[2], u[3], u[1] or "-", s, "-" if last is None else f"{last[1]} {last[2]}",
                     ",".join(c[1] for c in dm) or "-", ",".join(c[1] for c in lm) or "-",
                     ",".join(sorted(units[u]))]
            fh.write("\t".join(cells) + "\n")


def move_cases():
    a, b, c, e = frozenset("a"), frozenset("b"), frozenset("c"), NONE
    return [
        ("1 synced at the end, extra live keys allowed", who_moved([(a, a), (a, a | b)]) == ("synced", 1, [], [])),
        ("2 the live copy moved", who_moved([(a, a), (a, b)]) == ("live", 0, [], [1])),
        ("3 the dark copy moved", who_moved([(a, a), (b, a)]) == ("dark", 0, [1], [])),
        ("4 both moved in one commit", who_moved([(a, a), (b, c)]) == ("both", 0, [1], [1])),
        ("5 the live copy moved, then the dark one", who_moved([(a, a), (a, b), (c, b)]) == ("both", 0, [2], [1])),
        ("6 never synced: moves count once both sides hold the unit",
         who_moved([(a, e), (a, b), (c, b)]) == ("never", None, [2], [])
         and who_moved([(e, a), (b, a), (b, c)]) == ("never", None, [], [2])),
        ("7 the last sync counts", who_moved([(a, a), (a, b), (b, b), (b, c)]) == ("live", 2, [], [3])),
        ("8 a dark side without the unit is not synced", who_moved([(a, a), (e, a)]) == ("dark", 0, [1], [])),
        ("9 a dark key the live side lacks breaks the sync",
         who_moved([(a | b, a | b), (a | b, a)]) == ("live", 0, [], [1])),
        ("10 both moved to the same text: still synced", who_moved([(a, a), (b, b)]) == ("synced", 1, [], [])),
    ]


def git_cases():
    log = parse_log(f"{MARK}aaa a 2026-01-01 fix: x y\n\np/q.rs\nr.rs\n{MARK}bbb b 2026-01-02 \n\nr.rs\n")
    return [
        ("11 log: marks, subjects with spaces, names",
         log == [("aaa", "a", "2026-01-01", "fix: x y", {"p/q.rs", "r.rs"}), ("bbb", "b", "2026-01-02", "", {"r.rs"})]),
        ("12 cat-file: a blob, a missing object, an empty blob",
         parse_batch(b"x blob 3\nabc\nc:p missing\ny blob 0\n\n", 3) == ["abc", None, ""]),
        ("13 a clean replay has no gaps", gaps([("c1", None, "x"), ("c2", "x", "y")], "y") == []),
        ("14 a skipped change shows at the next commit and at the pin",
         gaps([("c1", None, "x"), ("c2", "z", "y")], "w") == ["c2", "pin"]),
        ("15 a file deleted and added back", gaps([("c1", None, "x"), ("c2", "x", None), ("c3", None, "x")], "x") == []),
    ]


def unit_cases():
    files = {"a.rs": "fn a() { 1 }\nfn b() { 2 }\n", "r.rs": "impl R { fn n() {} }\nfn a() { 1 }\n"}
    ix, holds = fc.index(files), holders(files)
    log = [("s0", "a", "d", "", {"d.rs"}), ("s1", "b", "d", "", {"n.rs"}), ("s2", "c", "d", "", {"d.rs"}),
           ("s3", "e", "d", "", {"d.rs", "n.rs"})]
    hist = [("s0", "a", "d", "", {"d.rs", "l.rs"}), ("s1", "b", "d", "", {"l.rs"}), ("s2", "c", "d", "", {"x.rs"}),
            ("s3", "e", "d", "", {"d.rs"})]
    after = {(0, "d.rs"): "fn f() { 1 }", (0, "l.rs"): "fn f() { 1 }", (1, "l.rs"): "fn f() { 2 }", (2, "x.rs"): "",
             (3, "d.rs"): "fn f() { 3 }"}
    seq = sweep(hist, after, {("d.rs", "", "fn", "f"): {"l.rs"}})[("d.rs", "", "fn", "f")]
    src = [("r0", "r:a", "d", "", {"p/d.rs"}), ("r1", "r:b", "d", "", {"p/d.rs", "p/n.rs"}),
           ("r2", "r:c", "d", "", {"p/d.rs"})]
    main = [("m0", "f", "d", "", {"pq/d.rs"}), ("m1", "g", "d", "", {"p/d.rs", "p/n.rs"}),
            ("m2", "h", "d", "", {"p/d.rs", "pq/d.rs"})]
    return [
        ("16 dark-only: after the namesake existed, and not touching it", dark_only(log, "d.rs", {"n.rs"}) == [2]),
        ("17 dark-only: no namesake ever, no commit", dark_only(log, "d.rs", {"m.rs"}) == []),
        ("18 changed units and their holders; present and absent units are left out",
         changed_units("fn a() { 1 }\nfn b() { 3 }\nfn c() {}\nimpl R { fn n() { 4 } }", ix, holds)
         == {("", "fn", "b"): {"a.rs"}, ("implR", "fn", "n"): {"r.rs"}}),
        ("19 namesakes: the holders of present or changed names, impl headers aside",
         namesakes("impl R { fn q() {} }", ix, holds) == set() and namesakes("fn b() { 3 }", ix, holds) == {"a.rs"}
         and namesakes("fn a() { 1 }", ix, holds) == {"a.rs", "r.rs"}),
        ("20 sweep: one state per commit that touched a watched file",
         [c[0] for c, _ in seq] == ["s0", "s1", "s3"] and who_moved([s for _, s in seq])[0] == "both"),
        ("21 squash subjects: the commit of the one import of the prefix; a subject that quotes one is not one",
         squash_rev("Squashed 'p/' content from commit abc\nSquashed 'pq/' content from commit def\n"
                    "Revert \"Squashed 'p/' content from commit abc\"\n", "p") == "abc"),
        ("22 seam: the source's last text must be what the import left; a later file is no seam",
         seam({"a.rs": "x", "b.rs": None, "c.rs": "y"}, {"p/a.rs": "x", "p/b.rs": "q", "p/c.rs": "z"}, "p")
         == ["p/c.rs"]),
        ("24 dark-only over the log of the source whose prefix holds the file, then main's; pq/ is not under p/",
         dark_only_all(["p/d.rs", "pq/d.rs"], {"p/d.rs": {"p/n.rs"}, "pq/d.rs": {"p/n.rs"}}, [("p", src)], main)
         == {"p/d.rs": [src[2], main[2]], "pq/d.rs": [main[2]]}),
    ]


def subdir_cases():
    """Case 23: a scratch repo replayed from its docs/sub, as a run from docs/lookahead/0.73 replays this one."""
    p = "crates/p/src/a.rs"
    with tempfile.TemporaryDirectory() as tmp:
        shas = oc.scratch_repo(tmp, [{p: "fn a() { 1 }\n", "docs/sub/x.md": "x\n"}, {p: "fn a() { 2 }\n"}])
        got = oc.at(os.path.join(tmp, "docs", "sub"), replay, ".", shas[-1], {p}, {p: "fn a() { 2 }\n"})
    return [("23 from a subdirectory replay() finds both commits of crates/p/src/a.rs, and no gap",
             isinstance(got, tuple) and [c[0] for c in got[0]] == shas and got[2] == {})]


def source_cases():
    """Case 25: a scratch source repo, the tree that p/ imported, replayed for a dark file with no changed unit."""
    ns, doc = {"p/a.rs": {"p/n.rs"}}, "/// doc\nfn a() { 1 }\n"
    with tempfile.TemporaryDirectory() as tmp:
        shas = oc.scratch_repo(tmp, [{"a.rs": "fn a() { 1 }\n", "n.rs": "fn a() { 1 }\n"}, {"a.rs": doc},
                                     {"x.md": "x\n"}])
        got = oc.at(tmp, source_states, "p", tmp, shas[-1], {}, ns, {"p/a.rs": doc, "p/n.rs": "fn a() { 1 }\n"})
        tag = pp.basename(tmp)
    ok = isinstance(got, tuple) and got[:2] == ({}, {}) and [c[0] for c in got[2]] == shas[:2]
    name = got[2][1][1].split(":", 1) if ok else ["", ""]
    return [("25 a source replays a dark file that has no changed unit, and its namesake; the commit that touched "
             "only the dark file is dark-only, named <repo's basename>:<commit>",
             ok and dark_only_all(["p/a.rs"], ns, [("p", got[2])], [])["p/a.rs"] == [got[2][1]]
             and name[0] == tag and len(name[1]) >= 7 and shas[1].startswith(name[1]))]


def self_test():
    """Rule 7: the move, git-parse, unit, subdirectory and source cases, in number order."""
    cases = sorted(move_cases() + git_cases() + unit_cases() + subdir_cases() + source_cases(),
                   key=lambda c: int(c[0].split()[0]))
    for c, ok in cases:
        print(("ok   " if ok else "FAIL ") + c)
    bad = sum(not ok for _, ok in cases)
    print(f"{len(cases) - bad}/{len(cases)} cases pass")
    return 1 if bad else 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pin")
    ap.add_argument("--verdict", nargs="+", default=["stale"], choices=fc.VERDICTS)
    ap.add_argument("--source", action="append", default=[], metavar="PREFIX=DIR")
    ap.add_argument("--tsv")
    ap.add_argument("--self-test", action="store_true")
    return ap


def first_texts(after):
    """{path: its text after its first commit on main}."""
    first = {}
    for i, p in sorted(after):
        first.setdefault(p, after[(i, p)])
    return first


def add_sources(pin, specs, units, ns, seqs, first, broken):
    """Put each unit's states in the PREFIX=DIR source clones, up to their import, in front of its states on main;
    add the clones' gaps to broken and return each clone's (prefix, log)."""
    srcs = []
    for spec in specs:
        prefix, repo = spec.split("=", 1)
        prefix = prefix.rstrip("/")
        got, bad, log = before(pin, prefix, repo, units, ns, first)
        broken.update(bad)
        srcs.append((prefix, log))
        for u, s in got.items():
            seqs[u] = s + seqs[u]
    return srcs


def run(pin, verdict, specs, tsv):
    """Replay the units of the dark files whose verdict is in verdict, print their history and, if tsv is set,
    write it there; 1 if a replay has a gap."""
    files, dark, units, ns = select(pin, set(verdict))
    log, after, broken = replay(".", pin, set(dark).union(*ns.values()), files)
    seqs = sweep(log, after, units)
    srcs = add_sources(pin, specs, units, ns, seqs, first_texts(after), broken)
    pre = sum(len(sl) for _, sl in srcs)
    print(f"unit history at {pin}: {len(dark)} {'/'.join(verdict)} dark files, {len(units)} changed units; "
          f"replayed {len(log)} first-parent commits on main and {pre} in sources, {len(broken)} paths with gaps")
    for p, g in broken.items():
        print(f"  GAP {p}: {', '.join(g)}")
    if broken:
        return 1
    res = results(seqs)
    report(dark, res, dark_only_all(dark, ns, srcs, log))
    if tsv:
        write_tsv(tsv, units, res)
    return 0


def main():
    ap = parser()
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.pin:
        ap.error("--pin or --self-test")
    try:
        return run(oc.git("rev-parse", "--short=10", a.pin).decode().strip(), a.verdict, a.source, a.tsv)
    except oc.Vacuous as e:
        print(f"unit_history: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
