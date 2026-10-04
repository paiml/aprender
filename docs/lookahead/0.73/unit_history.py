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
aside), after the first of those files existed. They catch what the keys leave out, such as a doc comment. The replay
checks itself: before each commit a file must hold the text the previous commit left, and after the last one the
pin's text; a source's last text must be the text that its import left on main. A gap fails the run.

Usage: unit_history.py --pin REV [--verdict V ...] [--source PREFIX=DIR ...] [--tsv PATH] | --self-test
"""
import argparse
import os.path as pp
import subprocess
import sys
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
    final holding the text each path must end on."""
    log = parse_log(git(repo, "log", "--first-parent", "--diff-merges=first-parent", "--reverse",
                        f"--format={MARK}%H %h %cs %s", "--name-only", rev, "--", *sorted(paths)).decode())
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


def before(pin, prefix, repo, units, first):
    """({unit: states}, broken, commits) from repo, the source of the pin's squashed import of prefix: the states of
    the units whose dark file and live holders all lie under prefix, over repo's first-parent history up to the
    imported commit. first holds each path's text after its first commit on main."""
    rev = squash_rev(git(".", "log", "--format=%s", f"--grep=^Squashed '{prefix}/' content from commit ", pin)
                     .decode(), prefix)
    mine = {u: live for u, live in units.items() if all(p.startswith(prefix + "/") for p in (u[0], *live))}
    paths = sorted({p[len(prefix) + 1:] for u, live in mine.items() for p in (u[0], *live)})
    if not paths:
        return {}, {}, 0
    at_rev = dict(zip(paths, read(repo, [f"{rev}:{p}" for p in paths])))
    log, after, broken = replay(repo, rev, set(paths), at_rev)
    tag = pp.basename(pp.abspath(repo))
    log = [(c[0], f"{tag}:{c[1]}", c[2], c[3], {f"{prefix}/{p}" for p in c[4]}) for c in log]
    after = {(i, f"{prefix}/{p}"): t for (i, p), t in after.items()}
    bad = {f"{prefix}/{p}": g for p, g in broken.items()}
    bad.update({p: ["import"] for p in seam(at_rev, first, prefix)})
    return sweep(log, after, mine), bad, len(log)


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


def report_dark_only(dark, log, donly):
    """Print the dark-only commits of each dark file that has one."""
    hit = {d: cs for d, cs in donly.items() if cs}
    print(f"  dark-only commits: {len(hit)} of {len(dark)} files")
    for d, cs in sorted(hit.items()):
        for i in cs:
            print(f"    {short(d)}: {log[i][1]} {log[i][2]} {log[i][3][:80]}")


def report(dark, log, res, donly):
    """Print the unit states, the files that hold a dark, both or never unit, and the dark-only commits."""
    st = Counter(r[0] for r in res.values())
    print("  units: " + ", ".join(f"{s} {st[s]}" for s in STATES))
    keep = open_units(res)
    print(f"  files: {len(keep)} of {len(dark)} hold a dark, both or never unit")
    for d, us in sorted(keep.items()):
        print(f"    {short(d)}: " + "; ".join(us))
    report_dark_only(dark, log, donly)


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
    ]


def self_test():
    """Rule 7: the move, git-parse and unit cases, in number order."""
    cases = sorted(move_cases() + git_cases() + unit_cases(), key=lambda c: int(c[0].split()[0]))
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


def add_sources(pin, specs, units, seqs, first, broken):
    """Put each unit's states in the PREFIX=DIR source clones, up to their import, in front of its states on main;
    add the clones' gaps to broken and return how many source commits were replayed."""
    pre = 0
    for spec in specs:
        prefix, repo = spec.split("=", 1)
        got, bad, n = before(pin, prefix.rstrip("/"), repo, units, first)
        broken.update(bad)
        pre += n
        for u, s in got.items():
            seqs[u] = s + seqs[u]
    return pre


def run(pin, verdict, specs, tsv):
    """Replay the units of the dark files whose verdict is in verdict, print their history and, if tsv is set,
    write it there; 1 if a replay has a gap."""
    files, dark, units, ns = select(pin, set(verdict))
    log, after, broken = replay(".", pin, set(dark).union(*ns.values()), files)
    seqs = sweep(log, after, units)
    pre = add_sources(pin, specs, units, seqs, first_texts(after), broken)
    print(f"unit history at {pin}: {len(dark)} {'/'.join(verdict)} dark files, {len(units)} changed units; "
          f"replayed {len(log)} first-parent commits on main and {pre} in sources, {len(broken)} paths with gaps")
    for p, g in broken.items():
        print(f"  GAP {p}: {', '.join(g)}")
    if broken:
        return 1
    res = results(seqs)
    report(dark, log, res, {d: dark_only(log, d, ns[d]) for d in dark})
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
    return run(oc.git("rev-parse", "--short=10", a.pin).decode().strip(), a.verdict, a.source, a.tsv)


if __name__ == "__main__":
    sys.exit(main())
