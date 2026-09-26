#!/usr/bin/env python3
"""carry_forward_gate.py -- an rc is not cut while a fix from the previous release line
is missing from it, unless that fix is listed as intentionally dropped.

WHY THIS EXISTS
---------------
#4273 (split decode attention, ~20x Qwen3.5 decode) shipped in v0.69.5-rc.1 and never
reached main or car/0.70.0. Nothing compared the two lines. The 0.69.x -> car audit on
2026-09-26 then found 95 code commits of the 0.69.x tags whose content was not on
car, including 9 perf and 18 correctness commits.

WHAT "CARRIED" MEANS
--------------------
For every non-merge commit in merge-base(prev, cand)..prev:
  1. carried (patch-id): a commit in merge-base..cand has the same `git patch-id --stable`.
  2. carried (content): release lines come back through SQUASH merges, which defeat
     patch-id. So the commit's added lines (stripped, >= MIN_LEN chars) are looked up in
     the same file at cand. At least CONTENT_MIN of them must be present. A commit that
     only deletes is judged the other way: its removed lines must be gone from cand.
  3. dropped: a row in the drops file names its sha (prefix >= 7) or matches its subject.
  4. otherwise MISSING, and the gate refuses.
A commit with no judgeable line (only short lines, binaries, pure renames) is `empty`. It
counts as carried, and it is listed so a reader can see what was not judged.

Known limit: a later commit on cand that rewrites the same lines makes a real carry read
as MISSING. The drops file is the escape hatch; its reason column says why. It never
fails open silently.

  carry_forward_gate.py --cand <ref> [--prev <ref>] [--drops <tsv>] [--no-fetch]
      --prev defaults to the newest tag of the previous release line: the highest
      vX.Y.Z[-rc.N] whose (X, Y) is below the version cand's workspace declares.
  carry_forward_gate.py --self-test

EXIT CODES: 0 every commit carried or dropped; 1 MISSING commits (listed); 2 ENV/usage
(history unreadable, no merge-base even after deepening, bad drops file). Never 0 on 2.
"""
import os
import re
import subprocess
import sys
import tempfile

MIN_LEN = 12
CONTENT_MIN = 0.8
VERSION_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(?:-rc\.(\d+))?$")


class EnvError(Exception):
    pass


def git(*args, check=True, cwd=None):
    r = subprocess.run(["git", *args], capture_output=True, text=True, errors="replace", cwd=cwd)
    if check and r.returncode != 0:
        raise EnvError("git %s: %s" % (" ".join(args), r.stderr.strip()[:300]))
    return r.stdout


def tag_key(name):
    m = VERSION_TAG.match(name)
    if not m:
        return None
    x, y, z, rc = m.groups()
    # a final sorts above its own rcs
    return (int(x), int(y), int(z), int(rc) if rc is not None else 1 << 30)


def pick_prev_tag(tags, cand_version):
    """Highest version tag on a lower (X, Y) line than cand_version 'X.Y.Z'."""
    cx, cy = (int(p) for p in cand_version.split(".")[:2])
    best = None
    for t in tags:
        k = tag_key(t)
        if k and (k[0], k[1]) < (cx, cy) and (best is None or k > tag_key(best)):
            best = t
    return best


def cand_version(cand, cwd=None):
    toml = git("show", "%s:Cargo.toml" % cand, cwd=cwd)
    m = re.search(r'^\[workspace\.package\][^\[]*?^version\s*=\s*"(\d+\.\d+\.\d+)', toml, re.M | re.S)
    if not m:
        raise EnvError("no [workspace.package] version in %s:Cargo.toml" % cand)
    return m.group(1)


def load_drops(path):
    """Rows: `sha<TAB><sha prefix>|subject<TAB><regex>` then <TAB><reason>. # comments."""
    shas, subjects = {}, []
    if not path:
        return shas, subjects
    with open(path) as f:
        for n, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 3 or not cols[2].strip():
                raise EnvError("%s:%d: want <sha|subject> TAB <key> TAB <reason>" % (path, n))
            kind, key, reason = cols[0], cols[1], cols[2]
            if kind == "sha":
                if not re.fullmatch(r"[0-9a-f]{7,40}", key):
                    raise EnvError("%s:%d: sha key %r is not 7-40 hex" % (path, n, key))
                shas[key] = reason
            elif kind == "subject":
                try:
                    subjects.append((re.compile(key), reason))
                except re.error as e:
                    raise EnvError("%s:%d: bad regex: %s" % (path, n, e))
            else:
                raise EnvError("%s:%d: kind %r is not sha|subject" % (path, n, kind))
    return shas, subjects


def patch_ids(rng, cwd=None):
    log = subprocess.run(["git", "log", "--no-merges", "-p", "--format=commit %H", rng],
                         capture_output=True, cwd=cwd)
    if log.returncode != 0:
        raise EnvError("git log %s failed" % rng)
    pid = subprocess.run(["git", "patch-id", "--stable"], input=log.stdout, capture_output=True, cwd=cwd)
    out = {}
    for line in pid.stdout.decode(errors="replace").splitlines():
        p, c = line.split()
        out[c] = p
    return out


class Tree:
    """Files of one ref, read through ONE `git cat-file --batch` (a git show per file
    made a 141k-line commit take 22 s)."""

    def __init__(self, ref, cwd=None):
        self.ref, self.cache = ref, {}
        self.proc = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, cwd=cwd)

    def lines(self, path):
        if path not in self.cache:
            self.proc.stdin.write(("%s:%s\n" % (self.ref, path)).encode())
            self.proc.stdin.flush()
            head = self.proc.stdout.readline().decode(errors="replace").split()
            if len(head) == 3 and head[1] == "blob":
                body = self.proc.stdout.read(int(head[2]) + 1)[:-1]  # + the trailing LF
                self.cache[path] = {l.strip() for l in body.decode(errors="replace").splitlines()}
            else:
                # "<name> missing": the file is absent at ref; a tree/submodule has no lines
                if len(head) == 3:
                    self.proc.stdout.read(int(head[2]) + 1)
                self.cache[path] = set()
        return self.cache[path]

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def content_score(sha, tree, cwd=None):
    """(present fraction, judged line count). Added lines must be present at cand; a
    deletion-only commit is judged by its removed lines being absent."""
    diff = git("show", "--format=", "-U0", "--no-renames", sha, cwd=cwd)
    add_tot = add_hit = del_tot = del_hit = 0
    new = old = None
    for l in diff.splitlines():
        if l.startswith("--- "):
            old = l[6:] if l.startswith("--- a/") else None
        elif l.startswith("+++ "):
            new = l[6:] if l.startswith("+++ b/") else None
        elif l.startswith("+") and new:
            t = l[1:].strip()
            if len(t) >= MIN_LEN:
                add_tot += 1
                add_hit += t in tree.lines(new)
        elif l.startswith("-") and old:
            t = l[1:].strip()
            if len(t) >= MIN_LEN:
                del_tot += 1
                del_hit += t not in tree.lines(old)
    if add_tot:
        return add_hit / add_tot, add_tot
    if del_tot:
        return del_hit / del_tot, del_tot
    return 1.0, 0


def shallow_cut(base, refs, cwd=None):
    """True when a shallow boundary may be hiding ancestry of the range.

    A boundary on base's own ancestry turns older commits, still reachable from prev by
    a second path, into range members: a 09-01 boundary once made the 63 commits of
    a5efa1443..v0.69.5-rc.1 read as 300. Such a commit is older than the boundary
    that hid it. So any range commit whose committer time is not newer than every
    reachable boundary counts as a cut. The fix is to deepen, never to judge."""
    path = git("rev-parse", "--git-path", "shallow", cwd=cwd).strip()
    if cwd and not os.path.isabs(path):
        path = os.path.join(cwd, path)
    try:
        with open(path) as f:
            bounds = [l.strip() for l in f if l.strip()]
    except FileNotFoundError:
        return False
    tips = [base, *refs]
    reach = [b for b in bounds if any(
        subprocess.run(["git", "merge-base", "--is-ancestor", b, t], cwd=cwd).returncode == 0 for t in tips)]
    if not reach:
        return False
    newest = max(int(git("log", "-1", "--format=%ct", b, cwd=cwd)) for b in reach)
    for ref in refs:
        times = git("log", "--format=%ct", "%s..%s" % (base, ref), cwd=cwd).split()
        if any(int(t) <= newest for t in times):
            return True
    return False


def ensure_history(prev, cand, fetch, cwd=None):
    def mb():
        r = subprocess.run(["git", "merge-base", prev, cand], capture_output=True, text=True, cwd=cwd)
        base = r.stdout.strip() if r.returncode == 0 else ""
        return base if base and not shallow_cut(base, (prev, cand), cwd=cwd) else ""
    base = mb()
    if base or not fetch:
        return base
    # rc-cut.yml checks out at depth 1: deepen until the lines meet above every boundary
    for since in ("45 days ago", "180 days ago"):
        git("fetch", "--quiet", "--no-tags", "--shallow-since=%s" % since, "origin",
            "+refs/tags/%s:refs/tags/%s" % (prev, prev), cand, check=False, cwd=cwd)
        base = mb()
        if base:
            return base
    git("fetch", "--quiet", "--no-tags", "--unshallow", "origin", check=False, cwd=cwd)
    return mb()


def judge(prev, cand, drops, fetch=True, cwd=None):
    """-> (rows, base); rows are (verdict, sha, subject, detail)."""
    base = ensure_history(prev, cand, fetch, cwd=cwd)
    if not base:
        raise EnvError("no merge-base of %s and %s clear of a shallow boundary, even after deepening" % (prev, cand))
    shas, subjects = drops
    cand_pids = set(patch_ids("%s..%s" % (base, cand), cwd=cwd).values())
    prev_pids = patch_ids("%s..%s" % (base, prev), cwd=cwd)
    tree = Tree(cand, cwd=cwd)
    rows = []
    for line in git("log", "--no-merges", "--reverse", "--format=%H%x09%s", "%s..%s" % (base, prev), cwd=cwd).splitlines():
        sha, subj = line.split("\t", 1)
        if prev_pids.get(sha) in cand_pids:
            rows.append(("carried", sha, subj, "patch-id"))
            continue
        frac, n = content_score(sha, tree, cwd=cwd)
        if n == 0:
            rows.append(("empty", sha, subj, "no judgeable line"))
        elif frac >= CONTENT_MIN:
            rows.append(("carried", sha, subj, "content %.2f of %d" % (frac, n)))
        else:
            why = next((r for k, r in shas.items() if sha.startswith(k)), None)
            if why is None:
                why = next((r for rx, r in subjects if rx.search(subj)), None)
            if why is not None:
                rows.append(("dropped", sha, subj, why))
            else:
                rows.append(("MISSING", sha, subj, "content %.2f of %d" % (frac, n)))
    tree.close()
    return rows, base


def report(rows, prev, cand, base):
    counts = {}
    for v, *_ in rows:
        counts[v] = counts.get(v, 0) + 1
    print("carry-forward %s -> %s (base %s): %s" % (prev, cand, base[:9],
          ", ".join("%d %s" % (counts[k], k) for k in sorted(counts))))
    for v, sha, subj, detail in rows:
        if v in ("MISSING", "dropped", "empty"):
            print("  %-7s %s  %s  [%s]" % (v, sha[:9], subj[:100], detail[:80]))
    return 1 if counts.get("MISSING") else 0


def self_test():
    fail = 0

    def check(label, got, want):
        nonlocal fail
        ok = got == want
        fail |= not ok
        print("  %s %s%s" % ("ok  " if ok else "FAIL", label, "" if ok else "\n       want %r\n       got  %r" % (want, got)))

    print("carry_forward_gate self-test")
    # pick_prev_tag: a final beats its rcs; rc numbers compare numerically; same line excluded
    tags = ["v0.69.3", "v0.69.5-rc.1", "v0.69.5-rc.10", "v0.69.5-rc.9", "v0.70.0-rc.1", "v0.68.9", "v0.69.x", "nightly"]
    check("prev tag = newest of the lower line, rc.10 > rc.9", pick_prev_tag(tags, "0.70.0"), "v0.69.5-rc.10")
    check("a final outranks its own rcs", pick_prev_tag(tags + ["v0.69.5"], "0.70.0"), "v0.69.5")
    check("the candidate's own line never counts", pick_prev_tag(["v0.70.0-rc.3"], "0.70.0"), None)
    check("a lower major.minor across a major", pick_prev_tag(["v0.99.1", "v1.0.0"], "1.1.0"), "v1.0.0")

    with tempfile.TemporaryDirectory() as d:
        def g(*a):
            return git(*a, cwd=d)

        def commit(path, text, msg):
            p = os.path.join(d, path)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(text)
            g("add", "-A")
            g("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg)
            return g("rev-parse", "HEAD").strip()

        g("init", "-q", "-b", "main")
        g("config", "user.name", "t")
        g("config", "user.email", "t@t")
        g("config", "commit.gpgsign", "false")
        g("config", "core.hooksPath", os.devnull)  # a global pre-commit hook must not judge the fixture
        commit("src/a.rs", "fn keep_this_original_line() {}\n", "base")
        base = g("rev-parse", "HEAD").strip()
        g("checkout", "-q", "-b", "rel")
        picked = commit("src/b.rs", "fn cherry_picked_fix_line() {}\n", "fix: picked")
        squashed = commit("src/c.rs", "fn squashed_fix_line_one() {}\nfn squashed_fix_line_two() {}\n", "fix: squashed")
        lost = commit("src/d.rs", "fn the_lost_decode_fix() {}\n", "perf: lost (#4273)")
        dropped = commit("Cargo.toml", 'version_line_bump = "0.69.5"\n', "release: bump to 0.69.5")
        by_sha = commit("src/e.rs", "fn superseded_on_car_by_rewrite() {}\n", "fix: superseded")
        deleted = commit("src/a.rs", "", "refactor: delete the original line")
        short = commit("src/f.rs", "x\n", "tiny")
        g("tag", "v0.69.5-rc.1")
        g("checkout", "-q", "main")
        g("cherry-pick", "-x", picked)  # same patch-id, new sha
        with open(os.path.join(d, "src/c.rs"), "w") as f:
            f.write("fn squashed_fix_line_one() {}\nfn squashed_fix_line_two() {}\n")
        with open(os.path.join(d, "src/z.rs"), "w") as f:
            f.write("fn unrelated_car_work_line() {}\n")
        g("add", "-A")
        g("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "fold: squash of the rel fixes + car work")
        dfile = os.path.join(d, "drops.tsv")
        with open(dfile, "w") as f:
            f.write("# kind\tkey\treason\nsubject\t^release: bump\tversion bumps are per line\nsha\t%s\tsuperseded by car's rewrite, ruled 2026-09-26\n" % by_sha[:9])
        rows, mb = judge("v0.69.5-rc.1", "main", load_drops(dfile), fetch=False, cwd=d)
        got = {sha: v for v, sha, _, _ in rows}
        check("merge-base found", mb, base)
        check("cherry-pick is carried by patch-id", (got[picked], [r[3] for r in rows if r[1] == picked][0]), ("carried", "patch-id"))
        check("squash-merged fix is carried by content", got[squashed], "carried")
        check("a fix absent from cand is MISSING", got[lost], "MISSING")
        check("subject drop row drops the bump", got[dropped], "dropped")
        check("sha-prefix drop row drops it", got[by_sha], "dropped")
        check("deletion-only commit NOT applied on cand is MISSING", got[deleted], "MISSING")
        check("short-line commit is 'empty', not silently carried", got[short], "empty")
        check("the gate refuses while anything is MISSING", report(rows, "v0.69.5-rc.1", "main", mb), 1)
        # mutant: drop every drops row -> the two dropped commits turn MISSING
        rows2, _ = judge("v0.69.5-rc.1", "main", ({}, []), fetch=False, cwd=d)
        check("without drops, both dropped rows are MISSING", sorted(v for v, *_ in rows2 if v != "carried" and v != "empty"), ["MISSING"] * 4)
        # carry the lost fix + the deletion -> with drops the gate passes
        g("cherry-pick", lost)
        with open(os.path.join(d, "src/a.rs"), "w") as f:
            f.write("")
        g("add", "-A")
        g("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "carry the deletion")
        rows3, mb3 = judge("v0.69.5-rc.1", "main", load_drops(dfile), fetch=False, cwd=d)
        check("all carried or dropped -> exit 0", report(rows3, "v0.69.5-rc.1", "main", mb3), 0)
        # The real shape: base's own ancestry is cut at Y, and X (older than Y) is still
        # reachable from prev through a side branch, so X would read as a range member.
        sh = os.path.join(d, "shallow")
        os.makedirs(sh)

        def h(*a, date=None):
            env = dict(os.environ, GIT_COMMITTER_DATE=date, GIT_AUTHOR_DATE=date) if date else None
            r = subprocess.run(["git", *a], capture_output=True, text=True, cwd=sh, env=env)
            if r.returncode:
                raise RuntimeError(r.stderr)
            return r.stdout.strip()

        def hc(path, text, msg, date):
            with open(os.path.join(sh, path), "w") as f:
                f.write(text)
            h("add", "-A")
            h("commit", "-q", "-m", msg, date=date)
            return h("rev-parse", "HEAD")

        h("init", "-q", "-b", "main")
        for k, v in (("user.name", "t"), ("user.email", "t@t"), ("commit.gpgsign", "false"), ("core.hooksPath", os.devnull)):
            h("config", k, v)
        x = hc("x.rs", "fn old_main_history_line() {}\n", "X old main work", "2026-09-01T00:00:00Z")
        y = hc("y.rs", "fn boundary_commit_line() {}\n", "Y", "2026-09-02T00:00:00Z")
        hc("b.rs", "fn base_commit_line_here() {}\n", "B base", "2026-09-03T00:00:00Z")
        h("checkout", "-q", "-b", "side", x)
        hc("s.rs", "fn side_branch_fix_line() {}\n", "S side", "2026-09-04T00:00:00Z")
        h("checkout", "-q", "-b", "rel", "main")
        h("merge", "-q", "--no-ff", "-m", "merge side", "side", date="2026-09-05T00:00:00Z")
        h("checkout", "-q", "main")
        hc("s.rs", "fn side_branch_fix_line() {}\n", "C carries S", "2026-09-06T00:00:00Z")
        full, _ = judge("rel", "main", ({}, []), fetch=False, cwd=sh)
        check("full history: range is S only, carried", [(v, r[:4]) for v, _, r, _ in full], [("carried", "S si")])
        with open(os.path.join(sh, ".git", "shallow"), "w") as f:
            f.write(y + "\n")
        try:
            judge("rel", "main", ({}, []), fetch=False, cwd=sh)
            check("a boundary on base's ancestry is ENV, never judged", "judged", "EnvError")
        except EnvError:
            check("a boundary on base's ancestry is ENV, never judged", "EnvError", "EnvError")
        with open(dfile, "w") as f:
            f.write("sha\tabc\tno reason col\n")
        try:
            load_drops(dfile)
            check("malformed drops row is ENV, never a pass", "accepted", "EnvError")
        except EnvError:
            check("malformed drops row is ENV, never a pass", "EnvError", "EnvError")
        try:
            judge("v0.69.5-rc.1", "no-such-ref", ({}, []), fetch=False, cwd=d)
            check("unreadable cand is ENV", "judged", "EnvError")
        except EnvError:
            check("unreadable cand is ENV", "EnvError", "EnvError")
    return 1 if fail else 0


def main(argv):
    args, i = {}, 0
    while i < len(argv):
        a = argv[i]
        if a == "--self-test":
            return self_test()
        if a == "--no-fetch":
            args["no_fetch"] = True
            i += 1
        elif a in ("--cand", "--prev", "--drops") and i + 1 < len(argv):
            args[a[2:]] = argv[i + 1]
            i += 2
        elif a in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            print("carry_forward_gate: bad argument %r" % a, file=sys.stderr)
            return 2
    if "cand" not in args:
        print("carry_forward_gate: --cand <ref> is required", file=sys.stderr)
        return 2
    try:
        prev = args.get("prev")
        if not args.get("no_fetch"):
            # rc-cut.yml checks out the default branch at depth 1: bring cand in by name
            # or sha. Tags are LISTED remotely; fetching them would pull their history.
            git("fetch", "--quiet", "--no-tags", "--depth=1", "origin", args["cand"], check=False)
        if not prev:
            if args.get("no_fetch"):
                tags = git("tag", "-l", "v*").split()
            else:
                tags = [l.split("refs/tags/", 1)[1] for l in git("ls-remote", "--tags", "--refs", "origin", "v*").splitlines()]
            prev = pick_prev_tag(tags, cand_version(args["cand"]))
            if not prev:
                raise EnvError("no tag of a previous release line below %s" % cand_version(args["cand"]))
        rows, base = judge(prev, args["cand"], load_drops(args.get("drops")), fetch=not args.get("no_fetch"))
        return report(rows, prev, args["cand"], base)
    except EnvError as e:
        print("carry_forward_gate: ENV: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
