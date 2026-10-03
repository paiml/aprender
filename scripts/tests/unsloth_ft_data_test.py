#!/usr/bin/env python3
"""Self-test of unsloth_ft_data.py (the canonical APR_FT_DATA builder).

Builds from a throwaway git repo (no network, no cargo, no GPU) and checks: row
count and width, byte-for-byte determinism, the commit (not the working tree) is
what is read, only crates/**/*.rs is used, files are joined in path order with
a newline after each, and too little source refuses. Then every mutant below
must make at least one case fail.

  python3 scripts/tests/unsloth_ft_data_test.py     # exit 0 = all pass, all mutants killed
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BUILDER = os.path.join(HERE, "..", "bench", "unsloth_ft_data.py")
A = "a" * 4999 + "é"  # a multi-byte char: widths are counted in characters
B = "b" * 5000
STREAM = A + "\n" + B + "\n"


def git(repo, *args):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t", GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return subprocess.run(["git", "-C", repo] + list(args), env=env, capture_output=True,
                          text=True, check=True).stdout.strip()


def put(repo, rel, text):
    path = os.path.join(repo, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def make_repo(root):
    """Commit 1 has the canonical sources; commit 2 changes them; the tree is then dirtied."""
    repo = os.path.join(root, "repo")
    os.makedirs(repo)
    git(repo, "init", "-q")
    for rel, text in (("crates/b/b.rs", B), ("crates/a.rs", A), ("crates/c.md", "MD" * 3000),
                      ("src/y.rs", "Y" * 3000)):
        put(repo, rel, text)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "one")
    first = git(repo, "rev-parse", "HEAD")
    put(repo, "crates/a.rs", "z" * 5000)
    git(repo, "commit", "-q", "-am", "two")
    put(repo, "crates/b/b.rs", "w" * 5000)  # uncommitted
    return repo, first


def build(builder, repo, rev, out, rows=4, chars=1000):
    p = subprocess.run([sys.executable, builder, "--repo", repo, "--rev", rev, "--out", out,
                        "--rows", str(rows), "--chars", str(chars)], capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def read_rows(out):
    with open(out, encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def c_rows_and_width(b, repo, first, tmp):
    out = os.path.join(tmp, "d1.jsonl")
    rc, _, _ = build(b, repo, first, out)
    rows = read_rows(out) if rc == 0 else []
    return rc == 0 and len(rows) == 4 and all(len(r) == 1000 for r in rows)


def c_stream_order_and_newlines(b, repo, first, tmp):
    out = os.path.join(tmp, "d2.jsonl")
    rc, _, _ = build(b, repo, first, out, rows=10, chars=1000)
    return rc == 0 and "".join(read_rows(out)) == STREAM[:10000]


def c_only_crates_rs(b, repo, first, tmp):
    out = os.path.join(tmp, "d3.jsonl")
    rc, _, _ = build(b, repo, first, out, rows=10, chars=1000)
    text = "".join(read_rows(out)) if rc == 0 else "MDY"
    return "MD" not in text and "Y" not in text


def c_deterministic_and_sha(b, repo, first, tmp):
    o1, o2 = os.path.join(tmp, "e1.jsonl"), os.path.join(tmp, "e2.jsonl")
    r1, s1, _ = build(b, repo, first, o1)
    r2, _, _ = build(b, repo, first, o2)
    if r1 or r2:
        return False
    with open(o1, "rb") as f1, open(o2, "rb") as f2:
        body1, body2 = f1.read(), f2.read()
    meta = json.loads(s1)
    return (body1 == body2 and meta["sha256"] == hashlib.sha256(body1).hexdigest()
            and meta["rev"] == first and meta["rows"] == 4)


def c_reads_the_commit(b, repo, first, tmp):
    out = os.path.join(tmp, "d4.jsonl")
    rc, _, _ = build(b, repo, first, out, rows=1, chars=1000)
    return rc == 0 and read_rows(out)[0] == "a" * 1000


def c_head_is_commit_two(b, repo, first, tmp):
    out = os.path.join(tmp, "d5.jsonl")
    rc, _, _ = build(b, repo, "HEAD", out, rows=7, chars=1000)
    rows = read_rows(out) if rc == 0 else []
    return len(rows) == 7 and rows[0] == "z" * 1000 and rows[6] == "b" * 1000


def c_too_little_refuses(b, repo, first, tmp):
    out = os.path.join(tmp, "d6.jsonl")
    rc, _, err = build(b, repo, first, out, rows=11, chars=1000)
    return rc == 3 and "need 11" in err and not os.path.exists(out)


CASES = [c_rows_and_width, c_stream_order_and_newlines, c_only_crates_rs,
         c_deterministic_and_sha, c_reads_the_commit, c_head_is_commit_two, c_too_little_refuses]

MUTANTS = [
    ("D1 every file, not just .rs", 'if n.endswith(b".rs")', "if n"),
    ("D2 short data allowed", "if len(rows) < a.rows:", "if False:"),
    ("D3 rows one char short", "stream[i * chars:(i + 1) * chars]", "stream[i * chars:(i + 1) * chars - 1]"),
    ("D4 no newline between files", 't + "\\n" for t in texts', "t for t in texts"),
    ("D5 always HEAD", "commit = resolve(a.repo, a.rev)", 'commit = resolve(a.repo, "HEAD")'),
    ("D6 row cap dropped", "min(rows, len(stream) // chars)", "len(stream) // chars"),
    ("D7 reverse path order", "return sorted(n.decode() for n in names if",
     "return sorted((n.decode() for n in names if"),
]
# D7 needs the closing edit too; see mutate().
D7_TAIL = ('n.endswith(b".rs"))\n', 'n.endswith(b".rs")), reverse=True)\n')


def failures(builder, repo, first, tmp):
    return [c.__name__ for c in CASES if not c(builder, repo, first, tmp)]


def mutate(src, name, old, new):
    out = src.replace(old, new)
    if name.startswith("D7"):
        out = out.replace(*D7_TAIL)
    return out


def survivors(src, repo, first, tmp):
    left = []
    for name, old, new in MUTANTS:
        if src.count(old) != 1:
            left.append(name + " (edit does not apply)")
            continue
        path = os.path.join(tmp, "mutant.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(mutate(src, name, old, new))
        if not failures(path, repo, first, tmp):
            left.append(name)
    return left


def main():
    with open(BUILDER, encoding="utf-8") as f:
        src = f.read()
    with tempfile.TemporaryDirectory() as tmp:
        repo, first = make_repo(tmp)
        failed = failures(BUILDER, repo, first, tmp)
        left = survivors(src, repo, first, tmp)
    for name in failed:
        print("FAIL case", name)
    for name in left:
        print("SURVIVED", name)
    print("%d checks, %d failed" % (len(CASES) + len(MUTANTS), len(failed) + len(left)))
    return 1 if failed or left else 0


if __name__ == "__main__":
    sys.exit(main())
