#!/usr/bin/env python3
"""Build the canonical APR_FT_DATA for beat-unsloth-finetune-throughput-v1 (contract 1.2.0).

  unsloth_ft_data.py --rev <commit> --out data.jsonl [--rows 1000] [--chars 8192] [--repo .]

The rows come from this repository's own MIT-licensed Rust sources (crates/**/*.rs)
AT A PINNED COMMIT, read from git objects, not the working tree: the same --rev
gives the same bytes on any host, and no third-party licence is involved.

The tracked files are sorted by path and joined (each one followed by a newline)
into one stream, which is cut into rows of exactly --chars characters. 8192
characters of Rust is far more than 512 Qwen tokens (code runs about 3-4
characters per token), so every row fills seq_len and nothing is padded. The
verdict enforces this at measurement time: label_tokens_timed must equal
timed_steps * batch * grad_accum * (seq_len - 1) on both sides.

Prints one JSON line {rev, rows, chars, sha256}; exit 3 if the commit has too
little source for --rows rows. 1000 rows = 250 steps * batch 4 (50 warmup + 200
timed), so every row is trained exactly once.
"""
import argparse
import hashlib
import json
import subprocess
import sys


def git(repo, *args, data=None):
    return subprocess.run(["git", "-C", repo] + list(args), input=data, capture_output=True,
                          check=True).stdout


def resolve(repo, rev):
    return git(repo, "rev-parse", "--verify", rev + "^{commit}").decode().strip()


def sources(repo, commit):
    names = git(repo, "ls-tree", "-r", "--name-only", "-z", commit, "--", "crates").split(b"\0")
    return sorted(n.decode() for n in names if n.endswith(b".rs"))


def read_blobs(repo, commit, paths):
    """Contents of commit:path for each path, in order, via one cat-file --batch."""
    req = "".join("%s:%s\n" % (commit, p) for p in paths).encode()
    out, pos, blobs = git(repo, "cat-file", "--batch", data=req), 0, []
    for _ in paths:
        header_end = out.index(b"\n", pos)
        size = int(out[pos:header_end].split()[2])
        blobs.append(out[header_end + 1:header_end + 1 + size].decode("utf-8", "replace"))
        pos = header_end + 1 + size + 1
    return blobs


def rows_from(texts, rows, chars):
    stream = "".join(t + "\n" for t in texts)
    return [stream[i * chars:(i + 1) * chars] for i in range(min(rows, len(stream) // chars))]


def encode(rows):
    return "".join(json.dumps({"text": r}, sort_keys=True) + "\n" for r in rows).encode("utf-8")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--chars", type=int, default=8192)
    ap.add_argument("--repo", default=".")
    a = ap.parse_args(argv)
    commit = resolve(a.repo, a.rev)
    paths = sources(a.repo, commit)
    rows = rows_from(read_blobs(a.repo, commit, paths), a.rows, a.chars)
    if len(rows) < a.rows:
        print("only %d full rows of %d chars at %s, need %d" % (len(rows), a.chars, commit, a.rows),
              file=sys.stderr)
        return 3
    body = encode(rows)
    with open(a.out, "wb") as f:
        f.write(body)
    print(json.dumps(dict(rev=commit, rows=len(rows), chars=a.chars,
                          sha256=hashlib.sha256(body).hexdigest())))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
