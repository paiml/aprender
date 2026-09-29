#!/usr/bin/env python3
"""Byte-exact reimplementation of `git patch-id` (#4421).

`git patch-id --verbatim` needs git >= 2.40. lambda and intel run git 2.34.1
(measured 2026-09-25: rc 129, "unknown option `verbatim'"), so the pr-review
receipt binding cannot lean on the native command on every runner.

This mirrors get_one_patchid() / flush_one_hunk() from git v2.53
builtin/patch-id.c and diff.c line for line. It is not trusted on its own:
scripts/lib/pr_review_patch_id.sh --self-test compares it against the native
command wherever the native command exists (--stable on git 2.34, --verbatim on
git >= 2.40), and a disagreement is a defect.

usage: git_patch_id.py --stable|--verbatim|--unstable  < diff
"""
import hashlib
import re
import sys

HEX40 = re.compile(rb"[0-9a-fA-F]{40}")
DIGITS = re.compile(rb"[0-9]*")
ZERO = "0" * 40
# C isspace() in the "C" locale.
SPACE = b" \t\n\v\f\r"


def flush_one_hunk(result, ctx):
    """diff.c flush_one_hunk: add the hunk's sha1 into result, bytewise, with carry."""
    digest = ctx.digest()
    carry = 0
    for i in range(20):
        carry += result[i] + digest[i]
        result[i] = carry & 0xFF
        carry >>= 8
    return hashlib.sha1()


def scan_hunk_header(line):
    """Return (before, after), or None when the header does not parse."""
    q = 4
    n = len(DIGITS.match(line, q).group(0))
    if line[q + n:q + n + 1] == b",":
        q += n + 1
        before = int(DIGITS.match(line, q).group(0) or b"0")
        n = len(DIGITS.match(line, q).group(0))
    else:
        before = 1
    if n == 0 or line[q + n:q + n + 1] != b" " or line[q + n + 1:q + n + 2] != b"+":
        return None
    r = q + n + 2
    n = len(DIGITS.match(line, r).group(0))
    if line[r + n:r + n + 1] == b",":
        r += n + 1
        after = int(DIGITS.match(line, r).group(0) or b"0")
        n = len(DIGITS.match(line, r).group(0))
    else:
        after = 1
    if n == 0:
        return None
    return before, after


def parse_index_line(line):
    """`index <pre>..<post>[ mode]` -> (pre_oid, post_oid), or None when there is no `..`."""
    oid1_end = line.find(b"..")
    if oid1_end < 0:
        return None
    oid2_end = line.find(b" ", oid1_end)
    if oid2_end < 0:
        oid2_end = len(line) - 1
    return line[6:oid1_end][:40], line[oid1_end + 2:oid2_end][:40]


def is_binary_marker(line):
    return line.startswith(b"GIT binary patch") or line.startswith(b"Binary files")


def oid_candidate(line):
    """The text after a `commit ` / `From ` prefix, else the line itself."""
    if line.startswith(b"commit "):
        return line[7:]
    if line.startswith(b"From "):
        return line[5:]
    return line


def get_one_patchid(lines, pos, stable, verbatim):
    """Returns (patchlen, result_hex, next_oid_hex, new_pos)."""
    patchlen = 0
    before = after = -1
    diff_is_binary = False
    pre_oid = post_oid = b""
    ctx = hashlib.sha1()
    result = bytearray(20)
    next_oid = None

    while pos < len(lines):
        line = lines[pos]
        pos += 1
        if line.startswith(b"\\ ") and len(line) > 12 and oid_candidate(line) is line:
            if verbatim:
                ctx.update(line)
            continue
        p = oid_candidate(line)

        if HEX40.match(p):
            next_oid = p[:40].decode().lower()
            break

        if not patchlen and not line.startswith(b"diff "):
            continue

        if before == -1:
            if is_binary_marker(line):
                diff_is_binary = True
                before = 0
                ctx.update(pre_oid)
                ctx.update(post_oid)
                if stable:
                    ctx = flush_one_hunk(result, ctx)
                continue
            elif line.startswith(b"index "):
                ids = parse_index_line(line)
                if ids is not None:
                    pre_oid, post_oid = ids
                continue
            elif line.startswith(b"--- "):
                before = after = 1
            elif not line[:1].isalpha():
                break

        if diff_is_binary:
            if line.startswith(b"diff "):
                diff_is_binary = False
                before = -1
            continue

        if before == 0 and after == 0:
            if line.startswith(b"@@ -"):
                parsed = scan_hunk_header(line)
                if parsed is not None:
                    before, after = parsed
                continue
            if not line.startswith(b"diff "):
                break
            if stable:
                ctx = flush_one_hunk(result, ctx)
            before = after = -1

        c = line[:1]
        if c in (b"-", b" "):
            before -= 1
        if c in (b"+", b" "):
            after -= 1

        data = line if verbatim else bytes(b for b in line if b not in SPACE)
        patchlen += len(data)
        ctx.update(data)

    flush_one_hunk(result, ctx)
    return patchlen, result.hex(), next_oid or ZERO, pos


def main(argv):
    mode = argv[1] if len(argv) > 1 else "--unstable"
    if mode not in ("--stable", "--verbatim", "--unstable"):
        sys.stderr.write("usage: git_patch_id.py --stable|--verbatim|--unstable\n")
        return 2
    verbatim = mode == "--verbatim"
    stable = mode != "--unstable"
    lines = sys.stdin.buffer.read().splitlines(keepends=True)
    pos = 0
    oid = ZERO
    out = sys.stdout
    while True:
        patchlen, result, nxt, pos = get_one_patchid(lines, pos, stable, verbatim)
        if patchlen:
            out.write("%s %s\n" % (result, oid))
        oid = nxt
        if pos >= len(lines):
            break
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
