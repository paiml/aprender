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


class _State:
    """The scanner state get_one_patchid threads through one patch."""

    def __init__(self):
        self.before = self.after = -1
        self.diff_is_binary = False
        self.pre_oid = self.post_oid = b""
        self.ctx = hashlib.sha1()
        self.result = bytearray(20)


def _header_line(st, line, stable):
    """A line seen before a hunk header count is known. Returns "continue", "break" or None."""
    if is_binary_marker(line):
        st.diff_is_binary = True
        st.before = 0
        st.ctx.update(st.pre_oid)
        st.ctx.update(st.post_oid)
        if stable:
            st.ctx = flush_one_hunk(st.result, st.ctx)
        return "continue"
    if line.startswith(b"index "):
        ids = parse_index_line(line)
        if ids is not None:
            st.pre_oid, st.post_oid = ids
        return "continue"
    if line.startswith(b"--- "):
        st.before = st.after = 1
        return None
    return None if line[:1].isalpha() else "break"


def _between_hunks(st, line, stable):
    """A line seen when the previous hunk is fully counted. Returns "continue", "break" or None."""
    if line.startswith(b"@@ -"):
        parsed = scan_hunk_header(line)
        if parsed is not None:
            st.before, st.after = parsed
        return "continue"
    if not line.startswith(b"diff "):
        return "break"
    if stable:
        st.ctx = flush_one_hunk(st.result, st.ctx)
    st.before = st.after = -1
    return None


def get_one_patchid(lines, pos, stable, verbatim):
    """Returns (patchlen, result_hex, next_oid_hex, new_pos)."""
    patchlen = 0
    st = _State()
    next_oid = None

    while pos < len(lines):
        line = lines[pos]
        pos += 1
        if line.startswith(b"\\ ") and len(line) > 12:
            if verbatim:
                st.ctx.update(line)
            continue
        p = oid_candidate(line)

        if HEX40.match(p):
            next_oid = p[:40].decode().lower()
            break

        if not patchlen and not line.startswith(b"diff "):
            continue

        if st.before == -1:
            act = _header_line(st, line, stable)
            if act == "continue":
                continue
            if act == "break":
                break

        if st.diff_is_binary:
            if line.startswith(b"diff "):
                st.diff_is_binary = False
                st.before = -1
            continue

        if st.before == 0 and st.after == 0:
            act = _between_hunks(st, line, stable)
            if act == "continue":
                continue
            if act == "break":
                break

        c = line[:1]
        if c in (b"-", b" "):
            st.before -= 1
        if c in (b"+", b" "):
            st.after -= 1

        data = line if verbatim else bytes(b for b in line if b not in SPACE)
        patchlen += len(data)
        st.ctx.update(data)

    flush_one_hunk(st.result, st.ctx)
    return patchlen, st.result.hex(), next_oid or ZERO, pos


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
