#!/usr/bin/env python3
"""include_fmt.py ROOT: every include!d .rs file that rustfmt would change (#4151).

`cargo fmt --all -- --check` (CI's fmt gate) never formats a file pulled in with `include!()`:
rustfmt follows `mod` declarations, not `include!`. Measured: two 141/142-column lines in
apr-cli's golden_output.rs (include!d from qa.rs) passed it with rc 0, twice (#4140).

This enumerates the include!d .rs files FROM SOURCE: every git-tracked .rs file under ROOT, and every
literal `include!("x.rs")` in it, resolved against the including file's directory and kept when the
target exists and is tracked. It then runs the pinned rustfmt (`rustfmt --edition <the owning crate's
edition> --check`, cwd ROOT so rust-toolchain.toml applies) on them in batches, and prints one line
per include!d file rustfmt would change:
    MISFORMATTED <path>
followed by `SCANNED <n> include!d file(s), <m> misformatted`. rustfmt follows `mod` children, so a
diff it reports for a file OUTSIDE the include!d set is ignored here; cargo fmt owns those.
Exits 0 after a complete scan. Exits 2 when it could not answer: no rustfmt, zero include!d files
(a scan that finds nothing is broken, not clean), or a rustfmt failure no file can be blamed for.
"""
import os
import re
import subprocess
import sys
import tomllib

INCLUDE = re.compile(r'(?<![A-Za-z0-9_])include!\(\s*"([^"]+\.rs)"\s*\)')
DIFF_IN = re.compile(r"^Diff in (.+?):\d+:\s*$")
ERROR_AT = re.compile(r"^\s*--> (.+?):\d+:\d+")
BATCH = 150


def tracked_rs(root):
    out = subprocess.run(["git", "-C", root, "ls-files", "-z", "--", "*.rs"], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return [p for p in out.stdout.split("\0") if p]


def include_targets(root, files):
    tracked, found = set(files), set()
    for f in files:
        try:
            src = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for m in INCLUDE.finditer(src):
            t = os.path.normpath(os.path.join(os.path.dirname(f), m.group(1)))
            if t in tracked:
                found.add(t)
    return sorted(found)


def edition_of(root, rel, cache):
    """The edition of the crate owning `rel`: the nearest ancestor Cargo.toml with a [package]."""
    d = os.path.dirname(rel)
    while True:
        man = os.path.join(root, d, "Cargo.toml")
        if man in cache:
            if cache[man]:
                return cache[man]
        elif os.path.isfile(man):
            try:
                pkg = tomllib.loads(open(man).read()).get("package")
            except (OSError, tomllib.TOMLDecodeError):
                pkg = None
            ed = None
            if pkg:
                ed = pkg.get("edition", "2015")
                if isinstance(ed, dict):
                    ws = tomllib.loads(open(os.path.join(root, "Cargo.toml")).read())
                    ed = ws.get("workspace", {}).get("package", {}).get("edition", "2021")
            cache[man] = ed
            if ed:
                return ed
        if not d:
            return "2021"
        d = os.path.dirname(d)


def main(argv):
    if len(argv) != 2:
        print("usage: include_fmt.py ROOT", file=sys.stderr)
        return 2
    root = os.path.realpath(argv[1])
    files = tracked_rs(root)
    if files is None:
        print("include_fmt: cannot list tracked files in %s" % root, file=sys.stderr)
        return 2
    targets = include_targets(root, files)
    if not targets:
        print("include_fmt: no include!d .rs file found under %s - the scan is broken, not clean" % root, file=sys.stderr)
        return 2
    by_ed, cache = {}, {}
    for t in targets:
        by_ed.setdefault(edition_of(root, t, cache), []).append(t)
    tset, bad = set(targets), set()
    for ed, group in sorted(by_ed.items()):
        for i in range(0, len(group), BATCH):
            chunk = group[i:i + BATCH]
            try:
                r = subprocess.run(["rustfmt", "--edition", ed, "--check", "--"] + chunk,
                                   cwd=root, capture_output=True, text=True)
            except FileNotFoundError:
                print("include_fmt: rustfmt is not on PATH - cannot check (not a pass)", file=sys.stderr)
                return 2
            blamed = set()
            for ln in (r.stdout + "\n" + r.stderr).splitlines():
                m = DIFF_IN.match(ln) or ERROR_AT.match(ln)
                if m:
                    p = os.path.relpath(os.path.realpath(os.path.join(root, m.group(1))), root)
                    if p in tset:
                        blamed.add(p)
            if r.returncode != 0 and not blamed:
                # a failure no include!d file can be blamed for: re-check this chunk one file at a time
                for f in chunk:
                    one = subprocess.run(["rustfmt", "--edition", ed, "--check", "--", f], cwd=root,
                                         capture_output=True, text=True)
                    if one.returncode != 0 and ("Diff in" in one.stdout or "error" in one.stderr):
                        blamed.add(f)
                if not blamed:
                    print("include_fmt: rustfmt failed (rc %d) and no include!d file could be blamed:\n%s"
                          % (r.returncode, (r.stderr or r.stdout)[-800:]), file=sys.stderr)
                    return 2
            bad |= blamed
    for b in sorted(bad):
        print("MISFORMATTED %s" % b)
    print("SCANNED %d include!d file(s), %d misformatted" % (len(targets), len(bad)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
