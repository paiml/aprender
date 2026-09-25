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

# tomllib is python 3.11+, and the clean-room guard runners run 3.10 without tomli: a
# module-level `import tomllib` died there with ModuleNotFoundError and took guard-cargo
# red (B2 #4315, job 107931281356; the same death as #3626). This script reads ONE key
# (a crate's `edition`), so it falls back to a reader for exactly that key.
try:
    import tomllib as _toml
except ImportError:
    try:
        import tomli as _toml
    except ImportError:
        _toml = None
if os.environ.get("INCLUDE_FMT_FORCE_FALLBACK") == "1":
    _toml = None

TOML_ERRORS = (ValueError,)  # tomllib/tomli.TOMLDecodeError subclass ValueError
SECTION = re.compile(r"^\s*\[\s*([A-Za-z0-9_.-]+)\s*\]\s*(?:#.*)?$")
EDITION = re.compile(r"""^\s*edition\s*(?:=\s*(?:"([^"]*)"|'([^']*)')|\.workspace\s*=\s*true|=\s*\{[^}]*workspace\s*=\s*true[^}]*\})\s*(?:#.*)?$""")
EDITION_KEY = re.compile(r"""^\s*(?:package\s*\.\s*|workspace\s*\.\s*package\s*\.\s*)?["']?edition["']?\s*[.=]""")
ML_DELIMS = ('"' * 3, "'" * 3)
ML_OPEN = re.compile(r"""^\s*[A-Za-z0-9_.-]+\s*=\s*("{3}|'{3})""")


class FallbackUnreadable(Exception):
    """The fallback reader met an edition it cannot read. NOT in TOML_ERRORS: edition_of would
    swallow it and walk up to another crate's edition, a silent wrong answer (#4315 quorum)."""


def load_toml(text):
    """The manifest as a dict: the real parser when there is one, else only what edition_of reads,
    [package] and [workspace.package] with their `edition` (a string, or {"workspace": True}).
    The fallback refuses (FallbackUnreadable) rather than guess at an edition line it cannot read."""
    if _toml is not None:
        return _toml.loads(text)
    doc, cur, in_ml = {}, None, None
    for n, line in enumerate(text.splitlines(), 1):
        if in_ml:  # inside a multi-line string: nothing here is a key
            if in_ml in line:
                # The first delimiter closes it. Anything after that could reopen it or hide an
                # escaped quote; the fallback refuses rather than track it.
                rest = line.split(in_ml, 1)[1]
                if any(d in rest for d in ML_DELIMS) or "\\" in line.split(in_ml, 1)[0][-1:]:
                    raise FallbackUnreadable("line %d: an ambiguous multi-line string end: %r" % (n, line.strip()))
                in_ml = None
            continue
        if line.lstrip().startswith("["):  # every header, [[bin]] included, leaves the section
            m = SECTION.match(line)
            name = m.group(1) if m else None
            if name == "package":
                cur = doc.setdefault("package", {})
            elif name == "workspace.package":
                cur = doc.setdefault("workspace", {}).setdefault("package", {})
            else:
                cur = None
            continue
        # A dotted `package.edition = ...` is readable anywhere; a bare `edition` only in a section we read.
        if EDITION_KEY.match(line) and (cur is not None or "." in line.split("=", 1)[0]):
            m = EDITION.match(line)
            if not m or cur is None:
                raise FallbackUnreadable("line %d: %r" % (n, line.strip()))
            ed = m.group(1) if m.group(1) is not None else m.group(2)
            cur["edition"] = ed if ed is not None else {"workspace": True}
        # A multi-line string opens only as `key = """` / `key = '''`. A triple quote anywhere else (a
        # comment, inside a one-line string) is refused rather than guessed at: guessing wrong skips the
        # real edition line with no error (#4315 quorum round 2).
        for delim in ML_DELIMS:
            if delim in line:
                m = ML_OPEN.match(line)
                if not m or m.group(1) != delim:
                    raise FallbackUnreadable("line %d: a %s outside `key = %s`: %r" % (n, delim, delim, line.strip()))
                if line.count(delim) % 2 == 1:
                    in_ml = delim
                break
    return doc


# (manifest, want [package].edition, want [workspace.package].edition); "RAISE" = the fallback must refuse
Q3, A3 = ML_DELIMS
SELF_TEST = [
    ('[package]\nedition = "2021"\n', "2021", None),
    ("[package]\nedition = '2018'\n", "2018", None),
    ('[package]\nedition = "2024"  # pinned\n', "2024", None),
    ('[package]\nedition.workspace = true\n', {"workspace": True}, None),
    ('[package]\nedition = { workspace = true }\n', {"workspace": True}, None),
    ('[workspace.package]\nedition = "2024"\n', None, "2024"),
    ('[package]\nname = "x"\n\n[[bin]]\nname = "b"\nedition = "2015"\n', None, None),
    ('[package]\nedition = "2021"\n[[bin]]\nedition = "2015"\n', "2021", None),
    ('[package]\nedition = "2021"\ndescription = ' + Q3 + '\nedition = "2018"\n' + Q3 + '\n', "2021", None),
    ('[package]\nedition = "2021"\nreadme = ' + Q3 + 'one line, closed' + Q3 + '\nname = "x"\n', "2021", None),
    ('[package]\n# a stray ' + Q3 + ' in a comment\nedition = "2021"\n', "RAISE", None),
    ('[package]\ndescription = ' + Q3 + '\nend' + Q3 + ' # ' + Q3 + '\nedition = "2021"\n', "RAISE", None),
    ('[package]\ndescription = ' + Q3 + '\nsaid \\' + Q3 + '\nedition = "2021"\n', "RAISE", None),
    ("[package]\ndescription = '" + Q3 + "'\nedition = \"2021\"\n", "RAISE", None),
    ("[package]\ndescription = " + A3 + "\nedition = '2018'\n" + A3 + "\n", None, None),
    ('[package.metadata]\nedition = "2015"\n[package]\nedition = "2021"\n', "2021", None),
    ('[dependencies]\nedition = "1"\n', None, None),
    ('package.edition = "2021"\n', "RAISE", None),
    ('[package]\nedition = 2021\n', "RAISE", None),
]


def _pick(doc):
    return (doc.get("package", {}).get("edition"),
            doc.get("workspace", {}).get("package", {}).get("edition"))


def self_test(root):
    """The fallback reader against the case table, then (when this python has tomllib/tomli)
    against the real parser on the table and on every tracked Cargo.toml under ROOT. Exit 0 pass, 1 fail."""
    global _toml
    real, fails = _toml, 0
    _toml = None
    for text, want_pkg, want_ws in SELF_TEST:
        try:
            got = _pick(load_toml(text))
        except FallbackUnreadable:
            got = ("RAISE", None)
        if got != (want_pkg, want_ws):
            print("FAIL  fallback %r: got %r, want %r" % (text, got, (want_pkg, want_ws)))
            fails += 1
    n = 0
    if real is not None:
        texts = [t for t, w, _ in SELF_TEST if w != "RAISE"]
        out = subprocess.run(["git", "-C", root, "ls-files", "-z", "--", "*Cargo.toml"],
                             capture_output=True, text=True)
        for rel in [p for p in out.stdout.split("\0") if p]:
            try:
                texts.append(open(os.path.join(root, rel)).read())
            except OSError:
                pass
        for text in texts:
            try:
                want = _pick(real.loads(text))
            except TOML_ERRORS:
                continue
            try:
                got = _pick(load_toml(text))
            except FallbackUnreadable as e:
                got = ("RAISE", str(e))
            n += 1
            if got != want:
                print("FAIL  parity %r...: fallback %r, real parser %r" % (text[:60], got, want))
                fails += 1
    _toml = real
    parity = "%d manifest(s) match the real parser" % n if real is not None else "parity skipped: no tomllib/tomli"
    print("include_fmt self-test: %d case(s), %s: %s" % (len(SELF_TEST), parity, "FAIL" if fails else "PASS"))
    return 1 if fails else 0


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
                pkg = load_toml(open(man).read()).get("package")
            except (OSError,) + TOML_ERRORS:
                pkg = None
            ed = None
            if pkg:
                ed = pkg.get("edition", "2015")
                if isinstance(ed, dict):
                    ws = load_toml(open(os.path.join(root, "Cargo.toml")).read())
                    ed = ws.get("workspace", {}).get("package", {}).get("edition", "2021")
            cache[man] = ed
            if ed:
                return ed
        if not d:
            return "2021"
        d = os.path.dirname(d)


def main(argv):
    if len(argv) == 3 and argv[1] == "--self-test":
        return self_test(os.path.realpath(argv[2]))
    if len(argv) != 2:
        print("usage: include_fmt.py ROOT | --self-test ROOT", file=sys.stderr)
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
    try:
        for t in targets:
            by_ed.setdefault(edition_of(root, t, cache), []).append(t)
    except FallbackUnreadable as e:
        print("include_fmt: no tomllib/tomli, and the fallback cannot read an edition (%s) - cannot check" % e,
              file=sys.stderr)
        return 2
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
