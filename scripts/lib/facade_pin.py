#!/usr/bin/env python3
"""Read or rewrite a facade's `upstream` version pin, in every spelling cargo accepts.

aprender#2628. The shell used a line-anchored regex that matched only the INLINE
form:

    upstream = { path = "...", version = "0.63.0", package = "aprender-contracts" }

Cargo treats the multi-line table as IDENTICAL:

    [dependencies.upstream]
    path = "..."
    version = "0.63.0"
    package = "aprender-contracts"

A manifest written that way was invisible to the enumeration, so `set_facade_pins`
never rewrote it AND `--check` never inspected it -- and `--check` then reported
rc=0 "all consistent" with a stale pin. The writer and its own validator shared
one blind spot.

aprender#2635 -- WHY THERE IS A HAND PARSER HERE, AND WHY THAT IS THE NARROW CHOICE.

The first fix read the manifest with `tomllib`. `tomllib` entered the stdlib in
3.11. The dev box where it was verified runs 3.13; the CI runner host runs
3.10.12, so on the runner every invocation died with ModuleNotFoundError, this
module printed NOTHING, `facade_edges()` produced NO ordering edges, and R3 --
the rule that stops a facade publishing before its upstream -- had nothing left
to check. The case table caught it; the live scan would have gone quietly inert,
which is the exact fail-open this file exists to prevent.

`cargo metadata` remains rejected for the reason recorded in the first fix: it
needs a RESOLVABLE workspace, and the --self-test fixtures deliberately are not
one, so it makes the guard unable to run against its own case table.

A tomllib-then-tomli-then-hand-parser chain was rejected too. `tomli` is not
guaranteed on the runner, and more importantly a chain means the branch taken
depends on the host: the 3.13 dev box would exercise one parser and the 3.10
runner another, so "verified here" would not transfer to "works there" -- which
is precisely the defect being fixed, one level down. ONE parser runs everywhere.

The cost is that this parser is not a general TOML implementation, and it is not
trying to be. It resolves ONE construct: a dependency named or renamed
`upstream`, and its `version`, under the three top-level dependency tables, in
the spellings cargo accepts. Everything else in the file is skipped, not
interpreted. Two things keep that honest:

  * `--self-test` drives a must-match / must-not-match case table (below).
  * where `tomllib` IS importable -- this dev box, the 3.11+ hosts -- the
    self-test additionally runs it as a DIFFERENTIAL ORACLE over every real
    manifest in the tree and requires the two to agree. tomllib is thus still
    the authority on TOML, but only at TEST time; no runtime path imports it,
    so no host runs code another host did not.

Writing stays a targeted text edit so comments and formatting survive --
round-tripping through a TOML writer would reformat the manifest.

  read    <manifest>            -> prints the version, or nothing if no upstream
                                   exit 0; exit 3 if the file could not be read
  package <manifest>            -> prints the upstream's real package name
                                   (the `package = ` rename target), or nothing
  write   <manifest> <version>  -> exit 0 if rewritten, 1 if no pin was found
  --self-test                   -> case table + differential oracle
"""
import os
import re
import sys

# The three dependency tables cargo resolves from the registry. Target-specific
# tables (`[target.'cfg(unix)'.dependencies]`) are deliberately NOT included:
# the tomllib version this replaces did not read them either, and a facade does
# not pin its upstream per-target. `_split_key` keeps them out by construction
# rather than by luck -- a target table's key path has four components, not two.
DEP_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")

# A bare or quoted TOML key, optionally dotted.
_KEY = r"(?:\"[^\"]*\"|'[^']*'|[A-Za-z0-9_-]+)"
_KV = re.compile(r"^\s*(" + _KEY + r"(?:\s*\.\s*" + _KEY + r")*)\s*=\s*(.*)$")

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"'}


def _consume_in_string(line, i, quote, out):
    """Copy one character from inside a string. -> (next_i, quote_or_None)."""
    ch = line[i]
    # Escapes are only special in a basic (double-quoted) string.
    if quote == '"' and ch == "\\" and i + 1 < len(line):
        out.append(ch)
        out.append(line[i + 1])
        return i + 2, quote
    out.append(ch)
    return i + 1, (None if ch == quote else quote)


def _strip_comment(line):
    """Drop a trailing `#` comment, but not a `#` inside a string."""
    out = []
    quote = None
    i = 0
    n = len(line)
    while i < n:
        if quote is not None:
            i, quote = _consume_in_string(line, i, quote, out)
            continue
        ch = line[i]
        if ch == "#":
            break
        if ch in "\"'":
            quote = ch
        out.append(ch)
        i += 1
    return "".join(out)


def _split_key(text):
    """Split a dotted TOML key into components, honouring quoting.

    `target."cfg(unix)".dependencies` -> ['target', 'cfg(unix)', 'dependencies'],
    so a target table can never be mistaken for a top-level dependency table.
    """
    parts = []
    cur = []
    quote = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = None
            else:
                cur.append(ch)
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            i += 1
            continue
        if ch == ".":
            parts.append("".join(cur).strip())
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    parts.append("".join(cur).strip())
    return parts


def _unquote(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        inner = value[1:-1]
        if value[0] == '"':
            inner = re.sub(
                r"\\(.)", lambda mo: _ESCAPES.get(mo.group(1), mo.group(1)), inner
            )
        return inner
    return value


# One `key = value` pair inside an inline table.
_INLINE_PAIR = re.compile(
    r"(?:^|,)\s*(" + _KEY + r")\s*=\s*"
    r"(\"(?:[^\"\\]|\\.)*\"|'[^']*'|\[[^\]]*\]|[^,}]+)"
)


def _parse_value(text):
    """A dependency's value: an inline table -> dict, a bare string -> str.

    TOML 1.0 forbids a newline inside an inline table, so `{ ... }` is always
    complete on the line it opens on. That is a guarantee of the format, not an
    assumption about how these manifests happen to be written.
    """
    text = text.strip()
    if text.startswith("{"):
        body = text[1 : text.rindex("}")] if "}" in text else text[1:]
        table = {}
        for mo in _INLINE_PAIR.finditer(body):
            table[_unquote(mo.group(1))] = _unquote(mo.group(2))
        return table
    if text[:1] in "\"'":
        return _unquote(text)
    return None


def _header_path(line):
    """The table path a header line opens, or None if the line is not a header."""
    if line.startswith("[["):
        # An array-of-tables header. Never a dependency table we read; return a
        # path that cannot match so the keys that follow are ignored.
        return ["\x00array-of-tables"]
    if line.startswith("[") and line.endswith("]"):
        return _split_key(line[1:-1])
    return None


def _dep_slot(full):
    """-> (dep_table, key) if `full` addresses the upstream dependency, else None.

    key is None for `[dependencies] upstream = ...` (the whole value) and
    'version'/'package' for `[dependencies.upstream] version = ...`.
    """
    if len(full) < 2 or full[0] not in DEP_TABLES or full[1] != "upstream":
        return None
    if len(full) == 2:
        return (full[0], None)
    if len(full) == 3 and full[2] in ("version", "package"):
        return (full[0], full[2])
    return None


def _record(found, full, raw):
    slot = _dep_slot(full)
    if slot is None:
        return
    dep_table, key = slot
    if key is None:
        found.setdefault(dep_table, _parse_value(raw))
        return
    current = found.setdefault(dep_table, {})
    if isinstance(current, dict):
        current[key] = _unquote(raw)


def parse_upstream(path):
    """-> {dep_table: value} for the `upstream` dependency, value str or dict."""
    found = {}
    table = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = _strip_comment(raw).strip()
            if not line:
                continue
            head = _header_path(line)
            if head is not None:
                table = head
                continue
            mo = _KV.match(line)
            if mo:
                _record(found, table + _split_key(mo.group(1)), mo.group(2))
    return found


def read_pin(path):
    deps = parse_upstream(path)
    for table in DEP_TABLES:
        dep = deps.get(table)
        if isinstance(dep, dict) and dep.get("version"):
            return dep["version"]
        if isinstance(dep, str):
            return dep
    return None


def read_package(path):
    """The real crate the `upstream` rename points at, for ordering-graph edges."""
    deps = parse_upstream(path)
    for table in DEP_TABLES:
        dep = deps.get(table)
        if isinstance(dep, dict):
            return dep.get("package") or "upstream"
    return None


def write_pin(path, new):
    src = open(path).read()

    # 1. inline table on one line
    out, n = re.subn(
        r'(?m)^(upstream\s*=\s*\{[^}\n]*?version\s*=\s*")[^"]*(")',
        lambda mo: mo.group(1) + new + mo.group(2),
        src,
    )
    if n:
        open(path, "w").write(out)
        return n

    # 2. the version key inside a [...dependencies.upstream] table, scoped to
    #    that table so no other section's `version` is touched.
    lines = src.splitlines(keepends=True)
    inside = False
    done = 0
    for i, ln in enumerate(lines):
        head = re.match(r"\s*\[([^\]]+)\]\s*$", ln)
        if head:
            name = head.group(1).strip()
            inside = name.split(".")[-1] == "upstream" and "dependencies" in name
            continue
        if inside:
            mo = re.match(r'(\s*version\s*=\s*")[^"]*(".*?)(\r?\n?)$', ln)
            if mo:
                lines[i] = mo.group(1) + new + mo.group(2) + (mo.group(3) or "\n")
                done += 1
    if done:
        open(path, "w").write("".join(lines))
    return done


# ---------------------------------------------------------------------------
# Case table. Each row is (label, manifest text, expected read_pin,
# expected read_package). The must-NOT-match rows matter as much as the
# must-match ones: a parser that answers "0.1.0" to everything passes every
# positive row on its own.
CASES = [
    (
        "inline table, the original spelling",
        '[package]\nname = "f"\n\n[dependencies]\n'
        'upstream = { path = "../up", version = "1.2.3", package = "up" }\n',
        "1.2.3",
        "up",
    ),
    (
        "multi-line [dependencies.upstream] table -- the #2628 defect",
        '[package]\nname = "f"\n\n[dependencies.upstream]\n'
        'path = "../up"\nversion = "1.2.3"\npackage = "up"\n',
        "1.2.3",
        "up",
    ),
    (
        "multi-line table, keys in the other order",
        '[dependencies.upstream]\npackage = "up"\nversion = "9.9.9"\n',
        "9.9.9",
        "up",
    ),
    (
        "bare string version",
        '[dependencies]\nupstream = "4.5.6"\n',
        "4.5.6",
        None,
    ),
    (
        "no rename: package defaults to the dependency name",
        '[dependencies]\nupstream = { version = "1.0.0" }\n',
        "1.0.0",
        "upstream",
    ),
    (
        "dev-dependencies is read too",
        '[dev-dependencies]\nupstream = { version = "2.0.0", package = "up" }\n',
        "2.0.0",
        "up",
    ),
    (
        "build-dependencies is read too",
        '[build-dependencies.upstream]\nversion = "3.0.0"\npackage = "up"\n',
        "3.0.0",
        "up",
    ),
    (
        "dependencies wins over dev-dependencies",
        '[dependencies]\nupstream = { version = "1.0.0", package = "a" }\n'
        '[dev-dependencies]\nupstream = { version = "2.0.0", package = "b" }\n',
        "1.0.0",
        "a",
    ),
    (
        "quoted key name",
        '[dependencies]\n"upstream" = { version = "1.4.0", package = "up" }\n',
        "1.4.0",
        "up",
    ),
    (
        "extra whitespace everywhere",
        "[ dependencies ]\n   upstream   =   { version   =   '1.5.0' }\n",
        "1.5.0",
        "upstream",
    ),
    (
        "trailing comment after the inline table",
        '[dependencies]\nupstream = { version = "1.6.0" }  # pinned by bump-version\n',
        "1.6.0",
        "upstream",
    ),
    (
        "a `#` inside a string is not a comment",
        '[dependencies]\nupstream = { version = "1.7.0", package = "up#x" }\n',
        "1.7.0",
        "up#x",
    ),
    # ---- must NOT match ----
    (
        "NO-MATCH a commented-out pin is not a pin",
        '[dependencies]\n# upstream = { version = "1.2.3" }\n',
        None,
        None,
    ),
    (
        "NO-MATCH path-only dependency has no registry pin, but does name a package",
        '[dependencies]\nupstream = { path = "../up", package = "up" }\n',
        None,
        "up",
    ),
    (
        "NO-MATCH a different dependency named similarly",
        '[dependencies]\nupstream-extra = { version = "1.2.3" }\n',
        None,
        None,
    ),
    (
        "NO-MATCH a serde dependency is not the upstream",
        '[dependencies]\nserde = { version = "1.0", features = ["derive"] }\n',
        None,
        None,
    ),
    (
        "NO-MATCH `upstream` under a non-dependency table",
        '[package.metadata]\nupstream = "1.2.3"\n',
        None,
        None,
    ),
    (
        "NO-MATCH a target-specific table is not a top-level dependency table",
        '[target."cfg(unix)".dependencies]\nupstream = { version = "1.2.3" }\n',
        None,
        None,
    ),
    (
        "NO-MATCH the version key of a NEIGHBOURING table does not leak in",
        '[dependencies.upstream]\npath = "../up"\npackage = "up"\n'
        '\n[dependencies.other]\nversion = "8.8.8"\n',
        None,
        "up",
    ),
    (
        "NO-MATCH `[package] version` is not the pin",
        '[package]\nname = "f"\nversion = "0.4.0"\n',
        None,
        None,
    ),
    (
        "NO-MATCH a manifest with no dependencies at all",
        '[package]\nname = "signpost"\nversion = "0.4.0"\n',
        None,
        None,
    ),
]


WRITE_CASES = (
    (
        "write round-trips the inline spelling",
        '[dependencies]\nupstream = { path = "../up", version = "1.2.3", package = "up" }\n',
    ),
    (
        "write round-trips the multi-line spelling",
        '[dependencies.upstream]\npath = "../up"\nversion = "1.2.3"\npackage = "up"\n',
    ),
)


def _write(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _check_read_cases(path):
    """The must-match / must-not-match table. -> number of failures."""
    fails = 0
    for label, text, want_pin, want_pkg in CASES:
        _write(path, text)
        got_pin, got_pkg = read_pin(path), read_package(path)
        if got_pin != want_pin or got_pkg != want_pkg:
            print(
                "FAIL  %s\n      pin: got %r want %r\n      pkg: got %r want %r"
                % (label, got_pin, want_pin, got_pkg, want_pkg)
            )
            fails += 1
        else:
            print("ok    %s" % label)
    return fails


def _check_write_cases(path):
    """A writer that edits a spelling the reader cannot see is the #2628 shape."""
    fails = 0
    for label, text in WRITE_CASES:
        _write(path, text)
        n = write_pin(path, "7.7.7")
        got = read_pin(path)
        if not n or got != "7.7.7":
            print("FAIL  %s: wrote %r, read back %r" % (label, n, got))
            fails += 1
        else:
            print("ok    %s" % label)
    return fails


def _oracle_pin(tomllib, path):
    """read_pin's answer, computed by tomllib instead. None if unparsable."""
    with open(path, "rb") as fh:
        try:
            doc = tomllib.load(fh)
        except tomllib.TOMLDecodeError:
            return None, False
    for table in DEP_TABLES:
        dep = doc.get(table, {}).get("upstream")
        if isinstance(dep, dict) and dep.get("version"):
            return dep["version"], True
        if isinstance(dep, str):
            return dep, True
    return None, True


def _compare_to_oracle(tomllib, subjects):
    """subjects: iterable of (label, path). -> (failures, compared)."""
    fails = 0
    checked = 0
    for label, real in subjects:
        oracle, parsed = _oracle_pin(tomllib, real)
        if not parsed:
            continue
        mine = read_pin(real)
        if mine != oracle:
            print(
                "FAIL  oracle disagrees on %s: hand %r tomllib %r"
                % (label, mine, oracle)
            )
            fails += 1
        checked += 1
    return fails, checked


def _real_manifests(root):
    for dirpath, _dirs, files in os.walk(os.path.join(root, "crates")):
        if "Cargo.toml" in files:
            yield os.path.join(dirpath, "Cargo.toml")


def _check_oracle(path):
    """Differential oracle: where tomllib exists it is the authority on TOML.

    Runs at TEST time only. No runtime path imports it, so a 3.10 host and a
    3.13 host execute the same resolver -- which is the point of aprender#2635.
    """
    try:
        import tomllib  # min-python-ok
    except ImportError:
        print(
            "note  tomllib absent (python %d.%d) -- differential oracle SKIPPED "
            "on this host; the case table above still ran"
            % (sys.version_info[0], sys.version_info[1])
        )
        return 0

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fails = 0
    checked = 0
    for label, text, _p, _k in CASES:
        _write(path, text)
        f, c = _compare_to_oracle(tomllib, [(label, path)])
        fails += f
        checked += c
    f, c = _compare_to_oracle(tomllib, ((m, m) for m in _real_manifests(root)))
    fails += f
    checked += c
    print("ok    differential oracle: hand parser == tomllib on %d manifests" % checked)
    return fails


def _self_test():
    import tempfile

    path = os.path.join(tempfile.mkdtemp(), "Cargo.toml")
    fails = _check_read_cases(path)
    fails += _check_write_cases(path)

    # Vacuity: a parser returning None for everything satisfies every NO-MATCH
    # row, so the table must carry enough must-MATCH rows to exclude that.
    if len([c for c in CASES if c[2] is not None]) < 5:
        print("FAIL  the case table has too few must-MATCH rows to be falsifying")
        fails += 1

    fails += _check_oracle(path)

    if fails:
        print("\nSELF-TEST FAILED (%d)" % fails)
        return 1
    print("\nSELF-TEST PASSED (%d rows)" % (len(CASES) + len(WRITE_CASES)))
    return 0


def _emit(resolver, path):
    """Print what `resolver` finds, or nothing. exit 3 if the file is unreadable."""
    try:
        value = resolver(path)
    except OSError:
        return 3
    if value:
        print(value)
    return 0


def main(argv):
    if len(argv) >= 2 and argv[1] == "--self-test":
        return _self_test()
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    mode, path = argv[1], argv[2]
    readers = {"read": read_pin, "package": read_package}
    if mode in readers:
        return _emit(readers[mode], path)
    if mode != "write" or len(argv) < 4:
        return 2
    return 0 if write_pin(path, argv[3]) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
