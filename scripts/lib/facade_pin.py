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

Reading uses tomllib (authoritative about TOML structure, and unlike
`cargo metadata` it does not require a resolvable workspace, so the --self-test
fixtures still work). Writing is a targeted text edit so comments and formatting
survive -- round-tripping through a TOML writer would reformat the manifest.

  read    <manifest>            -> prints the version, or nothing if no upstream
                                   exit 0; exit 3 if the file could not be parsed
  package <manifest>            -> prints the upstream's real package name
                                   (the `package = ` rename target), or nothing
  write   <manifest> <version>  -> exit 0 if rewritten, 1 if no pin was found
"""
import re
import sys
import tomllib


def read_pin(path):
    with open(path, "rb") as fh:
        m = tomllib.load(fh)
    for table in ("dependencies", "dev-dependencies", "build-dependencies"):
        dep = m.get(table, {}).get("upstream")
        if isinstance(dep, dict) and dep.get("version"):
            return dep["version"]
        if isinstance(dep, str):
            return dep
    return None


def read_package(path):
    """The real crate the `upstream` rename points at, for ordering-graph edges."""
    with open(path, "rb") as fh:
        m = tomllib.load(fh)
    for table in ("dependencies", "dev-dependencies", "build-dependencies"):
        dep = m.get(table, {}).get("upstream")
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


def main(argv):
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    mode, path = argv[1], argv[2]
    if mode == "read":
        try:
            pin = read_pin(path)
        except Exception:
            return 3
        if pin:
            print(pin)
        return 0
    if mode == "package":
        try:
            pkg = read_package(path)
        except Exception:
            return 3
        if pkg:
            print(pkg)
        return 0
    if mode == "write":
        if len(argv) < 4:
            return 2
        return 0 if write_pin(path, argv[3]) else 1
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
