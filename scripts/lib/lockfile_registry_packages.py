"""Print the name of every crates.io-sourced package in a Cargo.lock.

Separate file, not an inline heredoc: bashrs parses embedded python as shell and
reports phantom SC1007/SC1078 errors on ordinary assignments and quotes.

A package block looks like:

    [[package]]
    name = "serde"
    version = "1.0.0"
    source = "registry+https://github.com/rust-lang/crates.io-index"

`source` is ABSENT for path/workspace members and carries a `git+` prefix for git
dependencies; neither is a registry package. Parsing must be block-scoped, or a
`source` line leaks onto the preceding package and reports a workspace member as
coming from crates.io.

argv: <Cargo.lock path>
"""
import sys

name = None
source = None
out = []


def flush():
    if name is not None and source is not None and source.startswith("registry+"):
        out.append(name)


with open(sys.argv[1], encoding="utf-8", errors="replace") as fh:
    for raw in fh:
        line = raw.strip()
        if line == "[[package]]":
            flush()
            name = None
            source = None
        elif line.startswith("name = "):
            name = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("source = "):
            source = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("[") and line != "[[package]]":
            # Any other table ends the package section.
            flush()
            name = None
            source = None
flush()

for n in out:
    print(n)
