#!/usr/bin/env python3
"""Can this helper actually load under the interpreter running me right now?

aprender#2635, the dynamic half of scripts/check_python_helpers_min_version.sh.
Two questions, both answered WITHOUT executing the file under audit:

  1. does it compile?            (the builtin `compile`, so syntax newer than this
                                  interpreter is a hard failure, not a surprise
                                  at gate time)
  2. does every module it imports RESOLVE?  (`importlib.util.find_spec`, which
                                  locates a module without importing it, so a
                                  helper with side effects is never run)

Imports carrying the `# min-python-ok` marker are skipped -- those are declared
guarded imports, and the shell gate prints each one so the excuse is visible.

Prints one `MISSING <line> <module>` per unresolvable import, or `SYNTAX <msg>`,
and exits 1 if anything failed. Silent + exit 0 means the helper loads here.
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import py_imports  # noqa: E402  (needs the path insert above)


def _resolves(module):
    """Is `module` importable here? find_spec locates without importing."""
    try:
        return importlib.util.find_spec(module.split(".")[0]) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _compiles(path):
    """-> None if the file parses under this interpreter, else the message."""
    try:
        with open(path, "rb") as fh:
            compile(fh.read(), path, "exec")
    except (SyntaxError, ValueError) as exc:
        return str(exc).strip()
    return None


def check(path):
    syntax = _compiles(path)
    if syntax is not None:
        return ["SYNTAX %s" % syntax]

    # A helper may import a sibling helper; resolve from its own directory the
    # way the interpreter will when it is actually invoked.
    here = os.path.dirname(os.path.abspath(path))
    if here not in sys.path:
        sys.path.insert(0, here)

    return [
        "MISSING %d %s" % (lineno, mod)
        for lineno, mod, kind in py_imports.imports(path)
        if kind != "marked" and not _resolves(mod)
    ]


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    rc = 0
    for path in argv[1:]:
        for problem in check(path):
            print(problem)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
