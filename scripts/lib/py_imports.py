#!/usr/bin/env python3
"""List the modules a python file imports, without executing it.

aprender#2635. Used by scripts/check_python_helpers_min_version.sh to decide
whether a helper on a release-gate path can run under the minimum Python the
fleet provides.

Parsing with `ast` rather than importing the file matters twice over: importing
would run module-level code (a gate helper must never be executed just to be
audited), and it would answer only for the interpreter doing the auditing --
which is the whole defect being guarded against.

Each import is printed as:

    <line>\t<module>\t<marked|plain>

`module` is the dotted name as written. `marked` means the source line carries
the `# min-python-ok` declared-intent marker, i.e. the author asserts the import
is guarded (inside a try/except ImportError) or otherwise cannot run at gate
time. The gate prints every marked line rather than hiding it.

Only `ast` and `sys` are used, both stdlib since forever, so this file cannot
itself become the thing that fails to load on an old interpreter.
"""
import ast
import sys

MARKER = "# min-python-ok"


def _module_names(node):
    """The dotted module names this node imports; [] if none, None if not an import."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        # `from . import x` has no module; a relative import cannot be a
        # stdlib module, so it is not interesting here.
        if node.level or not node.module:
            return []
        return [node.module]
    return None


def _line_kind(lines, lineno):
    """'marked' if the source line carries the declared-intent marker."""
    text = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
    return "marked" if MARKER in text else "plain"


def imports(path):
    with open(path, "rb") as fh:
        src = fh.read()
    lines = src.decode("utf-8", "replace").splitlines()
    out = []
    # ast.walk, not tree.body: an import nested inside a function or a
    # try/except is still an import that runs at gate time.
    for node in ast.walk(ast.parse(src, filename=path)):
        names = _module_names(node)
        if not names:
            continue
        lineno = getattr(node, "lineno", 0)
        kind = _line_kind(lines, lineno)
        out.extend((lineno, name, kind) for name in names)
    return sorted(set(out))


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    rc = 0
    for path in argv[1:]:
        try:
            rows = imports(path)
        except (OSError, SyntaxError) as exc:
            print("ERROR\t%s\t%s" % (path, exc), file=sys.stderr)
            rc = 1
            continue
        for lineno, name, kind in rows:
            print("%d\t%s\t%s" % (lineno, name, kind))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
