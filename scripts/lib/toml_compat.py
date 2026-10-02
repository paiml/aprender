"""tomllib where the interpreter has it (3.11+), else a [package]-only reader.

The x86 runner host ships python 3.10 with neither tomllib nor tomli, and
`import tomllib` killed the tarball guards there before one row ran (#4429
x86-main, guard-cargo step 57). The callers read only `["package"]["name"]` and
`["package"]["version"]` of a Cargo.toml, so the fallback parses exactly that:
single-line `key = "basic"` / `key = 'literal'` strings in the [package] table.
Anything it does not understand in [package] is left out, so a caller asking
for it gets KeyError -- a refusal, never a guessed value.
"""
import re

try:
    from tomllib import TOMLDecodeError, loads  # noqa: F401
except ModuleNotFoundError:

    class TOMLDecodeError(ValueError):
        pass

    _PACKAGE = re.compile(r"^\[\s*package\s*\]\s*(#.*)?$")
    _PAIR = re.compile(r"""^([A-Za-z0-9_-]+)\s*=\s*(?:"([^"\\]*)"|'([^']*)')\s*(#.*)?$""")

    def loads(text):
        table, package, seen, in_ml = None, {}, False, None
        for raw in text.splitlines():
            line = raw.strip()
            if in_ml:
                # inside a multi-line string: its lines are text, never keys
                if line.count(in_ml) % 2 == 1:
                    in_ml = None
                continue
            if not line or line.startswith("#"):
                continue
            for q in ('"""', "'''"):
                if line.count(q) % 2 == 1:
                    in_ml = q
                    break
            if in_ml:
                continue
            if line.startswith("["):
                # Any header but exactly [package] (a [[bin]], [package.metadata.x],
                # [target.'cfg(unix)'.dependencies]) ends the package table.
                table = "package" if _PACKAGE.match(line) else None
                if table:
                    if seen:
                        raise TOMLDecodeError("duplicate [package] table")
                    seen = True
                continue
            if table != "package":
                continue
            m = _PAIR.match(line)
            if m:
                key = m.group(1)
                if key in package:
                    raise TOMLDecodeError("duplicate key %r in [package]" % key)
                package[key] = m.group(2) if m.group(2) is not None else m.group(3)
        return {"package": package} if seen else {}
