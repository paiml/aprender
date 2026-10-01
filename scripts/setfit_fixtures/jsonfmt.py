"""Human-readable JSON writer for the fixture corpus (D-11).

``json.dump(..., indent=2)`` puts every element of a flat numeric array on its own
line. A gradients fixture holds ~110k floats, so that produces a 110k-line file that is
"indented" but not readable by any useful definition, and unreviewable in a diff.

This writer indents STRUCTURE (objects, and lists of objects) while keeping numeric
payloads inline and wrapped. The shape of a fixture stays legible, and a changed number
shows up as a short line-level diff instead of a 110k-line rewrite.

Floats are emitted through ``%.9g``. Nine significant digits is the documented
round-trip width for IEEE-754 binary32, so this is lossless for the f32 values every
fixture records, while roughly halving the byte size versus full float64 repr.
"""

from __future__ import annotations

import json
import math

WRAP_COLUMNS = 96


def _fmt_scalar(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"refusing to serialise non-finite value {v!r} into a fixture")
        # %.9g round-trips binary32 exactly; float() then repr() gives the shortest
        # form that still round-trips, so this never loses a bit of the f32 value.
        return repr(float(f"{v:.9g}"))
    if v is None:
        return "null"
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    raise TypeError(f"unsupported type {type(v)!r}")


def _is_numeric_list(obj: list) -> bool:
    return all(isinstance(x, (int, float, bool)) for x in obj)


def _dumps(obj: object, indent: int) -> str:
    pad = " " * indent
    inner = " " * (indent + 2)

    if isinstance(obj, dict):
        if not obj:
            return "{}"
        parts = [
            f"{inner}{json.dumps(k, ensure_ascii=False)}: {_dumps(v, indent + 2)}"
            for k, v in obj.items()
        ]
        return "{\n" + ",\n".join(parts) + f"\n{pad}}}"

    if isinstance(obj, list):
        if not obj:
            return "[]"
        if _is_numeric_list(obj):
            # Inline, wrapped. This is the whole point of the module.
            out: list[str] = []
            line = inner
            for i, v in enumerate(obj):
                tok = _fmt_scalar(v) + ("," if i < len(obj) - 1 else "")
                if len(line) + len(tok) + 1 > WRAP_COLUMNS and line.strip():
                    out.append(line.rstrip())
                    line = inner
                line += tok + " "
            if line.strip():
                out.append(line.rstrip())
            if len(out) == 1:
                return "[" + out[0].strip() + "]"
            return "[\n" + "\n".join(out) + f"\n{pad}]"
        parts = [f"{inner}{_dumps(v, indent + 2)}" for v in obj]
        return "[\n" + ",\n".join(parts) + f"\n{pad}]"

    return _fmt_scalar(obj)


def dumps(obj: object) -> str:
    return _dumps(obj, 0) + "\n"


def write(path, obj) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dumps(obj))
