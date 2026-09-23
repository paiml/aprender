#!/usr/bin/env python3
"""gguf_census: the (quant type, k, n) shapes the held GGUF inventory actually uses (#3968).

A GPU whitelist entry admits a quant TYPE; a GEMV kernel is exercised at a SHAPE. #3951 measured the gap:
IQ4_XS was admitted on one model's [2560, 9216] while another held model used eight other shapes, one with
a non-power-of-two super-block count. This reads the tensor-info table of every held GGUF (header only, no
weights) and emits every 2-D weight's (type, k, n), so conformance rows are GENERATED from what is held,
never hand-listed.

k is ne[0] (the row length, the reduction axis of a GEMV), n is ne[1] (the output rows). A 1-D tensor is a
vector, never a GEMV, and is skipped.

CLI: gguf_census.py scan <dir>... --host <name> [--depth 1]   -> the committed census JSON on stdout
     exit 0 ok · 2 usage/ENV · a file that is not a readable GGUF is listed under `unreadable`, never dropped

The output IS the committed evidence, byte for byte: evidence/gpu-shape-census/census-<host>.json is written by
    python3 scripts/lib/gguf_census.py scan ~/models ~/.apr/models ~/.cache/apr/models --host <host> > census-<host>.json
with no transform after it (#3968 quorum: a hand step between a scanner and its evidence makes the evidence
unverifiable). It records its own roots (as given, `~` kept, so no host-specific absolute path), its depth, and
every GGUF it did NOT read because it sits deeper than --depth (`skipped_for_depth`), so a coverage gap is visible
in the evidence rather than silent.

--depth N reads GGUFs in a root and in up to N levels of subdirectories below it (0 = the root only).
"""

from __future__ import annotations

import argparse
from datetime import date
import json
import os
import struct
import sys
from pathlib import Path

SCHEMA = "gguf-census/v1"
# GGUF metadata value types -> struct format (fixed-size ones)
FIXED = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
STRING, ARRAY = 8, 9
# ggml_type -> name, the whitelist's vocabulary (crates/aprender-serve/src/gguf/dtype.rs)
GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
              10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
              17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
              23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16"}
# elements per block: 256 for the K and IQ super-block types, 32 for the legacy blocks, 1 for float types
BLOCK = {0: 1, 1: 1, 30: 1, 2: 32, 3: 32, 6: 32, 7: 32, 8: 32, 9: 32, 20: 32}


def block_of(qtype: int) -> int:
    return BLOCK.get(qtype, 256)


class Reader:
    def __init__(self, f):
        self.f = f

    def take(self, fmt: str):
        size = struct.calcsize("<" + fmt)
        buf = self.f.read(size)
        if len(buf) != size:
            raise ValueError("truncated header")
        return struct.unpack("<" + fmt, buf)[0]

    def string(self) -> str:
        n = self.take("Q")
        if n > 1 << 24:
            raise ValueError(f"implausible string length {n}")
        return self.f.read(n).decode("utf-8", "replace")

    def skip_value(self, vtype: int) -> None:
        if vtype in FIXED:
            self.f.seek(struct.calcsize("<" + FIXED[vtype]), os.SEEK_CUR)
        elif vtype == STRING:
            self.f.seek(self.take("Q"), os.SEEK_CUR)
        elif vtype == ARRAY:
            etype, count = self.take("I"), self.take("Q")
            if etype in FIXED:
                self.f.seek(struct.calcsize("<" + FIXED[etype]) * count, os.SEEK_CUR)
            else:
                for _ in range(count):
                    self.skip_value(etype)
        else:
            raise ValueError(f"unknown metadata value type {vtype}")


def read_tensors(path: Path) -> tuple[str, list]:
    """(architecture, [(name, qtype, dims)]) from the GGUF header."""
    with open(path, "rb") as f:
        r = Reader(f)
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF (bad magic)")
        version = r.take("I")
        if version < 2:
            raise ValueError(f"GGUF v{version} is not supported (v2+)")
        n_tensors, n_kv = r.take("Q"), r.take("Q")
        arch = None
        for _ in range(n_kv):
            key, vtype = r.string(), r.take("I")
            if key == "general.architecture" and vtype == STRING:
                arch = r.string()
            else:
                r.skip_value(vtype)
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.take("Q") for _ in range(r.take("I"))]
            qtype = r.take("I")
            r.take("Q")  # data offset
            tensors.append((name, qtype, dims))
        return arch or "unknown", tensors


def scan(dirs: list, depth: int) -> dict:
    files, unreadable, skipped, missing = [], [], [], []
    seen = set()
    for d in dirs:
        root = Path(os.path.expanduser(d))
        if not root.is_dir():
            missing.append(d)
            continue
        for p in sorted(root.rglob("*.gguf")):
            if not p.is_file():
                continue
            rel = p.relative_to(root)
            if len(rel.parts) - 1 > depth:  # parts - 1 = subdirectory levels below the root
                skipped.append(f"{d.rstrip('/')}/{rel}")
                continue
            real = p.resolve()
            if real in seen:
                continue
            seen.add(real)
            try:
                arch, tensors = read_tensors(p)
            except (OSError, ValueError, struct.error) as e:
                unreadable.append({"file": f"{d.rstrip('/')}/{rel}", "why": str(e)})
                continue
            shapes = {}
            for name, qtype, dims in tensors:
                if len(dims) != 2:
                    continue
                key = (qtype, int(dims[0]), int(dims[1]))
                shapes.setdefault(key, []).append(name)
            files.append({"file": p.name, "bytes": p.stat().st_size, "architecture": arch,
                          "shapes": [{"qtype": q, "type": GGML_TYPES.get(q, f"ggml{q}"), "k": k, "n": n,
                                      "block": block_of(q), "tensors": len(names), "example": names[0]}
                                     for (q, k, n), names in sorted(shapes.items())]})
    return {"schema": SCHEMA, "roots": list(dirs), "depth": depth, "missing_roots": missing, "files": files,
            "unreadable": unreadable, "skipped_for_depth": skipped}


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="gguf_census.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("dirs", nargs="+")
    s.add_argument("--depth", type=int, default=1)
    s.add_argument("--host", required=True, help="the host name the census is committed under (census-<host>.json)")
    a = ap.parse_args(argv)
    doc = {**scan(a.dirs, a.depth), "host": a.host, "measured": date.today().isoformat()}
    json.dump(doc, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
