#!/usr/bin/env python3
"""The model ladder's HELD universe, read from each file's own header (#3846).

The inventory used to be a filename glob. A glob has the same omission property as
the hand-kept list it replaced: a file whose name does not carry its quantisation
(any `.apr`, `Qwen3-0.6B-BF16.gguf`) is invisible to every gate, and nobody reviews
the string match. This module lists EVERY model file at depth 1 in the inventory
dirs, reads architecture + quantisation from the header, and classifies each one
against the contract's `inventory.scope`:

  swept          -- the ladder measures it (header in scope, or a filename pattern
                    matched: the union only grows the universe, never shrinks it)
  held_not_swept -- on the host, not measured, with the scope rule that excluded it

A file whose header cannot be read is held_not_swept with reason `unreadable: ...`;
the checker refuses that, because an unclassifiable file is exactly the blind spot.

Usage: model_ladder_universe.py <ladder.yaml> <dir:dir:...> <pattern,pattern,...>
Prints one JSON object: {"held": N, "swept": [{file, path, arch, quant, why}],
"held_not_swept": [{file, arch, quant, reason}]}.
"""
import fnmatch
import json
import os
import struct
import sys

EXTS = (".gguf", ".apr", ".safetensors")

# llama.cpp LLAMA_FTYPE -> name (general.file_type)
GGUF_FTYPE = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0",
    37: "TQ2_0", 38: "MXFP4_MOE",
}
# GGUF value types -> struct format of a fixed-width scalar
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
           10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9


def _read(f, n):
    b = f.read(n)
    if len(b) != n:
        raise ValueError("truncated header")
    return b


def _string(f):
    (n,) = struct.unpack("<Q", _read(f, 8))
    return _read(f, n).decode("utf-8", "replace")


def _skip_value(f, t):
    if t in _SCALAR:
        f.seek(struct.calcsize(_SCALAR[t]), 1)
    elif t == _STRING:
        (n,) = struct.unpack("<Q", _read(f, 8))
        f.seek(n, 1)
    elif t == _ARRAY:
        et, count = struct.unpack("<IQ", _read(f, 12))
        if et in _SCALAR:
            f.seek(struct.calcsize(_SCALAR[et]) * count, 1)
        else:
            for _ in range(count):
                _skip_value(f, et)
    else:
        raise ValueError(f"unknown GGUF value type {t}")


def read_gguf(path):
    """(arch, quant) from general.architecture and general.file_type."""
    arch = quant = None
    with open(path, "rb") as f:
        if _read(f, 4) != b"GGUF":
            raise ValueError("no GGUF magic")
        (version,) = struct.unpack("<I", _read(f, 4))
        if version < 2:
            raise ValueError(f"GGUF v{version} is not supported")
        _tensors, kvs = struct.unpack("<QQ", _read(f, 16))
        for _ in range(kvs):
            key = _string(f)
            (t,) = struct.unpack("<I", _read(f, 4))
            if key == "general.architecture" and t == _STRING:
                arch = _string(f)
            elif key == "general.file_type" and t in _SCALAR:
                (v,) = struct.unpack(_SCALAR[t], _read(f, struct.calcsize(_SCALAR[t])))
                quant = GGUF_FTYPE.get(int(v), f"ftype{int(v)}")
            else:
                _skip_value(f, t)
            if arch and quant:
                break
    if not arch:
        raise ValueError("no general.architecture")
    return arch, quant or "unknown"


def read_apr(path):
    """(arch, quant) from the APR v2 JSON metadata (offset from the header)."""
    with open(path, "rb") as f:
        head = _read(f, 64)
        if head[:4] != b"APR\x00":
            raise ValueError("no APR magic")
        (off,) = struct.unpack_from("<I", head, 12)
        f.seek(off)
        raw = f.read(64 << 20).decode("utf-8", "replace")
    meta, _ = json.JSONDecoder().raw_decode(raw)
    arch = meta.get("architecture")
    if not arch:
        raise ValueError("no architecture in APR metadata")
    q = meta.get("quantization")
    quant = (q.get("quant_type") if isinstance(q, dict) else None) or "unquantized"
    return arch, quant


def read_safetensors(path):
    """(arch, quant): arch from the sibling config.json, quant = the dominant dtype."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", _read(f, 8))
        if n > (256 << 20):
            raise ValueError("implausible safetensors header length")
        header = json.loads(_read(f, n))
    counts = {}
    for k, v in header.items():
        if k != "__metadata__" and isinstance(v, dict):
            counts[v.get("dtype", "?")] = counts.get(v.get("dtype", "?"), 0) + 1
    if not counts:
        raise ValueError("no tensors")
    quant = max(sorted(counts), key=counts.get)
    cfg = os.path.join(os.path.dirname(path), "config.json")
    arch = "unknown"
    if os.path.isfile(cfg):
        arch = json.load(open(cfg)).get("model_type") or "unknown"
    return arch, quant


READERS = {".gguf": read_gguf, ".apr": read_apr, ".safetensors": read_safetensors}


def _norm(s):
    return s.lower().replace("_", "").replace(".", "").replace("-", "")


def classify(fmt, arch, quant, scope):
    """(swept, reason). The reason names the scope rule that decided it."""
    formats = scope.get("formats") or []
    if fmt not in formats:
        return False, f"format {fmt} not in inventory.scope.formats {formats}"
    archs = {_norm(a) for a in scope.get("archs") or []}
    if _norm(arch) in archs:
        return True, f"arch {arch} in inventory.scope.archs"
    for q in scope.get("quants") or []:
        if fnmatch.fnmatch(quant.upper(), q.upper()):
            return True, f"quant {quant} matches inventory.scope.quants {q}"
    return False, (f"arch {arch} not in inventory.scope.archs and quant {quant} "
                   f"matches no inventory.scope.quants")


def universe(dirs, patterns, scope):
    pats = [p.lower() for p in patterns]
    swept, held_not, seen = [], [], set()
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            ext = os.path.splitext(f)[1].lower()
            if f in seen or not os.path.isfile(p) or ext not in EXTS:
                continue
            seen.add(f)
            by_name = any(fnmatch.fnmatch(f.lower(), pat) for pat in pats)
            try:
                arch, quant = READERS[ext](p)
            except (OSError, ValueError, UnicodeDecodeError) as e:
                if by_name:  # the ladder measures it anyway; the run decides its verdict
                    swept.append({"file": f, "path": p, "arch": "unreadable",
                                  "quant": "unreadable", "why": "filename pattern"})
                else:
                    held_not.append({"file": f, "arch": "unreadable", "quant": "unreadable",
                                     "reason": f"unreadable: {e}"})
                continue
            ok, why = classify(ext[1:], arch, quant, scope)
            if ok or by_name:
                swept.append({"file": f, "path": p, "arch": arch, "quant": quant,
                              "why": why if ok else "filename pattern"})
            else:
                held_not.append({"file": f, "arch": arch, "quant": quant, "reason": why})
    return {"held": len(swept) + len(held_not), "swept": swept, "held_not_swept": held_not}


def main(argv):
    import yaml
    inv = yaml.safe_load(open(argv[1]))["ladder"].get("inventory") or {}
    scope = inv.get("scope")
    if not isinstance(scope, dict) or not scope.get("formats"):
        print("decline: the ladder declares no inventory.scope {formats, archs, quants} (#3846)",
              file=sys.stderr)
        return 2
    print(json.dumps(universe(argv[2].split(":"), argv[3].split(","), scope)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
