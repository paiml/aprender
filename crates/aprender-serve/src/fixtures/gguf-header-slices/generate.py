#!/usr/bin/env python3
"""Cut a REAL GGUF header down to a committable fixture (#3609).

A GGUF's metadata section carries its tokenizer: the vocabulary, the merges, and
the keys that say which special tokens the model declares. The real Qwen3.5-0.8B
header is 10.9 MB (248,320 tokens, 247,587 merges), too large to commit. What
the #3609 tests need from it is the real KEY SET and real values: Qwen3.5
declares `tokenizer.ggml.eos_token_id` and has NO `tokenizer.ggml.unknown_token_id`
key at all, while TinyLlama declares `unknown_token_id = 0` (`<unk>`).

So the output is the source header with:
  * every key, and every scalar and string value, copied byte-for-byte;
  * the tokenizer's per-token arrays (tokens, token_type, scores, merges) cut
    to their first --keep entries, which is the ONLY change;
  * tensor_count = 0 and no tensor data (a header, not a model).

The truncation is named here and in MANIFEST.json, never hidden. MANIFEST.json
records the source file's name, the sha256 of its header bytes, and the output's
sha256, so a reader can check that the fixture came from what it claims to.

usage: generate.py SOURCE.gguf OUT.gguf-header --keep N
"""
import argparse
import hashlib
import json
import os
import struct
import sys

# The per-token arrays: the only values this script changes.
SLICED = {
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.scores",
    "tokenizer.ggml.merges",
}
SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
STRING, ARRAY = 8, 9


def read_exact(f, n):
    b = f.read(n)
    if len(b) != n:
        sys.exit(f"truncated source: wanted {n} bytes, got {len(b)}")
    return b


def read_string_raw(f):
    """The string's length prefix and bytes, verbatim."""
    head = read_exact(f, 8)
    (n,) = struct.unpack("<Q", head)
    return head + read_exact(f, n)


def read_value_raw(f, vtype):
    """(raw bytes of the value, element list for arrays or None)."""
    if vtype == STRING:
        return read_string_raw(f), None
    if vtype == ARRAY:
        head = read_exact(f, 12)
        etype, n = struct.unpack("<IQ", head)
        elems = []
        for _ in range(n):
            if etype == STRING:
                elems.append(read_string_raw(f))
            elif etype in SCALAR_SIZE:
                elems.append(read_exact(f, SCALAR_SIZE[etype]))
            else:
                sys.exit(f"unsupported nested array element type {etype}")
        return head + b"".join(elems), (etype, elems)
    if vtype in SCALAR_SIZE:
        return read_exact(f, SCALAR_SIZE[vtype]), None
    sys.exit(f"unsupported value type {vtype}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--keep", type=int, required=True)
    a = ap.parse_args()

    with open(a.source, "rb") as f:
        if read_exact(f, 4) != b"GGUF":
            sys.exit("source is not a GGUF file")
        (version,) = struct.unpack("<I", read_exact(f, 4))
        tensor_count, kv_count = struct.unpack("<QQ", read_exact(f, 16))
        kvs, sliced = [], {}
        for _ in range(kv_count):
            key_raw = read_string_raw(f)
            key = key_raw[8:].decode("utf-8")
            vtype_raw = read_exact(f, 4)
            (vtype,) = struct.unpack("<I", vtype_raw)
            value_raw, arr = read_value_raw(f, vtype)
            if key in SLICED:
                if arr is None:
                    sys.exit(f"{key} is not an array in the source")
                etype, elems = arr
                keep = elems[: a.keep]
                value_raw = struct.pack("<IQ", etype, len(keep)) + b"".join(keep)
                sliced[key] = {"source_len": len(elems), "kept": len(keep)}
            kvs.append(key_raw + vtype_raw + value_raw)
        header_end = f.tell()
        f.seek(0)
        source_header_sha256 = hashlib.sha256(read_exact(f, header_end)).hexdigest()

    out = b"GGUF" + struct.pack("<I", version) + struct.pack("<QQ", 0, len(kvs)) + b"".join(kvs)
    with open(a.out, "wb") as g:
        g.write(out)

    manifest_path = os.path.join(os.path.dirname(os.path.abspath(a.out)), "MANIFEST.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as m:
            manifest = json.load(m)
    manifest[os.path.basename(a.out)] = {
        "source_file": os.path.basename(a.source),
        "source_gguf_version": version,
        "source_tensor_count": tensor_count,
        "source_header_bytes": header_end,
        "source_header_sha256": source_header_sha256,
        "keys": len(kvs),
        "sliced_arrays": sliced,
        "keep": a.keep,
        "output_bytes": len(out),
        "output_sha256": hashlib.sha256(out).hexdigest(),
    }
    with open(manifest_path, "w") as m:
        json.dump(manifest, m, indent=2, sort_keys=True)
        m.write("\n")
    print(f"{a.out}: {len(out)} bytes, {len(kvs)} keys, sliced {sorted(sliced)}")


if __name__ == "__main__":
    main()
