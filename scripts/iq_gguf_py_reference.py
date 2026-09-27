#!/usr/bin/env python3
"""Emit llama.cpp gguf-py's dequantization of every tensor of one IQ type.

#3963 (and retroactively #3950): the GPU kernel is judged against aprender-serve's
CPU decoder, so the CPU decoder must itself be proven against an INDEPENDENT
decoder on the real bytes. gguf-py is independent in the way that matters: it
decodes from its own hex-encoded grid (``grid_hex`` / ``grid_map``), not from
``quantize::iq_grids``, so a transcription error shared by our grid constant and
our decoder cannot hide behind agreement.

Writes ``<outdir>/<tensor name>.f32`` (little-endian float32, row-major, shape
``[ne1, ne0]``) and ``<outdir>/MANIFEST.tsv`` (name, ne1, ne0, sha256 of the raw
quantized bytes). The Rust side
``quantize::iq_gguf_py_parity_tests::every_tensor_decodes_like_gguf_py`` compares
aprender's decoder against these files element by element.

Refuses rather than emitting nothing: an unknown type name, or a file holding no
tensor of the type, exits non-zero.

usage: iq_gguf_py_reference.py <model.gguf> <IQ2_XXS|IQ3_XXS|IQ2_S|...> <outdir>
  PYTHONPATH must include llama.cpp's gguf-py (e.g. /mnt/nvme-raid0/llama.cpp-master/gguf-py)
"""
import hashlib
import os
import sys

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__.strip().splitlines()[-2], file=sys.stderr)
        return 2
    path, type_name, outdir = sys.argv[1:]
    try:
        qtype = GGMLQuantizationType[type_name]
    except KeyError:
        print(f"unknown ggml type name {type_name!r}", file=sys.stderr)
        return 2
    os.makedirs(outdir, exist_ok=True)
    reader = GGUFReader(path)
    rows = []
    for t in reader.tensors:
        if t.tensor_type != qtype or len(t.shape) != 2:
            continue
        ne0, ne1 = int(t.shape[0]), int(t.shape[1])
        raw = np.asarray(t.data).tobytes()
        deq = dequantize(t.data, qtype).astype("<f4").reshape(ne1, ne0)
        deq.tofile(os.path.join(outdir, f"{t.name}.f32"))
        rows.append((t.name, ne1, ne0, hashlib.sha256(raw).hexdigest()))
    if not rows:
        print(f"{path} holds no 2-D {type_name} tensor: nothing to emit", file=sys.stderr)
        return 1
    with open(os.path.join(outdir, "MANIFEST.tsv"), "w", encoding="utf-8") as m:
        for r in rows:
            m.write("\t".join(map(str, r)) + "\n")
    print(f"gguf-py reference: {len(rows)} {type_name} tensors -> {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
