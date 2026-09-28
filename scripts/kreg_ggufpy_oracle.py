#!/usr/bin/env python3
"""Write the INDEPENDENT oracle for KREG-001 AC-3 parity receipts (aprender#4539).

For each GGML type: seeded random blocks (scale fields set finite and small), seeded
activations, and llama.cpp gguf-py's dequantization of those blocks dotted with the
activations in float64. gguf-py decodes from its own tables (IQ grids from its own
``grid_hex``), so a transcription error shared by our decoder and our kernel cannot
hide behind agreement, which the in-tree oracle cannot rule out.

Writes ``<outdir>/<TYPE>/weights.bin`` (the quantized bytes, row-major ``[out_dim]``
rows), ``x.bin`` (little-endian float32 ``[in_dim]``), ``ref.bin`` (little-endian
float64 ``[out_dim]``), ``mag.bin`` (little-endian float64 ``[out_dim]``: |W|·|x| from
the same gguf-py dequantization, the magnitude KTEST-02's EM-DOT bound is scaled by) and
``meta.json`` (shape, seed, gguf-py origin, sha256 of each
file). ``kernel_registry::parity`` measures the Rust kernels against these files.

Refuses rather than writing a vacuous oracle: an unknown type, a non-finite or
all-zero reference, exits non-zero.

usage: PYTHONPATH=<llama.cpp>/gguf-py kreg_ggufpy_oracle.py <outdir> <TYPE>...
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
from gguf import GGMLQuantizationType, GGML_QUANT_SIZES
from gguf.quants import dequantize

IN_DIM = 1024
OUT_DIM = 32

# (offset, low, high) of every f16 scale field in one block; the rest of the block is random.
SCALES = {
    "Q4_1": [(0, 0.002, 0.02), (2, -0.01, 0.01)],
    "Q5_0": [(0, 0.002, 0.02)],
    "Q5_1": [(0, 0.002, 0.02), (2, -0.01, 0.01)],
    "Q2_K": [(80, 0.002, 0.02), (82, 0.0, 0.01)],
    "Q3_K": [(108, 0.002, 0.02)],
    "IQ2_XXS": [(0, 0.002, 0.02)],
    "IQ3_XXS": [(0, 0.002, 0.02)],
    "IQ4_NL": [(0, 0.002, 0.02)],
    "IQ3_S": [(0, 0.002, 0.02)],
    "IQ2_S": [(0, 0.002, 0.02)],
    "IQ4_XS": [(0, 0.002, 0.02)],
}


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def origin() -> dict:
    import gguf

    path = os.path.dirname(os.path.dirname(os.path.abspath(gguf.__file__)))
    try:
        commit = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"gguf_py": os.path.abspath(gguf.__file__), "llama_cpp_commit": commit}


def emit(outdir: str, name: str) -> None:
    if name not in SCALES:
        sys.exit(f"unknown or unsupported type {name}")
    qtype = GGMLQuantizationType[name]
    block_elems, block_bytes = GGML_QUANT_SIZES[qtype]
    seed = 4539 + int(qtype)
    rng = np.random.default_rng(seed)
    blocks = OUT_DIM * IN_DIM // block_elems
    w = rng.integers(0, 256, size=(blocks, block_bytes), dtype=np.uint8)
    for off, lo, hi in SCALES[name]:
        w[:, off : off + 2] = rng.uniform(lo, hi, size=blocks).astype(np.float16).view(np.uint8).reshape(blocks, 2)
    x = rng.uniform(-1.0, 1.0, size=IN_DIM).astype(np.float32)
    deq = dequantize(w.reshape(OUT_DIM, -1), qtype).astype(np.float32)
    if deq.shape != (OUT_DIM, IN_DIM):
        sys.exit(f"{name}: gguf-py returned shape {deq.shape}")
    ref = deq.astype(np.float64) @ x.astype(np.float64)
    mag = np.abs(deq.astype(np.float64)) @ np.abs(x.astype(np.float64))
    if not np.all(np.isfinite(ref)) or not np.any(ref):
        sys.exit(f"{name}: a non-finite or all-zero reference measures nothing")
    d = os.path.join(outdir, name)
    os.makedirs(d, exist_ok=True)
    files = {"weights.bin": w.tobytes(), "x.bin": x.astype("<f4").tobytes(), "ref.bin": ref.astype("<f8").tobytes(), "mag.bin": mag.astype("<f8").tobytes()}
    for f, b in files.items():
        with open(os.path.join(d, f), "wb") as fh:
            fh.write(b)
    meta = {
        "schema": "kreg-ggufpy-oracle/v1",
        "ggml_type": name,
        "ggml_type_id": int(qtype),
        "in_dim": IN_DIM,
        "out_dim": OUT_DIM,
        "seed": seed,
        "oracle": "gguf_py_dequant_f64",
        **origin(),
        "sha256": {f: sha256(b) for f, b in files.items()},
    }
    with open(os.path.join(d, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    print(f"{name}: max|ref| {np.max(np.abs(ref)):.4g}")


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    for name in sys.argv[2:]:
        emit(sys.argv[1], name)


if __name__ == "__main__":
    main()
