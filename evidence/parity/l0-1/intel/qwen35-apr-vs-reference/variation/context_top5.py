#!/usr/bin/env python3
"""Top-5 context (PMAT-3091 variation): for prompt 4 positions 1, 6, 20 print top-5 (id, logit, piece)
from llama per-token, llama batched and apr, plus the decoded input pieces for positions 0..8 of
prompts 4 and the original prompt 0 pieces at 28 and 73. Context only; no threshold.
Usage: context_top5.py <variation-dir> <orig-subject-dir> <orig-per-token-dir> <orig-batched.bin> <model> <gguf-py>"""
import sys

import numpy as np

vd, sd, ptd, orig_bat, model, gpy = sys.argv[1:7]
sys.path.insert(0, gpy)
from gguf import GGUFReader  # noqa: E402

f = GGUFReader(model).fields["tokenizer.ggml.tokens"]
vocab = [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]


def load(path):
    buf = open(path, "rb").read()
    _, n, v = (int(x) for x in np.frombuffer(buf, "<i4", 3, 8))
    return np.frombuffer(buf, "<i4", n, 20), np.frombuffer(buf, "<f4", n * v, 20 + 4 * n).reshape(n, v)


def top5(row):
    idx = np.argsort(-row.astype(np.float64), kind="stable")[:5]
    return " | ".join(f"{int(i)} {float(row[i]):.4f} {vocab[int(i)]!r}" for i in idx)


def show(label, paths, positions):
    ids, pt = load(paths[0])
    _, bat = load(paths[1])
    _, apr = load(paths[2])
    print(f"== {label}: input pieces 0..8: " + " ".join(f"{k}:{vocab[int(ids[k])]!r}" for k in range(min(9, len(ids)))))
    for p in positions:
        print(f"-- {label} pos {p} input {int(ids[p])} {vocab[int(ids[p])]!r}")
        print(f"   llama per-token: {top5(pt[p])}")
        print(f"   llama batched  : {top5(bat[p])}")
        print(f"   apr            : {top5(apr[p])}")


show("prompt-4", [f"{vd}/p4-per-token.bin", f"{vd}/p4-batched.bin", f"{vd}/p4-apr.bin"], [1, 6, 20])
show("prompt-0", [f"{ptd}/per-token-run1.bin", orig_bat, f"{sd}/apr-intel-run1.bin"], [28, 73])
