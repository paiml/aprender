#!/usr/bin/env python3
"""PMAT-3091 layer observer: is a sub-layer step made by an engine's matmul, or carried in on its input?

Usage: kernel_isolation.py MODEL_GGUF GGUF_PY LAYER_TYPES_TSV LAYERS \
                           LABEL:LLAMA_DUMP_DIR:APR_OBS_DIR:POSLIST [LABEL:...]

For each (layer, pos) and each engine E in {llama, apr}, a float64 reference is computed from gguf-py's
dequantization of the weight applied to E's OWN dumped input, and E's dumped output is scored against it:
  DeltaNet:  z-N               = W(attn_gate)  . attn_norm-N
             linear_attn_out-N = W(ssm_out)    . final_output-N
  attention: attn_output-N     = W(attn_output). attn_gated-N
  both:      ffn_out-N         = W(ffn_down) . (silu(W(ffn_gate) . attn_post_norm-N) * W(ffn_up) . attn_post_norm-N)
A small rel_l2 for an engine means its kernel reproduces the float64 math on its own input; the cross-engine
difference at that output is then inherited from the input. A large one means the kernel itself departs.
Output: TSV on stdout (label pos layer output weights qtypes engine rel_l2_vs_ref cos_vs_ref). No threshold.
"""
import sys

import numpy as np


def read_f32(path):
    return np.fromfile(path, dtype="<f4").astype(np.float64)


def load_weights(model, gguf_py, layers):
    sys.path.insert(0, gguf_py)
    import gguf  # noqa: E402
    from gguf.quants import dequantize  # noqa: E402
    want = {"blk.%d.%s.weight" % (L, n) for L in layers
            for n in ("attn_gate", "ssm_out", "attn_output", "ffn_gate", "ffn_up", "ffn_down")}
    out = {}
    for t in gguf.GGUFReader(model).tensors:
        if t.name in want:
            n_in, n_out = int(t.shape[0]), int(t.shape[1])
            w = dequantize(np.asarray(t.data), t.tensor_type).reshape(n_out, n_in).astype(np.float64)
            out[t.name] = (w, t.tensor_type.name)
    return out


def silu(x):
    return x / (1.0 + np.exp(-x))


def score(ref, got):
    d = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    c = float(np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got)))
    return float(d), c


def checks(W, L, is_attn):
    b = "blk.%d." % L
    ffn = ("ffn_out-%d" % L, "attn_post_norm-%d" % L, (b + "ffn_gate.weight", b + "ffn_up.weight", b + "ffn_down.weight"),
           lambda x: W[b + "ffn_down.weight"][0] @ (silu(W[b + "ffn_gate.weight"][0] @ x) * (W[b + "ffn_up.weight"][0] @ x)))
    if is_attn:
        return [("attn_output-%d" % L, "attn_gated-%d" % L, (b + "attn_output.weight",),
                 lambda x: W[b + "attn_output.weight"][0] @ x), ffn]
    return [("z-%d" % L, "attn_norm-%d" % L, (b + "attn_gate.weight",), lambda x: W[b + "attn_gate.weight"][0] @ x),
            ("linear_attn_out-%d" % L, "final_output-%d" % L, (b + "ssm_out.weight",), lambda x: W[b + "ssm_out.weight"][0] @ x),
            ffn]


def load_layer_types(lt_path):
    with open(lt_path) as f:
        next(f)
        return {int(p[0]): p[1] for p in (ln.rstrip("\n").split("\t") for ln in f)}


def check_rows(W, lt, label, pos, L, dirs):
    for outn, inn, wnames, fn in checks(W, L, lt[L] == "full_attention"):
        for eng, d in dirs:
            ref = fn(read_f32("%s/pos%d/%s.f32" % (d, pos, inn)))
            rel, cos = score(ref, read_f32("%s/pos%d/%s.f32" % (d, pos, outn)))
            print("%s\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%.6f\t%.6f" % (
                label, pos, L, lt[L], outn, ",".join(w.split(".", 2)[2] for w in wnames),
                ",".join(W[w][1] for w in wnames), eng, rel, cos))


def main():
    model, gguf_py, lt_path, layers = sys.argv[1:5]
    layers = [int(x) for x in layers.split(",")]
    lt = load_layer_types(lt_path)
    W = load_weights(model, gguf_py, layers)
    print("\t".join(["label", "pos", "layer", "type", "output", "weights", "qtypes", "engine", "rel_l2_vs_ref", "cos_vs_ref"]))
    for spec in sys.argv[5:]:
        label, ldir, adir, poslist = spec.split(":")
        for pos in [int(p) for p in poslist.split(",")]:
            for L in layers:
                check_rows(W, lt, label, pos, L, (("llama", ldir), ("apr", adir)))


if __name__ == "__main__":
    sys.exit(main())
