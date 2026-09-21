#!/usr/bin/env python3
"""PMAT-3091: recover each engine's final-norm vector from its LOGITS through the tied lm_head.

Qwen3.5-0.8B has no output.weight; both engines use token_embd.weight (Q6_K) as lm_head, so
logits = W @ result_norm with W = dequant(token_embd) (248320 x 1024, full column rank).
Least squares h = argmin ||W h - logits|| recovers result_norm up to each engine's lm_head kernel error.
  * CONTROL: fit llama's own logits, compare to llama's DUMPED result_norm (validates the method).
  * SUBJECT: fit apr's logits -> apr's result_norm (derived, not dumped), compare to llama's dump.
  * l_out-23 direction: result_norm = rmsnorm(l_out-23) * output_norm.weight, so h / weight is
    l_out-23 up to a positive scale; only COSINE is meaningful for that row.
The relative fit residual ||W h - logits|| / ||logits|| is reported per fit.
Usage: derive_final_norm.py MODEL GGUF_PY LAYERWISE_TSV_OUT  LABEL:LLAMA_DUMP_DIR:APR_BIN:POSLIST ...
No threshold anywhere.
"""
import struct
import sys

import numpy as np


def load_w(model, gguf_py):
    sys.path.insert(0, gguf_py)
    import gguf  # noqa: E402
    from gguf.quants import dequantize  # noqa: E402
    r = gguf.GGUFReader(model)
    t = {x.name: x for x in r.tensors}
    emb = t["token_embd.weight"]
    n_embd, n_vocab = int(emb.shape[0]), int(emb.shape[1])
    w = dequantize(np.asarray(emb.data), emb.tensor_type).reshape(n_vocab, n_embd).astype(np.float64)
    norm_w = np.asarray(t["output_norm.weight"].data, dtype=np.float64).reshape(-1)
    return w, norm_w


def rawlogits_row(path, pos):
    with open(path, "rb") as f:
        assert f.read(8) == b"APRRAWLG"
        _v, n_pos, n_vocab = struct.unpack("<Iii", f.read(12))
        f.seek(20 + 4 * n_pos + 4 * pos * n_vocab)
        return np.frombuffer(f.read(4 * n_vocab), dtype="<f4").astype(np.float64)


def f32(path):
    return np.fromfile(path, dtype="<f4").astype(np.float64)


def cmp(ref, sub):
    d = sub - ref
    nr = np.linalg.norm(ref)
    return float(ref @ sub / (nr * np.linalg.norm(sub))), float(np.abs(d).max()), float(np.linalg.norm(d) / nr)


class Fitter:
    def __init__(self, w):
        self.w = w
        self.q, self.r = np.linalg.qr(w, mode="reduced")

    def fit(self, y):
        h = np.linalg.solve(self.r, self.q.T @ y)
        return h, float(np.linalg.norm(self.w @ h - y) / np.linalg.norm(y))


def rows_for(fitter, norm_w, label, ldir, apr_bin, pos):
    ll_logits = f32("%s/pos%d/result_output.f32" % (ldir, pos))
    ll_norm = f32("%s/pos%d/result_norm.f32" % (ldir, pos))
    ll_l23 = f32("%s/pos%d/l_out-23.f32" % (ldir, pos))
    h_ll, res_ll = fitter.fit(ll_logits)
    h_apr, res_apr = fitter.fit(rawlogits_row(apr_bin, pos))
    out = []
    c = cmp(ll_norm, h_ll)
    out.append([label, pos, "CONTROL llama_lstsq_result_norm vs llama_dump_result_norm", *c, res_ll])
    c = cmp(ll_norm, h_apr)
    out.append([label, pos, "SUBJECT apr_lstsq_result_norm vs llama_dump_result_norm", *c, res_apr])
    c = cmp(ll_l23, h_ll / norm_w)
    out.append([label, pos, "CONTROL llama_lstsq_l_out-23_direction vs llama_dump_l_out-23 (cos only)", c[0], "U", "U", res_ll])
    c = cmp(ll_l23, h_apr / norm_w)
    out.append([label, pos, "SUBJECT apr_lstsq_l_out-23_direction vs llama_dump_l_out-23 (cos only)", c[0], "U", "U", res_apr])
    return out


def main():
    model, gguf_py, out_tsv = sys.argv[1:4]
    w, norm_w = load_w(model, gguf_py)
    fitter = Fitter(w)
    lines = ["label\tpos\trow\tcos\tmax_abs_diff\trel_l2\tfit_rel_residual"]
    for spec in sys.argv[4:]:
        label, ldir, apr_bin, plist = spec.split(":")
        for pos in (int(p) for p in plist.split(",")):
            for r in rows_for(fitter, norm_w, label, ldir, apr_bin, pos):
                lines.append("\t".join("%.6f" % x if isinstance(x, float) else str(x) for x in r))
    with open(out_tsv, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
