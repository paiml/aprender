#!/usr/bin/env python3
"""PMAT-3091 layerwise comparator: llama.cpp eval-callback dumps vs apr, per (tensor, position).

Usage:
  compare_layerwise.py LABEL LLAMA_DUMP_DIR LLAMA_PER_TOKEN_BIN APR_EMBD_DIR APR_LOGITS_BIN \
                       POSITIONS LAYER_TYPES_TSV MODEL_GGUF GGUF_PY_DIR

Per row: cosine, max |apr - llama|, relative L2 ||apr - llama|| / ||llama||.
apr side reachable through realizar's public API: the token-embedding row (APR_EMBD_DIR/pos<P>/embd.f32)
and the logits row (APRRAWLG file). l_out-<il> and result_norm are NOT reachable without editing
crates/, so those rows carry status U with only llama's norm.
Extra columns on the embd row: the same row dequantized by gguf-py (a third Q6_K implementation).
Self-check: the llama callback's result_output at pos P must equal row P of LLAMA_PER_TOKEN_BIN.
No threshold anywhere. Output TSV on stdout.
"""
import struct
import sys

import numpy as np

HDR = ["label", "pos", "token_id", "tensor", "layer", "layer_type", "llama_l2", "apr_l2",
       "cos", "max_abs_diff", "rel_l2", "status", "note"]


def read_f32(path):
    return np.fromfile(path, dtype="<f4").astype(np.float64)


def read_rawlogits(path):
    with open(path, "rb") as f:
        assert f.read(8) == b"APRRAWLG", path
        _ver, n_pos, n_vocab = struct.unpack("<Iii", f.read(12))
        ids = np.frombuffer(f.read(4 * n_pos), dtype="<i4")
        rows = np.frombuffer(f.read(4 * n_pos * n_vocab), dtype="<f4").reshape(n_pos, n_vocab)
    return ids, rows


def metrics(ref, sub):
    d = sub - ref
    nr = float(np.linalg.norm(ref))
    cos = float(np.dot(ref, sub) / (nr * float(np.linalg.norm(sub))))
    return cos, float(np.max(np.abs(d))), float(np.linalg.norm(d)) / nr


def fmt(x):
    return "%.6f" % x if isinstance(x, float) else str(x)


def emit(cols):
    print("\t".join(fmt(c) for c in cols))


def gguf_embd_rows(model, gguf_py, ids):
    sys.path.insert(0, gguf_py)
    import gguf  # noqa: E402
    from gguf.quants import dequantize  # noqa: E402
    r = gguf.GGUFReader(model)
    t = next(x for x in r.tensors if x.name == "token_embd.weight")
    n_embd = int(t.shape[0])
    raw = np.asarray(t.data).reshape(int(t.shape[1]), -1)  # one byte row per token
    rows = {i: dequantize(raw[i], t.tensor_type).reshape(-1)[:n_embd].astype(np.float64) for i in ids}
    return t.tensor_type.name, rows


def layer_types(path):
    with open(path) as f:
        next(f)
        return {int(p[0]): p[1] for p in (ln.rstrip("\n").split("\t") for ln in f)}


def tensor_list(n_layer):
    return ["model.input_embed"] + ["l_out-%d" % i for i in range(n_layer)] + ["result_norm", "result_output"]


def apr_vector(tname, pos, apr_embd_dir, apr_rows):
    if tname == "model.input_embed":
        return read_f32("%s/pos%d/embd.f32" % (apr_embd_dir, pos))
    if tname == "result_output":
        return apr_rows[pos].astype(np.float64)
    return None


def row_for(ctx, pos, tname):
    label, ldir, lt, apr_embd_dir, apr_rows, ids, gg = ctx
    ref = read_f32("%s/pos%d/%s.f32" % (ldir, pos, tname))
    layer = int(tname.rsplit("-", 1)[1]) if tname.startswith("l_out-") else -1
    ltype = lt.get(layer, "-")
    sub = apr_vector(tname, pos, apr_embd_dir, apr_rows)
    base = [label, pos, int(ids[pos]), tname, layer, ltype, float(np.linalg.norm(ref))]
    if sub is None:
        emit(base + ["U", "U", "U", "U", "U", "apr hidden state not reachable via public API"])
        return
    cos, mx, rel = metrics(ref, sub)
    note = ""
    if tname == "model.input_embed":
        qname, rows = gg
        g = rows[int(ids[pos])]
        gc, gm, gr = metrics(ref, g)
        ac, am, ar = metrics(g, sub)
        note = "token_embd=%s; ggufpy-vs-llama cos=%.9f max_abs=%.3e rel_l2=%.3e; apr-vs-ggufpy cos=%.9f max_abs=%.3e rel_l2=%.3e" % (
            qname, gc, gm, gr, ac, am, ar)
    emit(base + [float(np.linalg.norm(sub)), cos, mx, rel, "measured", note])


def self_check(ldir, pos, ref_rows):
    cb = read_f32("%s/pos%d/result_output.f32" % (ldir, pos))
    same = bool(np.array_equal(cb.astype(np.float32), ref_rows[pos]))
    print("# self-check pos %d: callback result_output == per-token bin row: %s" % (pos, same), file=sys.stderr)
    return same


def main():
    label, ldir, lbin, apr_embd_dir, apr_bin, positions, lt_path, model, gguf_py = sys.argv[1:10]
    pos_list = [int(p) for p in positions.split(",")]
    lids, lrows = read_rawlogits(lbin)
    aids, arows = read_rawlogits(apr_bin)
    if not np.array_equal(lids, aids):
        print("token ids differ between llama and apr files", file=sys.stderr)
        return 2
    lt = layer_types(lt_path)
    gg = gguf_embd_rows(model, gguf_py, sorted({int(lids[p]) for p in pos_list}))
    ctx = (label, ldir, lt, apr_embd_dir, arows, lids, gg)
    emit(HDR)
    ok = all([self_check(ldir, p, lrows) for p in pos_list])
    for p in pos_list:
        for tname in tensor_list(len(lt)):
            row_for(ctx, p, tname)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
