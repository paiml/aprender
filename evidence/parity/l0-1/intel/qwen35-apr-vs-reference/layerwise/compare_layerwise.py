#!/usr/bin/env python3
"""PMAT-3091 layerwise comparator: llama.cpp eval-callback dumps vs apr, per (tensor, position).

Usage:
  compare_layerwise.py LABEL LLAMA_DUMP_DIR LLAMA_PER_TOKEN_BIN APR_EMBD_DIR APR_LOGITS_BIN \
                       POSITIONS LAYER_TYPES_TSV MODEL_GGUF GGUF_PY_DIR [APR_OBS_DIR]

Per row: cosine, max |apr - llama|, relative L2 ||apr - llama|| / ||llama||.
Without APR_OBS_DIR (the first layerwise pass): the apr side is the token-embedding row
(APR_EMBD_DIR/pos<P>/embd.f32) and the logits row (APRRAWLG file); l_out-<il> and result_norm carry
status U with only llama's norm.
With APR_OBS_DIR (the layer-observer pass): the tensor list is every name in LLAMA_DUMP_DIR/manifest.tsv
at that position, in dump order, and the apr side is APR_OBS_DIR/pos<P>/<name>.f32 as written by
qwen35_layer_obs (forward_single_qwen35_observed). A name apr did not write is status U.
The DeltaNet state tensors (state_predelta, new_state) are also scored with apr's [h][j][i] memory order
transposed to [h][i][j]; both are reported, in the note.
Extra columns on the embd row: the same row dequantized by gguf-py (a third Q6_K implementation).
Self-check: the llama callback's result_output at pos P must equal row P of LLAMA_PER_TOKEN_BIN.
No threshold anywhere. Output TSV on stdout.
"""
import os
import struct
import sys

import numpy as np

HDR = ["label", "pos", "token_id", "tensor", "layer", "layer_type", "llama_l2", "apr_l2",
       "cos", "max_abs_diff", "rel_l2", "status", "note"]
STATE_NAMES = ("state_predelta", "new_state")


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
    ns = float(np.linalg.norm(sub))
    cos = float(np.dot(ref, sub) / (nr * ns)) if nr > 0 and ns > 0 else float("nan")
    rel = float(np.linalg.norm(d)) / nr if nr > 0 else float("nan")
    return cos, float(np.max(np.abs(d))), rel


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


def layer_of(tname):
    head, _, tail = tname.rpartition("-")
    return int(tail) if head and tail.isdigit() else -1


def tensor_list(n_layer, ldir, pos, obs_dir):
    if obs_dir is None:
        return ["model.input_embed"] + ["l_out-%d" % i for i in range(n_layer)] + ["result_norm", "result_output"]
    names = []
    with open(os.path.join(ldir, "manifest.tsv")) as f:
        next(f)
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if int(p[2]) == pos:
                names.append(p[0])
    return names


def apr_vector(tname, pos, apr_embd_dir, apr_rows, obs_dir):
    if tname == "result_output":
        return apr_rows[pos].astype(np.float64)
    if obs_dir is not None:
        path = "%s/pos%d/%s.f32" % (obs_dir, pos, tname)
        return read_f32(path) if os.path.exists(path) else None
    if tname == "model.input_embed":
        return read_f32("%s/pos%d/embd.f32" % (apr_embd_dir, pos))
    return None


def embd_note(gg, ids, pos, ref, sub):
    qname, rows = gg
    g = rows[int(ids[pos])]
    gc, gm, gr = metrics(ref, g)
    ac, am, ar = metrics(g, sub)
    return "token_embd=%s; ggufpy-vs-llama cos=%.9f max_abs=%.3e rel_l2=%.3e; apr-vs-ggufpy cos=%.9f max_abs=%.3e rel_l2=%.3e" % (
        qname, gc, gm, gr, ac, am, ar)


def state_note(ref, sub):
    n = int(round((ref.size / 16) ** 0.5))
    if n * n * 16 != ref.size or sub.size != ref.size:
        return "state size %d not 16*n*n" % ref.size
    t = sub.reshape(16, n, n).transpose(0, 2, 1).reshape(-1)
    c, m, r = metrics(ref, t)
    return "apr [h][j][i] transposed to [h][i][j]: cos=%.6f max_abs=%.6e rel_l2=%.6f" % (c, m, r)


def row_for(ctx, pos, tname):
    label, ldir, lt, apr_embd_dir, apr_rows, ids, gg, obs_dir = ctx
    ref = read_f32("%s/pos%d/%s.f32" % (ldir, pos, tname))
    layer = layer_of(tname)
    base = [label, pos, int(ids[pos]), tname, layer, lt.get(layer, "-"), float(np.linalg.norm(ref))]
    sub = apr_vector(tname, pos, apr_embd_dir, apr_rows, obs_dir)
    if sub is None or sub.size != ref.size:
        why = "apr hidden state not reachable via public API" if sub is None else "size apr %d vs llama %d" % (sub.size, ref.size)
        emit(base + ["U", "U", "U", "U", "U", why])
        return
    cos, mx, rel = metrics(ref, sub)
    note = ""
    if tname == "model.input_embed":
        note = embd_note(gg, ids, pos, ref, sub)
    elif tname.rpartition("-")[0] in STATE_NAMES:
        note = state_note(ref, sub)
    emit(base + [float(np.linalg.norm(sub)), cos, mx, rel, "measured", note])


def self_check(ldir, pos, ref_rows):
    cb = read_f32("%s/pos%d/result_output.f32" % (ldir, pos))
    same = bool(np.array_equal(cb.astype(np.float32), ref_rows[pos]))
    print("# self-check pos %d: callback result_output == per-token bin row: %s" % (pos, same), file=sys.stderr)
    return same


def load_logits(lbin, apr_bin):
    lids, lrows = read_rawlogits(lbin)
    aids, arows = read_rawlogits(apr_bin)
    if not np.array_equal(lids, aids):
        print("token ids differ between llama and apr files", file=sys.stderr)
        return None
    return lids, lrows, arows


def main():
    label, ldir, lbin, apr_embd_dir, apr_bin, positions, lt_path, model, gguf_py = sys.argv[1:10]
    obs_dir = sys.argv[10] if len(sys.argv) > 10 else None
    pos_list = [int(p) for p in positions.split(",")]
    loaded = load_logits(lbin, apr_bin)
    if loaded is None:
        return 2
    lids, lrows, arows = loaded
    lt = layer_types(lt_path)
    gg = gguf_embd_rows(model, gguf_py, sorted({int(lids[p]) for p in pos_list}))
    ctx = (label, ldir, lt, apr_embd_dir, arows, lids, gg, obs_dir)
    return emit_all(ctx, pos_list, lrows)


def emit_all(ctx, pos_list, lrows):
    ldir, lt, obs_dir = ctx[1], ctx[2], ctx[7]
    emit(HDR)
    ok = all([self_check(ldir, p, lrows) for p in pos_list])
    for p in pos_list:
        for tname in tensor_list(len(lt), ldir, p, obs_dir):
            row_for(ctx, p, tname)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
