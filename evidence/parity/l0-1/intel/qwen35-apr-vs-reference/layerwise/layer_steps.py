#!/usr/bin/env python3
"""PMAT-3091 layer observer: relative-L2 steps along the residual stream, from compare_layerwise.py TSVs.

Usage: layer_steps.py [--ref LABEL:POS] CURVES_TSV [CURVES_TSV ...]
       (rows: label pos token_id tensor layer layer_type ... rel_l2 status)

Residual stream, per (label, pos): model.input_embed, then for each layer N attn_residual-N (after the
mixer add) and l_out-N (after the FFN add), then result_norm. The step at a point is
rel_l2(point) - rel_l2(previous point); a mixer step is charged to attn_residual-N, an FFN step to l_out-N.
Each step row also carries the rel_l2 of the sub-layer's own output (linear_attn_out-N / attn_output-N for the
mixer, ffn_out-N for the FFN), which is context, not part of the step.
Output: a TSV of every step on stdout, then '#'-prefixed summary lines: the largest step per (label, pos);
for the largest step of the REF (label, pos) (default: the first one read), the step and its rank at every other
(label, pos); and the 8 points where the REF step most exceeds the LARGEST step any other (label, pos) takes at
that point (excess = ref step - max other step).
No threshold anywhere.
"""
import sys


def load(paths):
    rows = {}
    for path in paths:
        with open(path) as f:
            hdr = next(f).rstrip("\n").split("\t")
            for ln in f:
                r = dict(zip(hdr, ln.rstrip("\n").split("\t")))
                if r["status"] == "measured":
                    rows[(r["label"], int(r["pos"]), r["tensor"])] = r
    return rows


def keys_in_order(rows):
    seen = []
    for (label, pos, _t) in rows:
        if (label, pos) not in seen:
            seen.append((label, pos))
    return seen


def stream(rows, label, pos):
    pts = [("model.input_embed", -1, "-", "embed", None)]
    n = 0
    while (label, pos, "l_out-%d" % n) in rows:
        lt = rows[(label, pos, "l_out-%d" % n)]["layer_type"]
        mixer = "linear_attn_out-%d" % n if "deltanet" in lt else "attn_output-%d" % n
        pts.append(("attn_residual-%d" % n, n, lt, "mixer", mixer))
        pts.append(("l_out-%d" % n, n, lt, "ffn", "ffn_out-%d" % n))
        n += 1
    pts.append(("result_norm", -1, "-", "final_norm", None))
    return pts


def steps(rows, label, pos):
    out = []
    prev = None
    for tname, layer, lt, point, sub in stream(rows, label, pos):
        r = rows.get((label, pos, tname))
        if r is None:
            prev = None
            continue
        rel = float(r["rel_l2"])
        if prev is not None:
            s = rows.get((label, pos, sub)) if sub else None
            out.append({"label": label, "pos": pos, "tensor": tname, "layer": layer, "type": lt, "point": point,
                        "rel_before": prev, "rel_after": rel, "step": rel - prev, "cos": float(r["cos"]),
                        "sub": sub or "-", "sub_rel": s["rel_l2"] if s else "-", "sub_cos": s["cos"] if s else "-"})
        prev = rel
    return out


def excess_lines(table, ref):
    others = [k for k in table if k != ref]
    by_point = {k: {s["tensor"]: s["step"] for s in table[k]} for k in table}
    ex = []
    for s in table[ref]:
        o = [by_point[k][s["tensor"]] for k in others if s["tensor"] in by_point[k]]
        if o:
            ex.append((s["step"] - max(o), s, max(o)))
    ex.sort(key=lambda e: -e[0])
    for e, s, mo in ex[:8]:
        print("# excess %s pos %d at %s (layer %d %s, %s): ref step %.6f, max other step %.6f, excess %.6f" % (
            ref[0], ref[1], s["tensor"], s["layer"], s["type"], s["point"], s["step"], mo, e))


COLS = ["label", "pos", "tensor", "layer", "type", "point", "rel_before", "rel_after", "step", "cos",
        "sub", "sub_rel", "sub_cos"]


def parse_args(args):
    if args[:1] == ["--ref"]:
        lab, _, p = args[1].partition(":")
        return (lab, int(p)), args[2:]
    return None, args


def print_table(rows):
    print("\t".join(COLS))
    table = {}
    for label, pos in keys_in_order(rows):
        table[(label, pos)] = steps(rows, label, pos)
        for s in table[(label, pos)]:
            print("\t".join("%.6f" % s[c] if isinstance(s[c], float) else str(s[c]) for c in COLS))
    return table


def largest_lines(table):
    for key, ss in table.items():
        best = max(ss, key=lambda s: s["step"])
        print("# largest step %s pos %d: %s (layer %d %s, %s) rel_l2 %.6f -> %.6f step %.6f" % (
            key[0], key[1], best["tensor"], best["layer"], best["type"], best["point"],
            best["rel_before"], best["rel_after"], best["step"]))


def rank_line(first, key, ss):
    ranked = sorted(ss, key=lambda s: -s["step"])
    idx = next((i for i, s in enumerate(ranked) if s["tensor"] == first["tensor"]), None)
    if idx is None:
        return
    s = ranked[idx]
    print("# at %s: %s pos %d step %.6f rank %d/%d (largest there: %s %.6f)" % (
        first["tensor"], key[0], key[1], s["step"], idx + 1, len(ranked), ranked[0]["tensor"], ranked[0]["step"]))


def main():
    ref, args = parse_args(sys.argv[1:])
    table = print_table(load(args))
    ref = ref or next(iter(table))
    largest_lines(table)
    first = max(table[ref], key=lambda s: s["step"])
    for key, ss in table.items():
        rank_line(first, key, ss)
    excess_lines(table, ref)


if __name__ == "__main__":
    main()
