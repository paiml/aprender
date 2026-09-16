#!/usr/bin/env python3
"""PMAT-3091 kvconfig tables from compare_kvconfig.sh outputs. Usage: tables.py CMP_DIR ON_VS_A_SUBLAYER_P4_TSV
Logits: per (prompt, apr mode, reference config) the compare_raw_logits.py summary (min/mean/median cosine, n<0.98,
argmax mismatches) and logits_gap.py's Frobenius rel L2; gap removed = 1 - frob(mode, cfg) / frob(off, A).
Sub-layer (p4 pos 1): the residual-stream steps from layer_steps.py for ON vs C, and the first point in forward
order whose printed rel_l2 (6 decimals, the comparator's own print precision) is non-zero, for ON vs C and ON vs A.
No threshold anywhere. Markdown on stdout, JSON to CMP_DIR/tables.json."""
import json
import sys

O, ON_A = sys.argv[1], sys.argv[2]
P, M, C = ["orig", "p1", "p2", "p3", "p4"], ["off", "on"], ["A", "B", "C"]


def summary(path):
    s = {}
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if f[0] == "summary":
            s[f[1]] = f[2]
    return s


def frob(p, c):
    out = {}
    for line in open(f"{O}/gap-{p}-vs-{c}.tsv"):
        f = line.rstrip("\n").split("\t")
        if f[0] == p:
            out[f[1]] = float(f[2])
    return out


res = {"logits": [], "gap_removed_vs_offA": {}}
print("## logits (apr vs llama per-token reference)\n")
print("| prompt | apr | ref | min cos | mean cos | median cos | n<0.98 | argmax mismatches | Frobenius rel L2 | gap removed vs OFF-vs-A |")
print("|---|---|---|---|---|---|---|---|---|---|")
for p in P:
    base = frob(p, "A")["off"]
    row = {"prompt": p}
    for m in M:
        row[f"{m}_vs"] = {}
        for c in C:
            s = summary(f"{O}/logits-{p}-{m}-vs-{c}.tsv")
            fr = frob(p, c)[m]
            gr = 1.0 - fr / base
            row[f"{m}_vs"][c] = {"min_cos": float(s["min_cosine"]), "mean_cos": float(s["mean_cosine"]),
                                 "median_cos": float(s["median_cosine"]), "n_below_0_98": int(s["n_below_0_98"]),
                                 "argmax": int(s["n_argmax_mismatches"]), "frob": fr, "gap_removed_vs_offA": round(gr, 6)}
            res["gap_removed_vs_offA"].setdefault(f"{m}{c}", []).append(round(gr, 6))
            print(f"| {p} | {m.upper()} | {c} | {s['min_cosine']} | {s['mean_cosine']} | {s['median_cosine']} | {s['n_below_0_98']} | {s['n_argmax_mismatches']} | {fr:.6f} | {100 * gr:.1f}% |")
    res["logits"].append(row)


def first_nonzero(path, label, pos):
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if f[0] == label and f[1] == pos and f[11] == "measured" and float(f[10]) > 0.0:
            return f"{f[3]} (layer {f[4]} {f[5]}) rel_l2 {f[10]}"
    return None


def steps(path, pos="1"):
    rows = []
    for line in open(path):
        if line.startswith("#") or line.startswith("label"):
            continue
        f = line.rstrip("\n").split("\t")
        if f[0] == "p4" and f[1] == pos:
            rows.append(f)
    return rows


print("\n## sub-layer, p4 pos 1: apr ON vs config C (residual stream steps)\n")
print("| point | layer | type | rel before | rel after | step | sub-layer out | sub rel |")
print("|---|---|---|---|---|---|---|---|")
st = steps(f"{O}/layer_steps_on_vs_C.tsv")
for f in st:
    print(f"| {f[2]} | {f[3]} | {f[4]} | {f[6]} | {f[7]} | {f[8]} | {f[10]} | {f[11]} |")
big = max(st, key=lambda f: float(f[8]))
off_st = steps(f"{O}/layer_steps_off_vs_C.tsv")
off_big = max(off_st, key=lambda f: float(f[8]))
res["residual_under_C"] = {
    "largest_step_on_vs_C_p4_pos1": f"{big[2]} (layer {big[3]} {big[4]}, {big[5]}) {big[6]} -> {big[7]} step {big[8]}",
    "largest_step_off_vs_C_p4_pos1": f"{off_big[2]} (layer {off_big[3]} {off_big[4]}, {off_big[5]}) step {off_big[8]}",
    "first_nonzero_on_vs_C_p4_pos1": first_nonzero(f"{O}/sublayer_p4_on_vs_C.tsv", "p4", "1"),
    "first_nonzero_on_vs_A_p4_pos1": first_nonzero(ON_A, "p4", "1"),
    "first_nonzero_on_vs_C_p4_pos0": first_nonzero(f"{O}/sublayer_p4_on_vs_C.tsv", "p4", "0"),
    "first_nonzero_on_vs_A_p4_pos0": first_nonzero(ON_A, "p4", "0"),
    "final_rel_on_vs_C_p4_pos1": st[-1][7],
}
print("\n```\n" + json.dumps(res["residual_under_C"], indent=1) + "\n```")
for line in open(f"{O}/layer_steps_on_vs_C.tsv"):
    if line.startswith("# largest step"):
        print(line.rstrip())
json.dump(res, open(f"{O}/tables.json", "w"), indent=1)
