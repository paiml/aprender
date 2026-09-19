#!/usr/bin/env python3
"""PMAT-3091 emulation: markdown tables for EMULATION.md from compare_emulation.sh outputs.

Usage: tables.py CMP_DIR
Reads logits-<prompt>-<variant>.json, gap-<prompt>.tsv, sublayer_{p4,orig}_{off,on}.tsv,
kernel_isolation_{off,on}.tsv. Prints markdown on stdout. Numbers only; no threshold, no verdict.
"""
import csv
import json
import os
import sys

PROMPTS = ["orig", "p1", "p2", "p3", "p4"]
VARIANTS = ["off", "on", "only-Q4_K", "only-Q5_K", "only-Q6_K", "only-Q8_0"]
SITES = [("p4", 0), ("p4", 1), ("p4", 2), ("p4", 3), ("orig", 4), ("orig", 28)]


def read_tsv(path):
    with open(path) as f:
        return [r for r in csv.DictReader((ln for ln in f if not ln.startswith("#")), delimiter="\t")]


def logits_table(d):
    print("| prompt | variant | min cos (pos) | mean cos | median cos | n < 0.98 | argmax mismatches | max abs diff |")
    print("|---|---|---|---|---|---|---|---|")
    for p in PROMPTS:
        for v in VARIANTS:
            s = json.load(open(os.path.join(d, f"logits-{p}-{v}.json")))["summary"]
            print(f"| {p} | {v} | {s['min_cosine']:.6f} ({s['min_cosine_pos']}) | {s['mean_cosine']:.6f} | "
                  f"{s['median_cosine']:.6f} | {s['n_below_0_98']} | {s['n_argmax_mismatches']} | {s['max_abs_diff']:.6f} |")


def gap_table(d):
    print("| prompt | variant | Frobenius rel L2 | mean per-pos rel L2 | median per-pos rel L2 | gap removed (Frob) | gap removed (mean) | per-pos rel L2 |")
    print("|---|---|---|---|---|---|---|---|")
    for p in PROMPTS:
        for r in read_tsv(os.path.join(d, f"gap-{p}.tsv")):
            print(f"| {r['prompt']} | {r['variant']} | {r['frob_rel_l2']} | {r['mean_pos_rel_l2']} | {r['median_pos_rel_l2']} | "
                  f"{r['gap_removed_frac_frob']} | {r['gap_removed_frac_mean']} | {r['pos_rel_l2']} |")


def sublayer_index(d):
    idx = {}
    for mode in ("off", "on"):
        for label in ("p4", "orig"):
            for r in read_tsv(os.path.join(d, f"sublayer_{label}_{mode}.tsv")):
                idx[(mode, r["label"], int(r["pos"]), r["tensor"])] = r
    return idx


def layer0_table(idx):
    names = ["final_output-0", "linear_attn_out-0", "attn_residual-0", "l_out-0"]
    print("| label pos | " + " | ".join(f"{n} OFF | {n} ON" for n in names) + " |")
    print("|---|" + "---|" * (2 * len(names)))
    for label, pos in SITES:
        cells = [f"{idx[(m, label, pos, n)]['rel_l2']}" for n in names for m in ("off", "on")]
        print(f"| {label} {pos} | " + " | ".join(cells) + " |")


def curve_table(idx, label, pos):
    print(f"| point ({label} pos {pos}) | rel L2 OFF | rel L2 ON | cos OFF | cos ON |")
    print("|---|---|---|---|---|")
    points = ["model.input_embed"] + [f"{k}-{n}" for n in range(24) for k in ("attn_residual", "l_out")] + ["result_norm", "result_output"]
    for t in points:
        a, b = idx.get(("off", label, pos, t)), idx.get(("on", label, pos, t))
        if a and b:
            print(f"| {t} | {a['rel_l2']} | {b['rel_l2']} | {a['cos']} | {b['cos']} |")


def kernel_summary(d):
    print("| mode | output | qtypes | engine | rows | rel L2 vs float64 dequant min | max |")
    print("|---|---|---|---|---|---|---|")
    for mode in ("off", "on"):
        groups = {}
        for r in read_tsv(os.path.join(d, f"kernel_isolation_{mode}.tsv")):
            groups.setdefault((r["output"].rsplit("-", 1)[0], r["qtypes"], r["engine"]), []).append(float(r["rel_l2_vs_ref"]))
        for (out, qt, eng), vals in sorted(groups.items()):
            print(f"| {mode} | {out} | {qt} | {eng} | {len(vals)} | {min(vals):.6f} | {max(vals):.6f} |")


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    d = argv[1]
    idx = sublayer_index(d)
    for title, fn in (("logits", lambda: logits_table(d)), ("gap", lambda: gap_table(d)),
                      ("layer0", lambda: layer0_table(idx)), ("curve p4 1", lambda: curve_table(idx, "p4", 1)),
                      ("kernel", lambda: kernel_summary(d))):
        print(f"\n### {title}\n")
        fn()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
