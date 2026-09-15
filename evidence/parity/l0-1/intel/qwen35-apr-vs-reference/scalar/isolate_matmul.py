#!/usr/bin/env python3
"""PMAT-3091 scalar kernel isolation: for every DeltaNet layer at the dumped positions, push (a) llama-scalar-C's OWN
dumped input and (b) apr(scalar)'s own dumped input through apr's scalar-emulated matmul (scalar_isolate), for the two
pure matmuls in the observer: attn_norm -> z (attn_gate) and final_output -> linear_attn_out (ssm_out).
Reports bit-equal counts of kernel(llama_in) vs llama_out and kernel(apr_in) vs apr_out, and the amplification
rel_l2(kernel(apr_in), kernel(llama_in)) / rel_l2(apr_in, llama_in). No threshold."""
import subprocess, sys, numpy as np

W = "/tmp/scalar3091"
ISO = f"{W}/iso"
SETS = [("p4", f"{W}/ref/SC-sub-p4", f"{W}/runs/scalar-dump-p4", [0, 1, 2, 3]),
        ("orig", f"{W}/ref/SC-sub-orig", f"{W}/runs/scalar-dump-orig", [4, 28])]
PAIRS = [("attn_norm", "z", "attn_gate"), ("final_output", "linear_attn_out", "ssm_out")]


def f32(path):
    return np.fromfile(path, dtype=np.float32)


def rel(a, b):
    return float(np.linalg.norm(a.astype(np.float64) - b) / np.linalg.norm(b.astype(np.float64)))


def bits_equal(a, b):
    return int((a.view(np.uint32) == b.view(np.uint32)).sum())


def cases(wts):
    layers = sorted({int(k.split(".")[1]) for k in wts})
    return [(lab, ld, ad, p, il, pair) for lab, ld, ad, poss in SETS for p in poss for il in layers for pair in PAIRS]


def jobs_for(wts, case):
    lab, ld, ad, p, il, (src, dst, w) = case
    _, _, qt, shape, wf = wts[f"blk.{il}.{w}.weight"]
    i, o = shape.split(",")
    return [f"matmul\t{wf}\t{qt}\t{i}\t{o}\t{d}/pos{p}/{src}-{il}.f32\t{ISO}/{lab}-p{p}-{dst}-{il}-{eng}.f32"
            for eng, d in (("llama", ld), ("apr", ad))]


def row(wts, case):
    lab, ld, ad, p, il, (src, dst, w) = case
    li, ai = f32(f"{ld}/pos{p}/{src}-{il}.f32"), f32(f"{ad}/pos{p}/{src}-{il}.f32")
    lo, ao = f32(f"{ld}/pos{p}/{dst}-{il}.f32"), f32(f"{ad}/pos{p}/{dst}-{il}.f32")
    kl, ka = f32(f"{ISO}/{lab}-p{p}-{dst}-{il}-llama.f32"), f32(f"{ISO}/{lab}-p{p}-{dst}-{il}-apr.f32")
    ir, kr = rel(ai, li), rel(ka, kl)
    amp = kr / ir if ir else float("nan")
    counts = (bits_equal(kl, lo), lo.size, bits_equal(ka, ao), ao.size)
    line = (f"{lab}\t{p}\t{il}\t{src}->{dst}\t{wts[f'blk.{il}.{w}.weight'][1]}\t{lo.size}\t{counts[0]}\t{counts[2]}"
            f"\t{ir:.3e}\t{kr:.3e}\t{amp:.1f}\t{rel(ao, lo):.3e}")
    return line, counts


def main():
    wts = {l.split("\t")[0]: l.rstrip("\n").split("\t") for l in open(f"{ISO}/weights.tsv")}
    cs = cases(wts)
    open(f"{ISO}/jobs.tsv", "w").write("\n".join(j for c in cs for j in jobs_for(wts, c)) + "\n")
    r = subprocess.run([sys.argv[1], f"{ISO}/jobs.tsv"], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=1800)
    print(f"# {r.stdout.strip()} rc={r.returncode} {r.stderr.strip()[:300]}", file=sys.stderr)
    print("label\tpos\tlayer\tkernel\tqtype\tn\tK(llama_in)==llama_out_bits\tK(apr_in)==apr_out_bits\tin_rel_l2\tK_out_rel_l2\tamplification\tout_rel_l2_dumps")
    tot = np.zeros(4, dtype=np.int64)
    for c in cs:
        line, counts = row(wts, c)
        print(line)
        tot += counts
    print(f"# totals: K(llama_in)==llama_out {tot[0]}/{tot[1]} elements; K(apr_in)==apr_out {tot[2]}/{tot[3]}", file=sys.stderr)


if __name__ == "__main__":
    main()
