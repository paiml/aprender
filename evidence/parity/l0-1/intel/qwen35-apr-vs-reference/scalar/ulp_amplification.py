#!/usr/bin/env python3
"""PMAT-3091: the amplification FLOOR of one Q8_K-quantized matmul. Both engines run the identical kernel; this feeds
it the SAME f32 activation row, then the same row with ONE element moved by exactly 1 ulp, and reports how far the
OUTPUT moved. Answers: is ~1e-6 rel L2 reachable at all while llama quantizes activations to Q8_K between layers?
usage: ulp_amplification.py <scalar_isolate> <out.tsv>   No threshold judged."""
import subprocess, sys, numpy as np

W = "/tmp/scalar3091"
ISO = f"{W}/iso"
OUT = "/mnt/nvme-raid0/parity-tmp/scalar3091-it5plus/ulp"
INPUT = f"{W}/ref/SC-sub-p4/pos0/attn_norm-0.f32"
WEIGHTS = ["blk.0.attn_qkv.weight", "blk.0.attn_gate.weight"]


def rel_l2(a, b):
    d = np.linalg.norm(a.astype(np.float64) - b.astype(np.float64))
    n = np.linalg.norm(b.astype(np.float64))
    return float(d / n) if n else float(d)


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    wts = {l.split("\t")[0]: l.rstrip("\n").split("\t") for l in open(f"{ISO}/weights.tsv")}
    x = np.fromfile(INPUT, dtype=np.float32)
    idxs = [0, 1, 2, 3, 100, 511, 1023, int(np.argmax(np.abs(x))), int(np.argmin(np.abs(x)))]
    jobs = []
    for wn in WEIGHTS:
        _, _, qt, shape, wf = wts[wn]
        i_dim, o_dim = shape.split(",")
        for e in idxs:
            jobs.append(f"ulpamp\t{wf}\t{qt}\t{i_dim}\t{o_dim}\t{INPUT}\t{e}\t{OUT}/{wn}-{e}-base.f32\t{OUT}/{wn}-{e}-pert.f32")
    open(f"{OUT}/jobs_ulp.tsv", "w").write("\n".join(jobs) + "\n")
    r = subprocess.run([sys.argv[1], f"{OUT}/jobs_ulp.tsv"], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=1800)
    print(f"# scalar_isolate rc={r.returncode} {r.stdout.strip()} {r.stderr.strip()[:200]}", file=sys.stderr)
    rows = ["weight\tqtype\telem\tx_value\tin_rel_l2\tout_rel_l2\tamplification\tout_bit_equal\tn_out"]
    for wn in WEIGHTS:
        _, _, qt, shape, _ = wts[wn]
        for e in idxs:
            y0 = np.fromfile(f"{OUT}/{wn}-{e}-base.f32", dtype=np.float32)
            y1 = np.fromfile(f"{OUT}/{wn}-{e}-pert.f32", dtype=np.float32)
            xp = np.fromfile(f"{OUT}/{wn}-{e}-pert.f32.in", dtype=np.float32)
            ri, ro = rel_l2(xp, x), rel_l2(y1, y0)
            be = int((y0.view(np.uint32) == y1.view(np.uint32)).sum())
            amp = ro / ri if ri else float("nan")
            rows.append(f"{wn}\t{qt}\t{e}\t{x[e]:.9g}\t{ri:.6e}\t{ro:.6e}\t{amp:.3f}\t{be}\t{y0.size}")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
