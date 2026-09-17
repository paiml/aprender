#!/usr/bin/env python3
"""PMAT-3091 scalar: full-precision walk of every dumped point, llama callback order, apr(scalar) vs llama-scalar-C.
usage: walk_points.py <label> <llama_dump_dir> <apr_dump_dir>   -> TSV on stdout, summary on stderr. No threshold judged:
rel_l2 = |a-l|/|l|, bit_equal = element count with identical f32 bits."""
import sys, numpy as np


def apr_name(name, layer):
    if layer == "-1" or name.endswith(f"-{layer}"):
        return name
    return f"{name}-{layer}"


def manifest(d, llama):
    rows = [l.rstrip("\n").split("\t") for l in open(f"{d}/manifest.tsv")][1:]
    if llama:
        return [(r[0], int(r[1]), int(r[2]), r[5]) for r in rows]
    return [(apr_name(r[0], r[1]), int(r[1]), int(r[2]), r[4]) for r in rows]


def fmt4(v):
    return ",".join(f"{x:.9g}" for x in v[:4])


def measure(ld, ad, f, af):
    l = np.fromfile(f"{ld}/{f}", dtype=np.float32)
    a = np.fromfile(f"{ad}/{af}", dtype=np.float32)
    if l.size != a.size:
        return l, a, None
    be = int((l.view(np.uint32) == a.view(np.uint32)).sum())
    nl = float(np.linalg.norm(l.astype(np.float64)))
    rel = float(np.linalg.norm(a.astype(np.float64) - l) / nl) if nl else float(np.linalg.norm(a))
    mad = float(np.max(np.abs(a.astype(np.float64) - l)))
    return l, a, (be, rel, mad)


def main():
    label, ld, ad = sys.argv[1:4]
    apr = {(n, p): f for n, _, p, f in manifest(ad, False)}
    rows = manifest(ld, True)
    print("label\tpos\tidx\ttensor\tlayer\tn\tbit_equal\trel_l2\tmax_abs\tllama_first4\tapr_first4")
    first, mx, missing = None, (0.0, None), 0
    for idx, (n, il, p, f) in enumerate(rows):
        if (n, p) not in apr:
            missing += 1
            continue
        l, a, m = measure(ld, ad, f, apr[(n, p)])
        if m is None:
            print(f"{label}\t{p}\t{idx}\t{n}\t{il}\tSIZE {l.size} vs {a.size}")
            continue
        be, rel, mad = m
        print(f"{label}\t{p}\t{idx}\t{n}\t{il}\t{l.size}\t{be}\t{rel:.6e}\t{mad:.6e}\t{fmt4(l)}\t{fmt4(a)}")
        mx = max(mx, (rel, f"pos{p} {n}"), key=lambda t: t[0])
        if be != l.size and first is None:
            first = f"pos{p} idx{idx} {n} bit_equal={be}/{l.size} rel_l2={rel:.3e}"
    print(f"# {label}: matched={len(rows) - missing} missing_in_apr={missing} max_rel_l2={mx[0]:.6e} at {mx[1]}; first_not_bit_equal={first}", file=sys.stderr)


if __name__ == "__main__":
    main()
