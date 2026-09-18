#!/usr/bin/env python3
"""PMAT-3091 emulation residual: is the first switch-ON departure (layer-3 attention) llama's f16 KV cache?

Usage: kv_f16_check.py LLAMA_DUMP_DIR APR_DUMP_ON APR_DUMP_OFF POSLIST LAYERLIST
llama.cpp d1d3c3396 stores K and V as f16 (`llama_kv_cache ... K (f16), V (f16)`, Flash Attention enabled; see
llama_repack_probe.excerpt.log). At position 0 a single key has softmax weight 1, so attn_pregate-N is V_0 as read
back from the cache. Per (mode, pos, layer): rel L2 of apr's attn_norm-N input vs llama, rel L2 of attn_pregate-N vs
llama, the same after rounding apr's pregate to f16, and how many elements are then bit-equal. No threshold.
"""
import sys

import numpy as np


def rd(path):
    return np.fromfile(path, dtype="<f4")


def rel(a, b):
    return float(np.linalg.norm(a.astype(np.float64) - b) / np.linalg.norm(b.astype(np.float64)))


def main(argv):
    if len(argv) != 6:
        print(__doc__, file=sys.stderr)
        return 2
    llama, on_dir, off_dir = argv[1:4]
    positions = [int(p) for p in argv[4].split(",")]
    layers = [int(x) for x in argv[5].split(",")]
    print("mode\tpos\tlayer\tattn_norm_in_rel\tpregate_rel\tpregate_rel_after_f16_round_of_apr\tbit_equal_after_f16_round\tn")
    for mode, adir in (("on", on_dir), ("off", off_dir)):
        for pos in positions:
            for ly in layers:
                ll = rd(f"{llama}/pos{pos}/attn_pregate-{ly}.f32")
                ap = rd(f"{adir}/pos{pos}/attn_pregate-{ly}.f32")
                an = rel(rd(f"{adir}/pos{pos}/attn_norm-{ly}.f32"), rd(f"{llama}/pos{pos}/attn_norm-{ly}.f32"))
                r16 = ap.astype(np.float16).astype(np.float32)
                print(f"{mode}\t{pos}\t{ly}\t{an:.6f}\t{rel(ap, ll):.6f}\t{rel(r16, ll):.6f}\t{int((r16 == ll).sum())}\t{ll.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
