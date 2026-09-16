#!/usr/bin/env python3
"""PMAT-3091 emulation: the apr-vs-llama logits gap with the ggml vec_dot emulation OFF, ON, and per qtype.

Usage: logits_gap.py PROMPT REF_BIN OFF_BIN VARIANT=BIN [VARIANT=BIN ...]
       (APRRAWLG files as compare_raw_logits.py reads them; token ids must match)

Per variant (off first): Frobenius rel L2 ||sub - ref||_F / ||ref||_F over all positions, the mean and median of the
per-position rel L2 ||sub_k - ref_k|| / ||ref_k||, and the fraction of the OFF gap removed:
1 - frob_rel_l2(variant) / frob_rel_l2(off) and 1 - mean_pos_rel_l2(variant) / mean_pos_rel_l2(off).
Also the per-position rel L2 at the positions listed in POSITIONS_ENV (default none). TSV on stdout. No threshold.
"""
import os
import sys

import numpy as np

MAGIC = b"APRRAWLG"


def load(path):
    buf = open(path, "rb").read()
    if buf[:8] != MAGIC:
        raise ValueError(f"{path}: bad magic")
    _, n_pos, n_vocab = (int(v) for v in np.frombuffer(buf, dtype="<i4", count=3, offset=8))
    off = 20 + 4 * n_pos
    if len(buf) != off + 4 * n_pos * n_vocab:
        raise ValueError(f"{path}: bad size")
    ids = np.frombuffer(buf, dtype="<i4", count=n_pos, offset=20)
    return ids, np.frombuffer(buf, dtype="<f4", offset=off).reshape(n_pos, n_vocab).astype(np.float64)


def gap(ref, sub):
    diff = sub - ref
    per_pos = np.linalg.norm(diff, axis=1) / np.linalg.norm(ref, axis=1)
    return float(np.linalg.norm(diff) / np.linalg.norm(ref)), per_pos


def row(prompt, name, ref, sub, base, positions):
    frob, per_pos = gap(ref, sub)
    base = base or (frob, float(per_pos.mean()))
    pos_txt = ",".join(f"{p}:{per_pos[p]:.6f}" for p in positions)
    line = (f"{prompt}\t{name}\t{frob:.6f}\t{per_pos.mean():.6f}\t{np.median(per_pos):.6f}"
            f"\t{1 - frob / base[0]:.6f}\t{1 - per_pos.mean() / base[1]:.6f}\t{pos_txt}")
    return line, base


def main(argv):
    if len(argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2
    prompt, ref_path, off_path = argv[1:4]
    variants = [("off", off_path)] + [tuple(a.split("=", 1)) for a in argv[4:]]
    rid, ref = load(ref_path)
    positions = [int(p) for p in os.environ.get("POSITIONS_ENV", "").split(",") if p]
    print("prompt\tvariant\tfrob_rel_l2\tmean_pos_rel_l2\tmedian_pos_rel_l2\tgap_removed_frac_frob"
          "\tgap_removed_frac_mean\tpos_rel_l2")
    base = None
    for name, path in variants:
        sid, sub = load(path)
        if not np.array_equal(rid, sid):
            print(f"logits_gap: token ids differ for {name}", file=sys.stderr)
            return 2
        line, base = row(prompt, name, ref, sub, base, positions)
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
