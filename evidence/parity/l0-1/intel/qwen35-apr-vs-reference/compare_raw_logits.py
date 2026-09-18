#!/usr/bin/env python3
"""Per-position comparison of two APRRAWLG raw-logit files (PMAT-3091).

Layout (both files): char[8] "APRRAWLG" | uint32 version=1 | int32 n_pos | int32 n_vocab |
int32 token_ids[n_pos] | float32 logits[n_pos*n_vocab], little-endian.

Per position: cosine of the full logit vectors (float64), argmax agreement, max |diff|,
plus near-tie context: the reference's top1-top2 logit gap, apr's rank of the reference
argmax, and apr's logit gap from its own argmax to the reference argmax.
Prints a TSV table plus summary lines. Reports numbers only; no threshold, no verdict.
Exit: 0 ok, 2 bad/mismatched input.

Usage: compare_raw_logits.py <reference.bin> <subject.bin> [--json out.json]
"""
import json
import sys

import numpy as np

MAGIC = b"APRRAWLG"
HEADER = ("pos\ttoken_id\tcosine\targmax_ref\targmax_apr\targmax_match\tmax_abs_diff"
          "\tref_top1_top2_gap\tapr_rank_of_ref_argmax\tapr_gap_to_ref_argmax")


def load(path):
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:8] != MAGIC:
        raise ValueError(f"{path}: bad magic {buf[:8]!r}")
    version, n_pos, n_vocab = (int(v) for v in np.frombuffer(buf, dtype="<i4", count=3, offset=8))
    if version != 1:
        raise ValueError(f"{path}: version {version}")
    off = 20 + 4 * n_pos
    want = off + 4 * n_pos * n_vocab
    if len(buf) != want:
        raise ValueError(f"{path}: size {len(buf)} != expected {want}")
    ids = np.frombuffer(buf, dtype="<i4", count=n_pos, offset=20).astype(np.int64)
    logits = np.frombuffer(buf, dtype="<f4", count=n_pos * n_vocab, offset=off)
    return ids, logits.reshape(n_pos, n_vocab).astype(np.float64)


def parse_args(argv):
    args = list(argv[1:])
    json_out = None
    if "--json" in args:
        i = args.index("--json")
        json_out = args[i + 1] if i + 1 < len(args) else None
        del args[i:i + 2]
    if len(args) != 2:
        return None
    return args[0], args[1], json_out


def check_pair(rid, ref, sid, sub):
    """Return an error string, or None when the two files are comparable."""
    if ref.shape != sub.shape or rid.shape != sid.shape or not np.array_equal(rid, sid):
        return f"shape/token mismatch ref={ref.shape} sub={sub.shape}"
    bad_ref, bad_sub = int((~np.isfinite(ref)).sum()), int((~np.isfinite(sub)).sum())
    if bad_ref or bad_sub:
        return f"non-finite logits present (ref={bad_ref} sub={bad_sub})"
    return None


def compare_row(k, token_id, a, b):
    am_r, am_s = int(np.argmax(a)), int(np.argmax(b))
    top2 = np.partition(a, -2)[-2:]
    return dict(
        pos=k, token_id=int(token_id),
        cosine=float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
        argmax_ref=am_r, argmax_apr=am_s, argmax_match=am_r == am_s,
        max_abs_diff=float(np.max(np.abs(a - b))),
        ref_top1_top2_gap=float(top2[1] - top2[0]),
        apr_rank_of_ref_argmax=int((b > b[am_r]).sum()) + 1,
        apr_gap_to_ref_argmax=float(b[am_s] - b[am_r]),
    )


def format_row(r):
    return (f"{r['pos']}\t{r['token_id']}\t{r['cosine']:.6f}\t{r['argmax_ref']}\t{r['argmax_apr']}"
            f"\t{r['argmax_match']}\t{r['max_abs_diff']:.6f}\t{r['ref_top1_top2_gap']:.6f}"
            f"\t{r['apr_rank_of_ref_argmax']}\t{r['apr_gap_to_ref_argmax']:.6f}")


def summarize(rows, n_vocab):
    cosines = np.array([r["cosine"] for r in rows])
    diffs = [r["max_abs_diff"] for r in rows]
    kmin = int(np.argmin(cosines))
    below = [r["pos"] for r in rows if r["cosine"] < 0.98]
    mism = [r["pos"] for r in rows if not r["argmax_match"]]
    return dict(
        n_positions=len(rows), n_vocab=int(n_vocab),
        min_cosine=round(float(cosines[kmin]), 6), min_cosine_pos=kmin,
        mean_cosine=round(float(cosines.mean()), 6),
        median_cosine=round(float(np.median(cosines)), 6),
        positions_below_0_98=below, n_below_0_98=len(below),
        argmax_mismatches=mism, n_argmax_mismatches=len(mism),
        max_abs_diff=round(max(diffs), 6),
        max_abs_diff_pos=int(np.argmax(diffs)),
    )


def load_pair(ref_path, sub_path):
    """Return (ids, ref, sub) or raise ValueError when the files cannot be compared."""
    try:
        rid, ref = load(ref_path)
        sid, sub = load(sub_path)
    except OSError as e:
        raise ValueError(str(e)) from e
    err = check_pair(rid, ref, sid, sub)
    if err:
        raise ValueError(err)
    return rid, ref, sub


def emit(rows, summary, json_out):
    print(HEADER)
    for r in rows:
        print(format_row(r))
    for key, val in summary.items():
        print(f"summary\t{key}\t{val}")
    if json_out:
        with open(json_out, "w") as f:
            json.dump(dict(summary=summary, rows=rows), f, indent=1)
            f.write("\n")


def main(argv):
    parsed = parse_args(argv)
    if parsed is None:
        print(__doc__, file=sys.stderr)
        return 2
    ref_path, sub_path, json_out = parsed
    try:
        rid, ref, sub = load_pair(ref_path, sub_path)
    except ValueError as e:
        print(f"compare_raw_logits: {e}", file=sys.stderr)
        return 2
    rows = [compare_row(k, rid[k], ref[k], sub[k]) for k in range(ref.shape[0])]
    emit(rows, summarize(rows, ref.shape[1]), json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
