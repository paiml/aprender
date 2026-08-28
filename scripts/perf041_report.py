#!/usr/bin/env python3
"""PERF-041 report: decompose serialization_index(c) into its two factors.

    serialization_index(c) = wall(c) / wall_fast(1)
                           = path_penalty * scaling_index(c)
      path_penalty  = latency_batched(1) / latency_fast(1)
      scaling_index = latency(c)         / latency_batched(1)

Also prints the TOKEN-NORMALIZED index, which the wall-clock one cannot
distinguish itself from when the two bands generate different numbers of
tokens:

    token_index(c) = c * agg_tok_s(1) / agg_tok_s(c)

Both are 1.0 under perfect sharing and c under full serialization. They agree
only if generation lengths match; a gap between them IS the confound, surfaced
rather than absorbed.
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys


def load(out_dir: str) -> dict[tuple[str, int], list[dict]]:
    bands: dict[tuple[str, int], list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        with open(path) as fh:
            try:
                rec = json.load(fh)
            except json.JSONDecodeError:
                continue
        if "error" in rec:
            print(f"  SKIP {os.path.basename(path)}: {rec['error']}")
            continue
        label = rec.get("label", "")
        mode = label.split("-")[0] if label else "?"
        bands.setdefault((mode, int(rec["c"])), []).append(rec)
    return bands


def med(recs: list[dict], key: str) -> float:
    return statistics.median(r[key] for r in recs)


def print_bands(bands: dict) -> None:
    print()
    print("=== PERF-041 bands (median over replicates) ===")
    print(f"{'mode':<8}{'c':>3}{'reps':>6}{'lat_p50_s':>12}{'agg_tok_s':>12}"
          f"{'tok_p50':>10}{'tok_min':>9}{'tok_max':>9}")
    for (mode, c) in sorted(bands):
        recs = bands[(mode, c)]
        print(f"{mode:<8}{c:>3}{len(recs):>6}{med(recs,'latency_p50_s'):>12.3f}"
              f"{med(recs,'agg_tok_s'):>12.1f}{med(recs,'tokens_p50'):>10.0f}"
              f"{min(r['tokens_min'] for r in recs):>9d}"
              f"{max(r['tokens_max'] for r in recs):>9d}")


def print_penalty(bands: dict, lat_fast1: float) -> float | None:
    """path_penalty = latency_batched(1) / latency_fast(1), or None if unmeasured."""
    forced1 = bands.get(("forced", 1))
    if not forced1:
        print("\nperf041: no forced c=1 band -- path_penalty UNMEASURED")
        return None
    lat_forced1 = med(forced1, "latency_p50_s")
    penalty = lat_forced1 / lat_fast1
    print()
    print("=== path_penalty (the term no recorded run contains) ===")
    print(f"  latency_fast(1)    = {lat_fast1:.3f}s  CUDA-graph replay "
          f"(generate_gpu_resident_streaming)")
    print(f"  latency_batched(1) = {lat_forced1:.3f}s  eager launch "
          f"(batched_decode_step, APR_FORCE_BATCHED_PATH=1)")
    print(f"  path_penalty       = {penalty:.2f}x")
    return penalty


def _index_row(c: int, recs: list[dict], lat_fast1: float, agg_fast1: float,
               penalty: float | None) -> tuple[str, bool]:
    """One row of the decomposition table, plus whether it satisfies index(c) < c."""
    ser = med(recs, "latency_p50_s") / lat_fast1
    tok = c * agg_fast1 / med(recs, "agg_tok_s")
    scaling = ser / penalty if penalty else float("nan")
    holds = c == 1 or ser < c
    pass_ser = "PASS" if holds else "FAIL"
    if c == 1 or penalty is None:
        pass_scl = "-"
    else:
        pass_scl = "PASS" if scaling < c else "FAIL"
    row = (f"{c:>3}{ser:>12.2f}{tok:>13.2f}{scaling:>15.2f}"
           f"{pass_ser:>6}{pass_scl:>12}")
    return row, holds


def print_index(bands: dict, lat_fast1: float, agg_fast1: float,
                penalty: float | None) -> bool:
    print()
    print("=== index, decomposed ===")
    print(f"{'c':>3}{'ser_index':>12}{'token_index':>13}{'scaling_index':>15}"
          f"{'<c?':>6}{'scaling<c?':>12}")
    ok = True
    for (mode, c) in sorted(bands):
        if mode != "fast":
            continue
        row, holds = _index_row(c, bands[(mode, c)], lat_fast1, agg_fast1, penalty)
        ok = ok and holds
        print(row)
    return ok


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/perf041"
    bands = load(out_dir)
    if not bands:
        print("perf041: no bands to report")
        return 1
    print_bands(bands)
    fast1 = bands.get(("fast", 1))
    if not fast1:
        print("\nperf041: no fast c=1 band -- cannot form the index")
        return 1
    penalty = print_penalty(bands, med(fast1, "latency_p50_s"))
    ok = print_index(bands, med(fast1, "latency_p50_s"),
                     med(fast1, "agg_tok_s"), penalty)
    print()
    print("postcondition: serialization_index(c) < c for every c > 1 -> "
          + ("PASS" if ok else "FAIL"))
    print("NOTE: the postcondition is NOT restated in terms of scaling_index. "
          "The decomposition explains the number; it does not discharge it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
