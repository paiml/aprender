#!/usr/bin/env python3
"""Recompute APR-PERF-GATE-001 4.4.2 per-cell wall-clock from a measured t_req.

Formula (campaign-scope 5, spec 4.4.2):
  T_band = 2c warmup requests + 5 s quiesce + max(60 s, n * t_req) + drain
  n      = max(30, 8c)
apr serialises (max_in_flight = 1), so the sampling phase costs n * t_req of
wall-clock regardless of c. Drain is not modelled (>= 0), so every figure is a
LOWER bound.
"""
import sys

BANDS = [1, 4, 8, 16]
REPLICATES = 3
QUIESCE = 5.0


def cell(t_req, bands=BANDS, replicates=REPLICATES):
    rows = []
    total = 0.0
    for c in bands:
        n = max(30, 8 * c)
        warm = 2 * c
        band = warm * t_req + QUIESCE + max(60.0, n * t_req)
        rows.append((c, warm, n, band))
        total += band * replicates
    return rows, total


def fmt(s):
    return f"{s/3600:.2f} h" if s >= 3600 else f"{s/60:.1f} min"


for label, t_req in [(a.split("=")[0], float(a.split("=")[1])) for a in sys.argv[1:]]:
    rows, total = cell(t_req)
    print(f"== {label}  t_req = {t_req:.1f} s ==")
    for c, warm, n, band in rows:
        print(f"   c={c:<3} warmup={warm:<3} n={n:<4} one replicate = {fmt(band)}   x3 = {fmt(band*3)}")
    print(f"   apr lane, 4 bands x N=3  = {fmt(total)}  ({total:.0f} s)")
    reqs = sum((2 * c + max(30, 8 * c)) for c in BANDS) * REPLICATES
    print(f"   total requests issued    = {reqs}")
    calib = 2 * t_req + QUIESCE + max(60.0, 30 * t_req)   # 2 warmup + quiesce + 30 sampled
    print(f"   30-request c=1 calibration = {fmt(calib)}"
          f"   (sampled requests alone: {fmt(30*t_req)})")
    print()
