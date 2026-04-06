#!/usr/bin/env python3
"""NumPy GEMM sweep for cgp compete. Tests multiple sizes, reports in trueno benchmark format.

Usage:
  OMP_NUM_THREADS=1 python3 benchmarks/numpy_gemm_sweep.py   # single-thread
  python3 benchmarks/numpy_gemm_sweep.py                      # all threads
"""
import os, sys, time

# Must set env BEFORE numpy import (OpenBLAS reads at load time)
single_thread = "--single-thread" in sys.argv or os.environ.get("OMP_NUM_THREADS") == "1"
import numpy as np

label = "1T" if single_thread else f"{os.cpu_count()}T"
print(f"NumPy {np.__version__} GEMM benchmark ({label})\n")

for size in [256, 512, 1024]:
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)
    # Warmup
    for _ in range(5):
        _ = a @ b
    # Min-of-10
    best = float('inf')
    for _ in range(10):
        t0 = time.perf_counter()
        _ = a @ b
        best = min(best, time.perf_counter() - t0)
    ms = best * 1000
    gflops = 2 * size**3 / best / 1e9
    print(f"  Matrix Multiplication ({size}x{size}x{size})...     {ms:.2f} ms  ({gflops:.2f} GFLOPS)")
