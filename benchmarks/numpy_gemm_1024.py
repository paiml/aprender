#!/usr/bin/env python3
"""NumPy GEMM benchmark for cgp compete. Outputs: TIME_MS GFLOPS"""
import time, numpy as np, sys

size = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
a = np.random.randn(size, size).astype(np.float32)
b = np.random.randn(size, size).astype(np.float32)

# Warmup
for _ in range(5):
    _ = a @ b

# Benchmark min-of-10
best = float('inf')
for _ in range(10):
    t0 = time.perf_counter()
    _ = a @ b
    elapsed = time.perf_counter() - t0
    best = min(best, elapsed)

ms = best * 1000
gflops = 2 * size**3 / best / 1e9
print(f"Matrix Multiplication ({size}x{size}x{size})...     {ms:.2f} ms  ({gflops:.2f} GFLOPS)")
