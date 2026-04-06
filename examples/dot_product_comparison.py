#!/usr/bin/env python3
"""
Dot Product Performance Comparison: NumPy vs PyTorch

This example demonstrates the performance characteristics documented in trueno's
benchmarks by comparing NumPy and PyTorch for dot product operations.

Trueno (Rust SIMD) typically achieves:
- 1.6x faster than NumPy
- 2.8x faster than PyTorch
- 11.9x faster than scalar Rust

Run with:
    uv run examples/dot_product_comparison.py

Requirements:
    uv pip install numpy torch
    # Or use the benchmarks environment: cd benchmarks && uv run ../examples/dot_product_comparison.py
"""

import time
import numpy as np
import torch
from typing import Tuple


def benchmark_numpy_dot(size: int, iterations: int = 1000) -> Tuple[float, float]:
    """Benchmark NumPy dot product"""
    # Generate random data
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)

    # Warmup
    for _ in range(10):
        _ = np.dot(a, b)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = np.dot(a, b)
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # Convert to microseconds

    return np.mean(times), np.std(times)


def benchmark_pytorch_dot(size: int, iterations: int = 1000) -> Tuple[float, float]:
    """Benchmark PyTorch dot product (CPU)"""
    # Generate random data
    a = torch.randn(size, dtype=torch.float32)
    b = torch.randn(size, dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = torch.dot(a, b)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = torch.dot(a, b)
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # Convert to microseconds

    return np.mean(times), np.std(times)


def main():
    print("=" * 80)
    print("Dot Product Performance Comparison: NumPy vs PyTorch")
    print("=" * 80)
    print("\n⚡ Trueno Performance Context:")
    print("  - Trueno AVX-512: 11.9x faster than scalar Rust")
    print("  - Trueno AVX-512: 1.6x faster than NumPy")
    print("  - Trueno AVX-512: 2.8x faster than PyTorch")
    print("\nThis script benchmarks NumPy and PyTorch for comparison.")
    print("See benchmarks/README.md for complete trueno benchmarks.\n")
    print("=" * 80)

    sizes = [100, 1_000, 10_000, 100_000, 1_000_000]
    iterations = 1000

    print(f"\nRunning {iterations} iterations per benchmark...")
    print(f"\n{'Size':<12} {'NumPy (μs)':<15} {'PyTorch (μs)':<15} {'Winner':<15} {'Speedup':<10}")
    print("-" * 80)

    for size in sizes:
        np_mean, np_std = benchmark_numpy_dot(size, iterations)
        pt_mean, pt_std = benchmark_pytorch_dot(size, iterations)

        if np_mean < pt_mean:
            winner = "NumPy"
            speedup = pt_mean / np_mean
        else:
            winner = "PyTorch"
            speedup = np_mean / pt_mean

        print(f"{size:<12,} {np_mean:>10.2f} ± {np_std:>5.2f}  "
              f"{pt_mean:>10.2f} ± {pt_std:>5.2f}  "
              f"{winner:<15} {speedup:>6.2f}x")

    print("\n" + "=" * 80)
    print("Key Findings:")
    print("=" * 80)
    print("\n✅ Compute-Intensive Operations (like dot product):")
    print("   - Benefit significantly from SIMD optimization")
    print("   - Trueno's AVX-512 backend provides 2-12x speedup")
    print("   - NumPy typically faster than PyTorch for CPU operations")
    print("\n🎯 When to use Trueno:")
    print("   - High-performance compute (ML inference, signal processing)")
    print("   - When you need predictable, low-latency operations")
    print("   - Systems programming where Python overhead is unacceptable")
    print("\n📊 Run full benchmarks: cd benchmarks && ./run_all.sh")
    print("=" * 80)


if __name__ == "__main__":
    main()
