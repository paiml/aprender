#!/usr/bin/env python3
"""
Matrix Multiplication Performance Comparison: NumPy vs PyTorch

This example demonstrates the performance characteristics of matrix multiplication,
comparing NumPy and PyTorch implementations.

Trueno (Rust SIMD + GPU) typically achieves:
- GPU: 2-10x faster than scalar for 500×500+ matrices
- SIMD: 7x faster than naive O(n³) for 128×128 matrices
- Automatic backend selection based on matrix size

Run with:
    uv run examples/matrix_multiply_comparison.py

Requirements:
    uv pip install numpy torch
    # Or use the benchmarks environment: cd benchmarks && uv run ../examples/matrix_multiply_comparison.py
"""

import time
import numpy as np
import torch
from typing import Tuple


def benchmark_numpy_matmul(size: int, iterations: int = 100) -> Tuple[float, float]:
    """Benchmark NumPy matrix multiplication"""
    # Generate random matrices
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)

    # Warmup
    for _ in range(5):
        _ = np.matmul(a, b)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = np.matmul(a, b)
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # Convert to microseconds

    return np.mean(times), np.std(times)


def benchmark_pytorch_matmul(size: int, iterations: int = 100) -> Tuple[float, float]:
    """Benchmark PyTorch matrix multiplication (CPU)"""
    # Generate random matrices
    a = torch.randn(size, size, dtype=torch.float32)
    b = torch.randn(size, size, dtype=torch.float32)

    # Warmup
    for _ in range(5):
        _ = torch.matmul(a, b)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = torch.matmul(a, b)
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # Convert to microseconds

    return np.mean(times), np.std(times)


def format_time(microseconds: float) -> str:
    """Format time in appropriate unit"""
    if microseconds < 1_000:
        return f"{microseconds:.2f} μs"
    elif microseconds < 1_000_000:
        return f"{microseconds / 1_000:.2f} ms"
    else:
        return f"{microseconds / 1_000_000:.2f} s"


def main():
    print("=" * 90)
    print("Matrix Multiplication Performance Comparison: NumPy vs PyTorch")
    print("=" * 90)
    print("\n⚡ Trueno Performance Context:")
    print("  - GPU Backend: 2-10x faster than scalar for 500×500+ matrices")
    print("  - SIMD Backend: ~7x faster than naive O(n³) for 128×128 matrices")
    print("  - Automatic backend selection (SIMD threshold: 64×64)")
    print("\nThis script benchmarks NumPy and PyTorch for comparison.")
    print("See benchmarks/README.md for complete trueno benchmarks.\n")
    print("=" * 90)

    # Matrix sizes to test
    sizes = [64, 128, 256, 512, 1024]
    iterations_map = {
        64: 1000,
        128: 500,
        256: 200,
        512: 50,
        1024: 10,
    }

    print(f"\n{'Size':<10} {'NumPy Time':<20} {'PyTorch Time':<20} {'Winner':<12} {'Speedup':<10}")
    print("-" * 90)

    for size in sizes:
        iterations = iterations_map[size]
        print(f"{size}×{size:<6}", end=" ", flush=True)

        np_mean, np_std = benchmark_numpy_matmul(size, iterations)
        pt_mean, pt_std = benchmark_pytorch_matmul(size, iterations)

        if np_mean < pt_mean:
            winner = "NumPy"
            speedup = pt_mean / np_mean
        else:
            winner = "PyTorch"
            speedup = np_mean / pt_mean

        np_time_str = format_time(np_mean)
        pt_time_str = format_time(pt_mean)

        print(f"{np_time_str:<20} {pt_time_str:<20} {winner:<12} {speedup:>6.2f}x")

    print("\n" + "=" * 90)
    print("Key Findings:")
    print("=" * 90)
    print("\n✅ Matrix Multiplication Characteristics:")
    print("   - O(n³) computational complexity")
    print("   - Benefits from cache optimization and SIMD")
    print("   - GPU acceleration effective for large matrices (>500×500)")
    print("\n🎯 Trueno Optimization Strategy:")
    print("   - Small matrices (<64×64): Naive O(n³)")
    print("   - Medium matrices (64-500): SIMD-optimized with transpose")
    print("   - Large matrices (>500×500): GPU compute shaders")
    print("\n📊 Performance Insights:")
    print("   - NumPy uses highly optimized BLAS libraries (OpenBLAS/MKL)")
    print("   - PyTorch adds framework overhead for small matrices")
    print("   - Trueno provides predictable performance with zero Python overhead")
    print("\n💡 When to use Trueno:")
    print("   - Real-time systems requiring predictable latency")
    print("   - Embedded systems without Python runtime")
    print("   - WebAssembly deployment (browser/edge)")
    print("\n📊 Run full benchmarks: cd benchmarks && ./run_all.sh")
    print("=" * 90)


if __name__ == "__main__":
    main()
