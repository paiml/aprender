#!/usr/bin/env python3
"""
Activation Functions Performance Comparison: NumPy vs PyTorch

This example demonstrates the performance of common neural network activation
functions, comparing NumPy and PyTorch implementations.

Trueno (Rust SIMD) typically provides:
- SIMD-optimized implementations of ReLU, Sigmoid, Tanh
- Predictable, low-latency execution
- 2-4x speedup for compute-intensive activations

Run with:
    uv run examples/activation_comparison.py

Requirements:
    uv pip install numpy torch
    # Or use the benchmarks environment: cd benchmarks && uv run ../examples/activation_comparison.py
"""

import time
import numpy as np
import torch
from typing import Tuple, Callable


def benchmark_numpy_activation(
    size: int,
    activation_fn: Callable,
    iterations: int = 1000
) -> Tuple[float, float]:
    """Benchmark NumPy activation function"""
    # Generate random data
    x = np.random.randn(size).astype(np.float32)

    # Warmup
    for _ in range(10):
        _ = activation_fn(x)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = activation_fn(x)
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # Convert to microseconds

    return np.mean(times), np.std(times)


def benchmark_pytorch_activation(
    size: int,
    activation_fn: Callable,
    iterations: int = 1000
) -> Tuple[float, float]:
    """Benchmark PyTorch activation function (CPU)"""
    # Generate random data
    x = torch.randn(size, dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = activation_fn(x)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = activation_fn(x)
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # Convert to microseconds

    return np.mean(times), np.std(times)


def main():
    print("=" * 90)
    print("Activation Functions Performance Comparison: NumPy vs PyTorch")
    print("=" * 90)
    print("\n⚡ Trueno Performance Context:")
    print("  - SIMD-optimized activation functions")
    print("  - 2-4x speedup for compute-intensive activations")
    print("  - Zero Python overhead for ML inference pipelines")
    print("\nThis script benchmarks NumPy and PyTorch for comparison.")
    print("See benchmarks/README.md for complete trueno benchmarks.\n")
    print("=" * 90)

    # Activation functions to test
    activations = [
        ("ReLU", lambda x: np.maximum(0, x), lambda x: torch.relu(x)),
        ("Sigmoid", lambda x: 1 / (1 + np.exp(-x)), lambda x: torch.sigmoid(x)),
        ("Tanh", lambda x: np.tanh(x), lambda x: torch.tanh(x)),
        ("Exp", lambda x: np.exp(x), lambda x: torch.exp(x)),
    ]

    size = 10_000
    iterations = 1000

    print(f"\nBenchmarking {size:,} elements with {iterations} iterations per test...\n")
    print(f"{'Activation':<15} {'NumPy (μs)':<15} {'PyTorch (μs)':<15} {'Winner':<12} {'Speedup':<10}")
    print("-" * 90)

    for name, np_fn, pt_fn in activations:
        np_mean, np_std = benchmark_numpy_activation(size, np_fn, iterations)
        pt_mean, pt_std = benchmark_pytorch_activation(size, pt_fn, iterations)

        if np_mean < pt_mean:
            winner = "NumPy"
            speedup = pt_mean / np_mean
        else:
            winner = "PyTorch"
            speedup = np_mean / pt_mean

        print(f"{name:<15} {np_mean:>10.2f} ± {np_std:>3.2f}  "
              f"{pt_mean:>10.2f} ± {pt_std:>3.2f}  "
              f"{winner:<12} {speedup:>6.2f}x")

    print("\n" + "=" * 90)
    print("Key Findings:")
    print("=" * 90)
    print("\n✅ Activation Function Characteristics:")
    print("   - ReLU: Very fast (element-wise comparison + clamp)")
    print("   - Sigmoid: Moderate (requires exp computation)")
    print("   - Tanh: Moderate (hyperbolic function)")
    print("   - Exp: Expensive (transcendental function)")
    print("\n🎯 Performance Insights:")
    print("   - NumPy's vectorized ops are highly optimized")
    print("   - PyTorch adds minimal overhead for CPU operations")
    print("   - Transcendental functions (exp, sigmoid) benefit from SIMD")
    print("\n💡 When to use Trueno:")
    print("   - ML inference pipelines in Rust/WASM")
    print("   - Real-time systems requiring predictable latency")
    print("   - Embedded ML without Python runtime")
    print("   - Edge deployment (WebAssembly)")
    print("\n📊 Trueno Advantages:")
    print("   - Zero Python overhead")
    print("   - Predictable memory usage")
    print("   - Cross-platform (x86, ARM, WASM)")
    print("   - GPU fallback for large batches")
    print("\n📊 Run full benchmarks: cd benchmarks && ./run_all.sh")
    print("=" * 90)


if __name__ == "__main__":
    main()
