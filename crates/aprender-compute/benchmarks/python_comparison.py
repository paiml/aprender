#!/usr/bin/env python3
"""Benchmark comparing Trueno vs NumPy vs PyTorch for 1D vector operations."""

import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch


@dataclass
class BenchmarkDef:
    """Benchmark definition."""

    name: str
    numpy_fn: Callable
    torch_fn: Callable
    binary: bool = False  # True if operation takes two inputs


# Benchmark definitions - declarative, no duplication
BENCHMARKS = [
    # Binary operations
    BenchmarkDef("add", lambda a, b: a + b, lambda a, b: a + b, binary=True),
    BenchmarkDef("sub", lambda a, b: a - b, lambda a, b: a - b, binary=True),
    BenchmarkDef("mul", lambda a, b: a * b, lambda a, b: a * b, binary=True),
    BenchmarkDef("div", lambda a, b: a / (b + 1.0), lambda a, b: a / (b + 1.0), binary=True),
    BenchmarkDef("dot", np.dot, torch.dot, binary=True),
    # Unary reductions
    BenchmarkDef("sum", np.sum, torch.sum),
    BenchmarkDef("max", np.max, torch.max),
    BenchmarkDef("min", np.min, torch.min),
    BenchmarkDef("argmax", np.argmax, torch.argmax),
    BenchmarkDef("argmin", np.argmin, torch.argmin),
    # Norms
    BenchmarkDef("norm_l2", np.linalg.norm, torch.norm),
    BenchmarkDef("norm_l1", lambda a: np.linalg.norm(a, ord=1), lambda a: torch.norm(a, p=1)),
    # Activations
    BenchmarkDef("relu", lambda a: np.maximum(0, a), torch.relu),
    BenchmarkDef("sigmoid", lambda a: 1 / (1 + np.exp(-a)), torch.sigmoid),
    BenchmarkDef("tanh", np.tanh, torch.tanh),
    BenchmarkDef("exp", np.exp, torch.exp),
    # Elementwise
    BenchmarkDef("abs", np.abs, torch.abs),
    BenchmarkDef("clamp", lambda a: np.clip(a, -1.0, 1.0), lambda a: torch.clamp(a, -1.0, 1.0)),
]

SIZES = [100, 1_000, 10_000, 100_000, 1_000_000]
ITERATIONS = 100


@dataclass
class Result:
    """Benchmark result."""

    mean_ns: float
    std_ns: float
    iterations: int


@dataclass
class BenchmarkResults:
    """All benchmark results."""

    numpy: dict = field(default_factory=dict)
    pytorch_cpu: dict = field(default_factory=dict)


def make_numpy_data(size: int, binary: bool) -> tuple:
    """Create NumPy test data."""
    a = np.random.randn(size).astype(np.float32)
    return (a, np.random.randn(size).astype(np.float32)) if binary else (a,)


def make_torch_data(size: int, binary: bool) -> tuple:
    """Create PyTorch test data."""
    a = torch.randn(size, dtype=torch.float32)
    return (a, torch.randn(size, dtype=torch.float32)) if binary else (a,)


def measure(fn: Callable, data: tuple, iterations: int) -> Result:
    """Measure function execution time."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn(*data)
        elapsed = (time.perf_counter() - start) * 1e9
        times.append(elapsed)
    return Result(
        mean_ns=statistics.mean(times),
        std_ns=statistics.stdev(times) if len(times) > 1 else 0,
        iterations=iterations,
    )


def run_benchmark(bench: BenchmarkDef, size: int) -> tuple[Result, Result]:
    """Run a single benchmark for one size."""
    np_data = make_numpy_data(size, bench.binary)
    pt_data = make_torch_data(size, bench.binary)
    np_result = measure(bench.numpy_fn, np_data, ITERATIONS)
    pt_result = measure(bench.torch_fn, pt_data, ITERATIONS)
    return np_result, pt_result


def run_all() -> BenchmarkResults:
    """Run all benchmarks."""
    results = BenchmarkResults()

    for bench in BENCHMARKS:
        print(f"\n{bench.name}:")
        results.numpy[bench.name] = {}
        results.pytorch_cpu[bench.name] = {}

        for size in SIZES:
            np_res, pt_res = run_benchmark(bench, size)
            results.numpy[bench.name][str(size)] = vars(np_res)
            results.pytorch_cpu[bench.name][str(size)] = vars(pt_res)
            print(f"  {size:>10,} | NumPy: {np_res.mean_ns:>10.0f} ns | PyTorch: {pt_res.mean_ns:>10.0f} ns")

    return results


def print_summary(results: BenchmarkResults) -> None:
    """Print performance summary."""
    print("\n" + "=" * 60)
    print("SUMMARY: NumPy vs PyTorch")
    print("=" * 60)

    for name in results.numpy:
        print(f"\n{name.upper()}:")
        for size in SIZES:
            key = str(size)
            np_time = results.numpy[name][key]["mean_ns"]
            pt_time = results.pytorch_cpu[name][key]["mean_ns"]
            ratio = pt_time / np_time
            faster = "NumPy" if ratio > 1.0 else "PyTorch"
            print(f"  {size:>10,}: {faster:>8} {abs(ratio):>5.2f}x faster")


def save_results(results: BenchmarkResults, path: str = "python_results.json") -> None:
    """Save results to JSON."""
    data = {"numpy": results.numpy, "pytorch_cpu": results.pytorch_cpu}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {path}")


def main() -> None:
    """Run benchmark suite."""
    print("=" * 60)
    print("Trueno vs NumPy vs PyTorch Benchmark")
    print("=" * 60)

    results = run_all()
    print_summary(results)
    save_results(results)


if __name__ == "__main__":
    main()
