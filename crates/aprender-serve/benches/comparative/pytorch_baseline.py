#!/usr/bin/env python3
"""
PyTorch Baseline Benchmark for Realizar Comparison

This script measures PyTorch inference performance on canonical datasets
to establish baseline metrics for comparison with Realizar.

Datasets: MNIST, CIFAR-10, Fashion-MNIST (matching alimentar)
Metrics: Latency (p50, p95, p99), Throughput, Memory

Setup (using uv):
uv install torch torchvision
    uv install psutil scikit-learn  # optional

Usage:
    uv run pytorch_baseline.py --dataset mnist --batch-size 32 --iterations 1000
    uv run pytorch_baseline.py --all --output results.json
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import datasets, transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not installed. Install with: uv install torch torchvision")


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    framework: str
    dataset: str
    batch_size: int
    iterations: int
    latencies_us: List[float]
    throughput_samples_per_sec: float
    memory_mb: float

    @property
    def p50(self) -> float:
        sorted_lat = sorted(self.latencies_us)
        return sorted_lat[len(sorted_lat) // 2]

    @property
    def p95(self) -> float:
        sorted_lat = sorted(self.latencies_us)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def p99(self) -> float:
        sorted_lat = sorted(self.latencies_us)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def mean(self) -> float:
        return statistics.mean(self.latencies_us)

    @property
    def std_dev(self) -> float:
        return statistics.stdev(self.latencies_us) if len(self.latencies_us) > 1 else 0.0

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "p50_us": self.p50,
            "p95_us": self.p95,
            "p99_us": self.p99,
            "mean_us": self.mean,
            "std_dev_us": self.std_dev,
        }


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST/Fashion-MNIST (28x28 grayscale)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNN_CIFAR(nn.Module):
    """Simple CNN for CIFAR-10 (32x32 RGB)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleLinear(nn.Module):
    """Simple linear model for Iris (tabular)."""

    def __init__(self, input_features: int = 4, num_classes: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def load_dataset(name: str, batch_size: int):
    """Load dataset with PyTorch DataLoader."""

    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        model = SimpleCNN(num_classes=10)

    elif name == "fashion_mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
        model = SimpleCNN(num_classes=10)

    elif name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        model = SimpleCNN_CIFAR(num_classes=10)

    elif name == "iris":
        # Load Iris manually (not in torchvision)
        from sklearn.datasets import load_iris
        iris = load_iris()
        X = torch.tensor(iris.data, dtype=torch.float32)
        y = torch.tensor(iris.target, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X, y)
        model = SimpleLinear(input_features=4, num_classes=3)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Single-threaded for fair comparison
        pin_memory=False
    )

    return loader, model


def benchmark_inference(
    dataset_name: str,
    batch_size: int,
    iterations: int,
    warmup: int = 50,
    device: str = "cpu"
) -> BenchmarkResult:
    """Run inference benchmark."""

    print(f"Benchmarking {dataset_name} (batch_size={batch_size}, iterations={iterations})")

    # Load dataset and model
    loader, model = load_dataset(dataset_name, batch_size)
    model = model.to(device)
    model.eval()

    # Get a single batch for benchmarking
    data_iter = iter(loader)
    sample_batch = next(data_iter)
    if isinstance(sample_batch, (list, tuple)):
        inputs = sample_batch[0].to(device)
    else:
        inputs = sample_batch.to(device)

    # Warm-up phase
    print(f"  Warming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(inputs)

    # Force garbage collection
    gc.collect()
    if device == "cuda":
        torch.cuda.synchronize()

    # Measure memory before
    mem_before = get_memory_mb()

    # Benchmark phase
    print(f"  Benchmarking ({iterations} iterations)...")
    latencies = []
    total_samples = 0

    with torch.no_grad():
        for i in range(iterations):
            start = time.perf_counter_ns()
            output = model(inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter_ns()

            latency_us = (end - start) / 1000.0  # Convert to microseconds
            latencies.append(latency_us)
            total_samples += inputs.shape[0]

    # Measure memory after
    mem_after = get_memory_mb()
    memory_mb = max(mem_after - mem_before, 0)

    # Calculate throughput
    total_time_s = sum(latencies) / 1_000_000  # Convert us to seconds
    throughput = total_samples / total_time_s if total_time_s > 0 else 0

    result = BenchmarkResult(
        framework="pytorch",
        dataset=dataset_name,
        batch_size=batch_size,
        iterations=iterations,
        latencies_us=latencies,
        throughput_samples_per_sec=throughput,
        memory_mb=memory_mb
    )

    print(f"  Results: p50={result.p50:.2f}us, p99={result.p99:.2f}us, throughput={throughput:.1f} samples/s")

    return result


def run_all_benchmarks(
    batch_sizes: List[int] = [1, 8, 32],
    iterations: int = 1000,
    device: str = "cpu"
) -> List[BenchmarkResult]:
    """Run benchmarks on all datasets."""

    datasets = ["mnist", "fashion_mnist", "cifar10"]

    # Check for sklearn for Iris
    try:
        import sklearn
        datasets.append("iris")
    except ImportError:
        print("Warning: sklearn not installed, skipping Iris dataset")

    results = []

    for dataset_name in datasets:
        for batch_size in batch_sizes:
            try:
                result = benchmark_inference(
                    dataset_name=dataset_name,
                    batch_size=batch_size,
                    iterations=iterations,
                    device=device
                )
                results.append(result)
            except Exception as e:
                print(f"Error benchmarking {dataset_name} (batch={batch_size}): {e}")

    return results


def print_results_table(results: List[BenchmarkResult]):
    """Print results as formatted table."""

    print("\n" + "=" * 80)
    print("PyTorch Baseline Benchmark Results")
    print("=" * 80)
    print(f"{'Dataset':<15} {'Batch':<6} {'p50 (us)':<12} {'p99 (us)':<12} {'Throughput':<15} {'Memory':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r.dataset:<15} {r.batch_size:<6} {r.p50:<12.2f} {r.p99:<12.2f} {r.throughput_samples_per_sec:<15.1f} {r.memory_mb:<10.1f}")

    print("=" * 80)


def save_results(results: List[BenchmarkResult], output_path: str):
    """Save results to JSON file."""

    data = {
        "framework": "pytorch",
        "version": torch.__version__ if HAS_TORCH else "N/A",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": [r.to_dict() for r in results]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch Baseline Benchmark")
    parser.add_argument("--dataset", type=str, choices=["mnist", "fashion_mnist", "cifar10", "iris"],
                        help="Dataset to benchmark")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to use")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--output", type=str, help="Output JSON file path")

    args = parser.parse_args()

    if not HAS_TORCH:
        print("Error: PyTorch is required. Install with: uv install torch torchvision")
        sys.exit(1)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {args.device}")
    print()

    if args.all:
        results = run_all_benchmarks(
            batch_sizes=[1, 8, 32],
            iterations=args.iterations,
            device=args.device
        )
    elif args.dataset:
        result = benchmark_inference(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            iterations=args.iterations,
            warmup=args.warmup,
            device=args.device
        )
        results = [result]
    else:
        parser.print_help()
        sys.exit(1)

    print_results_table(results)

    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
