#!/usr/bin/env python3
"""
MNIST Inference Benchmark: PyTorch vs Aprender/Realizar

Scientifically reproducible benchmark comparing inference latency
for equivalent models on MNIST classification.

## Methodology (per Box et al. 2005, Georges et al. 2007)

1. Fixed random seeds for reproducibility
2. Warm-up phase excluded from measurements
3. Large sample size (10,000 iterations) for statistical significance
4. Report: mean, std, 95% CI, percentiles (p50, p95, p99)
5. Same model architecture: Logistic Regression (784 -> 10)
6. Same input data: Single MNIST sample (784 floats)

## Usage

    cd benches/comparative
    uv sync
    uv run mnist_benchmark.py

## Output

    pytorch_mnist_results.json - Machine-readable results
    Stdout - Human-readable report

## Citation

    See realizar CITATION.cff
"""

import gc
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

# Configuration - MUST match Rust benchmark
SEED = 42
INPUT_DIM = 784  # 28x28 MNIST
NUM_CLASSES = 2  # Binary: digit 0 vs others (matching aprender LogisticRegression)
WARMUP_ITERATIONS = 100
BENCHMARK_ITERATIONS = 10_000
TRAINING_SAMPLES = 1000

# Check dependencies
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    TORCH_VERSION = torch.__version__
except ImportError:
    HAS_TORCH = False
    TORCH_VERSION = "N/A"
    print("ERROR: PyTorch not installed. Run: uv install torch", file=sys.stderr)
    sys.exit(1)


@dataclass
class BenchmarkConfig:
    """Benchmark configuration for reproducibility."""
    seed: int
    input_dim: int
    num_classes: int
    warmup_iterations: int
    benchmark_iterations: int
    training_samples: int
    pytorch_version: str
    python_version: str
    platform: str
    cpu: str
    timestamp: str


@dataclass
class BenchmarkResult:
    """Statistical results from benchmark run."""
    name: str
    iterations: int
    mean_us: float
    std_us: float
    ci_95_lower: float
    ci_95_upper: float
    p50_us: float
    p95_us: float
    p99_us: float
    min_us: float
    max_us: float
    throughput_per_sec: float

    @classmethod
    def from_latencies(cls, name: str, latencies_ns: List[float]) -> "BenchmarkResult":
        """Compute statistics from raw latency measurements."""
        # Convert to microseconds
        latencies_us = [t / 1000.0 for t in latencies_ns]
        n = len(latencies_us)

        mean = statistics.mean(latencies_us)
        std = statistics.stdev(latencies_us) if n > 1 else 0.0
        se = std / (n ** 0.5)
        ci_95 = 1.96 * se

        sorted_lat = sorted(latencies_us)

        return cls(
            name=name,
            iterations=n,
            mean_us=mean,
            std_us=std,
            ci_95_lower=mean - ci_95,
            ci_95_upper=mean + ci_95,
            p50_us=sorted_lat[n // 2],
            p95_us=sorted_lat[int(n * 0.95)],
            p99_us=sorted_lat[int(n * 0.99)],
            min_us=sorted_lat[0],
            max_us=sorted_lat[-1],
            throughput_per_sec=1_000_000 / mean if mean > 0 else 0,
        )


class LogisticRegression(nn.Module):
    """
    Simple Logistic Regression for MNIST.

    Architecture: 784 -> 10 (no hidden layers)
    Equivalent to aprender::LogisticRegression
    """

    def __init__(self, input_dim: int = INPUT_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for MNIST.

    Architecture: 784 -> 128 -> 10
    For comparison with more complex models.
    """

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = 128,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def generate_mnist_data(seed: int = SEED) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate test MNIST-like data.

    Uses deterministic generation matching the Rust benchmark.
    Real MNIST could be loaded with torchvision, but test
    data ensures exact reproducibility across Rust/Python.
    """
    torch.manual_seed(seed)

    # Generate samples matching Rust implementation
    X = torch.zeros(TRAINING_SAMPLES, INPUT_DIM)
    y = torch.zeros(TRAINING_SAMPLES, dtype=torch.long)

    for i in range(TRAINING_SAMPLES):
        for j in range(INPUT_DIM):
            # Same formula as Rust: ((i * 17 + j * 31) % 256) / 255.0
            pixel = ((i * 17 + j * 31) % 256) / 255.0
            X[i, j] = pixel
        # Binary classification: 0 vs not-0 (matching Rust exactly)
        y[i] = 0 if i % 10 == 0 else 1

    return X, y


def train_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                epochs: int = 10, lr: float = 0.01) -> nn.Module:
    """Train model on data."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def benchmark_inference(model: nn.Module, sample: torch.Tensor,
                        name: str) -> BenchmarkResult:
    """
    Benchmark single-sample inference latency.

    Methodology:
    1. Run WARMUP_ITERATIONS to stabilize CPU caches
    2. Run BENCHMARK_ITERATIONS, measuring each
    3. Compute statistics on measurements
    """
    model.eval()

    # Warmup phase (excluded from measurements)
    with torch.no_grad():
        for _ in range(WARMUP_ITERATIONS):
            _ = model(sample)

    # Force garbage collection before measurement
    gc.collect()

    # Measurement phase
    latencies_ns: List[float] = []

    with torch.no_grad():
        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter_ns()
            _ = model(sample)
            end = time.perf_counter_ns()
            latencies_ns.append(end - start)

    return BenchmarkResult.from_latencies(name, latencies_ns)


def get_cpu_info() -> str:
    """Get CPU model name."""
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        elif platform.system() == "Darwin":
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            return result.stdout.strip()
    except Exception:
        pass
    return platform.processor() or "Unknown"


def run_benchmark() -> dict:
    """Run complete benchmark suite."""

    # Configuration
    config = BenchmarkConfig(
        seed=SEED,
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        warmup_iterations=WARMUP_ITERATIONS,
        benchmark_iterations=BENCHMARK_ITERATIONS,
        training_samples=TRAINING_SAMPLES,
        pytorch_version=TORCH_VERSION,
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.release()}",
        cpu=get_cpu_info(),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    print("=" * 70)
    print("MNIST Inference Benchmark: PyTorch")
    print("=" * 70)
    print()
    print("## Configuration")
    print(f"  Seed: {config.seed}")
    print(f"  Input dimensions: {config.input_dim}")
    print(f"  Output classes: {config.num_classes}")
    print(f"  Training samples: {config.training_samples}")
    print(f"  Warmup iterations: {config.warmup_iterations}")
    print(f"  Benchmark iterations: {config.benchmark_iterations}")
    print()
    print("## Environment")
    print(f"  PyTorch: {config.pytorch_version}")
    print(f"  Python: {config.python_version}")
    print(f"  Platform: {config.platform}")
    print(f"  CPU: {config.cpu}")
    print()

    # Generate data
    print("## Generating test MNIST data...")
    torch.manual_seed(SEED)
    X, y = generate_mnist_data(SEED)
    print(f"  Data shape: {X.shape}")
    print()

    # Create single inference sample (784 floats, matching Rust)
    sample = torch.zeros(1, INPUT_DIM)
    for j in range(INPUT_DIM):
        sample[0, j] = (j % 256) / 255.0

    results: List[BenchmarkResult] = []

    # Benchmark 1: Logistic Regression
    print("## Training LogisticRegression...")
    logreg = LogisticRegression()
    logreg = train_model(logreg, X, y, epochs=10)

    print("## Benchmarking LogisticRegression inference...")
    logreg_result = benchmark_inference(logreg, sample, "PyTorch LogisticRegression")
    results.append(logreg_result)
    print_result(logreg_result)
    print()

    # Benchmark 2: MLP
    print("## Training MLP (784->128->10)...")
    mlp = MLP()
    mlp = train_model(mlp, X, y, epochs=10)

    print("## Benchmarking MLP inference...")
    mlp_result = benchmark_inference(mlp, sample, "PyTorch MLP")
    results.append(mlp_result)
    print_result(mlp_result)
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("| Model                    | p50 (us) | p99 (us) | Throughput/sec |")
    print("|--------------------------|----------|----------|----------------|")
    for r in results:
        print(f"| {r.name:<24} | {r.p50_us:>8.2f} | {r.p99_us:>8.2f} | {r.throughput_per_sec:>14,.0f} |")
    print()

    return {
        "config": asdict(config),
        "results": [asdict(r) for r in results],
    }


def print_result(result: BenchmarkResult):
    """Print formatted benchmark result."""
    print(f"  Iterations: {result.iterations:,}")
    print(f"  Mean: {result.mean_us:.2f} us")
    print(f"  Std Dev: {result.std_us:.2f} us")
    print(f"  95% CI: [{result.ci_95_lower:.2f}, {result.ci_95_upper:.2f}] us")
    print(f"  p50: {result.p50_us:.2f} us")
    print(f"  p95: {result.p95_us:.2f} us")
    print(f"  p99: {result.p99_us:.2f} us")
    print(f"  Min: {result.min_us:.2f} us")
    print(f"  Max: {result.max_us:.2f} us")
    print(f"  Throughput: {result.throughput_per_sec:,.0f} inferences/sec")


def main():
    """Main entry point."""
    results = run_benchmark()

    # Save results
    output_path = Path(__file__).parent / "pytorch_mnist_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")
    print()
    print("To compare with Aprender/Realizar, run:")
    print("  cargo run --example mnist_apr_benchmark --release --features aprender-serve")
    print()
    print("Then run the comparison script:")
    print("  uv run compare_mnist.py")


if __name__ == "__main__":
    main()
