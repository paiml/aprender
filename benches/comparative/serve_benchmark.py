#!/usr/bin/env python3
"""
HTTP Serving Benchmark: TorchServe vs Realizar

Compares end-to-end HTTP inference latency for MNIST classification.

## Methodology (per Georges et al. 2007)

1. Warm-up phase excluded from measurements
2. Large sample size (1,000 requests) for statistical significance
3. Report: mean, std, p50, p95, p99

## Prerequisites

For Realizar:
    cargo run --example serve_mnist --release --features aprender-serve

For TorchServe:
    pip install torchserve torch-model-archiver
    torch-model-archiver --model-name mnist --serialized-file model.pt ...
    torchserve --start --model-store model_store --models mnist=mnist.mar

## Usage

    cd benches/comparative
    uv run serve_benchmark.py --realizar http://localhost:3000
    uv run serve_benchmark.py --torchserve http://localhost:8080

## Output

    serve_benchmark_results.json - Machine-readable results
"""

import json
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import argparse
import platform

# Configuration
WARMUP_REQUESTS = 100
BENCHMARK_REQUESTS = 1000
INPUT_DIM = 784  # MNIST 28x28

# Check dependencies
try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Statistical results from HTTP benchmark."""
    name: str
    iterations: int
    mean_us: float
    std_us: float
    p50_us: float
    p95_us: float
    p99_us: float
    min_us: float
    max_us: float
    throughput_per_sec: float
    error_count: int

    @classmethod
    def from_latencies(cls, name: str, latencies_us: List[float], errors: int) -> "BenchmarkResult":
        """Compute statistics from raw latency measurements."""
        if not latencies_us:
            return cls(name=name, iterations=0, mean_us=0, std_us=0, p50_us=0,
                      p95_us=0, p99_us=0, min_us=0, max_us=0, throughput_per_sec=0,
                      error_count=errors)

        n = len(latencies_us)
        mean = statistics.mean(latencies_us)
        std = statistics.stdev(latencies_us) if n > 1 else 0.0
        sorted_lat = sorted(latencies_us)

        return cls(
            name=name,
            iterations=n,
            mean_us=mean,
            std_us=std,
            p50_us=sorted_lat[n // 2],
            p95_us=sorted_lat[int(n * 0.95)],
            p99_us=sorted_lat[int(n * 0.99)],
            min_us=sorted_lat[0],
            max_us=sorted_lat[-1],
            throughput_per_sec=1_000_000 / mean if mean > 0 else 0,
            error_count=errors,
        )


def generate_sample() -> List[float]:
    """Generate a single MNIST-like sample (784 floats)."""
    return [(j % 256) / 255.0 for j in range(INPUT_DIM)]


def benchmark_realizar(base_url: str) -> BenchmarkResult:
    """Benchmark Realizar HTTP endpoint."""
    url = f"{base_url.rstrip('/')}/predict"
    sample = generate_sample()
    payload = {"features": sample}

    # Health check
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/health", timeout=5)
        if resp.status_code != 200:
            print(f"ERROR: Realizar health check failed: {resp.status_code}")
            return BenchmarkResult.from_latencies("Realizar", [], 1)
    except Exception as e:
        print(f"ERROR: Cannot connect to Realizar: {e}")
        return BenchmarkResult.from_latencies("Realizar", [], 1)

    print("Warming up Realizar...")
    for _ in range(WARMUP_REQUESTS):
        requests.post(url, json=payload)

    print(f"Benchmarking Realizar ({BENCHMARK_REQUESTS} requests)...")
    latencies_us: List[float] = []
    errors = 0

    for i in range(BENCHMARK_REQUESTS):
        start = time.perf_counter_ns()
        try:
            resp = requests.post(url, json=payload)
            if resp.status_code != 200:
                errors += 1
                continue
        except Exception:
            errors += 1
            continue
        end = time.perf_counter_ns()
        latencies_us.append((end - start) / 1000.0)  # ns to µs

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{BENCHMARK_REQUESTS}")

    return BenchmarkResult.from_latencies("Realizar", latencies_us, errors)


def benchmark_torchserve(base_url: str, model_name: str = "mnist") -> BenchmarkResult:
    """Benchmark TorchServe HTTP endpoint."""
    url = f"{base_url.rstrip('/')}/predictions/{model_name}"
    sample = generate_sample()
    # TorchServe format varies by handler - adjust as needed
    payload = {"data": sample}

    # Health check
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/ping", timeout=5)
        if resp.status_code != 200:
            print(f"ERROR: TorchServe health check failed: {resp.status_code}")
            return BenchmarkResult.from_latencies("TorchServe", [], 1)
    except Exception as e:
        print(f"ERROR: Cannot connect to TorchServe: {e}")
        return BenchmarkResult.from_latencies("TorchServe", [], 1)

    print("Warming up TorchServe...")
    for _ in range(WARMUP_REQUESTS):
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception:
            pass

    print(f"Benchmarking TorchServe ({BENCHMARK_REQUESTS} requests)...")
    latencies_us: List[float] = []
    errors = 0

    for i in range(BENCHMARK_REQUESTS):
        start = time.perf_counter_ns()
        try:
            resp = requests.post(url, json=payload, timeout=5)
            if resp.status_code != 200:
                errors += 1
                continue
        except Exception:
            errors += 1
            continue
        end = time.perf_counter_ns()
        latencies_us.append((end - start) / 1000.0)

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{BENCHMARK_REQUESTS}")

    return BenchmarkResult.from_latencies("TorchServe", latencies_us, errors)


def print_result(result: BenchmarkResult):
    """Print formatted benchmark result."""
    print(f"\n## {result.name}")
    print(f"  Requests: {result.iterations:,}")
    print(f"  Errors: {result.error_count}")
    print(f"  Mean: {result.mean_us:.2f} µs")
    print(f"  Std Dev: {result.std_us:.2f} µs")
    print(f"  p50: {result.p50_us:.2f} µs")
    print(f"  p95: {result.p95_us:.2f} µs")
    print(f"  p99: {result.p99_us:.2f} µs")
    print(f"  Throughput: {result.throughput_per_sec:,.0f} req/sec")


def main():
    parser = argparse.ArgumentParser(
        description="HTTP Serving Benchmark: TorchServe vs Realizar"
    )
    parser.add_argument("--realizar", type=str, help="Realizar server URL (e.g., http://localhost:3000)")
    parser.add_argument("--torchserve", type=str, help="TorchServe server URL (e.g., http://localhost:8080)")
    parser.add_argument("--model-name", type=str, default="mnist", help="TorchServe model name")
    args = parser.parse_args()

    if not args.realizar and not args.torchserve:
        print("ERROR: Specify at least one server: --realizar or --torchserve")
        sys.exit(1)

    print("=" * 70)
    print("HTTP Serving Benchmark: MNIST Classification")
    print("=" * 70)
    print()
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"Warmup requests: {WARMUP_REQUESTS}")
    print(f"Benchmark requests: {BENCHMARK_REQUESTS}")
    print(f"Input dimension: {INPUT_DIM}")
    print()

    results = []

    if args.realizar:
        result = benchmark_realizar(args.realizar)
        print_result(result)
        results.append(result)

    if args.torchserve:
        result = benchmark_torchserve(args.torchserve, args.model_name)
        print_result(result)
        results.append(result)

    # Comparison
    if len(results) == 2 and all(r.iterations > 0 for r in results):
        r1, r2 = results
        speedup = r2.p50_us / r1.p50_us if r1.p50_us > 0 else 0
        print()
        print("=" * 70)
        print("COMPARISON")
        print("=" * 70)
        print()
        print(f"| Framework | p50 (µs) | p99 (µs) | Throughput/sec |")
        print(f"|-----------|----------|----------|----------------|")
        for r in results:
            print(f"| {r.name:<9} | {r.p50_us:>8.0f} | {r.p99_us:>8.0f} | {r.throughput_per_sec:>14,.0f} |")
        print()
        print(f"Speedup (p50): {speedup:.1f}x")

    # Save results
    output = {
        "config": {
            "warmup_requests": WARMUP_REQUESTS,
            "benchmark_requests": BENCHMARK_REQUESTS,
            "input_dim": INPUT_DIM,
            "platform": f"{platform.system()} {platform.release()}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "results": [asdict(r) for r in results],
    }

    output_path = Path(__file__).parent / "serve_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
