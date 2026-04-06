#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.0",
#     "numpy>=1.24",
#     "pynvml>=11.0",
# ]
# ///
"""
CUDA Training Benchmark: PyTorch vs Entrenar

Compares training performance between PyTorch and Entrenar's CUDA kernels.
Measures GPU utilization, tokens/second, and memory bandwidth.

SPEC-FT-001 v3.3.0 Targets:
  - GPU Utilization: >70%
  - Throughput: >100 tokens/second

Run with:
    uv run scripts/benchmark_cuda_training.py

    Or directly (uv handles dependencies automatically):
    ./scripts/benchmark_cuda_training.py

Prerequisites:
    - NVIDIA GPU with CUDA support
    - uv installed: curl -LsSf https://astral.sh/uv/install.sh | sh
"""

import subprocess
import time
import json
from dataclasses import dataclass
from typing import Optional
import sys


@dataclass
class BenchmarkResult:
    """Results from a training benchmark run"""
    name: str
    total_time_s: float
    tokens_processed: int
    tokens_per_second: float
    avg_gpu_utilization: float
    peak_memory_mb: float
    forward_time_s: float
    backward_time_s: float
    optimizer_time_s: float


def check_cuda_available() -> bool:
    """Check if CUDA is available via PyTorch"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_gpu_info() -> dict:
    """Get GPU information using pynvml"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return {
            "name": name,
            "total_memory_gb": memory.total / 1e9,
            "free_memory_gb": memory.free / 1e9,
        }
    except Exception as e:
        return {"name": "Unknown", "total_memory_gb": 0, "free_memory_gb": 0, "error": str(e)}


def monitor_gpu_utilization(duration_s: float, interval_s: float = 0.1) -> list[float]:
    """Monitor GPU utilization over a duration"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        utilizations = []
        start = time.perf_counter()
        while time.perf_counter() - start < duration_s:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utilizations.append(util.gpu)
            time.sleep(interval_s)

        pynvml.nvmlShutdown()
        return utilizations
    except Exception:
        return []


def benchmark_pytorch_training(
    batch_size: int = 8,
    seq_len: int = 128,
    hidden_size: int = 768,
    vocab_size: int = 32000,
    num_steps: int = 100,
    warmup_steps: int = 5,
) -> BenchmarkResult:
    """Benchmark PyTorch training loop"""
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  PyTorch device: {device}")

    # Simulate LM head: hidden -> vocab logits
    # This is the bottleneck operation in transformer training
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False).to(device)
    optimizer = torch.optim.AdamW(lm_head.parameters(), lr=1e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Generate synthetic data
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size * seq_len, hidden_size, device=device)
    targets = torch.randint(0, vocab_size, (batch_size * seq_len,), device=device)

    # Warmup
    for _ in range(warmup_steps):
        logits = lm_head(hidden_states)
        loss = loss_fn(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()

    # Benchmark
    forward_times = []
    backward_times = []
    optimizer_times = []

    total_start = time.perf_counter()

    for step in range(num_steps):
        # Forward
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = lm_head(hidden_states)
        loss = loss_fn(logits, targets)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        forward_times.append(t1 - t0)

        # Backward
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        backward_times.append(t3 - t2)

        # Optimizer
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        t5 = time.perf_counter()
        optimizer_times.append(t5 - t4)

    total_time = time.perf_counter() - total_start

    # Get memory usage
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6

    tokens_processed = batch_size * seq_len * num_steps
    tokens_per_second = tokens_processed / total_time

    # Estimate GPU utilization from timing
    kernel_time = sum(forward_times) + sum(backward_times) + sum(optimizer_times)
    gpu_util = (kernel_time / total_time) * 100

    return BenchmarkResult(
        name="PyTorch",
        total_time_s=total_time,
        tokens_processed=tokens_processed,
        tokens_per_second=tokens_per_second,
        avg_gpu_utilization=gpu_util,
        peak_memory_mb=peak_memory_mb,
        forward_time_s=sum(forward_times),
        backward_time_s=sum(backward_times),
        optimizer_time_s=sum(optimizer_times),
    )


def benchmark_entrenar_cuda(
    batch_size: int = 8,
    seq_len: int = 128,
    hidden_size: int = 768,
    vocab_size: int = 32000,
    num_steps: int = 100,
) -> Optional[BenchmarkResult]:
    """
    Benchmark Entrenar CUDA training.

    Runs the cuda_training_benchmark example and parses the output.
    """
    print("  Running Entrenar CUDA benchmark...")

    try:
        # Run the Rust benchmark example
        result = subprocess.run(
            ["cargo", "run", "--example", "cuda_training_benchmark",
             "--release", "--features", "cuda"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd="/home/noah/src/entrenar",
        )

        if result.returncode != 0:
            print(f"  ERROR: Entrenar benchmark failed")
            print(f"  stderr: {result.stderr[:500]}")
            return None

        # Parse output for metrics
        output = result.stdout

        # Extract metrics from output (looking for specific patterns)
        tokens_per_sec = 0.0
        gpu_util = 0.0
        total_time = 0.0

        for line in output.split('\n'):
            if 'Tokens/second:' in line:
                try:
                    tokens_per_sec = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif 'GPU Utilization:' in line:
                try:
                    gpu_util = float(line.split(':')[1].strip().replace('%', ''))
                except (ValueError, IndexError):
                    pass
            elif 'Total wall time:' in line:
                try:
                    total_time = float(line.split(':')[1].strip().replace('s', ''))
                except (ValueError, IndexError):
                    pass

        if tokens_per_sec == 0:
            print("  WARNING: Could not parse Entrenar output")
            print(f"  Output preview: {output[:1000]}")
            return None

        return BenchmarkResult(
            name="Entrenar CUDA",
            total_time_s=total_time,
            tokens_processed=int(tokens_per_sec * total_time),
            tokens_per_second=tokens_per_sec,
            avg_gpu_utilization=gpu_util,
            peak_memory_mb=0,  # Not reported by current benchmark
            forward_time_s=0,
            backward_time_s=0,
            optimizer_time_s=0,
        )

    except subprocess.TimeoutExpired:
        print("  ERROR: Entrenar benchmark timed out")
        return None
    except FileNotFoundError:
        print("  ERROR: cargo not found - is Rust installed?")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def print_results(results: list[BenchmarkResult]):
    """Print benchmark results in a formatted table"""
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\n{'Framework':<20} {'Tokens/s':>12} {'GPU Util':>10} {'Time (s)':>10}")
    print("-" * 52)

    for r in results:
        print(f"{r.name:<20} {r.tokens_per_second:>12.0f} {r.avg_gpu_utilization:>9.1f}% {r.total_time_s:>10.2f}")

    print("\n" + "-" * 70)
    print("  SPEC-FT-001 v3.3.0 VERIFICATION")
    print("-" * 70)

    entrenar = next((r for r in results if "Entrenar" in r.name), None)
    if entrenar:
        tokens_pass = entrenar.tokens_per_second >= 100
        gpu_pass = entrenar.avg_gpu_utilization >= 70

        print(f"\n  [{'PASS' if tokens_pass else 'FAIL'}] Tokens/second >= 100: {entrenar.tokens_per_second:.0f}")
        print(f"  [{'PASS' if gpu_pass else 'FAIL'}] GPU Utilization >= 70%: {entrenar.avg_gpu_utilization:.1f}%")

        if tokens_pass and gpu_pass:
            print("\n  RESULT: ALL TARGETS MET")
        else:
            print("\n  RESULT: TARGETS NOT MET")
    else:
        print("\n  WARNING: Entrenar benchmark did not complete")

    # Compare with PyTorch if both available
    pytorch = next((r for r in results if "PyTorch" in r.name), None)
    if pytorch and entrenar:
        speedup = entrenar.tokens_per_second / pytorch.tokens_per_second if pytorch.tokens_per_second > 0 else 0
        print(f"\n  Speedup vs PyTorch: {speedup:.2f}x")


def main():
    print("=" * 70)
    print("  CUDA Training Benchmark - SPEC-FT-001 v3.3.0")
    print("=" * 70)

    # Check CUDA
    if not check_cuda_available():
        print("\nERROR: CUDA not available")
        print("This benchmark requires an NVIDIA GPU with CUDA support.")
        sys.exit(1)

    # Get GPU info
    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info['name']}")
    print(f"Memory: {gpu_info['total_memory_gb']:.1f} GB total, {gpu_info['free_memory_gb']:.1f} GB free")

    # Configuration
    config = {
        "batch_size": 8,
        "seq_len": 128,
        "hidden_size": 768,
        "vocab_size": 32000,
        "num_steps": 100,
    }

    print(f"\nConfiguration:")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Sequence length: {config['seq_len']}")
    print(f"  Hidden size: {config['hidden_size']}")
    print(f"  Vocab size: {config['vocab_size']}")
    print(f"  Steps: {config['num_steps']}")

    results = []

    # Benchmark PyTorch
    print("\n[1/2] Benchmarking PyTorch...")
    try:
        pytorch_result = benchmark_pytorch_training(**config)
        results.append(pytorch_result)
        print(f"  Completed: {pytorch_result.tokens_per_second:.0f} tokens/s")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Benchmark Entrenar
    print("\n[2/2] Benchmarking Entrenar CUDA...")
    entrenar_result = benchmark_entrenar_cuda(**config)
    if entrenar_result:
        results.append(entrenar_result)
        print(f"  Completed: {entrenar_result.tokens_per_second:.0f} tokens/s")

    # Print results
    if results:
        print_results(results)
    else:
        print("\nNo benchmark results to display.")
        sys.exit(1)


if __name__ == "__main__":
    main()
