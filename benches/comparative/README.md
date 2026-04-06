# Comparative Benchmarks: Aprender vs PyTorch

Scientifically reproducible benchmarks comparing **Aprender (.apr)** inference performance against **PyTorch**.

## Quick Start

```bash
# 1. Run Aprender benchmark (from project root)
cargo run --example mnist_apr_benchmark --release --features aprender-serve

# 2. Run PyTorch benchmark
cd benches/comparative
uv sync
uv run mnist_benchmark.py

# 3. Generate comparison report
uv run compare_mnist.py
```

## Results

See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for the full comparison report.

### Summary (Example Results)

| Metric | Aprender | PyTorch | Speedup |
|--------|----------|---------|---------|
| p50 (us) | ~3-5 | ~30-50 | **10-15x** |
| Throughput/sec | ~200-300K | ~20-30K | **10-15x** |

## Methodology

Following established benchmarking guidelines:

- **Box et al. (2005)**: Statistics for Experimenters
- **Georges et al. (2007)**: Statistically Rigorous Java Performance Evaluation
- **MLPerf Inference**: Industry standard methodology

### Key Principles

1. **Reproducibility**: Fixed random seed (42), identical data generation
2. **Statistical rigor**: 10,000 iterations, 100 warmup, 95% CI
3. **Fair comparison**: Same model architecture, same input, same machine
4. **Transparency**: Full methodology documented, all code open source

### What We Compare

```
┌─────────────────────────────────────────────────────────────┐
│                    IDENTICAL TASK                           │
│         LogisticRegression: 784 inputs → 10 classes         │
│         Input: Single MNIST sample (784 floats)             │
│         Measurement: Single inference latency               │
└─────────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
   ┌───────────────┐               ┌───────────────┐
   │   PyTorch     │               │   Aprender    │
   │   ─────────   │               │   ─────────   │
   │ torch.nn      │               │ aprender::    │
   │ .Linear(784,  │               │ Logistic      │
   │  10)          │               │ Regression    │
   └───────────────┘               └───────────────┘
```

## File Structure

```
benches/comparative/
├── README.md                    # This file
├── BENCHMARK_RESULTS.md         # Full comparison report (generated)
├── pyproject.toml               # Python dependencies
├── mnist_benchmark.py           # PyTorch benchmark script
├── compare_mnist.py             # Comparison analysis script
├── pytorch_mnist_results.json   # PyTorch results (generated)
├── aprender_mnist_results.json  # Aprender results (generated)
└── comparison_summary.json      # Summary metrics (generated)
```

## Statistical Methods

### Measurements

- **Warmup**: 100 iterations (excluded from analysis)
- **Sample size**: 10,000 iterations
- **Metrics**: Mean, Std Dev, p50, p95, p99, Min, Max
- **Confidence interval**: 95% CI using t-distribution

### Significance Testing

- **Test**: Welch's t-test (unequal variances)
- **Null hypothesis**: mean(Aprender) = mean(PyTorch)
- **Significance level**: α = 0.05

### Effect Size

- **Measure**: Cohen's d
- **Interpretation**: negligible (<0.2), small (0.2-0.5), medium (0.5-0.8), large (>0.8)

## Configuration

Both benchmarks use identical configuration:

| Parameter | Value |
|-----------|-------|
| Seed | 42 |
| Input dimensions | 784 (28×28) |
| Output classes | 10 |
| Training samples | 1,000 |
| Warmup iterations | 100 |
| Benchmark iterations | 10,000 |

### Data Generation

test MNIST data using deterministic formula:

```python
# Python
pixel = ((i * 17 + j * 31) % 256) / 255.0

# Rust (identical)
let pixel = ((i * 17 + j * 31) % 256) as f32 / 255.0;
```

## Why Aprender is Faster

1. **Zero Python overhead**: No interpreter, no GIL, no dynamic dispatch
2. **Native compilation**: Rust compiles to optimized machine code
3. **No tensor framework overhead**: Direct matrix operations
4. **Predictable performance**: No JIT warmup, consistent latency
5. **Memory efficiency**: No Python object overhead

## Requirements

### Rust Side

```toml
# Cargo.toml
[dependencies]
aprender = { version = "0.10", features = ["default"] }
```

### Python Side

```toml
# pyproject.toml
[project]
dependencies = [
    "torch>=2.0",
    "torchvision>=0.15",
]
```

## References

1. Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). *Statistics for Experimenters: Design, Innovation, and Discovery*. Wiley.

2. Georges, A., Buytaert, D., & Eeckhout, L. (2007). Statistically Rigorous Java Performance Evaluation. *OOPSLA '07*.

3. MLPerf Inference: https://mlcommons.org/en/inference-datacenter-21/

## Citation

If you use these benchmarks, please cite:

```bibtex
@software{realizar,
  title = {Realizar: Pure Rust ML Inference Engine},
  author = {Pragmatic AI Labs},
  url = {https://github.com/paiml/realizar},
  year = {2025}
}

@software{aprender,
  title = {Aprender: Pure Rust Machine Learning Library},
  author = {Pragmatic AI Labs},
  url = {https://github.com/paiml/aprender},
  year = {2025}
}
```
