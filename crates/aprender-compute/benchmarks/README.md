# Trueno Benchmark Suite

Comprehensive performance comparison between **Trueno** (Rust SIMD library), **NumPy** (Python standard), and **PyTorch** (Deep learning framework).

## 📊 Goal

Validate that Trueno achieves **within 20% of NumPy/PyTorch performance** for 1D vector operations (v0.3.0 success criteria).

## 🚀 Quick Start

Run the complete benchmark suite:

```bash
./benchmarks/run_all.sh
```

This will:
1. ✅ Run Trueno benchmarks (Rust/Criterion) - ~5-10 minutes
2. ✅ Run Python benchmarks (NumPy/PyTorch) - ~2-3 minutes
3. ✅ Generate comparison report

**Results**:
- `benchmarks/comparison_report.md` - Human-readable markdown report
- `benchmarks/comparison_summary.json` - Machine-readable JSON data
- `target/criterion/` - Detailed Criterion benchmark data

## 📋 Operations Benchmarked

### Element-wise Operations (9)
- `add`, `sub`, `mul`, `div` - Basic arithmetic
- `scale`, `abs`, `clamp` - Transformations
- `fma`, `lerp` - Advanced operations

### Reductions (8)
- `sum`, `max`, `min` - Basic reductions
- `argmax`, `argmin` - Index finding
- `norm_l1`, `norm_l2`, `norm_linf` - Vector norms

### Activation Functions (8)
- `relu` - Rectified Linear Unit
- `sigmoid`, `tanh` - Classic activations
- `gelu`, `swish` - Modern transformer activations
- `exp`, `ln`, `log2` - Transcendental functions

**Total**: 25 operations × 5 sizes = 125 benchmark configurations

## 📏 Test Sizes

- **100** elements - Small vectors (cache-friendly)
- **1,000** elements - Medium vectors
- **10,000** elements - Large vectors (SIMD sweet spot)
- **100,000** elements - Very large (memory-bound)
- **1,000,000** elements - Extreme (bandwidth-limited)

## 🎯 Success Criteria (v0.3.0)

| Criteria | Target | Status |
|----------|--------|--------|
| Within 20% of NumPy | ≥80% of operations | 🔄 Testing |
| Faster than NumPy | ≥40% of operations | 🔄 Testing |
| Faster than PyTorch | ≥50% of operations | 🔄 Testing |

## 🔧 Running Individual Components

### Rust Benchmarks Only

```bash
cargo bench --all-features
```

Results in: `target/criterion/<operation>/<backend>/<size>/`

### Python Benchmarks Only

```bash
uv run benchmarks/python_comparison.py
```

Results in: `benchmarks/python_results.json`

### Generate Comparison Report

```bash
uv run benchmarks/compare_results.py
```

Requires both Rust and Python benchmarks to be run first.

## 📦 Dependencies

### Rust
- Criterion.rs (included in `dev-dependencies`)
- Trueno with all features enabled

### Python (via UV - Rust-based package manager)
```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
cd benchmarks
uv pip install numpy torch
```

UV is a Rust-based Python package manager that's significantly faster than pip.
Dependencies are defined in `benchmarks/pyproject.toml`.

## 📈 Expected Results

### Memory-Bound Operations (~1x SIMD benefit)
- `add`, `sub`, `mul`, `div`, `scale`, `abs`
- **Why**: Memory bandwidth saturation limits SIMD advantage
- **Expectation**: Trueno ≈ NumPy (both memory-bound)

### Compute-Bound Operations (4-12x SIMD benefit)
- `dot`, `sum`, `max`, `min`, `norm_l2`, `norm_l1`
- **Why**: SIMD parallelism fully utilized
- **Expectation**: Trueno 1.2-2x faster than NumPy

### Activation Functions (2-4x SIMD benefit)
- `relu`, `sigmoid`, `tanh`, `gelu`, `swish`
- **Why**: Moderate computational intensity
- **Expectation**: Trueno ≈ NumPy (both optimized)

## 🔍 Interpreting Results

### Performance Ratios

- **< 1.0**: Trueno is **faster**
- **0.8 - 1.2**: Trueno is **within 20%** (success criteria) ✅
- **> 1.2**: Trueno is **slower** ⚠️

### Example Output

```
✅ 0.85x - Trueno is 1.18x faster than NumPy
✓  1.15x - Trueno is within 20% of NumPy
⚠️ 1.45x - Trueno is 1.45x slower than NumPy
```

## 🐛 Troubleshooting

### Python dependencies missing

```bash
cd benchmarks && uv pip install numpy torch
```

### Criterion benchmarks not found

```bash
cargo bench --all-features
```

Must run Rust benchmarks before comparison.

### Permission denied on run_all.sh

```bash
chmod +x benchmarks/run_all.sh
```

## 📊 Viewing Results

### Markdown Report

```bash
cat benchmarks/comparison_report.md
```

Or open in any markdown viewer.

### JSON Data

```bash
python3 -m json.tool benchmarks/comparison_summary.json
```

### Criterion HTML Reports

```bash
open target/criterion/report/index.html  # macOS
xdg-open target/criterion/report/index.html  # Linux
```

## 🎓 Methodology

### Trueno (Rust)
- Uses Criterion.rs for statistical rigor
- Warm-up iterations to stabilize caches
- Outlier detection and removal
- Reports mean ± std deviation

### Python (NumPy/PyTorch)
- Uses `time.perf_counter()` for nanosecond precision
- 100 iterations per benchmark
- Statistical analysis (mean, std deviation)
- Separate CPU-only measurements

### Fairness
- All frameworks use **float32** (f32) precision
- Same random input data generation
- CPU-only comparisons (no GPU acceleration)
- Single-threaded execution

## 📚 References

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [NumPy Benchmarks](https://numpy.org/doc/stable/benchmarking.html)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

## 🤝 Contributing

To add new benchmarks:

1. **Rust**: Add to `benches/vector_ops.rs` or `benches/matrix_ops.rs`
2. **Python**: Add to `benchmarks/python_comparison.py`
3. **Update**: Add operation to comparison analysis

See `CLAUDE.md` for development guidelines.

---

**Last Updated**: 2025-11-20
**Version**: v0.3.0-rc
**Contact**: [GitHub Issues](https://github.com/paiml/trueno/issues)
