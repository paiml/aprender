# Trueno Examples

This directory contains examples demonstrating Trueno's high-performance compute capabilities and comparisons with NumPy/PyTorch.

## 📁 Examples Overview

### Rust Examples (Trueno Native)

| Example | Description | Command |
|---------|-------------|---------|
| `quickstart.rs` | **⭐ Start here!** All core features in one file | `cargo run --example quickstart` |
| `performance_demo.rs` | Compare Scalar vs SSE2/AVX backends | `cargo run --release --example performance_demo` |
| `matrix_operations.rs` | Matrix multiplication and transpose | `cargo run --release --example matrix_operations` |
| `activation_functions.rs` | Neural network activations (ReLU, Sigmoid, etc.) | `cargo run --release --example activation_functions` |
| `backend_detection.rs` | Auto-detection of available SIMD backends | `cargo run --release --example backend_detection` |
| `ml_similarity.rs` | Cosine similarity for ML applications | `cargo run --release --example ml_similarity` |
| `symmetric_eigen.rs` | Eigendecomposition for PCA/spectral analysis | `cargo run --release --example symmetric_eigen` |
| `hash_demo.rs` | SIMD-optimized hashing for KV stores | `cargo run --release --example hash_demo` |
| `gpu_batch_demo.rs` | GPU batch operations (requires `gpu` feature) | `cargo run --release --features gpu --example gpu_batch_demo` |
| `gpu_monitor_demo.rs` | GPU monitoring and metrics | `cargo run --release --features gpu --example gpu_monitor_demo` |
| `gpu_tiled_reduction.rs` | GPU tiled reduction operations | `cargo run --release --features gpu --example gpu_tiled_reduction` |
| `tiled_reduction_demo.rs` | TensorView and PartitionView demo | `cargo run --release --example tiled_reduction_demo` |
| `perf_tui.rs` | Interactive TUI performance dashboard | `cargo run --release --example perf_tui` |
| `regression_test.rs` | Numerical regression testing | `cargo run --release --example regression_test` |
| `vocab_bench.rs` | Vocabulary processing benchmark | `cargo run --release --example vocab_bench` |
| `profile_vocab.rs` | Vocabulary profiling | `cargo run --release --example profile_vocab` |
| `execution_graph.rs` | Execution path graph demo | `cargo run --release --features execution-graph --example execution_graph` |
| `ml_tuner_demo.rs` | ML-based kernel selection | `cargo run --release --features ml-tuner --example ml_tuner_demo` |
| `ml_tuner_evolution.rs` | ML tuner evolution demo | `cargo run --release --features ml-tuner --example ml_tuner_evolution` |
| `model_tracing.rs` | Model-level inference tracing | `cargo run --release --example model_tracing` |
| `tiling_demo.rs` | Tiling compute blocks | `cargo run --release --example tiling_demo` |
| `tile_profiler_demo.rs` | Tile profiler demo | `cargo run --release --example tile_profiler_demo` |
| `blis_benchmark.rs` | BLIS-style GEMM benchmark | `cargo run --release --example blis_benchmark` |
| `simd_comparison.rs` | SIMD backend comparison | `cargo run --release --example simd_comparison` |
| `simd_softmax_quant.rs` | SIMD softmax + quantization | `cargo run --release --example simd_softmax_quant` |

#### Benchmark Examples

| Example | Description | Command |
|---------|-------------|---------|
| `benchmark_matrix_suite.rs` | Matrix operation benchmarks | `cargo run --release --example benchmark_matrix_suite` |
| `benchmark_matvec.rs` | Matrix-vector multiplication | `cargo run --release --example benchmark_matvec` |
| `benchmark_matvec_parallel.rs` | Parallel matrix-vector ops | `cargo run --release --example benchmark_matvec_parallel` |
| `benchmark_parallel.rs` | Parallel computation benchmarks | `cargo run --release --example benchmark_parallel` |

### CUDA/PTX Examples (trueno-gpu)

| Example | Description | Command |
|---------|-------------|---------|
| `ptx_quickstart` | **⭐ Start here!** Basic PTX code generation | `cargo run -p trueno-gpu --example ptx_quickstart` |
| `gemm_kernel` | GEMM kernel generation (naive/tiled) | `cargo run -p trueno-gpu --example gemm_kernel` |
| `cuda_monitor` | GPU monitoring and metrics | `cargo run -p trueno-gpu --example cuda_monitor` |
| `flash_attention_cuda` | Flash Attention implementation | `cargo run -p trueno-gpu --example flash_attention_cuda` |
| `simple_attention_cuda` | Basic multi-head attention | `cargo run -p trueno-gpu --example simple_attention_cuda` |
| `q4k_gemm` | Quantized GEMM (Q4_K format) | `cargo run -p trueno-gpu --example q4k_gemm` |
| `q5k_q6k_gemm` | Q5_K/Q6_K quantized GEMM (PARITY-116/117) | `cargo run -p trueno-gpu --example q5k_q6k_gemm` |
| `register_allocation` | PTX register allocation demo | `cargo run -p trueno-gpu --example register_allocation` |
| `gpu_pixels_render` | GPU pixel rendering | `cargo run -p trueno-gpu --example gpu_pixels_render` |
| `dump_ptx` | Dump raw PTX output | `cargo run -p trueno-gpu --example dump_ptx` |
| `satd_kernels` | SATD (video codec) kernels | `cargo run -p trueno-gpu --example satd_kernels` |
| `lz4_compression` | **🗜️ LZ4 compression kernel** | `cargo run -p trueno-gpu --example lz4_compression` |
| `lz4_file_compress` | **📦 LZ4 file compression CLI** | `cargo run -p trueno-gpu --example lz4_file_compress -- bench` |
| `ptx_optimize` | PTX optimization passes demo | `cargo run -p trueno-gpu --example ptx_optimize` |
| `bench_kernel_gen` | Kernel generation benchmarks | `cargo run -p trueno-gpu --example bench_kernel_gen` |

**Note**: PTX generation examples work without a GPU. Runtime examples (cuda_monitor, flash_attention_cuda) require an NVIDIA GPU with CUDA drivers.

### Python Examples (NumPy/PyTorch Comparison)

| Example | Description | Command |
|---------|-------------|---------|
| `dot_product_comparison.py` | **⚡ Dot product benchmark** | `uv run examples/dot_product_comparison.py` |
| `matrix_multiply_comparison.py` | **🔢 Matrix multiplication benchmark** | `uv run examples/matrix_multiply_comparison.py` |
| `activation_comparison.py` | **🧠 Activation functions benchmark** | `uv run examples/activation_comparison.py` |

## 🚀 Quick Start

### Running Python Comparisons

1. Install UV (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Run any comparison example (UV handles dependencies automatically):
```bash
uv run examples/dot_product_comparison.py
uv run examples/matrix_multiply_comparison.py
uv run examples/activation_comparison.py
```

Alternatively, use the benchmarks environment:
```bash
cd benchmarks
uv run ../examples/dot_product_comparison.py
```

### Running Rust Examples

```bash
cargo run --release --example performance_demo
cargo run --release --example matrix_operations
```

## 📊 What Do These Examples Show?

### Dot Product Comparison (`dot_product_comparison.py`)

**Key Insights:**
- Demonstrates compute-intensive operations where SIMD excels
- Shows NumPy vs PyTorch performance characteristics
- Highlights Trueno's 1.6x advantage over NumPy (11.9x over scalar)

**Expected Output:**
```
Size          NumPy (μs)      PyTorch (μs)    Winner          Speedup
--------------------------------------------------------------------------------
100               0.82 ±  0.15      1.85 ±  0.22  NumPy           2.26x
1,000             3.21 ±  0.18      6.45 ±  0.31  NumPy           2.01x
10,000           25.67 ±  1.23     58.32 ±  2.45  NumPy           2.27x
```

**Trueno Context:**
- Trueno AVX-512: **11.9x faster than scalar**
- Trueno AVX-512: **1.6x faster than NumPy**
- Trueno AVX-512: **2.8x faster than PyTorch**

### Matrix Multiplication Comparison (`matrix_multiply_comparison.py`)

**Key Insights:**
- Shows O(n³) complexity scaling
- Demonstrates when GPU acceleration becomes effective
- Highlights optimized BLAS libraries in NumPy

**Expected Output:**
```
Size       NumPy Time           PyTorch Time         Winner       Speedup
------------------------------------------------------------------------------------------
64×64      59.87 μs             125.34 μs            NumPy          2.09x
128×128    434.23 μs            678.45 μs            NumPy          1.56x
256×256    2.67 ms              3.45 ms              NumPy          1.29x
512×512    19.82 ms             25.67 ms             NumPy          1.29x
```

**Trueno Context:**
- SIMD backend: **~7x faster than naive O(n³)** for 128×128
- GPU backend: **2-10x faster** than scalar for 500×500+
- Automatic backend selection based on matrix size

### Activation Functions Comparison (`activation_comparison.py`)

**Key Insights:**
- Compares common ML activation functions
- Shows relative costs (ReLU << Tanh < Sigmoid < Exp)
- Demonstrates SIMD benefits for transcendental functions

**Expected Output:**
```
Activation      NumPy (μs)      PyTorch (μs)    Winner       Speedup
------------------------------------------------------------------------------------------
ReLU                2.34 ±  0.12      5.67 ±  0.23  NumPy          2.42x
Sigmoid            15.67 ±  0.45     32.34 ±  1.12  NumPy          2.06x
Tanh                8.92 ±  0.34     18.45 ±  0.67  NumPy          2.07x
Exp                12.45 ±  0.56     28.91 ±  1.23  NumPy          2.32x
```

**Trueno Context:**
- SIMD-optimized implementations
- 2-4x speedup for compute-intensive activations
- Zero Python overhead for ML inference

## 🎯 Performance Summary

| Operation | Trueno vs Scalar | Trueno vs NumPy | Trueno vs PyTorch |
|-----------|-----------------|-----------------|-------------------|
| **Dot Product** | 11.9x faster | 1.6x faster | 2.8x faster |
| **Matrix Multiply** | 7x faster (128×128) | ~1x (competitive) | ~1.5x faster |
| **Activations** | 2-4x faster | ~1x (competitive) | ~2x faster |

## 💡 When to Use Trueno

✅ **Ideal Use Cases:**
- Real-time systems requiring predictable latency
- Embedded systems without Python runtime
- WebAssembly deployment (browser/edge)
- ML inference pipelines in Rust
- Systems programming with high-performance compute needs

⚠️ **When NumPy/PyTorch May Be Better:**
- Rapid prototyping in Python
- Large ecosystem of Python ML libraries
- Training large neural networks (PyTorch GPU)
- Interactive data exploration (Jupyter notebooks)

## 📚 More Resources

- **Comprehensive Benchmarks**: See `benchmarks/README.md`
- **Performance Analysis**: See `docs/performance-analysis.md`
- **API Documentation**: See `docs/` directory
- **Project README**: See root `README.md`

## 🤝 Contributing

To add new examples:

1. **Rust examples**: Add to this directory with `.rs` extension
2. **Python examples**: Add comparison scripts with NumPy/PyTorch
3. **Update this README**: Document the new example
4. **Follow TDD**: Ensure examples are well-tested

See `CLAUDE.md` for development guidelines.

---

**Last Updated**: 2026-01-25
**Version**: v0.14.1
**Contact**: [GitHub Issues](https://github.com/paiml/trueno/issues)
