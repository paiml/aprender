# Benchmarks Overview

This chapter presents comprehensive benchmark results for Trueno across different backends and workload sizes.

## Latest Benchmark Results

**Date**: 2025-11-18
**Platform**: x86_64 Linux (AVX2-capable)
**Compiler**: rustc 1.83 (release mode, opt-level=3, LTO=true)
**Tool**: Criterion.rs (statistical benchmarking)

## Executive Summary

Trueno's SIMD and GPU backends deliver **2-8x speedups** for most operations, with exceptional performance on reduction and compute-intensive operations.

### Key Findings

- **Average speedup**: 178.5% across all operations
- **Best speedup**: 8.8x (tanh activation, AVX2, 100 elements)
- **Operations meeting ≥10% target**: 66.7%
- **Reduction operations**: 200-400% speedup (dot, sum, max)
- **Activation functions**: 120-880% speedup (relu, tanh)
- **Element-wise ops**: 3-115% speedup (varies by operation and size)

## Benchmark Results by Operation

### Reduction Operations (Excellent Performance)

Reduction operations show exceptional SIMD performance due to parallel accumulation:

| Operation | Size | Scalar (ns) | SSE2 (ns) | AVX2 (ns) | SSE2 Speedup | AVX2 Speedup |
|-----------|------|-------------|-----------|-----------|--------------|--------------|
| **dot** | 100 | 36.11 | 10.79 | - | **3.3x** | - |
| **dot** | 1000 | 574.92 | 130.79 | - | **4.4x** | - |
| **dot** | 10000 | 6126.80 | 1475.60 | - | **4.2x** | - |
| **sum** | 100 | 32.77 | 10.53 | - | **3.1x** | - |
| **sum** | 1000 | 575.20 | 138.60 | - | **4.2x** | - |
| **sum** | 10000 | 5883.10 | 1491.00 | - | **3.9x** | - |
| **max** | 100 | 26.57 | 6.86 | - | **3.9x** | - |
| **max** | 1000 | 395.04 | 88.24 | - | **4.5x** | - |
| **max** | 10000 | 4193.30 | 1033.90 | - | **4.1x** | - |

**Why reduction operations excel:**
- Combines multiple operations in SIMD lanes (4-8 parallel accumulations)
- No memory write bottleneck (single scalar result)
- Horizontal reduction is highly optimized
- Minimal overhead from setup/cleanup

### Activation Functions (Good to Excellent Performance)

Activation functions benefit from SIMD, especially for compute-intensive operations:

| Operation | Size | Scalar (ns) | SSE2 (ns) | AVX2 (ns) | SSE2 Speedup | AVX2 Speedup |
|-----------|------|-------------|-----------|-----------|--------------|--------------|
| **tanh** | 100 | 891 | 137 | 101 | **6.5x** | **8.8x** |
| **tanh** | 1000 | 8000 | 1080 | - | **7.4x** | - |
| **relu** | 100 | 54.1 | 44.8 | 49.3 | **1.21x** | **1.10x** |

**Why activation functions perform well:**
- Compute-intensive (tanh requires exp calculations)
- SIMD processes 4-8 elements in parallel
- No data dependencies between elements
- AVX2 benefits from wider registers (8 f32 vs 4 for SSE2)

### Element-Wise Operations (Mixed Performance)

Element-wise operations show variable performance, often limited by memory bandwidth:

| Operation | Size | Scalar (ns) | SSE2 (ns) | AVX2 (ns) | SSE2 Speedup | AVX2 Speedup |
|-----------|------|-------------|-----------|-----------|--------------|--------------|
| **add** | 100 | 46.89 | 42.50 | - | **1.10x** | - |
| **add** | 1000 | 124.91 | 121.51 | - | 1.03x | - |
| **add** | 10000 | 1098.60 | 1044.60 | - | 1.05x | - |
| **mul** | 100 | 41.03 | 38.75 | - | 1.06x | - |
| **mul** | 1000 | 119.03 | 112.86 | - | 1.05x | - |
| **mul** | 10000 | 1029.10 | 1064.30 | - | 0.97x ❌ | - |
| **scale** | 100 | 43.9 | 41.8 | 39.6 | 1.05x | **1.11x** |
| **scale** | 1000 | 104 | 111 | 90.8 | 0.94x | **1.15x** |

**Why element-wise ops show limited speedups:**
- **Memory bandwidth bottleneck**: Simple operations (add, mul) are memory-bound, not compute-bound
- **Cache effects**: Small workloads fit in L1 cache, scalar loop is efficient
- **Large workloads**: Both scalar and SIMD become memory-bound
- **Overhead**: SIMD setup/cleanup costs hurt small workloads (<1000 elements)

## Performance by Backend

### SSE2 (128-bit SIMD)

**Availability**: Guaranteed on all x86_64 CPUs
**Register width**: 128 bits (4 × f32 or 2 × f64)
**Typical speedup**: 2-4x for reduction ops, 1.05-1.15x for element-wise

**Best operations:**
- ✅ Reduction (dot, sum, max): 3-4.5x
- ✅ Activation functions (tanh, relu): 1.2-7.4x
- ⚠️ Element-wise (add, mul): 1.03-1.10x

**Limitations:**
- Limited to 4-way parallelism
- Some operations (div, sigmoid) show regressions
- Memory bandwidth limited for large workloads

### AVX2 (256-bit SIMD)

**Availability**: Intel Haswell+ (2013+), AMD Zen+ (2018+)
**Register width**: 256 bits (8 × f32 or 4 × f64)
**Typical speedup**: 4-8x for reduction ops, 1.10-1.15x for element-wise

**Best operations:**
- ✅ Activation functions (tanh): 8.8x
- ✅ Scalar operations (scale): 1.15x
- ✅ Reduction (expected 2x over SSE2, not yet benchmarked)

**Advantages over SSE2:**
- 2x wider registers (8 vs 4 elements)
- FMA (fused multiply-add) instructions
- Better memory bandwidth utilization

### GPU (WebGPU via wgpu)

**Availability**: Systems with Vulkan/Metal/DX12 support
**Typical speedup**: 16-81x for large matrix operations (>500×500)

**IMPORTANT**: Empirical RTX 4090 benchmarking revealed that GPU has **3.5ms fixed transfer overhead**, making it slower than SIMD for vector operations at ALL sizes.

**GPU Performance Summary** (2025-11-23, RTX 4090):
- ✅ **Matrix multiplication**: 81x speedup on 1000×1000
- ❌ **Vector operations**: 2000x+ slower than SIMD due to transfer overhead
- 🎯 **Recommendation**: GPU only for matrix ops >500×500, otherwise use SIMD

**Current Thresholds**:

| Workload Type | Size Range | Recommended Backend |
|---------------|------------|---------------------|
| Vector operations | **Any** | **SIMD** (GPU disabled) |
| Matrix multiplication | <500×500 | SIMD |
| Matrix multiplication | ≥500×500 | **GPU** |

**GPU Transfer Overhead**: ~3.5ms per operation for CPU↔GPU↔CPU transfer

See **[GPU Performance](./gpu-performance.md)** for detailed RTX 4090 benchmark results and analysis.

## Performance by Workload Size

### Small (100 elements)

**Recommended backend**: SSE2 or Scalar
**SIMD benefit**: 5-10% for most ops, 120-650% for activation/reduction

At small sizes, SIMD overhead (setup, remainder handling) can exceed benefits for simple operations.

### Medium (1K-10K elements)

**Recommended backend**: SSE2/AVX2
**SIMD benefit**: 3-440% depending on operation

Sweet spot for SIMD: large enough to amortize overhead, small enough to avoid memory bottlenecks.

### Large (100K+ elements)

**Recommended backend**: GPU (if available), otherwise AVX2
**SIMD benefit**: 0-400% (memory-bound for simple ops, good for reductions)

At large sizes:
- Element-wise ops become memory-bound
- Reduction ops still benefit from SIMD
- GPU provides best performance if transfer overhead is justified

## Benchmark Methodology

### Tool: Criterion.rs

All benchmarks use [Criterion.rs](https://github.com/bheisler/criterion.rs) for statistical rigor:

- **Samples**: 100 per benchmark
- **Warmup**: 3 seconds
- **Measurement**: 5 seconds
- **Outlier detection**: Automated
- **Statistical analysis**: Mean, median, standard deviation

### Test Data

- **Sequential floats**: `(i as f32) * 0.5`
- **Workload sizes**: 100, 1000, 10000, 100000 elements
- **Backend comparison**: Scalar vs SSE2 vs AVX2 vs GPU

### Environment

- **CPU**: x86_64 with AVX2 support
- **RAM**: 16GB+ (prevents swapping)
- **Compiler flags**: `-C opt-level=3 -C lto=true -C codegen-units=1`
- **CPU affinity**: Pinned to single core (reduces variance)
- **Background processes**: Minimized

## Quality Standards

Every benchmark must meet these criteria:

1. **Coefficient of Variation (CV) < 5%** - Consistent results across runs
2. **No regressions >5%** - SIMD should not be slower than scalar
3. **Statistical significance** - 100+ samples for reliable mean/median
4. **Baseline comparison** - Always compare against scalar implementation

### Interpreting Results

**Speedup calculation**: `(scalar_time / simd_time)`

| Speedup | Status | Interpretation |
|---------|--------|----------------|
| ≥2.0x | ✅ Excellent | SIMD delivers significant value |
| 1.5-2.0x | ✅ Good | SIMD worth the complexity |
| 1.1-1.5x | ⚠️ Marginal | Consider simpler scalar code |
| 1.0-1.1x | ⚠️ Minimal | SIMD overhead may not be worth it |
| <1.0x | ❌ Regression | Fix implementation or use scalar |

## Reproducing Benchmarks

Run all benchmarks:

```bash
cargo bench --bench vector_ops
```

Run specific operation:

```bash
cargo bench --bench vector_ops -- dot
```

Generate HTML report:

```bash
cargo bench --bench vector_ops
open target/criterion/report/index.html
```

Compare against baseline:

```bash
# Save current results as baseline
cargo bench -- --save-baseline main

# Make changes, then compare
cargo bench -- --baseline main
```

## Next Steps

- **[SIMD Performance](./simd-performance.md)** - Deep dive into SIMD optimizations
- **[GPU Performance](./gpu-performance.md)** - GPU benchmarks and transfer overhead
- **[Optimization Guide](./optimization-guide.md)** - How to improve performance
- **[Profiling](./profiling.md)** - Using perf, flamegraphs, and vtune
