# SIMD Performance Analysis

**Date**: 2025-11-18
**System**: x86_64 Linux (AVX2-capable)
**Benchmark Tool**: Criterion.rs

This chapter provides a deep dive into Trueno's SIMD performance characteristics, analyzing when SIMD provides speedups and when it doesn't.

## Executive Summary

Comprehensive benchmarking reveals **mixed results** across operations. While some operations show excellent SIMD speedups (tanh: 6.5-8.8x), many element-wise operations show **minimal** or **negative** speedups, especially for SSE2.

### Key Findings

1. **Activation functions** (relu, tanh): Good to excellent SIMD speedups (1.2-8.8x)
2. **Reduction operations** (dot, sum, max): Excellent SIMD speedups (3-4.5x)
3. **Element-wise operations** (add, sub, div, fma): Minimal or negative SIMD benefit
4. **SSE2 backend**: Frequently slower than scalar for simple operations
5. **Small workloads** (<1000 elements): SIMD overhead often exceeds benefit

## Performance by Operation Category

### Excellent SIMD Performance (>5x speedup)

| Operation | Size | Scalar | SSE2 | AVX2 | SSE2 Speedup | AVX2 Speedup |
|-----------|------|--------|------|------|--------------|--------------|
| **tanh** | 100 | 891 ns | 137 ns | 101 ns | **6.5x** | **8.8x** |
| **tanh** | 1000 | 8.0 µs | 1.08 µs | - | **7.4x** | - |

**Why tanh excels:**
- Compute-intensive operation (requires exp calculations)
- SIMD processes 4-8 exponentials in parallel
- No memory bottleneck (compute dominates)
- AVX2's wider registers (8 vs 4 elements) provide 2x improvement over SSE2

### Good SIMD Performance (1.1-2x speedup)

| Operation | Size | Scalar | SSE2 | AVX2 | SSE2 Speedup | AVX2 Speedup |
|-----------|------|--------|------|------|--------------|--------------|
| **relu** | 100 | 54.1 ns | 44.8 ns | 49.3 ns | **1.21x** | **1.10x** |
| **scale** | 100 | 43.9 ns | 41.8 ns | 39.6 ns | 1.05x | **1.11x** |
| **scale** | 1000 | 104 ns | 111 ns | 90.8 ns | 0.94x | **1.15x** |
| **div** | 100 | 58.3 ns | 55.7 ns | 53.3 ns | 1.05x | 1.09x |

### Poor SIMD Performance (<1.1x or negative)

| Operation | Size | Scalar | SSE2 | AVX2 | SSE2 Speedup | AVX2 Speedup |
|-----------|------|--------|------|------|--------------|--------------|
| **sigmoid** | 100 | 364 ns | 405 ns | 393 ns | **0.90x** ❌ | **0.93x** ❌ |
| **fma** | 100 | 46.8 ns | 48.8 ns | 42.8 ns | **0.96x** ❌ | 1.09x |
| **sub** | 100 | 46.0 ns | 59.9 ns | 49.9 ns | **0.77x** ❌ | **0.92x** ❌ |
| **div** | 1000 | 142 ns | 218 ns | 142 ns | **0.65x** ❌ | 1.00x |

## Root Cause Analysis

### 1. Memory Bandwidth Bottleneck

For simple operations, memory access dominates compute time. SIMD can't help with RAM speed.

### 2. SIMD Overhead for Small Workloads

Fixed ~20-50ns overhead per operation from setup, alignment checks, and remainder handling.

### 3. Suboptimal Implementations

Some operations (div, sigmoid) show regressions requiring investigation.

## Next Steps

- Fix SSE2 div, sigmoid, fma, sub implementations
- Implement adaptive backend selection
- Benchmark against NumPy/PyTorch

## Related Chapters

- [Benchmarks Overview](./benchmarks.md)
- [Optimization Guide](./optimization-guide.md)
- [Profiling](./profiling.md)
