# SIMD Performance Analysis

**Date**: 2025-11-18
**System**: x86_64 Linux (AVX2-capable)
**Rust Version**: nightly
**Benchmark Tool**: Criterion.rs

## Executive Summary

Comprehensive benchmarking of Trueno SIMD backends reveals **mixed results** across operations. While some operations show excellent SIMD speedups (tanh: 6.5-8.8x), many element-wise operations show **minimal** or **negative** speedups, especially for SSE2.

### Key Findings

1. **Activation functions** (relu, tanh): Good SIMD speedups (1.2-8.8x)
2. **Element-wise operations** (add, sub, div, fma): Minimal or negative SIMD benefit
3. **SSE2 backend**: Frequently slower than scalar baseline
4. **Small workloads (<1000 elements)**: SIMD overhead often exceeds benefit

## Detailed Results

### Excellent SIMD Performance (>5x speedup)

| Operation | Size | Scalar | SSE2 | AVX2 | SSE2 Speedup | AVX2 Speedup |
|-----------|------|--------|------|------|--------------|--------------|
| **tanh** | 100 | 891 ns | 137 ns | 101 ns | **6.5x** | **8.8x** |
| **tanh** | 1000 | 8.0 µs | 1.08 µs | - | **7.4x** | - |

###Good SIMD Performance (1.1-2x speedup)

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
| **sigmoid** | 1000 | 3.11 µs | 2.71 µs | 2.65 µs | 1.15x | 1.17x |
| **sigmoid** | 10000 | 9.28 µs | 12.0 µs | 9.43 µs | **0.77x** ❌ | 0.98x |
| **sigmoid** | 100000 | 78.3 µs | 103 µs | 83.5 µs | **0.76x** ❌ | 0.94x |
| **fma** | 100 | 46.8 ns | 48.8 ns | 42.8 ns | **0.96x** ❌ | 1.09x |
| **fma** | 1000 | 134 ns | 145 ns | 143 ns | **0.92x** ❌ | **0.94x** ❌ |
| **fma** | 10000 | 1.47 µs | 1.59 µs | 1.43 µs | **0.92x** ❌ | 1.03x |
| **fma** | 100000 | 16.2 µs | 17.5 µs | 16.2 µs | **0.93x** ❌ | 1.00x |
| **sub** | 100 | 46.0 ns | 59.9 ns | 49.9 ns | **0.77x** ❌ | **0.92x** ❌ |
| **sub** | 1000 | 111 ns | 122 ns | 113 ns | **0.91x** ❌ | **0.98x** ❌ |
| **div** | 1000 | 142 ns | 218 ns | 142 ns | **0.65x** ❌ | 1.00x |
| **div** | 10000 | 1.11 µs | 1.87 µs | - | **0.59x** ❌ | - |

## Root Cause Analysis

### Why SIMD is Underperforming

1. **Memory Bandwidth Bottleneck**
   - For simple operations (add, sub), memory access dominates compute time
   - SIMD doesn't help when memory-bound
   - Confirmed by similar performance across backends for 100K element operations

2. **SIMD Overhead for Small Workloads**
   - Loop setup, data alignment, remainder handling
   - Fixed ~20-50ns overhead per operation
   - Only profitable for workloads >1000 elements (operation-dependent)

3. **Suboptimal Implementations**
   - SSE2 div: 0.59-0.65x speedup suggests implementation issue
   - sigmoid: Exp approximation may be expensive for SIMD
   - Potential issues: excessive branching, unaligned loads, poor instruction scheduling

4. **Cache Effects**
   - Small workloads (100-1000 elements) fit in L1 cache: scalar is efficient
   - Large workloads (100K elements): both scalar and SIMD are memory-bound

## Recommendations

### Immediate Optimizations (STOP THE LINE)

1. **Fix SSE2 div implementation** (0.59x speedup is unacceptable)
   - Review for branching, alignment issues
   - Compare against hand-optimized div implementations
   - Target: ≥1.5x speedup over scalar

2. **Optimize sigmoid/gelu/swish** (currently 0.76-0.93x for SSE2)
   - Current exp approximation may be too expensive
   - Consider lookup tables or faster polynomial approximations
   - Target: ≥1.5x speedup for 10K+ elements

3. **Fix fma and sub implementations**
   - SSE2 consistently slower (0.77-0.96x)
   - Likely alignment or remainder handling issues
   - Target: ≥1.5x speedup for 10K+ elements

### Longer-Term Optimizations (Kaizen)

1. **Adaptive Backend Selection**
   - Currently selects backend at Vector creation
   - Should consider operation type and workload size
   - Example: Use scalar for sigmoid <1000 elements, SIMD for ≥1000

2. **Implement AVX-512**
   - Potential 16-wide operations (2x over AVX2)
   - Most beneficial for tanh, relu (compute-bound operations)

3. **Benchmark Against NumPy/PyTorch**
   - Validate performance against industry standards
   - Target: Within 20% of NumPy for 1D ops (per ROADMAP)

## Testing Methodology

All benchmarks run with:
- Criterion.rs default settings (100 samples, 3s warmup)
- Release build with `--release`
- CPU affinity pinned to single core
- Background processes minimized

Sizes tested: 100, 1000, 10000, 100000 elements

## Next Steps

1. **STOP THE LINE**: Fix SSE2 div, sigmoid, fma, sub implementations
2. Run mutation testing to verify SIMD correctness
3. Add adaptive backend selection based on operation type and size
4. Benchmark against NumPy/PyTorch for ROADMAP v0.3.0 deliverable
5. Document performance regression tests to prevent future slowdowns

---

**Status**: WASM SIMD128 complete, CPU SIMD optimizations in progress
**TDG Score**: 87.7/100 (A-)
**ROADMAP**: v0.3.0 Phase 1 - completing comprehensive benchmarks
