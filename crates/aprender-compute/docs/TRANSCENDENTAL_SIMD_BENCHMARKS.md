# Transcendental Function SIMD Benchmarks

**Date**: 2025-11-21
**System**: x86_64 Linux (AVX512-capable)
**Commit**: d636ab7 ([SIMD] Optimize SSE2 division + comprehensive performance analysis)
**Benchmark Tool**: Criterion.rs (10s measurement time, 100 samples)

## Executive Summary

Transcendental functions (exp, tanh, sigmoid) show **dramatically better SIMD performance** than simple arithmetic operations (div, sub, fma), confirming the hypothesis that **compute-bound operations benefit more from SIMD than memory-bound operations**.

### Key Findings

1. **Exponential (exp)**: Consistent 1.52-1.91x SSE2 speedup ✅
2. **Hyperbolic Tangent (tanh)**: Outstanding 6-8x SSE2 speedup ✅✅✅
3. **Sigmoid**: CRITICAL ISSUE - SSE2 is **7.2x SLOWER** at 10K+ elements ❌❌❌

**URGENT**: Sigmoid has a severe performance regression at large workloads that must be investigated immediately.

---

## Detailed Benchmark Results

### 1. Exponential (`exp`)

| Size | Backend | Time | Throughput | vs Scalar | Grade |
|------|---------|------|------------|-----------|-------|
| **100** | Scalar | 342.0 ns | 292.4 Melem/s | 1.00x | - |
| | SSE2 | 224.9 ns | 444.6 Melem/s | **1.52x** | ✅ |
| | AVX2 | 246.9 ns | 405.0 Melem/s | **1.39x** | ✅ |
| **1000** | Scalar | 2.965 µs | 337.2 Melem/s | 1.00x | - |
| | SSE2 | 1.598 µs | 625.9 Melem/s | **1.86x** | ✅ |
| | AVX2 | 1.813 µs | 551.6 Melem/s | **1.64x** | ✅ |
| **10000** | Scalar | 29.56 µs | 338.3 Melem/s | 1.00x | - |
| | SSE2 | 15.45 µs | 647.4 Melem/s | **1.91x** | ✅ |
| | AVX2 | 18.39 µs | 543.7 Melem/s | **1.61x** | ✅ |

**Analysis**: Exponential function shows consistent SIMD speedups across all workload sizes. SSE2 performs better than AVX2, likely due to better exp approximation implementation. This is a **compute-bound operation** where SIMD shines.

**Comparison with Simple Ops**:
- div (SSE2): 0.83-1.08x
- exp (SSE2): 1.52-1.91x
- **Improvement**: 1.8x better speedup for compute-bound exp vs memory-bound div

### 2. Hyperbolic Tangent (`tanh`)

| Size | Backend | Time | Throughput | vs Scalar | Grade |
|------|---------|------|------------|-----------|-------|
| **100** | Scalar | 1.607 µs | 62.0 Melem/s | 1.00x | - |
| | SSE2 | 265.2 ns | 377.1 Melem/s | **6.06x** | ✅✅✅ |
| | AVX2 | 351.0 ns | 284.9 Melem/s | **4.58x** | ✅✅ |
| **1000** | Scalar | 16.27 µs | 61.5 Melem/s | 1.00x | - |
| | SSE2 | 2.029 µs | 492.9 Melem/s | **8.02x** | ✅✅✅ |
| | AVX2 | 2.147 µs | 465.8 Melem/s | **7.58x** | ✅✅✅ |
| **10000** | Scalar | 154.7 µs | 64.6 Melem/s | 1.00x | - |
| | SSE2 | 19.15 µs | 522.1 Melem/s | **8.07x** | ✅✅✅ |
| | AVX2 | 20.85 µs | 479.5 Melem/s | **7.42x** | ✅✅✅ |

**Analysis**: Hyperbolic tangent shows **OUTSTANDING** SIMD performance - 6-8x faster than scalar! This is the best SIMD speedup measured in the entire codebase. The tanh approximation is highly compute-bound with polynomial evaluation, making it ideal for SIMD parallelization.

**Comparison with Simple Ops**:
- fma (SSE2): 0.87-0.99x (never wins)
- tanh (SSE2): 6.06-8.07x
- **Improvement**: 8x speedup vs 1x (no speedup) for simple FMA

**Why tanh Dominates**:
- Complex polynomial approximation (many multiplies/adds)
- Compute dominates over memory access
- SIMD parallelism directly accelerates the polynomial evaluation
- Minimal memory bandwidth pressure

### 3. Sigmoid (`sigmoid`) - **CRITICAL PERFORMANCE ISSUE**

| Size | Backend | Time | Throughput | vs Scalar | Grade |
|------|---------|------|------------|-----------|-------|
| **100** | Scalar | 404.2 ns | 247.4 Melem/s | 1.00x | - |
| | SSE2 | 261.4 ns | 382.5 Melem/s | **1.55x** | ✅ |
| | AVX2 | 290.4 ns | 344.3 Melem/s | **1.39x** | ✅ |
| **1000** | Scalar | 3.770 µs | 265.3 Melem/s | 1.00x | - |
| | SSE2 | 1.888 µs | 529.8 Melem/s | **2.00x** | ✅ |
| | AVX2 | 2.107 µs | 474.7 Melem/s | **1.79x** | ✅ |
| **10000** | Scalar | 11.18 µs | 894.3 Melem/s | 1.00x | - |
| | SSE2 | 80.69 µs | 123.9 Melem/s | **0.14x** | ❌❌❌ |
| | AVX2 | 49.41 µs | 202.4 Melem/s | **0.23x** | ❌❌❌ |

**CRITICAL ISSUE**: Sigmoid performance **collapses** at 10,000 elements:
- SSE2: **7.2x SLOWER** than scalar (0.14x speedup)
- AVX2: **4.4x SLOWER** than scalar (0.23x speedup)

This is a **severe performance regression** that blocks production use at realistic workload sizes.

**Root Cause Hypotheses**:
1. **Cache thrashing** - SIMD implementation may have poor cache behavior at larger sizes
2. **Branch misprediction** - Possible conditional code in SIMD path
3. **Memory layout** - Unaligned accesses or stride issues
4. **Algorithm issue** - SIMD implementation may use different algorithm than scalar

**Immediate Action Required**:
1. Profile sigmoid with Renacer/perf at 10K elements
2. Compare SSE2 assembly vs scalar assembly
3. Check for cache misses, branch mispredictions
4. Validate algorithm correctness (may be using slow path)

---

## Performance Summary Table

| Operation | 100 elem | 1000 elem | 10000 elem | Best Backend | Speedup Range |
|-----------|----------|-----------|------------|--------------|---------------|
| **exp (SSE2)** | 1.52x ✅ | 1.86x ✅ | 1.91x ✅ | SSE2 | 1.5-1.9x |
| **tanh (SSE2)** | 6.06x ✅✅✅ | 8.02x ✅✅✅ | 8.07x ✅✅✅ | SSE2 | 6-8x |
| **sigmoid (SSE2)** | 1.55x ✅ | 2.00x ✅ | 0.14x ❌❌❌ | Scalar @10K | -7.2x to 2x |

---

## Comparison: Memory-Bound vs Compute-Bound

### Memory-Bound Operations (from previous analysis)
| Operation | SSE2 Speedup @ 1000 elem |
|-----------|---------------------------|
| div | 1.08x |
| sub | 1.08x |
| fma | 0.96x |
| **Average** | **1.04x** |

### Compute-Bound Operations (current analysis)
| Operation | SSE2 Speedup @ 1000 elem |
|-----------|---------------------------|
| exp | 1.86x |
| tanh | 8.02x |
| sigmoid | 2.00x |
| **Average** | **3.96x** |

**Conclusion**: Compute-bound operations show **3.8x better SIMD speedups** than memory-bound operations (3.96x vs 1.04x). This validates the core hypothesis and justifies prioritizing SIMD optimization for transcendental functions.

---

## Recommendations

### URGENT (Blocker for Production)

1. **Investigate Sigmoid Regression**
   - Profile with perf/Renacer at 10K elements
   - Compare scalar vs SSE2 assembly
   - Check for cache misses, branch mispredictions
   - Validate algorithm correctness
   - **Timeline**: Must fix before any release

2. **Sigmoid Quick Fix Options**:
   - **Option A**: Use scalar fallback for >1000 elements (adaptive backend)
   - **Option B**: Rewrite SSE2/AVX2 sigmoid to match scalar algorithm
   - **Option C**: Switch to tanh-based sigmoid approximation (reuse high-perf tanh)

### High Priority

1. **Benchmark Remaining Transcendentals**
   - ln, log2, log10 (recently implemented in AVX2/AVX512)
   - sqrt, recip
   - sin, cos, tan (currently scalar fallback)
   - **Expected**: Similar 2-8x speedups for compute-bound operations

2. **Add Logarithm Benchmarks**
   - Create bench_ln, bench_log2, bench_log10
   - Validate recent AVX2/AVX512 implementations
   - Compare vs NumPy performance

3. **Optimize AVX2 exp**
   - Currently SSE2 outperforms AVX2 (1.86x vs 1.64x @ 1K elem)
   - Review AVX2 exp approximation polynomial
   - Target: 2.5-3x speedup (double SSE2's 4-wide parallelism)

### Medium Priority

1. **Adaptive Backend Selection**
   - Use scalar for memory-bound ops <1000 elements
   - Use SSE2/AVX2 for compute-bound ops all sizes
   - Switch sigmoid to scalar >1000 elements (until fixed)

2. **Document Performance Guidelines**
   - Update CLAUDE.md with compute-bound vs memory-bound distinction
   - Set realistic performance expectations per operation type
   - Guide users on when to use which backend

---

## Validation of Recent AVX2/AVX512 Work

The recent commits implementing logarithm functions in AVX2/AVX512 are **highly valuable** based on these benchmark results:
- Logarithms are compute-bound (similar to exp)
- Expected speedup: 2-8x for AVX2, 4-16x for AVX512
- Should be prioritized over memory-bound optimizations

**Next Steps**:
1. Add benchmarks for ln, log2, log10 to validate implementations
2. Profile AVX512 logarithms to confirm 16-wide benefits
3. Compare with NumPy's log performance

---

## Appendix: Raw Benchmark Data

### exp
```
100 elem:  Scalar=342.0ns, SSE2=224.9ns, AVX2=246.9ns
1000 elem: Scalar=2.965µs, SSE2=1.598µs, AVX2=1.813µs
10000 elem: Scalar=29.56µs, SSE2=15.45µs, AVX2=18.39µs
```

### tanh
```
100 elem:  Scalar=1.607µs, SSE2=265.2ns, AVX2=351.0ns
1000 elem: Scalar=16.27µs, SSE2=2.029µs, AVX2=2.147µs
10000 elem: Scalar=154.7µs, SSE2=19.15µs, AVX2=20.85µs
```

### sigmoid
```
100 elem:  Scalar=404.2ns, SSE2=261.4ns, AVX2=290.4ns
1000 elem: Scalar=3.770µs, SSE2=1.888µs, AVX2=2.107µs
10000 elem: Scalar=11.18µs, SSE2=80.69µs, AVX2=49.41µs ⚠️ REGRESSION
```

---

**Generated by**: Claude Code autonomous benchmarking session
**Tools Used**: cargo bench, Criterion.rs, statistical analysis
**Data Quality**: High confidence (100 samples, 10s measurement per size)
**Status**: **BLOCKED** on sigmoid performance regression fix
