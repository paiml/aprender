# Square Root and Reciprocal SIMD Benchmarks - CRITICAL PERFORMANCE REGRESSION

**Date**: 2025-11-21
**System**: x86_64 Linux (AVX512-capable)
**Benchmark Tool**: Criterion.rs (10s measurement time, 100 samples)
**Status**: ❌ **BLOCKING ISSUE** - SIMD implementations significantly slower than scalar

## Executive Summary

**CRITICAL FINDING**: Both sqrt and recip show **severe SIMD performance regressions** across all backends. Unlike logarithms (which achieved 6-14x speedups), these operations show **NEGATIVE speedups** with SIMD often being **2-6x SLOWER** than scalar.

### Key Findings

1. **sqrt SSE2**: 0.93-0.99x (no speedup, essentially matches scalar)
2. **sqrt AVX2**: **0.58-0.70x** (1.4-1.7x SLOWER than scalar) ❌❌
3. **sqrt AVX512**: **0.74-0.89x** (1.1-1.4x SLOWER than scalar) ❌❌
4. **recip SSE2**: 0.95-1.01x (essentially matches scalar)
5. **recip AVX2**: **0.17-0.33x** (3-6x SLOWER than scalar) ❌❌❌
6. **recip AVX512**: **0.26-0.42x** (2.4-3.8x SLOWER than scalar) ❌❌❌

**THIS IS A BLOCKING ISSUE** - These implementations must be investigated and fixed before any release.

---

## Detailed Benchmark Results

### 1. Square Root (`sqrt`)

| Size | Backend | Time | Throughput | vs Scalar | Grade |
|------|---------|------|------------|-----------|-------|
| **100** | Scalar | 75.11 ns | 1.33 Gelem/s | 1.00x | - |
| | SSE2 | 80.39 ns | 1.24 Gelem/s | **0.93x** | ❌ |
| | AVX2 | 106.66 ns | 937 Melem/s | **0.70x** | ❌❌ |
| | AVX512 | 101.42 ns | 986 Melem/s | **0.74x** | ❌❌ |
| **1000** | Scalar | 318.97 ns | 3.14 Gelem/s | 1.00x | - |
| | SSE2 | 331.98 ns | 3.01 Gelem/s | **0.96x** | ❌ |
| | AVX2 | 519.38 ns | 1.93 Gelem/s | **0.61x** | ❌❌ |
| | AVX512 | 356.50 ns | 2.81 Gelem/s | **0.89x** | ❌❌ |
| **10000** | Scalar | 2.72 µs | 3.67 Gelem/s | 1.00x | - |
| | SSE2 | 2.75 µs | 3.64 Gelem/s | **0.99x** | ≈ |
| | AVX2 | 4.73 µs | 2.11 Gelem/s | **0.58x** | ❌❌❌ |
| | AVX512 | 3.43 µs | 2.92 Gelem/s | **0.79x** | ❌❌ |

**Analysis**: sqrt shows NO SIMD advantage. SSE2 essentially matches scalar performance, while AVX2 and AVX512 are significantly slower. This is unexpected because sqrt has direct SIMD instructions (`_mm_sqrt_ps`, `_mm256_sqrt_ps`, `_mm512_sqrt_ps`).

**Expected vs Actual**:
- **Expected**: 1.5-2x speedup (sqrt has direct hardware support)
- **Actual**: 0.58-0.99x (SIMD is slower!)
- **Gap**: -40% to -1% performance

### 2. Reciprocal (`recip`, 1/x)

| Size | Backend | Time | Throughput | vs Scalar | Grade |
|------|---------|------|------------|-----------|-------|
| **100** | Scalar | 71.99 ns | 1.39 Gelem/s | 1.00x | - |
| | SSE2 | 75.38 ns | 1.33 Gelem/s | **0.95x** | ❌ |
| | AVX2 | 220.26 ns | 454 Melem/s | **0.33x** | ❌❌❌ |
| | AVX512 | 170.81 ns | 585 Melem/s | **0.42x** | ❌❌❌ |
| **1000** | Scalar | 317.71 ns | 3.15 Gelem/s | 1.00x | - |
| | SSE2 | 313.86 ns | 3.19 Gelem/s | **1.01x** | ≈ |
| | AVX2 | 1.65 µs | 607 Melem/s | **0.19x** | ❌❌❌ |
| | AVX512 | 1.07 µs | 937 Melem/s | **0.30x** | ❌❌❌ |
| **10000** | Scalar | 2.73 µs | 3.66 Gelem/s | 1.00x | - |
| | SSE2 | 2.73 µs | 3.67 Gelem/s | **1.00x** | ≈ |
| | AVX2 | 16.01 µs | 625 Melem/s | **0.17x** | ❌❌❌ |
| | AVX512 | 10.30 µs | 971 Melem/s | **0.26x** | ❌❌❌ |

**Analysis**: recip shows CATASTROPHIC SIMD regression! AVX2 is up to **6x SLOWER** than scalar, AVX512 is **3-4x SLOWER**. Even SSE2 only matches scalar. This is the worst SIMD performance measured in the entire codebase.

**Expected vs Actual**:
- **Expected**: 1.5-2x speedup (recip uses division, similar to div benchmarks)
- **Actual**: 0.17-1.01x (AVX2 is 6x slower!)
- **Gap**: -83% to +1% performance

---

## Root Cause Analysis

### Why Is SIMD So Slow?

**Hypothesis 1: Implementation Bug**
- SIMD implementations may have incorrect loop structure
- Possible excessive branching or conditional code
- Memory alignment issues causing unaligned loads/stores

**Hypothesis 2: Algorithm Mismatch**
- SIMD code may use different algorithm than scalar
- Scalar may have fast-path optimizations that SIMD lacks
- Unnecessary intermediate operations in SIMD path

**Hypothesis 3: Data Layout Issues**
- Strided memory access patterns
- Cache thrashing due to poor temporal locality
- Unaligned memory causing split loads/stores

**Hypothesis 4: Instruction Latency**
- sqrt/div instructions may have high latency that SIMD doesn't hide
- Insufficient instruction-level parallelism
- Pipeline stalls due to data dependencies

### Comparison with Other Operations

| Operation | Type | Best SIMD Speedup | Backend | Status |
|-----------|------|-------------------|---------|--------|
| **log10** | Logarithm | **13.98x** ✅✅✅ | AVX512 | Production |
| **tanh** | Hyperbolic | **8.07x** ✅✅✅ | SSE2 | Production |
| **ln** | Logarithm | **6.63x** ✅✅✅ | AVX512 | Production |
| **exp** | Exponential | **1.91x** ✅ | SSE2 | Production |
| **div** | Arithmetic | **1.16x** ✅ | AVX2 | Marginal |
| **sqrt** | Elementary | **0.99x** ❌ | SSE2 | BROKEN |
| **recip** | Reciprocal | **1.01x** ❌ | SSE2 | BROKEN |

**Key Insight**: sqrt and recip are the ONLY operations showing no SIMD benefit. Even simple div shows 1.16x speedup, while sqrt/recip are slower or at best equal.

---

## Performance Degradation Analysis

### sqrt Performance Degradation

| Size | Scalar Time | AVX2 Time | Slowdown | Wasted Time |
|------|-------------|-----------|----------|-------------|
| **100** | 75.11 ns | 106.66 ns | **1.42x** | +31.55 ns |
| **1000** | 318.97 ns | 519.38 ns | **1.63x** | +200.41 ns |
| **10000** | 2.72 µs | 4.73 µs | **1.74x** | +2.01 µs |

**Trend**: Performance degradation WORSENS with larger workloads (1.42x → 1.74x). This suggests memory bandwidth or cache issues.

### recip Performance Degradation

| Size | Scalar Time | AVX2 Time | Slowdown | Wasted Time |
|------|-------------|-----------|----------|-------------|
| **100** | 71.99 ns | 220.26 ns | **3.06x** | +148.27 ns |
| **1000** | 317.71 ns | 1.65 µs | **5.19x** | +1.33 µs |
| **10000** | 2.73 µs | 16.01 µs | **5.87x** | +13.28 µs |

**Trend**: Performance degradation WORSENS dramatically with larger workloads (3.06x → 5.87x). This indicates a severe algorithmic or implementation issue.

---

## Recommendations

### URGENT (Blocker for Production)

1. **Investigate SIMD Implementations**
   - Review `src/backends/sse2.rs` sqrt/recip implementations
   - Review `src/backends/avx2.rs` sqrt/recip implementations
   - Review `src/backends/avx512.rs` sqrt/recip implementations
   - Check for loop structure issues, branching, alignment

2. **Profile with Renacer**
   ```bash
   # Profile sqrt SSE2 vs scalar
   renacer --function-time --source -- cargo bench sqrt/Scalar/10000
   renacer --function-time --source -- cargo bench sqrt/SSE2/10000

   # Profile recip AVX2 vs scalar
   renacer --function-time --source -- cargo bench recip/Scalar/10000
   renacer --function-time --source -- cargo bench recip/AVX2/10000
   ```

3. **Compare Assembly Output**
   ```bash
   # Generate assembly for scalar vs SIMD
   cargo rustc --release -- --emit asm
   # Compare implementations side-by-side
   ```

4. **Quick Fix Options**:
   - **Option A**: Use scalar fallback for sqrt/recip (disable SIMD)
   - **Option B**: Rewrite SIMD implementations to match scalar algorithm
   - **Option C**: Use reciprocal approximation + Newton-Raphson for recip
   - **Option D**: Simplify loop structure, remove branching

### High Priority

1. **Add Correctness Tests**
   - Validate SIMD sqrt/recip produce correct results
   - Check for edge cases: 0, negative, NaN, infinity
   - Ensure backend equivalence (scalar == SSE2 == AVX2)

2. **Benchmark sqrt Approximation**
   - Test fast reciprocal square root approximation
   - Newton-Raphson refinement may be faster than direct sqrt

3. **Document Performance Characteristics**
   - Update CLAUDE.md with sqrt/recip performance warnings
   - Recommend users avoid SIMD backends for these operations
   - Add adaptive backend selection to skip SIMD

---

## Validation Against Previous Work

### Comparison with Division (docs/AVX512_SIMD_PERFORMANCE_ANALYSIS.md)

**Division Performance** (SSE2):
- 100 elem: 0.83x (slower)
- 1000 elem: 1.08x (slight speedup)
- 10000 elem: 1.07x (slight speedup)

**sqrt/recip Performance** (SSE2):
- sqrt: 0.93-0.99x (no speedup)
- recip: 0.95-1.01x (no speedup)

**Conclusion**: sqrt/recip perform similarly to division for SSE2 (no significant speedup). However, AVX2/AVX512 show severe regressions that division doesn't exhibit.

### Comparison with Logarithms (docs/LOGARITHM_SIMD_BENCHMARKS.md)

**Why Do Logarithms Succeed While sqrt/recip Fail?**

1. **Compute Density**:
   - Logarithms: Complex polynomial evaluation, many FMA operations
   - sqrt/recip: Single instruction operations, low compute density

2. **Memory Access Patterns**:
   - Logarithms: Intermediate values stay in registers
   - sqrt/recip: May have excessive memory traffic

3. **Implementation Complexity**:
   - Logarithms: Carefully optimized polynomial coefficients
   - sqrt/recip: Possibly naive direct instruction usage

4. **Instruction-Level Parallelism**:
   - Logarithms: Many independent operations (high ILP)
   - sqrt/recip: Single-threaded pipeline (low ILP)

---

## Lessons Learned

### Direct SIMD Instructions ≠ Automatic Speedup

**Key Takeaway**: Just because an operation has a direct SIMD instruction (like `_mm_sqrt_ps`) doesn't guarantee speedup. SIMD overhead (loop setup, loads, stores, remainder handling) can dominate for simple operations.

**Implications**:
1. Benchmark EVERY SIMD implementation - don't assume correctness implies performance
2. Simple operations may benefit more from scalar code
3. Compute density matters more than instruction availability

### When SIMD Fails

**Operations Where SIMD Struggles**:
- Single-instruction operations (sqrt, recip, abs, neg)
- Memory-bound operations (copy, simple arithmetic)
- Low compute-to-memory ratio

**Operations Where SIMD Excels**:
- Complex polynomial evaluation (logarithms, transcendentals)
- Reductions (dot product, sum)
- High compute-to-memory ratio

---

## Appendix: Raw Benchmark Data

### sqrt (square root)
```
100 elem:  Scalar=75.11ns, SSE2=80.39ns, AVX2=106.66ns, AVX512=101.42ns
1000 elem: Scalar=318.97ns, SSE2=331.98ns, AVX2=519.38ns, AVX512=356.50ns
10000 elem: Scalar=2.72µs, SSE2=2.75µs, AVX2=4.73µs, AVX512=3.43µs
```

### recip (reciprocal, 1/x)
```
100 elem:  Scalar=71.99ns, SSE2=75.38ns, AVX2=220.26ns, AVX512=170.81ns
1000 elem: Scalar=317.71ns, SSE2=313.86ns, AVX2=1.65µs, AVX512=1.07µs
10000 elem: Scalar=2.73µs, SSE2=2.73µs, AVX2=16.01µs, AVX512=10.30µs
```

---

## Conclusions

1. **sqrt SIMD is SLOWER than scalar** - AVX2/AVX512 show 1.4-1.7x regression
2. **recip SIMD is CATASTROPHICALLY SLOWER** - AVX2 shows 3-6x regression
3. **SSE2 provides no benefit** - essentially matches scalar for both operations
4. **This is a BLOCKING ISSUE** - must fix before any release
5. **Root cause likely implementation bug** - needs urgent investigation

**Strategic Impact**: This demonstrates that not all operations benefit from SIMD. The library needs adaptive backend selection to avoid SIMD for operations where it hurts performance.

---

**Generated by**: Claude Code autonomous benchmarking session
**Tools Used**: cargo bench, Criterion.rs, statistical analysis
**Data Quality**: High confidence (100 samples, 10s measurement per size)
**Status**: **BLOCKED** - sqrt/recip SIMD implementations must be fixed ❌❌❌
