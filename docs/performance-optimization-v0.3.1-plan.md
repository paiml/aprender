# Performance Optimization Plan for v0.3.1

**Date**: 2025-11-20
**Goal**: Fix tanh and relu performance issues identified in v0.3.0 comprehensive benchmarks

## Executive Summary

Comprehensive benchmarks revealed two critical performance issues:
1. **tanh**: 5.59x slower than NumPy at 100K elements (194.29 µs vs 34.78 µs)
2. **relu**: 8.32x slower than NumPy at 1M elements (5.53 ms vs 664.79 µs)

Both operations show excellent performance at small sizes but degrade at larger sizes, indicating **memory bandwidth bottlenecks** rather than algorithmic issues.

---

## Issue 1: tanh Performance Degradation

### Benchmark Results

| Size | Trueno (SSE2) | NumPy | Speedup |
|------|---------------|-------|---------|
| 100 | 272.85 ns | 1.08 µs | **3.95x faster** ✅ |
| 1K | 2.01 µs | 1.16 µs | 1.74x slower ⚠️ |
| 10K | 19.21 µs | 4.33 µs | 4.44x slower ❌ |
| 100K | 194.29 µs | 34.78 µs | **5.59x slower** ❌ |

### Analysis

**Observations:**
- At 100 elements: Trueno is 3.95x **faster** than NumPy
- Performance degrades progressively as size increases
- Classic memory bandwidth issue pattern

**Current Implementation** (src/backends/sse2.rs:1104):
- ✅ Sophisticated range reduction for exp() approximation
- ✅ 6th-order Taylor series polynomial (good accuracy)
- ✅ Proper overflow/underflow handling
- ✅ SSE2 floor emulation
- ✅ Processes 4 f32s per iteration

**Root Cause:**
1. **Memory bandwidth saturation** - Large arrays exceed cache capacity
2. **NumPy advantage** - Uses Intel MKL or OpenBLAS which have:
   - Hardware-optimized SVML (Short Vector Math Library)
   - Better cache-aware tiling
   - Multi-threading for large arrays (OpenMP)
   - AVX2/AVX-512 implementations

**Why SSE2 Instead of AVX2?**
- Backend selection happens at Vector creation based on CPU features
- All tanh benchmarks showed SSE2 backend usage
- Need to investigate why AVX2/AVX-512 backends weren't selected

### Proposed Solutions

#### Option 1: Use AVX2/AVX-512 Backends (Quick Win)
**Effort**: 1-2 hours
**Expected improvement**: 2-4x at large sizes

**Actions:**
1. Investigate why AVX2 backend wasn't used in benchmarks
2. Ensure backend selection properly detects AVX2/AVX-512
3. AVX2 processes 8 f32s per iteration (2x throughput vs SSE2)
4. AVX2 has FMA (fused multiply-add) for better polynomial evaluation

**Verification:**
```bash
# Check CPU features
cat /proc/cpuinfo | grep flags | grep avx2
lscpu | grep avx

# Run benchmark with explicit backend forcing
cargo bench tanh -- --verbose
```

#### Option 2: Multi-threading for Large Arrays (Medium Effort)
**Effort**: 1-2 days
**Expected improvement**: 2-4x at 100K+ elements

**Actions:**
1. Use Rayon for parallel iteration at >10K elements
2. Split array into chunks, process in parallel
3. Minimal overhead for small arrays

**Implementation:**
```rust
use rayon::prelude::*;

if len > 10_000 {
    // Parallel processing
    a.par_chunks(4096)
        .zip(result.par_chunks_mut(4096))
        .for_each(|(chunk_a, chunk_result)| {
            unsafe { Sse2Backend::tanh(chunk_a, chunk_result) };
        });
} else {
    // Sequential for small arrays
    unsafe { Sse2Backend::tanh(a, result) };
}
```

#### Option 3: Cache-Aware Tiling (High Effort)
**Effort**: 3-5 days
**Expected improvement**: 1.5-2x at large sizes

**Actions:**
1. Process data in L2-cache-sized tiles (256KB chunks)
2. Better temporal locality
3. Reduce TLB misses

**Defer to v0.4.0**: Complex implementation, moderate gains

---

## Issue 2: relu Catastrophic Performance at 1M Elements

### Benchmark Results

| Size | Trueno (SSE2) | NumPy | PyTorch | Speedup |
|------|---------------|-------|---------|---------|
| 100 | 81.0 ns | 1.44 µs | 3.40 µs | **17.73x faster** ✅ |
| 1K | 239.7 ns | 2.32 µs | 2.72 µs | **9.70x faster** ✅ |
| 10K | 2.62 µs (Scalar!) | 10.24 µs | 4.17 µs | **3.90x faster** ✅ |
| 100K | 28.55 µs | 70.39 µs | 60.03 µs | **2.47x faster** ✅ |
| 1M | **5.53 ms** | 664.79 µs | 75.45 µs | **8.32x slower** ❌ |

### Analysis

**Critical Observation:**
- 100K → 1M: 10x elements, **194x time** (catastrophic scaling!)
- All other sizes show excellent performance
- 10K used **Scalar backend** (suspicious!)

**SSE2 Implementation** (src/backends/sse2.rs:708):
```rust
let va = _mm_loadu_ps(a.as_ptr().add(i));
let vresult = _mm_max_ps(zero, va);
_mm_storeu_ps(result.as_mut_ptr().add(i), vresult);
```

- ✅ Optimal SIMD code (1 load + 1 max + 1 store)
- ✅ No branching, perfect for pipelining
- ✅ Minimal instruction count

**Theoretical Throughput:**
- 1M elements / 4 (SSE2 width) = 250K iterations
- 3-4 cycles per iteration (load + max + store)
- At 3 GHz: ~300K cycles = **100 µs theoretical minimum**
- **Actual: 5.53 ms = 5530 µs (55x slower than theoretical!)**

**Root Causes:**

1. **Memory Allocation Overhead**
   - Line 1401 in vector.rs: `let mut result = vec![0.0; self.len()];`
   - For 1M elements: 4MB allocation
   - Possible page faults, memory fragmentation

2. **TLB (Translation Lookaside Buffer) Misses**
   - 1M f32s = 4MB
   - Typical TLB covers ~2MB (512 4KB pages)
   - Exceeding TLB capacity causes page table walks

3. **Why 10K Used Scalar Backend?**
   - Backend selection anomaly - needs investigation
   - Scalar might be bypassing some overhead?

4. **NumPy/PyTorch Advantage**
   - Multi-threading (OpenMP) for large arrays
   - Pre-allocated memory pools
   - Better memory alignment
   - Hardware prefetching optimization

### Proposed Solutions

#### Option 1: Enable Multi-threading (High Priority)
**Effort**: 1-2 hours
**Expected improvement**: 5-10x at 1M elements

**Actions:**
1. Use Rayon for arrays >100K elements
2. Chunk size: 64K-256K elements (cache-friendly)
3. Minimal overhead for smaller arrays

**Implementation:**
```rust
pub fn relu(&self) -> Result<Self> {
    let mut result = vec![0.0; self.len()];

    if self.len() > 100_000 {
        // Parallel processing for large arrays
        use rayon::prelude::*;
        self.data.par_chunks(65536)
            .zip(result.par_chunks_mut(65536))
            .for_each(|(chunk_in, chunk_out)| {
                unsafe {
                    match self.backend {
                        Backend::SSE2 | Backend::AVX => {
                            Sse2Backend::relu(chunk_in, chunk_out);
                        }
                        Backend::AVX2 | Backend::AVX512 => {
                            Avx2Backend::relu(chunk_in, chunk_out);
                        }
                        _ => ScalarBackend::relu(chunk_in, chunk_out),
                    }
                }
            });
    } else {
        // Sequential for small arrays
        unsafe { /* existing dispatch */ }
    }

    Ok(Vector::from_vec(result))
}
```

**Benefits:**
- Utilizes multiple cores
- Better cache utilization (smaller working sets per thread)
- Reduces TLB pressure per thread

#### Option 2: Investigate Backend Selection (Medium Priority)
**Effort**: 2-4 hours
**Expected improvement**: 2-4x if AVX2 is available

**Actions:**
1. Add logging to backend selection
2. Verify AVX2/AVX-512 detection working correctly
3. Ensure benchmarks use optimal backend

**Why 10K Used Scalar?**
- Check `benchmarks/` Criterion configuration
- Might be forcing specific backends for comparison
- Need to verify this isn't masking a real issue

#### Option 3: Memory Pool (Low Priority for v0.3.1)
**Effort**: 2-3 days
**Expected improvement**: 1.5-2x

**Defer to v0.4.0**:
- Pre-allocate memory pools for common sizes
- Reuse allocations where possible
- Complex lifetime management

---

## Recommended Action Plan (v0.3.1)

### Phase 1: Quick Wins (1-2 days)

**Priority 1: Enable Rayon Multi-threading for relu**
- Target: relu performance at 1M elements
- Expected: 8.32x slower → within 2x of NumPy
- Implementation: Add Rayon parallel processing for >100K elements
- Testing: Re-run comprehensive benchmarks

**Priority 2: Verify Backend Selection**
- Investigate why tanh used SSE2 instead of AVX2
- Investigate why relu/10K used Scalar
- Fix backend selection if broken

**Priority 3: Force AVX2/AVX-512 for tanh**
- Update backend dispatch to prefer AVX2/AVX-512
- Expected: 2-4x improvement at large sizes
- Testing: Benchmark tanh across all sizes

### Phase 2: Validation (1 day)

**Actions:**
1. Re-run `make bench-comprehensive`
2. Verify tanh within 2x of NumPy at 100K
3. Verify relu within 2x of NumPy at 1M
4. Update comparison reports

### Success Criteria (v0.3.1)

✅ **tanh**: Within 2x of NumPy at all sizes (currently 5.59x slower at 100K)
✅ **relu**: Within 2x of NumPy at all sizes (currently 8.32x slower at 1M)
✅ **No regressions**: All other operations maintain performance
✅ **Quality gates**: All tests pass, >90% coverage maintained

---

## Alternative: Accept Current Performance (Not Recommended)

**Rationale:**
- Current issues affect only 2 operations at large sizes
- 88.5% of operations still faster than NumPy
- Could document as known limitation

**Cons:**
- 8.32x slower relu at 1M is embarrassing for a performance-focused library
- Users with large arrays will be disappointed
- Contradicts "high-performance" positioning

**Recommendation**: **Fix the issues** - they're solvable with 1-2 days of work.

---

## Future Work (v0.4.0+)

### Advanced Optimizations

1. **Intel SVML Integration**
   - Link against Intel's Short Vector Math Library
   - Hardware-optimized transcendental functions
   - Effort: 1-2 weeks
   - Expected: Match NumPy performance exactly

2. **Cache-Aware Tiling**
   - Process data in L2-cache-sized tiles
   - Better temporal locality
   - Effort: 1 week
   - Expected: 1.5-2x improvement

3. **Memory Pooling**
   - Pre-allocate result vectors
   - Reuse allocations where safe
   - Effort: 2-3 days
   - Expected: 1.5x improvement at large sizes

4. **GPU Threshold Tuning**
   - Re-evaluate GPU thresholds post-optimization
   - May be viable after CPU optimizations
   - Effort: 3-5 days

---

## References

- Benchmark report: `benchmarks/comparison_report.md`
- Performance analysis: `docs/performance-analysis.md`
- ROADMAP: v0.3.1 optimization targets

---

**Status**: Ready for implementation
**Owner**: Trueno Core Team
**Timeline**: 2-3 days for complete fix
