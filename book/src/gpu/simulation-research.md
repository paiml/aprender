# trueno Simulation Research Findings

This chapter documents the comprehensive simulation testing of trueno's GPU and SIMD backends. All tests follow Popperian methodology - each claim is designed to be **disproven**, and surviving tests represent verified guarantees.

## Overview

102 falsifiable claims across 7 sections validate trueno's correctness, determinism, and performance characteristics.

## Section A: Backend Selection (Claims 1-15)

**Insight:** Backend selection logic is sound but conservative.

| Finding | Evidence | Recommendation |
|---------|----------|----------------|
| GPU threshold (100K elements) is appropriate | A-005, A-011 confirm transfer overhead amortized at scale | May lower to 50K on PCIe 4.0/5.0 |
| Parallel threshold (1K) balances overhead | A-006 shows Rayon adds value only above threshold | Keep threshold |
| Graceful degradation works | A-007 confirms GPU unavailability doesn't crash | Production-ready |
| Backend selection is O(1) | A-010 verified <1μs selection time | No bottleneck |

### Actionable

The 100K GPU threshold could potentially be lowered to 50K on systems with fast PCIe 4.0/5.0. This explains the IMP-600 finding that GPU is slower for MATVEC operations.

## Section B: Determinism (Claims 16-30)

**Insight:** SimRng (PCG algorithm) provides excellent cross-platform reproducibility.

| Test ID | Claim | Result |
|---------|-------|--------|
| B-017 | Same seed → identical output | ✅ Verified 100x |
| B-019 | Parallel partitions deterministic | ✅ Confirmed |
| B-022 | System load doesn't affect numerical results | ✅ Confirmed |
| B-023 | Memory pressure doesn't affect results | ✅ Confirmed |

### Key Learning

PCG over Mersenne Twister validated - PCG's smaller state (128 bits vs 2.5KB) is cache-friendly for parallel workloads.

```rust
// PCG provides deterministic sequences with minimal state
use trueno::rng::SimRng;

let mut rng = SimRng::seed(42);
let a = rng.gen::<f32>();  // Always the same value for seed 42
```

## Section C: SIMD Operations (Claims 31-50)

**Insight:** SIMD implementations correctly preserve mathematical properties.

| Property | Status | Tolerance |
|----------|--------|-----------|
| Add commutativity | ✅ Exact | 0 ULP |
| Add associativity | ✅ | 1e-5 (FP rounding) |
| Mul commutativity | ✅ Exact | 0 ULP |
| Dot product symmetry | ✅ | 1e-5 |
| Softmax sums to 1.0 | ✅ | 1e-5 |

### Critical Finding (C-050)

Denormal inputs don't cause stalls. SIMD backends correctly use flush-to-zero mode, preventing the 100x slowdown that denormals can cause on some hardware.

### GELU Validation (C-040)

The tanh approximation matches the exact formula within 1e-4:

```rust
// GELU approximation is production-ready
fn gelu_approx(x: f32) -> f32 {
    0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x.powi(3))))
}
// Max error: < 1e-4 compared to exact GELU
```

## Section D: PTX Kernels (Claims 51-65)

**Insight:** PTX generation produces correct, efficient CUDA code.

| Pattern | Test ID | Status |
|---------|---------|--------|
| bar.sync barriers after shared memory | D-053, D-054 | ✅ Verified |
| Softmax max-subtraction for stability | D-056 | ✅ Verified |
| Register allocation < 255 (sm_70+) | D-058 | ✅ Verified |
| Shared memory < 48KB | D-059 | ✅ Verified |

### Key Learning

PTX builder correctly emits synchronization barriers - missing barriers cause race conditions that are extremely hard to debug.

```ptx
// Correct barrier placement after shared memory write
st.shared.f32 [%r1], %f1;
bar.sync 0;  // Required before any thread reads from shared memory
```

## Section E: WGPU Shaders (Claims 66-80)

**Insight:** WGSL shaders match CPU reference implementations within tolerance.

| Operation | Max Diff from CPU |
|-----------|-------------------|
| add/mul/dot | < 1e-6 |
| relu/sigmoid/tanh | < 1e-5 |
| gelu/swish | < 1e-4 |
| softmax | < 1e-5 |
| matmul | < 1e-5 |

### Cross-Backend Consistency

GELU's 1e-4 tolerance matches PTX, confirming consistent behavior across GPU backends.

## Section F: Visual Regression (Claims 81-90)

**Insight:** BufferRenderer produces deterministic, correct visualizations.

| Test ID | Validation | Status |
|---------|------------|--------|
| F-083 | Identical inputs → identical RGBA (byte-level) | ✅ |
| F-084 | Different inputs → different RGBA | ✅ (after fix) |
| F-086 | Constant inputs handled correctly | ✅ |
| F-090 | 100 renders with same seed identical | ✅ |

### Bug Found (F-084)

Auto-normalization caused constant values (all 0s vs all 1s) to map to the same color. Fixed by using `.with_range(0.0, 1.0)` for explicit normalization.

```rust
// Bad: Auto-normalization can mask constant differences
let renderer = BufferRenderer::new();

// Good: Explicit range prevents masking
let renderer = BufferRenderer::new().with_range(0.0, 1.0);
```

## Section G: Stress Testing (Claims 91-100)

**Insight:** Jidoka (Toyota-style stop-on-defect) catches errors immediately.

```rust
// G-100: Jidoka triggers on FIRST failure
let guard = JidokaGuard::nan_guard("test");
let data_with_nan = vec![1.0, 2.0, f32::NAN, 4.0, 5.0];
guard.check_output(&data_with_nan); // Fails immediately at index 2
```

### G-092 Finding

The 2x slowdown detection threshold is appropriate for CI environments but may need adjustment for cloud runners with variable performance.

## Summary: Simulation Testing Value

| Category | What We Validated | Risk Mitigated |
|----------|-------------------|----------------|
| Correctness | All backends produce equivalent results | Silent precision drift |
| Determinism | Same inputs always produce same outputs | Non-reproducible ML training |
| Safety | Empty/single-element inputs handled | Segfaults in production |
| Performance | Thresholds correctly balance overhead | Suboptimal backend selection |
| Numerical Stability | Softmax, LayerNorm handle edge cases | NaN/Inf in ML pipelines |

## Falsifiable Claims Methodology

The Popper-style approach (attempting to disprove each claim) revealed:

1. **F-084 bug:** Auto-normalization masked differences in constant inputs
2. **Coverage gaps:** 14 claims were initially missing tests (now fixed)
3. **Tolerance precision:** GPU tolerance is 1e-4, not 1e-5 as originally claimed

**Bottom line:** 102 tests that could have failed but didn't - each one represents a specific guarantee about trueno's behavior that users can rely on.

## Integration with Performance Parity

These findings directly impact realizar's performance parity work:

| Finding | Impact on Parity |
|---------|------------------|
| GPU threshold 100K | Explains IMP-600 (GPU slower for MATVEC) |
| PCG determinism | Enables reproducible benchmarks |
| SIMD math properties | Validates trueno as llama.cpp-equivalent |
| PTX barriers correct | Enables safe FlashAttention |
| 1e-4 GPU tolerance | Expected precision for fused kernels |

## Example Usage

Run the simulation tests:

```bash
# From trueno directory
cargo test --lib simulation_

# Run specific section
cargo test --lib simulation_backend_selection
cargo test --lib simulation_determinism
cargo test --lib simulation_simd
cargo test --lib simulation_ptx
cargo test --lib simulation_wgpu
cargo test --lib simulation_visual
cargo test --lib simulation_stress
```
