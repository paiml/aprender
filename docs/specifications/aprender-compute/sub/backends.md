# Sub-spec: Multi-Backend Architecture

**Parent:** [trueno-spec.md](../trueno-spec.md) Sections 3, 4

---

## 1. Backend Selection

`Backend::Auto` resolves at `Vector` creation time via `is_x86_feature_detected!()`. This runs once — not per-operation.

**Priority order:**
1. CUDA (NVIDIA GPU + parallel workload)
2. wgpu (cross-platform GPU + >100K elements)
3. AVX-512 (Zen4/Sapphire Rapids+)
4. AVX2+FMA (preferred x86_64)
5. AVX
6. SSE2 (baseline x86_64)
7. NEON (ARM64)
8. SIMD128 (WASM)
9. Scalar (always available)

## 2. OpComplexity

GPU dispatch thresholds depend on operation complexity:

| Complexity | Examples | GPU threshold |
|------------|----------|---------------|
| Low | add, mul, relu | >1M elements |
| Medium | dot, reduce, softmax | >100K elements |
| High | matmul, conv2d, attention | >10K elements |

Below threshold → SIMD. Above → GPU (if available).

## 3. Backend Story Policy

**Every operation MUST work on ALL backends.** No exceptions.

Implementation checklist for new operation `frobulate()`:

1. **Contract first:** `contracts/frobulate-v1.yaml`
2. **Register binding:** `../provable-contracts/contracts/trueno/binding.yaml`
3. **Trait method:** `VectorBackend::frobulate()` in `src/backends/mod.rs`
4. **Scalar:** `src/backends/scalar/` — pure Rust, baseline correctness
5. **SSE2:** `src/backends/sse2/` — 4x f32 per iteration
6. **AVX2:** `src/backends/avx2/` — 8x f32, FMA if applicable
7. **AVX-512:** `src/backends/avx512/` — 16x f32
8. **NEON:** `src/backends/neon/` — 4x f32 (ARM)
9. **WASM:** `src/backends/wasm/` — 4x f32 (SIMD128)
10. **wgpu shader:** `src/backends/gpu/shaders/`
11. **wgpu device:** `src/backends/gpu/device/` — sync + async methods
12. **Integration test:** `tests/backend_story.rs`

If GPU acceleration is not beneficial (e.g., inherently sequential), the GPU method MUST:
- Fall back to CPU implementation
- Document why in a comment
- Still pass the backend story test

## 4. Dispatch Implementation

```rust
// src/backends/mod.rs — simplified dispatch pattern
match self.backend {
    Backend::Avx512 => unsafe { avx512::frobulate(a, result) },
    Backend::Avx2   => unsafe { avx2::frobulate(a, result) },
    Backend::Sse2   => unsafe { sse2::frobulate(a, result) },
    Backend::Neon   => unsafe { neon::frobulate(a, result) },
    Backend::Wasm   => unsafe { wasm::frobulate(a, result) },
    Backend::Scalar => scalar::frobulate(a, result),
}
```

## 5. Enforcement

- **Integration test:** `tests/backend_story.rs` tests all backends
- **CI:** runs backend story tests on every PR
- **Contract:** FALSIFY tests verify backend equivalence (tolerance < 1e-5)
