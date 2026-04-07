# Sub-spec: CPU SIMD Backends

**Parent:** [trueno-spec.md](../trueno-spec.md) Sections 5, 8

---

## 1. Lane Widths

| ISA | Register | f32/lane | f64/lane | Detection |
|-----|----------|----------|----------|-----------|
| SSE2 | 128-bit | 4 | 2 | Baseline x86_64 (always available) |
| AVX | 256-bit | 8 | 4 | `is_x86_feature_detected!("avx")` |
| AVX2+FMA | 256-bit | 8 | 4 | `is_x86_feature_detected!("avx2","fma")` |
| AVX-512 | 512-bit | 16 | 8 | `is_x86_feature_detected!("avx512f")` |
| NEON | 128-bit | 4 | 2 | Baseline ARM64 |
| SIMD128 | 128-bit | 4 | 2 | WASM target |

## 2. Implementation Pattern

```rust
#[target_feature(enable = "avx2")]
unsafe fn op_avx2(a: &[f32], result: &mut [f32]) {
    let lanes = 8; // 256-bit / 32-bit
    let chunks = a.len() / lanes;

    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * lanes));
        let vr = /* operation on va */;
        _mm256_storeu_ps(result.as_mut_ptr().add(i * lanes), vr);
    }

    // CRITICAL: scalar remainder
    for i in (chunks * lanes)..a.len() {
        result[i] = /* scalar operation */;
    }
}
```

**Mandatory rules:**
- Every function has `#[target_feature(enable = "...")]`
- Every `unsafe` block has `// SAFETY:` comment
- Remainder handling is NOT optional — forgetting it is the #1 SIMD bug
- Use `_loadu_` (unaligned) unless alignment is guaranteed and documented

## 3. FMA Pattern (AVX2)

Fused multiply-add avoids intermediate rounding:

```rust
// a * b + c — single instruction, better precision
let result = _mm256_fmadd_ps(a, b, c);
```

Use FMA for dot products, matrix multiply, and any multiply-accumulate pattern.

## 4. Horizontal Reduction

Reducing a SIMD register to a single scalar (e.g., sum, max):

```rust
// AVX2 horizontal sum: 256-bit → scalar
#[target_feature(enable = "avx2")]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);        // upper 128
    let lo = _mm256_castps256_ps128(v);           // lower 128
    let sum128 = _mm_add_ps(lo, hi);              // 4 floats
    let shuf = _mm_movehdup_ps(sum128);           // [1,1,3,3]
    let sum64 = _mm_add_ps(sum128, shuf);         // [0+1,_,2+3,_]
    let shuf2 = _mm_movehl_ps(sum64, sum64);      // [2+3,_,_,_]
    let sum32 = _mm_add_ss(sum64, shuf2);         // [0+1+2+3]
    _mm_cvtss_f32(sum32)
}
```

## 5. WASM SIMD128

Same patterns as SSE2 but with `wasm32::` intrinsics. 4x f32 per lane. No GPU support in standard WASM — WebGPU is separate and accessed through the wgpu backend.

Build: `cargo build --target wasm32-unknown-unknown`

## 6. Common Pitfalls

1. **Forgetting remainder** — `len % lanes` elements MUST be handled
2. **Missing `#[target_feature]`** — compiler won't emit SIMD without it
3. **Aligned vs unaligned loads** — use `_loadu_` unless you guarantee alignment
4. **Cross-lane operations** — AVX2 permute/shuffle operates on 128-bit lanes, not the full 256 bits
5. **NEON differences** — ARM NEON has no horizontal operations in older ISA revisions
