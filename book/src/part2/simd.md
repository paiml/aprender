# SIMD Vectorization

SIMD (Single Instruction, Multiple Data) vectorization is the primary optimization target in Phase 3. The Trueno crate provides portable SIMD backends that accelerate element-wise and reduction operations across CPU architectures.

## Supported SIMD Backends

| Backend | Architecture | Register Width | Typical Speedup |
|---------|-------------|---------------|-----------------|
| AVX2 | x86-64 (Haswell+) | 256-bit (8 x f32) | 4-8x |
| AVX-512 | x86-64 (Skylake-X+) | 512-bit (16 x f32) | 8-16x |
| NEON | ARM (ARMv8+) | 128-bit (4 x f32) | 2-4x |
| Scalar | All | 32/64-bit | 1x (baseline) |

## Automatic Detection

Trueno detects the best available SIMD instruction set at runtime using `cpuid` (x86) or feature registers (ARM). When the `BackendSelector` returns `Backend::SIMD`, it maps to `trueno::Backend::Auto`, letting Trueno pick the optimal instruction set:

```rust
pub fn to_trueno_backend(backend: Backend) -> trueno::Backend {
    match backend {
        Backend::Scalar => trueno::Backend::Scalar,
        Backend::SIMD   => trueno::Backend::Auto,
        Backend::GPU    => trueno::Backend::GPU,
    }
}
```

## When SIMD Is Selected

The MoE router selects SIMD for:

- **Low complexity** operations (element-wise add, multiply) at 1M+ elements
- **Medium complexity** operations (reductions, dot product) at 10K-100K elements
- **High complexity** operations (matrix multiply) at 1K-10K elements

Below these thresholds, scalar code is sufficient. Above them, GPU dispatch becomes beneficial.

## Code Patterns That Benefit

| Pattern | Python | Trueno (SIMD) |
|---------|--------|--------------|
| Vector addition | `np.add(a, b)` | `a.add(&b)` |
| Element-wise multiply | `a * b` | `a.mul(&b)` |
| Dot product | `np.dot(a, b)` | `a.dot(&b)` |
| Sum reduction | `np.sum(a)` | `a.sum()` |
| Matrix multiply | `a @ b` | `mat_a.matmul(&mat_b)` |

## Example: Vector Addition

```rust
use trueno::Vector;

let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
let b = Vector::from_slice(&[5.0, 6.0, 7.0, 8.0]);
let c = a.add(&b).unwrap();
// c = [6.0, 8.0, 10.0, 12.0]
// Automatically uses AVX2/AVX-512/NEON based on CPU
```

## Verifying SIMD Usage

```bash
# Check which SIMD features are available
rustc --print cfg | grep target_feature

# Verify Trueno detected the correct backend
RUST_LOG=trueno=debug cargo run 2>&1 | grep "Selected backend"
```

## Portability

Code using `trueno::Backend::Auto` compiles and runs on any platform. On systems without SIMD support, Trueno falls back to scalar loops with identical results. No conditional compilation or feature flags are needed in user code.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
