# First Principles Thinking

**First Principles Thinking** means building from fundamental truths rather than adopting existing frameworks with their inherited assumptions and technical debt.

## Core Principle

> Own every layer. External frameworks are borrowed complexity — first-principles implementations are permanent assets.

The Sovereign AI Stack builds each capability from scratch in pure Rust, producing a vertically integrated system with no opaque dependencies.

## Why First Principles?

### The Framework Tax

Traditional ML stacks depend on layers of borrowed complexity:

| Layer | Typical Stack | Sovereign AI Stack |
|-------|--------------|-------------------|
| Compute | PyTorch (C++/CUDA) | trueno (Rust, AVX2/AVX-512/NEON, wgpu) |
| ML | scikit-learn (Python/C) | aprender (Rust) |
| Inference | ONNX Runtime (C++) | realizar (Rust, fused quantized kernels) |
| Serving | Flask/FastAPI (Python) | batuta serve (Rust, async) |
| Distribution | Ray (Python/C++) | repartir (Rust, work-stealing) |
| Speech | Whisper (Python/PyTorch) | whisper-apr (Rust, WASM-first) |

Each external dependency brings: build complexity, ABI instability, Python runtime overhead, and opaque failure modes.

### What First Principles Gives You

```
No Python runtime    → Deploy as a single static binary
No C++ dependencies  → Cross-compile to any target
No CUDA SDK          → GPU via wgpu (Vulkan/Metal/DX12/WebGPU)
No framework lock-in → Swap any layer independently
WASM support         → Run ML in the browser
```

## First Principles in Batuta

### Compute: trueno

Instead of wrapping BLAS/LAPACK, trueno implements SIMD kernels directly:

```rust
// First principles: hand-written AVX2 dot product
// No opaque C library — every instruction is visible and auditable
#[cfg(target_arch = "x86_64")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();
    for i in (0..a.len()).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    hsum_avx2(sum)
}
```

### ML: aprender

Algorithms implemented from the math, not wrapped from scikit-learn:

```rust
// First principles: Random Forest from decision theory
// Not a binding to a C library — pure Rust, fully auditable
let model = RandomForest::builder()
    .n_trees(100)
    .max_depth(10)
    .criterion(SplitCriterion::Gini)
    .build(&training_data)?;
```

## The Stack Builds on Itself

Each layer depends only on the layers below it — no circular or external dependencies:

```
trueno          → SIMD/GPU primitives (no dependencies)
aprender        → ML algorithms (depends on trueno)
realizar        → Inference runtime (depends on trueno + aprender)
whisper-apr     → Speech recognition (depends on all three)
batuta          → Orchestrates everything
```

## Benefits

1. **Total auditability** - Every computation is visible in Rust source
2. **No supply chain risk** - No opaque native binaries in the dependency tree
3. **Cross-platform** - WASM, embedded, server — all from the same codebase
4. **Performance ownership** - Optimize any layer directly, no FFI boundaries
5. **Privacy by construction** - No telemetry, no cloud calls, sovereign by default

---

**Navigate:** [Table of Contents](../SUMMARY.md)
