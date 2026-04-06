# Introduction

**Trueno** (Spanish: "thunder") is a high-performance Rust library providing unified compute primitives across three execution targets: CPU SIMD, GPU, and WebAssembly. The name reflects the library's mission: to deliver thunderous performance through intelligent hardware acceleration.

## The Problem: Performance vs Portability

Modern applications face a critical tradeoff:

- **Hand-optimized assembly**: Maximum performance (2-50x speedup), but unmaintainable and platform-specific
- **Portable high-level code**: Easy to write and maintain, but leaves performance on the table
- **Unsafe SIMD intrinsics**: Good performance, but riddled with `unsafe` code and platform-specific complexity

Traditional approaches force you to choose between performance, safety, and portability. **Trueno chooses all three.**

## The Solution: Write Once, Optimize Everywhere

Trueno's core philosophy is **write once, optimize everywhere**:

```rust
use trueno::Vector;

// Single API call, multiple backend implementations
let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
let b = Vector::from_slice(&[5.0, 6.0, 7.0, 8.0]);
let result = a.add(&b)?;

// Automatically selects best backend:
// - AVX2 on modern Intel/AMD (4-8x speedup)
// - NEON on ARM64 (2-4x speedup)
// - GPU for large workloads (10-50x speedup)
// - WASM SIMD128 in browsers (2x speedup)
```

## Key Features

### 1. Multi-Target Execution

Trueno runs on **three execution targets** with a unified API:

| Target | Backends | Use Cases |
|--------|----------|-----------|
| **CPU SIMD** | SSE2, AVX, AVX2, AVX-512 (x86)<br>NEON (ARM)<br>SIMD128 (WASM) | General-purpose compute, small to medium workloads |
| **GPU** | CUDA (NVIDIA via trueno-gpu)<br>Vulkan, Metal, DX12, WebGPU via `wgpu` | Large workloads (100K+ elements), parallel operations |
| **WebAssembly** | SIMD128 portable | Browser/edge deployment, serverless functions |

### 2. Runtime Backend Selection

Trueno automatically selects the best available backend at runtime:

```
┌─────────────────────────────────────────────────┐
│           Trueno Public API (Safe)              │
│  compute(), map(), reduce(), transform()        │
└─────────────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌────────┐   ┌─────────┐   ┌──────────┐
   │  SIMD  │   │   GPU   │   │   WASM   │
   │ Backend│   │ Backend │   │  Backend │
   └────────┘   └─────────┘   └──────────┘
        │             │             │
   ┌────┴────┐   ┌────┴────┐   ┌───┴─────┐
   │ Runtime │   │CUDA/wgpu│   │ SIMD128 │
   │ Detect  │   │ Compute │   │ Portable│
   └─────────┘   └─────────┘   └─────────┘
   │  │  │  │       │
   SSE2 AVX NEON   PTX
      AVX512     (trueno-gpu)
```

**Backend Selection Priority:**
1. GPU (if available + workload > 100K elements)
2. AVX-512 (if CPU supports)
3. AVX2 (if CPU supports)
4. AVX (if CPU supports)
5. SSE2 (baseline x86_64)
6. NEON (ARM64)
7. SIMD128 (WASM)
8. Scalar fallback

### 3. Zero Unsafe in Public API

All `unsafe` code is isolated to backend implementations:

```rust
// ✅ SAFE public API
pub fn add(&self, other: &Self) -> Result<Self> {
    // Safe bounds checking, validation
    if self.len() != other.len() {
        return Err(TruenoError::SizeMismatch { ... });
    }

    // ❌ UNSAFE internal implementation (isolated)
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe { self.add_avx2(other) }
    } else {
        self.add_scalar(other) // Safe fallback
    }
}
```

**Safety guarantees:**
- Public API is 100% safe Rust
- All bounds checked before dispatching to backends
- Miri validation for undefined behavior
- 286 documented SAFETY invariants in backend code

### 4. Proven Performance

Trueno delivers **2-50x speedups** over scalar code:

| Operation | Size | Scalar | SSE2 | AVX2 | AVX-512 | GPU |
|-----------|------|--------|------|------|---------|-----|
| `add_f32` | 1K | 1.0x | 2.1x | 4.3x | 8.2x | - |
| `add_f32` | 100K | 1.0x | 2.0x | 4.1x | 8.0x | 3.2x |
| `add_f32` | 1M | 1.0x | 2.0x | 4.0x | 7.9x | 12.5x |
| `dot_product` | 1M | 1.0x | 3.1x | 6.2x | 12.1x | 18.7x |

All benchmarks validated with:
- Coefficient of variation < 5%
- 100+ iterations for statistical significance
- No regressions > 5% vs baseline

### 5. Extreme TDD Quality

Trueno is built with **EXTREME TDD** methodology:

- **97%+ test coverage** (verified with `make coverage`, 4600+ tests)
- **Property-based testing** (commutativity, associativity, distributivity)
- **Backend equivalence tests** (scalar vs SIMD vs GPU produce identical results)
- **Mutation testing** (>80% mutation kill rate with `cargo mutants`)
- **Zero tolerance for defects** (all quality gates must pass)

## Real-World Impact: The FFmpeg Case Study

**FFmpeg** (the world's most-used video codec library) contains:
- **390 assembly files** (~180,000 lines, 11% of codebase)
- **Platform-specific implementations** for x86, ARM, MIPS, PowerPC
- **Speedups**: SSE2 (2-4x), AVX2 (4-8x), AVX-512 (8-16x)

**Problems with hand-written assembly:**
- ❌ Unsafe (raw pointers, no bounds checking)
- ❌ Unmaintainable (390 files, must update all platforms)
- ❌ Non-portable (separate implementations per CPU)
- ❌ Expertise barrier (requires assembly knowledge)

**Trueno's value proposition:**
- ✅ **Safety**: Zero unsafe in public API
- ✅ **Portability**: Single source → x86/ARM/WASM/GPU
- ✅ **Performance**: 85-95% of hand-tuned assembly
- ✅ **Maintainability**: Rust type system catches errors at compile time

## Who Should Use Trueno?

Trueno is designed for:

1. **ML/AI Engineers** - NumPy-like compute primitives for Rust (use with aprender for training)
2. **Systems Programmers** - Eliminate unsafe SIMD intrinsics
3. **Game Developers** - Fast vector math for physics/graphics
4. **Scientific Computing** - High-performance numerical operations
5. **WebAssembly Developers** - Portable SIMD for browsers/edge
6. **Transpiler Authors** - Safe SIMD target for Depyler/Decy/Ruchy

## Design Principles

Trueno follows five core principles:

1. **Write once, optimize everywhere** - Single algorithm, multiple backends
2. **Safety via type system** - Zero unsafe in public API
3. **Performance must be proven** - Every optimization validated with benchmarks (≥10% speedup)
4. **Extreme TDD** - >90% coverage, mutation testing, property-based tests
5. **Toyota Way** - Kaizen (continuous improvement), Jidoka (built-in quality)

## What's Next?

- **[Getting Started](./getting-started/installation.md)** - Install Trueno and run your first program
- **[Architecture](./architecture/overview.md)** - Understand the multi-backend design
- **[API Reference](./api-reference/vector-operations.md)** - Explore available operations
- **[Performance](./performance/benchmarks.md)** - See benchmark results and optimization techniques
- **[Examples](./examples/vector-math.md)** - Learn from real-world use cases

## Project Status

Trueno is under active development at **Pragmatic AI Labs**:

- **Current Version**: 0.14.6 (15 phases delivered: vectors, matrices, SIMD, GPU, BLIS GEMM, quantization, attention, LZ4, and more)
- **Tests**: 4800+ tests across trueno and trueno-gpu
- **Coverage**: 95%+ line coverage (TDG grade: A+)
- **License**: MIT/Apache-2.0 dual-licensed
- **Repository**: [github.com/paiml/trueno](https://github.com/paiml/trueno)
- **Issues**: [github.com/paiml/trueno/issues](https://github.com/paiml/trueno/issues)

**Scope:**
- **Trueno**: Compute primitives (vectors, matrices, SIMD, GPU, quantization kernels) — NumPy equivalent
- **Aprender**: ML framework with autograd and training — PyTorch equivalent

Trueno is the compute backend for higher-level ML libraries. For neural networks and training, see [aprender](https://github.com/paiml/aprender).
