# Trueno Specification

**Unified high-performance compute primitives across CPU SIMD, NVIDIA CUDA, wgpu, and WebAssembly.**

Version 1.0 · April 2026 · Pragmatic AI Labs · [paiml/trueno](https://github.com/paiml/trueno)

---

> **Canonical spec.** This is the ONE specification for trueno. Detail lives in
> `docs/specifications/sub/`. Old specs in `docs/specifications/*.md` are
> superseded by this document — do not create new top-level spec files.

---

## Table of Contents

| # | Section | Sub-spec |
|---|---------|----------|
| 1 | [Philosophy](#1-philosophy) | |
| 2 | [Provable-Contract-First Design](#2-provable-contract-first-design) | [sub/contracts.md](sub/contracts.md) |
| 3 | [Multi-Backend Architecture](#3-multi-backend-architecture) | [sub/backends.md](sub/backends.md) |
| 4 | [Backend Story Policy](#4-backend-story-policy) | [sub/backends.md](sub/backends.md) |
| 5 | [CPU SIMD Backends](#5-cpu-simd-backends) | [sub/simd.md](sub/simd.md) |
| 6 | [CUDA Backend (trueno-gpu)](#6-cuda-backend-trueno-gpu) | [sub/cuda.md](sub/cuda.md) |
| 7 | [wgpu Backend](#7-wgpu-backend) | [sub/wgpu.md](sub/wgpu.md) |
| 8 | [WASM Backend](#8-wasm-backend) | [sub/simd.md](sub/simd.md) |
| 9 | [Layout Mandate (Q4K/Q6K)](#9-layout-mandate-q4kq6k) | [sub/layout.md](sub/layout.md) |
| 10 | [Crate Architecture](#10-crate-architecture) | |
| 11 | [Quality Gates](#11-quality-gates) | [sub/quality.md](sub/quality.md) |
| 12 | [Testing Requirements](#12-testing-requirements) | [sub/quality.md](sub/quality.md) |
| 13 | [Coverage](#13-coverage) | [sub/quality.md](sub/quality.md) |
| 14 | [Profiling & Tracing](#14-profiling--tracing) | [sub/profiling.md](sub/profiling.md) |
| 15 | [Blackwell Infrastructure](#15-blackwell-infrastructure) | [sub/cuda.md](sub/cuda.md) |
| 16 | [Safety Model](#16-safety-model) | |
| 17 | [Performance Contracts](#17-performance-contracts) | [sub/contracts.md](sub/contracts.md) |
| 18 | [BLIS GEMM Engine](#18-blis-gemm-engine) | [sub/blis.md](sub/blis.md) |
| 19 | [ComputeBrick & Profiling](#19-computebrick--profiling) | [sub/brick.md](sub/brick.md) |
| 20 | [PTX Optimizer](#20-ptx-optimizer) | [sub/cuda.md](sub/cuda.md) |
| 21 | [Runtime Contracts](#21-runtime-contracts) | [sub/contracts.md](sub/contracts.md) |
| 22 | [Activation One Path Rule](#22-activation-one-path-rule) | |
| 23 | [Contract-Aware Tracing (Tier 3)](#23-contract-aware-tracing-tier-3) | [sub/deep-integration.md](sub/deep-integration.md) |
| 24 | [Stack Integration](#24-stack-integration) | |
| 25 | [Development Commands](#25-development-commands) | |

---

## 1. Philosophy

Trueno exists because hand-written assembly is unsafe, unmaintainable, and non-portable. A single Rust source produces optimized code for x86, ARM, WASM, NVIDIA CUDA, and cross-platform GPU — with zero `unsafe` in the public API.

**Core invariants:**
- Write once, optimize everywhere via runtime dispatch
- Every optimization must prove ≥10% speedup via benchmarks
- >90% test coverage, mutation testing, property-based tests
- Contract-first: no kernel ships without a provable contract

---

## 2. Provable-Contract-First Design

**Every kernel implementation MUST begin with a YAML contract in `contracts/`.** The contract is the specification; the Rust code is the implementation. This is non-negotiable.

### The Contract-First Workflow

```
1. Write YAML contract       → contracts/my-kernel-v1.yaml
2. Define equations           → mathematical specification + pre/postconditions
3. Define FALSIFY tests       → how to disprove correctness
4. Define proof obligations   → formal properties (tolerance, equivalence)
5. Register binding           → ../provable-contracts/contracts/trueno/binding.yaml
6. Generate scaffold          → pv generate contracts/my-kernel-v1.yaml
7. Implement kernel           → fill in scaffold with real logic
8. Run FALSIFY tests + lint   → pv test + pv lint (7 gates)
9. Run Kani harnesses         → cargo kani (bounded model checking)
10. Merge only if all pass
```

### Escape-Proof Six-Stage Pipeline

It must be *impossible* to ship code that violates a contract. Six stages, each gates the next:

```
A. Equation (YAML)          → mathematical ground truth must exist
B. Lean 4 Proof             → theorem must have no sorry
C. YAML Validation          → pv lint Gates 1-7 must pass
D. build.rs Codegen         → sets CONTRACT_* env vars from binding.yaml
E. #[contract] Proc Macro   → checks env vars, inserts debug_assert pre/post
F. Test Execution           → cargo test runs FALSIFY tests
```

### Build-Time Enforcement (build.rs)

`build.rs` reads `../provable-contracts/contracts/trueno/binding.yaml`, sets `CONTRACT_{STEM}_{EQUATION}={status}` env vars, and enforces the **AllImplemented policy** — any `not_implemented` binding panics the build.

```
binding.yaml → parse → CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX=implemented
                      → CONTRACT_GEMM_BACKWARD_TILED_V1_BACKWARD_A_GEMM=implemented
                      → if any not_implemented: panic!()
```

### Binding Stats

| Metric | Count |
|--------|-------|
| Total bindings | 38 |
| Implemented | 38 |
| Critical path equations | 8 (softmax, matmul, silu, gelu, execute_matmul, PipelineCache, GemmBackward, arithmetic_intensity) |

### Contract Inventory (28 local contracts)

| Domain | Contracts |
|--------|-----------|
| Core kernels | gemv, softmax, elementwise, transpose |
| BLAS | level3, trsm |
| FFT | stockham, bluestein, 2d, 3d |
| Image | conv2d, resize, canny, color, histogram |
| Sparse | formats, spmv, spmm, spgemm, bsr |
| Solvers | cholesky, lu, qr, svd |
| Random | philox, threefry |
| Quantization | tensor-contraction |
| GPU | dimension-independent-kernels |

### Verification Ladder

| Level | Method | Tool | Trueno status |
|-------|--------|------|--------------|
| L5 | Theorem proving | Lean 4 | 53 theorems, 22 domains, 0 sorry, all eqs covered |
| L4 | Bounded model check | Kani | YAML-defined, not yet in CI |
| L3 | Property-based test | proptest | Active |
| L2 | Falsification test | `#[test]` | Active, all contracts |
| L1 | Type system + traits | rustc | Active, AllImplemented enforced |
| L0 | Code review + lint | pv lint, pmat comply | Active |

See [sub/contracts.md](sub/contracts.md) for binding.yaml schema, escape analysis, `#[contract]` macro, trait enforcement, `pv lint` gates, KAIZEN workflow, and contract schema reference.

---

## 3. Multi-Backend Architecture

```
Public API (safe) → Backend Dispatch → {SIMD, CUDA, wgpu, WASM, Scalar}
```

**Default backend selection** (`Backend::Auto`, resolved once at `Vector` creation via OnceLock):
1. **AVX2+FMA** — preferred x86_64 (safer than AVX-512 for memory-bound ops)
2. **AVX** — fallback x86_64
3. **SSE2** — baseline x86_64
4. **NEON** — ARM64
5. **SIMD128** — WASM
6. **Scalar** — always available

**AVX-512** is NOT auto-selected — only used for ComputeBound operations via `select_backend_for_operation()`. GPU backends (CUDA, wgpu) are dispatched separately based on workload size and OpComplexity.

See [sub/backends.md](sub/backends.md) for dispatch logic and OpComplexity thresholds.

---

## 4. Backend Story Policy

**ZERO TOLERANCE: every operation MUST work on ALL backends.** No exceptions. If GPU acceleration is not beneficial, the GPU method falls back to CPU and documents why.

When adding a new operation:
1. **Write contract FIRST** (`contracts/my-op-v1.yaml`) — equations, FALSIFY tests, proof obligations
2. Register binding in `../provable-contracts/contracts/trueno/binding.yaml`
3. Add to `VectorBackend` trait (`src/backends/mod.rs`)
4. Implement in all backend modules: `scalar/`, `sse2/`, `avx2/`, `avx512/`, `neon/`, `wasm/`, `gpu/`, `q4k/`, `q6k/`
5. Add WGSL shader if GPU-accelerable
6. Add sync + async device methods
7. Add integration test in `tests/backend_story.rs`

Enforcement: `tests/backend_story.rs` + CI.

---

## 5. CPU SIMD Backends

| Backend | Width | Elements (f32) | Detection |
|---------|-------|-----------------|-----------|
| SSE2 | 128-bit | 4 | Baseline x86_64 |
| AVX | 256-bit | 8 | `is_x86_feature_detected!("avx")` |
| AVX2+FMA | 256-bit | 8 | Preferred for most ops |
| AVX-512 | 512-bit | 16 | ComputeBound ops only (`avx512f` feature flag) |
| NEON | 128-bit | 4 | Baseline ARM64 |

**Critical patterns:**
- Always handle remainder: `len % lane_width` with scalar fallback
- Wrap intrinsics in `#[target_feature(enable = "...")]` functions
- Every `unsafe` block needs a `// SAFETY:` comment

See [sub/simd.md](sub/simd.md) for lane widths, FMA patterns, and horizontal reduction techniques.

---

## 6. CUDA Backend (trueno-gpu)

Pure Rust PTX generation — no nvcc, no LLVM, no external toolchains. The `trueno-gpu` crate generates PTX strings from Rust at compile-time or runtime.

**Available kernels:** GEMM (naive/tiled/tensor core), Softmax, LayerNorm, Attention (FlashAttention-style), Q4_K dequantization, 6 backward kernels (activations, cross_entropy, gemm, layer_norm, rms_norm, softmax).

**Key APIs:** `PtxModule`, `PtxKernel`, `KernelBuilder`, `Kernel::emit_ptx()`.

**Testing without GPU:** All `build_ptx()` / `emit_ptx()` functions are pure string generators — test by checking `.version`, `.entry`, `.target` directives.

See [sub/cuda.md](sub/cuda.md) for PTX generation details, register allocation, Blackwell workarounds, and the dimension-independent kernel plan.

---

## 7. wgpu Backend

Cross-platform GPU compute via Vulkan/Metal/DX12/WebGPU. No CUDA required.

**Inference:** `WgslForwardPass` — RMSNorm, GEMV (cooperative K-reduction, vec4 loads), SiLU, RoPE. GEMV for M=1, tiled GEMM for M>1. 27.6 tok/s on Radeon Pro W5700X.

**Training:** 9 shaders in `src/backends/gpu/shaders/backward.rs` — 6 backward (silu, gemm_a, gemm_b, rmsnorm, rope, cross_entropy), plus adamw_step optimizer, nf4_dequant, and cross_entropy_forward. All FALSIFY tests pass. Enables full training loop on AMD/Intel/Apple.

**GPU threshold:** Only dispatch to GPU for >100K elements (PCIe transfer ~0.5ms).

See [sub/wgpu.md](sub/wgpu.md) for shader source locations, `GpuMatmulCache`, and provable contracts.

---

## 8. WASM Backend

Portable SIMD128 for browser/edge deployment. 4x f32 per lane (vs 8x AVX2). No GPU support in standard WASM — WebGPU is separate.

Build: `cargo build --target wasm32-unknown-unknown`

---

## 9. Layout Mandate (Q4K/Q6K)

**LAYOUT-002:** The Sovereign AI Stack uses **row-major exclusively** for APR/GGUF data. Column-major kernels exist for internal BLAS-style ops only.

Garbage inference output (`"olumbia+lsi nunca/localENTS"`) = wrong kernel layout. Aprender handles GGUF→APR transpose during import (`src/format/converter/write.rs`).

Pipeline: `GGUF (col-major) → aprender transpose → APR (row-major) → realizar → trueno row-major kernels`

See [sub/layout.md](sub/layout.md) for kernel selection guide and fused Q4K spec reference.

---

## 10. Crate Architecture

```
trueno/                  Main crate (CPU SIMD + wgpu)
├── src/backends/        scalar/, sse2/, avx2/, avx512/, neon/, wasm/, gpu/, q4k/, q6k/
├── src/vector/          Vector<T> + VectorOps trait
├── src/matrix/          matmul, transpose
├── src/blis/            BLIS micro-kernel delegation
├── src/brick/           ComputeBrick, BrickProfiler, quant_ops
├── src/eigen/           Eigendecomposition
├── src/monitor/         GPU monitoring, ComputeDevice trait
├── src/tiling/          Cache-aware tiling
├── src/tuner/           ML-based backend tuner
└── src/error.rs         TruenoError

trueno-gpu/              CUDA sub-crate (pure Rust PTX)
├── src/ptx/             PTX builder, instructions, registers, optimizer
├── src/kernels/         gemm, softmax, layernorm, attention, quantize, backward, lz4
├── src/driver/          CUDA driver FFI
└── src/memory/          DeviceBuffer, HostBuffer, pool

crates/                  Domain sub-crates
├── cbtop                Compute Block Top TUI + adaptive ML
├── trueno-fft           FFT (Stockham, Bluestein, 2D, 3D)
├── trueno-image         Image processing (conv2d, resize, canny)
├── trueno-quant         Quantization (Q4K, Q5K, Q6K, NF4)
├── trueno-rand          RNG (Philox, ThreeFry)
├── trueno-solve         Solvers (Cholesky, LU, QR, SVD)
├── trueno-sparse        Sparse (CSR, SELL, BSR, SpMV, SpGEMM)
└── trueno-tensor        Tensor contraction

contracts/               28 YAML provable contracts (source of truth)
```

---

## 11. Quality Gates

**Every commit:** clippy clean, all tests pass, ≥90% coverage, rustfmt, PMAT TDG ≥ B+.

**Every PR:** Tests for new code (all 5 categories), rustdoc, benchmarks prove ≥10% speedup, mutation testing ≥80% kill rate, contract FALSIFY tests pass.

**Every release:** CI green, repo-score ≥90/110, changelog updated, semver bump, git tag.

```bash
cargo clippy --all-features -- -D warnings
cargo test --all-features
make coverage                                    # ≥90% or commit blocked
pmat analyze tdg --min-grade B+
pmat repo-score . --min-score 90
cargo mutants --timeout 120 --minimum-pass-rate 80
```

See [sub/quality.md](sub/quality.md) for coverage enforcement, test categories, and mutation testing details.

---

## 12. Testing Requirements

Five mandatory test categories for every operation:

1. **Unit** — correctness, empty inputs, NaN/infinity/subnormal edge cases
2. **Property-based** (proptest) — commutativity, associativity, distributivity
3. **Backend equivalence** — all backends produce identical results (f32 tolerance < 1e-5)
4. **Mutation** — ≥80% kill rate (`cargo mutants`)
5. **Benchmark** — prove ≥10% speedup vs scalar baseline

---

## 13. Coverage

**≥90% line coverage is non-negotiable.** Enforced by `make coverage-check` and CI.

- ONLY use `make coverage` — never `cargo llvm-cov` directly, never `cargo-tarpaulin`
- New code must have 100% coverage
- HTML report: `target/coverage/html/index.html`

| Component | Minimum | Target |
|-----------|---------|--------|
| Public API | 100% | 100% |
| SIMD backends | 90% | 95% |
| GPU backend | 85% | 90% |
| Overall | **90%** | **95%+** |

---

## 14. Profiling & Tracing

Renacer v0.5.0+ for syscall tracing, function profiling, flamegraphs, and OTLP export.

```bash
make profile                     # benchmark profiling
make profile-flamegraph          # flamegraph
make profile-otlp-jaeger         # traces → Jaeger (localhost:16686)
```

**Golden trace validation** (`renacer.toml`): CI fails if syscall count or latency exceeds budget. Captures baseline traces for backend_detection, matrix_ops, activations, similarity.

See [sub/profiling.md](sub/profiling.md) for OTLP best practices and golden trace details.

---

## 15. Blackwell Infrastructure

**JIT Bug (trueno#200):** `cuModuleLoadDataEx` fails on sm_121 during active GPU work. Forward kernels work after pre-warming; backward kernels crash during training. Inference unaffected (cuBLAS/SIMD path).

**Fix (trueno#203):** Dimension-independent kernels (M,K,N as runtime params → ~15 types vs 50+ variants) + pre-compiled cubin pipeline: `build.rs → nvcc → include_bytes!() → zero JIT at runtime`.

Contract: `contracts/dimension-independent-kernels-v1.yaml` (6 FALSIFY tests).

---

## 16. Safety Model

- `unsafe` ONLY in backend implementations — never in public API
- Every `unsafe` block has a `// SAFETY:` comment explaining invariants
- SIMD intrinsics wrapped in `#[target_feature]` functions
- Public APIs are bounds-checked with `Result<T, TruenoError>`
- SIMD loops always handle remainder with scalar fallback

---

## 17. Performance Contracts

Every contract in `contracts/` tracks measured performance:

```yaml
performance:
  baseline: scalar
  measured_ratio: 1.53           # vs scalar baseline
  measured_throughput: "16.3 Gelem/s"
  regression_threshold: 5%      # CI fails on >5% regression
```

Benchmark validation: ≥100 iterations, CV <5%, results saved to `target/criterion/`.

### Performance Targets

**Mandatory: ≥1.5x ndarray on ALL operations at ALL sizes.**
**Stretch: ≥2.0x ndarray on all operations.**

Comparison baseline: ndarray 0.17 (matrixmultiply 0.3 backend). Trueno v0.17.0 with `--features parallel` on AMD Ryzen Threadripper 7960X (24-core, AVX-512).

| Op | Size | Trueno (ns) | ndarray (ns) | Ratio | Target | Stretch | Status |
|----|------|-------------|--------------|-------|--------|---------|--------|
| Transpose | 64 | 167 | 1,262 | **7.56x** | ≥1.5x | ≥2.0x | ✅ stretch met |
| Transpose | 128 | 1,828 | 8,032 | **4.40x** | ≥1.5x | ≥2.0x | ✅ stretch met |
| Transpose | 256 | 10,247 | 52,111 | **5.09x** | ≥1.5x | ≥2.0x | ✅ stretch met |
| Transpose | 512 | 73,166 | 419,690 | **5.74x** | ≥1.5x | ≥2.0x | ✅ stretch met |
| GEMM | 64 | 4,300 | 5,160 | **1.20x** | ≥1.5x | ≥2.0x | ⬆ below target |
| GEMM | 128 | 33,100 | 36,600 | **1.10x** | ≥1.5x | ≥2.0x | ⬆ below target |
| GEMM | 256 | 170,000 | 274,000 | **1.61x** | ≥1.5x | ≥2.0x | ✅ target met (was 1.02x) |
| GEMM | 512 | 864,000 | 2,206,000 | **2.55x** | ≥1.5x | ≥2.0x | ✅ stretch met (was 1.01x) |
| GEMM | 1024 | 3,970,000 | 17,500,000 | **4.41x** | ≥1.5x | ≥2.0x | ✅ stretch met (was 0.97x) |
| GEMV | 64 | 143 | 686 | **4.80x** | ≥1.5x | ≥2.0x | ✅ stretch met |
| GEMV | 128 | 590 | 2,483 | **4.21x** | ≥1.5x | ≥2.0x | ✅ stretch met |
| GEMV | 256 | 2,317 | 9,304 | **4.02x** | ≥1.5x | ≥2.0x | ✅ stretch met |
| GEMV | 512 | 11,771 | 35,861 | **3.05x** | ≥1.5x | ≥2.0x | ✅ stretch met |
| GEMV | 1024 | 57,630 | 141,360 | **2.45x** | ≥1.5x | ≥2.0x | ✅ stretch met |
| Vec Add | 1K | 51 | 132 | **2.58x** | ≥1.5x | ≥2.0x | ✅ stretch met |
| Vec Add | 10K | 878 | 896 | **1.02x** | ≥1.5x | ≥2.0x | ❌ bandwidth ceiling |
| Vec Add | 100K | 10,271 | 10,588 | **1.03x** | ≥1.5x | ≥2.0x | ❌ bandwidth ceiling |
| Vec Add | 1M | 108,740 | 112,980 | **1.04x** | ≥1.5x | ≥2.0x | ❌ bandwidth ceiling |
| ReLU | 1K | 36 | 85 | **2.34x** | ≥1.5x | ≥2.0x | ✅ stretch met |
| ReLU | 10K | 550 | 570 | **1.04x** | ≥1.5x | ≥2.0x | ❌ bandwidth ceiling |
| ReLU | 100K | 6,222 | 6,394 | **1.03x** | ≥1.5x | ≥2.0x | ❌ bandwidth ceiling |
| Softmax | 128 | 91 | 350 | **3.85x** | ≥1.5x | ≥2.0x | ✅ stretch met |
| Softmax | 1K | 483 | 2,794 | **5.78x** | ≥1.5x | ≥2.0x | ✅ stretch met |
| Softmax | 4K | 1,817 | 11,336 | **6.24x** | ≥1.5x | ≥2.0x | ✅ stretch met |
| Softmax | 32K | 13,838 | 88,983 | **6.43x** | ≥1.5x | ≥2.0x | ✅ stretch met |

**Score: 17/26 ops ≥1.5x target (65%). 16/26 ≥2.0x stretch (62%). 13/26 ≥3.0x (50%).**

v0.17.0 final results: GEMM 512 1.01x→2.55x and GEMM 1024 0.97x→4.41x (rayon parallel, NR=8 BLIS with AVX-512, 24-core dispatch). GEMM 256 1.02x→1.61x (parallel dispatch at 8M FLOP threshold). All softmax 3.17-6.65x, all transpose 4.38-7.52x, all GEMV 2.87-4.53x. Elementwise vec add/ReLU at 10K-1M sizes are 1.02-1.08x (memory-bandwidth ceiling).

### GPU Performance (RTX 4090, CUDA 12.8)

**cuBLAS FP16 GEMM throughput** (vendor-optimized baseline):

| Size | Time (µs) | TFLOP/s | Efficiency (of 330T peak) |
|------|-----------|---------|--------------------------|
| 256×256 | 4.2 | 8.1 | 2.4% |
| 512×512 | 9.0 | 29.9 | 9.1% |
| 1024×1024 | 25.8 | 83.3 | 25.2% |
| 2048×2048 | 247 | 69.4 | 21.0% |
| 4096×4096 | 1,362 | 100.9 | 30.6% |

Peak: 100.9 TFLOP/s at 4096×4096 (30.6% of theoretical 330 TFLOP/s FP16 tensor core peak).

**PTX GEMM (pure Rust, no nvcc) vs cuBLAS — FP32:**

| Size | PTX (µs) | cuBLAS (µs) | Ratio | Verdict |
|------|----------|-------------|-------|---------|
| 32×32 | 3.7 | 7.6 | **PTX 2.0× faster** | ✅ |
| 64×64 | 5.3 | 9.4 | **PTX 1.8× faster** | ✅ |
| 128×128 | 8.6 | 5.7 | cuBLAS 1.5× faster | competitive |
| 256×256 | 15.9 | 11.8 | cuBLAS 1.3× faster | competitive |

Key finding: trueno-gpu's pure-Rust PTX GEMM kernels **beat cuBLAS** at small matrix sizes (≤64) due to lower kernel launch overhead. cuBLAS wins at 128+ where its vendor-optimized tiling dominates. All sizes within 2× — competitive with NVIDIA's own library.

**PTX generation performance** (pure Rust → PTX string):
- Module emit: 272ns
- Kernel build: 84ns
- Module build: 16.5ns

1,047/1,047 kernel PTX generation tests pass.

### Optimizations Applied (v0.17.0, April 2026)

1. **Rayon parallel GEMM dispatch**: Parallel outer-loop tiling at 8M FLOP threshold. 24-core Threadripper 7960X scales near-linearly for large GEMM. GEMM 1024: 0.97x→4.41x. GEMM 512: 1.01x→2.55x. GEMM 256: 1.02x→1.61x.

2. **NR=8 BLIS with AVX-512 microkernel**: Row-major C SIMD load/store with 16×8 AVX-512 microkernel. Native 512-bit execution on Zen 4 (not double-pump). SIMD A packing for full throughput.

3. **LLVM autovectorization for bandwidth-bound ReLU**: Hand-written AVX2 intrinsics were 40% SLOWER than LLVM autovectorized loop. Root cause: calling convention overhead from `#[target_feature]` functions forces the compiler to save/restore SIMD registers at each call boundary. Fix: use simple `for i in 0..n { output[i] = input[i].max(0.0) }` which LLVM vectorizes optimally with zero calling convention overhead.

4. **AVX-512 BLIS 5-loop** (`gemm_blis_avx512_packed`): Full BLIS cache-blocked GEMM with MR_512=16, NR_512=8 packing for 257-768 dimensions.

5. **MC cache blocking optimization**: MC=72→128 (16×MR). Reduces packing cycles by 1.78x for the ic-loop. Zen 4 L2 = 1MB/core; MC×KC×4B = 128×256×4 = 128KB << 1MB.

6. **`gemm_direct_rowmajor`** (prior): Zero-pack row-major GEMM for ≤128×128. No packing overhead. Broadcast A from row-major, SIMD load B contiguously.

7. **AVX-512 frequency throttling awareness (Zen 4)**: AMD Zen 4 reduces clock by ~15-30% for AVX-512 instructions. For bandwidth-bound ops, lower clock = fewer bytes/second. Solution: use AVX2 (full clock) for bandwidth-bound ops, AVX-512 only for compute-bound (GEMM where FMA throughput dominates).

### Root Cause Analysis (remaining gaps)

1. **GEMM 64 (1.20x), 128 (1.10x)**: Single-threaded near compute peak -- both libraries at ~80-90% of theoretical FMA throughput. Rayon overhead exceeds parallel benefit at these sizes. These are effectively at parity for the compute regime. **Fix**: Inline microkernel for ≤128 avoiding pack entirely, or lower parallel FLOP threshold for Threadripper.

2. **Vec add/ReLU 10K-1M (1.02-1.08x)**: Memory-bandwidth ceiling. Total working set (2-3 arrays x 40KB-4MB) is serviced at ~95% of peak DRAM bandwidth by both libraries. This is a physical limit, not an algorithmic gap. **Fix (PMAT-021)**: (a) fused op API (relu+add in single pass, halving bandwidth); (b) in-place operation variants to eliminate output allocation.

### Benchmark Command

```bash
cargo bench --bench gemm_comparison --features parallel
```

---

## 18. BLIS GEMM Engine

`src/blis/` implements BLIS-style blocked GEMM with cache hierarchy optimization (L2→L1→registers). Micro-kernels: `8x6` AVX2 true-ASM (BLIS blocked path), `8x8` AVX2+FMA 4-way K-unrolled (small-matrix path), `8x8` direct rowmajor (zero-pack path, ≤128), `16x8` AVX-512 (Zen 4 / Intel, BLIS + small paths), `8x8` NEON.

**Dispatch hierarchy** (in `gemm_blis`):
1. **m\*n\*k < 4096**: `gemm_reference` (scalar, correctness only)
2. **m,n ≤ 128, m%8=0, n%8=0**: `gemm_direct_rowmajor` — zero packing, row-major C SIMD load/store, broadcast A elements, 4-way K-unrolled. Eliminates ~2µs overhead for 64×64.
3. **m,n ≤ 256, AVX-512, m%16=0, n%8=0**: `gemm_small_avx512_16x8` — 16×8 tiles, pre-packed B, SIMD transpose A. 128 outputs/tile (2.67× AVX2 8×6).
4. **m,n ≤ 256, m%8=0, n%8=0**: `gemm_small_nopack_8x8` — pre-packed B, SIMD transpose A, 8×8 µkernel.
5. **>256, AVX2+FMA**: `gemm_blis_nr8_rowmajor_c` — BLIS 5-loop with NR=8, row-major C SIMD load/store. MC=64, KC=256, NC=1024 (matching matrixmultiply's cache params). Eliminates scalar C tile overhead: 16 SIMD ops vs 96 scalar ops per tile.
6. **>256 fallback**: BLIS 5-loop with AVX2 8×6 microkernel, MC=128, KC=256, NC=4096.

**Cache blocking constants** (April 2026):
- AVX2: MR=8, NR=6, MC=128, KC=256, NC=4096
- AVX-512: MR_512=16, NR_512=8, MC_512=128, KC_512=256, NC_512=4096
4. **m,n ≤ 256**: `gemm_small_strided_avx2` — direct strided 8×6 access.
5. **else**: 5-loop BLIS blocked with MC=72, KC=256, NC=4096, MR=8, NR=6, thread-local packed buffers.

**Parallel GEMM:** Adaptive thread count scaling — 4M FLOP single-thread threshold, 4 threads for <16M FLOPs, full pool for larger. HeijunkaScheduler partitions M dimension with balanced load.

**Toyota Production System integration:**
- **Jidoka** — `JidokaGuard` stops on numerical error (NaN, divergence >1e-3 from reference)
- **Heijunka** — `HeijunkaScheduler` for load-balanced parallel GEMM
- **Kaizen** — `BlisProfiler` tracks per-level (L3/L2/L1/micro) timing

Backend selection via `BackendCostModel` with roofline analysis. `gemm_profiled()` returns profiling stats alongside results.

See [sub/blis.md](sub/blis.md) for micro-kernel patterns, packing layout, and cost model.

---

## 19. ComputeBrick & Profiling

`src/brick/` provides token-centric compute units — self-verifying blocks with budgets, assertions, and backends.

**Key types:**
- `ComputeBrick` — Composable compute unit with pre/postconditions
- `BrickProfiler` — O(1) hot-path profiling via `BrickId` enum (PAR-200)
- `ExecutionGraph` — Full execution path tracking with kernel checksums
- `ModelTracer` — Model-level inference tracing with tensor stats, attention weights, logit evolution

**Quantization ops:** `BlockQ5K`, `BlockQ6K`, `DotQ5KOp`, `DotQ6KOp` (llama.cpp compatible). Fused ops: `FusedQKVOp`, `FusedGateUpOp` for transformer inference.

**Integration:** `BrickTuner::get_tuner_recommendations()` in `src/tuner/` uses profiler data for kernel selection. SyncMode (Eager/Deferred) controls GPU synchronization granularity.

See [sub/brick.md](sub/brick.md) for the full brick taxonomy, profiling protocol, and tracing API.

---

## 20. PTX Optimizer

`trueno-gpu/src/ptx/optimize/` implements multi-pass PTX optimization:

| Pass | What it does | Reference |
|------|-------------|-----------|
| FMA fusion | `mul` + `add` → `fma` pattern matching | Click & Paleczny 1995 |
| Tile validation | Validate tile constraints, prevent register spill | Volkov & Demmel 2008 |
| Loop splitting | Split loops at conditional boundaries | NVIDIA CUDA Tile IR |
| TKO (Token ordering) | Memory dependency tracking, barrier elimination | NVIDIA Tile IR model |
| Barrier safety | Detect early-exit-before-barrier bugs (PARITY-114) | Five Whys 2026 |

Applied via `optimize()` in sequence. All passes are pure functions on PTX AST — no GPU required for testing.

---

## 21. Runtime Contracts

`src/contracts.rs` enforces kernel-level preconditions/postconditions at runtime. `src/generated_contracts.rs` is auto-generated from YAML via `pv codegen`.

**Three-layer contract hierarchy:**
1. **aprender** (import) — `enforce_architecture_completeness()`: validate tensor names
2. **realizar** (load) — `contract_gate::validate_model_load()`: validate architecture
3. **trueno** (kernel) — `contracts::validate_weight_buffer()`: validate bytes & layout

`STACK_LAYOUT = RowMajor` — the ONLY layout trueno kernels accept. Hard-errors on violation (no silent defaults).

Generated contracts use `debug_assert!()` — zero cost in release builds. Covers: activation kernels (gelu, relu, silu), matmul pre/postconditions, position encoding, active learning.

---

## 22. Activation One Path Rule

`src/activations.rs` defines canonical scalar activation functions per UCBD §4 (One Path Rule):

`silu_scalar()`, `gelu_scalar()`, `sigmoid_scalar()`, `relu_scalar()`, `tanh_scalar()`, plus `f16_to_f32()`/`f32_to_f16()` conversions.

**Downstream crates (aprender, realizar, entrenar, whisper-apr) MUST import from here** — re-implementing is a contract violation. SIMD-vectorized versions exist in `backends/*/ops/activations` but delegate to these canonical implementations for correctness.

---

## 23. Contract-Aware Tracing (Tier 3)

Tiers 1 (compile-time) and 2 (CI) enforce contracts statically. Tier 3 enforces at runtime via tracing integration — closing the gap between "contract says X" and "system does X in production."

### Architecture

```
Contract YAML → ContractRegistry (startup)
                    ↓
BrickProfiler ──→ budget check ──→ violation event
ModelTracer   ──→ postcondition check ──→ violation event
                    ↓
ContractTracingLayer (tracing::Layer)
                    ↓
Structured diagnostics (SARIF-compatible)
```

### Gap Closures

| Gap | Problem | Fix |
|-----|---------|-----|
| Gap 2 | ComputeBrick budget is hardcoded | `ComputeBrick::from_contract()` derives `TokenBudget` from roofline YAML |
| Gap 3 | ModelTracer observes but doesn't verify | `end_forward()` checks MLT-01..05 against contract invariants |

### ContractTracingLayer

A `tracing::Layer` that intercepts spans tagged with `contract.id` and verifies postconditions on span close. Violations emit structured `tracing::error!` events with contract ID, obligation, and measured value.

**Performance budget:** ≤130ns per check (NCCLbpf demonstrates this is achievable for GPU data paths). BrickProfiler's deferred sync mode batches checks with existing finalization — zero per-kernel overhead on hot path.

### ModelTracer Contract Hooks

`end_forward()` verifies existing trace data against contract invariants:
- **MLT-01**: no NaN/Inf in activations (activation-kernel-v1 postcondition)
- **MLT-02**: attention weights sum to 1 per row (softmax-kernel-v1 postcondition)
- **MLT-03**: logit magnitudes within bounds (model-config-algebra-v1)
- **MLT-04**: quantization error ≤ contract threshold (quantization-ordering-v1)
- **MLT-05**: KV cache utilization ≤ capacity (gpu-decode-profiling-v1)

Zero additional collection overhead — reuses existing trace data.

### Rust-Native Path (Future)

When Rust MCP-759 contracts stabilize (`#[contracts::requires]`/`#[contracts::ensures]`), YAML postcondition checks can migrate to compiler-inserted assertions with zero cost in release builds via `-Z contract-checks=off`.

**References:** ProofWright (arXiv:2511.12294), Volta (arXiv:2511.12638), NCCLbpf (arXiv:2603.11438), Rust MCP-759

See [sub/deep-integration.md](sub/deep-integration.md) for full design, code examples, and enforcement pipeline.

---

## 24. Stack Integration

Trueno is the compute foundation for the Sovereign AI Stack:
- **aprender** — tensor operations, format conversion
- **realizar** — fused inference kernels (uses trueno Q4K/Q6K)
- **entrenar** — training (blocked on Blackwell fix, trueno#200)
- **Depyler** — `np.dot()` → `trueno::Vector::dot()`
- **PMAT** — quality gates, pre-commit hooks, TDG grading

Stack-wide search: `batuta oracle --rag "your question"`

---

## 25. Development Commands

```bash
# Build
cargo build --all-features

# Test
cargo test --all-features

# Coverage (ONLY this command)
make coverage

# Lint
cargo clippy --all-features -- -D warnings && cargo fmt -- --check

# Bench
cargo bench --no-fail-fast

# Profile
make profile && make profile-flamegraph

# Quality
pmat analyze tdg --min-grade B+ && pmat repo-score . --min-score 90

# Code search (never grep for code discovery)
pmat query "simd kernel" --limit 10

# CUDA tests (requires GPU)
cargo test -p trueno-gpu --features cuda

# Stack search
batuta oracle --rag "your question"
```
