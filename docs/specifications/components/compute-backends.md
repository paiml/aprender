# Compute Backends

Version: 1.0
Status: proposed
Date: 2026-04-09

**Version**: 1.0.0
**Status**: Active
**Parent**: [aprender-spec.md](../aprender-spec.md) §3
**GH Issues**: aprender#559, trueno#200, trueno#203

---

## 1. Overview

The stack provides compute across two dispatch layers:

**Layer 1 — trueno `Backend` enum (CPU + wgpu):**
Scalar, SSE2, AVX, AVX2, AVX-512, NEON, WasmSIMD, GPU (wgpu). Auto-
selects best available at runtime. User code is backend-agnostic.

**Layer 2 — GPU kernel dispatch (realizar + trueno-gpu):**
Custom CUDA PTX, cuBLAS GEMM, cuBLASLt FP8, wgpu WGSL shaders. Selected
per-operation within the GPU path. cuBLAS and PTX coexist — PTX for M=1
decode (bandwidth-bound), cuBLAS for M>1 prefill (compute-bound).

```rust
use trueno::{Vector, Matrix, Backend};

// Layer 1: CPU/wgpu dispatch via Backend enum
let result = Matrix::matmul(&a, &b); // Auto-selects AVX2/NEON/GPU
```

Each kernel has a **provable equivalence contract** (see
[provable-contracts.md](provable-contracts.md)) guaranteeing identical
numerical behavior across compute paths.

### Selection Priority

```
Layer 2 GPU (if CUDA available + parity gate passes):
  Custom PTX (M=1 decode) | cuBLAS (M>1 prefill) | cuBLASLt (FP8)
Layer 2 GPU (if wgpu available):
  WGSL compute shaders (all vendors)
Layer 1 CPU:
  AVX-512 → AVX2 → AVX → SSE2 → NEON → WasmSIMD → Scalar
```

### Per-Kernel Equivalence Contract

Every kernel has a contract with `equivalence` proof obligations:
```yaml
proof_obligations:
  - type: equivalence
    property: "SIMD matches scalar"
    formal: "max_ulp_error(simd(x), scalar(x)) <= 2"
  - type: equivalence
    property: "wgpu matches CPU"
    formal: "cosine(wgpu(x), cpu(x)) >= 0.98"
  - type: equivalence
    property: "PTX matches CPU"
    formal: "cosine(ptx(x), cpu(x)) >= 0.98"
```

This is how the sm_121 JIT bug (GH-559) was caught — the parity contract
failed, the system routed to wgpu automatically.

---

## 2. SIMD Backend (CPU)

**Always available.** The default backend on all platforms.

### 2.1 Implementation

- Hand-written SIMD intrinsics: AVX2+FMA, AVX-512, SSE2, NEON
- BLIS-style microkernels for matmul (avx2.rs, neon.rs)
- Hand-optimized hot paths for quantized tensor ops
- Rayon parallelism for matrix operations

### 2.2 Quantized Kernels

| Kernel | Description |
|--------|-------------|
| `fused_q4k_parallel_matvec` | Q4K dequant + matrix-vector multiply |
| `fused_q6k_parallel_matvec` | Q6K dequant + matrix-vector multiply |
| `fused_q8k_matvec` | Q8K dequant + matrix-vector multiply |
| `generic_fused_gate_up_matvec_into` | Fused SwiGLU gate+up (halves rayon dispatches) |

### 2.3 Performance

For 7B Q4K single-token decode: ~3 tok/s (memory-bandwidth bound on CPU).
Sufficient for testing, development, and CPU-only deployment.

---

## 3. wgpu Backend (Cross-Platform GPU)

**Targets**: AMD, Intel, NVIDIA, Apple Silicon — any GPU with Vulkan 1.2+,
Metal, or DX12 support.

### 3.1 How It Works

wgpu is a Rust implementation of the WebGPU specification. It compiles WGSL
(WebGPU Shading Language) compute shaders to native GPU code at runtime:

```
WGSL shader source
    │
  wgpu runtime
    │
    ├── Vulkan (Linux, Windows) ── NVIDIA, AMD, Intel
    ├── Metal (macOS, iOS) ────── Apple Silicon, older AMD
    ├── DX12 (Windows) ────────── NVIDIA, AMD, Intel
    └── WebGPU (browser) ─────── Any browser with WebGPU
```

One WGSL shader runs everywhere. No vendor-specific code.

### 3.2 Why wgpu Is Not "Cross-Platform CUDA"

| Aspect | CUDA | wgpu/Vulkan |
|--------|------|-------------|
| Scope | Full compute platform (cuBLAS, cuDNN, NCCL) | GPU access API |
| Vendor | NVIDIA only | All vendors |
| Ecosystem | Mature (15+ years) | Younger for compute |
| Warp shuffle | Native (`__shfl_sync`) | Subgroup ops (variable size) |
| Shared memory | Explicit, configurable | `var<workgroup>` (driver-managed) |
| Tensor cores | Full (WMMA PTX) | Limited (`VK_KHR_cooperative_matrix`) |
| Profiling | Nsight, nvprof | RenderDoc, limited compute profiling |

wgpu gives us **portability**. CUDA gives us **peak NVIDIA performance**.
We use both.

### 3.3 Implemented WGSL Shaders

**trueno** (39 WGSL shaders in `backends/gpu/shaders/`):
- basic_ops.rs: 19 shaders (matmul 16x16, CUTLASS-style 64x64 GEMM,
  add, sub, mul, div, dot, scale, clamp, etc.)
- reductions.rs: 5 shaders (max, sum, softmax exp, workgroup barrier)
- advanced.rs: 6 shaders (advanced operations)
- backward.rs: 9 shaders (backward pass for autograd)

**realizar** (inference-specific WGSL):

| Kernel | PMAT | Status |
|--------|------|--------|
| RMSNorm | PMAT-336 | Done |
| Q4K dequant+GEMV | PMAT-363 | Done |
| Bias add | PMAT-356 | Done |
| RoPE | PMAT-358 | Done |
| Attention | PMAT-361 | Done |
| LM Head | PMAT-347 | Done |
| SwiGLU/SiLU | PMAT-346 | Done (overflow fixed) |
| KV Cache | PMAT-344 | Partial |
| End-to-end forward | — | Not yet wired |

### 3.4 Compute Shader Limitations

**Variable subgroup size.** NVIDIA=32, AMD=64, Intel=variable. Reduction
algorithms must handle any subgroup size.

**No explicit shared memory banking.** `var<workgroup>` exists but the
driver controls allocation. Sufficient for RMSNorm reductions and tiled GEMV.

**No tensor core access (practical).** `VK_KHR_cooperative_matrix` exists
but adoption is limited. Irrelevant for M=1 decode (bandwidth-bound, not
compute-bound). Matters at M≥4 prefill.

### 3.5 Performance

For 7B Q4K decode on GB10: ~30 tok/s (80% memory bandwidth efficiency).
~83% of custom CUDA PTX performance, but runs on any GPU vendor.

---

## 4. CUDA PTX Backend (Custom Kernels)

**Targets**: NVIDIA GPUs (sm_50+). Highest performance for quantized ops.

### 4.1 Implementation

Custom PTX kernels written in-house for fused quantized operations:

- Fused Q4K dequantize + GEMV (reads quantized data directly)
- RMSNorm with in-register reduction
- RoPE with fused position encoding
- SwiGLU with fused gate + activation
- Attention with fused softmax

PTX is NVIDIA's virtual ISA — forward-compatible across GPU generations.
The driver JIT-compiles PTX → native SASS at load time.

### 4.2 The sm_121 JIT Bug (GH-559)

On Blackwell (sm_121), NVIDIA's driver JIT produces numerically incorrect
SASS from valid PTX:

| Evidence | Value |
|----------|-------|
| GPU/CPU logit cosine similarity | -0.005 (uncorrelated) |
| PyTorch on same GPU (uses nvcc) | 1.000 (perfect) |

**Root cause**: NVIDIA's driver contains three compilation pipelines:

```
1. nvcc (offline)     → Full compiler, used by PyTorch/cuBLAS     ✓
2. NVRTC (runtime)    → Same backend as nvcc, library API          ✓
3. Driver JIT         → Lightweight compiler, used by our PTX     ✗ on sm_121
```

### 4.3 NVRTC Fix

Replace driver JIT with NVRTC for sm_120+ GPUs:

```
Before (broken on Blackwell):
  PTX → cuModuleLoadData → driver JIT → wrong SASS

After (fixed):
  PTX → nvrtcCompileProgram(--gpu-architecture=sm_121) → cubin → correct SASS
```

```rust
pub fn from_ptx(ctx: &CudaContext, ptx: &str) -> Result<Self, GpuError> {
    let (major, _) = ctx.compute_capability()?;
    if major >= 12 {
        Self::from_ptx_nvrtc(ctx, ptx)  // Bypass buggy JIT
    } else {
        Self::from_ptx_jit(ctx, ptx)    // Pre-Blackwell: JIT works
    }
}
```

### 4.4 Performance

For 7B Q4K decode: ~36 tok/s (95% memory bandwidth efficiency).
3.3x faster than cuBLAS FP16 because it reads quantized data directly
(0.5625 B/elem vs 2.0 B/elem). Per Ivanov et al. (2021), autoregressive
LLM decode is memory-bandwidth bound, not compute bound.

---

## 5. cuBLAS Accelerator (NVIDIA GEMM)

**Not a separate backend** — an accelerator within the GPU path. Lives in
`trueno-gpu/src/driver/{cublas.rs, cublaslt.rs}`. Used alongside custom
PTX kernels within the same GPU inference pass.

### 5.1 Purpose

cuBLAS provides pre-built, hand-tuned GEMM (matrix multiply) kernels that
no handwritten shader can match. Used when:

- Batched prefill (M > 1, compute-bound, tensor cores matter)
- Full training backward pass (GEMM-dominated)
- FP16/BF16 operations where quantized kernels don't apply

### 5.2 Implementation

Hand-written FFI bindings (not bindgen) in trueno-gpu:
- `cublas_sys.rs`: `cublasCreate_v2`, `cublasGemmEx`, data types
- `cublas.rs`: Safe RAII wrapper with `gemm_f16()`, `gemm_f32()`
- `cublaslt_sys.rs` + `cublaslt.rs`: cuBLASLt for FP8 E4M3 GEMM
- FP8 path: Q4K → dequant → FP8 E4M3 → cuBLASLt GEMM → FP16 → FP32
- Plan caching for (m, n, k) shapes (PMAT-086)

### 5.3 Integration Points

| Crate | Use |
|-------|-----|
| realizar | Prefill GEMM, FP8 E4M3 GEMM, FP16 fallback |
| entrenar | Training backward pass, cuBLAS tensor cores (ALB-075) |

### 5.3 Relationship to Custom PTX

They coexist. For M=1 decode, custom Q4K PTX beats cuBLAS (bandwidth wins).
For M>1 prefill, cuBLAS wins (compute wins, tensor cores engaged).

```
Single-token decode:  Custom Q4K PTX  (bandwidth-bound, less data read)
Batched prefill:      cuBLAS FP16     (compute-bound, tensor cores)
Training backward:    cuBLAS          (GEMM-dominated)
```

### 5.4 Performance

7B model prefill (91 tokens): 314ms with cuBLAS batched (8.2x over serial).
RTX 4090 decode: 25ms/token (40 tok/s).

---

## 6. WASM Backend (Browser)

**Target**: `wasm32-unknown-unknown` for browser deployment.

- Scalar fallback when no SIMD available
- WebGPU via wgpu when browser supports it
- Streaming model loading for large models
- APR sharding for >2GB models in browser

---

## 7. decy: CUDA Kernel Harvesting Pipeline

**Crate**: `decy` (standalone, not a trueno dependency)

### 7.1 Problem

The CUDA ecosystem has thousands of battle-tested open-source kernels
(FlashAttention, CUTLASS tile schedulers, fused attention variants).
Rewriting them from scratch in WGSL or PTX is error-prone and wasteful.

### 7.2 Solution

decy transpiles C/CUDA → safe Rust with minimal `unsafe`:

```
Open-source CUDA C++ (e.g., FlashAttention)
        │
   decy transpile
        │
   Safe Rust + inline PTX / GPU intrinsics
   (<5 unsafe blocks per 1000 LOC)
        │
   trueno backend kernel (pure Rust, no C++ toolchain)
```

### 7.3 Pipeline Stages

| Stage | Crate | Purpose |
|-------|-------|---------|
| Parse | decy-parser | C/CUDA AST extraction |
| Lower | decy-hir | HIR (High-level IR) construction |
| Analyze | decy-analyzer | Type analysis, control flow |
| Ownership | decy-ownership | Pointer → borrow/lifetime inference |
| Verify | decy-verify | Safety verification |
| Codegen | decy-codegen | Rust source emission |

### 7.4 Key Benefit

Eliminates the C++ build toolchain dependency. trueno ships pure Rust
kernels that were originally proven in the CUDA ecosystem — best of both
worlds: CUDA's battle-tested algorithms + Rust's safety guarantees.

---

## 8. Backend Dispatch and Parity Gate

### 8.1 Runtime Dispatch

At model load time, the parity gate validates correctness by comparing a
one-token forward pass between each GPU backend and CPU reference:

```
Load Model → CPU Forward (reference, always correct)
                │
    ┌───────────┼───────────┐
    │           │           │
CUDA Forward  wgpu Forward  cuBLAS Forward
    │           │           │
cosine ≥ 0.98? cosine?     cosine?
    │           │           │
    └── select best passing backend ──┘
```

### 8.2 Performance Budget (7B Q4K Decode)

| Backend | Bandwidth Efficiency | tok/s | vs cuBLAS FP16 |
|---------|---------------------|-------|----------------|
| CUDA Q4K PTX | 95% | ~36 | 3.3x faster |
| wgpu Q4K WGSL | 80% | ~30 | 2.7x faster |
| cuBLAS FP16 | 100% (but 3.5x data) | ~11 | baseline |
| CPU SIMD | N/A | ~3 | 0.3x |

Key insight: for M=1 decode, arithmetic intensity is below the roofline
knee — performance is determined by memory bandwidth, not FLOPs (Ivanov
et al. 2021). Reading less data (Q4K) beats faster compute (tensor cores).

### 8.3 Parity Contract

```yaml
# contracts/gpu-parity-v2.yaml
equations:
  parity: "cosine(gpu_logits, cpu_logits) >= 0.98"

proof_obligations:
  - "exists b in {wgpu, cuda, nvrtc}: cosine(b, cpu) >= 0.98"
  - "select(model, device) is deterministic"
```

---

## 9. References

1. Ivanov et al. (2021) "Data Movement Is All You Need." MLSys.
2. Dettmers et al. (2022) "GPTQ: Accurate Post-Training Quantization."
3. NVIDIA PTX ISA v8.5 — Forward compatibility specification.
4. Xu et al. (2024) "Efficient Parallel Reductions on GPUs using Subgroup Ops."
5. Chatterjee et al. (2025) "ProofWright: Agentic Formal Verification of
   CUDA." arXiv:2511.12294. — Provable memory/thread/functional correctness
   for CUDA kernels.
6. Arora et al. (2025) "TensorRight: Automated Verification of Tensor Graph
   Rewrites." arXiv:2511.17838. — SMT-based verification of tensor layout
   transformations.
7. Gond et al. (2026) "LLM-42: Enabling Determinism in LLM Inference."
   arXiv:2601.17768. — Verified rollback for GPU floating-point non-determinism.
8. Zhou et al. (2025) "Linear Layouts: Robust Code Generation Using F2."
   arXiv:2505.23819. — Algebraic formalization of tensor layouts.
9. Qiu et al. (2024) "Tenspiler: A Verified Lifting-Based Compiler for
   Tensor Operations." arXiv:2404.18249. — Verified multi-backend dispatch.
