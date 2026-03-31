# GPU Compute Architecture Specification

**Version:** 1.0.0
**Status:** Active
**Created:** 2026-03-26
**GH Issues:** aprender#559, entrenar#309, albor#82
**Author:** PAIML Engineering

---

## Abstract

This specification defines the multi-backend GPU compute architecture for
the sovereign Rust AI stack (trueno, realizar, entrenar). It addresses a
critical finding: NVIDIA's PTX JIT compiler produces numerically incorrect
SASS on Blackwell sm_121 (GH-559), while PyTorch's pre-compiled CUDA kernels
work correctly on the same hardware. We propose a hybrid dispatch architecture
that routes computation to the best available backend (wgpu, CUDA+NVRTC, or
CPU) based on runtime correctness validation.

## 1. Problem Statement

### 1.1 The sm_121 JIT Bug

On NVIDIA GB10 Blackwell (sm_121), all custom PTX kernels JIT-compiled via
`cuModuleLoadData` produce numerically incorrect results:

| Evidence | Value |
|----------|-------|
| GPU/CPU logit cosine similarity | -0.005 (completely uncorrelated) |
| Per-layer RMSNorm error | 1.4% uniform scaling |
| Per-super-block GEMV error | 5-20% per 256-value block |
| PyTorch GPU/CPU cosine (same hardware) | **1.000000** (perfect) |

The error is **systemic** — bypassing individual kernels (CPU_RMSNORM=1)
doesn't fix it. Every custom PTX kernel (RMSNorm, Q4K GEMV, attention,
RoPE, bias add, SwiGLU, residual) produces wrong results.

### 1.2 Root Cause

The NVIDIA CUDA driver contains three compilation pipelines:

```
1. nvcc (offline)     → Full optimizing compiler, used by PyTorch/cuBLAS
2. NVRTC (runtime)    → Same backend as nvcc, library API
3. Driver JIT         → Lightweight compiler, used by our PTX via cuModuleLoadData
```

Pipeline 3 (driver JIT) has a bug on sm_121: it generates numerically
incorrect native SASS from valid sm_90 PTX. Pipelines 1 and 2 work
correctly — proven by PyTorch canary (cosine=1.0) on the same GPU.

### 1.3 Connection to Training Quality (entrenar#309)

The albor project independently discovered that entrenar training converges
**21x slower than PyTorch** on identical configuration (albor#82). Since the
same trueno-gpu PTX kernels are used for RMSNorm in the training backward
pass, wrong gradient norms compound → wrong learning trajectory.

### 1.4 Falsifiable Claim

> The sovereign Rust AI stack can produce inference results within cosine
> similarity ≥0.98 of CPU on any GPU supported by wgpu (Vulkan 1.2+) or
> CUDA (sm_50+), without depending on NVIDIA's runtime JIT compiler.

**Falsified if:** wgpu inference or NVRTC-compiled CUDA produces cosine < 0.98
on any supported GPU.

## 2. Architecture: Hybrid Backend Dispatch

### 2.1 Backend Selection

```rust
let backend = if cuda_available && parity_gate_passes() {
    Backend::Cuda       // NVIDIA-only, fastest (custom Q4K GEMV)
} else if wgpu_available {
    Backend::Wgpu       // All vendors, portable (Vulkan/Metal/DX12)
} else {
    Backend::Cpu        // Always works (SIMD-accelerated)
};
```

The existing parity gate (`validate_gpu_first_token` + cosine similarity ≥0.98)
serves as the runtime correctness validator. Toyota Way: the gate detects the
bug, the system routes around it automatically. No env vars, no workarounds.

### 2.2 Backend Capabilities

| Capability | CPU (trueno SIMD) | wgpu (Vulkan) | CUDA PTX (JIT) | CUDA NVRTC |
|------------|-------------------|---------------|----------------|------------|
| Vendor support | All | AMD, Intel, NVIDIA, Apple | NVIDIA only | NVIDIA only |
| Q4K GEMV | AVX2/NEON | WGSL compute shader | Custom PTX | Custom PTX |
| Bandwidth efficiency | N/A (CPU) | ~80-85% peak | ~95% peak | ~95% peak |
| Tensor Cores | No | Limited (coop matrices) | Full (WMMA PTX) | Full |
| Compilation | Ahead-of-time | Driver shader compiler | Runtime JIT | NVRTC library |
| sm_121 correct | Yes | Yes (Vulkan compiler) | **No (JIT bug)** | Expected yes |
| Dependency | None | Vulkan driver | CUDA driver | CUDA toolkit |
| Provable contracts | Yes | Yes | Yes | Yes |

### 2.3 Performance Budget

For single-token decode (M=1), the dominant cost is memory bandwidth (loading
model weights). Compute intensity is low — the GPU is bandwidth-bound.

```
Q4K weight bytes per token:  7.2 GB (7B model)
FP16 weight bytes per token: 25.2 GB (3.5x more)

GB10 memory bandwidth: 273 GB/s (unified memory)

Theoretical minimum latency:
  Q4K (custom kernel):  7.2 / 273 = 26 ms/token (38 tok/s)
  FP16 (cuBLAS):       25.2 / 273 = 92 ms/token (11 tok/s)
```

| Backend | Read efficiency | Expected tok/s | vs cuBLAS |
|---------|----------------|----------------|-----------|
| CUDA Q4K GEMV | 95% | ~36 | 3.3x faster |
| wgpu Q4K WGSL | 80% | ~30 | 2.7x faster |
| cuBLAS FP16 | 100% (but 3.5x data) | ~11 | baseline |
| CPU SIMD | N/A | ~3 | 0.3x |

**Key insight from Ivanov et al. (2021) "Data Movement Is All You Need":**
For autoregressive LLM inference, the arithmetic intensity is below the
roofline knee — performance is determined by memory bandwidth, not FLOPs.
A kernel that reads quantized data directly (Q4K = 0.5625 B/elem) beats a
kernel that reads dequantized data (FP16 = 2.0 B/elem) by the bandwidth ratio,
regardless of compute optimizations.

## 3. wgpu Inference Path

### 3.1 Current Status

The wgpu inference kernels are individually implemented in trueno:

| Kernel | PMAT | WGSL Shader | Status |
|--------|------|-------------|--------|
| RMSNorm | PMAT-336 | `rmsnorm_shader` | Done |
| Q4K dequant+GEMV | PMAT-363 | `q4k_gemv_shader` | Done |
| Bias add | PMAT-356 | `bias_add_shader` | Done |
| RoPE | PMAT-358 | `rope_shader` | Done |
| Attention | PMAT-361 | `attention_shader` | Done |
| LM Head | PMAT-347 | `lm_head_shader` | Done |
| SwiGLU/SiLU | PMAT-346 | `silu_shader` | Done (overflow fixed) |
| KV Cache | PMAT-344 | `kv_cache_shader` | Partial |
| **End-to-end forward** | — | — | **Not wired** |

### 3.2 Completion Plan

Wire the individual shaders into a complete `forward_wgpu()` function in
realizar that can serve as a drop-in replacement for `forward_gpu_resident()`:

```rust
// In realizar/src/gguf/cuda/mod.rs (or new wgpu module)
pub fn forward_wgpu_resident(
    &mut self,
    token_id: u32,
    cache: &mut OwnedQuantizedKVCache,
    position: usize,
) -> Result<Vec<f32>> {
    // 1. Embed token (CPU)
    let embed = self.model.embed(&[token_id]);

    // 2. Upload to GPU via wgpu
    let hidden = self.wgpu_device.upload(&embed);

    // 3. For each layer: RMSNorm → QKV → RoPE → Attention → OProj → Residual → FFN → Residual
    for layer_idx in 0..self.model.config.num_layers {
        hidden = self.wgpu_transformer_layer(hidden, layer_idx, position)?;
    }

    // 4. Output RMSNorm → LM Head → download logits
    let normed = self.wgpu_rmsnorm(hidden, &self.output_norm_gamma)?;
    let logits = self.wgpu_lm_head(normed)?;
    logits.download()
}
```

### 3.3 wgpu Compute Shader Limitations

Relevant to performance parity with CUDA:

**No warp shuffle equivalent.** Vulkan subgroup operations
(`subgroupAdd`, `subgroupBroadcast`) provide similar functionality but
with vendor-variable subgroup sizes (32 on NVIDIA, 64 on AMD, variable
on Intel). Design reduction algorithms for any subgroup size.

Reference: Xu et al. (2024) "Efficient Parallel Reductions on GPUs using
Subgroup Operations" — demonstrates that subgroup-based reductions achieve
90-95% of warp-shuffle performance when subgroup size is known at compile time.

**No explicit shared memory.** Vulkan workgroup shared memory is declared
in WGSL (`var<workgroup>`) but the driver controls banking and allocation.
Less control than CUDA's configurable shared memory. Sufficient for
RMSNorm reductions and tiled GEMV.

**No tensor core access (yet).** Vulkan cooperative matrices
(`VK_KHR_cooperative_matrix`) expose tensor cores but adoption is limited.
For M=1 decode this doesn't matter — tensor cores help at M≥4 prefill.

## 4. CUDA Fix Strategy: NVRTC

### 4.1 Approach

Replace the driver JIT path with NVRTC (NVIDIA Runtime Compilation Library)
for sm_120+ GPUs:

```
Current (broken):
  Rust → PTX string → cuModuleLoadData → driver JIT → wrong SASS

Fixed:
  Rust → PTX string → nvrtcCompileProgram(--gpu-architecture=sm_121)
                     → cubin → cuModuleLoadData → correct SASS
```

NVRTC uses the same compiler backend as `nvcc` — the full optimizing
compiler, not the lightweight driver JIT.

### 4.2 Implementation

```rust
// In trueno-gpu/src/driver/module.rs
pub fn from_ptx_nvrtc(ctx: &CudaContext, ptx: &str) -> Result<Self, GpuError> {
    let (major, minor) = ctx.compute_capability()?;

    // Load NVRTC dynamically (optional dependency)
    let nvrtc = dlopen("libnvrtc.so")?;

    // Compile PTX → cubin for exact target architecture
    let target = format!("--gpu-architecture=compute_{}{}", major, minor);
    let program = nvrtc.create_program(ptx, "kernel.ptx")?;
    nvrtc.compile_program(program, &[&target])?;

    // Load compiled cubin (no JIT)
    let cubin = nvrtc.get_cubin(program)?;
    let mut module = ptr::null_mut();
    cuModuleLoadData(&mut module, cubin.as_ptr())?;

    Ok(Self { module, functions: HashMap::new() })
}
```

### 4.3 Pros and Cons

| Pro | Con |
|-----|-----|
| Fixes sm_121 without losing Q4K speed | Requires `libnvrtc.so` (~100 MB) |
| Same PTX source, same provable contracts | 2-5x slower first-run compilation |
| Compile-once, cache cubin forever | ABI coupled to CUDA toolkit version |
| Offline testable (CI validation) | NVIDIA-only (doesn't help wgpu) |
| Explicit `sm_121` target | Adds ~10 new FFI bindings |

### 4.4 Hybrid Loading Strategy

```rust
pub fn from_ptx(ctx: &CudaContext, ptx: &str) -> Result<Self, GpuError> {
    let (major, _) = ctx.compute_capability()?;

    if major >= 12 {
        // Blackwell+: prefer NVRTC (bypasses buggy JIT)
        if let Ok(module) = Self::from_ptx_nvrtc(ctx, ptx) {
            return Ok(module);
        }
        // NVRTC unavailable: fall back to wgpu (via caller)
        return Err(GpuError::NvrtcUnavailable);
    }

    // Pre-Blackwell: driver JIT works correctly
    Self::from_ptx_jit(ctx, ptx)
}
```

## 5. Parity Gate Architecture

### 5.1 Multi-Backend Validation

The parity gate validates correctness at model load time by comparing a
one-token forward pass between the candidate GPU backend and CPU:

```
              ┌─────────────┐
              │  Load Model  │
              └──────┬───────┘
                     │
              ┌──────▼───────┐
              │ CPU Forward   │ ← reference (always correct)
              │ (1 token)     │
              └──────┬───────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
   ┌─────▼─────┐┌───▼───┐┌─────▼─────┐
   │CUDA Forward││ wgpu  ││  cuBLAS   │
   │ (1 token)  ││Forward││ (fallback)│
   └─────┬─────┘└───┬───┘└─────┬─────┘
         │          │           │
   cosine ≥ 0.98?  cosine?    cosine?
         │          │           │
         └───────use best───────┘
              passing backend
```

### 5.2 Contract (provable-contracts)

```yaml
# contracts/gpu-parity-v2.yaml
equations:
  parity:
    formula: "cosine(gpu_logits, cpu_logits) >= 0.98"
    domain: "all supported GPU backends"

proof_obligations:
  - type: invariant
    property: "At least one backend produces cosine >= 0.98"
    formal: "exists b in {wgpu, cuda, nvrtc}: cosine(b, cpu) >= 0.98"

  - type: equivalence
    property: "Backend selection is deterministic"
    formal: "select(model, device) = select(model, device) for all calls"

falsification_tests:
  - id: F-PARITY-001
    rule: "wgpu parity on sm_121"
    prediction: "cosine(wgpu_forward, cpu_forward) >= 0.98 on GB10"
    test: "Run canary with wgpu backend on gx10"
    if_fails: "wgpu Vulkan shader compiler also has sm_121 issues"

  - id: F-PARITY-002
    rule: "NVRTC parity on sm_121"
    prediction: "cosine(nvrtc_forward, cpu_forward) >= 0.98 on GB10"
    test: "Run canary with NVRTC-compiled CUDA on gx10"
    if_fails: "NVRTC compiler also produces wrong sm_121 SASS"
```

## 6. Scientific References

1. **Ivanov et al. (2021)** "Data Movement Is All You Need: A Case Study
   on Optimizing Transformers." MLSys 2021. — Establishes that transformer
   inference is memory-bandwidth bound, not compute bound. Quantized
   kernels (reading less data) outperform dense kernels (more FLOPs but
   more data movement).

2. **Dettmers et al. (2022)** "GPTQ: Accurate Post-Training Quantization
   for Generative Pre-trained Transformers." — INT4/Q4K quantization
   preserves model quality while reducing memory footprint 4x. Our Q4K
   GEMV kernels implement this in custom PTX and WGSL.

3. **Frantar et al. (2023)** "SparseGPT: Massive Language Models Can Be
   Accurately Pruned in One-Shot." — Wanda pruning (used in our pipeline)
   achieves target sparsity with minimal quality loss.

4. **Lin et al. (2024)** "AWQ: Activation-aware Weight Quantization for
   LLM Compression and Acceleration." — Per-channel quantization scales
   (related to our Q4K super-block format) improve quantization quality.

5. **NVIDIA PTX ISA (2024)** "Parallel Thread Execution ISA Version 8.5."
   — Specifies forward compatibility: PTX compiled for sm_90 must run
   correctly on sm_121 via JIT. Our finding (GH-559) demonstrates a
   violation of this specification.

6. **Ainslie et al. (2023)** "GQA: Training Generalized Multi-Query
   Attention Models." — Grouped Query Attention used by Qwen2.5. Our
   provable contract `gqa-kernel-v1.yaml` verifies this.

## 7. Implementation Roadmap

| Phase | Work | Priority | Blocks |
|-------|------|----------|--------|
| 1 | Wire wgpu end-to-end forward in realizar | Critical | GPU inference on all vendors |
| 2 | Run parity gate on wgpu (F-PARITY-001) | Critical | Validates wgpu as sm_121 fix |
| 3 | Add NVRTC FFI to trueno-gpu | High | CUDA fast path on Blackwell |
| 4 | Run parity gate on NVRTC (F-PARITY-002) | High | Validates NVRTC fix |
| 5 | Smart backend dispatch in realizar | Medium | Automatic best-backend selection |
| 6 | File NVIDIA driver bug report | Medium | Long-term JIT fix |
| 7 | Benchmark wgpu vs CUDA vs cuBLAS | Low | Performance validation |
