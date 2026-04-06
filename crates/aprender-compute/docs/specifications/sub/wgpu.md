# Sub-spec: wgpu Backend

**Parent:** [trueno-spec.md](../trueno-spec.md) Section 7

---

## 1. Overview

Cross-platform GPU compute via Vulkan/Metal/DX12/WebGPU. Enables GPU acceleration on AMD, Intel Arc, and Apple Silicon without CUDA.

## 2. Inference Path

**Key types:**
- `GpuMatmulCache` — persistent weight buffers + cached pipeline + GEMV dispatch
- `WgslForwardPass` — multi-pass single-submit transformer layer

**Shaders** (`src/backends/gpu/shaders/basic_ops.rs`):
- `GEMV_SHADER` — cooperative K-reduction, vec4 loads, 256 threads/workgroup
- `MATMUL_SHADER` — tiled 16×16 shared-memory GEMM (M>1 prefill)

**Dispatch rule:** M=1 → GEMV shader, M>1 → GEMM shader.

**Performance** (Radeon Pro W5700X, Qwen2.5-Coder-1.5B):
- 27.6 tok/s decode (81% of CPU SIMD)
- 1.29ms/layer (28 layers = 36ms full forward)
- Peak 90.6 GFLOPS

## 3. Training Path

9 shaders in `src/backends/gpu/shaders/backward.rs`:

| Shader | Purpose |
|--------|---------|
| `silu_backward` | SiLU activation gradient |
| `gemm_backward_a` | Weight gradient (dL/dA) |
| `gemm_backward_b` | Input gradient (dL/dB) |
| `rmsnorm_backward` | RMSNorm gradient |
| `rope_backward` | Rotary position embedding gradient |
| `cross_entropy_backward` | Cross-entropy loss gradient |
| `cross_entropy_forward` | Cross-entropy forward (used in training loop) |
| `adamw_step` | AdamW optimizer update |
| `nf4_dequant` | NF4 4-bit dequantization (QLoRA) |

Dispatch functions: `src/backends/gpu/device/backward.rs`

All 8 FALSIFY tests pass on AMD Radeon Pro W5700X via Vulkan (FALSIFY-WGPU-TRAIN-001 through 008). Numerical tolerance < 1e-5 vs CPU reference.

## 4. Provable Contracts

- **Inference:** `wgpu-forward-pass-v1.yaml`
  - `rmsnorm_correctness`: 4.77e-7 max error vs CPU
  - `gemv_dispatch`: M=1→GEMV, M>1→GEMM
  - `vec4_alignment`: K%4==0 enforced by shader
- **Training:** `wgpu-training-v1.yaml` (8 FALSIFY tests)

## 5. GPU Threshold

Only dispatch to GPU for >100K elements. PCIe/bus transfer costs ~0.5ms — below this threshold, SIMD is faster.

```rust
const GPU_MIN_SIZE: usize = 100_000;
fn should_use_gpu(size: usize) -> bool {
    size >= GPU_MIN_SIZE && gpu_available()
}
```
