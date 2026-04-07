# Sub-spec: CUDA Backend (trueno-gpu)

**Parent:** [trueno-spec.md](../trueno-spec.md) Sections 6, 15

---

## 1. Philosophy

Own the stack. trueno-gpu generates PTX from pure Rust — no nvcc, no LLVM, no external toolchains. PTX is a text format; generating it is string manipulation.

## 2. Key APIs

| Type | Purpose |
|------|---------|
| `PtxModule` | Top-level PTX module (.version, .target, globals) |
| `PtxKernel` | Single kernel entry point |
| `KernelBuilder` | Fluent API for constructing kernels |
| `Kernel` trait | `emit_ptx() -> String` — the core contract |

## 3. Available Kernels

| Kernel | Variants | Contract |
|--------|----------|----------|
| `GemmKernel` | naive, tiled, tensor_core | `blas-level3-v1.yaml` |
| `SoftmaxKernel` | warp shuffle reduction | `softmax-kernel-v1.yaml` |
| `LayerNormKernel` | with gamma/beta | — |
| `AttentionKernel` | standard, causal | — |
| `QuantizeKernel` | Q4_K fused with matmul | — |
| 6 backward kernels | activations, cross_entropy, gemm, layer_norm, rms_norm, softmax | `dimension-independent-kernels-v1.yaml` |

## 4. Testing Without GPU

All `emit_ptx()` functions are pure Rust string generators:

```bash
# No GPU needed — tests validate PTX text output
cargo test -p trueno-gpu property_tests

# With GPU hardware
cargo test -p trueno-gpu --features cuda
cargo test -p trueno-gpu --test pixel_fkr --features "cuda gpu-pixels"
```

Check for `.version`, `.entry`, `.target` directives in emitted PTX.

## 5. Blackwell sm_121 JIT Bug (trueno#200)

**Problem:** `cuModuleLoadDataEx` fails with `CUDA_ERROR_UNKNOWN` on Blackwell GPUs when called during active GPU work. The NVIDIA JIT compiler crashes non-deterministically under load.

**What's affected:**
- Forward kernels: Work after pre-warming (load all variants before training)
- Backward kernels: Crash during training (compiled on-demand when GPU is active)
- Inference: NOT affected (uses cuBLAS/SIMD, no custom PTX at runtime)

**Workaround (`from_ptx_direct`):** Compile PTX to cubin at init time, before any GPU work. Load only pre-compiled cubin blobs during training.

## 6. Dimension-Independent Kernels (trueno#203)

**Current (broken):** Dimensions (M, K, N) baked into PTX source → 50+ kernel variants → JIT compilation for each new shape → Blackwell crash.

**Target:** Dimensions as runtime parameters → ~15 kernel types → each compiled once.

**Pre-compiled cubin pipeline (permanent fix):**

```
build.rs → nvcc (offline) → cubin for sm_80/sm_89/sm_121
→ include_bytes!() → zero JIT at runtime
```

Contract: `contracts/dimension-independent-kernels-v1.yaml` (6 FALSIFY tests, 20 BAKED→10 OK kernel analysis).

## 7. Register Allocation

trueno-gpu includes register allocation with liveness tracking (`src/ptx/registers/`). Registers are allocated per-kernel, freed when no longer live. This minimizes register pressure and avoids spilling to local memory.

## 8. Memory Management

`DeviceBuffer` and `HostBuffer` with a memory pool (`src/memory/pool.rs`) that tracks fragmentation. GPU memory is a scarce resource — the pool avoids allocation/deallocation churn during inference loops.
