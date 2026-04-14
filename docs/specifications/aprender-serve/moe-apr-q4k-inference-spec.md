# SPEC-MOE-APR: MoE Inference for APR Q4K Models

Version: 2.0
Status: in-progress
Date: 2026-04-13

**Document ID:** SPEC-MOE-APR-001
**Version:** 2.0.0
**Status:** IN PROGRESS (correctness PASS, performance FAIL — 100x gap vs llama.cpp)
**Author:** PAIML Engineering
**Date:** 2026-04-11
**Priority:** P0 — Blocks PMAT-521 (ALB-010 teacher loading for SHIP-TWO Model 2)
**Parent:** SPEC-SHIP-TWO-001
**Contract:** `contracts/aprender/moe-apr-q4k-inference-v1.yaml`
**Target Model:** Qwen3-Coder-30B-A3B-Instruct (128 experts, top-8, 48 MoE layers)
**Citations:**
- [C1] Shazeer et al. (2017) "Sparsely-Gated Mixture-of-Experts" arXiv:1701.06538
- [C2] Fedus et al. (2022) "Switch Transformers" arXiv:2101.03961
- [C3] Jiang et al. (2024) "Mixtral of Experts" arXiv:2401.04088
- [C4] Dai et al. (2024) "DeepSeek-V2" arXiv:2405.04434
- [C5] Cai et al. (2024) "A Survey on MoE" arXiv:2407.06204

---

## 1. Abstract

APR Q4K MoE inference is not implemented. The GGUF loader hardcodes
`moe_experts: None` for every block. The APR Q4K loading path tries dense
tensor names (`model.layers.N.mlp.up_proj.weight`) and fails on MoE models
that have per-expert tensors (`model.layers.N.mlp.experts.E.up_proj.weight`).

This blocks ALB-010 (teacher model for SHIP-TWO Model 2). The Qwen3-Coder-30B
APR file has 18,867 tensors (128 experts × 3 projections × 48 layers + globals)
but inference fails immediately with "tensor not found."

---

## 2. Five Whys

1. Why does `apr run qwen3-coder-30b-q4k.apr` fail? → "tensor not found: model.layers.0.mlp.up_proj.weight"
2. Why is that tensor missing? → MoE models have `mlp.experts.E.up_proj.weight`, not `mlp.up_proj.weight`
3. Why doesn't the loader try expert names? → APR Q4K loading path only knows dense tensor layout
4. Why no MoE loading? → SafeTensors MoE loading exists (line 721 in `safetensors_infer_convert.rs`) but APR Q4K path was never wired to it
5. Why was it never wired? → ALB-010 steps 1-5 built MoE dispatch and SafeTensors loading. Steps 6-8 (APR loading) were blocked waiting for this spec

---

## 3. Architecture: Qwen3-Coder-30B-A3B

| Parameter | Value |
|-----------|-------|
| hidden_size | 2048 |
| num_hidden_layers | 48 |
| num_attention_heads | 32 |
| num_key_value_heads | 4 |
| head_dim | 128 |
| moe_intermediate_size | 768 |
| num_experts | 128 |
| num_experts_per_tok | 8 |
| decoder_sparse_step | 1 (all layers MoE) |
| vocab_size | 151936 |
| rope_theta | 10,000,000 |
| norm_topk_prob | **false** (HF config default — was incorrectly set to true) |
| shared_expert | **NO** (unlike Qwen3.5-35B) |
| Total params | 30.5B |
| Active params/token | 3.3B |

---

## 4. Three Tensor Layouts

| Format | Router | Expert Gate | Expert Up | Expert Down |
|--------|--------|-------------|-----------|-------------|
| **GGUF** | `blk.N.ffn_gate_inp.weight` [128,2048] | `blk.N.ffn_gate_exps.weight` [128,768,2048] | `blk.N.ffn_up_exps.weight` [128,768,2048] | `blk.N.ffn_down_exps.weight` [128,2048,768] |
| **SafeTensors** | `model.layers.N.mlp.gate.weight` | `model.layers.N.mlp.experts.E.gate_proj.weight` [768,2048] | `...experts.E.up_proj.weight` [768,2048] | `...experts.E.down_proj.weight` [2048,768] |
| **APR** (ours) | Same as SafeTensors (imported from HF) | Same as SafeTensors | Same as SafeTensors | Same as SafeTensors |

---

## 5. Existing Code vs Gaps

| Component | Status | Location |
|-----------|--------|----------|
| F32 MoE dispatch (softmax + top-k + SwiGLU) | **DONE** | `moe_dispatch.rs` |
| SafeTensors MoE loading (Layout 1+2) | **DONE** | `safetensors_infer_convert.rs:721` |
| APR Q4K MoE loading | **MISSING** | `loader_apr_quantized.rs` (dense only) |
| MoE config from APR metadata | **MISSING** | `config.rs:from_apr()` |
| Q4K quantized expert dispatch | **MISSING** | Only F32 dispatch exists |
| MoE-aware forward block | **PARTIAL** | SafeTensors path has it, APR path doesn't |
| `norm_topk_prob` flag | **MISSING** | Not in MoeExpertWeights |

---

## 6. Implementation Plan (Updated v2.0)

### Phase 1: MoE metadata extraction — **DONE** (2026-04-11)

- APR path: infer from tensor names + shape
- GGUF path: read from `{arch}.expert_count` / `{arch}.expert_feed_forward_length` metadata
- Config fields: `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`

### Phase 2: MoE tensor loading — **DONE** (2026-04-13)

- APR path: per-expert tensor names (`model.layers.N.mlp.experts.E.*`)
- GGUF path: packed 3D tensors (`blk.N.ffn_gate_exps.weight` [ne0,ne1,ne2])
- Unpack 3D → per-expert slices via stride-based byte offsets

### Phase 3: Q4K MoE dispatch — **DONE** (2026-04-11)

- `moe_forward_q4k`: softmax router → top-k → per-expert SwiGLU → weighted sum
- `moe_expert_swiglu`: gate+up split → SiLU(gate)*up → down

### Phase 4: Forward block wiring — **DONE** (2026-04-13)

- MoE detection via `layer.moe_gate_weight.is_some()`
- Skip dense FFN down_proj for MoE layers (MoE output already includes down)
- `qwen3moe` arch constraint: has_qk_norm=true, eps=1e-6

### Phase 5: Stride-based MoE dispatch — **TODO** (P0, 4h)

**Eliminate per-expert data copy.** Currently each expert's Q4K data is copied
into a separate `Vec<u8>`. Instead, pass the original mmap'd 3D tensor data
with an offset parameter to the Q4K matmul:

- Add `fused_q4k_parallel_matvec_at_offset(data, input, in_dim, out_dim, byte_offset)`
- Expert e's offset = `e * (total_bytes / num_experts)`
- Eliminates 113 MB of copies per layer (128 experts × 884KB each)
- Expected speedup: 5-10x (memory bandwidth was the bottleneck)

### Phase 6: Fused multi-expert kernel — **TODO** (P1, 8h)

**Process all top-k experts in a single rayon dispatch.** Currently 8 separate
rayon dispatches per layer (384 per token). Fuse into one:

- Allocate output buffers for all 8 experts upfront
- Single `par_iter` over expert_indices with closure that does SwiGLU per expert
- Reduces rayon dispatch overhead from 384 to 48 per token
- Expected speedup: 2-3x on top of Phase 5

### Phase 7: CUDA MoE kernel — **BLOCKED** (P1, driver issue)

**GPU-native expert dispatch implemented but blocked by GB10 driver:**

- CUDA MoE infrastructure built: PackedMoeRef, indexed weights, expert GEMV dispatch
- `cuda_moe_ffn`: router on CPU, expert matmuls via `q4k_gemv_indexed_async` with stride offsets
- **Blocked:** `cuMemHostRegister` fails on GB10 aarch64 for large mmap'd tensors (113MB packed experts)
- `cuMemcpyHtoD` also fails with CUDA_ERROR_INVALID_VALUE after registration attempt
- **Dense models work on GPU** (32B Qwen2.5-Coder confirmed at ~0.3 tok/s GPU)
- Root cause: Blackwell unified memory driver limitation for large registered buffers
- **Workaround:** Use llama.cpp as teacher server (92 tok/s confirmed)
- **Future fix:** trueno-gpu direct pointer passing (no GpuBuffer abstraction), or CUDA driver update

### Phase 8: Port candle fused MoE WMMA kernel — **TODO** (P0, 8h)

**Five Whys (revised 2026-04-14):**
1. Why 1.76 tok/s? → 1,152 separate CPU matmuls per token
2. Why separate matmuls? → Our Q4K GEMV assumes standalone 2D tensor, can't handle 3D packed stride
3. Why can't we fix the stride? → Wrong approach. The kernel must handle expert routing INSIDE
4. Why don't we have a fused MoE kernel? → Haven't ported candle's pattern yet
5. Where is the reference? → `candle/candle-kernels/src/moe/moe_wmma_gguf.cu` — PROVEN, working code

**Prior art (from our own stack):**
- Dense model parity: **1.01x llama.cpp** at c=4 (PMAT-105, qwen-coder-deploy)
- Candle fused MoE kernel: WMMA tensor cores, Q4K/Q6K dequant, sorted token routing
- llama.cpp `mul_mat_vec_q_moe`: stride_channel_x into packed 3D, one kernel all experts

**Implementation: Port candle `moe_wmma_gguf.cu` to trueno-gpu:**

1. Copy `candle-kernels/src/moe/moe_wmma_gguf.cu` + `moe_utils.cuh` + `gguf.cuh`
2. Add to trueno-gpu as `MoeGemmKernel` (alongside existing `GemmKernel`)
3. Wire into `CudaExecutor::fused_moe_gemv(packed_3d_ptr, input, expert_ids, weights, ...)`
4. Dispatch from `cuda_moe_ffn`: router on CPU → expert_ids to GPU → one kernel for all experts
5. Remove per-expert GEMV loop

**Expected result:** ONE kernel launch per MoE layer (vs 24 now) = 50-90 tok/s target

---

## 7. Performance Gap Analysis (v2.0, 2026-04-13)

### 7.1 Measured Results

| Engine | Model | Hardware | tok/s | Notes |
|--------|-------|----------|-------|-------|
| **llama.cpp** | Qwen3-Coder-30B-A3B Q4_K_M | Blackwell GB10, GPU | **88.9** | `ggml_mul_mat_id` native 3D dispatch |
| **apr serve** | Same model, same hardware | Same, CPU fallback | **0.9** | Per-expert Q4K matmul, no batching |
| **Gap** | | | **100x** | **UNACCEPTABLE — must fix** |

### 7.2 Five Whys (Performance)

1. Why 0.9 tok/s? → Each token requires 8 expert forward passes × 48 layers = 384 matmuls
2. Why are 384 matmuls slow? → Each expert matmul is a separate `fused_q4k_parallel_matvec` call with memory allocation + rayon dispatch overhead
3. Why separate matmuls? → We unpack 3D packed tensor into per-expert Vec<u8> copies, then matmul each independently
4. Why copy? → `ggml_mul_mat_id` indexes into the 3D tensor with stride-based access (zero-copy). We allocate per-expert buffers.
5. Why no stride-based access? → Our Q4K matmul only supports 2D [out_dim, in_dim] tensors. Need to extend to support expert_id indexing into 3D packed data.

### 7.3 Root Cause: Architectural Mismatch

llama.cpp uses `ggml_mul_mat_id`:
- **Zero-copy**: indexes directly into 3D packed tensor via `nb[2]` stride
- **GPU-accelerated**: single kernel launch for all 8 active experts
- **Batched**: processes multiple expert matmuls in one kernel (CUDA `mmvq` kernel with expert ID)

Our implementation:
- **Copies per-expert data** into separate `OwnedQuantizedTensor` (128 × 884KB = 113MB per layer, ×48 layers)
- **Sequential expert matmuls** via 8 separate `fused_q4k_parallel_matvec` calls per layer
- **No GPU dispatch** for MoE (WGPU bypass triggers CPU fallback)
- **Rayon overhead** per matmul call (thread pool dispatch × 384 per token)

### 7.4 Performance Fix Plan

**Phase 5: Stride-based MoE dispatch (P0, estimated 4h)**

Replace per-expert copy+matmul with stride-based access into the 3D packed tensor:

```rust
// CURRENT (slow): copy per-expert data, then matmul
let expert_data = packed_3d[e * expert_bytes..(e+1) * expert_bytes].to_vec();
let result = fused_q4k_parallel_matvec(&expert_data, input, in_dim, out_dim);

// TARGET (fast): pass offset + stride, no copy
let result = fused_q4k_parallel_matvec_strided(
    &packed_3d_data,        // entire 3D tensor (mmap, zero-copy)
    input,
    in_dim,
    out_dim,
    expert_offset,          // e * nb[2]
);
```

**Phase 6: Fused multi-expert kernel (P1, estimated 8h)**

Process all 8 active experts in a single rayon dispatch:

```rust
// Process all top-k experts in one parallel pass
let expert_outputs = fused_moe_q4k_topk(
    &packed_gate_data,      // 3D gate tensor
    &packed_up_data,        // 3D up tensor
    &packed_down_data,      // 3D down tensor
    input,
    &selected_experts,      // [(expert_idx, weight); top_k]
    hidden_dim,
    moe_intermediate,
);
```

**Phase 7: CUDA MoE kernel (P1, estimated 16h)**

Port llama.cpp's `mmvq` kernel pattern:
- Single CUDA kernel launch for all 8 experts
- Expert ID passed as parameter, not separate launches
- Uses `ggml_mul_mat_id` equivalent in trueno-gpu

### 7.5 Performance Targets

| Phase | Expected tok/s | Gap to llama.cpp | Blocking? |
|-------|---------------|-------------------|-----------|
| Current (v2.0) | 0.9 | 100x | YES |
| Phase 5 (stride) | ~10-15 | ~6-9x | Unblocks teacher generation |
| Phase 6 (fused) | ~25-40 | ~2-4x | Acceptable for production |
| Phase 7 (CUDA) | ~80-100 | ~1x | Parity |

## 8. Acceptance Criteria (Updated v2.0)

| ID | Criterion | Threshold | Status |
|----|-----------|-----------|--------|
| AC-MOE-001 | MoE metadata extracted | num_experts=128, top_k=8 | **PASS** |
| AC-MOE-002 | All expert tensors loaded (GGUF packed 3D) | 128 experts × 48 layers | **PASS** |
| AC-MOE-003 | Router weight loaded per layer | 48 router tensors [128,2048] F32 | **PASS** |
| AC-MOE-004 | `apr serve` produces coherent Python on MoE | FALSIFY-MOE-006 | **PASS** (2026-04-13) |
| AC-MOE-005 | No OOM during loading | Peak RSS < model size × 2 | **PASS** |
| AC-MOE-006 | Throughput ≥ 10 tok/s (Phase 5 target) | `apr serve` + probar benchmark | **FAIL** (0.9 tok/s) |
| AC-MOE-007 | Throughput within 4x of llama.cpp (Phase 6) | probar A/B benchmark | PENDING |
| AC-MOE-008 | Throughput parity with llama.cpp (Phase 7) | probar A/B benchmark | PENDING |

---

## 9. Falsification Tests (Updated v2.0)

| ID | Hypothesis Falsified If... | Status | Mitigation |
|----|---------------------------|--------|------------|
| FALSIFY-MOE-001 | Expert tensor count wrong | **PASS** | Fixed: GGUF 3D packed loading |
| FALSIFY-MOE-002 | Router softmax produces NaN | **PASS** | Max-subtract softmax |
| FALSIFY-MOE-003 | Top-8 selection wrong | **PASS** | Sort-based with bounds check |
| FALSIFY-MOE-004 | Expert weights all zero | **PASS** | Verified via MOE-TRACE |
| FALSIFY-MOE-005 | OOM during loading | **PASS** | mmap + per-expert slicing |
| FALSIFY-MOE-006 | Output is garbage | **PASS** (2026-04-13) | Root cause: double down_proj + missing arch constraint |
| FALSIFY-MOE-007 | GGUF 3D packed format fails | **PASS** | dims.reverse() + stride-based slicing |
| FALSIFY-MOE-008 | Throughput < 10 tok/s | **FAIL** (0.9 tok/s) | Phase 5: stride-based dispatch (no copy) |
| FALSIFY-MOE-009 | Throughput < 25% of llama.cpp | **FAIL** (1%) | Phase 6+7: fused kernel + CUDA |

## 10. Bugs Found and Fixed (v2.0)

| Bug | Root Cause | Fix | Found Via |
|-----|-----------|-----|-----------|
| "tensor not found" on GGUF MoE | No GGUF 3D packed tensor loading | Added `unpack_moe_experts_gate_up/down` | FALSIFY-MOE-007 |
| num_experts=2048 (wrong) | `dims.reverse()` in parser; used dims[2] instead of dims[0] | Fixed index after reading `parse_tensor_info` in reading.rs | pmat query (reference A) |
| Garbage output (_accel repeating) | Double down_proj: `moe_expert_swiglu` applies down, then caller applies dummy down again | Skip down_proj for MoE layers in forward loop | Layer tracing (reference B) |
| Arch constraints not matching | GGUF says "qwen3moe", fallback matches "qwen3_moe" (underscore), codegen YAML missing entirely | Added "qwen3moe" to arch-constraints-v1.yaml | pmat query + HF transformers (reference D) |
| norm_topk_prob wrong | Hardcoded `true` when experts > 0; HF config says `false` for Qwen3-Coder-30B | Needs fix: read from GGUF metadata or default false | HF config (reference D) |

---

## 9. Files to Modify

| File | Change |
|------|--------|
| `crates/aprender-serve/src/gguf/config.rs` | Extract MoE config from APR metadata |
| `crates/aprender-serve/src/gpu/adapters/apr_q4k.rs` | MoE tensor loading + expert packing |
| `crates/aprender-serve/src/gpu/scheduler/moe_dispatch.rs` | Add Q4K dispatch variant |
| `crates/aprender-serve/src/gpu/scheduler/types.rs` | Add `norm_topk_prob` to `MoeExpertWeights` |
| `contracts/aprender/moe-apr-q4k-inference-v1.yaml` | New provable contract |

---

## 10. References

| Reference | Location |
|-----------|----------|
| Existing MoE dispatch | `crates/aprender-serve/src/gpu/scheduler/moe_dispatch.rs` |
| SafeTensors MoE loading | `crates/aprender-serve/src/safetensors_infer_convert.rs:721` |
| MoeExpertWeights struct | `crates/aprender-serve/src/gpu/scheduler/types.rs:117` |
| APR Q4K tensor loading | `crates/aprender-serve/src/gguf/loader_apr_quantized.rs` |
| APR config extraction | `crates/aprender-serve/src/gguf/config.rs:359` |
| MoE config from tensor inference | `crates/aprender-serve/src/gpu/adapters/apr_q4k.rs:482` |
| Qwen3 MoE family contract | `contracts/model-families/qwen3.yaml` |
| SHIP-TWO parent spec | `docs/specifications/aprender-train/ship-two-models-spec.md` |
| candle FusedMoeGGUF | `candle-transformers/src/fused_moe.rs` |
| llama.cpp tensor mapping | `gguf-py/gguf/tensor_mapping.py` |

---

*End of specification SPEC-MOE-APR-001.*
