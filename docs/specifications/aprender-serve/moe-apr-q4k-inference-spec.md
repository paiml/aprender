# SPEC-MOE-APR: MoE Inference for APR Q4K Models

Version: 1.0
Status: proposed
Date: 2026-04-11

**Document ID:** SPEC-MOE-APR-001
**Version:** 1.0.0
**Status:** PROPOSED
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
| norm_topk_prob | true |
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

## 6. Implementation Plan

### Phase 1: APR MoE metadata extraction (1h)

In `crates/aprender-serve/src/gguf/config.rs`, extend `from_apr()` to read:
- `num_experts` from APR custom metadata
- `num_experts_per_tok` from APR custom metadata
- `moe_intermediate_size` from APR custom metadata
- Fallback: infer from tensor names (count `experts.N` patterns in layer 0)

### Phase 2: APR Q4K MoE tensor loading (3h)

In the APR Q4K loading path, detect MoE model (num_experts > 0) and:
- Load router: `model.layers.N.mlp.gate.weight`
- For each expert E in 0..num_experts:
  - Load `model.layers.N.mlp.experts.E.gate_proj.weight` (Q4K)
  - Load `model.layers.N.mlp.experts.E.up_proj.weight` (Q4K)
  - Load `model.layers.N.mlp.experts.E.down_proj.weight` (Q4K)
- Pack into `MoeExpertWeights` fused format
- Skip dense FFN loading when MoE detected

### Phase 3: Q4K MoE dispatch (2h)

Add `moe_forward_q4k()` that:
- Runs softmax router on F32 (dequant router weight once)
- Selects top-8 experts per token
- For each selected expert: `fused_q4k_parallel_matvec` on quantized expert weights
- SwiGLU activation: `SiLU(gate) * up`, then down projection
- Weighted sum of expert outputs
- Renormalize weights if `norm_topk_prob`

### Phase 4: Wire into forward block (1h)

In the APR forward path, check `layer.moe_gate_weight.is_some()`:
- If MoE: call `moe_forward_q4k()` instead of dense FFN
- If dense: existing path unchanged

---

## 7. Acceptance Criteria

| ID | Criterion | Threshold | Measurement |
|----|-----------|-----------|-------------|
| AC-MOE-001 | APR MoE metadata extracted | num_experts=128, top_k=8 | Unit test |
| AC-MOE-002 | All 128×48 expert tensors loaded | 18,432 expert tensors (128×3×48) | Count assertion |
| AC-MOE-003 | Router weight loaded per layer | 48 router tensors [128,2048] | Shape assertion |
| AC-MOE-004 | `apr run` produces non-garbage on MoE model | Coherent Python on "def fibonacci" | Manual + oracle |
| AC-MOE-005 | Inference completes without OOM | Peak RSS < 24 GB (Q4K model is 17 GB) | Memory check |
| AC-MOE-006 | Throughput > 0.5 tok/s (CPU Q4K, 30B MoE) | Measurable generation | Timer |

---

## 8. Falsification Tests

| ID | Hypothesis Falsified If... | Mitigation |
|----|---------------------------|------------|
| FALSIFY-MOE-001 | Expert tensor count != 128×3×48 | Fix expert enumeration loop |
| FALSIFY-MOE-002 | Router softmax produces NaN | Add numerically stable softmax with max-subtract |
| FALSIFY-MOE-003 | Top-8 selection returns wrong experts | Sort-based selection with tie-breaking |
| FALSIFY-MOE-004 | Expert weights are all zero after load | Verify Q4K dequant produces non-zero on sample |
| FALSIFY-MOE-005 | OOM during 128-expert loading | Load experts lazily or use mmap |
| FALSIFY-MOE-006 | Output is garbage (repetitive/nonsense) | Trace layer-by-layer, compare with SafeTensors path |

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
