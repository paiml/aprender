# SHIP-007 v3 — APR attention code audit vs HF Qwen2 reference

**Date**: 2026-05-03 (extension of `findings-v2-attention-math.md`)
**Goal**: Cross-reference APR's attention forward against HF Transformers
Qwen2 to identify the algebraic source of the 1.4e-3 cosine drop between
qkv_bias and attention.

## Audit table

| Sub-operation | APR location | HF Qwen2 reference | Status |
|---|---|---|---|
| RoPE rotation formula | `helpers.rs:334-338` (split-half: x[i]=x1·cos − x2·sin, x[i+½d]=x1·sin + x2·cos) | `apply_rotary_pos_emb` with `rotate_half(x) = (-x2, x1)` | **MATCH** ✓ |
| RoPE freq computation | `1.0 / theta^(2·i/d)` for i ∈ [0, d/2) | `inv_freq[k] = 1/theta^(2k/d)` for k ∈ [0, d/2) | **MATCH** ✓ |
| rope_theta value | 1000000.0 (from `metadata.rope_theta`) | 1000000.0 (Qwen2.5 config) | **MATCH** ✓ |
| Attention scale | `1.0 / sqrt(head_dim)` (head_dim=128, scale ≈ 0.0884) | `1/sqrt(head_dim)` | **MATCH** ✓ |
| Causal mask | `for j in 0..=i` (triangular, inclusive) | `attention_mask` triangular | **MATCH** ✓ (semantically) |
| Softmax | f32 max-subtract-then-exp, sum-norm | f32 max-subtract softmax | **MATCH** ✓ |
| QKV bias | applied post-matmul, pre-RoPE (`add_bias` before `apply_rope_f32`) | `q_proj`/`k_proj`/`v_proj` Linear(bias=True) — bias inside the Linear | **MATCH** ✓ (same logical position) |
| Q indexing per head | `q_start = i*hidden_dim + h*head_dim` | reshape to (b,h,s,d) | **MATCH** ✓ (logical equivalence) |
| K indexing per kv_head | `k_start = j*kv_dim + kv_head*head_dim` with `kv_head = head/group_size` | GQA with num_kv_heads, head sharing | **MATCH** ✓ (GQA-7:1) |
| V weighted sum | `attn_out[d] += p · v_all[v_start + d]` | `attn_output = matmul(attn_probs, v)` | **MATCH** ✓ |

## Where the 1.4e-3 cosine drop CANNOT come from

Each item in the table above was algebraically verified vs HF. The bug
is NOT in any of:
- RoPE rotation direction (would produce cos ≈ 0)
- RoPE base/theta (would produce cos < 0.5 for sequences > 1)
- Attention scale (would shift softmax distribution but rarely 1.4e-3)
- Causal mask structure (would produce wildly different attention scores)
- Softmax numerical precision (max-subtract makes this stable)

## Remaining hypotheses for the 1.4e-3 cosine drop

1. **Numerical accumulation ordering**. HF uses torch.matmul which dispatches
   to BLAS with vectorized parallel summation. APR uses nested scalar
   for-loops with sequential accumulation. The order-of-summation difference
   in 128-element dot products can produce ~1-2 ULPs per dot, integrated
   across the attention pattern this could compound to ~1e-3 cosine drift.
   **Plausible but feels like the upper end** for this magnitude.

2. **F32-dequant vs FP16 weights**. APR's `forward_traced` uses F32
   dequantized Q4K weights (line 38 comment: "Q4K layers not used in
   traced forward (uses F32 for accuracy)"). HF uses original FP16 weights.
   The Q4K → F32 dequant is lossy (~1e-3 RMS per element). When these
   slightly-off Q values are dotted against slightly-off K values (also
   from dequant Q4K), the product compounds the error. **This is a
   structural systematic error, not a randomly-distributed one.**

3. **Subtle FP precision in the V@scores accumulation**. APR accumulates
   `attn_out[d] += p · v[d]` in F32. For a sequence of 7 tokens, this
   sums up to 7 weighted values per output element. Summation order is
   forward (j=0,1,2,...,i). HF does the same logical sum but via BLAS
   which may reorder.

4. **Some Qwen2-specific detail not yet identified.** Qwen2 doesn't have
   QK-norm (that's Qwen3+). Doesn't have sliding-window attention by default.
   Standard MHA with GQA-7:1.

## Conclusion

**No algebraic bug found in APR's attention.** The 1.4e-3 cosine drop is
most likely a combination of:
(a) Q4K dequant precision loss in the QKV matmul outputs (already accounted
    for at qkv_matmul=0.99969, qkv_bias=0.9999975), AND
(b) compounding of those errors through the Q@Kᵀ scaled-dot-product against
    similarly-imprecise K values.

The cos=0.9986 is consistent with **systematic precision loss from Q4K dequant
through the attention math**, NOT with a structural algorithmic bug.

## Implication for SHIP-007 fix

If the 1.4e-3 drop is precision-loss artifact (not bug), then the actual
SHIP-007 root cause may be **further downstream** than layer-0 attention.
Worth checking:
- Whether the same diff at layer-1, layer-13 (mid-network), layer-27 (last layer)
  shows accumulating drift or stable noise floor
- Whether the lm_head logits cosine of 0.9969 is consistent with an
  argmax-flip threshold (i.e., is the drift concentrated at a few critical
  logit indices, or spread across all 152064?)
- The `apr run` quality issue may NOT be a forward-pass bug but a
  sampling/decoding bug (temperature, top-k, top-p configuration mismatch).

## Next narrowing steps

1. **Multi-layer cosine sweep**: capture stages at layers 0, 1, 13, 27 on
   both APR and HF sides; observe whether cosine drift grows monotonically
   (precision-loss accumulation) or has a discontinuity (structural bug
   triggered by a specific layer's data distribution).

2. **Logit argmax check**: top-5 logit comparison on both sides. If APR's
   top token ID matches HF's top-1 token ID, the drift is "noise" (won't
   affect quality much). If they disagree, the drift is bug-relevant.

3. **Repeat attention with F32 weights end-to-end**: convert canonical 7B
   teacher to FP16 safetensors (already exists: 15GB at
   `/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.safetensors`),
   run `apr trace --save-tensor` on it, compare against HF FP16. If cos
   improves to >0.999 across all stages, confirms (b) above and the bug
   is just Q4K precision.

## Repro

```bash
# v2 bisection (already run)
bash scripts/run_v2_bisection.sh  # produces refined cosine table

# Suggested next: multi-layer cosine sweep
APR=/tmp/save-tensor-step3-smoke
HF=/tmp/qwen25-coder-7b-hf-fp16-stages-v2
for layer in 0; do
    for stage in attn_norm qkv_matmul qkv_bias attention attn_out; do
        cos=$(apr diff --values "$APR/layer-$layer/$stage.bin" "$HF/layer-$layer/$stage.bin" --limit 1 \
            | grep "cosine sim" | head -1 | awk '{print $4}')
        echo "L$layer/$stage: $cos"
    done
done
```
