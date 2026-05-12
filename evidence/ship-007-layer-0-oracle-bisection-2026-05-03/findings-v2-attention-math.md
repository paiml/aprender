# SHIP-007 layer-0 oracle bisection v2 — bug narrowed to attention math

**Date**: 2026-05-03 (refinement of `findings.md`)
**Iteration**: v2 with HF-side qkv_matmul / qkv_bias / attention captures added.

## Refined cosine table

| Stage | Cosine sim | RMS diff | Δ from prev | Interpretation |
|---|---|---|---|---|
| embedding | 1.0000000000 | 4.67e-10 | (start) | bit-identical |
| attn_norm | 0.9999999483 | 7.12e-5 | -5e-8 | RMSNorm correct |
| **qkv_matmul** | **0.9996914319** | 2.30e-2 | -3.1e-4 | Q4K matmul noise (acceptable) |
| qkv_bias | 0.9999975262 | 2.30e-2 | +2.8e-4 | bias dampens relative noise |
| **attention** | **0.9985879806** | 9.46e-3 | **-1.4e-3** | **← BIG DROP — bug is here** |
| attn_out | 0.9966403287 | 1.46e-2 | -1.9e-3 | O-proj amplifies the upstream error |
| ffn_* (downstream) | 0.996-0.999 | 1e-2 to 3e-2 | (carries) | downstream artifacts |
| final_norm | 0.9932669898 | 3.03e-1 | (whole) | accumulates 28 layers |
| lm_head | 0.9969170161 | 2.37e-1 | (whole) | last-token logits |

## Bug location: between qkv_bias and attention

The cosine is healthy through `qkv_bias` (0.9999975, essentially perfect) and
drops by ~1.4e-3 at `attention`. That is the FIRST place where APR forward
introduces error materially above Q4K noise floor.

Operations between these two stages (per `inference.rs` and HF Qwen2Attention):
1. **RoPE** applied to Q and K per-head (`apply_rope_f32`)
2. **Attention scaling** `score *= 1/sqrt(head_dim)`
3. **Q@Kᵀ** scaled-dot-product (per-head, GQA-7:1 grouped)
4. **Causal mask** + **softmax** over scores
5. **softmax_scores @ V** (weighted sum over values)

The bug is in one of these 5 operations. O-projection (which takes attention
→ attn_out) has its own additional error (~1.9e-3 cosine drop) but that is a
SEPARATE quantitative concern — not the primary divergence cascade.

## Implications

- **The bug is NOT in QKV matmul** (cos=0.99969 is at the expected Q4K floor for a 3584×4608 matmul).
- **The bug is NOT in QKV bias add** (cos=0.99999, bit-equivalent within FP precision).
- **The bug IS in the attention math itself** — RoPE/scale/softmax/mask/V-weight chain.
- **O-proj has its own ADDITIONAL Q4K error** but that's secondary.

## Why the qkv_matmul/qkv_bias inversion?

`qkv_matmul` cos=0.99969 vs `qkv_bias` cos=0.9999975 — this is correct, not
a bug:

- `qkv_matmul` is the pre-bias matmul output (Q4K input × Q4K weight).
  Q4K-vs-FP16 matmul has ~3e-4 element-wise error in cosine; this matches.
- `qkv_bias` adds a deterministic FP16 bias vector to that output. Since the
  bias is bit-identical on both sides, adding it shifts the output distribution
  by the same amount. This shifts the post-bias cosine closer to 1 because the
  deterministic component (bias) now dominates the tensor's direction over
  the small noisy matmul component.

In other words: `qkv_bias_cos > qkv_matmul_cos` is a mathematical artifact of
adding a deterministic vector to a slightly-noisy one, not evidence of a bug.

## Next milestones

1. **Capture finer stages within attention math** to narrow further:
   - Post-RoPE Q
   - Post-RoPE K
   - Q@Kᵀ raw scores
   - Post-scale scores
   - Post-softmax probs
   - Post-V@ pre-concat (per-head)
   This requires deeper instrumentation of HF's monolithic self_attn.

2. **Compare APR's scratch_swiglu_ffn-style internal capture for attention**
   (PR #1167 introduced sub-FFN telemetry; analogous sub-attention telemetry
   would mirror this).

3. **Audit each attention sub-op against `../candle` and `../pytorch`**
   per `feedback_stack_research_repos`. Focus areas:
   - RoPE freq table (theta=1000000 for Qwen2.5)
   - GQA-7:1 head grouping (PMAT-FFN-FUSION lesson learned)
   - Softmax fp32 accumulator
   - Causal mask construction

## Tools
- APR side: `apr trace --save-tensor "all"` (PR #1421 SHIP-007 PR-C-real step 3)
- HF side: `scripts/generate_qwen25_coder_fp16_stages.py` v2 (this PR)
- Diff: `apr diff --values <apr>.bin <hf>.bin --limit 1` (PR #1413)

## Reproducer

```bash
# APR side
/mnt/nvme-raid0/targets/aprender/release/apr trace \
    /mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr \
    --save-tensor "all" --save-tensor-dir /tmp/save-tensor-step3-smoke

# HF side (v2 with qkv + attention)
uv run --with torch --with transformers --with accelerate --with safetensors \
    python scripts/generate_qwen25_coder_fp16_stages.py \
    --output /tmp/qwen25-coder-7b-hf-fp16-stages-v2 --device cpu

# Bisect attention block specifically
APR=/tmp/save-tensor-step3-smoke
HF=/tmp/qwen25-coder-7b-hf-fp16-stages-v2
for stage in qkv_matmul qkv_bias attention attn_out; do
    apr diff --values "$APR/layer-0/$stage.bin" "$HF/layer-0/$stage.bin" --limit 1 \
        | grep -E "max_abs_diff|cosine|RMS"
done
```
