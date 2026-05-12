# SHIP-007 layer-0 oracle bisection — APR Q4K vs HF FP16 element-wise diff

**Date**: 2026-05-03
**Host**: noah-Lambda-Vector (lambda-labs)
**APR side**: `apr trace --save-tensor` on `/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr` (Q4K, RTX 4090)
**HF side**: `Qwen/Qwen2.5-Coder-7B-Instruct` FP16 forward pass on CPU (HF cache hit, no download)
**Tokens**: `[3838, 374, 220, 17, 10, 17, 30]` ("What is 2+2?", 7 tokens, identical on both sides — confirms tokenizer parity)
**Tool**: `/mnt/nvme-raid0/targets/aprender/release/apr diff --values <apr>.bin <hf>.bin` (PR #1413 APRT diff)

## Result

| Stage | RMS diff | Cosine sim | Notes |
|---|---|---|---|
| embedding | 4.67e-10 | **1.0000000000** | bit-identical (token table is unquantized) |
| attn_norm | 7.12e-5 | **0.9999999483** | within Q4K noise floor — RMSNorm correct |
| **attn_out** | 1.46e-2 | **0.9966403287** | ← **FIRST SIGNIFICANT DROP** |
| ffn_norm | 1.60e-2 | 0.9959173286 | (carries attn_out drift) |
| ffn_gate | 3.16e-2 | 0.9996581102 | downstream artifact |
| ffn_up | 2.04e-2 | 0.9963500750 | downstream artifact |
| ffn_silu | 1.10e-2 | 0.9979571225 | downstream artifact |
| ffn_swigl | 4.94e-3 | 0.9984939085 | downstream artifact |
| ffn_out | 2.05e-2 | 0.9980404691 | downstream artifact |
| post_ffn_residual | 2.44e-2 | 0.9981704939 | downstream artifact |
| final_norm | 3.03e-1 | 0.9932669898 | (whole-model — accumulates 28 layers) |
| lm_head | 2.37e-1 | 0.9969170161 | (whole-model — last-token logits) |

## Interpretation

1. **Embedding is correct.** Cos=1.000000 means APR's token-embedding lookup matches HF byte-for-byte (modulo a 4e-10 RMS round-off). The input pipeline is fine.

2. **AttnNorm is correct.** Cos=0.99999995 is well within Q4K noise (typical floor ~1e-4 RMS). The pre-attention RMSNorm matches HF.

3. **AttnOut is WRONG.** Cos=0.9966 is way above noise floor. APR's post-O-proj attention output diverges from HF's by ~3.4e-3 cosine — substantial.

4. **All downstream stages carry the drift.** Cosines never recover after attn_out — the divergence cascades through FFN and beyond. This is consistent with a single layer-0 attention bug feeding into all later computations.

## Bug location (narrowed to attention block)

The bug is **inside the layer-0 attention block** between RMSNorm output (cos=0.99999995, correct) and the post-O-proj output (cos=0.9966, wrong).

Possible bug sites (in order of forward execution):
1. `qkv_matmul` — Q4K-fused matmul of normed input × QKV weights. **APR captures this stage but HF side does not (deferred per script doc).** Adding HF-side qkv_matmul capture is the single most impactful follow-up.
2. `qkv_bias` — bias add post-matmul.
3. RoPE on Q and K (per-head rotation).
4. Q@Kᵀ scaled-dot-product (with attention scale 1/√head_dim).
5. Softmax over scores (causal mask applied).
6. softmax_scores @ V (weighted sum).
7. O-projection (Q4K matmul of attention output × O-proj weights).

## Next milestone

**Extend the HF reference script to capture qkv_matmul, qkv_bias, attention** (the 3 stages currently deferred). Re-run the diff. The first stage at which APR-vs-HF cosine drops will pinpoint the bug to a single matmul or kernel within the attention block.

If qkv_matmul drops below 0.999, the bug is in Q4K matmul kernel.
If qkv_matmul is fine (≥0.999) but attention is below 0.999, the bug is in RoPE or softmax.
If attention is fine but attn_out drops, the bug is in O-projection.

## Reproducer

```bash
# APR side (live):
cargo build -p apr-cli --release --features inference,cuda
/mnt/nvme-raid0/targets/aprender/release/apr trace \
    /mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr \
    --save-tensor "all" \
    --save-tensor-dir /tmp/save-tensor-step3-smoke

# HF side (one-shot):
uv run --with torch --with transformers --with accelerate --with safetensors \
    python scripts/generate_qwen25_coder_fp16_stages.py \
    --output /tmp/qwen25-coder-7b-hf-fp16-stages --device cpu

# Diff each shared stage:
APR=/tmp/save-tensor-step3-smoke
HF=/tmp/qwen25-coder-7b-hf-fp16-stages
for stage in embedding attn_norm attn_out ffn_norm ffn_gate ffn_up ffn_silu ffn_swigl ffn_out post_ffn_residual; do
    apr diff --values "$APR/layer-0/$stage.bin" "$HF/layer-0/$stage.bin" --limit 1 \
        | grep -E "max_abs_diff|cosine|RMS"
done
```

## Implications for ship %

- **MODEL-1 SHIP-007 bug location: PINPOINTED to attention block.** Previously the bug was hypothesized to be at apr_transformer matmul Q4K dispatch (§28); this empirical bisection narrows it further to specifically the attention forward pass.
- The next code-side lever is **fixing the divergent matmul/kernel** — but first the HF reference needs the missing 3 stages to identify which specific kernel.
- Estimate: ~2-3 more iterations to (a) extend HF reference script, (b) re-bisect, (c) fix the kernel.
