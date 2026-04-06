# Known Architecture Edge Cases

Reference for non-standard `config.json` field patterns that require special
handling during contract mapping.

## Falcon (7B, 40B, 180B)

**Non-standard fields:**
- `multi_query: true` — Falcon-7B uses multi-query attention (1 KV head).
  Map to `gqa-kernel-v1` with n_kv_heads=1.
- `parallel_attn: true` — Attention and FFN computed in parallel (not
  sequential). This is an optimization pattern but does not change which
  contracts are needed; note it in the config summary.
- `alibi: true` — Some Falcon variants use ALiBi instead of RoPE. When
  present, require `alibi-kernel-v1` and do NOT require `rope-kernel-v1`.
- `num_kv_heads` field (note: NOT `num_key_value_heads`) — Falcon uses
  a different field name. Check both `num_key_value_heads` and `num_kv_heads`.
- `new_decoder_architecture: true` — Falcon-40B/180B flag indicating the
  updated architecture. Does not change contract requirements.

**Normalization**: Falcon uses `layer_norm_epsilon` (not `rms_norm_eps`),
so it maps to `layernorm-kernel-v1`.

## Phi / Phi-2

**Non-standard fields:**
- `partial_rotary_factor: 0.5` — Only half the head dimension uses RoPE.
  The remaining dimensions use learned absolute position embeddings.
  Requires BOTH `rope-kernel-v1` AND `absolute-position-v1`.
- `qk_layernorm: true` — QK normalization before attention dot product.
  Requires `qk-norm-v1`.
- `rotary_emb_base` — Phi uses this instead of `rope_theta`. Treat
  the same as `rope_theta` for contract mapping purposes.

**Activation**: Phi-2 uses `"gelu_new"` → `gelu-kernel-v1`.

**Normalization**: Phi-2 uses `layer_norm_eps` → `layernorm-kernel-v1`.

## Phi-3 / Phi-3.5

**Differences from Phi-2:**
- Uses `rms_norm_eps` → `rmsnorm-kernel-v1` (not LayerNorm)
- Uses standard `rope_theta` field
- May have `rope_scaling` for long-context variants → `rope-extrapolation-v1`
- `hidden_act = "silu"` → `silu-kernel-v1` + `swiglu-kernel-v1`

## Gemma2

**Non-standard fields:**
- `attn_logit_softcapping: 50.0` — Soft caps attention logits before softmax.
  The `attention-scaling-v1` contract (universal) should note this variant.
  The scaling becomes: `logits = softcap * tanh(logits / softcap)` applied
  after the standard `1/sqrt(d_k)` scaling.
- `final_logit_softcapping: 30.0` — Caps final output logits. Relevant to
  `inference-pipeline-v1`.
- `sliding_window: 4096` on alternating layers — Not all layers use sliding
  window. Layers alternate between full attention and sliding window.
  Still requires `sliding-window-attention-v1`.
- `query_pre_attn_scalar` — Custom scaling factor replacing `1/sqrt(d_k)`.
  Note in `attention-scaling-v1`.

## DeepSeek-V2 / DeepSeek-V3

**Non-standard fields:**
- `q_lora_rank` / `kv_lora_rank` — Multi-head Latent Attention (MLA) uses
  low-rank projections for KV compression. This is architecturally different
  from standard GQA. No dedicated MLA contract exists yet; flag as a gap.
- `qk_nope_head_dim` / `qk_rope_head_dim` — Split head dimensions between
  non-positional and RoPE dimensions. Requires `rope-kernel-v1`.
- `v_head_dim` — Value head dimension different from query head dim.
- `n_routed_experts` / `n_shared_experts` — MoE routing. No MoE contract yet.

**Contract gaps to flag**: MLA (multi-head latent attention), MoE routing.

## Qwen3.5 (Hybrid GDN)

**Non-standard fields:**
- Hybrid architecture with both standard attention layers and Gated Delta Net
  (GDN) recurrence layers.
- Look for `model_type` containing "gated_delta_net" or fields like
  `gdn_chunk_size`, `gdn_expansion_ratio`.
- Requires: `gated-delta-net-v1` + `hybrid-layer-dispatch-v1` + `conv1d-kernel-v1`
  in addition to all standard transformer contracts.
- The `hybrid-layer-dispatch-v1` contract governs which layer indices use
  attention vs GDN recurrence.

**Existing contracts**: `qwen35-shapes-v1`, `qwen35-e2e-verification-v1`,
`qwen35-hybrid-forward-v1` already exist in the repo.

## Mamba / Mamba2

**Non-standard fields:**
- `d_state` — SSM state dimension
- `d_conv` — Convolution width in Mamba block
- `expand` — Expansion factor for inner dimension
- `ssm_cfg` — SSM configuration dict (Mamba-1)
- `model_type = "mamba"` or `"mamba2"`

**Contract mapping**: Requires `ssm-kernel-v1`. Does NOT require standard
attention contracts (attention-kernel, kv-cache-*, softmax-kernel) since
Mamba is attention-free. Adjust the universal set accordingly:
- Keep: `matmul-kernel`, `linear-projection`, `embedding-lookup`,
  `embedding-algebra`, `tensor-shape-flow`, `cross-entropy-kernel`,
  `inference-pipeline`, `model-config-algebra`
- Drop: `attention-kernel`, `attention-scaling`, `softmax-kernel`,
  `kv-cache-sizing`, `kv-cache-equivalence`

## Jamba (Hybrid SSM + Attention)

**Non-standard fields:**
- Combines Mamba SSM layers with standard attention layers
- `attn_layer_period` / `attn_layer_offset` — Which layers use attention
- Has both SSM fields (`d_state`, `d_conv`) and attention fields

**Contract mapping**: Requires the full universal attention set PLUS
`ssm-kernel-v1` PLUS `hybrid-layer-dispatch-v1`. Effectively a superset.

## GPT-2 / GPT-Neo / GPT-J

**Legacy fields:**
- Uses `n_embd` instead of `hidden_size`
- Uses `n_head` instead of `num_attention_heads`
- Uses `n_layer` instead of `num_hidden_layers`
- `activation_function` instead of `hidden_act`

**Normalization**: `layer_norm_epsilon` → `layernorm-kernel-v1`

**Position**: Learned absolute position embeddings (no RoPE). Requires
`absolute-position-v1`. Does NOT require `rope-kernel-v1`.

## StarCoder / StarCoder2

**Non-standard fields:**
- `multi_query: true` (StarCoder-1) → `gqa-kernel-v1` with n_kv=1
- StarCoder2 uses `num_key_value_heads` normally
- `sliding_window` present in StarCoder2 → `sliding-window-attention-v1`

## Non-LLM Architectures (informational)

These architectures will trigger the non-LLM warning:

| model_type | Architecture | Notes |
|------------|-------------|-------|
| `vit` | Vision Transformer | No causal mask, no KV cache |
| `clip` | CLIP | Vision + text encoder (not decoder) |
| `whisper` | Whisper | Encoder-decoder, not causal-only |
| `wav2vec2` | Wav2Vec2 | Audio encoder |
| `bert` | BERT | Bidirectional encoder |
| `t5` | T5 | Encoder-decoder |
| `bart` | BART | Encoder-decoder |

For these, still attempt mapping but warn that the universal contract set
assumes causal decoder-only models.
