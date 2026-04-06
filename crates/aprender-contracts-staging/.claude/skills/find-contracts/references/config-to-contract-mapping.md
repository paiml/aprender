# Config-to-Contract Mapping Reference

Complete mapping from Hugging Face `config.json` fields to provable-contracts
YAML contract files.

## Universal Contracts (always required for transformer LLMs)

These 13 contracts are required by every autoregressive transformer model
regardless of specific configuration.

| Contract | Reason |
|----------|--------|
| `model-config-algebra-v1` | Config dimension consistency (h = n_h * d_k, etc.) |
| `softmax-kernel-v1` | Attention probability normalization |
| `matmul-kernel-v1` | Q*K^T and attn*V matrix multiplies |
| `linear-projection-v1` | Dense/linear layer forward pass |
| `embedding-lookup-v1` | Token-to-vector lookup |
| `embedding-algebra-v1` | Embedding dimension algebra |
| `tensor-shape-flow-v1` | Shape propagation through layers |
| `cross-entropy-kernel-v1` | Loss computation (logits → loss) |
| `inference-pipeline-v1` | End-to-end inference flow |
| `attention-kernel-v1` | Scaled dot-product attention |
| `attention-scaling-v1` | 1/sqrt(d_k) scaling factor |
| `kv-cache-sizing-v1` | KV cache memory bounds |
| `kv-cache-equivalence-v1` | Cached vs uncached equivalence |

## Conditional Contracts — Activation Functions

| config.json trigger | Contract | Notes |
|---------------------|----------|-------|
| `hidden_act = "silu"` | `silu-kernel-v1` | SiLU/Swish activation |
| `hidden_act = "silu"` | `swiglu-kernel-v1` | Most SiLU models use SwiGLU FFN (gate * silu(up)) |
| `hidden_act = "gelu"` | `gelu-kernel-v1` | Standard GELU |
| `hidden_act = "gelu_new"` | `gelu-kernel-v1` | Approximate GELU |
| `hidden_act = "gelu_fast"` | `gelu-kernel-v1` | Fast GELU approximation |
| `hidden_act = "gelu_pytorch_tanh"` | `gelu-kernel-v1` | PyTorch tanh-approx GELU |

**Decision rule**: If `hidden_act` contains the substring `"gelu"`, require
`gelu-kernel-v1`. If `hidden_act == "silu"`, require both `silu-kernel-v1`
and `swiglu-kernel-v1`.

## Conditional Contracts — Normalization

| config.json trigger | Contract | Notes |
|---------------------|----------|-------|
| `rms_norm_eps` present and not null | `rmsnorm-kernel-v1` | RMSNorm (LLaMA, Qwen, Mistral, Gemma) |
| `layer_norm_eps` or `layer_norm_epsilon` present | `layernorm-kernel-v1` | LayerNorm (GPT-2, Falcon, Phi) |

**Decision rule**: Check for the presence of either epsilon field. Some models
have both (hybrid normalization) — require both contracts in that case.

## Conditional Contracts — Position Encoding

| config.json trigger | Contract | Notes |
|---------------------|----------|-------|
| `rope_theta` present | `rope-kernel-v1` | Rotary Position Embedding |
| `rope_scaling` not null | `rope-extrapolation-v1` | Extended context via RoPE scaling (YaRN, linear, dynamic) |
| `partial_rotary_factor` present | `rope-kernel-v1` | Partial RoPE (Phi-style) |
| `partial_rotary_factor` present | `absolute-position-v1` | Non-rotary dimensions use learned absolute |
| `use_alibi = true` or `alibi = true` | `alibi-kernel-v1` | ALiBi position bias |

**Edge cases**:
- Phi models have `partial_rotary_factor` (e.g., 0.5) — only half the head
  dimension uses RoPE, the rest uses absolute position encoding.
- `rope_scaling.type` can be `"linear"`, `"dynamic"`, `"yarn"`, `"longrope"`,
  etc. All trigger `rope-extrapolation-v1`.

## Conditional Contracts — Attention Variants

| config.json trigger | Contract | Notes |
|---------------------|----------|-------|
| `num_key_value_heads` != `num_attention_heads` | `gqa-kernel-v1` | Grouped-Query Attention |
| `multi_query = true` | `gqa-kernel-v1` | MQA is GQA with n_kv=1 (Falcon) |
| `sliding_window` set (not null) | `sliding-window-attention-v1` | Sliding window attention (Mistral, Qwen2) |
| `qk_layernorm = true` or `qk_norm = true` | `qk-norm-v1` | QK normalization before dot product |
| `attn_logit_softcapping` set | `attention-scaling-v1` | Already universal, but note softcapping variant |

**Decision rule for GQA**: Compare `num_key_value_heads` to `num_attention_heads`.
If they differ (or if `multi_query` is true), the model uses grouped-query
attention and requires `gqa-kernel-v1`. If `num_key_value_heads` is absent,
assume MHA (no GQA contract needed).

## Conditional Contracts — Architecture-Specific

| config.json trigger | Contract | Notes |
|---------------------|----------|-------|
| `tie_word_embeddings = true` | `tied-embeddings-v1` | Shared input/output embedding matrix |
| `model_type` contains "gated_delta_net" or GDN architecture fields | `gated-delta-net-v1` | Gated delta net recurrence |
| GDN architecture | `hybrid-layer-dispatch-v1` | Hybrid attention + GDN layer routing |
| GDN architecture | `conv1d-kernel-v1` | Short convolution in GDN blocks |
| `ssm_cfg` or `d_state` or `d_conv` present | `ssm-kernel-v1` | State-space model (Mamba/Jamba) |
| SSM + attention layers | `hybrid-layer-dispatch-v1` | Hybrid SSM + attention routing |

## Model-Specific Contracts

Always check for these two contracts using the model type name:

| Pattern | Example for Qwen3.5 | Purpose |
|---------|---------------------|---------|
| `<model>-shapes-v1.yaml` | `qwen35-shapes-v1.yaml` | Concrete tensor shapes for the specific config |
| `<model>-e2e-verification-v1.yaml` | `qwen35-e2e-verification-v1.yaml` | End-to-end numerical verification |

**Model name normalization**:
- Use `model_type` from config
- Lowercase
- Remove hyphens and underscores
- Common mappings: `llama` → `llama`, `qwen2` → `qwen2`, `mistral` → `mistral`,
  `phi` → `phi`, `phi3` → `phi3`, `gemma2` → `gemma2`, `falcon` → `falcon`,
  `gpt2` → `gpt2`, `starcoder2` → `starcoder2`

## Quick Reference: Common Model Families

### LLaMA 3 / 3.1 / 3.2
- Universal (13) + `silu-kernel` + `swiglu-kernel` + `rmsnorm-kernel` + `rope-kernel` + `gqa-kernel`
- 3.1+ adds: `rope-extrapolation` (rope_scaling present)
- `tie_word_embeddings` often true → `tied-embeddings`

### Mistral / Mixtral
- Universal (13) + `silu-kernel` + `swiglu-kernel` + `rmsnorm-kernel` + `rope-kernel` + `gqa-kernel` + `sliding-window-attention`
- Mixtral adds MoE (no contract yet for MoE routing)

### Qwen2 / Qwen2.5 / Qwen3.5
- Universal (13) + `silu-kernel` + `swiglu-kernel` + `rmsnorm-kernel` + `rope-kernel` + `gqa-kernel`
- `tie_word_embeddings` varies by size
- Qwen3.5 may include GDN layers → `gated-delta-net` + `hybrid-layer-dispatch` + `conv1d-kernel`

### Gemma / Gemma2
- Universal (13) + `gelu-kernel` + `rmsnorm-kernel` + `rope-kernel`
- Gemma2: `attn_logit_softcapping` → note in `attention-scaling`
- Gemma2: `sliding_window` on alternating layers → `sliding-window-attention`

### Phi / Phi-2 / Phi-3
- Universal (13) + `gelu-kernel` + `layernorm-kernel` (Phi-2) or `rmsnorm-kernel` (Phi-3)
- `partial_rotary_factor` → `rope-kernel` + `absolute-position`
- Phi-2: `qk_layernorm = true` → `qk-norm`

### Falcon
- Universal (13) + `gelu-kernel` + `layernorm-kernel`
- `multi_query = true` → `gqa-kernel` (MQA, n_kv=1)
- `parallel_attn = true` (parallel attention + FFN, informational only)
- `alibi = true` (some Falcon variants) → `alibi-kernel`
