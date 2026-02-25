# Qwen3.5 Hybrid Attention Architecture

Qwen3.5-9B-Instruct introduces a **hybrid attention** architecture that alternates
between standard softmax attention and **Gated Delta Net** (GDN) linear attention
layers. This chapter explains the architecture, the contract-driven config pipeline,
and how to use it with `apr`.

## Architecture Overview

Qwen3.5 uses a 32-layer transformer with two types of attention:

| Layer Type | Mechanism | Complexity | State |
|-----------|-----------|------------|-------|
| Standard (softmax) | Q*K^T / sqrt(d) → softmax → V | O(n^2) | KV cache |
| Linear (GDN) | Recurrent state update | O(n) per step | Recurrent + Conv |

The `layer_types` field in `config.json` specifies which layers use which mechanism:

```json
{
  "layer_types": [
    "attention", "linear", "attention", "linear",
    ...
  ]
}
```

### Gated Delta Net Recurrence

Linear attention layers implement the **Gated Delta Net** equations:

```
GDN-1: S_t = exp(g_t) * S_{t-1} + k_t (x) delta_t
GDN-2: delta_t = beta_t * (v_t - S_{t-1}^T k_t)
GDN-3: o_t = S_t^T q_t
```

Where:
- `g_t = -exp(A_log) * softplus(a_t + dt_bias)` is the decay factor
- `beta_t = sigma(b_t)` is the update gate
- Q, K are L2-normalized
- State `S` is a `[num_v_heads, key_head_dim, value_head_dim]` matrix

### Key Differences from Qwen2

| Parameter | Qwen2-7B | Qwen3.5-9B |
|-----------|----------|------------|
| head_dim | 128 (computed) | 256 (explicit) |
| num_heads | 28 | 16 |
| num_kv_heads | 4 | 4 |
| attention_bias | true | **false** |
| layer_types | all standard | **hybrid** |
| vocab_size | 152064 | 248320 |

## Config Pipeline

The Qwen3.5 config flows through three stages:

```
SafeTensors config.json
    |
    v  (SafetensorsConfig with layer_types, head_dim, linear_* fields)
AprTransformerConfig
    |
    v  (config_to_gpu with explicit_head_dim, layer_types)
GpuModelConfig
    |
    v  (is_linear_layer(block_idx) dispatch)
forward_linear_block_incremental() or forward_block_incremental()
```

### Contract-Driven Validation

The `contracts/model-families/qwen3_5.yaml` contract enforces:

```yaml
constraints:
  attention_type: gqa
  has_bias: "false"        # No attention bias tensors
  activation: silu
  mlp_type: swiglu
  positional_encoding: rope
```

These constraints drive weight loading: `has_bias=false` means the loader
skips bias tensors entirely instead of loading zeros.

## Weight Loading

Linear attention layers have **different tensor names** from standard layers:

### Standard Attention Layer Tensors
```
model.layers.{n}.self_attn.q_proj.weight    [4096, 4096]
model.layers.{n}.self_attn.k_proj.weight    [1024, 4096]
model.layers.{n}.self_attn.v_proj.weight    [1024, 4096]
model.layers.{n}.self_attn.o_proj.weight    [4096, 4096]
```

### Gated Delta Net Layer Tensors
```
model.layers.{n}.self_attn.in_proj_qkvz.weight  [QKVZ_dim, 4096]
model.layers.{n}.self_attn.in_proj_ba.weight     [2*num_v_heads, 4096]
model.layers.{n}.self_attn.out_proj.weight        [4096, value_dim]
model.layers.{n}.self_attn.conv1d.weight          [conv_dim, 1, kernel]
model.layers.{n}.self_attn.A_log                  [num_v_heads]
model.layers.{n}.self_attn.dt_bias                [num_v_heads]
model.layers.{n}.self_attn.norm.weight            [value_dim]
```

The `in_proj_qkvz` tensor is a **combined** projection that gets split into
Q, K, V, and Z (gate) during loading:

```
in_proj_qkvz = [Q | K | V | Z]
  Q: [key_dim, hidden_dim]     -> qkv_weight (part)
  K: [key_dim, hidden_dim]     -> qkv_weight (part)
  V: [value_dim, hidden_dim]   -> qkv_weight (part)
  Z: [value_dim, hidden_dim]   -> linear_attn.z_weight
```

## CLI Usage

```bash
# Import Qwen3.5 from HuggingFace
apr import hf://Qwen/Qwen3.5-9B-Instruct -o qwen35.apr --arch qwen3_5

# Inspect hybrid attention config
apr inspect qwen35.apr | grep -E "layer_types|linear_"

# Run inference (realizar handles dispatch)
apr run qwen35.apr --prompt "What is 2+2?" --max-tokens 32

# QA validation
apr qa qwen35.apr --assert-tps 50
```

## Falsification Tests

The contract is protected by 8 Popperian falsification tests:

| Test | What it tries to break |
|------|----------------------|
| QWEN35-001 | Exact dimensions (4096, 256, 16, 4, 248320) |
| QWEN35-002 | has_bias must be false |
| QWEN35-003 | hidden_dim == num_heads * head_dim |
| QWEN35-004 | GQA divisibility (16 % 4 == 0) |
| QWEN35-005 | Shape template dimensions |
| QWEN35-006 | rope_theta = 1,000,000 |
| QWEN35-007 | Architecture class = Qwen3_5ForCausalLM |
| QWEN35-008 | SwiGLU MLP (silu + swiglu) |

Run: `cargo test -- falsify_mf_qwen35`

## Fine-Tuning Support

Qwen3.5-9B is wired into the `apr finetune` CLI for classification fine-tuning:

```bash
# Plan mode — shows config and trainable parameter count
apr finetune --task classify --model-size 9B --plan

# Output:
#   Model: 4096h x 32L
#   LoRA: rank=16, alpha=16.0, 64 adapters
#   Classifier: 4096->5 (20485 params)
#   Total trainable: 6,836,229 params
```

The `TransformerConfig::qwen3_5_9b()` factory in entrenar mirrors the contract:

| Config Field | Value | Source |
|-------------|-------|--------|
| hidden_size | 4096 | qwen3_5.yaml `hidden_dim` |
| num_attention_heads | 16 | qwen3_5.yaml `num_heads` |
| num_kv_heads | 4 | qwen3_5.yaml `num_kv_heads` |
| num_hidden_layers | 32 | qwen3_5.yaml `num_layers` |
| vocab_size | 248320 | qwen3_5.yaml `vocab_size` |
| use_bias | false | qwen3_5.yaml `has_bias` |
| head_dim() | 256 | 4096 / 16 |

CLI aliases: `--model-size 9B`, `--model-size qwen3.5-9b`, `--model-size qwen3.5`

### Fine-Tuning Falsification Tests

Seven additional tests (FALSIFY-FT-QWEN35-001..007) verify the config factory
matches the contract:

| Test | What it tries to break |
|------|----------------------|
| FT-QWEN35-001 | vocab_size must be 248320 (not Qwen2's 152064) |
| FT-QWEN35-002 | use_bias must be false |
| FT-QWEN35-003 | head_dim() must be 256 |
| FT-QWEN35-004 | num_hidden_layers must be 32 |
| FT-QWEN35-005 | num_kv_heads must be 4 |
| FT-QWEN35-006 | All YAML dimensions match config factory |
| FT-QWEN35-007 | CLI dispatch consistency (9B != Qwen2) |

Run: `cargo test -- falsify_ft_qwen35`
