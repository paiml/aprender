---
title: "Qwen3.5-9B: Quantization, Serving, and Fine-Tuning Pipeline"
issue: GH-278
status: In Progress
created: 2026-02-17
updated: 2026-02-17
---

# Qwen3.5-9B: Quantization, Serving, and Fine-Tuning Pipeline

**Goal:** Be the first to quantize, serve, and publish Qwen3.5-9B-Instruct in APR format.

**Primary Model:** [`Qwen/Qwen3.5-9B-Instruct`](https://huggingface.co/Qwen/Qwen3.5-9B-Instruct)
**Release Date:** 2026-02-16 (yesterday — first-mover window is NOW)
**License:** Apache 2.0

## Phases

| Phase | Scope | Blocked By | Target |
|-------|-------|------------|--------|
| **1** | Quantize + Serve + Publish APR | Nothing (implement today) | 2026-02-17 |
| 2 | SVD LoRA extraction from community fine-tunes | Phase 1 + community models | TBD |
| 3 | QLoRA fine-tuning via entrenar | trueno backward ops | TBD |

---

## Model Specification: Qwen3.5-9B-Instruct

Source: [HuggingFace transformers `Qwen3_5TextConfig`](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5)

### Architecture

| Parameter | Value | vs Qwen2.5-7B |
|-----------|-------|---------------|
| Architecture | `Qwen3_5ForCausalLM` | was `Qwen2ForCausalLM` |
| model_type | `qwen3_5` | was `qwen2` |
| vocab_size | 248,320 | was 152,064 (+63%) |
| hidden_size | 4,096 | was 3,584 |
| num_hidden_layers | 32 | was 28 |
| num_attention_heads | 16 | was 28 |
| num_key_value_heads | 4 | was 4 |
| **head_dim** | **256** | **was 128 (2x)** |
| intermediate_size | 12,288 | was 18,944 |
| max_position_embeddings | 262,144 | was 131,072 (2x) |
| rope_theta | 1,000,000 | same |
| rms_norm_eps | 1e-6 | same |
| **attention_bias** | **false** | **was true** |
| tie_word_embeddings | false | same |
| activation | silu (SwiGLU) | same |
| norm_type | rmsnorm | same |

### Hybrid Attention (NEW — not in Qwen2/Qwen3)

Qwen3.5 introduces **hybrid linear/quadratic attention**. Some layers use standard softmax
attention, others use linear attention with convolution:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `layer_types` | per-layer list: `"attention"` or `"linear"` | Routes each layer |
| `linear_conv_kernel_dim` | 4 | Conv kernel for linear attn |
| `linear_key_head_dim` | 128 | Key dim in linear layers |
| `linear_value_head_dim` | 128 | Value dim in linear layers |
| `linear_num_key_heads` | 16 | Key heads in linear layers |
| `linear_num_value_heads` | 32 | Value heads in linear layers |

**Impact:** Realizar needs a new linear attention forward path. Standard attention layers
work unchanged. The layer dispatch reads `layer_types` from config to route each block.

### Key Differences from Qwen2.5 (What Breaks)

1. **No attention bias** — q/k/v projection bias tensors don't exist. Import must not expect them.
2. **head_dim=256** — RoPE, attention score computation, KV cache all assume head_dim. Must update.
3. **vocab_size=248,320** — embedding/lm_head tensor shapes change. Tokenizer is different.
4. **Linear attention layers** — new forward path needed in realizar.
5. **Tensor naming** — likely `model.layers.{n}.self_attn.*` same pattern, but verify at import time.

### What Stays the Same

- GQA attention (Q=16 heads, KV=4 heads)
- SwiGLU MLP (gate_proj, up_proj, down_proj)
- RMSNorm (pre-normalization)
- RoPE positional encoding (theta=1M)
- Row-major layout (SafeTensors native)
- Q4K/Q6K quantization math (operates on tensor data, architecture-agnostic)

---

## Phase 1: Quantize + Serve + Publish (TODAY)

### Objective

Be the first to publish Qwen3.5-9B-Instruct in quantized APR format with working inference
via `apr serve`. No dependency on llama.cpp or GGUF ecosystem.

### Pipeline

```
apr pull hf://Qwen/Qwen3.5-9B-Instruct           # Download SafeTensors BF16
    │
    ▼
apr import Qwen3.5-9B-Instruct/ \                  # SafeTensors → APR Q4K
    --quantize q4k \
    --arch qwen3_5 \
    -o qwen3.5-9b-q4k.apr
    │
    ▼
apr validate qwen3.5-9b-q4k.apr --quality          # Contract + integrity check
apr tensors qwen3.5-9b-q4k.apr --stats             # Verify shapes + statistics
    │
    ▼
apr run qwen3.5-9b-q4k.apr \                       # Smoke test inference
    "What is 2+2? Answer with just the number." \
    --max-tokens 32
    │
    ▼
apr serve qwen3.5-9b-q4k.apr --port 8080           # OpenAI-compatible serving
    │
    ▼
Push to HuggingFace: paiml/Qwen3.5-9B-Instruct-APR # Publish FIRST
```

### Implementation Tasks

#### Task 1: Model Family Contract

**File:** `contracts/model-families/qwen3_5.yaml`

New contract derived from `qwen2.yaml` with Qwen3.5-specific parameters:

```yaml
family: qwen3_5
display_name: "Qwen3.5"
vendor: Alibaba
architectures:
  - Qwen3_5ForCausalLM
  - Qwen3_5ForConditionalGeneration
hf_pattern: "Qwen/Qwen3.5*"

size_variants:
  9b:
    parameters: "9B"
    hidden_dim: 4096
    num_layers: 32
    num_heads: 16
    num_kv_heads: 4
    intermediate_dim: 12288
    vocab_size: 248320
    max_position_embeddings: 262144
    head_dim: 256
    rope_theta: 1000000.0
    rms_norm_eps: 0.000001

constraints:
  attention_type: gqa
  activation: silu
  norm_type: rmsnorm
  has_bias: "false"           # KEY DIFFERENCE from Qwen2
  tied_embeddings: "false"
  positional_encoding: rope
  mlp_type: swiglu
  hybrid_attention: "true"    # NEW: some layers use linear attention

tensor_template:
  embedding: "model.embed_tokens.weight"
  lm_head: "lm_head.weight"
  final_norm: "model.norm.weight"
  per_layer:
    q_proj: "model.layers.{n}.self_attn.q_proj.weight"
    k_proj: "model.layers.{n}.self_attn.k_proj.weight"
    v_proj: "model.layers.{n}.self_attn.v_proj.weight"
    o_proj: "model.layers.{n}.self_attn.o_proj.weight"
    gate_proj: "model.layers.{n}.mlp.gate_proj.weight"
    up_proj: "model.layers.{n}.mlp.up_proj.weight"
    down_proj: "model.layers.{n}.mlp.down_proj.weight"
    input_layernorm: "model.layers.{n}.input_layernorm.weight"
    post_attention_layernorm: "model.layers.{n}.post_attention_layernorm.weight"
    # NO q/k/v bias tensors (attention_bias=false)

shape_template:
  embedding: "[vocab_size, hidden_dim]"
  lm_head: "[vocab_size, hidden_dim]"
  q_proj: "[num_heads * head_dim, hidden_dim]"
  k_proj: "[num_kv_heads * head_dim, hidden_dim]"
  v_proj: "[num_kv_heads * head_dim, hidden_dim]"
  o_proj: "[hidden_dim, num_heads * head_dim]"
  gate_proj: "[intermediate_dim, hidden_dim]"
  up_proj: "[intermediate_dim, hidden_dim]"
  down_proj: "[hidden_dim, intermediate_dim]"
  input_layernorm: "[hidden_dim]"
  post_attention_layernorm: "[hidden_dim]"

quantizations:
  - q4_k_m
  - q6_k
  - q8_0
  - f16
  - f32
```

**Note:** Linear attention layer tensors (conv weights, linear K/V projections) need to be
discovered at import time from the actual SafeTensors files. The contract above covers the
standard attention layers; linear layer tensors will be added after inspecting the model.

#### Task 2: Architecture Registration

Register `Qwen3_5ForCausalLM` in the model family loader so `apr import` auto-detects
the architecture from `config.json`.

**Files to modify:**
- `src/format/model_family.rs` — add `Qwen3_5` variant
- `src/format/model_family_loader.rs` — register `qwen3_5.yaml`
- `src/format/model_family_part_03.rs` — add `make_qwen3_5_config()` (derive from `make_qwen2_config`)

**Key changes from Qwen2:**
- `has_bias: false` — skip q/k/v/o bias tensor expectations
- `head_dim: 256` — affects shape validation for Q/K/V projections
- `vocab_size: 248320` — affects embedding/lm_head shape validation

#### Task 3: Import Pipeline (SafeTensors → APR Q4K)

The import pipeline (`src/format/converter/write.rs`) handles SafeTensors → APR conversion.
Q4K quantization operates on tensor data and is architecture-agnostic.

**Changes needed:**
- Handle missing bias tensors gracefully (skip instead of error when `has_bias=false`)
- Validate shape templates with head_dim=256 (q_proj shape: `[16*256, 4096] = [4096, 4096]`)
- Store linear attention layer metadata in APR header for realizar to read

**No changes needed:**
- Q4K/Q6K quantization math (same block format)
- Transpose logic (SafeTensors is already row-major)
- APR file format (metadata extensible)

#### Task 4: Realizar Inference — Standard Attention Layers

Standard attention layers in Qwen3.5 are identical to Qwen2 except:

1. **No bias** — skip bias addition in Q/K/V/O projections
2. **head_dim=256** — RoPE frequency computation, attention score scaling (`1/sqrt(256)`),
   KV cache slot size all key off head_dim

**Files in realizar:**
- `src/gpu/scheduler/kv.rs` — attention forward, RoPE, KV cache
- `src/layers/model.rs` — model loading, embedding lookup, lm_head projection
- `src/inference/norm.rs` — RMSNorm (unchanged)

**These are parameter changes, not architectural changes.** The attention code path is the
same; it just reads different dimensions from the model config.

#### Task 5: Realizar Inference — Linear Attention Layers

This is the novel work. Qwen3.5 hybrid architecture dispatches each layer to either
standard softmax attention or linear attention based on `layer_types` in config.

**Linear attention forward pass:**
```
input → LayerNorm → Linear Q/K/V Projection
    → Conv1D(kernel=4) on K
    → Linear attention: O = V * (K^T * Q)  [no softmax, O(n) not O(n²)]
    → Output projection
    → Residual + LayerNorm → SwiGLU MLP → Residual
```

**Implementation approach:**
- Add `LinearAttentionLayer` alongside existing `AttentionLayer` in realizar
- Layer dispatch reads `layer_types[i]` to select attention vs linear per block
- Linear attention uses different head counts (`linear_num_key_heads=16`,
  `linear_num_value_heads=32`) and dims (`linear_key_head_dim=128`,
  `linear_value_head_dim=128`)
- Conv1D with kernel_dim=4 on the key pathway

**Files to create/modify in realizar:**
- `src/gpu/scheduler/linear_attn.rs` — NEW: linear attention forward
- `src/gpu/scheduler/kv.rs` — dispatch per layer_type
- `src/layers/model.rs` — load linear attention weights, store layer_types config

**Trueno kernels needed:**
- Conv1D (kernel=4) — may already exist or can be implemented as grouped matmul
- Linear attention score computation (no softmax path)

#### Task 6: Tokenizer (248K Vocab)

Qwen3.5 uses a larger vocabulary (248,320 vs 152,064). The tokenizer is BPE-based
(same algorithm as Qwen2) but with expanded token set.

**Changes:**
- `apr pull` downloads `tokenizer.json` automatically from HF
- BPE tokenizer in realizar loads from `tokenizer.json` — no code change needed
  if it handles arbitrary vocab sizes
- Verify special tokens: `<|im_start|>`, `<|im_end|>`, thinking mode tokens

#### Task 7: Smoke Test + Validation

```bash
# Contract validation
apr validate qwen3.5-9b-q4k.apr --quality

# Tensor inspection
apr tensors qwen3.5-9b-q4k.apr --stats | head -20

# Inference smoke test
apr run qwen3.5-9b-q4k.apr "What is 2+2?" --max-tokens 32 --temperature 0

# Benchmark
apr bench qwen3.5-9b-q4k.apr --warmup 3 --measure 10

# Full QA
apr qa qwen3.5-9b-q4k.apr
```

**Success criteria:**
- `apr validate` passes all contract checks
- `apr run` produces coherent output containing "4" for the 2+2 test
- `apr bench` reports tok/s numbers
- No NaN/Inf in tensor statistics

#### Task 8: Publish to HuggingFace

Push quantized model to HF as `paiml/Qwen3.5-9B-Instruct-APR`:

**Variants to publish:**
- `qwen3.5-9b-q4k.apr` (~5 GB) — best size/quality tradeoff
- `qwen3.5-9b-q6k.apr` (~7 GB) — higher quality
- `qwen3.5-9b-q8.apr` (~9 GB) — near-lossless

**Model card contents:**
- Architecture details (hybrid attention, head_dim=256)
- Quantization method and quality metrics
- `apr serve` usage instructions
- Benchmark results (tok/s CPU + GPU)
- Provenance: SafeTensors BF16 → APR Q4K pipeline

### Implementation Order

The tasks have dependencies:

```
Task 1 (contract) ──► Task 2 (arch registration) ──► Task 3 (import)
                                                         │
                                                         ▼
Task 6 (tokenizer) ──► Task 4 (std attention) ──► Task 5 (linear attention)
                                                         │
                                                         ▼
                                                  Task 7 (validation)
                                                         │
                                                         ▼
                                                  Task 8 (publish)
```

**Critical path:** Tasks 1 → 2 → 3 → 4 → 5 → 7. Tasks 1-3 are small config/enum changes.
Task 4 is parameter adjustments. **Task 5 (linear attention) is the only significant new code.**

### Quantization Variants

| Format | Size (est.) | Quality | Use Case |
|--------|-------------|---------|----------|
| APR Q4_K | ~5 GB | Good | Default serving, fits 8GB VRAM |
| APR Q6_K | ~7 GB | Better | Quality-sensitive, fits 12GB VRAM |
| APR Q8_0 | ~9 GB | Near-lossless | Reference quality |
| APR F16 | ~18 GB | Lossless | Baseline comparison |

---

## Phase 2: SVD LoRA Extraction + Model Merging (Future)

**Prerequisite:** Phase 1 complete, Qwen3.5 community fine-tunes available on HuggingFace.

### Objective

Extract task-specific capabilities from community Qwen3/3.5 fine-tunes via SVD decomposition
and merge them into the base Qwen3.5-9B to create a stronger model without any training.

### Approach

1. Download 2-3 top-performing Qwen3.5 fine-tunes from HF (code, math, reasoning)
2. Compute task vectors: `delta = fine_tune - base`
3. SVD decompose deltas into low-rank A/B matrices (LoRA-equivalent)
4. Merge via `apr merge --strategy ties` or `apr merge --strategy dare`
5. Validate with `apr qa` + benchmark suite
6. Submit to Open LLM Leaderboard

### Stack Changes

- `apr finetune extract` — SVD extraction of task vectors (may already exist)
- `apr merge` — verify compatibility with Qwen3.5 tensor names
- Eval harness integration for leaderboard benchmarks

---

## Phase 3: QLoRA Fine-Tuning via Entrenar (Future)

**Prerequisite:** Phase 1 complete + trueno backward ops + entrenar Phase 1-3.

### Objective

True gradient-based fine-tuning of Qwen3.5-9B using QLoRA (4-bit frozen base + FP16
trainable adapters) via the entrenar training library.

### Approach

```yaml
# entrenar config (declarative)
model:
  path: qwen3.5-9b-q4k.apr
  layers: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

data:
  train: code-instructions.parquet
  batch_size: 4

optimizer:
  name: adamw
  lr: 2e-4
  weight_decay: 0.01

lora:
  rank: 64
  alpha: 16
  dropout: 0.05

quantize:
  bits: 4
  method: qlora
```

### Entrenar Feature Requests

1. **Qwen3.5 model family support** — LoRA target layer names matching `qwen3_5.yaml` contract
2. **Hybrid attention awareness** — apply LoRA to both standard and linear attention layers
3. **Linear attention backward** — trueno needs backward ops for conv1d + linear attention
4. **248K vocab handling** — embedding layer LoRA with large vocabulary
5. **Checkpoint export** — `entrenar export --format apr` for serving via `apr serve`

### Target Benchmarks

| Benchmark | Metric | Target |
|-----------|--------|--------|
| HumanEval | pass@1 | > base Qwen3.5-9B |
| MBPP | pass@1 | > base Qwen3.5-9B |
| GSM8K | accuracy | > base Qwen3.5-9B |
| MMLU | accuracy | > base Qwen3.5-9B |

### Training Data Candidates

- [OpenCoder-LLM/opc-sft-stage2](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2) — code SFT
- [TIGER-Lab/MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) — math reasoning
- [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) — DPO alignment

---

## References

- [Qwen3.5 HuggingFace Collection](https://huggingface.co/collections/Qwen/qwen35)
- [Qwen3.5 transformers docs](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5)
- [Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1)
- [Qwen3-8B config.json](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) (reference for Qwen3 dense arch)
- [Understanding and Implementing Qwen3 From Scratch](https://magazine.sebastianraschka.com/p/qwen3-from-scratch)
- [Unsloth Qwen3.5 fine-tuning guide](https://unsloth.ai/docs/models/qwen3.5)
- [Open LLM Leaderboard submission guide](https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/submitting)
