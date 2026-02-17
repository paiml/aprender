---
title: "Qwen3-8B: Performance Parity Pipeline"
issue: GH-279
status: Partially Complete (GH-280 blocks GPU parity)
created: 2026-02-17
updated: 2026-02-18
---

# Qwen3-8B: Performance Parity Pipeline

**Goal:** Import Qwen3-8B from SafeTensors, quantize to Q4K, export our own GGUF with full metadata, and achieve Ollama performance parity.

**Primary Model:** [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B)

## Motivation

We tested `apr qa` on a third-party Qwen3-8B GGUF from HuggingFace and found:
- 8/10 QA gates pass, but **GPU golden output = garbage** and **0.52x Ollama parity**
- Root cause: the GGUF had **sparse metadata** (only `general.architecture`) — no `rope_theta`, no `rms_norm_eps`, no context length
- Our approach: import SafeTensors directly (full metadata from `config.json`), quantize ourselves, export our own GGUF

## Architecture: Qwen3-8B

Source: [HuggingFace `Qwen/Qwen3-8B` config.json](https://huggingface.co/Qwen/Qwen3-8B)

### Parameters

| Parameter | Qwen3-8B | vs Qwen2-7B |
|-----------|----------|-------------|
| Architecture | `Qwen3ForCausalLM` | was `Qwen2ForCausalLM` |
| model_type | `qwen3` | was `qwen2` |
| vocab_size | 151,936 | same |
| hidden_size | 4,096 | was 3,584 |
| num_hidden_layers | 36 | was 28 |
| num_attention_heads | 32 | was 28 |
| num_key_value_heads | 8 | was 4 |
| head_dim | 128 | same |
| intermediate_size | 12,288 | was 18,944 |
| max_position_embeddings | 40,960 | was 131,072 |
| rope_theta | 1,000,000 | same |
| rms_norm_eps | 1e-6 | same |
| **attention_bias** | **false** | **was true** |
| **qk_norm** | **true** | **not present** |
| tie_word_embeddings | false | same |
| activation | silu (SwiGLU) | same |
| norm_type | rmsnorm | same |

### Key Differences from Qwen2

1. **No attention bias**: `attention_bias: false` — no Q/K/V bias tensors (3 fewer tensors per layer)
2. **QK normalization**: New `q_norm` and `k_norm` RMSNorm weight tensors per layer (2 extra tensors per layer, shape `[head_dim]`)
3. **Thinking mode**: Special tokens `<think>` (151667) and `</think>` (151668) for chain-of-thought reasoning
4. **Net tensor delta**: -3 bias + 2 norm = -1 tensor per layer vs Qwen2

### Tensor Inventory (8B)

| Tensor | Shape | Count |
|--------|-------|-------|
| `model.embed_tokens.weight` | [151936, 4096] | 1 |
| `lm_head.weight` | [151936, 4096] | 1 |
| `model.norm.weight` | [4096] | 1 |
| `model.layers.{0-35}.self_attn.q_proj.weight` | [4096, 4096] | 36 |
| `model.layers.{0-35}.self_attn.k_proj.weight` | [1024, 4096] | 36 |
| `model.layers.{0-35}.self_attn.v_proj.weight` | [1024, 4096] | 36 |
| `model.layers.{0-35}.self_attn.o_proj.weight` | [4096, 4096] | 36 |
| `model.layers.{0-35}.self_attn.q_norm.weight` | [128] | 36 |
| `model.layers.{0-35}.self_attn.k_norm.weight` | [128] | 36 |
| `model.layers.{0-35}.mlp.gate_proj.weight` | [12288, 4096] | 36 |
| `model.layers.{0-35}.mlp.up_proj.weight` | [12288, 4096] | 36 |
| `model.layers.{0-35}.mlp.down_proj.weight` | [4096, 12288] | 36 |
| `model.layers.{0-35}.input_layernorm.weight` | [4096] | 36 |
| `model.layers.{0-35}.post_attention_layernorm.weight` | [4096] | 36 |
| **Total** | | **399** |

## Pipeline

```bash
# Step 1: Download SafeTensors from HuggingFace
apr pull hf://Qwen/Qwen3-8B

# Step 2: Import to APR with Q4K quantization
apr import hf://Qwen/Qwen3-8B -o qwen3-8b-q4k.apr --quantize q4k --arch qwen3

# Step 3: Export our own GGUF (full metadata from config.json)
apr export qwen3-8b-q4k.apr --format gguf -o qwen3-8b-q4k.gguf

# Step 4: QA validation (all 10 gates)
apr qa qwen3-8b-q4k.gguf --assert-tps 100

# Step 5: Performance benchmark
apr bench qwen3-8b-q4k.gguf --warmup 3 --measure 10
```

## Performance Targets

| Metric | 3rd-party GGUF | Our GGUF | Target | Status |
|--------|---------------|----------|--------|--------|
| Ollama parity | 0.52x | **0.7x** (87 vs 132) | >= 1.0x | Grade D |
| GPU golden output | garbage | garbage (no QK norm in kernel) | coherent | GH-280 |
| CPU golden output | — | **coherent** (thinking mode works) | coherent | PASS |
| QA gates | 8/10 | **8/10** | 10/10 | 2 remain |
| CPU throughput | — | **74.4 tok/s** | 40+ | PASS |
| GPU speedup | — | **15.5x** (81 vs 5) | 2x+ | PASS |

### Root Cause of 0.52x Parity

The third-party GGUF had only `general.architecture` in metadata. Missing:
- `rope_freq_base` (rope_theta) — defaults to 10000 instead of 1000000, corrupts position encoding
- `context_length` — defaults too low, truncates KV cache
- `rms_norm_eps` — wrong epsilon produces numerical drift in layernorms
- `head_count_kv` — wrong GQA grouping produces garbage attention

Our pipeline imports from SafeTensors + `config.json`, which has all parameters. The GGUF export writes them as proper GGUF metadata keys.

## QK Normalization

Qwen3 applies RMSNorm to Q and K projections before attention score computation. This is a per-head normalization with learned weights:

```
Q_norm = RMSNorm(Q, q_norm_weight)  # shape: [batch, heads, seq, head_dim]
K_norm = RMSNorm(K, k_norm_weight)  # shape: [batch, kv_heads, seq, head_dim]
attn = softmax(Q_norm @ K_norm^T / sqrt(head_dim)) @ V
```

The `q_norm.weight` and `k_norm.weight` tensors have shape `[head_dim]` (128) and are broadcast across all heads. This stabilizes attention scores at scale and eliminates the need for attention bias.

## Contract

Model family contract: [`contracts/model-families/qwen3.yaml`](../../contracts/model-families/qwen3.yaml)

Key contract constraints:
- `has_bias: false` (no Q/K/V bias tensors — differs from Qwen2)
- `qk_norm: true` (Q/K RMSNorm — new in Qwen3)
- `attention_type: gqa` (8 KV heads for 32 Q heads)
- Same SwiGLU MLP, RoPE, RMSNorm as Qwen2

## Verification Checklist

- [x] `apr import` SafeTensors → APR succeeds with 399 tensors (8.28GB Q4K)
- [x] Contract validation passes (all shapes match qwen3.yaml)
- [x] `apr export` APR → GGUF produces valid GGUF with full metadata (267 dup tokens deduped)
- [x] GGUF metadata includes 19 keys (arch=qwen3, layers=36, heads=32/8kv, hidden=4096)
- [ ] `apr qa` passes all gates — **8/10 pass**
- [x] Tensor Contract, Metadata, Throughput (74.4 tok/s), GPU Speedup (15.5x), PTX, GPU State, Perf Regression
- [x] Ollama Parity: 0.7x (87 vs 132 tok/s) — PASS (threshold 0.2x) but not at 1.0x target
- [ ] Golden Output: CPU coherent (thinking mode), GPU garbage (QK norm not in kernel → GH-280)
- [ ] Format Parity: auto-discovers SafeTensors from `~/.apr/cache/hf/` (GH-279-2)
- [x] Thinking mode tokens present in tokenizer vocabulary (`<think>`=151667, `</think>`=151668)

## Bugs Found During Pipeline

### GH-279-1: GGUF token table dedup (FIXED)
Qwen3 tokenizer has 267 reserved tokens all mapped to `<unk>`. llama.cpp requires unique strings.
Fix: `ValidatedGgufMetadata::validate()` auto-dedupes with `_N` suffixes (commit `135de184`).

### GH-279-2: `apr import hf://` fails for sharded models (FIXED)
`apr import hf://Qwen/Qwen3-8B` now checks `~/.apr/cache/hf/` and falls back to `model.safetensors.index.json` for sharded models.
Fix: Added APR cache to `find_in_cache()` + sharded index fallback in `resolve_hf_source()`.

### GH-279-3: GPU parity failure (IN PROGRESS → PARTIALLY FIXED)
GPU forward pass diverges from CPU (cosine sim 0.000479). Root cause: QK norm not applied in GPU matmul path.
CPU argmax: 33975 | GPU argmax: 85222 | Max logit diff: 12084.8
Fixes applied: trueno PTX bar_sync label mismatch fixed (commit `d989451`), sm_70 baseline target for broad GPU compatibility. Remaining: QK norm kernel (GH-280).

### GH-279-4: Golden output gate thinks vs answers (FIXED)
Qwen3 uses thinking mode by default — generates `<think>` chain before answer. Golden output gate expects direct "4".
Fix: `strip_thinking_blocks()` strips `<think>...</think>` before `verify_output()`. No-op for non-thinking models.

### GH-284: CPU inference 2.4 tok/s vs Ollama 120 tok/s (FIXED)
Root cause: F32 matmul in `realizar::apr_transformer::helpers::f32_matmul()` was single-threaded scalar code.
Fix: Added rayon parallel chunking (`par_chunks_mut` with 64-element chunks) and AVX2 SIMD dot product (`simd_dot_f32`) for all F32 matmul operations (commit `55cfb95` in realizar).

### GH-282: PTX kernel load failures in `apr serve --gpu` (PARTIALLY FIXED)
Root cause 1: trueno PTX `bar_sync(id)` stored barrier ID in `.src` operand but emitter read from `.label` field. `bar_sync(1)` emitted as `bar.sync 0;` instead of `bar.sync 1;`.
Root cause 2: Hardcoded `sm_89` PTX target caused JIT failures on GPUs older than RTX 4090.
Fix: Set `.label("sync {id}")` in `bar_sync()`, changed default target to `sm_70` (Volta baseline), added `emit_ptx_for_target()` API (commit `d989451` in trueno).

## References

- [Qwen3 Technical Report](https://qwenlm.github.io/blog/qwen3/)
- [HuggingFace Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen3 model family contract](../../contracts/model-families/qwen3.yaml)
- [Qwen3.5 spec (GH-278)](./qwen3.5-fine-tune.md) — hybrid linear/quadratic attention (different architecture)
