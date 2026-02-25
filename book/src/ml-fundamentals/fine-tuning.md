# LoRA Fine-Tuning

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique that freezes
the base model weights and injects trainable low-rank matrices into attention projections.
Instead of updating all parameters (billions for modern LLMs), LoRA trains only a small
number of additional parameters (typically < 1% of the base model).

## How LoRA Works

For a pretrained weight matrix \\( W_0 \in \mathbb{R}^{d \times k} \\), LoRA constrains
the update to a low-rank decomposition:

\\[
W = W_0 + \Delta W = W_0 + BA
\\]

where \\( B \in \mathbb{R}^{d \times r} \\), \\( A \in \mathbb{R}^{r \times k} \\),
and the rank \\( r \ll \min(d, k) \\).

- **A** is initialized with random Gaussian values
- **B** is initialized to zero (so \\( \Delta W = 0 \\) at the start of training)
- Only **A** and **B** are updated during training
- The base model weights \\( W_0 \\) remain frozen

## Supported Base Models

| Model | CLI Flag | Hidden Size | Head Dim | Vocab | Bias |
|-------|----------|-------------|----------|-------|------|
| Qwen2.5-Coder-0.5B | `--model-size 0.5B` | 896 | 64 | 151,936 | Yes |
| Qwen3.5-9B-Instruct | `--model-size 9B` | 4,096 | 256 | 248,320 | No |

## CLI Quickstart

### Classification Fine-Tuning

```bash
# Plan mode (estimate VRAM, show config)
apr finetune --task classify --model-size 9B --plan

# Train with data
apr finetune --task classify \
    --model-size 9B \
    --data train.jsonl \
    --epochs 10 \
    --rank 16 \
    -o checkpoints/

# Qwen2.5 (smaller, for testing)
apr finetune --task classify \
    --model-size 0.5B \
    --data train.jsonl \
    -o checkpoints/
```

### General LoRA Fine-Tuning

```bash
# Plan mode
apr finetune model.apr --method lora --model-size 7B --plan

# Train
apr finetune model.apr --method lora --data train.jsonl -o adapter.apr

# Merge adapter back into base model
apr finetune merge model.apr --adapter adapter.apr -o merged.apr
```

## Qwen3.5-9B Specifics

Qwen3.5 introduces several architectural differences from Qwen2 that affect fine-tuning:

### No Attention Bias

Qwen3.5 does **not** use bias in Q/K/V/O projections (`use_bias=false`). This means:
- LoRA adapters target only the weight matrices, not bias vectors
- The LoRA parameter count is slightly lower than equivalent Qwen2 models

### Explicit Head Dimension

Qwen3.5-9B uses `head_dim=256` (vs Qwen2's typical 128). The attention projection
shapes are:

| Projection | Shape | Notes |
|-----------|-------|-------|
| Q proj | [4096, 4096] | 16 heads x 256 head_dim |
| K proj | [1024, 4096] | 4 KV heads x 256 head_dim |
| V proj | [1024, 4096] | 4 KV heads x 256 head_dim |
| O proj | [4096, 4096] | hidden_dim x (num_heads x head_dim) |

### Hybrid Attention

Qwen3.5 uses a mix of standard softmax attention and linear attention layers.
LoRA targets Q/V projections in both layer types, ensuring the adapter captures
both attention mechanisms.

### 248K Vocabulary

The larger vocabulary (248,320 tokens vs Qwen2's ~152K) affects embedding layer
dimensions but does not change the LoRA targeting strategy, which focuses on
attention projections.

## Classification Pipeline

The classification fine-tuning pipeline consists of:

1. **Base model config** -- loaded via `TransformerConfig::qwen3_5_9b()` or `qwen2_0_5b()`
2. **LoRA injection** -- rank-16 adapters on Q and V projections
3. **Classification head** -- mean pooling + linear layer (hidden_size -> num_classes)
4. **Training loop** -- epoch management, validation split, early stopping, LR scheduling
5. **Checkpointing** -- periodic saves to APR format

```
Input text
    |
    v
[Tokenize] -> [Embedding] -> [Transformer Layers (frozen + LoRA)] -> [Mean Pool]
    |
    v
[Classification Head (trainable)] -> [Softmax] -> [Cross-Entropy Loss]
```

## Contract Validation

Fine-tuning configs are validated against the model family contract:

- `contracts/model-families/qwen3_5.yaml` -- source of truth for dimensions
- `contracts/classification-finetune-v1.yaml` -- classification invariants
- `src/format/model_family_contract_falsify.rs` -- Popperian falsification tests

The falsification tests (FALSIFY-FT-QWEN35-001 through 007) verify that the
`TransformerConfig::qwen3_5_9b()` factory matches the YAML contract exactly.
If they diverge, the test suite catches it before any training runs.

## References

- Hu et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Dettmers et al. (2023). [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- `docs/specifications/qwen3.5-fine-tune.md` -- full specification
