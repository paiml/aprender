# Qwen Inference — LLM Inference with realizar

Aprender provides LLM inference through the `realizar` crate, accessible via the `apr` CLI
or Rust API. The `aprender` crate handles model format conversion and training; all inference
uses `realizar` for optimal throughput (225+ tok/s GPU, 30+ tok/s CPU on 7B Q4K).

## Quick Start (CLI)

```bash
# Run inference via apr CLI (recommended)
apr run model.safetensors --prompt "What is 2+2?" --max-tokens 32

# Chat mode with interactive conversation
apr chat model.gguf

# Serve as HTTP API
apr serve model.apr --port 8080
```

## Examples

### Qwen Chat Demo

Demonstrates Qwen2 model configuration and tokenization setup:

```bash
cargo run --example qwen_chat
```

### Qwen APR Native Format

Creates and loads a Qwen2-0.5B model in native APR v2 format:

```bash
cargo run --example qwen_apr_native
```

### Production Workflow

```bash
# Import from HuggingFace
apr import hf://Qwen/Qwen2-0.5B-Instruct -o qwen2-0.5b.apr

# Quantize for deployment
apr convert qwen2-0.5b.apr --quantize q4k -o qwen2-0.5b-q4k.apr

# Validate quality
apr qa qwen2-0.5b-q4k.apr

# Run inference
apr run qwen2-0.5b-q4k.apr --prompt "Hello!" --max-tokens 64
```

## Supported Model Formats

| Format | CPU | GPU | Notes |
|--------|-----|-----|-------|
| GGUF (Q4K, Q6K) | Yes | Yes | Best throughput, quantized |
| APR (native) | Yes | Yes | Embedded tokenizer, portable |
| SafeTensors (F32, F16) | Yes | Yes (if VRAM sufficient) | Large, full precision |

## Qwen3.5 (Hybrid Attention)

Qwen3.5-9B-Instruct introduces **hybrid attention** — alternating standard softmax
and Gated Delta Net linear attention layers. Key differences from Qwen2:

- **head_dim=256** (explicit, vs Qwen2's computed 128)
- **No attention bias** (`has_bias=false`)
- **Hybrid `layer_types`** — some layers are `"linear"`, using O(n) recurrence
- **vocab_size=248320** (vs 152064 for Qwen2)

```bash
# Import Qwen3.5 (hybrid layers auto-detected from config.json)
apr import hf://Qwen/Qwen3.5-9B-Instruct -o qwen35.apr --arch qwen3_5

# Verify hybrid config
apr inspect qwen35.apr
```

The `realizar` inference engine automatically dispatches to the correct attention
kernel per layer based on the `layer_types` config field. See the
[Qwen3.5 Hybrid Attention chapter](./qwen3.5-hybrid-attention.md) for details.

## Fine-Tuning

Both Qwen2.5 and Qwen3.5 models support classification fine-tuning via LoRA:

```bash
# Qwen3.5-9B: 6.8M trainable params (rank-16 LoRA on Q/V projections)
apr finetune --task classify --model-size 9B --plan
apr finetune --task classify --model-size 9B --data train.jsonl -o checkpoints/

# Qwen2.5-0.5B: 1.1M trainable params (smaller, good for testing)
apr finetune --task classify --model-size 0.5B --data train.jsonl -o checkpoints/
```

Key Qwen3.5 fine-tuning differences:
- **No attention bias** — LoRA adapters target weight matrices only
- **64 LoRA adapters** — 32 layers x 2 targets (Q + V projections)
- **head_dim=256** — larger attention projections than Qwen2

See the [LoRA Fine-Tuning chapter](../ml-fundamentals/fine-tuning.md) for theory and details.

## See Also

- [LoRA Fine-Tuning](../ml-fundamentals/fine-tuning.md)
- [Qwen3.5 Hybrid Attention (GH-278)](./qwen3.5-hybrid-attention.md)
- [Qwen Chat Demo](./qwen-chat.md)
- [Qwen APR Native](./qwen-apr-native.md)
- [Rosetta Stone Converter](./rosetta-stone.md)
- [Examples Reference](./examples-reference.md)
