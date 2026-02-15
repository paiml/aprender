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

## See Also

- [Qwen Chat Demo](./qwen-chat.md)
- [Qwen APR Native](./qwen-apr-native.md)
- [Rosetta Stone Converter](./rosetta-stone.md)
- [Examples Reference](./examples-reference.md)
