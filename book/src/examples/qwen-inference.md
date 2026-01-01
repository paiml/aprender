# Qwen Inference - LLM Inference Example

This example demonstrates Qwen2 model inference for text generation.

**Run command:**
```bash
cargo run --example qwen_inference --release --features inference
```

**Topics covered:**
- Qwen2 model loading (0.5B, 1.5B, 7B variants)
- SafeTensors weight loading
- Autoregressive text generation
- Sampling strategies (greedy, top-k, top-p)
- KV cache for efficient generation

**Important:** For optimized inference (225+ tok/s), use the `realizar` crate:
```bash
# Recommended approach via apr CLI
cargo run --bin apr --features inference -- run model.safetensors \
    --prompt "What is 2+2?" --max-tokens 32
```

**See also:**
- [Qwen Chat](./qwen-chat.md)
- [Qwen APR Native](./qwen-apr-native.md)
- [Examples Reference](./examples-reference.md)
