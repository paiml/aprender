# Quick Generate - Real Model Inference

The `quick_generate` example demonstrates real GGUF model inference with actual LLMs like TinyLlama, Qwen2, Phi-2, and others.

## Running the Example

```bash
# Use default model path
cargo run --example quick_generate --release

# Specify custom model
cargo run --example quick_generate --release -- /path/to/model.gguf
```

## Supported Models

| Model | Tokenizer | Tested |
|-------|-----------|--------|
| TinyLlama-1.1B | SentencePiece | Yes |
| Qwen2.5-0.5B | GPT-2 | Yes |
| Phi-2 | GPT-2 | Yes |
| LLaMA-2-7B | SentencePiece | Yes |

## Example Output

```
$ cargo run --example quick_generate --release -- ~/models/tinyllama.Q4_0.gguf

Prompt: 'Once upon a time'
Tokens: [9038, 2501, 263, 931]
, I was a dream, a dreaming of a dream...

Full tokens: [9038, 2501, 263, 931, 29892, 306, 471, 263, 12561, 29892, 263, 12561, 292, 310, 263, 12561, 856, 856, 856, 856, 856]
```

With Qwen2.5:

```
$ cargo run --example quick_generate --release -- ~/models/qwen2.5-0.5b.Q4_0.gguf

Prompt: 'Once upon a time'
Tokens: [12522, 5193, 264, 882]
, a group of friends decided to go on a trip to a nearby city. They packed their bags

Full tokens: [12522, 5193, 264, 882, 11, 264, 1880, 315, 4780, 6587, 311, ...]
```

## How It Works

### 1. Memory-Mapped Loading

The example uses `MappedGGUFModel` for efficient memory-mapped file access:

```rust
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
let model = OwnedQuantizedModel::from_mapped(&mapped).unwrap();
```

This approach:
- Loads only needed tensor data on-demand
- Reduces memory footprint
- Enables fast startup times

### 2. Automatic Tokenizer Detection

The tokenizer type is automatically detected from model metadata:

```rust
// Metadata key: tokenizer.ggml.model
// - "llama" or "sentencepiece" -> SentencePiece style (▁ = U+2581)
// - "gpt2" -> GPT-2 style (Ġ = U+0120)
```

This ensures correct encoding for different model families.

### 3. Greedy Token Generation

The example uses greedy sampling (always pick highest probability token):

```rust
let mut all_tokens = tokens;
for _ in 0..20 {
    let logits = model.forward(&all_tokens).unwrap();

    // Greedy: pick highest logit
    let (best_idx, _) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    all_tokens.push(best_idx as u32);
}
```

### 4. Token Decoding

Tokens are decoded to text with proper space handling:

```rust
let tok_str = &vocab[best_idx];
// Handle both GPT-2 style (Ġ) and SentencePiece style (▁) space tokens
print!("{}", tok_str.replace("▁", " ").replace('\u{0120}', " "));
```

## Quantization Support

The example supports all standard GGUF quantization formats:

| Format | Bits | Block Size | Status |
|--------|------|------------|--------|
| F32 | 32 | - | Supported |
| F16 | 16 | - | Supported |
| Q4_0 | 4 | 32 | Supported |
| Q4_1 | 4 | 32 | Supported |
| Q5_0 | 5 | 32 | Supported |
| Q8_0 | 8 | 32 | Supported |
| Q4_K | 4 | 256 | Supported |
| Q5_K | 5 | 256 | Supported |
| Q6_K | 6 | 256 | Supported |

## Performance

On a typical system:

| Model | Quantization | Tokens/sec |
|-------|--------------|------------|
| TinyLlama-1.1B | Q4_0 | 12-15 |
| Qwen2.5-0.5B | Q4_0 | 15-20 |
| Phi-2 2.7B | Q4_0 | 8-12 |

Use `--release` for 10-20x faster inference compared to debug builds.

## Customization

To modify generation parameters, edit the example:

```rust
// Change number of tokens to generate
for _ in 0..50 {  // Generate 50 tokens instead of 20

// Change prompt
let prompt = "The meaning of life is";
```

## Troubleshooting

### Model Not Found

```
Error: Failed to load model
```

Ensure the model path exists and is a valid GGUF file:

```bash
# Check file exists
ls -la /path/to/model.gguf

# Verify GGUF magic bytes
xxd /path/to/model.gguf | head -1
# Should show: 0000000: 4747 5546  (GGUF)
```

### NaN/Inf in Output

If you see garbled output or PAD tokens, ensure you're using the latest realizar with Q8_0 fixes:

```bash
git pull
cargo clean
cargo build --release
```

### Slow Performance

Always use `--release` for benchmarking:

```bash
# Debug (slow)
cargo run --example quick_generate

# Release (fast)
cargo run --example quick_generate --release
```

## See Also

- [Examples Reference](./examples-reference.md) - All examples
- [GGUF Format](../formats/gguf.md) - GGUF specification
- [Quantization](../quantization/what-is-quantization.md) - Quantization overview
- [Generation Strategies](../generation/sampling.md) - Other sampling methods
