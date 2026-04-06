# LLaMA 2 Training from Scratch

This example demonstrates training a LLaMA 2 architecture model from scratch with configurable sizes.

## Running the Example

```bash
cargo run --example llama2-train
```

## Code

```rust
{{#include ../../../examples/llama2/train.rs}}
```

## Model Configurations

Available configs in `examples/llama2/configs/`:

| Config | Layers | Hidden | Heads | Params |
|--------|--------|--------|-------|--------|
| 124m.toml | 12 | 768 | 12 | 124M |
| 350m.toml | 24 | 1024 | 16 | 350M |
| 774m.toml | 36 | 1280 | 20 | 774M |

## Training Output

```
🦙 LLaMA 2 Training from Scratch
================================

📋 Loading config from examples/llama2/configs/124m.toml
🔧 Building LLaMA model:
   - Layers: 12
   - Hidden size: 768
   - Heads: 12
   - Parameters: 162.4M

📚 Loading datasets:
   - Train batches: 31
   - Val batches: 31

🚀 Starting training...

Epoch 1/10
─────────────────
  Step 10: loss=2.5000, lr=3.00e-4
```

## Custom Configuration

```toml
[model]
vocab_size = 32000
hidden_size = 768
num_layers = 12
num_heads = 12
intermediate_size = 2048

[training]
batch_size = 4
learning_rate = 3e-4
epochs = 10
```
