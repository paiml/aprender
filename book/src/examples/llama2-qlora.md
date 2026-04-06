# LLaMA 2 QLoRA Fine-Tuning

This example demonstrates memory-efficient fine-tuning using 4-bit quantized base weights with LoRA adapters.

## Running the Example

```bash
cargo run --example llama2-finetune-qlora
```

## Code

```rust
{{#include ../../../examples/llama2/finetune_qlora.rs}}
```

## QLoRA Benefits

| Metric | LoRA | QLoRA | Savings |
|--------|------|-------|---------|
| Base weight memory | 14GB | 3.5GB | 75% |
| Total memory (7B) | 14GB | 4GB | 71% |
| Trainable params | 8M | 8M | Same |

## Quantization Details

- **Block size**: 64 elements per block
- **Bit width**: 4-bit symmetric
- **Scale factors**: FP16 per block
- **Dequantization**: On-the-fly during forward pass

## Usage

```rust
let qlora_layer = QLoRALayer::new(
    base_weights,    // Quantized to 4-bit
    4096,            // d_out
    4096,            // d_in
    64,              // rank
    128.0,           // alpha
);

// Forward pass dequantizes automatically
let output = qlora_layer.forward(&input);
```
