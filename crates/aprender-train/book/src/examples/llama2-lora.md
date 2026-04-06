# LLaMA 2 LoRA Fine-Tuning

This example demonstrates parameter-efficient fine-tuning of LLaMA 2 using LoRA adapters.

## Running the Example

```bash
cargo run --example llama2-finetune-lora
```

## Code

```rust
{{#include ../../../examples/llama2/finetune_lora.rs}}
```

## LoRA Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Rank | 16-64 | Low-rank dimension |
| Alpha | 32-128 | Scaling factor |
| Target modules | q_proj, v_proj | Attention projections |
| Dropout | 0.05 | LoRA dropout |

## Memory Comparison

| Method | Memory | Trainable Params |
|--------|--------|------------------|
| Full fine-tuning | 28GB | 7B (100%) |
| LoRA (r=64) | 14GB | 8M (0.1%) |

## Target Module Selection

```rust
let lora_config = LoRAConfig::new()
    .with_rank(64)
    .with_alpha(128)
    .with_targets(["q_proj", "k_proj", "v_proj", "o_proj"]);
```
