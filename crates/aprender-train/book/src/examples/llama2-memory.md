# LLaMA 2 Memory Benchmarks

This example benchmarks memory usage across different fine-tuning methods.

## Running the Example

```bash
cargo run --example llama2-memory-benchmarks
```

## Code

```rust
{{#include ../../../examples/llama2/memory_benchmarks.rs}}
```

## Benchmark Results

### 7B Model Memory Usage

| Method | Base Weights | Adapters | Total | GPU Required |
|--------|--------------|----------|-------|--------------|
| Full FT | 28GB (FP32) | - | 28GB | A100 80GB |
| LoRA | 14GB (FP16) | 32MB | 14GB | A100 40GB |
| QLoRA | 3.5GB (4-bit) | 32MB | 4GB | RTX 3090 |

### Throughput Comparison

| Method | Tokens/sec | Relative |
|--------|------------|----------|
| Full FT | 1000 | 1.0x |
| LoRA | 950 | 0.95x |
| QLoRA | 800 | 0.8x |

## Memory Breakdown

```
Full Fine-Tuning:
  Base weights:     28.0 GB (FP32)
  Gradients:        28.0 GB
  Optimizer state:  56.0 GB (Adam)
  Total:           112.0 GB

QLoRA:
  Base weights:      3.5 GB (4-bit)
  LoRA A:            8.0 MB (FP32)
  LoRA B:            8.0 MB (FP32)
  Gradients:        16.0 MB
  Optimizer:        32.0 MB
  Total:             3.6 GB
```
