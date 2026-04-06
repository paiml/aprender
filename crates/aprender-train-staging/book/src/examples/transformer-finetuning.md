# Transformer Fine-Tuning

This example demonstrates fine-tuning a large language model (Qwen2.5-Coder-1.5B) using entrenar's transformer training
pipeline.

## Overview

Fine-tuning a pre-trained transformer model involves:
1. Loading pre-trained weights from SafeTensors format
2. Preparing a domain-specific dataset (in this case, Rust documentation)
3. Running gradient descent to adapt the model to the new domain

## Configuration

Create a YAML configuration file:

```yaml
# qwen-rustdoc.yaml
model:
  path: /path/to/Qwen2.5-Coder-1.5B-Instruct
  mode: transformer
  config: /path/to/config.json

data:
  train: /path/to/train_lm.jsonl
  tokenizer: /path/to/tokenizer.json
  batch_size: 1
  seq_len: 64
  input_column: text

optimizer:
  name: adamw
  lr: 0.0001
  weight_decay: 0.01

training:
  epochs: 20
  mode: causal_lm
  gradient_accumulation: 1
  warmup_steps: 5
  grad_clip: 1.0
  output_dir: ./checkpoints
  seed: 42
```

## Running the Training

```bash
entrenar train qwen-rustdoc.yaml
```

## Expected Output

```
Entrenar: Training from qwen-rustdoc.yaml
✓ Config loaded and validated (Transformer mode)
  Model: /path/to/Qwen2.5-Coder-1.5B-Instruct
  Optimizer: adamw (lr=0.0001)
  Batch size: 1
  Epochs: 20
  Training mode: CausalLm

Loading model weights...
  Detected architecture: Qwen2
  Loaded 338 weight tensors
✓ Loaded pre-trained weights successfully

Starting transformer training...

Epoch  1/20: loss=16.655378, perplexity=17113502.00
Epoch  2/20: loss=13.596731, perplexity=803499.00
...
Epoch 20/20: loss=0.602336, perplexity=1.83

✓ Transformer training complete
  Final loss: 0.501395
  Best loss: 0.472123
  Steps completed: 200
```

## Performance Metrics

The training pipeline includes built-in profiling:

```
╔══════════════════════════════════════════════════════════════╗
║       ENTRENAR TRACE REPORT                                   ║
╚══════════════════════════════════════════════════════════════╝
Total Measured Time: 428.91s
────────────────────────────────────────────────────────────────
Step            | Count    | Duration        | % Time
────────────────────────────────────────────────────────────────
Matmul          | 174200   | 255.42s         |   59.55%
Transpose       | 67600    | 172.36s         |   40.19%
Alloc           | 174200   | 1.13s           |    0.26%
────────────────────────────────────────────────────────────────
```

## Key Points

- **CUDA Acceleration**: The matmul operations use realizar's CUDA backend when available
- **Memory Efficiency**: The pipeline uses careful memory management for 1.5B+ models
- **Convergence Tracking**: Loss and perplexity are tracked per epoch
- **Checkpointing**: Final model metadata is saved to the output directory

## Related Examples

- [LoRA Fine-Tuning](lora-finetuning.md) - Parameter-efficient fine-tuning
- [QLoRA Example](qlora-example.md) - Quantized LoRA for memory-constrained setups
- [Distillation](distillation.md) - Knowledge transfer between models
