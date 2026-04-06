# Fine-Tune Real: End-to-End Rust Test Generation

The `finetune_real` example demonstrates real-world fine-tuning of a pre-trained language model (Qwen2.5-Coder-0.5B) for
Rust test generation.

## Overview

This example showcases:
- Loading pre-trained SafeTensors models from HuggingFace
- Full fine-tuning vs LoRA fine-tuning comparison
- CUDA-accelerated training with real-time TUI monitoring
- Cross-entropy loss with causal language modeling
- Learning rate scheduling with cosine decay

## Running the Example

```bash
# Run training (producer mode)
cargo run --release --example finetune_real

# Monitor training in separate terminal (consumer mode)
cargo run --release --example finetune_real --features nvml -- --monitor --experiment ./experiments/finetune-real
```

## Architecture

### Training Pipeline

1. **Model Loading**: Downloads Qwen2.5-Coder-0.5B from HuggingFace cache
2. **Tokenization**: Pre-tokenizes Rust code samples with the Qwen2 tokenizer
3. **Experiment 1**: Full fine-tuning (all LM head weights trainable)
4. **Experiment 2**: LoRA fine-tuning (frozen base, rank-16 adapters)

### TUI Monitor

The training produces a real-time TUI showing:
- Epoch/step progress with colored bars
- Loss curve with Braille sparklines
- Epoch history table with labeled columns
- GPU telemetry (when `--features nvml` enabled)
- ETA and throughput metrics

## Example Output

```
═════════════════════════════════════════════════════════════════════════════════════
ENTRENAR  ● Running  00:02:26 Qwen2.5-Coder-0.5B  121 tok/s
─────────────────────────────────────────────────────────────────────────────────────
Epoch 18/18 ████████████████████ 100%    Step 16/16 ████████████████████ 100%
Loss 10.9077 ↑  Best 5.7973  LR 0.000010  Grad 25.71  ETA 00:00:00
─────────────────────────────────────────────────────────────────────────────────────
Loss History: [5.80 - 12.52] (200 steps)
─────────────────────────────────────────────────────────────────────────────────────
GPU: N/A
─────────────────────────────────────────────────────────────────────────────────────
Epoch      Loss       Min       Max          LR  Trend
    8    9.1035    6.0082   12.5236    0.000066      →
    9    9.0758    5.9802   12.2845    0.000031      →
   10    8.9277    5.9727   12.4896    0.000013      →
   11    8.7432    5.9697   12.3900    0.000010      ↓
   12    8.8717    5.9674   11.9185    0.000010      →
   13    8.4931    5.9635   10.9077    0.000010      ↓
  ... 7 earlier epochs
─────────────────────────────────────────────────────────────────────────────────────
Config: AdamW  Batch: 1  Checkpoint: ./experiments/finetune-real
═════════════════════════════════════════════════════════════════════════════════════
```

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | Qwen2.5-Coder-0.5B | 500M parameter code model |
| Epochs (Full) | 3 | Full fine-tuning epochs |
| Epochs (LoRA) | 15 | LoRA fine-tuning epochs |
| Learning Rate | 0.0002 (Full), 0.0006 (LoRA) | Initial LR with cosine decay |
| LoRA Rank | 16 | Low-rank adapter dimension |
| LoRA Alpha | 32 | Scaling factor |
| Batch Size | 1 | Per-sample training |

## Checkpoint Output

Checkpoints are saved as HuggingFace-complete directories containing:

| File | Description |
|------|-------------|
| `model.safetensors` | Classifier head + LoRA adapter weights |
| `model.apr` | Same weights in APR format |
| `metadata.json` | Epoch metrics (loss, accuracy, LR) |
| `config.json` | HF model architecture config |
| `adapter_config.json` | PEFT LoRA adapter config |
| `tokenizer.json` | BPE tokenizer (copied from base model) |

Publish directly to HuggingFace:

```bash
apr publish ./checkpoints/best/ paiml/my-model
```

## IPC State File

Training state is persisted to `experiments/finetune-real/training_state.json` for:
- External monitoring tools
- Crash recovery
- Experiment tracking

See [TUI Monitor Documentation](../monitor/dashboard.md) for details.
