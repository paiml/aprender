# Checkpointing

`ClassifyTrainer` saves HuggingFace-complete checkpoints automatically during training.
Each checkpoint is a self-contained directory that can be published directly to
HuggingFace Hub via `apr publish`.

## Checkpoint Contents

| File | Description |
|------|-------------|
| `model.safetensors` | Classifier head + LoRA adapter weights (SafeTensors format) |
| `model.apr` | Same weights in APR format (sovereign, used by realizador) |
| `metadata.json` | Epoch metrics: loss, accuracy, learning rate, throughput |
| `config.json` | HF model architecture config (hidden_size, num_layers, etc.) |
| `adapter_config.json` | PEFT LoRA config (rank, alpha, target_modules, task_type) |
| `tokenizer.json` | BPE tokenizer (copied from base model, when using `from_pretrained`) |

## When Checkpoints Are Saved

The trainer saves checkpoints in two cases:

1. **Best checkpoint** (`checkpoints/best/`): Saved whenever validation loss improves.
   This is the checkpoint you typically publish.

2. **Periodic checkpoints** (`checkpoints/epoch-N/`): Saved every `save_every` epochs
   (default: every 5 epochs).

## HuggingFace Compatibility

Checkpoints are designed for direct use with the HuggingFace ecosystem:

```python
# Load in Python with PEFT
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

base = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
model = PeftModel.from_pretrained(base, "./checkpoints/best/")
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/best/")
```

## Two-Command Publish Workflow

```bash
# 1. Train (checkpoints saved automatically)
apr finetune --task classify \
    --model-size 0.5B /path/to/base-model \
    --data corpus.jsonl \
    --epochs 3 \
    -o ./checkpoints/

# 2. Publish best checkpoint to HuggingFace
apr publish ./checkpoints/best/ paiml/shell-safety-classifier
```

## Configuration

```rust
use entrenar::finetune::classify_trainer::TrainingConfig;

let config = TrainingConfig {
    checkpoint_dir: PathBuf::from("./checkpoints"),
    save_every: 5,           // Save every 5 epochs
    early_stopping_patience: 10,
    ..TrainingConfig::default()
};
```

## GPU Training

When GPU training is active (CUDA feature), `save_checkpoint()` automatically
synchronizes GPU-updated transformer weights to CPU before saving. This ensures
checkpoints contain all trained parameters regardless of compute backend.
