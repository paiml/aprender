# First Training

Fine-tune a model with LoRA:

```bash
# Fine-tune with LoRA adapter
apr finetune model.gguf \
  --adapter lora \
  --rank 64 \
  --data train.jsonl \
  --output adapter.safetensors

# Apply the adapter
apr merge model.gguf adapter.safetensors -o fine-tuned.gguf
```

## Training Pipeline

For full training workflows:

```bash
# Plan (dry-run, shows what will happen)
apr train plan config.yaml

# Execute
apr train apply config.yaml

# Monitor in real-time
apr monitor
```

## Knowledge Distillation

```bash
apr distill \
  --teacher large-model.gguf \
  --student small-model.gguf \
  --data train.jsonl \
  -o distilled.gguf
```
