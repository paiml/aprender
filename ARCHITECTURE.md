# Entrenar Architecture

## Overview

Entrenar is a tape-based autograd training library with CUDA GPU support,
built for the PAIML sovereign AI stack.

## Module Architecture

```
entrenar
├── autograd/          # Tape-based automatic differentiation
│   └── ops/           # Differentiable operations (matmul, softmax, etc.)
├── train/             # Training loop infrastructure
│   ├── transformer_trainer/  # Transformer-specific training
│   │   ├── cuda_trainer     # GPU-resident CUDA training
│   │   ├── distributed_trainer  # DDP data parallelism
│   │   └── grad_accumulator    # Per-block gradient accumulation
│   ├── loss/          # Loss functions (CE, MSE, Huber)
│   └── metrics/       # Evaluation metrics
├── config/            # YAML configuration loading and validation
│   ├── schema         # Serde-derived config structs
│   └── validate       # JSON Schema validation (AI-05)
├── transformer/       # Transformer model architecture
├── lora/              # LoRA/QLoRA adapters
├── quant/             # Quantization (QAT/PTQ)
├── merge/             # Model merging (TIES/DARE/SLERP)
├── distill/           # Knowledge distillation
├── sovereign/         # Air-gapped deployment, data governance
├── finetune/          # Fine-tuning pipeline with distributed comms
│   ├── worker_client  # TCP-based gradient exchange client
│   └── gradient_server # Coordinator for AllReduce
├── io/                # Model I/O (SafeTensors, JSON, YAML)
└── eval/              # Evaluation framework with drift detection
```

## Pipeline Architecture

```
YAML Config → Schema Validation → Model Loading → Data Loading
    → Training Loop (CPU or CUDA) → Checkpoint Saving → Export
```

## No Correction Cascades

The architecture uses direct training, not correction models.
Each model is independently trained from data — no model exists
solely to correct another model's errors.
