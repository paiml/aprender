# YAML Mode Training (v1.0)

YAML Mode Training enables ML practitioners to configure, execute, and monitor model training using only YAML
configuration files. No code required.

## Core Principles (Toyota Way)

The YAML Mode Training system is built on Toyota Way manufacturing principles:

- **Muda Elimination**: No redundant code; configuration-only workflows
- **Poka-yoke**: Schema validation catches errors at parse time, not runtime
- **Jidoka**: Built-in quality with automatic checkpointing and early stopping
- **Heijunka**: Reproducible training through deterministic seeding
- **Kaizen**: Experiment tracking enables iterative refinement

## Quick Start

### Initialize a New Configuration

```bash
# Generate a minimal config
entrenar init --template minimal -o config.yaml

# Generate a LoRA fine-tuning config
entrenar init --template lora --name my-lora-exp -o lora.yaml

# Generate a QLoRA config with 4-bit quantization
entrenar init --template qlora -o qlora.yaml

# Generate a full config with all options
entrenar init --template full -o full.yaml
```

### Validate Configuration

```bash
entrenar validate config.yaml
```

### Run Training

```bash
entrenar train config.yaml
```

## Manifest Schema

### Required Fields

Every manifest must include these three fields:

```yaml
entrenar: "1.0"           # Specification version (required)
name: "my-experiment"     # Experiment identifier (required)
version: "1.0.0"          # Experiment version (required)
```

### Optional Global Fields

```yaml
description: "Fine-tune LLaMA on Alpaca dataset"
seed: 42                  # Global random seed for reproducibility
```

## Configuration Sections

### Data Configuration

```yaml
data:
  # Data source (supports local paths, hf://, pacha://, s3://)
  source: "hf://tatsu-lab/alpaca"
  format: "parquet"       # Auto-detected if omitted

  # Train/val/test split ratios
  split:
    train: 0.8
    val: 0.1
    test: 0.1
    stratify: "label"     # Column for stratified sampling
    seed: 42              # Split-specific seed

  # DataLoader settings
  loader:
    batch_size: 32
    shuffle: true
    num_workers: 4
    pin_memory: true
    drop_last: false
```

### Model Configuration

```yaml
model:
  # Model source (supports local paths, hf://, pacha://)
  source: "hf://meta-llama/Llama-2-7b"
  format: "safetensors"   # Auto-detected if omitted

  # Device placement
  device: "auto"          # auto, cpu, cuda, cuda:0, mps
  dtype: "float16"        # float32, float16, bfloat16

  # Freeze specific layers
  freeze:
    - "embed_tokens"
    - "layers.0"
```

### Optimizer Configuration

```yaml
optimizer:
  name: "adamw"           # sgd, adam, adamw, rmsprop, adagrad, lamb
  lr: 0.001               # Learning rate (required)
  weight_decay: 0.01
  betas: [0.9, 0.999]     # Adam/AdamW betas
  eps: 1e-8
```

### Scheduler Configuration

```yaml
scheduler:
  name: "cosine_annealing"  # step, cosine, linear, exponential, plateau, one_cycle

  warmup:
    steps: 1000           # Warmup steps
    start_lr: 1e-7        # Starting learning rate

  T_max: 10000            # Cosine annealing T_max
  eta_min: 1e-6           # Minimum learning rate
```

### Training Configuration

```yaml
training:
  # Duration (mutually exclusive - choose ONE)
  epochs: 10              # Number of epochs
  # max_steps: 50000      # OR maximum steps
  # duration: "2h30m"     # OR wall-clock time

  # Gradient settings
  gradient:
    accumulation_steps: 4
    clip_norm: 1.0

  # Mixed precision training
  mixed_precision:
    enabled: true
    dtype: "bfloat16"
    loss_scale: "dynamic"

  # Checkpointing
  checkpoint:
    save_every: 1000
    keep_last: 3
    save_best: true
    metric: "val_loss"
    mode: "min"

  # Early stopping (Jidoka)
  early_stopping:
    enabled: true
    metric: "val_loss"
    patience: 5
    min_delta: 0.001
    mode: "min"
```

### LoRA Configuration

```yaml
lora:
  enabled: true
  rank: 16                # Rank of low-rank matrices
  alpha: 32               # Scaling factor
  dropout: 0.05

  target_modules:         # Modules to apply LoRA to
    - q_proj
    - k_proj
    - v_proj
    - o_proj

  bias: "none"            # none, all, lora_only
  init_weights: "gaussian"
```

### QLoRA Configuration

For memory-efficient fine-tuning, add quantization to LoRA:

```yaml
lora:
  enabled: true
  rank: 64
  alpha: 128
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj

  # QLoRA specific
  quantize_base: true     # Quantize base model
  quantize_bits: 4        # 4-bit quantization
  double_quantize: true   # Double quantization
  quant_type: "nf4"       # nf4 or fp4
```

### Quantization Configuration

For post-training or quantization-aware training:

```yaml
quantize:
  enabled: true
  bits: 8                 # 2, 4, or 8
  scheme: "symmetric"     # symmetric, asymmetric, dynamic
  granularity: "per_channel"
  group_size: 128

  exclude:                # Layers to skip
    - "lm_head"
    - "embed_tokens"
```

### Monitoring Configuration

```yaml
monitoring:
  # Terminal visualization
  terminal:
    enabled: true
    refresh_rate: 100     # ms
    metrics:
      - loss
      - accuracy
      - learning_rate
    charts:
      - type: sparkline
        metric: loss
        window: 100
      - type: progress
        show_eta: true

  # Experiment tracking
  tracking:
    enabled: true
    backend: "trueno-db"  # trueno-db, mlflow, wandb, tensorboard
    project: "my-project"
    experiment: "{{ name }}-{{ timestamp }}"

  # System metrics
  system:
    enabled: true
    interval: 1000
    metrics:
      - cpu_percent
      - memory_mb
      - gpu_utilization
      - gpu_memory_mb

  # Alerts (Andon system)
  alerts:
    - condition: "loss > 10"
      action: "warn"
      message: "Loss explosion detected"
    - condition: "gpu_memory > 0.95"
      action: "halt"
      message: "GPU OOM imminent"
```

### Callbacks Configuration

```yaml
callbacks:
  - type: checkpoint
    trigger: epoch_end

  - type: lr_monitor
    trigger: step

  - type: gradient_monitor
    trigger: step
    interval: 100

  - type: sample_predictions
    trigger: epoch_end
    config:
      num_samples: 5
```

### Output Configuration

```yaml
output:
  # Output directory (supports template expressions)
  dir: "./experiments/{{ name }}/{{ timestamp }}"

  model:
    format: "safetensors"
    save_optimizer: true
    save_scheduler: true

  metrics:
    format: "parquet"
    include:
      - train_loss
      - val_loss
      - accuracy

  report:
    enabled: true
    format: "markdown"
    include_plots: true

  registry:
    enabled: true
    target: "pacha://models/{{ name }}:{{ version }}"
```

## Template Expressions

YAML Mode supports template expressions using `{{ }}` syntax:

| Expression | Description |
|-----------|-------------|
| `{{ name }}` | Experiment name |
| `{{ version }}` | Experiment version |
| `{{ timestamp }}` | ISO timestamp |
| `{{ date }}` | Date (YYYY-MM-DD) |
| `{{ seed }}` | Random seed |

## Validation (Poka-yoke)

The manifest is validated at parse time to catch errors early:

### Automatic Checks

- **Version compatibility**: Only `entrenar: "1.0"` supported
- **Required fields**: `name`, `version` must be non-empty
- **Type constraints**: Numbers, strings, arrays validated
- **Range constraints**: `lr > 0`, `batch_size >= 1`, `epochs >= 1`
- **Mutual exclusivity**: `epochs` XOR `max_steps` XOR `duration`
- **Split ratios**: Must sum to 1.0
- **Quantization bits**: Only 2, 4, or 8 allowed

### Example Validation Errors

```bash
$ entrenar validate invalid.yaml
Error: Unsupported entrenar version: 2.0. Supported versions: 1.0

$ entrenar validate bad-lr.yaml
Error: Invalid range for optimizer.lr: -0.001 (expected > 0)

$ entrenar validate bad-split.yaml
Error: Invalid split ratios: sum is 1.2 (expected 1.0)
```

## Complete Example

Here's a complete LLaMA-2 QLoRA fine-tuning configuration:

```yaml
entrenar: "1.0"
name: "llama2-alpaca-qlora"
version: "1.0.0"
description: "Fine-tune LLaMA-2-7B on Alpaca using QLoRA"
seed: 42

data:
  source: "hf://tatsu-lab/alpaca"
  split:
    train: 0.9
    val: 0.1
    seed: 42
  loader:
    batch_size: 4
    shuffle: true
    num_workers: 4

model:
  source: "hf://meta-llama/Llama-2-7b"
  device: "auto"
  dtype: "float16"

optimizer:
  name: "adamw"
  lr: 0.0002
  betas: [0.9, 0.999]
  weight_decay: 0.01

scheduler:
  name: "cosine_annealing"
  warmup:
    steps: 100
  T_max: 10000
  eta_min: 1e-6

training:
  epochs: 3
  gradient:
    accumulation_steps: 16
    clip_norm: 1.0
  mixed_precision:
    enabled: true
    dtype: "bfloat16"
  checkpoint:
    save_every: 500
    keep_last: 2
    save_best: true
  early_stopping:
    enabled: true
    patience: 3

lora:
  enabled: true
  rank: 64
  alpha: 128
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
  quantize_base: true
  quantize_bits: 4
  quant_type: "nf4"

monitoring:
  terminal:
    enabled: true
    metrics: [loss, accuracy, learning_rate]
  tracking:
    enabled: true
    backend: "trueno-db"
    project: "llama-finetune"

output:
  dir: "./experiments/llama2-alpaca/{{ timestamp }}"
  model:
    format: "safetensors"
```

## CLI Reference

### `entrenar init`

Generate a new training manifest from a template.

```bash
entrenar init [OPTIONS]

Options:
  -t, --template <TEMPLATE>  Template: minimal, lora, qlora, full [default: minimal]
  -o, --output <PATH>        Output file (stdout if not specified)
  --name <NAME>              Experiment name [default: my-experiment]
  --model <URI>              Model source path or URI
  --data <URI>               Data source path or URI
```

### `entrenar validate`

Validate a manifest without running training.

```bash
entrenar validate <CONFIG>

Options:
  --detailed                 Show detailed validation output
```

### `entrenar train`

Run training from a YAML manifest.

```bash
entrenar train <CONFIG> [OPTIONS]

Options:
  --dry-run                  Validate only, don't train
  --epochs <N>               Override epochs
  --lr <RATE>                Override learning rate
  --batch-size <N>           Override batch size
```

## Programmatic Usage

You can also use YAML Mode from Rust code:

```rust
use entrenar::yaml_mode::{load_manifest, validate_manifest, Template, generate_yaml};

// Load and validate a manifest
let manifest = load_manifest(Path::new("config.yaml"))?;

// Generate from template
let yaml = generate_yaml(Template::Qlora, "my-exp", Some("model.safetensors"), None);

// Manual validation
validate_manifest(&manifest)?;
```

## References

YAML Mode Training is informed by:

1. Liker, J. K. (2004). *The Toyota Way*. McGraw-Hill.
2. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
3. Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.
4. Sculley, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems.

See [docs/specifications/yaml-mode-train.md](../../../docs/specifications/yaml-mode-train.md) for the complete
specification with all 20 peer-reviewed citations.
