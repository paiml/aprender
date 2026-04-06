# Command-Line Interface

Entrenar provides a powerful CLI for training, validation, quantization, and model merging—all without writing code.

## Installation

```bash
cargo install entrenar
```

Or build from source:

```bash
cargo build --release
# Binary at target/release/entrenar
```

## Quick Reference

```bash
entrenar train config.yaml              # Train from YAML config
entrenar validate config.yaml           # Validate config without training
entrenar info config.yaml               # Display config information
entrenar quantize model.json -o q4.json # Quantize a model
entrenar merge a.json b.json -o out.json # Merge models
```

## Global Options

| Flag | Description |
|------|-------------|
| `-v, --verbose` | Enable verbose output |
| `-q, --quiet` | Suppress all output except errors |
| `--help` | Print help information |
| `--version` | Print version |

## Commands

### train

Train a model from a YAML configuration file.

```bash
entrenar train <CONFIG> [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `CONFIG` | Path to YAML configuration file |

**Options:**

| Option | Description |
|--------|-------------|
| `-o, --output-dir <DIR>` | Override output directory |
| `-r, --resume <PATH>` | Resume training from checkpoint |
| `-e, --epochs <N>` | Override number of epochs |
| `-b, --batch-size <N>` | Override batch size |
| `-l, --lr <RATE>` | Override learning rate |
| `--dry-run` | Validate config but don't train |
| `--save-every <N>` | Save checkpoint every N steps |
| `--log-every <N>` | Log metrics every N steps |
| `--seed <N>` | Random seed for reproducibility |

**Examples:**

```bash
# Basic training
entrenar train config.yaml

# Override hyperparameters
entrenar train config.yaml --epochs 50 --lr 0.0001 --batch-size 32

# Resume from checkpoint
entrenar train config.yaml --resume checkpoints/epoch_10.json

# Dry run to validate config
entrenar train config.yaml --dry-run

# Full example with all options
entrenar train config.yaml \
  --output-dir ./experiments/run1 \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-4 \
  --save-every 1000 \
  --log-every 100 \
  --seed 42 \
  --verbose
```

### validate

Validate a configuration file without training.

```bash
entrenar validate <CONFIG> [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `-d, --detailed` | Show detailed validation report |

**Examples:**

```bash
# Quick validation
entrenar validate config.yaml

# Detailed report
entrenar validate config.yaml --detailed
```

**Output:**

```
✓ Configuration is valid
  Model: llama-7b
  Optimizer: adamw (lr=0.0001)
  Epochs: 10
  Batch size: 8
  LoRA: rank=64, alpha=16
```

### info

Display information about a configuration.

```bash
entrenar info <CONFIG> [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `-f, --format <FORMAT>` | Output format: `text`, `json`, `yaml` (default: `text`) |

**Examples:**

```bash
# Human-readable output
entrenar info config.yaml

# JSON for scripting
entrenar info config.yaml --format json

# YAML output
entrenar info config.yaml --format yaml
```

### quantize

Quantize a model to reduce size and memory footprint.

```bash
entrenar quantize <MODEL> -o <OUTPUT> [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `MODEL` | Path to model file |

**Options:**

| Option | Description |
|--------|-------------|
| `-o, --output <PATH>` | Output path for quantized model (required) |
| `-b, --bits <N>` | Quantization bits: `4` or `8` (default: `4`) |
| `-m, --method <METHOD>` | Method: `symmetric`, `asymmetric` (default: `symmetric`) |
| `--per-channel` | Use per-channel quantization |
| `--calibration-data <PATH>` | Path to calibration data for PTQ |

**Examples:**

```bash
# 4-bit symmetric quantization (default)
entrenar quantize model.json -o model_q4.json

# 8-bit asymmetric quantization
entrenar quantize model.json -o model_q8.json --bits 8 --method asymmetric

# Per-channel with calibration
entrenar quantize model.json -o model_q4.json \
  --per-channel \
  --calibration-data calibration.json
```

**Quantization Methods:**

| Method | Description | Use Case |
|--------|-------------|----------|
| `symmetric` | Zero-centered quantization | General purpose, faster inference |
| `asymmetric` | Full range quantization | Better for non-symmetric weight distributions |

### merge

Merge multiple models using various algorithms.

```bash
entrenar merge <MODELS>... -o <OUTPUT> [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `MODELS` | Two or more model paths to merge |

**Options:**

| Option | Description |
|--------|-------------|
| `-o, --output <PATH>` | Output path for merged model (required) |
| `-m, --method <METHOD>` | Merge method (default: `ties`) |
| `-w, --weight <FLOAT>` | Interpolation weight for SLERP (0.0-1.0) |
| `-d, --density <FLOAT>` | Density threshold for TIES/DARE |
| `--weights <LIST>` | Comma-separated weights for weighted average |

**Merge Methods:**

| Method | Description | Parameters |
|--------|-------------|------------|
| `ties` | Trim, Elect Sign, Merge | `--density` (default: 0.2) |
| `dare` | Drop And REscale | `--density` (default: 0.5) |
| `slerp` | Spherical Linear Interpolation | `--weight` (default: 0.5) |
| `average` | Weighted average | `--weights` |

**Examples:**

```bash
# TIES merge (default)
entrenar merge model_a.json model_b.json -o merged.json

# SLERP with custom weight
entrenar merge model_a.json model_b.json -o merged.json \
  --method slerp --weight 0.7

# DARE with density
entrenar merge model_a.json model_b.json -o merged.json \
  --method dare --density 0.3

# Weighted average of 3 models
entrenar merge a.json b.json c.json -o merged.json \
  --method average --weights "0.5,0.3,0.2"
```

## Configuration File

The CLI works with YAML configuration files:

```yaml
# config.yaml
model:
  path: llama-7b.gguf
  type: llama

data:
  train: train.parquet
  validation: val.parquet
  batch_size: 8

optimizer:
  name: adamw
  lr: 0.0001
  weight_decay: 0.01

training:
  epochs: 10
  output_dir: ./checkpoints
  save_interval: 1000

lora:
  enabled: true
  rank: 64
  alpha: 16
  target_modules: [q_proj, k_proj, v_proj, o_proj]

quantization:
  enabled: false
  bits: 4
```

See [YAML Configuration](../declarative/yaml-config.md) for full schema.

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Configuration error |
| `2` | Runtime error |
| `3` | I/O error |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ENTRENAR_LOG` | Log level: `error`, `warn`, `info`, `debug`, `trace` |
| `ENTRENAR_CONFIG` | Default config file path |
| `CUDA_VISIBLE_DEVICES` | GPU device selection |

## Shell Completion

Generate shell completions:

```bash
# Bash
entrenar --generate-completion bash > ~/.local/share/bash-completion/completions/entrenar

# Zsh
entrenar --generate-completion zsh > ~/.zfunc/_entrenar

# Fish
entrenar --generate-completion fish > ~/.config/fish/completions/entrenar.fish
```

## Examples

### Complete Training Workflow

```bash
# 1. Validate configuration
entrenar validate config.yaml --detailed

# 2. Dry run to check setup
entrenar train config.yaml --dry-run

# 3. Start training
entrenar train config.yaml --verbose

# 4. Resume if interrupted
entrenar train config.yaml --resume checkpoints/latest.json

# 5. Quantize final model
entrenar quantize checkpoints/final.json -o model_q4.json --bits 4
```

### Model Merging Pipeline

```bash
# Train specialist models
entrenar train math_config.yaml
entrenar train code_config.yaml
entrenar train writing_config.yaml

# Merge specialists
entrenar merge \
  checkpoints/math_final.json \
  checkpoints/code_final.json \
  checkpoints/writing_final.json \
  -o merged_expert.json \
  --method ties \
  --density 0.3

# Quantize merged model
entrenar quantize merged_expert.json -o expert_q4.json
```

## Programmatic Usage

The CLI types are also available programmatically:

```rust
use entrenar::config::cli::{Cli, Command, TrainArgs};
use clap::Parser;

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Train(args) => {
            println!("Training with config: {:?}", args.config);
        }
        Command::Validate(args) => {
            // ...
        }
        _ => {}
    }
}
```

## Next Steps

- [YAML Configuration Schema](../declarative/yaml-config.md)
- [Training Best Practices](../best-practices/debugging-training.md)
- [Quantization Guide](../quantization/qlora-overview.md)
- [Model Merging](../merging/overview.md)
