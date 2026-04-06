# Declarative Training Examples

Examples demonstrating YAML-based declarative training configuration.

## Basic Training Config

```yaml
# train.yaml
model:
  path: base-model.gguf

data:
  train: train.parquet
  batch_size: 8

optimizer:
  name: adamw
  lr: 0.0001
  weight_decay: 0.01

training:
  epochs: 10
  grad_clip: 1.0
```

## LoRA Fine-tuning Config

```yaml
# lora-train.yaml
model:
  path: llama-7b.gguf

data:
  train: instruction-data.parquet
  batch_size: 4

optimizer:
  name: adamw
  lr: 0.0002

lora:
  rank: 64
  alpha: 16
  dropout: 0.1
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj

training:
  epochs: 3
```

## Running from YAML

```rust
use entrenar::config::train_from_yaml;

// Single-command training
let result = train_from_yaml("train.yaml")?;
println!("Final loss: {:.4}", result.final_loss);
```

Or via CLI:

```bash
entrenar train train.yaml
```

## Running the Example

```bash
cargo run --example train_from_yaml_example
```

## See Also

- [YAML Configuration](../declarative/yaml-config.md)
- [train_from_yaml Function](../declarative/train-from-yaml.md)
- [Optimizer Builders](../declarative/optimizer-builders.md)
