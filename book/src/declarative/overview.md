# Declarative Training Overview

**Declarative training** allows you to define complete training workflows in YAML configuration files (Ludwig-style).

## The Problem

Training code often mixes:
- Model architecture definitions
- Hyperparameter configurations
- Data loading logic
- Training loop boilerplate

**Result**: Hard to experiment, compare runs, or share configurations

## The Solution

Define training in YAML, execute with one function call:

```yaml
# config.yaml
model:
  path: models/llama-7b.gguf
data:
  train: data/train.parquet
  batch_size: 4
optimizer:
  name: adamw
  lr: 0.0001
  beta1: 0.9
  beta2: 0.999
  weight_decay: 0.01
training:
  epochs: 3
  grad_clip: 1.0
  output_dir: ./checkpoints
```

**Single-command training:**
```rust
use entrenar::config::train_from_yaml;

train_from_yaml("config.yaml")?;  // Complete workflow
```

**From `src/config/train.rs`**

## Configuration Schema

### Model Section

```yaml
model:
  path: path/to/model.gguf  # Model file path (required)
```

Currently supports:
- `.gguf` files (placeholder for Realizar integration)
- Placeholder models for testing

### Data Section

```yaml
data:
  train: path/to/train.parquet  # Training data path (required)
  batch_size: 4                  # Batch size (required)
```

Currently supports:
- `.parquet` files (placeholder for data loading)
- Synthetic data for examples

### Optimizer Section

```yaml
optimizer:
  name: adamw       # Optimizer type: sgd, adam, adamw (required)
  lr: 0.0001        # Learning rate (required)
  # Optional parameters:
  momentum: 0.9     # For SGD
  beta1: 0.9        # For Adam/AdamW
  beta2: 0.999      # For Adam/AdamW
  eps: 1e-8         # For Adam/AdamW
  weight_decay: 0.01  # For AdamW
```

**Supported optimizers:**
- `sgd` → Creates `SGD` optimizer
- `adam` → Creates `Adam` optimizer
- `adamw` → Creates `AdamW` optimizer

### Training Section

```yaml
training:
  epochs: 3                    # Number of training epochs (required)
  grad_clip: 1.0              # Gradient clipping threshold (optional)
  output_dir: ./checkpoints   # Where to save trained model (required)
```

## Optimizer Builders

From `src/config/builder.rs`:

```rust
pub fn build_optimizer(spec: &OptimSpec) -> Result<Box<dyn Optimizer>> {
    match spec.name.to_lowercase().as_str() {
        "sgd" => {
            let momentum = spec.params.get("momentum")
                .and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            Ok(Box::new(SGD::new(spec.lr, momentum)))
        }
        "adam" => {
            let beta1 = spec.params.get("beta1")
                .and_then(|v| v.as_f64()).unwrap_or(0.9) as f32;
            let beta2 = spec.params.get("beta2")
                .and_then(|v| v.as_f64()).unwrap_or(0.999) as f32;
            let eps = spec.params.get("eps")
                .and_then(|v| v.as_f64()).unwrap_or(1e-8) as f32;
            Ok(Box::new(Adam::new(spec.lr, beta1, beta2, eps)))
        }
        "adamw" => {
            // Similar with weight_decay parameter
            Ok(Box::new(AdamW::new(spec.lr, beta1, beta2, eps, weight_decay)))
        }
        name => Err(Error::ConfigError(format!("Unknown optimizer: {}", name))),
    }
}
```

## Workflow

The `train_from_yaml()` function orchestrates:

1. **Load config** from YAML file
2. **Validate config** (check paths exist, validate parameters)
3. **Build model** from model path
4. **Build optimizer** from optimizer spec
5. **Setup trainer** with training config
6. **Run training loop** for specified epochs
7. **Save trained model** to output directory

```rust
// From src/config/train.rs
pub fn train_from_yaml<P: AsRef<Path>>(config_path: P) -> Result<()> {
    // 1. Load and validate config
    let yaml_content = fs::read_to_string(config_path.as_ref())?;
    let spec: TrainSpec = serde_yaml::from_str(&yaml_content)?;
    validate_config(&spec)?;

    // 2. Build components
    let model = build_model(&spec)?;
    let optimizer = build_optimizer(&spec.optimizer)?;

    // 3. Setup trainer
    let mut train_config = TrainConfig::new().with_log_interval(100);
    if let Some(clip) = spec.training.grad_clip {
        train_config = train_config.with_grad_clip(clip);
    }

    let mut trainer = Trainer::new(
        model.parameters.into_iter().map(|(_, t)| t).collect(),
        optimizer,
        train_config,
    );
    trainer.set_loss(Box::new(MSELoss));

    // 4. Training loop
    for epoch in 0..spec.training.epochs {
        let avg_loss = trainer.train_epoch(batches.clone(), |x| x.clone());
        println!("Epoch {}/{}: loss={:.6}", epoch + 1, spec.training.epochs, avg_loss);
    }

    // 5. Save trained model
    let output_path = spec.training.output_dir.join("final_model.json");
    save_model(&final_model, &output_path, &save_config)?;

    Ok(())
}
```

## Example Usage

From `examples/train_from_yaml_example.rs`:

```rust
use entrenar::config::train_from_yaml;
use std::fs;

fn main() {
    // Ensure output directory exists
    fs::create_dir_all("./output").expect("Failed to create output directory");

    // Run training from YAML config
    match train_from_yaml("examples/config.yaml") {
        Ok(()) => {
            println!("=== Training Successful ===");
            println!("\nTrained model saved to: ./output/final_model.json");
        }
        Err(e) => {
            eprintln!("Training failed: {}", e);
            std::process::exit(1);
        }
    }
}
```

**Run with:**
```bash
cargo run --example train_from_yaml_example
```

## Validation

The `validate_config()` function checks:
- ✅ Model path exists
- ✅ Training data path exists
- ✅ Learning rate > 0
- ✅ Batch size > 0
- ✅ Epochs > 0
- ✅ Output directory is valid

**From `src/config/train.rs`**

## Tests

**5 builder tests** in `src/config/builder.rs`:
- SGD builder creates correct optimizer
- Adam builder extracts beta1/beta2/eps
- AdamW builder extracts weight_decay
- Unknown optimizer name returns error
- Missing required parameters handled

## Benefits

✅ **Reproducibility**: Config files capture entire training setup
✅ **Experimentation**: Easy to modify hyperparameters
✅ **Sharing**: Share configs instead of code
✅ **Version control**: Git-friendly YAML files
✅ **Documentation**: Self-documenting training runs

## Future Enhancements (v0.2.0+)

- Real GGUF model loading (via Realizar)
- Real Parquet data loading
- Support for validation sets
- Checkpointing during training
- TensorBoard logging

## Next Steps

- [YAML Configuration](./yaml-config.md) - Full schema reference
- [train_from_yaml Function](./train-from-yaml.md) - Implementation details
- [Optimizer Builders](./optimizer-builders.md) - Builder pattern
- [Examples](../examples/train-from-yaml.md) - Real examples

## Implementation

All declarative training code in `src/config/`:
- `train.rs` - train_from_yaml() function, TrainSpec, validation
- `builder.rs` - build_optimizer(), build_model()
- `mod.rs` - Public API exports
