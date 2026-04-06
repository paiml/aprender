# Research Artifact Management

This example demonstrates research experiment tracking and artifact management.

## Running the Example

```bash
cargo run --example research
```

## Code

```rust
{{#include ../../../examples/research.rs}}
```

## Features

- **Experiment tracking** - Log hyperparameters, metrics, artifacts
- **Reproducibility** - Capture git commit, random seeds, environment
- **Artifact storage** - Models, checkpoints, logs
- **Comparison** - Compare runs across experiments

## Usage

```rust
let mut experiment = Experiment::new("ablation-study")
    .with_tags(["lora", "7b", "alpaca"])
    .with_params(params);

experiment.log_metric("loss", loss_value, step);
experiment.log_artifact("model.safetensors", &model_bytes);

experiment.finish();
```

## Directory Structure

```
experiments/
├── ablation-study/
│   ├── run-001/
│   │   ├── params.yaml
│   │   ├── metrics.json
│   │   ├── artifacts/
│   │   │   └── model.safetensors
│   │   └── logs/
│   └── run-002/
└── baseline/
```
