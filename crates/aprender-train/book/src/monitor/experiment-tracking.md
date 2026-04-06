# Experiment Tracking

Entrenar provides a comprehensive experiment tracking system that integrates with distributed tracing for observability
across training runs.

## Overview

The experiment tracking system consists of three main components:

1. **ExperimentStorage** - A trait for persisting experiment data
2. **Run** - A struct that wraps runs with tracing integration
3. **TracingConfig** - Configuration for distributed tracing via Renacer

## Storage Backends

### InMemoryStorage

For testing and WASM environments:

```rust
use entrenar::storage::{ExperimentStorage, InMemoryStorage, RunStatus};

let mut storage = InMemoryStorage::new();

// Create an experiment
let exp_id = storage.create_experiment("my-experiment", None).unwrap();

// Create and start a run
let run_id = storage.create_run(&exp_id).unwrap();
storage.start_run(&run_id).unwrap();

// Log metrics
storage.log_metric(&run_id, "loss", 0, 0.5).unwrap();
storage.log_metric(&run_id, "loss", 1, 0.4).unwrap();

// Complete the run
storage.complete_run(&run_id, RunStatus::Success).unwrap();
```

### TruenoBackend (Production)

For production use with TruenoDB persistence (requires `monitor` feature):

```rust
use entrenar::storage::{ExperimentStorage, TruenoBackend, RunStatus};

let mut backend = TruenoBackend::new();

let exp_id = backend.create_experiment("production-training", Some(serde_json::json!({
    "model": "llama-7b",
    "learning_rate": 0.0001
}))).unwrap();

let run_id = backend.create_run(&exp_id).unwrap();
backend.start_run(&run_id).unwrap();

// Training loop...
for step in 0..1000 {
    let loss = train_step();
    backend.log_metric(&run_id, "loss", step, loss).unwrap();
}

backend.complete_run(&run_id, RunStatus::Success).unwrap();
```

## Run Struct with Tracing

The `Run` struct provides a higher-level API with automatic step tracking and distributed tracing:

```rust
use std::sync::{Arc, Mutex};
use entrenar::storage::{InMemoryStorage, ExperimentStorage};
use entrenar::run::{Run, TracingConfig};

let mut storage = InMemoryStorage::new();
let exp_id = storage.create_experiment("my-exp", None).unwrap();
let storage = Arc::new(Mutex::new(storage));

// Create a run with tracing enabled
let config = TracingConfig::default();
let mut run = Run::new(&exp_id, storage.clone(), config).unwrap();

// Log metrics - step auto-increments per metric key
run.log_metric("loss", 0.5).unwrap();  // step 0
run.log_metric("loss", 0.4).unwrap();  // step 1
run.log_metric("loss", 0.3).unwrap();  // step 2

// Or log with explicit step
run.log_metric_at("accuracy", 0, 0.85).unwrap();
run.log_metric_at("accuracy", 100, 0.92).unwrap();

// Finish the run
run.finish(entrenar::storage::RunStatus::Success).unwrap();
```

## TracingConfig

Configure distributed tracing behavior:

```rust
use entrenar::run::TracingConfig;

// Default: tracing enabled, no OTLP export
let config = TracingConfig::default();
assert!(config.tracing_enabled);

// Disable tracing for faster execution
let config = TracingConfig::disabled();

// Enable OTLP export for observability platforms
let config = TracingConfig::default()
    .with_otlp_export()
    .with_golden_trace_path("/tmp/golden-traces");
```

### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tracing_enabled` | `bool` | `true` | Creates Renacer spans for distributed tracing |
| `export_otlp` | `bool` | `false` | Export traces via OpenTelemetry Protocol |
| `golden_trace_path` | `Option<PathBuf>` | `None` | Path for golden trace storage |

## ExperimentStorage Trait

Implement custom storage backends by implementing the `ExperimentStorage` trait:

```rust
pub trait ExperimentStorage: Send + Sync {
    // Experiment lifecycle
    fn create_experiment(&mut self, name: &str, config: Option<serde_json::Value>) -> Result<String>;

    // Run lifecycle
    fn create_run(&mut self, experiment_id: &str) -> Result<String>;
    fn start_run(&mut self, run_id: &str) -> Result<()>;
    fn complete_run(&mut self, run_id: &str, status: RunStatus) -> Result<()>;
    fn get_run_status(&self, run_id: &str) -> Result<RunStatus>;

    // Metrics
    fn log_metric(&mut self, run_id: &str, key: &str, step: u64, value: f64) -> Result<()>;
    fn get_metrics(&self, run_id: &str, key: &str) -> Result<Vec<MetricPoint>>;

    // Artifacts
    fn log_artifact(&mut self, run_id: &str, key: &str, data: &[u8]) -> Result<String>;

    // Distributed tracing
    fn set_span_id(&mut self, run_id: &str, span_id: &str) -> Result<()>;
    fn get_span_id(&self, run_id: &str) -> Result<Option<String>>;
}
```

## Run States

Runs follow a state machine:

```
Pending -> Running -> Success
                   -> Failed
                   -> Cancelled
```

- **Pending**: Run created but not started
- **Running**: Training in progress
- **Success**: Training completed successfully
- **Failed**: Training failed with an error
- **Cancelled**: Training was manually stopped

## Artifacts

Store binary artifacts with content-addressable hashing:

```rust
let model_weights = std::fs::read("model.safetensors").unwrap();
let hash = storage.log_artifact(&run_id, "model.safetensors", &model_weights).unwrap();
// Returns: "sha256-a1b2c3d4e5f6..."
```

## MetricPoint

Metrics are stored as timestamped data points:

```rust
use entrenar::storage::MetricPoint;

let point = MetricPoint::new(step, value);
// Automatically captures current timestamp

// Or with explicit timestamp
let point = MetricPoint::with_timestamp(step, value, timestamp);
```

## Integration with Training Loop

```rust
use std::sync::{Arc, Mutex};
use entrenar::storage::{InMemoryStorage, ExperimentStorage, RunStatus};
use entrenar::run::{Run, TracingConfig};

fn train_model() -> Result<(), Box<dyn std::error::Error>> {
    // Setup storage
    let mut storage = InMemoryStorage::new();
    let exp_id = storage.create_experiment("llm-finetune", Some(serde_json::json!({
        "model": "llama-7b",
        "lora_rank": 64,
        "learning_rate": 0.0001
    })))?;

    let storage = Arc::new(Mutex::new(storage));

    // Create traced run
    let config = TracingConfig::default().with_otlp_export();
    let mut run = Run::new(&exp_id, storage.clone(), config)?;

    // Training loop
    for epoch in 0..10 {
        let train_loss = train_epoch();
        let val_loss = validate_epoch();

        run.log_metric("train_loss", train_loss)?;
        run.log_metric("val_loss", val_loss)?;

        println!("Epoch {}: train={:.4}, val={:.4}", epoch, train_loss, val_loss);
    }

    // Complete run
    run.finish(RunStatus::Success)?;

    Ok(())
}
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `monitor` | Enables TruenoBackend for production persistence |
| `tracing` | Enables Renacer distributed tracing integration |

Enable features in `Cargo.toml`:

```toml
[dependencies]
entrenar = { version = "0.2", features = ["monitor", "tracing"] }
```
