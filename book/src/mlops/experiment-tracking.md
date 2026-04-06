# Experiment Tracking

Track experiments, runs, metrics, parameters, and artifacts with a local-first SQLite backend.

## Toyota Principle: Genchi Genbutsu

"Go and see" - understand the situation through direct observation. Experiment tracking enables data-driven decisions by
capturing everything that happens during training.

## Quick Start

```rust
use entrenar::storage::{SqliteBackend, ExperimentStorage, RunStatus, ParameterValue};

// Open or create database
let mut backend = SqliteBackend::open("experiments.db")?;

// Create experiment
let exp_id = backend.create_experiment("gpt2-finetune", Some(serde_json::json!({
    "model": "gpt2",
    "dataset": "wikitext"
})))?;

// Create and start a run
let run_id = backend.create_run(&exp_id)?;
backend.start_run(&run_id)?;

// Log parameters
backend.log_param(&run_id, "learning_rate", ParameterValue::Float(1e-4))?;
backend.log_param(&run_id, "batch_size", ParameterValue::Int(32))?;
backend.log_param(&run_id, "optimizer", ParameterValue::String("adamw".into()))?;

// Log metrics over time
for epoch in 0..10 {
    let loss = train_epoch(&model);
    backend.log_metric(&run_id, "loss", epoch, loss)?;
    backend.log_metric(&run_id, "accuracy", epoch, evaluate(&model))?;
}

// Save artifact
let model_bytes = save_model(&model);
let artifact_hash = backend.log_artifact(&run_id, "model.safetensors", &model_bytes)?;

// Complete the run
backend.complete_run(&run_id, RunStatus::Success)?;
```

## Storage Backends

### SQLite (Default)

Local-first, zero-dependency storage:

```rust
use entrenar::storage::SqliteBackend;

// File-based storage
let backend = SqliteBackend::open("./experiments.db")?;

// In-memory for testing
let backend = SqliteBackend::open_in_memory()?;
```

### In-Memory

For testing and ephemeral experiments:

```rust
use entrenar::storage::InMemoryStorage;

let mut storage = InMemoryStorage::new();
```

## Parameter Types

```rust
use entrenar::storage::ParameterValue;

// Supported types
ParameterValue::String("adam".to_string())
ParameterValue::Int(32)
ParameterValue::Float(0.001)
ParameterValue::Bool(true)
ParameterValue::List(vec![
    ParameterValue::Int(128),
    ParameterValue::Int(256),
])
```

## Querying Experiments

```rust
// Get experiment by ID
let experiment = backend.get_experiment(&exp_id)?;

// List runs for experiment
let runs = backend.list_runs(&exp_id)?;

// Get metrics for a run
let loss_history = backend.get_metrics(&run_id, "loss")?;
for point in loss_history {
    println!("Step {}: {}", point.step, point.value);
}

// Get run status
let status = backend.get_run_status(&run_id)?;
```

## Filtering Runs

```rust
use entrenar::storage::{ParamFilter, FilterOp};

// Filter by parameter value
let filters = vec![
    ParamFilter {
        key: "learning_rate".to_string(),
        op: FilterOp::Lt,
        value: ParameterValue::Float(1e-3),
    },
    ParamFilter {
        key: "optimizer".to_string(),
        op: FilterOp::Eq,
        value: ParameterValue::String("adamw".into()),
    },
];

let runs = backend.filter_runs(&exp_id, &filters)?;
```

## Distributed Tracing

Link experiments to distributed traces:

```rust
// Set span ID for distributed tracing
backend.set_span_id(&run_id, "trace-abc-123")?;

// Retrieve span ID
let span_id = backend.get_span_id(&run_id)?;
```

## Cargo Run Example

```bash
# Run experiment tracking example
cargo run --example experiment_tracking

# With verbose output
cargo run --example experiment_tracking -- --verbose
```

## Database Schema

The SQLite backend uses the following tables:

- `experiments` - Experiment metadata
- `runs` - Individual training runs
- `metrics` - Time-series metrics
- `params` - Run parameters
- `artifacts` - Content-addressable artifact storage

## Best Practices

1. **Use descriptive experiment names** - Makes querying easier
2. **Log all hyperparameters** - Enables reproducibility
3. **Save artifacts with hashes** - Content-addressable storage prevents duplicates
4. **Set span IDs** - Enables distributed tracing across services
5. **Use parameter filtering** - Find best runs quickly

## See Also

- [MLOps Overview](./overview.md)
- [Model Registry](./model-registry.md)
- [Preflight Validation](./preflight.md)
