# Experiment Schema (Phase 5)

The experiment schema provides data structures for ML experiment tracking, designed to integrate with [entrenar](https://github.com/paiml/entrenar)'s `ExperimentStorage`.

## Schema Overview

```
ExperimentRecord (1) ──< RunRecord (N)
                             │
                             ├──< MetricRecord (N) [time-series]
                             └──< ArtifactRecord (N) [CAS]
```

## Record Types

### ExperimentRecord

Root entity representing a tracked experiment.

```rust
use trueno_db::experiment::ExperimentRecord;

// Simple creation
let experiment = ExperimentRecord::new("exp-001", "ResNet Training");

// With configuration
let config = serde_json::json!({
    "learning_rate": 0.01,
    "batch_size": 32,
    "model": "resnet50"
});

let experiment = ExperimentRecord::builder("exp-002", "Custom Config")
    .config(config)
    .build();

// Access fields
println!("ID: {}", experiment.experiment_id());
println!("Name: {}", experiment.name());
println!("Created: {}", experiment.created_at());
```

### RunRecord

Represents a single execution of an experiment with lifecycle tracking.

```rust
use trueno_db::experiment::{RunRecord, RunStatus};

// Create a run
let mut run = RunRecord::new("run-001", "exp-001");

// Start execution
run.start();
assert_eq!(run.status(), RunStatus::Running);

// Complete with status
run.complete(RunStatus::Success);
assert!(run.ended_at().is_some());

// With renacer span for distributed tracing
let run = RunRecord::builder("run-002", "exp-001")
    .renacer_span_id("span-abc-123")
    .build();
```

**RunStatus Variants:**
- `Pending` - Created but not started
- `Running` - Currently executing
- `Success` - Completed successfully
- `Failed` - Terminated with error
- `Cancelled` - User/system cancelled

### MetricRecord

Time-series optimized metric storage with step-based ordering.

```rust
use trueno_db::experiment::MetricRecord;

// Log training metrics
for step in 0..100 {
    let loss = 1.0 / (step as f64 + 1.0);
    let metric = MetricRecord::new("run-001", "loss", step, loss);
    // Store metric...
}

// With explicit timestamp
use chrono::{TimeZone, Utc};
let ts = Utc.with_ymd_and_hms(2025, 1, 15, 12, 0, 0).unwrap();

let metric = MetricRecord::builder("run-001", "accuracy", 100, 0.95)
    .timestamp(ts)
    .build();
```

### ArtifactRecord

Content-addressable storage for model checkpoints and outputs.

```rust
use trueno_db::experiment::ArtifactRecord;

let artifact = ArtifactRecord::new(
    "run-001",
    "model.pt",
    "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    104_857_600, // 100MB
);

println!("Key: {}", artifact.key());
println!("Hash: {}", artifact.cas_hash());
println!("Size: {} bytes", artifact.size_bytes());
```

## ExperimentStore

In-memory storage with time-series query capabilities.

```rust
use trueno_db::experiment::{
    ExperimentStore, ExperimentRecord, RunRecord, MetricRecord, RunStatus,
};

// Create store
let mut store = ExperimentStore::new();

// Add experiment and run
let experiment = ExperimentRecord::new("exp-001", "Training");
store.add_experiment(experiment);

let mut run = RunRecord::new("run-001", "exp-001");
run.start();
store.add_run(run);

// Log metrics during training
for step in 0..100 {
    store.add_metric(MetricRecord::new(
        "run-001",
        "loss",
        step,
        1.0 / (step as f64 + 1.0),
    ));
    store.add_metric(MetricRecord::new(
        "run-001",
        "accuracy",
        step,
        step as f64 / 100.0,
    ));
}

// Query metrics for visualization (ordered by step)
let loss_curve = store.get_metrics_for_run("run-001", "loss");
assert_eq!(loss_curve.len(), 100);
assert_eq!(loss_curve[0].step(), 0);  // First step
assert_eq!(loss_curve[99].step(), 99); // Last step

// Query runs for experiment
let runs = store.get_runs_for_experiment("exp-001");
```

## Serialization

All records support JSON serialization via serde:

```rust
use trueno_db::experiment::MetricRecord;

let metric = MetricRecord::new("run-001", "loss", 50, 0.25);

// Serialize
let json = serde_json::to_string(&metric)?;

// Deserialize
let restored: MetricRecord = serde_json::from_str(&json)?;
assert_eq!(metric.run_id(), restored.run_id());
```

## Integration with entrenar

This schema is designed to be the storage foundation for entrenar's experiment tracking:

```rust
// entrenar writes to trueno-db
let store = ExperimentStore::new();

// ExperimentStorage from entrenar uses this schema
// See: entrenar/docs/specifications/experiment-tracking-spec.md §3.1
```

## Time-Series Optimization

The `get_metrics_for_run` query returns metrics ordered by step, enabling:

- Loss curve visualization
- Learning rate schedules
- Accuracy progression
- Any time-series metric analysis

```rust
// Metrics are always returned in step order
let metrics = store.get_metrics_for_run("run-001", "loss");
for (i, m) in metrics.iter().enumerate() {
    assert_eq!(m.step(), i as u64);
}
```
