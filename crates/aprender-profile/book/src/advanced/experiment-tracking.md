# Experiment Span Tracking

Renacer provides specialized span types for tracking ML experiments, enabling syscall correlation during training runs. This integrates with the [entrenar](https://github.com/paiml/entrenar) experiment tracking framework (v1.8.0 specification).

## Overview

The experiment span module (`renacer::experiment_span`) provides:

- **SpanType::Experiment** - Specialized span type for ML training operations
- **ExperimentMetadata** - Structured metadata for training runs
- **Golden trace comparison** - Compare traces for behavioral equivalence

## SpanType Enum

Renacer classifies spans into three types:

```rust,ignore
use renacer::experiment_span::SpanType;

let syscall_span = SpanType::Syscall;    // System call (read, write, etc.)
let gpu_span = SpanType::Gpu;            // GPU operation (kernel, transfer)
let experiment_span = SpanType::Experiment; // ML experiment span
```

## ExperimentMetadata

Capture rich metadata about ML training operations:

```rust,ignore
use renacer::experiment_span::ExperimentMetadata;
use std::collections::HashMap;

let mut metrics = HashMap::new();
metrics.insert("accuracy".to_string(), 0.95);
metrics.insert("f1_score".to_string(), 0.92);

let metadata = ExperimentMetadata {
    model_name: "gpt-2".to_string(),
    epoch: Some(10),
    step: Some(1000),
    loss: Some(0.0025),
    metrics,
};

// Serialize to JSON for storage
let json = metadata.to_json();

// Parse from JSON
let parsed = ExperimentMetadata::from_json(&json).unwrap();
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | `String` | Model identifier (e.g., "gpt-2", "bert-base") |
| `epoch` | `Option<u32>` | Current training epoch |
| `step` | `Option<u64>` | Current training step |
| `loss` | `Option<f64>` | Current loss value |
| `metrics` | `HashMap<String, f64>` | Additional metrics (accuracy, perplexity, etc.) |

## Creating Experiment Spans

Use `ExperimentSpan::new_experiment()` to create spans:

```rust,ignore
use renacer::experiment_span::{ExperimentMetadata, ExperimentSpan};

let metadata = ExperimentMetadata {
    model_name: "llama-7b".to_string(),
    epoch: Some(5),
    step: Some(500),
    loss: Some(0.15),
    ..Default::default()
};

let span = ExperimentSpan::new_experiment("training_step", metadata);

// Span automatically has:
// - Generated trace_id (W3C format)
// - Generated span_id
// - Start timestamp
// - SpanType::Experiment

// End the span when done
let mut span = span;
span.end();

// Convert to SpanRecord for storage
let record = span.to_span_record();
```

## Golden Trace Comparison

Compare two traces for behavioral equivalence using `compare_traces()`:

```rust,ignore
use renacer::experiment_span::{compare_traces, EquivalenceScore};
use renacer::unified_trace::UnifiedTrace;

// Baseline trace (e.g., from original Python training)
let baseline = UnifiedTrace::new(1234, "python_train".to_string());

// Candidate trace (e.g., from Rust-transpiled training)
let candidate = UnifiedTrace::new(1234, "rust_train".to_string());

let score: EquivalenceScore = compare_traces(&baseline, &candidate);

println!("Syscall match: {:.1}%", score.syscall_match * 100.0);
println!("Timing variance: {:.1}%", score.timing_variance * 100.0);
println!("Semantic equiv: {:.1}%", score.semantic_equiv * 100.0);
println!("Overall: {:.1}%", score.overall() * 100.0);

if score.is_equivalent() {
    println!("Traces are behaviorally equivalent!");
}
```

### EquivalenceScore

The comparison returns three metrics:

| Metric | Range | Description |
|--------|-------|-------------|
| `syscall_match` | 0.0-1.0 | Syscall sequence similarity (LCS-based) |
| `timing_variance` | 0.0-1.0 | Timing difference (0 = identical) |
| `semantic_equiv` | 0.0-1.0 | Observable behavior match |

The `overall()` method computes a weighted score:
- 40% syscall match
- 20% timing (inverted)
- 40% semantic equivalence

The `is_equivalent()` method returns `true` if `overall() >= 0.85`.

## Integration with entrenar

When using entrenar's experiment tracking, Renacer spans enable syscall correlation:

```rust,ignore
use renacer::experiment_span::{ExperimentMetadata, ExperimentSpan};

// Called by entrenar's Run at each training step
fn track_training_step(
    model: &str,
    epoch: u32,
    step: u64,
    loss: f64,
    metrics: HashMap<String, f64>,
) -> ExperimentSpan {
    let metadata = ExperimentMetadata {
        model_name: model.to_string(),
        epoch: Some(epoch),
        step: Some(step),
        loss: Some(loss),
        metrics,
    };

    ExperimentSpan::new_experiment("training_step", metadata)
}
```

The experiment spans are automatically converted to `SpanRecord` for storage in trueno-db, enabling:

- Syscall correlation during training
- Performance analysis per epoch/step
- Trace comparison between implementations
- Anomaly detection in training behavior

## Use Cases

### 1. Transpiler Validation

Compare Python vs Rust training traces:

```rust,ignore
let python_trace = trace_python_training();
let rust_trace = trace_rust_training();

let score = compare_traces(&python_trace, &rust_trace);
assert!(score.is_equivalent(), "Transpilation changed behavior!");
```

### 2. Regression Detection

Compare traces across code changes:

```rust,ignore
let baseline = load_golden_trace("baseline.trace");
let current = run_current_training();

let score = compare_traces(&baseline, &current);
if !score.is_equivalent() {
    eprintln!("Regression detected: {:.1}% match", score.overall() * 100.0);
}
```

### 3. Performance Analysis

Track syscall patterns per training step:

```rust,ignore
for step in 0..num_steps {
    let span = ExperimentSpan::new_experiment("step", metadata.clone());

    // Training happens here (syscalls are traced)
    train_step(model, batch);

    span.end();
    storage.insert(span.to_span_record());
}

// Query syscall patterns per step
let patterns = storage.query_by_attribute("experiment.step");
```

## API Reference

### SpanType

```rust,ignore
pub enum SpanType {
    Syscall,    // Default - system calls
    Gpu,        // GPU operations
    Experiment, // ML experiment spans
}
```

### ExperimentMetadata

```rust,ignore
pub struct ExperimentMetadata {
    pub model_name: String,
    pub epoch: Option<u32>,
    pub step: Option<u64>,
    pub loss: Option<f64>,
    pub metrics: HashMap<String, f64>,
}

impl ExperimentMetadata {
    fn default() -> Self;
    fn to_json(&self) -> String;
    fn from_json(json: &str) -> Result<Self, serde_json::Error>;
    fn to_attributes(&self) -> HashMap<String, String>;
}
```

### ExperimentSpan

```rust,ignore
pub struct ExperimentSpan {
    pub trace_id: [u8; 16],
    pub span_id: [u8; 8],
    pub parent_span_id: Option<[u8; 8]>,
    pub name: String,
    pub span_type: SpanType,
    pub metadata: ExperimentMetadata,
    pub start_time_nanos: u64,
    pub end_time_nanos: u64,
    pub logical_clock: u64,
}

impl ExperimentSpan {
    fn new_experiment(name: &str, metadata: ExperimentMetadata) -> Self;
    fn new_experiment_with_parent(...) -> Self;
    fn end(&mut self);
    fn to_span_record(&self) -> SpanRecord;
}
```

### EquivalenceScore

```rust,ignore
pub struct EquivalenceScore {
    pub syscall_match: f64,
    pub timing_variance: f64,
    pub semantic_equiv: f64,
}

impl EquivalenceScore {
    fn overall(&self) -> f64;
    fn is_equivalent(&self) -> bool;
}
```

### compare_traces

```rust,ignore
pub fn compare_traces(
    baseline: &UnifiedTrace,
    candidate: &UnifiedTrace,
) -> EquivalenceScore;
```
