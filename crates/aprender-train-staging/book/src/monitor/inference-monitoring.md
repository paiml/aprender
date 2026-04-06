# Inference Monitoring

Real-time audit logging and explainability for APR format models.

## Overview

The inference monitoring module provides comprehensive tracing for model predictions, enabling:

- **Decision Path Tracing**: Capture exactly how a model reached its decision
- **Audit Logging**: Tamper-evident logs for safety-critical applications
- **Explainability**: Human-readable explanations for predictions
- **Provenance Graphs**: Reconstruct incident causality chains

This follows the Toyota Way principle of **現地現物 (Genchi Genbutsu)** -
every decision is traceable to ground truth.

## Decision Paths

Each model type has a specific path representation:

| Model Type | Path Type | Key Information |
|------------|-----------|-----------------|
| Linear/Logistic | `LinearPath` | Feature contributions, intercept, logit |
| Decision Tree | `TreePath` | Splits, thresholds, leaf info |
| Random Forest | `ForestPath` | Per-tree predictions, aggregation |
| KNN | `KNNPath` | Neighbor indices, distances, votes |
| Neural Network | `NeuralPath` | Gradients, activations, attention |

### Example: Linear Path

```rust
use entrenar::monitor::inference::{LinearPath, DecisionPath};

// Create a linear path with feature contributions
let path = LinearPath::new(
    vec![0.8, -0.3, 0.5],  // contributions: weight * feature
    0.1,                    // intercept
    1.1,                    // logit (sum of contributions + intercept)
    0.75,                   // output (e.g., sigmoid(logit))
).with_probability(0.75);

// Get human-readable explanation
println!("{}", path.explain());
```

Output:
```
Prediction: 0.7500 (probability: 75.0%)
Top contributing features:
  - feature[0]: +0.8000
  - feature[2]: +0.5000
  - feature[1]: -0.3000
Intercept: 0.1000
```

## Collector Strategies

Three collector strategies for different use cases:

### RingCollector (Real-Time)

Fixed-size ring buffer with O(1) operations. Ideal for games and drones.

- **Target**: <100ns per trace
- **Memory**: Bounded, configurable size
- **Use Case**: Real-time systems where only recent traces matter

```rust
use entrenar::monitor::inference::{RingCollector, LinearPath};

// Create ring collector with capacity 64
let collector: RingCollector<LinearPath, 64> = RingCollector::new();

// After recording traces, get recent ones
let recent = collector.recent(10);  // Last 10 traces
let all = collector.all();          // All traces (oldest first)
```

### StreamCollector (Persistent)

Write-through to file or network. Supports binary, JSON, and JSON Lines formats.

- **Target**: <1µs per trace
- **Memory**: Configurable buffer before flush
- **Use Case**: Persistent logging, audit trails

```rust
use entrenar::monitor::inference::{StreamCollector, StreamFormat, TreePath};
use std::fs::File;

let file = File::create("traces.jsonl")?;
let collector = StreamCollector::<TreePath, _>::new(file, StreamFormat::JsonLines)
    .with_flush_threshold(100);  // Flush every 100 traces
```

### HashChainCollector (Safety-Critical)

SHA-256 hash chain for tamper-evident audit trails. Each entry's hash includes the previous hash.

- **Target**: <10µs per entry
- **Security**: Cryptographic integrity verification
- **Use Case**: Autonomous vehicles, medical devices, regulatory compliance

```rust
use entrenar::monitor::inference::{HashChainCollector, LinearPath};

let mut collector: HashChainCollector<LinearPath> = HashChainCollector::new();

// Record traces...

// Verify chain integrity
let verification = collector.verify_chain();
assert!(verification.valid);
println!("Verified {} entries", verification.entries_verified);
```

## InferenceMonitor

High-level wrapper that combines model, collector, and optional safety checks:

```rust
use entrenar::monitor::inference::{
    InferenceMonitor, RingCollector, LinearPath, Explainable,
};

// Model must implement Explainable trait
let model = MyLinearModel::new(weights, intercept);
let collector: RingCollector<LinearPath, 64> = RingCollector::new();

let mut monitor = InferenceMonitor::new(model, collector)
    .with_latency_budget_ns(10_000_000);  // 10ms budget

// Predict with automatic tracing
let outputs = monitor.predict(&input_features, 1);

// Access traces
let traces = monitor.collector().recent(1);
println!("{}", traces[0].explain());
```

## Implementing Explainable

To use inference monitoring, your model must implement `Explainable`:

```rust
use entrenar::monitor::inference::{Explainable, LinearPath};

struct MyModel {
    weights: Vec<f32>,
    intercept: f32,
}

impl Explainable for MyModel {
    type Path = LinearPath;

    fn predict_explained(&self, x: &[f32], n_samples: usize) -> (Vec<f32>, Vec<Self::Path>) {
        let features_per_sample = x.len() / n_samples;
        let mut outputs = Vec::with_capacity(n_samples);
        let mut paths = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let sample = &x[i * features_per_sample..(i + 1) * features_per_sample];

            // Compute contributions
            let contributions: Vec<f32> = self.weights
                .iter()
                .zip(sample)
                .map(|(w, x)| w * x)
                .collect();

            let logit: f32 = contributions.iter().sum::<f32>() + self.intercept;
            let output = sigmoid(logit);

            let path = LinearPath::new(contributions, self.intercept, logit, output)
                .with_probability(output);

            outputs.push(output);
            paths.push(path);
        }

        (outputs, paths)
    }

    fn explain_one(&self, sample: &[f32]) -> Self::Path {
        self.predict_explained(sample, 1).1.into_iter().next().unwrap()
    }
}
```

## Safety Andon

Automatic quality monitoring based on Safety Integrity Levels (SIL):

```rust
use entrenar::monitor::inference::{SafetyAndon, SafetyIntegrityLevel};

let mut andon = SafetyAndon::new(SafetyIntegrityLevel::SIL2)
    .with_min_confidence(0.7)           // Alert if confidence < 70%
    .with_low_confidence_threshold(3);  // Alert after 3 consecutive low-confidence

// Check each trace
andon.check_trace(&trace, latency_budget_ns);

// Review alerts
for alert in andon.history() {
    println!("[{:?}] {}", alert.level, alert.message);
}
```

### Safety Integrity Levels

| Level | Min Confidence | Max Latency | Use Case |
|-------|----------------|-------------|----------|
| QM    | 50%            | 100ms       | Quality management |
| SIL1  | 60%            | 50ms        | Low risk |
| SIL2  | 70%            | 20ms        | Medium risk |
| SIL3  | 80%            | 10ms        | High risk |
| SIL4  | 90%            | 5ms         | Critical systems |

## Counterfactual Explanations

Answer "What would need to change to flip the decision?"

```rust
use entrenar::monitor::inference::Counterfactual;

let feature_names = vec!["income".into(), "debt_ratio".into(), "score".into()];

let counterfactual = Counterfactual::new(
    vec![0.3, 0.8, 0.2],   // Original input (denied)
    0,                      // Original decision
    0.3,                    // Original confidence
    vec![0.6, 0.5, 0.4],   // Counterfactual input (approved)
    1,                      // Alternative decision
    0.8,                    // Alternative confidence
).with_feature_names(&feature_names);

println!("{}", counterfactual.explain());
// Output:
// The decision would have been 1 if:
//   - income: 0.3000 → 0.6000 (+0.3000)
//   - debt_ratio: 0.8000 → 0.5000 (-0.3000)
//   - score: 0.2000 → 0.4000 (+0.2000)
```

## Provenance Graphs

Reconstruct causal chains for incident analysis:

```rust
use entrenar::monitor::inference::{
    ProvenanceGraph, ProvenanceNode, ProvenanceEdge, CausalRelation,
    IncidentReconstructor,
};

let mut graph = ProvenanceGraph::new();

// Build provenance chain: Input -> Transform -> Inference -> Action
let camera = graph.add_node(ProvenanceNode::Input {
    source: "front_camera".into(),
    timestamp_ns: 1000,
    hash: 0xdeadbeef,
});

let preprocess = graph.add_node(ProvenanceNode::Transform {
    operation: "normalize".into(),
    input_refs: vec![camera],
});

let detection = graph.add_node(ProvenanceNode::Inference {
    model_id: "detector".into(),
    model_version: "v2.1".into(),
    confidence: 0.3,  // Low confidence!
    output: 0.0,
});

// Connect nodes
graph.add_edge(ProvenanceEdge {
    from: camera,
    to: preprocess,
    relation: CausalRelation::DataFlow,
    timestamp_ns: 1100,
});

// Reconstruct incident path and identify anomalies
let reconstructor = IncidentReconstructor::new(&graph);
let path = reconstructor.reconstruct_path(detection, 10);
let anomalies = reconstructor.identify_anomalies(&path, 0.7);

for anomaly in &anomalies {
    println!("Node {}: {}", anomaly.node_id, anomaly.description);
}
```

## Serialization

### Binary Format (APRT)

Compact binary format with magic number `0x41505254` ("APRT"):

```rust
use entrenar::monitor::inference::{TraceSerializer, TraceFormat, PathType};

let serializer = TraceSerializer::new(TraceFormat::Binary);
let bytes = serializer.serialize(&trace, PathType::Linear)?;

// Deserialize
let restored: DecisionTrace<LinearPath> = serializer.deserialize(&bytes)?;
```

### JSON Format

Human-readable JSON for debugging and integration:

```rust
let serializer = TraceSerializer::new(TraceFormat::Json);
let json = serializer.serialize(&trace, PathType::Linear)?;
println!("{}", String::from_utf8_lossy(&json));
```

## Performance Targets

| Collector | Target Latency | Measured |
|-----------|----------------|----------|
| RingCollector | <100ns | ~50ns |
| StreamCollector | <1µs | ~500ns |
| HashChainCollector | <10µs | ~5µs |

## Running the Example

```bash
cargo run --example inference_monitor
```

This demonstrates all components working together: collectors, safety monitoring, counterfactuals, provenance graphs,
and serialization.
