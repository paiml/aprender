# Model Lineage

Entrenar provides comprehensive lineage tracking for experiment reproducibility using Lamport timestamps and causal
event ordering. The integrity module enables behavioral verification, trace storage policies, and promotion gates for ML
model deployment.

## Overview

The lineage system consists of three components:

1. **LamportTimestamp** - Logical clocks for causal ordering across distributed systems
2. **CausalLineage** - Event tracking with happens-before relationships
3. **BehavioralIntegrity** - Model promotion gates with metamorphic testing

## Lamport Timestamps

Lamport timestamps provide a logical clock for ordering events in distributed systems without relying on synchronized
physical clocks.

```rust
use entrenar::integrity::LamportTimestamp;

// Create timestamps for different nodes
let mut node_a = LamportTimestamp::new("node-a");
let mut node_b = LamportTimestamp::new("node-b");

// Increment on local events
node_a.increment();

// Merge when receiving messages (synchronizes clocks)
node_b.merge(&node_a);
node_b.increment();

// Check causal relationships
assert!(node_a.happens_before(&node_b));
```

### Happens-Before Relationship

The `happens_before()` method determines if one event causally precedes another:

```rust
use entrenar::integrity::LamportTimestamp;

let ts1 = LamportTimestamp::with_counter("node-1", 5);
let ts2 = LamportTimestamp::with_counter("node-1", 10);
let ts3 = LamportTimestamp::with_counter("node-2", 7);

// Same node, lower counter = happens before
assert!(ts1.happens_before(&ts2));

// Different nodes may be concurrent
println!("Concurrent: {}", ts1.is_concurrent_with(&ts3));
```

## Causal Lineage Tracking

Track experiment events with causal ordering:

```rust
use entrenar::integrity::{CausalLineage, LineageEvent, LineageEventType, LamportTimestamp};

let mut lineage = CausalLineage::new();

// Record experiment lifecycle events
let ts1 = LamportTimestamp::new("trainer-node");
let start = LineageEvent::new(ts1, LineageEventType::RunStarted, "run-001");
lineage.add_event(start);

// Log metrics during training
let ts2 = LamportTimestamp::with_counter("trainer-node", 2);
let metric = LineageEvent::new(ts2, LineageEventType::MetricLogged, "run-001");
lineage.add_event(metric);

// Save artifacts
let ts3 = LamportTimestamp::with_counter("trainer-node", 3);
let artifact = LineageEvent::new(ts3, LineageEventType::ArtifactSaved, "run-001")
    .with_context("checkpoint: epoch_10.pt");
lineage.add_event(artifact);

// Complete the run
let ts4 = LamportTimestamp::with_counter("trainer-node", 4);
let complete = LineageEvent::new(ts4, LineageEventType::RunCompleted, "run-001");
lineage.add_event(complete);
```

### Event Types

| Event Type | Description |
|------------|-------------|
| RunStarted | Training run initiated |
| MetricLogged | Metric recorded (loss, accuracy, etc.) |
| ArtifactSaved | Checkpoint or model artifact saved |
| RunCompleted | Training run finished |
| ModelPromoted | Model promoted to production |
| ModelRolledBack | Model rolled back to previous version |

### Querying Events

```rust
use entrenar::integrity::{CausalLineage, LineageEventType};

// Get all events in causal order
let events = lineage.events_in_order();

// Filter by run
let run_events = lineage.events_for_run("run-001");

// Filter by type
let promotions = lineage.events_of_type(LineageEventType::ModelPromoted);

// Get latest event for a run
let latest = lineage.latest_event_for_run("run-001");

// Check if one run precedes another
let precedes = lineage.run_precedes("run-001", "run-002");
```

## Trace Storage Policy

Configure how experiment traces are stored and retained:

```rust
use entrenar::integrity::{TraceStoragePolicy, CompressionAlgorithm};

// Custom policy
let policy = TraceStoragePolicy::new(
    CompressionAlgorithm::Zstd,  // Compression algorithm
    30,                           // Retention days
    10 * 1024 * 1024 * 1024,     // Max size (10 GB)
    0.5,                          // Sample rate (50%)
);

// Or use presets
let dev_policy = TraceStoragePolicy::development();
let prod_policy = TraceStoragePolicy::production();
let archive_policy = TraceStoragePolicy::archival();
```

### Compression Algorithms

| Algorithm | Ratio | Speed | Use Case |
|-----------|-------|-------|----------|
| None | 1.0x | Fastest | Debug, small traces |
| RLE | ~2.0x | Fast | Sparse data |
| LZ4 | ~2.5x | Fast | Real-time streaming |
| Zstd | ~4.0x | Moderate | General purpose |

### Policy Presets

| Preset | Compression | Retention | Sample Rate | Use Case |
|--------|-------------|-----------|-------------|----------|
| minimal | None | 7 days | 10% | CI/testing |
| development | LZ4 | 7 days | 100% | Local dev |
| production | Zstd | 90 days | 50% | Production |
| archival | Zstd | 365 days | 25% | Long-term storage |

### Sampling

Deterministic sampling ensures consistent trace collection:

```rust
use entrenar::integrity::TraceStoragePolicy;

let policy = TraceStoragePolicy::production();

// Check if a trace should be sampled
if policy.should_sample("trace-12345") {
    // Collect this trace
}

// Same trace ID always returns same result (deterministic)
assert_eq!(
    policy.should_sample("trace-12345"),
    policy.should_sample("trace-12345")
);
```

## Behavioral Integrity

Verify model behavior consistency before promotion:

```rust
use entrenar::integrity::{BehavioralIntegrity, BehavioralIntegrityBuilder};

let integrity = BehavioralIntegrityBuilder::new("model-v2.0")
    .equivalence_score(0.98)    // Output consistency
    .syscall_match(0.95)        // System call patterns
    .timing_variance(0.05)      // Inference timing consistency
    .semantic_equiv(0.97)       // Semantic output equivalence
    .test_count(10000)
    .build();

// Check if model passes promotion gate
if integrity.passes_gate(0.9) {
    println!("Model approved for production!");
} else {
    println!("Model failed quality gate");
}
```

### Composite Score

The composite score combines multiple metrics with configurable weights:

| Metric | Weight | Description |
|--------|--------|-------------|
| Equivalence | 30% | Output consistency across runs |
| Syscall Match | 20% | System call pattern matching |
| Timing (inverted) | 20% | Lower variance = better |
| Semantic Equiv | 30% | Semantic output equivalence |

```rust
use entrenar::integrity::BehavioralIntegrity;

let integrity = BehavioralIntegrity::new(0.9, 0.85, 0.1, 0.88, "model-v1");

// Get composite score (0.0 - 1.0)
let score = integrity.composite_score();
println!("Composite score: {:.1}%", score * 100.0);
```

### Metamorphic Violations

Track violations from metamorphic testing:

```rust
use entrenar::integrity::{
    BehavioralIntegrity, MetamorphicViolation, MetamorphicRelationType
};

let mut integrity = BehavioralIntegrity::new(0.9, 0.85, 0.1, 0.88, "model-v1");

// Record a violation
let violation = MetamorphicViolation::new(
    "MV-001",
    MetamorphicRelationType::Identity,
    "Model produces different outputs for identical inputs",
    "Input: [1.0, 2.0, 3.0]",
    "Expected: [0.5, 0.3, 0.2]",
    "Actual: [0.4, 0.4, 0.2]",
    0.7,  // Severity (0.0-1.0)
);

integrity.add_violation(violation);

// Analyze violations
let counts = integrity.violation_counts();
println!("Critical: {}, Warnings: {}, Minor: {}",
    counts.critical, counts.warnings, counts.minor);
```

### Metamorphic Relation Types

| Type | Description | Example |
|------|-------------|---------|
| Identity | f(x) = f(x) | Same input, same output |
| Additive | f(x+c) relates to f(x) | Translation invariance |
| Multiplicative | f(k*x) relates to f(x) | Scale invariance |
| Permutation | f(permute(x)) relates to f(x) | Order invariance |
| Negation | f(-x) relates to f(x) | Sign symmetry |
| Composition | f(g(x)) relates to g(f(x)) | Commutativity |

### Assessment Grades

```rust
use entrenar::integrity::{BehavioralIntegrity, IntegrityAssessment};

let integrity = BehavioralIntegrity::new(0.95, 0.92, 0.05, 0.94, "model-v1");

match integrity.assessment() {
    IntegrityAssessment::Excellent => println!("Ready for production"),
    IntegrityAssessment::Good => println!("Minor improvements needed"),
    IntegrityAssessment::Fair => println!("Significant work required"),
    IntegrityAssessment::Poor => println!("Major issues detected"),
    IntegrityAssessment::Critical => println!("Critical violations found"),
}
```

### Summary Report

```rust
use entrenar::integrity::BehavioralIntegrity;

let integrity = BehavioralIntegrity::new(0.95, 0.92, 0.05, 0.94, "model-v2.0")
    .with_test_count(10000);

println!("{}", integrity.summary());
// Output:
// Model: model-v2.0
// Composite Score: 94.5%
// Assessment: Excellent
// Violations: 0 critical, 0 warnings, 0 minor
// Tests Run: 10000
// Gate Status: PASS
```

## Integration with Experiment Tracking

Combine lineage tracking with behavioral integrity:

```rust
use entrenar::integrity::{
    CausalLineage, LineageEvent, LineageEventType, LamportTimestamp,
    BehavioralIntegrityBuilder,
};

let mut lineage = CausalLineage::new();

// Start training run
let ts1 = LamportTimestamp::new("trainer");
lineage.add_event(LineageEvent::new(ts1, LineageEventType::RunStarted, "run-001"));

// ... training completes ...

// Check behavioral integrity before promotion
let integrity = BehavioralIntegrityBuilder::new("candidate-model")
    .equivalence_score(0.95)
    .syscall_match(0.92)
    .timing_variance(0.08)
    .semantic_equiv(0.94)
    .test_count(5000)
    .build();

if integrity.passes_gate(0.9) {
    // Record promotion with integrity context
    let ts2 = LamportTimestamp::with_counter("trainer", 100);
    let context = format!(
        "score={:.2},assessment={}",
        integrity.composite_score(),
        integrity.assessment()
    );
    let promote = LineageEvent::new(ts2, LineageEventType::ModelPromoted, "run-001")
        .with_context(context);
    lineage.add_event(promote);

    println!("Model promoted to production!");
}
```

## Configuration

Configure integrity settings in your training config:

```yaml
# entrenar.yaml
integrity:
  lineage:
    enabled: true
    node_id: "trainer-001"

  trace_storage:
    compression: zstd
    retention_days: 90
    max_size_gb: 50
    sample_rate: 0.5

  behavioral:
    promotion_threshold: 0.9
    max_timing_variance: 0.2
    require_clean_violations: true
```
