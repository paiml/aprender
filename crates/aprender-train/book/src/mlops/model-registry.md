# Model Registry

Manage model lifecycle with staging workflows, promotion policies, and version comparisons.

## Toyota Principle: Kanban

Visual workflow management for model stages. Models flow through stages
(Development → Staging → Production → Archived) like parts through a factory.

## Quick Start

```rust
use entrenar::storage::{InMemoryRegistry, ModelRegistry, ModelStage};
use std::collections::HashMap;

let mut registry = InMemoryRegistry::new();

// Register a new model version
let metrics = HashMap::from([
    ("accuracy".to_string(), 0.95),
    ("f1_score".to_string(), 0.93),
]);

registry.register_model(
    "sentiment-classifier",
    1,
    metrics,
    "/artifacts/model-v1.safetensors"
)?;

// Transition through stages
registry.transition_stage("sentiment-classifier", 1, ModelStage::Staging)?;
registry.transition_stage("sentiment-classifier", 1, ModelStage::Production)?;
```

## Model Stages

```rust
use entrenar::storage::ModelStage;

// Available stages
ModelStage::Development  // Initial development
ModelStage::Staging      // Testing and validation
ModelStage::Production   // Serving traffic
ModelStage::Archived     // Retired from use
```

## Stage Transitions

```rust
// Valid transitions
// Development → Staging
registry.transition_stage("model", 1, ModelStage::Staging)?;

// Staging → Production
registry.transition_stage("model", 1, ModelStage::Production)?;

// Any stage → Archived
registry.transition_stage("model", 1, ModelStage::Archived)?;

// Invalid transitions return error
// Cannot go backwards: Production → Staging (error)
```

## Version Comparison

```rust
use entrenar::storage::VersionComparison;

// Compare two versions
let comparison = registry.compare_versions("model", 1, 2)?;

println!("Version {} vs {}", comparison.version_a, comparison.version_b);

for (metric, diff) in &comparison.metric_diffs {
    println!("  {}: {:.2}% change", metric, diff.percent_change);
}
```

## Promotion Policies

Automatically validate models before promotion:

```rust
use entrenar::storage::{PromotionPolicy, MetricRequirement, Comparison};

let policy = PromotionPolicy::new()
    .require_metric(MetricRequirement {
        name: "accuracy".to_string(),
        comparison: Comparison::Gte,
        threshold: 0.90,
    })
    .require_metric(MetricRequirement {
        name: "latency_p99_ms".to_string(),
        comparison: Comparison::Lte,
        threshold: 100.0,
    });

// Check if model meets policy
let result = registry.check_promotion_policy("model", 1, &policy)?;

if result.passed {
    registry.transition_stage("model", 1, ModelStage::Production)?;
} else {
    for failure in result.failures {
        eprintln!("Policy violation: {}", failure);
    }
}
```

## Querying the Registry

```rust
// Get specific version
let version = registry.get_version("model", 1)?;

// List all versions
let versions = registry.list_versions("model")?;

// Get models by stage
let production_models = registry.get_by_stage(ModelStage::Production)?;

// Get latest version in stage
let latest = registry.get_latest_in_stage("model", ModelStage::Production)?;
```

## Model Version Details

```rust
use entrenar::storage::ModelVersion;

let version: ModelVersion = registry.get_version("model", 1)?;

println!("Version: {}", version.version);
println!("Stage: {:?}", version.stage);
println!("Created: {}", version.created_at);
println!("Artifact: {}", version.artifact_path);

for (name, value) in &version.metrics {
    println!("  {}: {}", name, value);
}
```

## Audit Trail

Track all stage transitions:

```rust
use entrenar::storage::StageTransition;

let transitions = registry.get_transitions("model", 1)?;

for t in transitions {
    println!(
        "{}: {:?} → {:?}",
        t.timestamp, t.from_stage, t.to_stage
    );
}
```

## Cargo Run Example

```bash
# Register and promote a model
cargo run --example model_registry

# List all production models
cargo run --example model_registry -- --list-production
```

## Integration with CI/CD

```yaml
# .github/workflows/model-promotion.yml
name: Model Promotion

on:
  workflow_dispatch:
    inputs:
      model_name:
        required: true
      version:
        required: true

jobs:
  promote:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Check promotion policy
        run: |
          cargo run --example check_promotion -- \
            --model ${{ inputs.model_name }} \
            --version ${{ inputs.version }}
      - name: Promote to production
        if: success()
        run: |
          cargo run --example promote_model -- \
            --model ${{ inputs.model_name }} \
            --version ${{ inputs.version }} \
            --stage production
```

## Best Practices

1. **Define clear promotion policies** - Objective criteria for each stage
2. **Keep audit trails** - Track who promoted what and when
3. **Use semantic versioning** - Clear version numbering
4. **Archive old versions** - Don't delete, archive for rollback
5. **Automate promotion checks** - CI/CD integration

## See Also

- [MLOps Overview](./overview.md)
- [Experiment Tracking](./experiment-tracking.md)
- [Cloud Storage](./cloud-storage.md)
