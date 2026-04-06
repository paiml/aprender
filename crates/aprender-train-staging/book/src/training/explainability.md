# Explainability Callback

The `ExplainabilityCallback` integrates aprender's interpret module
into the training loop, providing feature attribution and importance
analysis during model evaluation.

## Overview

```rust
use entrenar::train::{
    ExplainabilityCallback, ExplainMethod, FeatureImportanceResult,
};

// Create callback with chosen method
let mut explainer = ExplainabilityCallback::new(ExplainMethod::PermutationImportance)
    .with_top_k(5)           // Track top 5 features
    .with_eval_samples(100)  // Use 100 samples for evaluation
    .with_feature_names(vec!["age".into(), "income".into(), "score".into()]);

trainer.add_callback(explainer);
```

## Available Methods

### ExplainMethod Enum

```rust
pub enum ExplainMethod {
    /// Permutation importance - fast, model-agnostic
    PermutationImportance,
    /// Integrated gradients - for differentiable models
    IntegratedGradients,
    /// Saliency maps - gradient-based attribution
    Saliency,
}
```

| Method | Speed | Use Case |
|--------|-------|----------|
| `PermutationImportance` | Fast | Any model, production monitoring |
| `IntegratedGradients` | Medium | Neural networks, precise attribution |
| `Saliency` | Fast | Neural networks, gradient visualization |

## Computing Attributions

The callback provides wrapper methods around aprender's interpret module:

### Permutation Importance

```rust
use aprender::primitives::Vector;

let x: Vec<Vector<f32>> = /* validation data */;
let y: Vec<f32> = /* targets */;

let importances = explainer.compute_permutation_importance(
    |sample| model.predict(sample),
    &x,
    &y,
);

// Record for this epoch
explainer.record_importances(epoch, importances);
```

### Integrated Gradients

```rust
let sample = Vector::from_slice(&[1.0, 2.0, 3.0]);
let baseline = Vector::from_slice(&[0.0, 0.0, 0.0]);

let attributions = explainer.compute_integrated_gradients(
    |x| model.predict(x),
    &sample,
    &baseline,
);
```

### Saliency Maps

```rust
let saliency = explainer.compute_saliency(
    |x| model.predict(x),
    &sample,
);
```

## Tracking Feature Importance

### Recording Per-Epoch Results

```rust
impl TrainerCallback for MyTrainingCallback {
    fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        // Compute importances on validation set
        let importances = self.explainer.compute_permutation_importance(
            |x| self.model.predict(x),
            &self.val_x,
            &self.val_y,
        );

        // Record sorted top-k importances
        self.explainer.record_importances(ctx.epoch, importances);

        CallbackAction::Continue
    }
}
```

### Querying Results

```rust
// Get all recorded results
let results: &[FeatureImportanceResult] = explainer.results();

for result in results {
    println!("Epoch {}: {:?}", result.epoch, result.importances);
}

// Get consistently important features across epochs
let consistent = explainer.consistent_top_features();
// Returns features ranked by: (1) frequency in top-k, (2) avg score
```

## FeatureImportanceResult

```rust
pub struct FeatureImportanceResult {
    /// Epoch when computed
    pub epoch: usize,
    /// Feature index to importance score (sorted by abs value)
    pub importances: Vec<(usize, f32)>,
    /// Method used for computation
    pub method: ExplainMethod,
}
```

## Complete Example

```rust
use entrenar::train::{
    Trainer, TrainConfig, ExplainabilityCallback, ExplainMethod,
    CallbackContext, CallbackAction, TrainerCallback,
};
use aprender::primitives::Vector;

// Simple linear model
fn predict(weights: &[f32], x: &Vector<f32>) -> f32 {
    weights.iter()
        .zip(x.as_slice())
        .map(|(w, xi)| w * xi)
        .sum()
}

fn main() {
    // Setup explainability callback
    let mut explainer = ExplainabilityCallback::new(ExplainMethod::PermutationImportance)
        .with_top_k(3)
        .with_feature_names(vec![
            "feature_0".into(),
            "feature_1".into(),
            "feature_2".into(),
        ]);

    // Validation data
    let val_x = vec![
        Vector::from_slice(&[1.0, 2.0, 3.0]),
        Vector::from_slice(&[2.0, 3.0, 4.0]),
        Vector::from_slice(&[3.0, 4.0, 5.0]),
    ];
    let val_y = vec![6.0, 9.0, 12.0];
    let weights = vec![1.0, 1.0, 1.0];

    // Compute and record importances
    let importances = explainer.compute_permutation_importance(
        |x| predict(&weights, x),
        &val_x,
        &val_y,
    );
    explainer.record_importances(0, importances);

    // Query results
    println!("Top features at epoch 0:");
    for (idx, score) in &explainer.results()[0].importances {
        let name = explainer.feature_names()
            .map(|n| n[*idx].as_str())
            .unwrap_or("unknown");
        println!("  {}: {:.4}", name, score);
    }
}
```

## Integration with Monitoring

Combine with `MonitorCallback` for comprehensive training observability:

```rust
trainer.add_callback(MonitorCallback::new());
trainer.add_callback(ExplainabilityCallback::new(ExplainMethod::PermutationImportance));
trainer.add_callback(EarlyStopping::new(5, 0.001));
```

This enables:
- Real-time loss/LR tracking (MonitorCallback)
- Feature importance trends (ExplainabilityCallback)
- Automatic early stopping (EarlyStopping)

## Use Cases

### Model Debugging

Identify which features drive predictions:

```rust
let top = explainer.consistent_top_features();
if top[0].0 != expected_important_feature {
    println!("Warning: Model may be using unexpected features");
}
```

### Feature Engineering Validation

Verify new features contribute positively:

```rust
// After adding new feature at index 5
let latest = explainer.results().last().unwrap();
let has_new_feature = latest.importances.iter().any(|(idx, _)| *idx == 5);
println!("New feature in top-k: {}", has_new_feature);
```

### Training Stability Analysis

Track feature importance stability across epochs:

```rust
let results = explainer.results();
if results.len() >= 2 {
    let prev = &results[results.len() - 2].importances;
    let curr = &results[results.len() - 1].importances;

    // Check if top feature changed
    if prev[0].0 != curr[0].0 {
        println!("Warning: Top feature changed between epochs");
    }
}
```

## See Also

- [Callback System](./callback-system.md) - Full callback documentation
- [Real-Time Monitoring](../monitor/overview.md) - Monitor integration
- [aprender interpret module](https://docs.rs/aprender/latest/aprender/interpret/) - Underlying explainability methods
