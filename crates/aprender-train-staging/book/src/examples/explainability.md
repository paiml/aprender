# Explainability (Feature Attribution)

This example demonstrates feature importance tracking using permutation importance during training.

## Running the Example

```bash
cargo run --example explainability
```

## Code

```rust
{{#include ../../../examples/explainability.rs}}
```

## Expected Output

```
=== Explainability Callback Example ===

Model: y = 3.0*x0 + 1.0*x1 + 0.5*x2
Expected: feature_0 should be most important

Configuration:
  Method: PermutationImportance
  Top-K: 3
  Eval samples: 50

Epoch 0:
  feature_0: 17.100000
  feature_1: 2.800000
  feature_2: 0.600000
```

## Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| Permutation Importance | Shuffle feature, measure loss increase | General feature ranking |
| SHAP | Shapley values for attribution | Individual predictions |
| Integrated Gradients | Gradient-based attribution | Neural networks |

## Usage

```rust
use entrenar::train::{ExplainabilityCallback, ExplainMethod};

trainer.add_callback(
    ExplainabilityCallback::new(ExplainMethod::PermutationImportance)
        .with_top_k(10)
        .with_eval_samples(100)
);
```
