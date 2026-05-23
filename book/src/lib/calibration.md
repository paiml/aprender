<!-- PCU: lib-calibration | contract: contracts/apr-page-lib-calibration-v1.yaml -->

# Module: `aprender::calibration`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/calibration.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/calibration.rs) or directory.

## Example

```rust
use aprender::calibration::{TemperatureScaling, PlattScaling, IsotonicRegression};
use aprender::calibration::{expected_calibration_error, brier_score};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::calibration` answers the question "are my model's confidence
scores trustworthy?" Modern classifiers are typically over-confident — the
top softmax probability is higher than the actual frequency of being
correct. This module ships three post-hoc calibrators (Temperature Scaling
for multi-class, Platt Scaling for binary, Isotonic Regression for non-
parametric) and four calibration metrics (Expected and Maximum Calibration
Error, Brier score, reliability diagrams).

## Key types

| Type | Description |
|------|-------------|
| `TemperatureScaling` | Single-parameter softmax rescaling. `fit(&[logits], &labels)` learns T; `calibrate(logits)` applies it. |
| `PlattScaling` | Logistic-sigmoid post-hoc calibration for binary classifiers. |
| `IsotonicRegression` | Non-parametric monotone calibrator. |
| `expected_calibration_error`, `maximum_calibration_error` | Bucketed calibration metrics. |
| `brier_score` | Mean-squared-error between probability and 0/1 outcome. |
| `reliability_diagram` | Returns `(avg_confidence, avg_accuracy)` per bin for plotting. |

## Usage patterns

### Pattern 1: Temperature scaling for a multi-class model

```rust
use aprender::calibration::TemperatureScaling;
use aprender::primitives::Vector;

// Per-sample logit vectors and integer labels (from a held-out val set).
let logits = vec![
    Vector::from_slice(&[2.0, 0.5, 0.1]),
    Vector::from_slice(&[0.2, 1.8, 0.4]),
    Vector::from_slice(&[0.1, 0.3, 2.5]),
];
let labels = vec![0_usize, 1, 2];

let mut ts = TemperatureScaling::new();
ts.fit(&logits, &labels);
println!("learned T = {:.3}", ts.temperature());

let probs = ts.predict_proba(&Vector::from_slice(&[2.0, 0.5, 0.1]));
println!("calibrated probs: {:?}", probs.as_slice());
```

### Pattern 2: Calibration metrics on binary predictions

```rust
use aprender::calibration::{expected_calibration_error, brier_score};

let predictions: Vec<f32> = vec![0.9, 0.8, 0.7, 0.6, 0.3, 0.1];
let labels: Vec<bool> = vec![true, true, false, true, false, false];

let ece = expected_calibration_error(&predictions, &labels, 5);
let brier = brier_score(&predictions, &labels);
println!("ECE={:.4}  Brier={:.4}", ece, brier);
```

## See also

- [`classification`](classification.md) — `LogisticRegression` outputs that may need calibration
- [`metrics`](metrics.md) — `accuracy`, `f1_score` complement calibration metrics
- [`tree`](tree.md) — random forests and gradient boosting are common calibration targets
- [`bayesian`](bayesian.md) — Bayesian models give well-calibrated posteriors out of the box

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
