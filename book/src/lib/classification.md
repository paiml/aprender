<!-- PCU: lib-classification | contract: contracts/apr-page-lib-classification-v1.yaml -->

# Module: `aprender::classification`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/classification/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/classification/mod.rs) or directory.

## Example

<!-- example-cost: skip -->
```rust
use aprender::classification::LogisticRegression;
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::classification` collects the linear and instance-based classifiers
that complement tree ensembles for tabular problems: logistic regression, the
linear support vector machine, Gaussian Naive Bayes, and k-nearest neighbours.
All four return `usize` class indices and follow the same `fit` / `predict`
contract — `predict` here returns `Vec<usize>`, not a `Vector<f32>`, because
the targets are discrete labels.

## Key types

| Type | Description |
|------|-------------|
| `LogisticRegression` | Multinomial logistic regression with optional class weighting (`ClassWeight`) and gradient-descent or L-BFGS fit modes (`FitMode`). |
| `LinearSVM` | Hinge-loss SVM solved with subgradient descent; sparse and fast on high-dimensional inputs. |
| `GaussianNB` | Gaussian Naive Bayes with per-class mean / variance estimates; closed-form, no iterative training. |
| `KNearestNeighbors` | K-nearest neighbours with selectable `DistanceMetric` (Euclidean, Manhattan, Cosine). |
| `ClassWeight` | Enum that controls class re-weighting in logistic regression (`Balanced` / `Custom`). |

## Usage patterns

### Pattern 1: Logistic regression on separable data

<!-- example-cost: skip -->
```rust
use aprender::metrics::classification::accuracy;
use aprender::prelude::*;

let x = Matrix::from_vec(6, 2, vec![
    1.0, 2.0, 2.0, 3.0, 3.0, 1.0,    // class 0
    6.0, 5.0, 7.0, 8.0, 8.0, 6.0,    // class 1
]).expect("valid 6x2 matrix");
let y: Vec<usize> = vec![0, 0, 0, 1, 1, 1];

let mut lr = LogisticRegression::new()
    .with_learning_rate(0.1)
    .with_max_iter(1000);
lr.fit(&x, &y).expect("logistic regression fit");

let preds = lr.predict(&x);
let acc = accuracy(&preds, &y);
assert!(acc >= 0.8, "training accuracy contract");
println!("accuracy: {:.2}", acc);
```

### Pattern 2: KNN with custom distance metric

<!-- example-cost: skip -->
```rust
use aprender::classification::{DistanceMetric, KNearestNeighbors};
use aprender::primitives::Matrix;

let x = Matrix::from_vec(4, 2, vec![
    0.0, 0.0,
    0.0, 1.0,
    5.0, 5.0,
    5.0, 6.0,
]).expect("valid 4x2 matrix");
let y = vec![0_usize, 0, 1, 1];

let mut knn = KNearestNeighbors::new(3)
    .with_metric(DistanceMetric::Euclidean);
knn.fit(&x, &y).expect("knn fit");

let probe = Matrix::from_vec(1, 2, vec![0.1, 0.1]).expect("probe matrix");
let prediction = knn.predict(&probe);
assert_eq!(prediction[0], 0, "near origin → class 0");
```

## See also

- [`tree`](tree.md) — decision tree / random forest classifiers
- [`ensemble`](ensemble.md) — boosted classifiers (`MixtureOfExperts`, etc.)
- [`metrics`](metrics.md) — `accuracy`, precision, recall, F1 in `metrics::classification`
- [`calibration`](calibration.md) — `TemperatureScaling` / `PlattScaling` to calibrate probabilistic outputs
- [`model_selection`](model_selection.md) — cross-validation for picking `k`, learning rate, or `C`

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
