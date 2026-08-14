<!-- PCU: lib-model_selection | contract: contracts/apr-page-lib-model_selection-v1.yaml -->

# Module: `aprender::model_selection`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/model_selection/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/model_selection/mod.rs) or directory.

## Example

<!-- example-cost: skip -->
```rust
use aprender::model_selection::{train_test_split, cross_validate, KFold, StratifiedKFold};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::model_selection` is the validation & tuning toolkit: `KFold` and
`StratifiedKFold` for fold generation, `train_test_split` for one-shot
holdout, `cross_validate` to run an `Estimator` over those folds, and
`grid_search_alpha` for tuning regularization strength. The
`CrossValidationResult` and `GridSearchResult` types pack per-fold scores
plus aggregated `mean` / `std` / `min` / `max` accessors so you can build
confidence intervals around your point estimates.

## Key types

| Type | Description |
|------|-------------|
| `KFold` | K-fold cross-validation split generator. Builder: `with_shuffle`, `with_random_state`. |
| `StratifiedKFold` | Same as `KFold` but preserves class proportions. |
| `CrossValidationResult` | Per-fold scores with `mean`, `std`, `min`, `max` reductions. |
| `GridSearchResult` | Best parameter + score from a hyperparameter sweep. |
| `cross_validate<E>` | Generic CV driver. Takes any `Estimator`. |
| `train_test_split` | One-shot holdout split. |
| `grid_search_alpha` | Bisection / linear sweep over regularization strength. |

## Usage patterns

### Pattern 1: 5-fold cross-validation of a linear model

<!-- example-cost: skip -->
```rust
use aprender::model_selection::{cross_validate, KFold};
use aprender::prelude::*;

let x = Matrix::from_vec(10, 1, (1..=10).map(|i| i as f32).collect()).expect("10x1");
let y = Vector::from_slice(&[2.1, 4.0, 6.1, 8.0, 10.1, 11.9, 14.0, 16.1, 17.9, 20.0]);

let kfold = KFold::new(5).with_shuffle(true).with_random_state(42);
let model_factory = || LinearRegression::new();

let result = cross_validate(model_factory, &x, &y, &kfold)
    .expect("cv runs cleanly");
println!("cv R²: mean={:.4}  std={:.4}", result.mean(), result.std());
```

### Pattern 2: Stratified split for classification

<!-- example-cost: skip -->
```rust
use aprender::model_selection::StratifiedKFold;
use aprender::primitives::Vector;

let y = Vector::from_slice(&[
    0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 1.0,
    2.0, 2.0,
]);

let skf = StratifiedKFold::new(3)
    .with_shuffle(true)
    .with_random_state(7);

let folds = skf.split(&y);
for (i, (train_idx, val_idx)) in folds.iter().enumerate() {
    println!("fold {}: train={}, val={}", i, train_idx.len(), val_idx.len());
}
```

## See also

- [`metrics`](metrics.md) — the score functions invoked by `cross_validate`
- [`linear_model`](linear_model.md) — `grid_search_alpha` is tailored for Ridge / Lasso / ElasticNet
- [`traits`](traits.md) — `cross_validate` is generic over `Estimator`
- [`automl`](automl.md) — higher-level automation that bundles model selection + tuning

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
