<!-- PCU: lib-tree | contract: contracts/apr-page-lib-tree-v1.yaml -->

# Module: `aprender::tree`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/tree/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/tree/mod.rs) or directory.

## Example

```rust
use aprender::tree::{DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::tree` is the decision-tree family. It owns CART-style decision
trees for both classification (`DecisionTreeClassifier`) and regression
(`DecisionTreeRegressor`), plus the ensemble methods that wrap them: random
forests (`RandomForestClassifier`, `RandomForestRegressor`) and gradient
boosting (`GradientBoostingClassifier`). These are the workhorses for
tabular data — fast to train, interpretable splits, no need to scale features.

## Key types

| Type | Description |
|------|-------------|
| `DecisionTreeClassifier` | CART classifier with Gini impurity, configurable max depth and min samples. |
| `DecisionTreeRegressor` | CART regressor with MSE splits. |
| `RandomForestClassifier` | Bagged forest with bootstrap + feature subsampling. |
| `RandomForestRegressor` | Random-forest regressor. |
| `GradientBoostingClassifier` | Stage-wise additive boosting with shallow trees. |
| `gini_impurity`, `gini_split` | Low-level helpers re-exported for custom split algorithms. |
| `Node`, `Leaf`, `TreeNode` | Internal tree-node types exposed for inspection. |

## Usage patterns

### Pattern 1: Decision tree with controlled depth

```rust
use aprender::metrics::classification::accuracy;
use aprender::prelude::*;

let x = Matrix::from_vec(6, 2, vec![
    1.0, 2.0, 2.0, 3.0, 3.0, 1.0,
    6.0, 5.0, 7.0, 8.0, 8.0, 6.0,
]).expect("6x2");
let y: Vec<usize> = vec![0, 0, 0, 1, 1, 1];

let mut tree = DecisionTreeClassifier::new().with_max_depth(3);
tree.fit(&x, &y).expect("decision tree fit");

let preds = tree.predict(&x);
let acc = accuracy(&preds, &y);
assert!(acc >= 0.8, "training accuracy contract");
```

### Pattern 2: Random-forest regressor

```rust
use aprender::tree::RandomForestRegressor;
use aprender::primitives::{Matrix, Vector};
use aprender::traits::Estimator;

let x = Matrix::from_vec(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("6x1");
let y = Vector::from_slice(&[1.5, 3.1, 4.4, 6.0, 7.7, 9.5]);

let mut rf = RandomForestRegressor::new(50);
rf.fit(&x, &y).expect("rf fit");

let probe = Matrix::from_vec(1, 1, vec![3.5]).expect("1x1");
let pred = rf.predict(&probe);
println!("rf(3.5) ≈ {:.2}", pred[0]);
```

## See also

- [`ensemble`](ensemble.md) — mixture-of-experts ensembles (differentiable routing)
- [`classification`](classification.md) — linear / instance-based classifiers
- [`linear_model`](linear_model.md) — linear regression baselines for tabular data
- [`metrics`](metrics.md) — `accuracy`, `r_squared` to evaluate tree models

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
