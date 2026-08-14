<!-- PCU: lib-linear_model | contract: contracts/apr-page-lib-linear_model-v1.yaml -->

# Module: `aprender::linear_model`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/linear_model/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/linear_model/mod.rs) or directory.

## Example

<!-- example-cost: trivial -->
```rust
use aprender::linear_model;
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::linear_model` provides ordinary least squares regression plus the
three regularized linear models that cover the vast majority of tabular
regression problems: ridge (L2), lasso (L1), and elastic net (L1 + L2). The
solvers are deterministic, single-threaded, and ship with closed-form solutions
where possible (Cholesky for OLS/Ridge, coordinate descent for Lasso). Reach
for this module when you need an interpretable linear baseline, a fast pre-NN
sanity check, or a feature-selection-friendly model via lasso sparsity.

## Key types

| Type | Description |
|------|-------------|
| `LinearRegression` | OLS regression via normal equations + Cholesky. Fits in O(p³). |
| `Ridge` | OLS with L2 penalty on coefficients. Closed-form solution with a regularization parameter `alpha`. |
| `Lasso` | OLS with L1 penalty, solved via coordinate descent. Produces sparse coefficient vectors. |
| `ElasticNet` | Combined L1 + L2 penalty. Tunable mixing ratio `l1_ratio` between 0 and 1. |

All four implement the [`Estimator`](traits.md) trait, so they share the same
`fit` / `predict` / `score` surface.

## Usage patterns

### Pattern 1: OLS regression with R² scoring

```rust
use aprender::prelude::*;

let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0])
    .expect("valid 4x1 matrix");
let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

let mut model = LinearRegression::new();
model.fit(&x, &y).expect("fit succeeds on well-conditioned data");

let preds = model.predict(&x);
let r2 = model.score(&x, &y);
assert!(r2 > 0.99, "perfectly linear data must score near 1.0");
println!("predictions: {:?}", preds.as_slice());
```

### Pattern 2: Lasso for feature selection

```rust
use aprender::prelude::*;

// 6 samples, 3 features — only the first feature is informative
let x = Matrix::from_vec(6, 3, vec![
    1.0, 0.5, 0.2,
    2.0, 0.6, 0.1,
    3.0, 0.7, 0.3,
    4.0, 0.5, 0.2,
    5.0, 0.6, 0.4,
    6.0, 0.5, 0.1,
]).expect("valid 6x3 matrix");
let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);

let mut lasso = Lasso::new(0.1);
lasso.fit(&x, &y).expect("lasso fit");

// Lasso zeroes out coefficients of uninformative features.
let coefs = lasso.coefficients();
println!("non-zero coefficients: {}",
    coefs.as_slice().iter().filter(|&&c| c.abs() > 1e-3).count());
```

## See also

- [`optim`](optim.md) — solvers (Cholesky, coordinate descent, FISTA) used internally
- [`loss`](loss.md) — MSE/MAE losses that drive the underlying optimization
- [`metrics`](metrics.md) — `r_squared`, `mse`, `rmse` used by `score()`
- [`regularization`](regularization.md) — broader regularization techniques beyond linear models
- [`model_selection`](model_selection.md) — `grid_search_alpha` for tuning the regularization strength

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
