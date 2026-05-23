<!-- PCU: lib-traits | contract: contracts/apr-page-lib-traits-v1.yaml -->

# Module: `aprender::traits`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/traits.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/traits.rs) or directory.

## Example

```rust
use aprender::traits::{Estimator, Transformer, UnsupervisedEstimator};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::traits` defines the three abstract contracts every ML algorithm in
the framework satisfies: `Estimator` for supervised learning (regression,
classification), `UnsupervisedEstimator` for clustering / anomaly detection,
and `Transformer` for stateful preprocessing. Every model elsewhere in
aprender — linear regression, random forest, KMeans, scalers — implements at
least one of these traits, which is what makes pipelines and cross-validation
generic over algorithm choice.

## Key types

| Type | Description |
|------|-------------|
| `Estimator` | Supervised learners: `fit(x, y)`, `predict(x) -> Vector<f32>`, `score(x, y) -> f32`. |
| `UnsupervisedEstimator` | Unsupervised learners with associated `Labels` type. `fit(x)`, `predict(x) -> Self::Labels`. |
| `Transformer` | Stateful preprocessors with `fit`, `transform`, and a default `fit_transform`. |

These are also re-exported at the crate root (`aprender::{Estimator,
Transformer, UnsupervisedEstimator}`) and through `aprender::prelude::*`.

## Usage patterns

### Pattern 1: Writing generic code over `Estimator`

```rust
use aprender::prelude::*;
use aprender::traits::Estimator;

fn fit_and_score<E: Estimator>(
    model: &mut E,
    x: &Matrix<f32>,
    y: &Vector<f32>,
) -> f32 {
    model.fit(x, y).expect("fit succeeds on valid data");
    model.score(x, y)
}

let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("4x1");
let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

let mut lr = LinearRegression::new();
let score = fit_and_score(&mut lr, &x, &y);
assert!(score > 0.99);
```

### Pattern 2: Implementing `Transformer`

```rust
use aprender::primitives::Matrix;
use aprender::traits::Transformer;
use aprender::error::Result;

struct Identity;

impl Transformer for Identity {
    fn fit(&mut self, _x: &Matrix<f32>) -> Result<()> { Ok(()) }
    fn transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>> {
        let data: Vec<f32> = x.as_slice().to_vec();
        Matrix::from_vec(x.n_rows(), x.n_cols(), data)
            .map_err(|e| aprender::error::AprenderError::ValidationError {
                message: e.to_string(),
            })
    }
}

// The default `fit_transform` is provided automatically.
let mut t = Identity;
let x = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("2x2");
let y = t.fit_transform(&x).expect("identity transform");
assert_eq!(y.n_rows(), 2);
```

## See also

- [`primitives`](primitives.md) — `Matrix` and `Vector` used in every signature
- [`error`](../lib/error.md) — `AprenderError` returned by all fallible trait methods
- [`linear_model`](linear_model.md) — flagship `Estimator` implementations
- [`preprocessing`](preprocessing.md) — flagship `Transformer` implementations
- [`cluster`](cluster.md) — flagship `UnsupervisedEstimator` implementations

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
