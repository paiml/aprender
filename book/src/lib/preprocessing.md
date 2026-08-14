<!-- PCU: lib-preprocessing | contract: contracts/apr-page-lib-preprocessing-v1.yaml -->

# Module: `aprender::preprocessing`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/preprocessing/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/preprocessing/mod.rs) or directory.

## Example

<!-- example-cost: trivial -->
```rust
use aprender::preprocessing::{StandardScaler, MinMaxScaler, RobustScaler, PCA, TSNE};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::preprocessing` is the feature-engineering layer. It owns the three
canonical scalers (`StandardScaler` for z-score normalisation, `MinMaxScaler`
for range scaling, `RobustScaler` for median + IQR), plus linear (PCA) and
non-linear (t-SNE) dimensionality reduction. All of these implement
[`Transformer`](traits.md) so they slot into pipelines, follow the
fit / transform / fit_transform contract, and support `inverse_transform`
where mathematically defined.

## Key types

| Type | Description |
|------|-------------|
| `StandardScaler` | Zero-mean, unit-variance scaling. Builder: `with_mean`, `with_std`. Supports `inverse_transform`. |
| `MinMaxScaler` | Linear rescaling to `[min, max]`. Builder: `with_range`. |
| `RobustScaler` | Median + IQR scaling that is insensitive to outliers. Builder: `with_centering`, `with_scaling`. |
| `PCA` | Principal Component Analysis. `explained_variance`, `explained_variance_ratio`, and `components` accessors. |
| `TSNE` | t-distributed Stochastic Neighbor Embedding for 2-D / 3-D visualisation. |

## Usage patterns

### Pattern 1: StandardScaler in a pipeline

```rust
use aprender::preprocessing::StandardScaler;
use aprender::primitives::Matrix;
use aprender::traits::Transformer;

let x = Matrix::from_vec(4, 2, vec![
    1.0, 100.0,
    2.0, 200.0,
    3.0, 300.0,
    4.0, 400.0,
]).expect("4x2");

let mut scaler = StandardScaler::new();
let x_scaled = scaler.fit_transform(&x).expect("standardize");

// After fitting, each column should have ~zero mean and ~unit variance.
println!("means: {:?}", scaler.mean());
println!("stds:  {:?}", scaler.std());

// Round-trip via inverse_transform.
let recovered = scaler.inverse_transform(&x_scaled).expect("inverse");
println!("recovered[0,0] = {:.2}", recovered.get(0, 0));
```

### Pattern 2: PCA dimensionality reduction

```rust
use aprender::preprocessing::PCA;
use aprender::primitives::Matrix;
use aprender::traits::Transformer;

// 6 samples in 3-D, projected down to 2 principal components.
let x = Matrix::from_vec(6, 3, vec![
    1.0, 2.0, 3.0,
    2.0, 4.0, 6.0,
    3.0, 6.0, 9.0,
    1.1, 2.1, 3.1,
    2.2, 4.2, 6.2,
    3.3, 6.3, 9.3,
]).expect("6x3");

let mut pca = PCA::new(2);
let x_low = pca.fit_transform(&x).expect("PCA fit_transform");
assert_eq!(x_low.shape(), (6, 2));

if let Some(ratio) = pca.explained_variance_ratio() {
    println!("explained variance ratio: {:?}", ratio);
}
```

## See also

- [`traits`](traits.md) — `Transformer` contract these types implement
- [`decomposition`](decomposition.md) — `ICA` for blind source separation
- [`data`](data.md) — `DataFrame` as the input pipeline upstream of scaling
- [`cluster`](cluster.md) — scale features before clustering for stable results

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
