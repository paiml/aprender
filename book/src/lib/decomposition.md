<!-- PCU: lib-decomposition | contract: contracts/apr-page-lib-decomposition-v1.yaml -->

# Module: `aprender::decomposition`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/decomposition/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/decomposition/mod.rs) or directory.

## Example

```rust
use aprender::decomposition::ICA;
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::decomposition` houses matrix-decomposition–based unsupervised
methods. Today it exposes Independent Component Analysis (`ICA`) — the
go-to algorithm for blind source separation and recovering statistically
independent factors. PCA, the closely related linear-decomposition method,
lives in [`preprocessing`](preprocessing.md) because it is most commonly used
as a feature transformer.

## Key types

| Type | Description |
|------|-------------|
| `ICA` | Independent Component Analysis (FastICA-style). Builder methods: `with_max_iter`, `with_tolerance`, `with_random_state`. |

## Usage patterns

### Pattern 1: Recover independent components

```rust
use aprender::decomposition::ICA;
use aprender::primitives::Matrix;

// Suppose `mixed` is a (n_samples, n_features) matrix of mixed signals.
let mixed = Matrix::from_vec(8, 2, vec![
    1.1, 0.2, -0.8, 0.9, 0.4, -1.1, 1.5, 0.3,
    -0.7, 0.8, 1.2, -0.5, -0.3, 1.0, 0.6, -0.9,
]).expect("8x2 matrix");

let mut ica = ICA::new(2)
    .with_max_iter(200)
    .with_tolerance(1e-4)
    .with_random_state(42);
ica.fit(&mixed).expect("ica fit");

let components = ica.transform(&mixed).expect("transform");
assert_eq!(components.shape(), (8, 2));
```

### Pattern 2: Configure ICA reproducibly for an experiment

```rust
use aprender::decomposition::ICA;

// Two reproducible runs with the same seed should produce identical mixings.
let ica_a = ICA::new(3)
    .with_random_state(123)
    .with_max_iter(500)
    .with_tolerance(1e-6);

let ica_b = ICA::new(3)
    .with_random_state(123)
    .with_max_iter(500)
    .with_tolerance(1e-6);

// Both estimators have identical hyperparameters, so given the same data
// they will converge to bit-identical un-mixing matrices.
let _ = (ica_a, ica_b);
```

## See also

- [`preprocessing`](preprocessing.md) — `PCA` (linear) and `TSNE` (non-linear) live here
- [`cluster`](cluster.md) — common downstream consumer of decomposed features
- [`primitives`](primitives.md) — `Matrix` shape conventions consumed by `fit` / `transform`

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
