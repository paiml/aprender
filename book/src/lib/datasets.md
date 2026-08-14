<!-- PCU: lib-datasets | contract: contracts/apr-cli-commands-v1.yaml -->

# Module: `aprender::datasets`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/datasets/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/datasets/mod.rs)

## Example

```rust
use aprender::datasets::{load_iris, make_blobs};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::datasets` provides the dataset generators and loaders that Pillar 1
(replace **and** beat scikit-learn) measures against. It mirrors
`sklearn.datasets`:

| function | mirrors |
|----------|---------|
| `make_blobs` | `sklearn.datasets.make_blobs` |
| `make_regression` | `sklearn.datasets.make_regression` |
| `make_classification` | `sklearn.datasets.make_classification` |
| `load_iris` | `sklearn.datasets.load_iris` |

The embedded real data — currently Iris — is sourced once from scikit-learn and
committed to the repository, so loading it has **no runtime Python or network
dependency**. That property is what lets the beat benchmarks in
`contracts/beat-sklearn-*.yaml` compare like against like without a Python
process in the loop.

Larger embedded sets (`load_digits`, `load_california_housing`) are not
implemented yet; they are tracked as a continuation of PMAT-720. This chapter
says so rather than implying a completeness the module does not have.

## See also

- [`aprender::data`](./data.md) — the columnar `DataFrame` these feed
- [`apr beat-run`](../cli/beat-run.md) — evaluates the beat contracts that
  consume these datasets
