<!-- PCU: lib-pipeline | contract: contracts/apr-cli-commands-v1.yaml -->

# Module: `aprender::pipeline`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/pipeline.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/pipeline.rs)

## Example

```rust
use aprender::pipeline::Pipeline;
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::pipeline` provides `Pipeline`, which chains transformers and ends in
a single estimator. It mirrors `sklearn.pipeline.Pipeline`:

- **`fit`** — each transformer is fit, then applied, in sequence; the final
  estimator is fit on the fully transformed data.
- **`predict` / `score`** — the same transformer chain is applied in
  transform-only mode before delegating to the estimator.

That asymmetry is the point of the type. Fitting a scaler on data that has
already been through the test-time path, or scoring against a scaler fit on the
scoring data, is the classic leakage bug; routing both through one object makes
it hard to write by accident.

Steps use trait objects (`Box<dyn Transformer>` and `Box<dyn Estimator>`) so a
pipeline can be heterogeneous — for example `StandardScaler` followed by
`LogisticRegression`.

## See also

- [`aprender::traits`](./traits.md) — the `Transformer` and `Estimator` traits a
  step must implement
- [`aprender::preprocessing`](./preprocessing.md) — the transformers most
  commonly used as steps
