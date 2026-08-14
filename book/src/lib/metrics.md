<!-- PCU: lib-metrics | contract: contracts/apr-page-lib-metrics-v1.yaml -->

# Module: `aprender::metrics`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/metrics/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/metrics/mod.rs) or directory.

## Example

```rust
use aprender::metrics::{r_squared, mse, mae, rmse};
use aprender::metrics::classification::{accuracy, f1_score, Average};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::metrics` is the evaluation toolkit — separate from `loss` because
metrics are non-differentiable scoring functions, not training signals.
Regression metrics (`mse`, `mae`, `rmse`, `r_squared`) live at the top level;
classification metrics (`accuracy`, `precision`, `recall`, `f1_score`) live
under `metrics::classification`; clustering metrics (`silhouette_score`,
`inertia`) sit alongside them; and specialized subdomains have their own
submodules: `drift`, `perplexity`, `ranking`, `percentile`, `grad_norm`,
`evaluator`.

## Key types

| Symbol | Description |
|--------|-------------|
| `r_squared`, `mse`, `mae`, `rmse` | Regression metrics (top-level free functions). |
| `metrics::classification::accuracy` | Accuracy on `&[usize]` predictions vs targets. |
| `metrics::classification::{precision, recall, f1_score, Average}` | Per-class / macro / micro / weighted averages. |
| `silhouette_score`, `inertia` | Cluster-quality scores. |
| `metrics::ranking::*` | Information-retrieval style ranking metrics (NDCG, MAP, MRR). |
| `metrics::perplexity::*` | Language-model perplexity. |
| `metrics::drift::*` | Distribution-drift detectors (KS, PSI, etc.). |

## Usage patterns

### Pattern 1: Regression scoring

```rust
use aprender::metrics::{r_squared, mse, rmse};
use aprender::primitives::Vector;

let y_true = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);
let y_pred = Vector::from_slice(&[2.9, 5.1, 7.2, 8.8]);

let r2 = r_squared(&y_pred, &y_true);
let mse_val = mse(&y_pred, &y_true);
let rmse_val = rmse(&y_pred, &y_true);

assert!(r2 > 0.99);
println!("R²={:.4}  MSE={:.4}  RMSE={:.4}", r2, mse_val, rmse_val);
```

### Pattern 2: Classification metrics with multiple averages

```rust
use aprender::metrics::classification::{accuracy, f1_score, precision, recall, Average};

let y_true: Vec<usize> = vec![0, 0, 1, 1, 2, 2, 1, 0];
let y_pred: Vec<usize> = vec![0, 1, 1, 1, 2, 0, 1, 0];

let acc = accuracy(&y_pred, &y_true);
let f1_macro = f1_score(&y_pred, &y_true, Average::Macro);
let p_micro = precision(&y_pred, &y_true, Average::Micro);
let r_weighted = recall(&y_pred, &y_true, Average::Weighted);

println!("acc={:.3}  f1_macro={:.3}  p_micro={:.3}  r_weighted={:.3}",
    acc, f1_macro, p_micro, r_weighted);
```

## See also

- [`loss`](loss.md) — differentiable losses (use these for training, not evaluation)
- [`calibration`](calibration.md) — `expected_calibration_error`, `brier_score`, reliability diagrams
- [`model_selection`](model_selection.md) — cross-validation orchestrates these metrics over folds
- [`interpret`](interpret.md) — feature-importance scoring complements aggregate metrics

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
