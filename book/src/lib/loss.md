<!-- PCU: lib-loss | contract: contracts/apr-page-lib-loss-v1.yaml -->

# Module: `aprender::loss`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/loss/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/loss/mod.rs) or directory.

## Example

<!-- example-cost: skip -->
```rust
use aprender::loss::{MSELoss, MAELoss, HuberLoss, Loss};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::loss` exposes the classic regression losses (MSE, MAE, Huber) and
the categorical cross-entropy loss used by classifiers, both as free
functions (`mse_loss`, `mae_loss`, `huber_loss`, `cross_entropy_loss`,
`triplet_loss`) and as types implementing the `Loss` trait so they can be
passed around generically by optimizers and training loops.

## Key types

| Type | Description |
|------|-------------|
| `Loss` | Trait. `forward(y_pred, y_true) -> f32` and (where supported) `backward`. |
| `MSELoss` | Mean Squared Error. Smooth, convex; the default for regression. |
| `MAELoss` | Mean Absolute Error. Robust to outliers but non-differentiable at 0. |
| `HuberLoss` | Quadratic near 0, linear past `delta`; combines MSE smoothness with MAE robustness. |
| `mse_loss`, `mae_loss`, `huber_loss`, `cross_entropy_loss`, `triplet_loss` | Free-function entry points. |

## Usage patterns

### Pattern 1: Compute MSE and MAE on a single prediction vector

<!-- example-cost: skip -->
```rust
use aprender::loss::{mse_loss, mae_loss, huber_loss};
use aprender::primitives::Vector;

let y_pred = Vector::from_slice(&[1.0, 2.0, 3.0]);
let y_true = Vector::from_slice(&[1.5, 2.0, 2.5]);

let mse = mse_loss(&y_pred, &y_true);
let mae = mae_loss(&y_pred, &y_true);
let huber = huber_loss(&y_pred, &y_true, 1.0);

println!("mse={:.4}  mae={:.4}  huber={:.4}", mse, mae, huber);
assert!(mse > 0.0 && mae > 0.0);
```

### Pattern 2: Pass losses to generic code via the `Loss` trait

<!-- example-cost: skip -->
```rust
use aprender::loss::{Loss, MSELoss, HuberLoss};
use aprender::primitives::Vector;

fn evaluate<L: Loss>(loss: &L, y_pred: &Vector<f32>, y_true: &Vector<f32>) -> f32 {
    loss.forward(y_pred, y_true)
}

let y_pred = Vector::from_slice(&[0.5, 1.5, 2.5]);
let y_true = Vector::from_slice(&[1.0, 1.0, 3.0]);

let mse = evaluate(&MSELoss, &y_pred, &y_true);
let huber = evaluate(&HuberLoss { delta: 1.0 }, &y_pred, &y_true);
println!("MSE={:.3}  Huber={:.3}", mse, huber);
```

## See also

- [`optim`](optim.md) — optimizers that minimise these losses
- [`metrics`](metrics.md) — closely related scoring functions for evaluation (not training)
- [`autograd`](autograd.md) — auto-differentiation of loss expressions over `Tensor`
- [`nn`](nn.md) — `nn::loss` re-exports loss types for module-level training

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
