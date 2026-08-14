<!-- PCU: lib-optim | contract: contracts/apr-page-lib-optim-v1.yaml -->

# Module: `aprender::optim`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/optim/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/optim/mod.rs) or directory.

## Example

```rust
use aprender::optim::{Adam, SGD, Optimizer};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::optim` is the unified optimization layer. It exposes both
**stochastic** optimizers (SGD, Adam) used in mini-batch deep-learning loops
and **batch** optimizers (L-BFGS, Conjugate Gradient, Damped Newton,
FISTA, Augmented Lagrangian, ADMM, Interior Point) used in convex / classical
ML. A single `Optimizer` trait fronts both modes: `step` updates parameters
in-place from a mini-batch gradient; `minimize` runs a full deterministic
optimization given an objective and gradient closure.

## Key types

| Type | Description |
|------|-------------|
| `Optimizer` | Unified trait. Implementors decide whether they support `step`, `minimize`, or both. |
| `SGD`, `Adam` | Stochastic optimizers for mini-batch training. SGD supports momentum. |
| `LBFGS`, `ConjugateGradient`, `DampedNewton` | Batch optimizers with deterministic line search. |
| `FISTA`, `ADMM`, `AugmentedLagrangian`, `InteriorPoint` | Convex / constrained optimization solvers. |
| `OptimizationResult`, `ConvergenceStatus` | Return type from `minimize`. |
| `prox::soft_threshold`, `prox::project_l2_ball`, `prox::project_box`, `prox::nonnegative` | Proximal operators (the `optim::prox` submodule). |

## Usage patterns

### Pattern 1: Adam on a small parameter vector

```rust
use aprender::optim::{Adam, Optimizer};
use aprender::primitives::Vector;

let mut opt = Adam::new(0.01);
let mut params = Vector::from_slice(&[1.0, 2.0, 3.0]);

for _step in 0..50 {
    // In real training, gradients come from autograd or a closure.
    let grad = Vector::from_slice(&[0.1, -0.1, 0.05]);
    opt.step(&mut params, &grad);
}
println!("final params: {:?}", params.as_slice());
```

### Pattern 2: L-BFGS minimization of a quadratic

```rust
use aprender::optim::{LBFGS, Optimizer, ConvergenceStatus};
use aprender::primitives::Vector;

let mut lbfgs = LBFGS::new(100, 1e-5, 10);
let objective = |x: &Vector<f32>| (x[0] - 5.0).powi(2);
let gradient = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 5.0)]);

let result = lbfgs.minimize(objective, gradient, Vector::from_slice(&[0.0]));
assert_eq!(result.status, ConvergenceStatus::Converged);
println!("solution: {:?}", result.solution.as_slice());
```

## See also

- [`loss`](loss.md) — loss functions whose gradients feed `step`
- [`nn`](nn.md) — module-level `nn::optim` mirrors this for `Module` parameter trees
- [`linear_model`](linear_model.md) — uses Cholesky + coordinate descent + FISTA internally
- [`regularization`](regularization.md) — pairs with proximal operators (`prox::soft_threshold` for L1)

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
