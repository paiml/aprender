<!-- PCU: lib-regularization | contract: contracts/apr-page-lib-regularization-v1.yaml -->

# Module: `aprender::regularization`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/regularization/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/regularization/mod.rs) or directory.

## Example

```rust
use aprender::regularization::{Mixup, LabelSmoothing, CutMix, StochasticDepth};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::regularization` collects **data-level** regularizers — the
augmentation techniques that complement parameter-norm regularizers (L1 / L2
live in [`optim::prox`](optim.md)) and dropout-style layers (those live in
[`nn`](nn.md)). It exposes Mixup, CutMix, Label Smoothing, Stochastic Depth,
R-Drop, and SpecAugment, all in their canonical forms from the literature.
These techniques typically lift test accuracy by 1–3 points on image and
audio benchmarks at the cost of slower convergence.

## Key types

| Type | Description |
|------|-------------|
| `Mixup` | Sample interpolation `λ * x1 + (1-λ) * x2` with label mixing. Parameter `alpha` controls the Beta(α, α) prior over λ. |
| `LabelSmoothing` | Replace hard `1`-hot labels with `1-ε` on the true class and `ε / (K-1)` elsewhere. |
| `CutMix`, `CutMixParams` | Patch-swap augmentation between two images. |
| `StochasticDepth`, `DropMode` | Per-layer skip with linearly decaying probability. |
| `RDrop` | Consistency loss over two forward passes with dropout. |
| `SpecAugment` | Time / frequency masking for speech features. |
| `cross_entropy_with_smoothing` | Free function for label-smoothed cross-entropy. |

## Usage patterns

### Pattern 1: Mixup augmentation

```rust
use aprender::regularization::Mixup;
use aprender::primitives::Vector;

let mixup = Mixup::new(0.4);  // alpha=0.4 is a common default
let lambda = mixup.sample_lambda();

let x1 = Vector::from_slice(&[1.0, 0.0, 0.0]);
let x2 = Vector::from_slice(&[0.0, 1.0, 0.0]);
let mixed_x = mixup.mix_samples(&x1, &x2, lambda);

let y1 = Vector::from_slice(&[1.0, 0.0]);
let y2 = Vector::from_slice(&[0.0, 1.0]);
let mixed_y = mixup.mix_labels(&y1, &y2, lambda);

println!("mixed input: {:?}", mixed_x.as_slice());
println!("mixed label: {:?}", mixed_y.as_slice());
```

### Pattern 2: Label smoothing

```rust
use aprender::regularization::{LabelSmoothing, cross_entropy_with_smoothing};
use aprender::primitives::Vector;

let smoother = LabelSmoothing::new(0.1);  // 10% smoothing
let smoothed = smoother.smooth_index(2, 5);  // true class = 2 out of 5
assert!((smoothed[2] - 0.92).abs() < 1e-2);    // (1 - 0.1) + 0.1/5 = 0.92
assert!((smoothed[0] - 0.02).abs() < 1e-2);    // 0.1 / 5 = 0.02

let logits = Vector::from_slice(&[1.0, 2.0, 3.0, 1.0, 0.5]);
let loss = cross_entropy_with_smoothing(&logits, 2, 0.1);
println!("smoothed CE = {:.4}", loss);
```

## See also

- [`nn`](nn.md) — `Dropout`, `Dropout2d`, `AlphaDropout`, `DropBlock`, `DropConnect`
- [`optim`](optim.md) — `prox::soft_threshold` for L1, weight-decay flags on optimizers
- [`linear_model`](linear_model.md) — `Lasso` / `Ridge` / `ElasticNet` already bake in L1/L2
- [`loss`](loss.md) — pair label smoothing with the matching cross-entropy loss

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
