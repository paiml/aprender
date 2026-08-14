<!-- PCU: lib-interpret | contract: contracts/apr-page-lib-interpret-v1.yaml -->

# Module: `aprender::interpret`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/interpret/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/interpret/mod.rs) or directory.

## Example

<!-- example-cost: trivial -->
```rust
use aprender::interpret::{Explainer, ShapExplainer, PermutationImportance, FeatureContributions};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::interpret` is the explainable-AI layer. It exposes the
`Explainer` trait plus four post-hoc explainers: `ShapExplainer` (KernelSHAP-
style sampled approximation), `PermutationImportance` (Breiman 2001),
`IntegratedGradients` (Sundararajan et al. 2017 for differentiable models),
and `FeatureContributions` (closed-form decomposition for linear models).
These produce per-feature attribution scores that answer "why did the model
make this prediction?"

## Key types

| Type | Description |
|------|-------------|
| `Explainer` | Trait defining `explain(sample) -> Vector<f32>` for per-feature attributions. |
| `ShapExplainer` | Sampled-SHAP estimator over a background distribution. |
| `PermutationImportance` | Permutation-based feature importance. Static `compute(predict, x, y, score)` constructor. |
| `IntegratedGradients` | Path-integral attributions for differentiable models. |
| `FeatureContributions` | Linear / additive feature attributions. `from_linear(weights, features, bias)` for instant decomposition. `top_features(k)` / `verify_sum(tol)` accessors. |

## Usage patterns

### Pattern 1: Closed-form attribution for a linear model

<!-- example-cost: skip -->
```rust
use aprender::interpret::FeatureContributions;
use aprender::primitives::Vector;

let weights = Vector::from_slice(&[2.0, -1.0, 0.5]);
let features = Vector::from_slice(&[3.0, 2.0, 4.0]);
let bias = 0.1;

let contributions = FeatureContributions::from_linear(&weights, &features, bias);
// Top 2 most influential features (by absolute contribution).
let top = contributions.top_features(2);
for (idx, contrib) in &top {
    println!("feature {} contributes {:.3}", idx, contrib);
}

// Sanity-check that contributions + bias ≈ model output.
assert!(contributions.verify_sum(1e-5));
```

### Pattern 2: Permutation importance

<!-- example-cost: skip -->
```rust
use aprender::interpret::PermutationImportance;
use aprender::primitives::Vector;

let predict_fn = |x: &Vector<f32>| -> f32 { 2.0 * x[0] + 0.5 * x[1] };
let score_fn = |preds: &[f32], y: &[f32]| -> f32 {
    let mse: f32 = preds.iter().zip(y).map(|(p, t)| (p - t).powi(2)).sum::<f32>()
        / preds.len() as f32;
    1.0 - mse  // higher is better
};

let x = vec![
    Vector::from_slice(&[1.0, 2.0]),
    Vector::from_slice(&[2.0, 1.0]),
    Vector::from_slice(&[3.0, 0.0]),
];
let y = vec![3.0_f32, 4.5, 6.0];

let pi = PermutationImportance::compute(predict_fn, &x, &y, score_fn);
let ranking = pi.ranking();
println!("feature ranking (most→least important): {:?}", ranking);
```

## See also

- [`explainable`](explainable.md) — broader explainability tooling beyond per-sample attributions
- [`metrics`](metrics.md) — aggregate scoring that complements per-feature attributions
- [`linear_model`](linear_model.md) — linear models pair naturally with `FeatureContributions::from_linear`
- [`tree`](tree.md) — tree models with built-in feature-importance accessors

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
