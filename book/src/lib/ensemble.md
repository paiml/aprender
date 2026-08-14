<!-- PCU: lib-ensemble | contract: contracts/apr-page-lib-ensemble-v1.yaml -->

# Module: `aprender::ensemble`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/ensemble/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/ensemble/mod.rs) or directory.

## Example

```rust
use aprender::ensemble::{MixtureOfExperts, SoftmaxGating, MoeConfig};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::ensemble` is the mixture-of-experts (MoE) corner of aprender. It
exposes a gating-network trait, a softmax gate, an `MoeConfig` builder for
top-k / capacity-factor / load-balance settings, and the `MixtureOfExperts`
container that routes inputs to expert estimators. Bagging-style ensembles
(random forest, gradient boosting) live in [`tree`](tree.md); this module
focuses on learned, differentiable routing — the workhorse of modern
sparsely-activated LLMs.

## Key types

| Type | Description |
|------|-------------|
| `MixtureOfExperts<E, G>` | Generic MoE that wraps any `Estimator` experts behind a `GatingNetwork`. |
| `MoeConfig` | Builder for top-k routing, capacity factor, expert dropout, load-balance weight. |
| `GatingNetwork` | Trait implemented by routers (softmax, dense, etc.). |
| `SoftmaxGating` | Reference gating network using a softmax over expert logits. |
| `MoeBuilder` | Fluent builder for assembling experts + gating + config. |

## Usage patterns

### Pattern 1: Configure an MoE

```rust
use aprender::ensemble::MoeConfig;

let cfg = MoeConfig::default()
    .with_top_k(2)
    .with_capacity_factor(1.25)
    .with_expert_dropout(0.0)
    .with_load_balance_weight(0.01);

println!("top_k={}, capacity={}", cfg.top_k, cfg.capacity_factor);
```

### Pattern 2: Inspect routing weights

```rust
use aprender::ensemble::{MixtureOfExperts, SoftmaxGating, MoeConfig};
// `MixtureOfExperts::<Expert, Gating>` is generic. In practice you assemble
// via the builder and supply concrete expert estimators; see
// `crates/aprender-core/examples/moe_*.rs` for full end-to-end demos.

// After fit, you can query the routing weights for a sample:
//     let weights = moe.get_routing_weights(&input_vec);
//     let usage = moe.expert_usage(&inputs_matrix);
// The load-balance auxiliary loss is exposed via:
//     let lb = moe.compute_load_balance_loss(&inputs_matrix);
```

## See also

- [`tree`](tree.md) — `RandomForestClassifier` / `RandomForestRegressor`, `GradientBoostingClassifier`
- [`models`](models.md) — production transformer / Qwen2 implementations that use MoE internally
- [`nn`](nn.md) — neural network primitives the gating networks are built from

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
