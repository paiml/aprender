<!-- PCU: lib-online | contract: contracts/apr-page-lib-online-v1.yaml -->

# Module: `aprender::online`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/online/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/online/mod.rs) or directory.

## Example

<!-- example-cost: trivial -->
```rust
use aprender::online::{OnlineLearner, OnlineLinearRegression, OnlineLogisticRegression};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::online` is the streaming-learning corner of aprender. The
top-level types implement classical online algorithms — `OnlineLearner` /
`PassiveAggressive` traits backing `OnlineLinearRegression` and
`OnlineLogisticRegression`, both with configurable learning-rate decay.
The submodules cover the broader continual-learning landscape: corpus
construction, continual pretraining (`cpt`), curriculum learning, direct
preference optimisation (`dpo`), drift detection, knowledge distillation
(`distillation`, `distillation_advanced`), per-layer merge, RLHF /
reinforcement learning from verifier (`rlvr`), and tokenizer surgery.

## Key types

| Type | Description |
|------|-------------|
| `OnlineLearner` | Trait. `partial_fit(x, y)` + `predict_one(x)` for one-at-a-time updates. |
| `PassiveAggressive` | Trait for PA-I / PA-II margin-based updates. |
| `OnlineLinearRegression` | Streaming linear regression. Builder: `with_config`. |
| `OnlineLogisticRegression` | Streaming binary logistic regression. `predict_proba_one`. |
| `OnlineLearnerConfig` | Configures learning rate, decay schedule, regularization. |
| `LearningRateDecay` | Enum: constant, inverse-time, exponential, etc. |
| `online::distillation`, `online::dpo`, `online::cpt`, `online::drift` | Sub-modules for advanced workflows. |

## Usage patterns

### Pattern 1: Streaming linear regression

<!-- example-cost: skip -->
```rust
use aprender::online::OnlineLinearRegression;

let mut model = OnlineLinearRegression::new(2);

// Pretend each (x, y) arrives one at a time.
let samples = [
    (vec![1.0_f64, 2.0], 3.0_f64),
    (vec![2.0, 1.0], 4.0),
    (vec![3.0, 0.5], 6.5),
    (vec![1.5, 1.5], 4.5),
];

for (x, _y) in &samples {
    let pred = model.predict_one(x).expect("predict");
    println!("pred = {:.3}  weights = {:?}", pred, model.weights());
    // In a real loop you'd call partial_fit(x, y) here to update the model.
}
```

### Pattern 2: Logistic regression with configurable decay

<!-- example-cost: skip -->
```rust
use aprender::online::{OnlineLogisticRegression, OnlineLearnerConfig, LearningRateDecay};

let cfg = OnlineLearnerConfig::default();
let mut clf = OnlineLogisticRegression::with_config(3, cfg);

let probe = vec![0.8_f64, -0.4, 0.1];
let p = clf.predict_proba_one(&probe).expect("predict");
println!("P(y=1) = {:.4}", p);

// LearningRateDecay variants let you anneal: constant, inverse-time, etc.
let _decay = LearningRateDecay::default();
```

## See also

- [`classification`](classification.md) — batch counterparts of these online learners
- [`linear_model`](linear_model.md) — batch linear regression for comparison
- [`models`](models.md) — large language models trained via the `online::distillation` sub-module
- [`drift` (in `metrics`)](metrics.md) — distribution-drift detectors that pair with online learning

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
