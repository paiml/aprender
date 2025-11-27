# Case Study: Mixture of Experts (MoE)

This case study demonstrates specialized ensemble learning using Mixture of Experts architecture. MoE enables multiple expert models with a learnable gating network that routes inputs to the most appropriate expert(s).

## Overview

```text
Input --> Gating Network --> Expert Weights
                 |
          +------+------+
          v      v      v
       Expert0 Expert1 Expert2
          v      v      v
          +------+------+
                 v
        Weighted Output
```

**Key Benefits:**
- **Specialization**: Each expert focuses on a subset of the problem
- **Conditional Compute**: Only top-k experts execute per input (sparse MoE)
- **Scalability**: Add experts without retraining others

## Quick Start

### Basic MoE with RandomForest Experts

```rust
use aprender::ensemble::{MixtureOfExperts, MoeConfig, SoftmaxGating};
use aprender::tree::RandomForestClassifier;

// Create gating network (routes inputs to experts)
let gating = SoftmaxGating::new(n_features, n_experts);

// Build MoE with 3 expert classifiers
let moe = MixtureOfExperts::builder()
    .gating(gating)
    .expert(RandomForestClassifier::new(100, 10))  // scope expert
    .expert(RandomForestClassifier::new(100, 10))  // type expert
    .expert(RandomForestClassifier::new(100, 10))  // method expert
    .config(MoeConfig::default().with_top_k(2))    // sparse: top 2
    .build()?;

// Predict (weighted combination of expert outputs)
let output = moe.predict(&input);
```

### Configuring MoE Behavior

```rust
let config = MoeConfig::default()
    .with_top_k(2)              // Activate top 2 experts per input
    .with_capacity_factor(1.25) // Load balancing headroom
    .with_expert_dropout(0.1)   // Regularization during training
    .with_load_balance_weight(0.01); // Encourage even expert usage
```

## Gating Networks

### SoftmaxGating

The default gating mechanism uses softmax over learned weights:

```rust
// Create gating: 4 input features, 3 experts
let gating = SoftmaxGating::new(4, 3);

// Temperature controls distribution sharpness
let sharp_gating = SoftmaxGating::new(4, 3).with_temperature(0.1);  // peaked
let uniform_gating = SoftmaxGating::new(4, 3).with_temperature(10.0); // uniform

// Get expert weights for input
let weights = gating.forward(&[1.0, 2.0, 3.0, 4.0]);
// weights: [0.2, 0.5, 0.3] (sums to 1.0)
```

### Custom Gating Networks

Implement the `GatingNetwork` trait for custom routing:

```rust
pub trait GatingNetwork: Send + Sync {
    fn forward(&self, x: &[f32]) -> Vec<f32>;
    fn n_features(&self) -> usize;
    fn n_experts(&self) -> usize;
}
```

## Persistence

### Binary Format (bincode)

```rust
// Save
moe.save("model.bin")?;

// Load
let loaded = MixtureOfExperts::<MyExpert, SoftmaxGating>::load("model.bin")?;
```

### APR Format (with header)

```rust
// Save with .apr header (ModelType::MixtureOfExperts = 0x0040)
moe.save_apr("model.apr")?;

// Verify format
let bytes = std::fs::read("model.apr")?;
assert_eq!(&bytes[0..4], b"APRN");
```

### Bundled Architecture

MoE uses **bundled persistence** - one `.apr` file contains everything:

```text
model.apr
├── Header (ModelType::MixtureOfExperts)
├── Metadata (MoeConfig)
└── Payload
    ├── Gating Network
    └── Experts[0..n]
```

**Benefits:**
- Atomic save/load (no partial states)
- Single file deployment
- Checksummed integrity

## Use Case: Error Classification

From GitHub issue #101 - depyler-oracle transpiler error classification:

```rust
// Problem: Single RandomForest handles all error types equally
// Solution: Specialized experts per error category

let moe = MixtureOfExperts::builder()
    .gating(SoftmaxGating::new(feature_dim, 3))
    .expert(scope_expert)   // E0425, E0412 (variable/import)
    .expert(type_expert)    // E0308, E0277 (casts, traits)
    .expert(method_expert)  // E0599 (API mapping)
    .config(MoeConfig::default().with_top_k(1))
    .build()?;

// Each expert specializes, improving accuracy on edge cases
```

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 1 | Experts activated per input |
| `capacity_factor` | 1.0 | Load balancing capacity multiplier |
| `expert_dropout` | 0.0 | Expert dropout rate (training) |
| `load_balance_weight` | 0.01 | Auxiliary loss weight |

## Performance

- **Sparse Routing**: Only `top_k` experts execute per input
- **Conditional Compute**: O(top_k) instead of O(n_experts)
- **Serialization**: ~1ms save/load for typical ensembles

## References

- [Outrageously Large Neural Networks (Shazeer et al., 2017)](https://arxiv.org/abs/1701.06538)
- [Switch Transformers (Fedus et al., 2022)](https://arxiv.org/abs/2101.03961)
- [Model Format Spec](../../../docs/specifications/model-format-spec.md) - Section 6.4
