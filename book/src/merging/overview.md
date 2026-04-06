# Model Merging Overview

**Model merging** combines multiple fine-tuned models into a single
unified model that retains capabilities from all source models.

## The Problem

When you fine-tune multiple models for different tasks, you end up with N separate models:

```
Base Model (7B params)
  ├→ Model A: Fine-tuned on coding tasks
  ├→ Model B: Fine-tuned on math problems
  └→ Model C: Fine-tuned on creative writing
```

**Challenge**: How do you create a single model that performs well on all three tasks without:
- Retraining from scratch (expensive)
- Serving N models in parallel (memory/latency overhead)
- Losing task-specific knowledge (catastrophic forgetting)

## The Solution: Weight Merging

Entrenar implements three state-of-the-art merging algorithms from Arcee AI:

### TIES (Task Inference via Elimination and Sign voting)

**Key Idea**: Resolve parameter conflicts by keeping top-k% changes and using sign voting

```rust
use entrenar::merge::TIESMerger;

// density=0.5 keeps top 50% of changes
// lambda=1.0 gives equal weight to all models
let merger = TIESMerger::new(0.5, 1.0);
let merged = merger.merge(&models)?;
```

**From `src/merge/ties.rs`**

### DARE (Drop And REscale)

**Key Idea**: Randomly drop parameter updates with Bernoulli masking, then rescale

```rust
use entrenar::merge::DAREMerger;

// drop_rate=0.9 means keep only 10% of updates
let merger = DAREMerger::new(0.9);
let merged = merger.merge(&models)?;
```

**From `src/merge/dare.rs`**

### SLERP (Spherical Linear intERPolation)

**Key Idea**: Interpolate on the weight manifold (preserves magnitude)

```rust
use entrenar::merge::SLERPMerger;

// t=0.5 gives 50-50 interpolation between two models
let merger = SLERPMerger::new(0.5);
let merged = merger.merge(&[model_a, model_b])?;
```

**From `src/merge/slerp.rs`**

## When to Use Each Algorithm

| Algorithm | Use Case | Best For |
|-----------|----------|----------|
| **TIES** | Multi-task merging (3+ models) | Resolving parameter conflicts across many tasks |
| **DARE** | Sparse fine-tuning merges | LoRA adapters, small delta updates |
| **SLERP** | Two-model interpolation | Smooth transitions, model averaging |

## Implementation Details

All merging algorithms in Entrenar are:
- ✅ **Tested**: Property-based tests for permutation invariance
- ✅ **Validated**: Works with full models and LoRA adapters
- ✅ **Type-safe**: Compile-time guarantees via Rust's type system

## Next Steps

- [TIES Algorithm](./ties.md) - Detailed TIES implementation
- [DARE Algorithm](./dare.md) - Drop and rescale mechanics
- [SLERP Algorithm](./slerp.md) - Spherical interpolation
- [Examples](../examples/merge-models.md) - Real-world merging examples

## References

Based on:
- **TIES-Merging** paper (Yadav et al., 2023)
- **DARE** paper (Yu et al., 2024)
- **SLERP** (classic computer graphics technique)
- **Arcee AI** merging research
