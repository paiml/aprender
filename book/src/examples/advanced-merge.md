# Case Study: Advanced Merge Strategies

**Ticket**: GH-442 | **Contract**: `merge.md §strategies`

## Overview

Six additional merge strategies for Arcee MergeKit parity: Task Arithmetic,
NuSLERP, MultiSLERP, DELLA, Breadcrumbs, and SCE.

## Strategies

| Strategy | Base Required | Models | Description |
|----------|:---:|:---:|-------------|
| `task-arithmetic` | yes | 2+ | Linear combination of task vectors |
| `nuslerp` | no | 2 | Enhanced SLERP with nlerp fallback |
| `multi-slerp` | no | 2+ | Barycentric SLERP for >2 models |
| `della` | yes | 2+ | Adaptive magnitude pruning (like DARE but magnitude-aware) |
| `breadcrumbs` | yes | 2+ | Task arithmetic + outlier removal |
| `sce` | no | 2+ | Variance-adaptive per-tensor weighting |

## API

```rust
use aprender::format::converter::merge::{MergeOptions, MergeStrategy};

// Task Arithmetic: base + Σ(scale_i * (model_i - base))
let opts = MergeOptions {
    strategy: MergeStrategy::TaskArithmetic,
    scales: Some(vec![0.7, 0.3]),
    base_model: Some("base.safetensors".into()),
    ..Default::default()
};

// DELLA: adaptive drop rate proportional to magnitude
let opts = MergeOptions {
    strategy: MergeStrategy::Della,
    drop_rate: 0.7,
    base_model: Some("base.safetensors".into()),
    ..Default::default()
};

// SCE: variance-adaptive per-tensor weighting
let opts = MergeOptions {
    strategy: MergeStrategy::Sce,
    weights: Some(vec![0.5, 0.5]),
    ..Default::default()
};
```

## New `MergeOptions` Fields

| Field | Type | Default | Used By |
|-------|------|---------|---------|
| `scales` | `Option<Vec<f32>>` | `None` (all 1.0) | TaskArithmetic, Breadcrumbs |
| `outlier_k` | `f32` | `3.0` | Breadcrumbs |

## Falsification Tests

| Test | Property |
|------|----------|
| FALSIFY-MERGE-ADV-001 | All strategies produce finite results |
| FALSIFY-MERGE-ADV-002 | Task arithmetic with zero scale returns base |
| FALSIFY-MERGE-ADV-003 | SCE result bounded by input values |

## Run the Example

```bash
cargo run --example advanced_merge
```
