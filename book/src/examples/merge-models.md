# Model Merging Examples

Examples demonstrating TIES, DARE, and SLERP model merging algorithms.

## Basic TIES Merge

```rust
use entrenar::merge::TiesMerge;

// Three fine-tuned models to merge
let models = vec![model_a, model_b, model_c];
let weights = vec![0.4, 0.3, 0.3];

// TIES: Trim + Sign Election + Merge
let merger = TiesMerge::new(0.2); // 20% density
let merged = merger.merge(&models, &weights);
```

## DARE with Dropout

```rust
use entrenar::merge::DareMerge;

// Drop 90% of delta weights, rescale remainder
let merger = DareMerge::new(0.9);
let merged = merger.merge(&base_model, &finetuned_model);
```

## SLERP Interpolation

```rust
use entrenar::merge::SlerpMerge;

// Spherical linear interpolation between two models
let merger = SlerpMerge::new();
let merged = merger.merge(&model_a, &model_b, 0.5);
```

## Running the Example

```bash
cargo run --example merge_models
```

## See Also

- [TIES Algorithm](../merging/ties.md)
- [DARE Algorithm](../merging/dare.md)
- [SLERP Algorithm](../merging/slerp.md)
