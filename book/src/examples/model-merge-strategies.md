# Case Study: Model Merge Strategies (GH-245)

Model merging combines multiple fine-tuned models into a single model without additional training. This is how many top-ranked open models on HuggingFace are created — merges, not trained from scratch.

## The 5 Strategies

| Strategy | Models | Requires Base | Key Parameter |
|----------|--------|---------------|---------------|
| Average | 2+ | No | — |
| Weighted | 2+ | No | `--weights` |
| SLERP | 2 only | No | `--weights` (interpolation t) |
| TIES | 2+ | Yes | `--density` |
| DARE | 2+ | Yes | `--drop-rate`, `--seed` |

### Average

Simple element-wise mean: `result = (model_a + model_b) / N`. Good baseline for ensemble-style merges.

### Weighted

Weighted element-wise sum: `result = w1*model_a + w2*model_b`. Weights must sum to 1.0.

### SLERP (Spherical Linear Interpolation)

Interpolates along the great circle between two weight vectors on a hypersphere. Preserves the magnitude of weights better than linear interpolation. Only works with exactly 2 models. Falls back to linear interpolation when vectors are nearly parallel.

### TIES (Trim, Elect Sign, Merge)

1. Compute task vectors: `delta_i = model_i - base`
2. Trim small values below `density * max(|delta|)` per tensor
3. Elect sign per element via majority vote across models
4. Average values agreeing with elected sign
5. Result: `base + merged_delta`

### DARE (Drop And Rescale)

1. Compute task vectors: `delta_i = model_i - base`
2. Randomly drop elements with probability `drop_rate`
3. Rescale remaining by `1 / (1 - drop_rate)`
4. Average rescaled deltas
5. Result: `base + avg(rescaled_deltas)`

## Running the Example

```bash
cargo run --example model_merge_strategies
```

## Rust API

```rust
use aprender::format::{apr_merge, MergeOptions, MergeStrategy};

// Average (default)
apr_merge(&[&model_a, &model_b], &output, MergeOptions::default())?;

// Weighted
apr_merge(&[&model_a, &model_b], &output, MergeOptions {
    strategy: MergeStrategy::Weighted,
    weights: Some(vec![0.7, 0.3]),
    ..Default::default()
})?;

// SLERP
apr_merge(&[&model_a, &model_b], &output, MergeOptions {
    strategy: MergeStrategy::Slerp,
    weights: Some(vec![0.3]),  // interpolation parameter t
    ..Default::default()
})?;

// TIES
apr_merge(&[&task_a, &task_b, &task_c], &output, MergeOptions {
    strategy: MergeStrategy::Ties,
    base_model: Some(base_path),
    density: 0.2,
    ..Default::default()
})?;

// DARE
apr_merge(&[&task_a, &task_b, &task_c], &output, MergeOptions {
    strategy: MergeStrategy::Dare,
    base_model: Some(base_path),
    drop_rate: 0.5,
    seed: 42,
    ..Default::default()
})?;
```

## CLI Usage

```bash
# Average
apr merge model_a.st model_b.st --strategy average -o merged.st

# Weighted
apr merge model_a.st model_b.st --strategy weighted --weights 0.7,0.3 -o merged.st

# SLERP
apr merge model_a.st model_b.st --strategy slerp --weights 0.3 -o merged.st

# TIES
apr merge task_a.st task_b.st task_c.st --strategy ties \
    --base-model base.st --density 0.2 -o merged.st

# DARE
apr merge task_a.st task_b.st task_c.st --strategy dare \
    --base-model base.st --drop-rate 0.5 --seed 42 -o merged.st
```

## Example Output

```
Input models:
  base (zeros):
    layer.bias [4] = [0.000, 0.000, 0.000, 0.000]
  model_a (diag 1,2,3,4):
    layer.bias [4] = [0.500, 0.500, 0.500, 0.500]
  model_b (diag 4,3,2,1):
    layer.bias [4] = [1.000, 1.000, 1.000, 1.000]

1. Average: [0.750, 0.750, 0.750, 0.750]
2. Weighted (0.7A + 0.3B): [0.650, 0.650, 0.650, 0.650]
3. SLERP (t=0.3): [0.650, 0.650, 0.650, 0.650]
4. TIES (density=0.2): [0.583, 0.583, 0.583, 0.583]
5. DARE (drop=0.5): [0.833, 1.000, 0.833, 0.333]  (stochastic)
```

Note how SLERP produces slightly different results from weighted interpolation (curved vs linear path), and DARE produces stochastic results (some elements dropped, others rescaled).

## See Also

- [Rosetta Stone — Universal Format Converter](./rosetta-stone.md)
- [APR CLI Tool](../tools/apr-cli.md)
