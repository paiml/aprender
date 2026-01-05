# Neural Network Pruning Theory

Neural network pruning is a model compression technique that removes redundant parameters to reduce model size and computational cost while maintaining accuracy.

## Overview

Modern neural networks are often over-parameterized, containing many weights that contribute little to the final prediction. Pruning identifies and removes these less important weights, producing a sparse model.

### Key Benefits

- **Reduced memory footprint** - Fewer parameters to store
- **Faster inference** - Less computation required
- **Energy efficiency** - Lower power consumption
- **Edge deployment** - Enables deployment on resource-constrained devices

## Pruning Criteria

### Magnitude-Based Pruning

The simplest and most effective pruning method uses weight magnitude as an importance metric.

#### L1 Magnitude (Absolute Value)

```
importance(w) = |w|
```

Weights with small absolute values contribute less to the output and can be safely removed.

#### L2 Magnitude (Squared Value)

```
importance(w) = w^2
```

Squared magnitude penalizes small weights more aggressively, creating a clearer separation between important and unimportant weights.

### Activation-Weighted Pruning (Wanda)

Wanda (Weights AND Activations) considers both weight magnitude and activation statistics:

```
importance(w_ij) = |w_ij| * sqrt(sum(x_j^2) / n)
```

This captures how much each weight contributes to the output given typical inputs, requiring calibration data to estimate activation statistics.

**Reference:** Sun et al. (2023) - "A Simple and Effective Pruning Approach for Large Language Models"

## Sparsity Patterns

### Unstructured Sparsity

Individual weights are pruned independently, achieving maximum flexibility and compression but limited hardware acceleration.

```
Original:  [0.5, 0.1, -0.8, 0.02]
Mask:      [1,   0,    1,   0   ]
Pruned:    [0.5, 0,   -0.8, 0   ]
```

### N:M Structured Sparsity

Exactly N non-zero values per M consecutive elements. Hardware-accelerated on NVIDIA Ampere+ GPUs.

**Common patterns:**
- **2:4** - 2 non-zeros per 4 elements (50% sparsity)
- **4:8** - 4 non-zeros per 8 elements (50% sparsity)

```
2:4 Pattern:
Original:  [0.5, 0.1, -0.8, 0.02]
Mask:      [1,   0,    1,   0   ]  // 2 ones per 4 elements
Pruned:    [0.5, 0,   -0.8, 0   ]
```

### Block Sparsity

Entire blocks of weights are pruned together, enabling efficient memory access patterns.

## Pruning Schedules

### One-Shot Pruning

Prune to target sparsity in a single step, typically after pre-training.

```rust
let schedule = PruningSchedule::OneShot { step: 1000 };
```

### Gradual Pruning

Incrementally increase sparsity over training, allowing the model to adapt.

```rust
let schedule = PruningSchedule::Gradual {
    start_step: 1000,
    end_step: 5000,
    initial_sparsity: 0.0,
    final_sparsity: 0.5,
    frequency: 500,  // Prune every 500 steps
};
```

### Cubic Pruning Schedule

The Zhu & Gupta (2017) cubic schedule provides smooth sparsity increase:

```
s_t = s_f * (1 - (1 - t/T)^3)
```

Where:
- `s_t` = sparsity at step t
- `s_f` = final target sparsity
- `T` = total pruning steps

This schedule prunes aggressively early (when model is more plastic) and gradually slows.

**Reference:** Zhu & Gupta (2017) - "To Prune or Not To Prune"

## Implementation in Aprender

### Computing Importance Scores

```rust
use aprender::pruning::{MagnitudeImportance, Importance};
use aprender::nn::Linear;

let layer = Linear::new(768, 768);

// L1 magnitude
let l1 = MagnitudeImportance::l1();
let scores = l1.compute(&layer, None)?;

// L2 magnitude
let l2 = MagnitudeImportance::l2();
let scores = l2.compute(&layer, None)?;
```

### Generating Sparsity Masks

```rust
use aprender::pruning::{generate_unstructured_mask, generate_nm_mask};

// 50% unstructured sparsity
let mask = generate_unstructured_mask(&scores.values, 0.5)?;

// 2:4 N:M sparsity
let nm_mask = generate_nm_mask(&scores.values, 2, 4)?;
```

### Applying Masks

```rust
let mut weights = layer.weight().clone();
mask.apply(&mut weights)?;

// Verify sparsity
let actual_sparsity = mask.sparsity();
assert!((actual_sparsity - 0.5).abs() < 0.01);
```

## Best Practices

1. **Start with magnitude pruning** - Simple, effective, no calibration needed
2. **Use gradual schedules for high sparsity** - Allows model adaptation
3. **Fine-tune after pruning** - Recover accuracy lost during pruning
4. **Validate with representative data** - Ensure pruned model generalizes
5. **Consider hardware targets** - Use N:M patterns for GPU acceleration

## Mathematical Properties

### Importance Scores

- All importance scores are non-negative: `importance(w) >= 0`
- Zero weights have zero importance: `importance(0) = 0`
- Masks are idempotent: `apply(apply(w, m), m) = apply(w, m)`

### Sparsity Definition

```
sparsity = num_zeros / total_elements
```

For a 50% sparse model, half the weights are exactly zero.

## References

1. Han et al. (2015) - "Learning both Weights and Connections for Efficient Neural Networks"
2. Zhu & Gupta (2017) - "To Prune, or Not to Prune: Exploring the Efficacy of Pruning for Model Compression"
3. Sun et al. (2023) - "A Simple and Effective Pruning Approach for Large Language Models"
4. Frantar & Alistarh (2023) - "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot"
