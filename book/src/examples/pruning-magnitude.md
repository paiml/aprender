# Case Study: Magnitude Pruning

This example demonstrates neural network pruning using magnitude-based importance scoring with Aprender's pruning module.

## Overview

Magnitude pruning is the simplest and most widely-used pruning technique. It removes weights with the smallest absolute values, based on the intuition that small weights contribute less to the network's output.

## Running the Example

```bash
cargo run --example pruning_magnitude
```

## Code Walkthrough

### 1. Create a Linear Layer

```rust
use aprender::nn::Linear;

let layer = Linear::new(16, 8);
let weights = layer.weight();
let total_params = weights.data().len();  // 128 parameters
```

### 2. Compute L1 Importance

L1 importance uses absolute value: `importance(w) = |w|`

```rust
use aprender::pruning::{MagnitudeImportance, Importance};

let l1_importance = MagnitudeImportance::l1();
let l1_scores = l1_importance.compute(&layer, None)?;

println!("Method: {}", l1_scores.method);  // "magnitude_l1"
println!("Min: {:.6}", l1_scores.stats.min);
println!("Max: {:.6}", l1_scores.stats.max);
println!("Mean: {:.6}", l1_scores.stats.mean);
```

### 3. Compute L2 Importance

L2 importance uses squared value: `importance(w) = w^2`

```rust
let l2_importance = MagnitudeImportance::l2();
let l2_scores = l2_importance.compute(&layer, None)?;
```

L2 penalizes small weights more aggressively than L1, creating clearer separation.

### 4. Generate Unstructured Mask

Create a mask that zeros out 50% of weights:

```rust
use aprender::pruning::generate_unstructured_mask;

let mask = generate_unstructured_mask(&l1_scores.values, 0.5)?;

println!("Achieved sparsity: {:.1}%", mask.sparsity() * 100.0);
println!("Non-zero weights: {}", mask.nnz());
println!("Pruned weights: {}", mask.num_zeros());
```

### 5. Generate N:M Structured Mask

2:4 sparsity keeps exactly 2 non-zeros per 4 consecutive elements:

```rust
use aprender::pruning::generate_nm_mask;

// Layer must have elements divisible by 4
let nm_layer = Linear::new(8, 8);  // 64 elements
let nm_scores = MagnitudeImportance::l1().compute(&nm_layer, None)?;

let nm_mask = generate_nm_mask(&nm_scores.values, 2, 4)?;
println!("Achieved sparsity: {:.1}%", nm_mask.sparsity() * 100.0);  // 50%
```

### 6. Apply Mask to Weights

```rust
let mut pruned_weights = weights.clone();
mask.apply(&mut pruned_weights)?;

// Verify zeros
let zeros_after: usize = pruned_weights
    .data()
    .iter()
    .filter(|&&v| v.abs() < 1e-10)
    .count();
```

## Expected Output

```
╔══════════════════════════════════════════════════════════════╗
║         Magnitude Pruning with Aprender                      ║
║         Prune neural networks by weight magnitude            ║
╚══════════════════════════════════════════════════════════════╝

📊 Creating Linear Layer (16 → 8)
   Weight shape: [8, 16]
   Total parameters: 128

🔬 Computing L1 Magnitude Importance
   Method: magnitude_l1
   Stats:
     - Min:  0.000123
     - Max:  0.987654
     - Mean: 0.456789
     - Std:  0.234567

✂️  Generating Unstructured Mask (50% sparsity)
   Achieved sparsity: 50.0%
   Non-zero weights: 64
   Pruned weights: 64

✂️  Generating 2:4 N:M Mask (50% structured sparsity)
   Pattern: 2:4 (2 non-zeros per 4 elements)
   Achieved sparsity: 50.0%
   Valid 2:4 groups: 16/16

📉 Applying Mask to Weights
   Zeros after pruning: 64 (50.0%)

╔══════════════════════════════════════════════════════════════╗
║                    Pruning Summary                           ║
╠══════════════════════════════════════════════════════════════╣
║  Original parameters:      128                               ║
║  Pruned parameters:         64 (50% reduction)               ║
║  Remaining parameters:      64                               ║
╚══════════════════════════════════════════════════════════════╝
```

## Key Concepts

### ImportanceScores

The `compute()` method returns `ImportanceScores` containing:
- `values` - Tensor of importance scores (same shape as weights)
- `method` - String identifier (e.g., "magnitude_l1")
- `stats` - Statistics (min, max, mean, std)

### SparsityMask

The mask is a binary tensor where:
- `1.0` = keep the weight
- `0.0` = prune (set to zero)

Key methods:
- `sparsity()` - Fraction of zeros (0.0 to 1.0)
- `nnz()` - Number of non-zeros
- `num_zeros()` - Number of zeros
- `apply(&mut tensor)` - Zero out masked weights

### N:M Sparsity Verification

The example verifies that every group of 4 elements has exactly 2 non-zeros:

```rust
for chunk in mask_data.chunks(4) {
    let nonzeros: usize = chunk.iter()
        .map(|&v| if v > 0.5 { 1 } else { 0 })
        .sum();
    assert_eq!(nonzeros, 2);  // Valid 2:4 pattern
}
```

## When to Use

- **L1 Magnitude** - General purpose, works well in most cases
- **L2 Magnitude** - When you want stronger separation between important/unimportant weights
- **Unstructured** - Maximum flexibility, best compression
- **2:4 N:M** - When targeting NVIDIA Ampere+ GPU acceleration

## Related Examples

- [Neural Network Pruning Theory](../ml-fundamentals/neural-network-pruning.md)
- [XOR Neural Network](./xor-neural-network.md)
- [Neural Network Training Pipeline](./neural-network-training.md)
