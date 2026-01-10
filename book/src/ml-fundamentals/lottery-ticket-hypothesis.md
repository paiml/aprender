# Lottery Ticket Hypothesis

The Lottery Ticket Hypothesis (LTH) reveals that dense neural networks contain sparse subnetworks ("winning tickets") that can train to full accuracy when reset to their initial weights.

## Overview

Frankle & Carbin (2018) discovered that randomly-initialized neural networks contain sparse subnetworks that, when trained in isolation from initialization, can match the test accuracy of the full network.

### Key Insight

> "A randomly-initialized, dense neural network contains a subnetwork that is initialized such that—when trained in isolation—it can match the test accuracy of the original network after training for at most the same number of iterations."

This challenges the common belief that over-parameterization is necessary for training success.

## The Algorithm: Iterative Magnitude Pruning (IMP)

The original LTH algorithm uses iterative magnitude pruning with weight rewinding:

### Algorithm Steps

1. **Initialize** network with random weights W₀
2. **Train** to convergence → W_T
3. **Prune** the p% smallest-magnitude weights globally
4. **Rewind** remaining weights to their values at W₀
5. **Repeat** steps 2-4 until target sparsity reached

### Mathematical Formulation

For target sparsity s after n rounds, the per-round pruning rate is:

```
p = 1 - (1 - s)^(1/n)
```

After k rounds, the remaining weights fraction is:

```
remaining(k) = (1 - p)^k
```

For example, with 90% target sparsity over 10 rounds:
- p ≈ 0.206 (20.6% per round)
- remaining(10) = 0.1 (10% of weights remain)

## Rewind Strategies

Different rewinding strategies affect which weights are restored:

### Early Rewinding (W₀)

Rewind to the initial random weights. This is the original LTH formulation.

```rust
RewindStrategy::Init
```

### Late Rewinding (W_k)

Rewind to weights from early in training (iteration k). Often k = 0.1T (10% of training).

```rust
RewindStrategy::Early { iteration: 100 }
```

**Reference:** Frankle et al. (2019) - "Stabilizing the Lottery Ticket Hypothesis"

### Learning Rate Rewinding

Rewind the learning rate schedule but keep late weights.

```rust
RewindStrategy::Late { fraction: 0.1 }  // Rewind to 10% through training
```

### No Rewinding

Keep final trained weights (standard pruning without LTH).

```rust
RewindStrategy::None
```

## Implementation in Aprender

### Basic Usage

```rust,ignore
use aprender::pruning::{LotteryTicketPruner, LotteryTicketConfig, RewindStrategy};
use aprender::nn::Linear;

// Create a model
let model = Linear::new(256, 128);

// Configure: 90% sparsity over 10 rounds, rewind to init
let config = LotteryTicketConfig::new(0.9, 10)
    .with_rewind_strategy(RewindStrategy::Init);

let pruner = LotteryTicketPruner::with_config(config);

// Find winning ticket
let ticket = pruner.find_ticket(&model).expect("ticket");

println!("Winning ticket:");
println!("  Sparsity: {:.1}%", ticket.sparsity * 100.0);
println!("  Remaining params: {}", ticket.remaining_parameters);
println!("  Compression: {:.1}x", ticket.compression_ratio());
```

### Builder Pattern

```rust,ignore
use aprender::pruning::{LotteryTicketPruner, RewindStrategy};

let pruner = LotteryTicketPruner::builder()
    .target_sparsity(0.95)
    .pruning_rounds(15)
    .rewind_strategy(RewindStrategy::Early { iteration: 500 })
    .global_pruning(true)
    .build();
```

### Applying the Ticket

```rust,ignore
// Apply winning ticket mask and rewind weights
let result = pruner.apply_ticket(&mut model, &ticket).expect("apply");

// The model now has:
// 1. Sparse weights (90% zeros)
// 2. Non-zero weights reset to initial values
```

### Tracking Sparsity History

```rust,ignore
let ticket = pruner.find_ticket(&model).expect("ticket");

println!("Sparsity progression:");
for (round, sparsity) in ticket.sparsity_history.iter().enumerate() {
    println!("  Round {}: {:.1}%", round + 1, sparsity * 100.0);
}
// Output:
// Round 1: 20.6%
// Round 2: 37.0%
// Round 3: 50.0%
// ...
// Round 10: 90.0%
```

## Why Winning Tickets Work

### The Lottery Analogy

Training a neural network is like buying lottery tickets:
- Each random initialization creates a "ticket" (subnetwork structure)
- The winning ticket has fortuitously good initial weights
- Over-parameterization increases the chance of containing a winning ticket

### Structural vs. Weight Importance

Winning tickets suggest that:
1. **Structure matters** - The connectivity pattern is crucial
2. **Initial weights matter** - Specific initializations enable training
3. **Pruning identifies structure** - Magnitude pruning discovers winning tickets

## Comparison with Standard Pruning

| Aspect | Standard Pruning | Lottery Ticket |
|--------|-----------------|----------------|
| When | After training | During/after training |
| Weights | Keep final trained | Rewind to initial |
| Retraining | From pruned state | From initialization |
| Goal | Compress trained model | Find trainable sparse subnet |

## Practical Considerations

### Computational Cost

LTH requires multiple train-prune-rewind cycles:

```
Total cost = rounds × training_cost
```

For 10 rounds, expect ~10x training time.

### Hyperparameter Sensitivity

Winning tickets can be sensitive to:
- Learning rate schedule
- Batch size
- Random seed (initialization)

### Scaling Challenges

Original LTH is harder to replicate at scale:
- Works well for small networks (MNIST, CIFAR)
- Requires "late rewinding" for larger models (ImageNet, BERT)

## Extensions and Variants

### One-Shot LTH

Find winning tickets without iterative pruning using sensitivity analysis:

```rust
// Approximate winning ticket in one shot
let pruner = LotteryTicketPruner::builder()
    .target_sparsity(0.9)
    .pruning_rounds(1)  // Single round
    .build();
```

### Supermask Training

Train only the mask, keeping weights frozen:

```
mask_i = σ(s_i)  // Learned scores
output = mask ⊙ W ⊙ x
```

**Reference:** Zhou et al. (2019) - "Deconstructing Lottery Tickets"

### Neural Network Pruning at Initialization

Skip training entirely—find winning tickets from random init:

**Reference:** Lee et al. (2019) - "SNIP: Single-Shot Network Pruning"

## Mathematical Properties

### Mask Properties

```
mask ∈ {0, 1}^n          // Binary mask
sparsity = sum(mask=0) / n
density = 1 - sparsity
```

### Idempotence

Applying a mask multiple times has the same effect:

```
apply(apply(W, m), m) = apply(W, m)
```

### Compression Ratio

```
compression = total_params / remaining_params
            = 1 / density
```

For 90% sparsity: compression = 10x

## Quality Metrics

### Winning Ticket Quality

A good winning ticket:
1. Achieves target accuracy when retrained
2. Matches or exceeds dense network accuracy
3. Trains in similar or fewer iterations

### Sparsity vs. Accuracy Trade-off

Typical behavior:
- Up to 80% sparsity: minimal accuracy loss
- 80-95% sparsity: gradual degradation
- >95% sparsity: significant accuracy drop

## References

1. Frankle, J., & Carbin, M. (2018). "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks." ICLR 2019.
2. Frankle, J., et al. (2019). "Stabilizing the Lottery Ticket Hypothesis." arXiv:1903.01611.
3. Zhou, H., et al. (2019). "Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask." NeurIPS 2019.
4. Lee, N., et al. (2019). "SNIP: Single-shot Network Pruning based on Connection Sensitivity." ICLR 2019.
5. Malach, E., et al. (2020). "Proving the Lottery Ticket Hypothesis: Pruning is All You Need." ICML 2020.
