# Pruning Schedules

Pruning schedules control when and how sparsity is introduced during training. The right schedule can significantly
impact final model quality.

## Overview

Entrenar supports three pruning schedules:

| Schedule | Sparsity Curve | Best For |
|----------|---------------|----------|
| OneShot | Step function | Post-training pruning |
| Gradual | Linear | Fine-tuning during training |
| Cubic | S-curve | High sparsity targets |

## OneShot Schedule

Applies target sparsity in a single step. Simple and effective for post-training compression.

```rust
use entrenar::prune::PruningSchedule;

let schedule = PruningSchedule::OneShot { step: 1000 };

// Sparsity transitions instantly at step 1000
assert_eq!(schedule.sparsity_at_step(999), 0.0);
assert_eq!(schedule.sparsity_at_step(1000), 1.0);  // Returns multiplier
assert_eq!(schedule.sparsity_at_step(2000), 1.0);
```

### Use Cases

- LLM pruning (SparseGPT, Wanda)
- Post-training compression
- When fine-tuning budget is limited

### Pros and Cons

**Pros:**
- Simplest to implement
- Works well with calibration-based methods
- No hyperparameter tuning for schedule

**Cons:**
- Can cause accuracy drop without fine-tuning
- Not ideal for very high sparsity (>70%)

## Gradual Schedule

Linearly interpolates from initial to final sparsity over a range of steps.

```rust
let schedule = PruningSchedule::Gradual {
    start_step: 1000,
    end_step: 5000,
    initial_sparsity: 0.0,
    final_sparsity: 0.5,
    frequency: 500,
};

// Before start: no pruning
assert_eq!(schedule.sparsity_at_step(500), 0.0);

// During pruning: linear interpolation
assert_eq!(schedule.sparsity_at_step(1000), 0.0);
assert_eq!(schedule.sparsity_at_step(3000), 0.25);
assert_eq!(schedule.sparsity_at_step(5000), 0.5);

// After end: stay at final sparsity
assert_eq!(schedule.sparsity_at_step(6000), 0.5);
```

### Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `start_step` | When to begin pruning | 10% of total steps |
| `end_step` | When to reach final sparsity | 50-80% of total steps |
| `initial_sparsity` | Starting sparsity | 0.0 |
| `final_sparsity` | Target sparsity | 0.3-0.9 |
| `frequency` | Steps between pruning updates | 100-1000 |

### Frequency Effect

The `frequency` parameter controls how often the mask is updated:

```rust
// Update mask every 500 steps
let frequent = PruningSchedule::Gradual {
    start_step: 0,
    end_step: 10000,
    initial_sparsity: 0.0,
    final_sparsity: 0.5,
    frequency: 500,  // 20 updates total
};

// Update mask every 2000 steps
let sparse_updates = PruningSchedule::Gradual {
    start_step: 0,
    end_step: 10000,
    initial_sparsity: 0.0,
    final_sparsity: 0.5,
    frequency: 2000,  // 5 updates total
};
```

More frequent updates allow finer control but add overhead.

## Cubic Schedule (Zhu & Gupta 2017)

Uses a cubic polynomial that prunes aggressively early and slows toward the end.

```rust
let schedule = PruningSchedule::Cubic {
    start_step: 0,
    end_step: 10000,
    final_sparsity: 0.7,
};

// Formula: s_t = s_f * (1 - (1 - t/T)^3)
```

### Mathematical Formula

The cubic schedule follows:

```
s_t = s_f * (1 - (1 - t/T)^3)
```

Where:
- `s_t` = sparsity at step t
- `s_f` = final target sparsity (e.g., 0.7)
- `t` = current step within pruning window
- `T` = total pruning steps (`end_step - start_step`)

### Sparsity Progression

For `final_sparsity = 0.7` over 10000 steps:

| Step | Progress | Sparsity |
|------|----------|----------|
| 0 | 0% | 0.0% |
| 2500 | 25% | 48.8% |
| 5000 | 50% | 61.3% |
| 7500 | 75% | 68.9% |
| 10000 | 100% | 70.0% |

```rust
let schedule = PruningSchedule::Cubic {
    start_step: 0,
    end_step: 10000,
    final_sparsity: 0.7,
};

// Verify progression
assert!((schedule.sparsity_at_step(0) - 0.0).abs() < 0.01);
assert!((schedule.sparsity_at_step(2500) - 0.488).abs() < 0.01);
assert!((schedule.sparsity_at_step(5000) - 0.613).abs() < 0.01);
assert!((schedule.sparsity_at_step(7500) - 0.689).abs() < 0.01);
assert!((schedule.sparsity_at_step(10000) - 0.7).abs() < 0.01);
```

### Why Cubic?

The cubic curve has desirable properties:

1. **Aggressive early pruning** - Model is most plastic early in training
2. **Gradual convergence** - Allows fine-tuning of remaining weights
3. **Smooth transitions** - No sudden sparsity jumps

### Reference

> Zhu, M., & Gupta, S. (2017). "To Prune, or Not to Prune: Exploring the Efficacy of Pruning for Model Compression."
arXiv:1710.01878

## Choosing a Schedule

### Decision Tree

```
Is this post-training compression?
├── Yes → OneShot
└── No → Target sparsity > 50%?
    ├── Yes → Cubic
    └── No → Gradual
```

### Recommendations by Scenario

| Scenario | Recommended Schedule |
|----------|---------------------|
| LLM compression (Wanda/SparseGPT) | OneShot |
| Training from scratch with pruning | Gradual |
| High sparsity (>70%) | Cubic |
| Quick experiments | OneShot |
| Production deployment | Gradual or Cubic |

## Configuration Validation

All schedules validate their parameters:

```rust
use entrenar::prune::PruningConfig;

// Invalid: end_step before start_step
let bad_config = PruningConfig::new()
    .with_schedule(PruningSchedule::Gradual {
        start_step: 5000,
        end_step: 1000,  // Invalid!
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 100,
    });

match bad_config.validate() {
    Ok(()) => unreachable!(),
    Err(e) => println!("Validation error: {}", e),
}
```

## Combining with Fine-Tuning

For best results, allocate training steps for recovery:

```rust
let total_steps = 100000;

// Prune during first 60% of training
let schedule = PruningSchedule::Cubic {
    start_step: 0,
    end_step: 60000,  // 60% of total
    final_sparsity: 0.7,
};

// Remaining 40% for fine-tuning at final sparsity
```
