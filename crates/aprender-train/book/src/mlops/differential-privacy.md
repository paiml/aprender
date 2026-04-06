# Differential Privacy

Privacy-preserving training with Differentially Private SGD (DP-SGD) and Rényi Differential Privacy accounting.

## Toyota Principle: Jidoka

Protect the quality of privacy guarantees. Just as Jidoka stops production when defects are detected, DP-SGD ensures
privacy bounds are never exceeded.

## Quick Start

```rust
use entrenar::optim::dp::{DPOptimizer, PrivacyEngine, PrivacyAccountant};
use entrenar::optim::AdamW;

// Configure privacy engine
let engine = PrivacyEngine::new()
    .with_noise_multiplier(1.0)
    .with_max_grad_norm(1.0)
    .with_target_epsilon(1.0)
    .with_target_delta(1e-5);

// Wrap base optimizer
let base_optimizer = AdamW::new(0.001);
let dp_optimizer = DPOptimizer::new(base_optimizer, engine);

// Training loop
for batch in dataloader {
    let loss = model.forward(&batch);
    let grads = loss.backward();

    // DP-SGD: clip gradients and add noise
    dp_optimizer.step(&mut model, grads)?;

    // Check privacy budget
    let (epsilon, delta) = dp_optimizer.get_privacy_spent()?;
    println!("Privacy: ε={:.2}, δ={:.2e}", epsilon, delta);

    if epsilon > target_epsilon {
        println!("Privacy budget exhausted!");
        break;
    }
}
```

## Privacy Engine Configuration

```rust
use entrenar::optim::dp::PrivacyEngine;

let engine = PrivacyEngine::new()
    // Noise multiplier (higher = more privacy, less utility)
    .with_noise_multiplier(1.0)

    // Per-sample gradient clipping threshold
    .with_max_grad_norm(1.0)

    // Target privacy budget
    .with_target_epsilon(1.0)
    .with_target_delta(1e-5)

    // Sample rate (batch_size / dataset_size)
    .with_sample_rate(0.01);
```

## Privacy Accounting

Track privacy expenditure with Rényi Differential Privacy (RDP):

```rust
use entrenar::optim::dp::PrivacyAccountant;

let accountant = PrivacyAccountant::new()
    .with_noise_multiplier(1.0)
    .with_sample_rate(0.01);

// After each step
accountant.step();

// Get privacy spent
let (epsilon, delta) = accountant.get_privacy_spent(1e-5)?;
println!("After {} steps: ε={:.2}", accountant.steps(), epsilon);

// Estimate steps until budget exhausted
let remaining = accountant.steps_remaining(target_epsilon, 1e-5)?;
println!("Steps remaining: {}", remaining);
```

## Noise Multiplier Selection

```rust
use entrenar::optim::dp::estimate_noise_multiplier;

// Find noise multiplier for target privacy
let noise_multiplier = estimate_noise_multiplier(
    target_epsilon,  // e.g., 1.0
    target_delta,    // e.g., 1e-5
    sample_rate,     // batch_size / dataset_size
    num_steps,       // total training steps
)?;

println!("Use noise_multiplier={:.2}", noise_multiplier);
```

## Per-Sample Gradient Clipping

```rust
use entrenar::optim::dp::clip_gradients;

// Clip per-sample gradients to max_norm
let clipped_grads = clip_gradients(&per_sample_grads, max_norm);

// Add calibrated Gaussian noise
let noisy_grads = add_noise(&clipped_grads, noise_multiplier * max_norm);
```

## Privacy-Utility Trade-off

| Noise Multiplier | Privacy (ε) | Utility Impact |
|-----------------|-------------|----------------|
| 0.5 | High ε (~10) | Minimal |
| 1.0 | Medium ε (~1) | Moderate |
| 2.0 | Low ε (~0.5) | Significant |
| 4.0 | Very Low ε (~0.1) | Severe |

## Integration with Training Loop

```rust
use entrenar::train::{Trainer, TrainerConfig};
use entrenar::optim::dp::{DPOptimizer, PrivacyEngine};

let config = TrainerConfig::default()
    .with_epochs(10)
    .with_batch_size(64);

let engine = PrivacyEngine::new()
    .with_noise_multiplier(1.0)
    .with_max_grad_norm(1.0)
    .with_target_epsilon(1.0)
    .with_target_delta(1e-5);

let mut trainer = Trainer::new(config)
    .with_dp_engine(engine);

// Train with privacy guarantees
trainer.fit(&model, &dataset)?;

// Get final privacy guarantee
let (epsilon, delta) = trainer.privacy_spent()?;
println!("Final privacy: ({:.2}, {:.2e})-DP", epsilon, delta);
```

## Cargo Run Example

```bash
# Train with differential privacy
cargo run --example dp_training

# With custom privacy parameters
cargo run --example dp_training -- \
    --epsilon 1.0 \
    --delta 1e-5 \
    --max-grad-norm 1.0
```

## Privacy Composition

For multiple queries/models:

```rust
use entrenar::optim::dp::compose_privacy;

let guarantees = vec![
    (0.5, 1e-5),  // Model 1
    (0.3, 1e-5),  // Model 2
    (0.2, 1e-5),  // Model 3
];

let (total_epsilon, total_delta) = compose_privacy(&guarantees)?;
println!("Total: ({:.2}, {:.2e})-DP", total_epsilon, total_delta);
```

## Best Practices

1. **Choose δ < 1/n** - Where n is dataset size
2. **Start with higher noise** - Tune down for utility
3. **Use larger batch sizes** - Better privacy-utility trade-off
4. **Monitor privacy budget** - Stop before exhaustion
5. **Validate with auditing** - Empirical privacy testing

## Privacy Guarantees

DP-SGD provides (ε, δ)-differential privacy, meaning:

> For any two neighboring datasets D and D' (differing by one record), and any output S:
>
> P[M(D) ∈ S] ≤ e^ε · P[M(D') ∈ S] + δ

Where M is the training mechanism.

## See Also

- [MLOps Overview](./overview.md)
- [Experiment Tracking](./experiment-tracking.md)
- [Optimizer Selection](../best-practices/optimizer-selection.md)
