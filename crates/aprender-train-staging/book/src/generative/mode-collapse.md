# Mode Collapse Detection

Mode collapse occurs when the generator produces limited variety.

## What is Mode Collapse?

- Generator ignores latent input
- All outputs look similar
- Discriminator can't distinguish (always ~0.5)
- Training becomes unstable

## Detection Methods

### Diversity Score

```rust
let collapse_score = gan.detect_mode_collapse(num_samples);

// collapse_score ∈ [0, 1]
// 0 = perfect diversity
// 1 = complete collapse (all identical)
```

### Implementation

```rust
pub fn detect_mode_collapse(&mut self, num_samples: usize) -> f32 {
    let latents = self.sample_latent(num_samples);
    let samples: Vec<Vec<u32>> = latents.iter()
        .map(|z| self.generator.generate(z))
        .collect();

    // Count unique sequences
    let unique: HashSet<_> = samples.iter().collect();
    let diversity = unique.len() as f32 / num_samples as f32;

    // Mode collapse score = 1 - diversity
    1.0 - diversity
}
```

## Prevention Strategies

### 1. Minibatch Discrimination

```rust
fn minibatch_features(batch: &[Vec<u32>]) -> Vec<f32> {
    let features: Vec<Vec<f32>> = batch.iter()
        .map(|x| embed(x))
        .collect();

    // Compute pairwise distances
    let mut diversity_features = Vec::new();
    for i in 0..features.len() {
        let distances: Vec<f32> = features.iter()
            .map(|f| l2_distance(&features[i], f))
            .collect();
        diversity_features.extend(distances);
    }
    diversity_features
}
```

### 2. Historical Averaging

```rust
fn historical_averaging_loss(
    current_params: &[f32],
    history: &VecDeque<Vec<f32>>,
) -> f32 {
    let avg: Vec<f32> = average_params(history);
    mse(current_params, &avg)
}
```

### 3. Unrolled GAN

```rust
fn unrolled_generator_loss(
    gan: &mut CodeGan,
    unroll_steps: usize,
) -> f32 {
    // Save discriminator state
    let d_state = gan.discriminator.clone();

    // Unroll discriminator updates
    for _ in 0..unroll_steps {
        gan.train_discriminator_step(&real_batch);
    }

    // Compute generator loss against unrolled discriminator
    let g_loss = gan.generator_loss();

    // Restore discriminator
    gan.discriminator = d_state;

    g_loss
}
```

## Monitoring Dashboard

```rust
// During training
for step in 0..total_steps {
    let g_loss = gan.train_generator_step();
    let d_loss = gan.train_discriminator_step(&batch);

    if step % 100 == 0 {
        let collapse = gan.detect_mode_collapse(100);
        println!(
            "Step {}: G={:.4} D={:.4} Collapse={:.2}%",
            step, g_loss, d_loss, collapse * 100.0
        );

        if collapse > 0.8 {
            eprintln!("WARNING: Mode collapse detected!");
            // Consider: reduce LR, add noise, restart
        }
    }
}
```
