# GAN Training Loop

Training GANs requires careful balancing of generator and discriminator.

## Basic Training Loop

```rust
use entrenar::generative::{CodeGan, CodeGanConfig};

let config = CodeGanConfig::default();
let mut gan = CodeGan::new(config);

for epoch in 0..num_epochs {
    for real_batch in data_loader {
        // 1. Train discriminator on real data
        let d_loss_real = gan.train_discriminator_real(&real_batch);

        // 2. Train discriminator on fake data
        let fake_batch = gan.generate_batch(batch_size);
        let d_loss_fake = gan.train_discriminator_fake(&fake_batch);

        // 3. Train generator
        let g_loss = gan.train_generator_step();

        gan.record_step(g_loss, d_loss_real + d_loss_fake);
    }

    // Log progress
    println!(
        "Epoch {}: G_loss={:.4}, D_loss={:.4}",
        epoch,
        gan.stats.avg_generator_loss(),
        gan.stats.avg_discriminator_loss()
    );
}
```

## Training Tricks

### n-step Discriminator Updates

Train discriminator more frequently than generator:

```rust
let d_steps_per_g_step = 5;

for _ in 0..d_steps_per_g_step {
    let d_loss = gan.train_discriminator_step(&real_batch);
}
let g_loss = gan.train_generator_step();
```

### Gradient Penalty (WGAN-GP)

```rust
fn gradient_penalty(
    discriminator: &Discriminator,
    real: &[u32],
    fake: &[u32],
) -> f32 {
    let alpha = rand::random::<f32>();
    let interpolated = interpolate_tokens(real, fake, alpha);

    let d_interp = discriminator.forward(&interpolated);
    let gradients = compute_gradients(discriminator, &interpolated);

    let penalty = (gradients.norm() - 1.0).powi(2);
    10.0 * penalty  // Lambda = 10
}
```

### Spectral Normalization

```rust
fn spectral_norm(weight: &mut Vec<Vec<f32>>) {
    let (u, sigma, _v) = power_iteration(weight, 10);
    for row in weight.iter_mut() {
        for w in row.iter_mut() {
            *w /= sigma;
        }
    }
}
```

## Tracking Training Progress

```rust
// Statistics tracking
let stats = &gan.stats;

println!("Steps: {}", stats.steps);
println!("Avg G loss: {:.4}", stats.avg_generator_loss());
println!("Avg D loss: {:.4}", stats.avg_discriminator_loss());
println!("Mode collapse score: {:.4}", stats.mode_collapse_score);
println!("Unique tokens: {}", stats.unique_tokens);
```
