# Code Generation GANs

GANs for generating valid Rust AST candidates.

## Configuration

```rust
use entrenar::generative::{CodeGanConfig, GeneratorConfig, DiscriminatorConfig};

let config = CodeGanConfig {
    generator: GeneratorConfig {
        latent_dim: 128,
        hidden_dims: vec![256, 512, 256],
        vocab_size: 100,
        max_seq_len: 64,
        dropout: 0.1,
        batch_norm: true,
    },
    discriminator: DiscriminatorConfig {
        vocab_size: 100,
        embed_dim: 64,
        hidden_dims: vec![128, 64],
        dropout: 0.1,
    },
    learning_rate_g: 0.0002,
    learning_rate_d: 0.0002,
    beta1: 0.5,
    beta2: 0.999,
};
```

## Training Loop

```rust
let mut gan = CodeGan::new(config);

for epoch in 0..num_epochs {
    for batch in data_loader {
        // Train discriminator
        let d_loss = gan.train_discriminator_step(&batch);

        // Train generator
        let g_loss = gan.train_generator_step();

        // Monitor for mode collapse
        let collapse_score = gan.detect_mode_collapse(100);
        if collapse_score > 0.8 {
            eprintln!("Warning: Mode collapse detected!");
        }
    }
}
```

## Evaluation

```rust
// Generate samples
let samples = gan.generate_batch(100);

// Check diversity
let diversity = gan.compute_diversity(&samples);
println!("Sample diversity: {:.2}", diversity);

// Interpolate between codes
let z1 = gan.sample_latent(1)[0].clone();
let z2 = gan.sample_latent(1)[0].clone();
let interpolated = gan.interpolate(&z1, &z2, 10);
```
