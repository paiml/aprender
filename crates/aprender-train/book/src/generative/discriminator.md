# Discriminator Architecture

The discriminator classifies code as real or generated.

## Architecture

```
Input: tokens ∈ N^seq_len
    ↓
Embedding(vocab_size → embed_dim)
    ↓
Flatten to (seq_len * embed_dim)
    ↓
Linear → ReLU
    ↓
Linear → ReLU
    ↓
...
    ↓
Linear → 1
    ↓
Sigmoid → P(real)
```

## Configuration

```rust
let config = DiscriminatorConfig {
    vocab_size: 100,        // Number of token types
    embed_dim: 64,          // Embedding dimension
    hidden_dims: vec![128, 64],  // MLP layers
    dropout: 0.1,           // Regularization
};
```

## Training Objective

Binary cross-entropy loss:

```rust
fn discriminator_loss(
    d_real: f32,    // D(x) for real sample
    d_fake: f32,    // D(G(z)) for generated sample
) -> f32 {
    // Maximize: log(D(x)) + log(1 - D(G(z)))
    let real_loss = -d_real.ln();
    let fake_loss = -(1.0 - d_fake).ln();
    real_loss + fake_loss
}
```

## Label Smoothing

Improve training stability:

```rust
fn train_with_smoothing(&mut self, real: &[u32], fake: &[u32]) {
    // Soft labels: 0.9 for real, 0.1 for fake
    let real_label = 0.9;
    let fake_label = 0.1;

    let d_real = self.discriminator.forward(real);
    let d_fake = self.discriminator.forward(fake);

    let loss = bce(d_real, real_label) + bce(d_fake, fake_label);
    self.backward(loss);
}
```

## Feature Matching

Alternative to standard GAN loss:

```rust
fn feature_matching_loss(
    discriminator: &Discriminator,
    real: &[u32],
    fake: &[u32],
) -> f32 {
    let real_features = discriminator.intermediate_features(real);
    let fake_features = discriminator.intermediate_features(fake);

    mse(&real_features, &fake_features)
}
```
