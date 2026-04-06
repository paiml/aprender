# Generator Architecture

The generator maps latent vectors to code token sequences.

## Architecture

```
Input: z ∈ R^latent_dim
    ↓
Linear(latent_dim → hidden[0])
    ↓
ReLU + BatchNorm (optional)
    ↓
Linear(hidden[0] → hidden[1])
    ↓
ReLU + BatchNorm (optional)
    ↓
...
    ↓
Linear(hidden[-1] → vocab_size * max_seq_len)
    ↓
Reshape to (max_seq_len, vocab_size)
    ↓
Softmax per position
    ↓
Argmax → Token IDs
```

## Configuration

```rust
let config = GeneratorConfig {
    latent_dim: 128,        // Size of noise vector
    hidden_dims: vec![256, 512, 256],  // MLP layers
    vocab_size: 100,        // Number of token types
    max_seq_len: 64,        // Maximum output length
    dropout: 0.1,           // Regularization
    batch_norm: true,       // Stabilize training
};
```

## Initialization

Xavier initialization for stable gradients:

```rust
fn xavier_init(fan_in: usize, fan_out: usize) -> f32 {
    let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
    rand::random::<f32>() * 2.0 * limit - limit
}
```

## Generation

```rust
let generator = Generator::new(config);
let z = LatentCode::sample(&mut rng, 128);
let tokens: Vec<u32> = generator.generate(&z);
```

## Temperature Sampling

Control diversity vs quality:

```rust
fn generate_with_temperature(&self, z: &LatentCode, temp: f32) -> Vec<u32> {
    let logits = self.forward(z);

    logits.chunks(self.vocab_size)
        .map(|chunk| {
            let scaled: Vec<f32> = chunk.iter()
                .map(|x| x / temp)
                .collect();
            softmax_sample(&scaled)
        })
        .collect()
}
```
