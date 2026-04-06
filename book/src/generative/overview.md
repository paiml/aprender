# Generative Models

Entrenar provides Generative Adversarial Networks (GANs) for code synthesis.

## Architecture

```
Latent Vector z ─┬─► Generator ─► AST Tokens ─┬─► Discriminator ─► Valid/Invalid
                 │                            │
                 │   Real AST Samples ────────┘
                 │
                 └── (sampled from N(0, I))
```

## Key Components

### Generator
- Maps latent vectors to Rust AST token sequences
- MLP architecture with configurable hidden layers
- Xavier/He initialization for stable training

### Discriminator
- Classifies code as real (valid) or fake (invalid)
- Token embedding + MLP architecture
- Binary cross-entropy loss

### Latent Space
- Standard normal distribution sampling
- SLERP interpolation for smooth transitions
- Supports latent space exploration

## Quick Start

```rust
use entrenar::generative::{CodeGan, CodeGanConfig};

let config = CodeGanConfig::default();
let mut gan = CodeGan::new(config);

// Generate code samples
let latent = gan.sample_latent(10);
for z in &latent {
    let code = gan.generator.generate(z);
    println!("{:?}", code);
}
```

## Use Cases

1. **Code Completion**: Generate likely next tokens
2. **Data Augmentation**: Create synthetic training data
3. **Novelty Search**: Explore code generation space
4. **Transfer Learning**: Pre-train on code distributions
