# Latent Space Interpolation

Explore the code generation space through latent vector manipulation.

## LatentCode Type

```rust
use entrenar::generative::LatentCode;

// Sample from standard normal
let z = LatentCode::sample(&mut rng, 128);

// Create from vector
let z = LatentCode::new(vec![0.0; 128]);

// Properties
println!("Dimension: {}", z.dim());
println!("Norm: {}", z.norm());
```

## Linear Interpolation (LERP)

Simple straight-line interpolation:

```rust
let z1 = LatentCode::new(vec![0.0; 128]);
let z2 = LatentCode::new(vec![1.0; 128]);

// t=0 gives z1, t=1 gives z2
let z_mid = z1.lerp(&z2, 0.5);
```

## Spherical Linear Interpolation (SLERP)

Interpolate along the surface of a hypersphere:

```rust
let z1 = LatentCode::sample(&mut rng, 128).normalize();
let z2 = LatentCode::sample(&mut rng, 128).normalize();

// SLERP maintains constant norm
let z_mid = z1.slerp(&z2, 0.5);
assert!((z_mid.norm() - 1.0).abs() < 0.1);
```

### When to Use SLERP

- Latent vectors are typically sampled from unit sphere
- SLERP avoids "dead zones" in latent space
- Smoother visual transitions for image GANs
- Better semantic interpolation for code GANs

## Interpolation for Code Generation

```rust
let mut gan = CodeGan::new(config);

// Generate two random codes
let z1 = gan.sample_latent(1)[0].clone();
let z2 = gan.sample_latent(1)[0].clone();

// Generate 11 intermediate samples
let samples = gan.interpolate(&z1, &z2, 10);

for (i, code) in samples.iter().enumerate() {
    println!("Step {}: {:?}", i, code);
}
```

## Latent Space Arithmetic

Discover semantic directions:

```rust
// Vector arithmetic in latent space
// e.g., "loop" - "if" + "match" might give switch-like patterns
let z_loop = encode_code("for i in 0..n { }");
let z_if = encode_code("if cond { }");
let z_match = encode_code("match x { }");

let z_result = z_loop.subtract(&z_if).add(&z_match);
let code = generator.generate(&z_result);
```
