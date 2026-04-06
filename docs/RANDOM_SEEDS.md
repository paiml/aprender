# Random Seed Management

This document describes how Realizar handles random seeds for reproducible inference.

## Seed Philosophy

**Principle**: All stochastic operations must be deterministic given a seed.

## Seed Sources

### 1. Explicit User Seed

```rust
// Rust API
let config = SamplingConfig {
    seed: Some(42),
    ..Default::default()
};

let output = model.generate("Hello", &config)?;
```

```bash
# CLI
realizar infer --seed 42 "Hello world"
```

```json
// REST API
{
  "prompt": "Hello",
  "seed": 42
}
```

### 2. Default Seed

When no seed is provided:
- **Default value**: 42
- **Rationale**: Reproducibility by default

```rust
impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            seed: Some(42),  // Explicit default for reproducibility
            temperature: 1.0,
            top_k: None,
            top_p: None,
            max_tokens: 100,
        }
    }
}
```

### 3. Random Seed

For non-reproducible generation:

```bash
realizar infer --seed random "Hello world"
```

```json
{
  "prompt": "Hello",
  "seed": "random"
}
```

## PRNG Implementation

### Algorithm

We use **xoshiro256++** for random number generation:
- Fast (0.75 ns per 64-bit number)
- Good statistical properties (passes BigCrush)
- Small state (256 bits)
- Reproducible across platforms

### Seeding Process

```rust
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn create_rng(seed: u64) -> Xoshiro256PlusPlus {
    // Expand 64-bit seed to 256-bit state deterministically
    let mut hasher = blake3::Hasher::new();
    hasher.update(&seed.to_le_bytes());
    let hash = hasher.finalize();

    let mut state = [0u8; 32];
    state.copy_from_slice(hash.as_bytes());

    Xoshiro256PlusPlus::from_seed(state)
}
```

## Stochastic Operations

### Temperature Sampling

```rust
fn sample_with_temperature(logits: &[f32], temp: f32, rng: &mut impl Rng) -> usize {
    let scaled: Vec<f32> = logits.iter().map(|x| x / temp).collect();
    let probs = softmax(&scaled);

    // Deterministic given RNG state
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for (i, p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    probs.len() - 1
}
```

### Top-K Sampling

```rust
fn sample_top_k(logits: &[f32], k: usize, rng: &mut impl Rng) -> usize {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);

    let probs = softmax(&indexed.iter().map(|x| x.1).collect::<Vec<_>>());

    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for (i, p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return indexed[i].0;
        }
    }
    indexed.last().unwrap().0
}
```

### Top-P (Nucleus) Sampling

```rust
fn sample_top_p(logits: &[f32], p: f32, rng: &mut impl Rng) -> usize {
    let probs = softmax(logits);
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cumsum = 0.0;
    let mut cutoff = indexed.len();
    for (i, (_, prob)) in indexed.iter().enumerate() {
        cumsum += prob;
        if cumsum > p {
            cutoff = i + 1;
            break;
        }
    }
    indexed.truncate(cutoff);

    // Re-normalize and sample
    let sum: f32 = indexed.iter().map(|x| x.1).sum();
    let normalized: Vec<f32> = indexed.iter().map(|x| x.1 / sum).collect();

    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for (i, p) in normalized.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return indexed[i].0;
        }
    }
    indexed.last().unwrap().0
}
```

## Determinism Guarantees

### Guaranteed Deterministic

| Operation | Condition |
|-----------|-----------|
| Greedy decoding | Always |
| Temperature sampling | Same seed |
| Top-K sampling | Same seed |
| Top-P sampling | Same seed |
| Tensor operations | Always (CPU) |

### Not Guaranteed Deterministic

| Operation | Reason |
|-----------|--------|
| GPU inference | Floating-point non-associativity |
| Multi-threaded inference | Thread scheduling |

## Verification

### Unit Test

```rust
#[test]
fn test_seed_reproducibility() {
    let model = Model::demo();
    let config = SamplingConfig {
        seed: Some(42),
        temperature: 0.7,
        ..Default::default()
    };

    let output1 = model.generate("Hello", &config).unwrap();
    let output2 = model.generate("Hello", &config).unwrap();

    assert_eq!(output1, output2, "Same seed must produce same output");
}
```

### Integration Test

```bash
#!/bin/bash
OUTPUT1=$(realizar infer --seed 42 "Hello")
OUTPUT2=$(realizar infer --seed 42 "Hello")

if [ "$OUTPUT1" != "$OUTPUT2" ]; then
    echo "FAIL: Outputs differ with same seed"
    exit 1
fi
echo "PASS: Reproducible with seed 42"
```

## Seed Documentation in Results

Always document seeds in benchmark/experiment results:

```markdown
## Experiment Configuration

- **Seed**: 42
- **Temperature**: 0.7
- **Top-K**: 40
- **Max Tokens**: 100
```

## References

1. Blackman, D., & Vigna, S. (2021). Scrambled Linear Pseudorandom Number Generators.
2. L'Ecuyer, P., & Simard, R. (2007). TestU01: A C library for empirical testing of random number generators.

---

**Document Version**: 1.0.0
**Last Updated**: 2025-12-10
