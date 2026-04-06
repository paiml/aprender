# ADR-0006: Inference Determinism Strategy

## Status

Accepted

## Date

2024-11-28

## Context

ML inference can be non-deterministic due to:
- Floating-point operation ordering
- Thread scheduling
- GPU execution order
- Random sampling in generation

Users need reproducible results for:
- Debugging and testing
- Scientific experiments
- Regulatory compliance

## Decision

Guarantee deterministic CPU inference given the same seed and input.

## Implementation

### 1. Fixed PRNG Seeding

```rust
pub struct SamplingConfig {
    pub seed: Option<u64>,  // None = random, Some(42) = deterministic
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            seed: Some(42),  // Reproducible by default
            temperature: 1.0,
            top_k: None,
            top_p: None,
        }
    }
}
```

### 2. Deterministic Operations

All tensor operations use:
- Sequential iteration (no parallel non-determinism)
- Fixed operation order (no reassociation)
- Stable sort algorithms

### 3. Greedy Decoding Path

```rust
pub fn greedy_sample(logits: &[f32]) -> usize {
    // Always deterministic - returns argmax
    logits.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}
```

## Consequences

### Positive
- Reproducible results for testing and debugging
- Scientific experiments are repeatable
- CI tests are deterministic

### Negative
- Cannot use non-deterministic GPU optimizations
- Single-threaded inference path (slower)
- Must document all sources of randomness

## Trade-offs

| Mode | Determinism | Performance |
|------|-------------|-------------|
| CPU + seed | Guaranteed | Baseline |
| CPU + random | Non-deterministic | Baseline |
| GPU + seed | Best effort | ~2x faster |
| GPU + random | Non-deterministic | ~2x faster |

## Verification

```rust
#[test]
fn test_determinism() {
    let model = Model::demo();
    let config = SamplingConfig {
        seed: Some(12345),
        temperature: 0.7,
        ..Default::default()
    };

    let results: Vec<_> = (0..10)
        .map(|_| model.generate("Test", &config).unwrap())
        .collect();

    assert!(results.windows(2).all(|w| w[0] == w[1]));
}
```

## References

- [Reproducibility in Deep Learning](https://pytorch.org/docs/stable/notes/randomness.html)
- [ML Reproducibility Checklist](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)
