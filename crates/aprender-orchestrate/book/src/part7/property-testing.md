# Property-Based Testing

Property-based testing verifies that invariants hold across thousands of randomly generated inputs. The Sovereign AI Stack uses `proptest` for numerical correctness and data structure validation.

## Core Concept

Instead of testing specific pairs, define properties that must always be true:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn normalize_produces_unit_vector(v in prop::collection::vec(-1000.0f32..1000.0, 3..128)) {
        let normalized = normalize(&v);
        let magnitude: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        prop_assert!((magnitude - 1.0).abs() < 1e-5);
    }
}
```

## Common Property Patterns

| Property | Description | Example |
|----------|-------------|---------|
| Round-trip | encode then decode equals original | serialize/deserialize |
| Idempotent | applying twice equals once | normalize, deduplicate |
| Invariant | condition always holds | sorted output, non-negative |
| Oracle | matches known-good implementation | Rust vs Python output |

## Strategy Composition

Build complex input generators from simple ones:

```rust
fn model_config_strategy() -> impl Strategy<Value = ModelConfig> {
    (1usize..=32, 64usize..=4096, 1usize..=64)
        .prop_map(|(layers, hidden, heads)| ModelConfig {
            num_layers: layers,
            hidden_size: hidden - (hidden % heads),
            num_heads: heads,
        })
}
```

## Shrinking

When proptest finds a failure, it shrinks to the minimal reproduction:

```
Minimal failing input: ModelConfig { num_layers: 1, hidden_size: 64, num_heads: 65 }
```

## Combining with Mutation Testing

Property tests are excellent mutation killers. A mutation changing `<` to `<=` will likely violate an invariant across thousands of inputs:

```bash
make mutants-fast    # Find surviving mutants
# Write property tests targeting survivors
make mutants         # Verify mutations are killed
```

## CI Integration

Property tests run as standard `cargo test`. CI can increase case count:

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]
    #[test]
    fn exhaustive_check(input in any::<u32>()) { /* ... */ }
}
```

---

**Navigate:** [Table of Contents](../SUMMARY.md)
