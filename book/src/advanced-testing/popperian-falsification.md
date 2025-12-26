# Popperian Falsification Testing

Karl Popper's criterion of demarcation states that scientific claims must be **falsifiable**—there must exist possible observations that would prove them false. We apply this rigorous standard to software testing.

> "A theory which is not refutable by any conceivable event is non-scientific. Irrefutability is not a virtue of a theory but a vice." — Karl Popper, *Conjectures and Refutations* (1963)

## Why Falsification Over Verification?

Traditional testing asks: "Does this work?"
Falsification testing asks: "Under what conditions would this **fail**?"

This shift in perspective is powerful because:

1. **Specificity**: Falsification conditions are precise and measurable
2. **Coverage**: Forces consideration of edge cases and failure modes
3. **Rigor**: Can never "prove" correctness, only fail to falsify
4. **Documentation**: Falsification conditions become living specifications

## Falsification Hierarchy

```
Level 0: Logical Falsification
  └─→ Type system prevents invalid states
  └─→ Example: "APR files always have valid headers"

Level 1: Unit Falsification
  └─→ Single function produces wrong output
  └─→ Example: "mel_spectrogram() matches librosa within 1e-5"

Level 2: Integration Falsification
  └─→ Components fail to interoperate
  └─→ Example: "apr import | apr run produces output"

Level 3: System Falsification
  └─→ End-to-end failure under realistic conditions
  └─→ Example: "Browser inference runs for 1 hour without crash"

Level 4: Performance Falsification
  └─→ Performance claims are not met
  └─→ Example: "Decode speed ≥ 50 tok/s on reference hardware"
```

## Writing Falsification Tests

A good falsification test has three parts:

1. **Claim**: What property should hold?
2. **Falsification Condition**: What observation would disprove it?
3. **Test Method**: How do we check for the falsification condition?

### Example: Quantization Determinism (BB3)

```rust
/// BB3: Quantization must be deterministic
/// Falsification: Same input produces different output
#[test]
fn test_bb3_quantization_deterministic() {
    let data: Vec<f32> = (0..128)
        .map(|i| (i as f32 - 64.0) * 0.01)
        .collect();
    let shape = vec![128];

    // Run quantization 10 times
    let mut results: Vec<Vec<u8>> = Vec::new();
    for _ in 0..10 {
        let quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");
        results.push(quantized.blocks.clone());
    }

    // All results must be identical
    let first = &results[0];
    for (i, result) in results.iter().enumerate().skip(1) {
        assert_eq!(
            first, result,
            "BB3 FALSIFIED: Quantization run {} differs from run 0",
            i
        );
    }
}
```

### Example: No Telemetry (DD3)

```rust
/// DD3: No telemetry symbols in binary
/// FALSIFICATION: Binary contains "telemetry", "analytics" dependencies
#[test]
fn dd3_no_telemetry_symbols() {
    let cargo_toml = include_str!("../Cargo.toml");

    let telemetry_patterns = [
        "telemetry", "analytics", "sentry", "datadog",
        "newrelic", "opentelemetry", "amplitude", "mixpanel",
    ];

    for pattern in telemetry_patterns {
        assert!(
            !cargo_toml.to_lowercase().contains(pattern),
            "DD3 FALSIFIED: Cargo.toml contains telemetry dependency: '{}'",
            pattern
        );
    }
}
```

## Aprender's Falsification Sections

The specification defines falsification tests in themed sections:

| Section | Domain | Example Tests |
|---------|--------|---------------|
| **AA** | Audio Processing | Resampling accuracy, streaming integrity |
| **BB** | Quantization | Round-trip error, determinism, GGUF compat |
| **CC** | Cross-Repository | APR format parity, version compatibility |
| **DD** | Sovereign Compliance | No telemetry, air-gap license, provenance |

### Running Falsification Tests

```bash
# Run BB (Quantization) falsification tests
cargo test --lib --features format-quantize tests_falsification_bb

# Run DD (Sovereign Compliance) tests
cargo test --test format_parity_tests -- dd

# Run CC (Cross-Repository) tests
cargo test --test format_parity_tests -- cc

# Run all falsification tests
cargo test falsif
```

## Connection to Property-Based Testing

Falsification testing pairs naturally with property-based testing. While falsification defines *what* should never happen, property-based testing generates *inputs* to try to make it happen:

```rust
proptest! {
    /// BB3: Quantization is deterministic for ANY input
    #[test]
    fn prop_quantization_deterministic(
        weights in prop::collection::vec(-1.0f32..1.0, 32..1024),
    ) {
        let shape = vec![weights.len()];
        let q1 = quantize(&weights, &shape, QuantType::Q8_0)?;
        let q2 = quantize(&weights, &shape, QuantType::Q8_0)?;
        prop_assert_eq!(q1.blocks, q2.blocks, "FALSIFIED: Non-deterministic");
    }
}
```

## Best Practices

1. **Name tests after their falsification condition**: `test_bb3_quantization_deterministic`
2. **Include section ID in assertion messages**: `"BB3 FALSIFIED: ..."`
3. **Document the claim and falsification condition in doc comments**
4. **Use synthetic data to avoid OOM with large models**
5. **Run tests with required features**: `--features format-quantize`

## Further Reading

- Popper, K. (1959). *The Logic of Scientific Discovery*. Hutchinson.
- Popper, K. (1963). *Conjectures and Refutations*. Routledge.
- Claessen & Hughes (2000). *QuickCheck*. ICFP '00.
- DeMillo et al. (1978). *Hints on Test Data Selection*. IEEE Computer.
