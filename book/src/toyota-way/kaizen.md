# Kaizen

Continuous improvement through small, incremental changes.

## Principle

| Japanese | English | Application |
|----------|---------|-------------|
| Kai | Change | Iterative updates |
| Zen | Good | Quality improvement |

## Kaizen Cycle

```
Plan → Do → Check → Act → (repeat)
```

## Applied to Code

### Before Kaizen

```rust
// Manual, error-prone
fn calculate_score() {
    let a = get_value_a();
    let b = get_value_b();
    let c = get_value_c();
    let score = a * 0.3 + b * 0.4 + c * 0.3;
}
```

### After Kaizen

```rust
// Extracted weights, testable
const WEIGHTS: [f32; 3] = [0.3, 0.4, 0.3];

fn calculate_score(values: &[f32]) -> f32 {
    values.iter()
        .zip(WEIGHTS.iter())
        .map(|(v, w)| v * w)
        .sum()
}
```

## Small Improvements

| Area | Improvement |
|------|-------------|
| Performance | Cache layout results |
| Readability | Extract helper functions |
| Testing | Add edge case tests |
| Safety | Replace unwrap with Result |

## Measurement

```rust
// Track improvement over time
struct MetricHistory {
    frame_times: Vec<f32>,
    test_coverage: Vec<f32>,
}

impl MetricHistory {
    fn trend(&self) -> Trend {
        // Compare recent to historical
    }
}
```

## Review Process

1. Identify small improvement
2. Implement change
3. Measure impact
4. Commit if positive
5. Repeat

## Verified Test

```rust
#[test]
fn test_kaizen_weighted_score() {
    // Extracted weights enable testing
    const WEIGHTS: [f32; 3] = [0.3, 0.4, 0.3];

    fn weighted_score(values: &[f32], weights: &[f32]) -> f32 {
        values.iter()
            .zip(weights.iter())
            .map(|(v, w)| v * w)
            .sum()
    }

    let values = [100.0, 80.0, 90.0];
    let score = weighted_score(&values, &WEIGHTS);

    // 30 + 32 + 27 = 89
    assert_eq!(score, 89.0);

    // Easy to test edge cases
    assert_eq!(weighted_score(&[0.0, 0.0, 0.0], &WEIGHTS), 0.0);
}
```
