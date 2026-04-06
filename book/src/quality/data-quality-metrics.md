# Data Quality Metrics

Measuring dataset quality for ML visualization.

## Core Dimensions

| Dimension | Description | Threshold |
|-----------|-------------|-----------|
| Completeness | Non-null values | ≥95% |
| Uniqueness | Distinct records | ≥99% |
| Validity | Format compliance | ≥98% |
| Consistency | Cross-field rules | ≥95% |
| Timeliness | Data freshness | App-specific |

## Completeness

```rust
fn completeness(values: &[Option<Value>]) -> f32 {
    let non_null = values.iter().filter(|v| v.is_some()).count();
    non_null as f32 / values.len() as f32
}
```

## Uniqueness

```rust
fn uniqueness<T: Hash + Eq>(values: &[T]) -> f32 {
    use std::collections::HashSet;
    let unique: HashSet<_> = values.iter().collect();
    unique.len() as f32 / values.len() as f32
}
```

## Validity

| Field Type | Validation Rule |
|------------|-----------------|
| Email | RFC 5322 pattern |
| Phone | E.164 format |
| Date | ISO 8601 |
| Currency | Numeric, 2 decimals |

## Data Card Requirements

```yaml
data_card:
  name: "Sales Dataset"
  version: "2024.1"
  rows: 1_000_000
  quality:
    completeness: 0.97
    uniqueness: 0.995
    validity: 0.99
  columns:
    - name: "revenue"
      type: "float64"
      null_count: 1234
```

## Quality Score

```rust
fn quality_score(metrics: &DataQuality) -> f32 {
    metrics.completeness * 0.3
        + metrics.uniqueness * 0.25
        + metrics.validity * 0.25
        + metrics.consistency * 0.2
}
```

## Verified Test

```rust
#[test]
fn test_data_quality_completeness() {
    // Completeness calculation
    let values = vec![
        Some(1), Some(2), None, Some(4), Some(5),
        Some(6), None, Some(8), Some(9), Some(10),
    ];

    let non_null = values.iter().filter(|v| v.is_some()).count();
    let completeness = non_null as f32 / values.len() as f32;

    assert_eq!(non_null, 8);
    assert_eq!(completeness, 0.8);

    // Threshold check
    let threshold = 0.95;
    assert!(completeness < threshold);  // Fails threshold
}
```
