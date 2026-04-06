# CLI Monitor Example

This example demonstrates drift monitoring on production data.

## Running the Example

```bash
cargo run --example cli_monitor
```

## Code

```rust
{{#include ../../../examples/cli_monitor.rs}}
```

## CLI Usage

```bash
# Monitor for drift using PSI
entrenar monitor data.parquet --threshold 0.2

# Monitor specific features
entrenar monitor data.parquet --features age,income,score

# Continuous monitoring
entrenar monitor data.parquet --watch --interval 60
```

## Drift Detection Methods

| Method | Threshold | Interpretation |
|--------|-----------|----------------|
| PSI | 0.1 | Minor drift |
| PSI | 0.2 | Significant drift |
| KS | 0.05 | Statistical significance |

## Output

```
Drift Detection Report
======================

Feature: age
  PSI: 0.08 (No significant drift)
  KS p-value: 0.23

Feature: income
  PSI: 0.31 (⚠️ Significant drift!)
  KS p-value: 0.001

Recommendation: Retrain model with recent data
```
