# CLI Audit Example

This example demonstrates the audit command for bias detection and fairness analysis.

## Running the Example

```bash
cargo run --example cli_audit
```

## Code

```rust
{{#include ../../../examples/cli_audit.rs}}
```

## CLI Usage

```bash
# Audit predictions for bias
entrenar audit predictions.parquet --type bias --threshold 0.8

# Audit with specific protected attributes
entrenar audit data.parquet --protected-attrs gender,race --threshold 0.1
```

## Audit Types

| Type | Description | Metrics |
|------|-------------|---------|
| `bias` | Detect prediction bias | Disparate impact ratio |
| `fairness` | Fairness constraints | Equal opportunity diff |
| `drift` | Data drift detection | PSI, KS statistic |

## Output Format

```json
{
  "audit_type": "bias",
  "threshold": 0.8,
  "passed": false,
  "metrics": {
    "disparate_impact": 0.72,
    "statistical_parity_diff": 0.15
  },
  "recommendations": [
    "Consider rebalancing training data",
    "Apply fairness constraints during training"
  ]
}
```
