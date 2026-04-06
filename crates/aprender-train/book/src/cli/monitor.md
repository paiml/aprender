# Monitor Command

The `entrenar monitor` command detects data drift using Population Stability Index (PSI) and other statistical measures.

## Usage

```bash
entrenar monitor <INPUT> [OPTIONS]
```

## Arguments

| Argument | Description |
|----------|-------------|
| `<INPUT>` | Path to current data file to monitor |

## Options

| Option | Description |
|--------|-------------|
| `--baseline <PATH>` | Path to baseline statistics file |
| `--threshold <T>` | PSI threshold for drift alert (default: 0.2) |
| `--interval <SECS>` | Monitoring interval in seconds (default: 60) |
| `--format <FORMAT>` | Output format: text, json, yaml (default: text) |

## Understanding PSI

Population Stability Index (PSI) measures how much a distribution has shifted:

```
PSI = Σ (actual_% - expected_%) × ln(actual_% / expected_%)
```

| PSI Value | Interpretation |
|-----------|----------------|
| < 0.1 | No significant shift |
| 0.1 - 0.2 | Moderate shift |
| > 0.2 | Significant shift (action required) |

## Examples

### Basic Drift Detection

```bash
entrenar monitor data/current.parquet
```

Output:
```
Monitoring: data/current.parquet
  Drift threshold (PSI): 0.2
Drift Monitoring Results:
  PSI score: 0.0042
  Threshold: 0.2000
  Severity: low
  Status: NO DRIFT
```

### Monitor with Baseline

```bash
entrenar monitor data/current.parquet --baseline data/baseline.parquet
```

### Custom Threshold

```bash
# Stricter threshold for production
entrenar monitor data/current.parquet --threshold 0.1
```

### JSON Output for Alerting

```bash
entrenar monitor data/current.parquet --format json
```

Output:
```json
{
  "psi_score": 0.0042,
  "threshold": 0.2,
  "status": "NO DRIFT",
  "severity": "low",
  "drift_detected": false,
  "buckets": {
    "baseline": [0.10, 0.15, 0.20, 0.25, 0.15, 0.10, 0.05],
    "current": [0.11, 0.14, 0.19, 0.26, 0.16, 0.09, 0.05]
  }
}
```

## Severity Levels

| Severity | PSI Range | Action |
|----------|-----------|--------|
| low | < 0.1 | No action needed |
| moderate | 0.1 - threshold | Consider investigation |
| high | > threshold | Immediate investigation required |

## CI/CD Integration

Use monitor in CI pipelines to detect drift:

```yaml
# GitHub Actions example
- name: Drift Detection
  run: |
    entrenar monitor data/production.parquet \
      --baseline data/training.parquet \
      --threshold 0.2
```

```bash
# Shell script with alerting
result=$(entrenar monitor data/current.parquet --format json)
drift_detected=$(echo $result | jq '.drift_detected')

if [ "$drift_detected" = "true" ]; then
    echo "ALERT: Data drift detected!"
    # Send alert to monitoring system
    exit 1
fi
```

## Continuous Monitoring

For continuous monitoring, use the `--interval` option:

```bash
# Monitor every 5 minutes
entrenar monitor data/stream.parquet --interval 300
```

## PSI Calculation Details

The PSI is calculated by:

1. Binning both distributions into buckets
2. Computing the percentage in each bucket
3. Applying the PSI formula

```rust
let mut psi = 0.0;
for (expected, actual) in baseline_buckets.iter().zip(current_buckets.iter()) {
    if *expected > 0.0 && *actual > 0.0 {
        psi += (*actual - *expected) * (*actual / *expected).ln();
    }
}
psi = psi.abs();
```

## Best Practices

1. **Establish baselines** - Use training data distribution as baseline
2. **Set appropriate thresholds** - Start with 0.2, adjust based on domain
3. **Monitor regularly** - Daily or per-batch in production
4. **Alert on drift** - Integrate with monitoring systems
5. **Investigate promptly** - Drift may indicate data quality issues

## See Also

- [CLI Overview](./overview.md) - General CLI reference
- [Audit Command](./audit.md) - Bias and fairness auditing
- [Drift Detection](../monitor/drift-detection.md) - Drift detection theory
- [Quality Gates](../monitor/quality-gates.md) - Jidoka quality gates
