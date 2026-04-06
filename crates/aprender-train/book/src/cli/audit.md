# Audit Command

The `entrenar audit` command performs bias detection, fairness analysis, privacy checks, and security audits on models
and datasets.

## Usage

```bash
entrenar audit <INPUT> [OPTIONS]
```

## Arguments

| Argument | Description |
|----------|-------------|
| `<INPUT>` | Path to model or data file to audit |

## Options

| Option | Description |
|--------|-------------|
| `--type <TYPE>` | Audit type: bias, fairness, privacy, security (default: bias) |
| `--protected-attr <ATTR>` | Protected attribute column for bias analysis |
| `--threshold <T>` | Pass/fail threshold (default: 0.8) |
| `--format <FORMAT>` | Output format: text, json, yaml (default: text) |

## Audit Types

### Bias Audit (default)

Detect demographic bias using statistical parity metrics:

```bash
entrenar audit predictions.parquet --type bias --threshold 0.8
```

Output:
```
Auditing: predictions.parquet
  Audit type: bias
  Threshold: 0.8
Bias Audit Results:
  Demographic parity ratio: 0.923
  Equalized odds: 0.970
  Threshold: 0.800
  Status: PASS
```

#### Metrics Calculated

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Demographic Parity Ratio | min(P(Y=1\|A=0), P(Y=1\|A=1)) / max(...) | 1.0 = perfect parity |
| Equalized Odds | 1 - \|TPR_A - TPR_B\| | 1.0 = equal true positive rates |

### Fairness Audit

Check model calibration and fairness across groups:

```bash
entrenar audit model.safetensors --type fairness
```

Output:
```
Fairness Audit Results:
  Calibration error: 0.050
  Status: PASS
```

### Privacy Audit

Scan for PII patterns in data:

```bash
entrenar audit data.parquet --type privacy
```

Output:
```
Privacy Audit Results:
  PII scan: Complete
  Email patterns: 0 found
  Phone patterns: 0 found
  SSN patterns: 0 found
  Status: PASS
```

### Security Audit

Check for security vulnerabilities in model files:

```bash
entrenar audit model.safetensors --type security
```

Output:
```
Security Audit Results:
  Pickle deserialization: Safe (SafeTensors)
  Code execution vectors: None
  Status: PASS
```

## JSON Output

Get machine-readable results for CI/CD integration:

```bash
entrenar audit predictions.parquet --type bias --format json
```

Output:
```json
{
  "audit_type": "bias",
  "demographic_parity_ratio": 0.923,
  "equalized_odds": 0.970,
  "threshold": 0.8,
  "pass": true
}
```

## Threshold Configuration

The `--threshold` option sets the minimum acceptable value:

| Audit Type | Threshold Meaning |
|------------|-------------------|
| bias | Minimum demographic parity ratio |
| fairness | Maximum acceptable calibration error = 1 - threshold |
| privacy | N/A (binary pass/fail) |
| security | N/A (binary pass/fail) |

```bash
# Strict bias threshold (>90% parity required)
entrenar audit data.parquet --type bias --threshold 0.9

# Relaxed threshold for development
entrenar audit data.parquet --type bias --threshold 0.7
```

## CI/CD Integration

Use audit commands in CI pipelines with exit codes:

```yaml
# GitHub Actions example
- name: Bias Audit
  run: |
    entrenar audit predictions.parquet --type bias --threshold 0.8
    if [ $? -ne 0 ]; then
      echo "Bias audit failed!"
      exit 1
    fi
```

```bash
# Shell script
entrenar audit model.safetensors --type security || {
    echo "Security audit failed!"
    exit 1
}
```

## Examples

### Complete Audit Pipeline

```bash
# 1. Security audit on model
entrenar audit model.safetensors --type security

# 2. Privacy audit on training data
entrenar audit data/train.parquet --type privacy

# 3. Bias audit on predictions
entrenar audit predictions.parquet \
  --type bias \
  --protected-attr gender \
  --threshold 0.8

# 4. Fairness audit
entrenar audit predictions.parquet --type fairness
```

### Audit with Protected Attribute

```bash
entrenar audit predictions.parquet \
  --type bias \
  --protected-attr race \
  --threshold 0.85 \
  --format json > audit_results.json
```

## Programmatic Usage

```rust
// Demographic parity calculation
let group_a_positive_rate = 0.72;
let group_b_positive_rate = 0.78;

let demographic_parity = (group_a_positive_rate / group_b_positive_rate)
    .min(group_b_positive_rate / group_a_positive_rate);

// Equalized odds calculation
let group_a_tpr = 0.85;
let group_b_tpr = 0.82;
let equalized_odds = 1.0 - (group_a_tpr - group_b_tpr).abs();

let pass = demographic_parity >= threshold;
```

## See Also

- [CLI Overview](./overview.md) - General CLI reference
- [Monitor Command](./monitor.md) - Drift detection
- [Quality Gates](../monitor/quality-gates.md) - Jidoka quality gates
