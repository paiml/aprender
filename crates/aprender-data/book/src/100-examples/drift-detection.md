# Drift Detection (Examples 56-65)

This section covers distribution drift detection between datasets.

## Examples 56-57: Basic Drift and KS Test

```rust
use alimentar::{ArrowDataset, DriftDetector};

let baseline = ArrowDataset::from_parquet("baseline.parquet")?;
let current = ArrowDataset::from_parquet("current.parquet")?;

let detector = DriftDetector::new(baseline);
let report = detector.detect(&current)?;

for (column, score) in &report.column_scores {
    println!("{}: drift={:.3}", column, score.drift_score);
}
```

## Examples 58-59: Chi-Square and PSI

```rust
use alimentar::{DriftDetector, DriftTest};

let detector = DriftDetector::new(baseline)
    .add_test(DriftTest::ChiSquare)  // For categorical
    .add_test(DriftTest::PSI(10));   // PSI with 10 buckets

let report = detector.detect(&current)?;
```

## Examples 60-61: Severity and Column-Level Drift

```rust
use alimentar::DriftDetector;

let detector = DriftDetector::new(baseline);
let report = detector.detect(&current)?;

// Overall severity
println!("Severity: {:?}", report.severity);

// Per-column analysis
for (col, score) in &report.column_scores {
    if score.drift_detected {
        println!("DRIFT in {}: {} ({:?})",
            col, score.drift_score, score.test_type);
    }
}
```

## Examples 62-64: Thresholds and Sketches

```rust
use alimentar::{DriftDetector, sketch::{DDSketch, TDigest}};

// Custom threshold
let detector = DriftDetector::new(baseline)
    .threshold(0.1); // 0.1 = 10% drift threshold

// Using sketches for streaming
let sketch = DDSketch::new();
for batch in streaming {
    sketch.insert_batch(&batch, "value")?;
}
let merged = DDSketch::merge(vec![sketch1, sketch2])?;
```

## Example 65: Drift Report Export

```rust
use alimentar::DriftDetector;

let detector = DriftDetector::new(baseline);
let report = detector.detect(&current)?;

// Export to JSON
let json = report.to_json()?;
std::fs::write("drift_report.json", json)?;
```

## CLI Usage

```bash
# Compare two datasets
alimentar drift compare baseline.parquet current.parquet

# Specific tests
alimentar drift detect --tests ks,psi baseline.parquet current.parquet

# JSON output
alimentar drift compare --format json baseline.parquet current.parquet

# Create sketch for incremental comparison
alimentar drift sketch data.parquet --output sketch.bin
alimentar drift merge sketch1.bin sketch2.bin --output merged.bin
```

## Drift Tests Available

| Test | Type | Description |
|------|------|-------------|
| KS | Numeric | Kolmogorov-Smirnov test |
| PSI | Numeric | Population Stability Index |
| ChiSquare | Categorical | Chi-squared test |
| JensenShannon | Both | JS divergence |
| Wasserstein | Numeric | Earth mover's distance |

## Key Concepts

- **Baseline**: Reference distribution
- **Sketches**: Memory-efficient summaries
- **Merge**: Combine sketches from distributed systems
- **Severity**: None/Low/Medium/High classification
