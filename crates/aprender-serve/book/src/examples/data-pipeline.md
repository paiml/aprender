# Data Pipeline - Alimentar Integration

End-to-end ML data pipeline demonstrating the integration between alimentar (data loading) and realizar (inference).

## Overview

This example shows a complete ML workflow:

1. **Load data** with alimentar's built-in Iris dataset
2. **Quality check** for data issues
3. **Transform** data (shuffle, normalize)
4. **Split** into train/test sets
5. **Inference** with a classifier
6. **Evaluate** predictions
7. **Monitor** for drift

## Running the Example

```bash
cargo run --example data_pipeline --features alimentar-data
```

## Pipeline Stages

### Stage 1: Load Data

```rust
use alimentar::datasets::{iris, CanonicalDataset};

let dataset = iris().expect("Failed to load Iris dataset");

println!("Samples: {}", dataset.len());           // 150
println!("Features: {:?}", dataset.feature_names()); // sepal_length, sepal_width, ...
println!("Classes: {}", dataset.num_classes());   // 3
```

The Iris dataset is embedded in alimentar - no download required.

### Stage 2: Quality Check

```rust
use alimentar::{QualityChecker, QualityReport};

let checker = QualityChecker::new();
let report: QualityReport = checker.check(dataset.data())?;

println!("Rows: {}", report.row_count);
println!("Columns: {}", report.column_count);
println!("Issues: {}", report.issues.len());
```

Quality checks detect:
- Missing values
- High duplicate ratios
- Outliers
- Type mismatches

### Stage 3: Apply Transforms

```rust
use alimentar::{Dataset, Shuffle};

let shuffle = Shuffle::with_seed(42);
let shuffled = dataset
    .data()
    .with_transform(&shuffle)?;
```

Available transforms:
- `Shuffle` - Randomize row order
- `Select` - Choose specific columns
- `Filter` - Remove rows by condition
- `Normalize` - Scale features
- `Take`/`Skip` - Subset rows

### Stage 4: Train/Test Split

```rust
let total = shuffled.len();
let train_size = (total as f64 * 0.8) as usize;

// Training: first 80%
// Test: remaining 20%
```

### Stage 5: Extract Features

```rust
use arrow::array::Float64Array;

let batch = shuffled.get_batch(0)?;

let sepal_length = batch
    .column_by_name("sepal_length")?
    .as_any()
    .downcast_ref::<Float64Array>()?;
```

Arrow RecordBatch provides zero-copy access to columnar data.

### Stage 6: Run Inference

```rust
let classifier = IrisClassifier::new();
let predictions = classifier.predict_batch(&test_samples);
```

The example uses a simple decision tree classifier based on petal measurements.

### Stage 7: Evaluation Metrics

```rust
for (pred, actual) in predictions.iter().zip(test_labels.iter()) {
    metrics.record(pred.species.as_str(), actual);
}

println!("Accuracy: {:.2}%", metrics.accuracy() * 100.0);
println!("Macro F1: {:.4}", metrics.macro_f1());
```

Per-class metrics (precision, recall, F1) are computed automatically.

### Stage 8: Drift Detection

```rust
use alimentar::{DriftDetector, DriftTest};

let detector = DriftDetector::new(reference_data)
    .with_test(DriftTest::KolmogorovSmirnov)
    .with_alpha(0.05);

let drift_report = detector.detect(&current_data)?;

if drift_report.num_drifted() > 0 {
    println!("Drift detected in: {:?}", drift_report.drifted_columns());
}
```

Drift detection compares feature distributions between reference and current data.

### Stage 9: DataLoader Batching

```rust
use alimentar::DataLoader;

let loader = DataLoader::new(dataset)
    .batch_size(32)
    .shuffle(true);

for batch in loader {
    // Process Arrow RecordBatch
    println!("Batch: {} rows", batch.num_rows());
}
```

DataLoader provides efficient batched iteration with optional shuffling.

## Example Output

```
=== Data Pipeline: Alimentar + Realizar ===

Stage 1: Loading Iris dataset with alimentar...
  Dataset: Iris flower dataset (Fisher, 1936)
  Samples: 150
  Features: ["sepal_length", "sepal_width", "petal_length", "petal_width"]
  Classes: 3

Stage 2: Data quality check...
  Total rows: 150
  Total columns: 5
  Issues found: 5
  Warning: HighDuplicateRatio { column: "species", ... }

Stage 3: Applying transforms...
  Applied: Shuffle (seed=42)
  Result: 150 samples

Stage 4: Train/test split (80/20)...
  Training samples: 120
  Test samples: 30

Stage 6: Running inference...
  Test samples: 30
  Throughput: 17M samples/sec

Stage 8: Drift detection...
  Overall drift detected: false
  All columns: OK (no significant drift)

============================================================
Pipeline Summary:
  Data source: alimentar::datasets::iris (embedded)
  Total samples: 150
  Train/Test split: 120/30
  Drift detected: false
============================================================
```

## Alimentar Features Used

| Feature | Description |
|---------|-------------|
| `datasets::iris` | Built-in Iris dataset |
| `QualityChecker` | Data quality validation |
| `Shuffle` | Row randomization transform |
| `DriftDetector` | Distribution drift detection |
| `DataLoader` | Batched iteration |
| `ArrowDataset` | Zero-copy Arrow storage |

## Dependencies

Add to `Cargo.toml`:

```toml
[dependencies]
alimentar = { version = "0.1", path = "../alimentar", features = ["local", "shuffle"] }

[features]
alimentar-data = ["dep:alimentar"]
```

## Next Steps

1. **Add more transforms**: Normalize features, filter outliers
2. **Use real model**: Replace classifier with aprender-trained model
3. **Production deployment**: Export to Lambda with realizar
4. **Continuous monitoring**: Set up drift alerts
