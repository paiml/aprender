# Iris Dataset

The classic Fisher's Iris dataset (1936) for classification tasks.

## Overview

- **Samples**: 150 (all embedded)
- **Features**: 4 numeric measurements
- **Classes**: 3 species (setosa, versicolor, virginica)
- **Task**: Multi-class classification

## Loading

```rust
use alimentar::datasets::{iris, CanonicalDataset};

let dataset = iris()?;
assert_eq!(dataset.len(), 150);
assert_eq!(dataset.num_features(), 4);
assert_eq!(dataset.num_classes(), 3);
```

## Schema

| Column | Type | Description |
|--------|------|-------------|
| sepal_length | f64 | Sepal length (cm) |
| sepal_width | f64 | Sepal width (cm) |
| petal_length | f64 | Petal length (cm) |
| petal_width | f64 | Petal width (cm) |
| species | string | "setosa", "versicolor", "virginica" |

## Feature Access

```rust
let dataset = iris()?;

// Get feature names
let names = dataset.feature_names();
// ["sepal_length", "sepal_width", "petal_length", "petal_width"]

// Extract features only (no labels)
let features = dataset.features()?;
assert_eq!(features.schema().fields().len(), 4);
```

## Label Access

```rust
let dataset = iris()?;

// String labels
let labels = dataset.labels();
// ["setosa", "setosa", ..., "virginica"]

// Numeric labels (0, 1, 2)
let numeric = dataset.labels_numeric();
// [0, 0, ..., 2]
```

## Class Distribution

The dataset is perfectly balanced:

| Class | Label | Count |
|-------|-------|-------|
| setosa | 0 | 50 |
| versicolor | 1 | 50 |
| virginica | 2 | 50 |

## Example: Simple Classification

```rust
use alimentar::datasets::{iris, CanonicalDataset};
use alimentar::DataLoader;

let dataset = iris()?;
let features = dataset.features()?;
let labels = dataset.labels_numeric();

// Create batched loader
let loader = DataLoader::new(features)
    .batch_size(32)
    .shuffle(true);

for batch in loader {
    println!("Batch: {} rows", batch.num_rows());
}
```

## Reference

Fisher, R.A. (1936). "The use of multiple measurements in taxonomic problems." *Annals of Eugenics*, 7(2), 179-188.
