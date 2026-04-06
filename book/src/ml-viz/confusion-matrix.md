# Confusion Matrices

Confusion matrices visualize the performance of classification models
by showing predicted vs actual class counts.

## Basic Confusion Matrix

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::ConfusionMatrix;

// 2x2 binary classification
let data = vec![
    50.0, 10.0,  // TN, FP
    5.0,  35.0,  // FN, TP
];

let cm = ConfusionMatrix::new(&data, 2).build();
```

**Test Reference**: `src/plots/heatmap.rs::test_confusion_matrix`

## With Class Names

```rust
use trueno_viz::plots::ConfusionMatrix;

let cm = ConfusionMatrix::new(&data, 2)
    .class_names(&["Negative", "Positive"])
    .build();
```

## Multiclass Confusion Matrix

```rust
use trueno_viz::plots::ConfusionMatrix;

// 3x3 multiclass
let data = vec![
    45.0, 3.0,  2.0,   // Class 0
    4.0,  38.0, 8.0,   // Class 1
    1.0,  7.0,  42.0,  // Class 2
];

let cm = ConfusionMatrix::new(&data, 3)
    .class_names(&["Cat", "Dog", "Bird"])
    .build();
```

## Normalization

### By Row (True Labels)

```rust
use trueno_viz::plots::{ConfusionMatrix, CmNormalize};

let cm = ConfusionMatrix::new(&data, 3)
    .normalize(CmNormalize::True)  // Row sums = 1
    .build();
```

### By Column (Predicted)

```rust
use trueno_viz::plots::{ConfusionMatrix, CmNormalize};

let cm = ConfusionMatrix::new(&data, 3)
    .normalize(CmNormalize::Pred)  // Column sums = 1
    .build();
```

### Overall

```rust
use trueno_viz::plots::{ConfusionMatrix, CmNormalize};

let cm = ConfusionMatrix::new(&data, 3)
    .normalize(CmNormalize::All)  // All cells sum to 1
    .build();
```

## Customization

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::{ConfusionMatrix, HeatmapPalette};

let cm = ConfusionMatrix::new(&data, 2)
    .palette(HeatmapPalette::Blues)
    .annotate(true)
    .annotation_format("{:.0}")  // Integer format
    .cell_border(true)
    .title("Model Performance")
    .xlabel("Predicted Label")
    .ylabel("True Label")
    .build();
```

## Metrics Display

Show metrics alongside matrix:

```rust
use trueno_viz::plots::ConfusionMatrix;

let cm = ConfusionMatrix::new(&data, 2)
    .show_metrics(true)
    .build();

// Prints: Accuracy, Precision, Recall, F1
let metrics = cm.metrics();
println!("Accuracy: {:.4}", metrics.accuracy);
println!("Precision: {:.4}", metrics.precision);
println!("Recall: {:.4}", metrics.recall);
println!("F1: {:.4}", metrics.f1);
```

## Complete Example

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::{ConfusionMatrix, CmNormalize, HeatmapPalette};

fn main() -> Result<()> {
    // Iris-like classification results
    let data = vec![
        48.0, 2.0,  0.0,   // Setosa
        3.0,  44.0, 3.0,   // Versicolor
        0.0,  5.0,  45.0,  // Virginica
    ];

    let cm = ConfusionMatrix::new(&data, 3)
        .class_names(&["Setosa", "Versicolor", "Virginica"])
        .normalize(CmNormalize::True)
        .palette(HeatmapPalette::Blues)
        .annotate(true)
        .annotation_format("{:.2}")
        .title("Iris Classification Results")
        .xlabel("Predicted Species")
        .ylabel("True Species")
        .build();

    cm.render_to_file("iris_confusion_matrix.png")?;

    // Print metrics
    let metrics = cm.metrics();
    println!("Overall Accuracy: {:.2}%", metrics.accuracy * 100.0);

    Ok(())
}
```

## Next Chapter

Continue to [PNG Encoding](../output/png.md) for output format details.
