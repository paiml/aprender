# ROC Curves

Receiver Operating Characteristic (ROC) curves visualize binary classifier
performance across all classification thresholds.

## Basic ROC Curve

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::RocCurve;

// Model predictions and true labels
let y_true = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0];
let y_scores = vec![0.1, 0.3, 0.8, 0.9, 0.4, 0.7, 0.2, 0.85];

let roc = RocCurve::new(&y_true, &y_scores).build();

// Verify AUC is computed
assert!(roc.auc() >= 0.0 && roc.auc() <= 1.0);
```

**Test Reference**: `src/plots/roc.rs::test_roc_basic`

## AUC Score

```rust
use trueno_viz::plots::RocCurve;

let roc = RocCurve::new(&y_true, &y_scores).build();

let auc = roc.auc();
println!("AUC: {:.4}", auc);

// Perfect classifier: AUC = 1.0
// Random classifier: AUC = 0.5
// Inverted classifier: AUC = 0.0
```

## Customization

### Colors and Style

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::RocCurve;

let roc = RocCurve::new(&y_true, &y_scores)
    .color(Rgba::new(66, 133, 244, 255))
    .line_width(2.0)
    .show_diagonal(true)  // Random classifier reference
    .diagonal_color(Rgba::new(150, 150, 150, 255))
    .build();
```

### Show AUC in Plot

```rust
use trueno_viz::plots::RocCurve;

let roc = RocCurve::new(&y_true, &y_scores)
    .show_auc(true)
    .auc_position(0.6, 0.2)  // x, y position
    .build();
```

## Multiple Models

Compare ROC curves for multiple classifiers:

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::RocCurve;

let roc = RocCurve::new(&[], &[])
    .add_model("Logistic Regression", &y_true, &lr_scores,
               Rgba::new(66, 133, 244, 255))
    .add_model("Random Forest", &y_true, &rf_scores,
               Rgba::new(52, 168, 83, 255))
    .add_model("SVM", &y_true, &svm_scores,
               Rgba::new(234, 67, 53, 255))
    .show_diagonal(true)
    .show_legend(true)
    .build();
```

## Confidence Intervals

With cross-validation or bootstrap:

```rust
use trueno_viz::plots::RocCurve;

let roc = RocCurve::new(&y_true, &y_scores)
    .confidence_interval(true)
    .ci_alpha(0.2)  // 20% transparency for CI band
    .build();
```

## Operating Point

Mark the threshold used in practice:

```rust
use trueno_viz::plots::RocCurve;

let roc = RocCurve::new(&y_true, &y_scores)
    .mark_threshold(0.5)
    .threshold_marker_size(8.0)
    .build();
```

## Labels

```rust
use trueno_viz::plots::RocCurve;

let roc = RocCurve::new(&y_true, &y_scores)
    .title("ROC Curve - Binary Classification")
    .xlabel("False Positive Rate (1 - Specificity)")
    .ylabel("True Positive Rate (Sensitivity)")
    .build();
```

## Complete Example

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::RocCurve;

fn main() -> Result<()> {
    // Simulated classifier outputs
    let y_true = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
    ];

    let model_a = vec![
        0.1, 0.2, 0.3, 0.25, 0.15, 0.8, 0.9, 0.85, 0.7, 0.95,
        0.4, 0.35, 0.2, 0.75, 0.65, 0.88, 0.3, 0.72, 0.28, 0.92,
    ];

    let model_b = vec![
        0.15, 0.25, 0.35, 0.3, 0.2, 0.7, 0.75, 0.72, 0.68, 0.82,
        0.45, 0.4, 0.25, 0.65, 0.6, 0.78, 0.35, 0.62, 0.32, 0.85,
    ];

    let roc = RocCurve::new(&[], &[])
        .add_model("Model A", &y_true, &model_a,
                   Rgba::new(66, 133, 244, 255))
        .add_model("Model B", &y_true, &model_b,
                   Rgba::new(234, 67, 53, 255))
        .show_diagonal(true)
        .show_auc(true)
        .show_legend(true)
        .title("Model Comparison - ROC Curves")
        .build();

    roc.render_to_file("roc_comparison.png")?;

    // Print AUC scores
    for (name, auc) in roc.auc_scores() {
        println!("{}: AUC = {:.4}", name, auc);
    }

    Ok(())
}
```

## Next Chapter

Continue to [PR Curves](./pr.md) for precision-recall analysis.
