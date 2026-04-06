#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Confusion Matrix ML Example
//!
//! Demonstrates creating a confusion matrix visualization for
//! evaluating classification model performance.
//!
//! Run with: `cargo run --example confusion_matrix_ml`

use trueno_viz::output::PngEncoder;
use trueno_viz::plots::{ConfusionMatrix, Normalization};
use trueno_viz::prelude::WithDimensions;

fn main() {
    println!("Confusion Matrix ML Example");
    println!("============================\n");

    // Step 1: Simulate classification results
    println!("Step 1: Simulating classification results...");
    let (y_true, y_pred, class_names) = simulate_classification();

    println!("  Total samples: {}", y_true.len());
    println!("  Classes: {class_names:?}");

    // Step 2: Build confusion matrix from predictions
    println!("\nStep 2: Building confusion matrix...");
    let cm = ConfusionMatrix::new()
        .from_predictions(&y_true, &y_pred, class_names.len())
        .labels(&class_names)
        .normalize(Normalization::None) // Show raw counts
        .dimensions(500, 500)
        .margin(60)
        .build()
        .expect("Failed to build confusion matrix");

    println!("  Classes: {}", cm.num_classes());
    println!("  Total predictions: {}", cm.total());

    // Step 3: Compute and display metrics
    println!("\nStep 3: Computing classification metrics...");
    let metrics = cm.metrics();

    println!("\n--- Per-Class Metrics ---");
    println!("{:>12} {:>10} {:>10} {:>10}", "Class", "Precision", "Recall", "F1-Score");
    println!("{}", "-".repeat(45));

    let f1_scores = metrics.f1_scores();
    for (i, name) in class_names.iter().enumerate() {
        println!(
            "{:>12} {:>10.3} {:>10.3} {:>10.3}",
            name, metrics.precision[i], metrics.recall[i], f1_scores[i]
        );
    }

    println!("\n--- Overall Metrics ---");
    println!("Accuracy:     {:.3}", metrics.accuracy);
    println!("Macro F1:     {:.3}", metrics.macro_f1());

    // Step 4: Render visualizations
    println!("\nStep 4: Rendering visualizations...");

    // Raw counts
    let fb = cm.to_framebuffer().expect("Failed to render");
    PngEncoder::write_to_file(&fb, "confusion_matrix_counts.png").expect("Failed to write PNG");
    println!("  Saved: confusion_matrix_counts.png");

    // Normalized by row (shows recall per class)
    let cm_norm = ConfusionMatrix::new()
        .from_predictions(&y_true, &y_pred, class_names.len())
        .normalize(Normalization::Row)
        .dimensions(500, 500)
        .margin(60)
        .build()
        .expect("Failed to build normalized confusion matrix");

    let fb_norm = cm_norm.to_framebuffer().expect("Failed to render");
    PngEncoder::write_to_file(&fb_norm, "confusion_matrix_normalized.png")
        .expect("Failed to write PNG");
    println!("  Saved: confusion_matrix_normalized.png");

    // Display the raw confusion matrix
    println!("\n--- Confusion Matrix (Raw Counts) ---");
    println!("Rows = Actual, Columns = Predicted\n");

    // Note: We reconstruct the matrix for display since we used from_predictions
    let n = class_names.len();
    let mut matrix = vec![vec![0u32; n]; n];
    for (&true_class, &pred_class) in y_true.iter().zip(y_pred.iter()) {
        matrix[true_class][pred_class] += 1;
    }

    print!("{:>12}", "");
    for name in &class_names {
        print!("{name:>10}");
    }
    println!();

    for (i, row) in matrix.iter().enumerate() {
        print!("{:>12}", class_names[i]);
        for &count in row {
            print!("{count:>10}");
        }
        println!();
    }

    println!("\nConfusion matrix successfully generated!");
}

/// Simulate a 3-class classification scenario.
///
/// Returns (`true_labels`, `predicted_labels`, `class_names`).
fn simulate_classification() -> (Vec<usize>, Vec<usize>, Vec<&'static str>) {
    let class_names = vec!["Setosa", "Versicolor", "Virginica"];

    // Simulated predictions with some errors
    // Class 0 (Setosa): Well classified
    // Class 1 (Versicolor): Some confusion with Virginica
    // Class 2 (Virginica): Some confusion with Versicolor

    let mut y_true = Vec::new();
    let mut y_pred = Vec::new();

    // Setosa samples (mostly correct)
    for _ in 0..45 {
        y_true.push(0);
        y_pred.push(0);
    } // TP
    for _ in 0..3 {
        y_true.push(0);
        y_pred.push(1);
    } // FN (predicted as Versicolor)
    for _ in 0..2 {
        y_true.push(0);
        y_pred.push(2);
    } // FN (predicted as Virginica)

    // Versicolor samples (some confusion)
    for _ in 0..2 {
        y_true.push(1);
        y_pred.push(0);
    } // FN (predicted as Setosa)
    for _ in 0..38 {
        y_true.push(1);
        y_pred.push(1);
    } // TP
    for _ in 0..10 {
        y_true.push(1);
        y_pred.push(2);
    } // FN (predicted as Virginica)

    // Virginica samples (some confusion)
    for _ in 0..1 {
        y_true.push(2);
        y_pred.push(0);
    } // FN (predicted as Setosa)
    for _ in 0..8 {
        y_true.push(2);
        y_pred.push(1);
    } // FN (predicted as Versicolor)
    for _ in 0..41 {
        y_true.push(2);
        y_pred.push(2);
    } // TP

    (y_true, y_pred, class_names)
}
