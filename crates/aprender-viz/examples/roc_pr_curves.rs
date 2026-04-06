#![allow(clippy::expect_used, clippy::unwrap_used)]
//! ROC and PR Curves Example
//!
//! Demonstrates creating ROC (Receiver Operating Characteristic) and
//! Precision-Recall curves for binary classifier evaluation.
//!
//! Run with: `cargo run --example roc_pr_curves`

use trueno_viz::output::PngEncoder;
use trueno_viz::plots::{compute_pr, compute_roc, PrCurve, RocCurve};
use trueno_viz::prelude::*;

fn main() {
    println!("ROC and PR Curves Example");
    println!("=========================\n");

    // Step 1: Simulate binary classification scores
    println!("Step 1: Simulating binary classifier predictions...");
    let (y_true, y_scores) = simulate_binary_classifier();

    let n_positive = y_true.iter().fold(0usize, |acc, &y| acc + usize::from(y == 1));
    let n_negative = y_true.len() - n_positive;

    println!("  Total samples: {}", y_true.len());
    println!(
        "  Positive: {} ({:.1}%)",
        n_positive,
        n_positive as f32 / y_true.len() as f32 * 100.0
    );
    println!(
        "  Negative: {} ({:.1}%)",
        n_negative,
        n_negative as f32 / y_true.len() as f32 * 100.0
    );

    // Step 2: Compute ROC curve
    println!("\nStep 2: Computing ROC curve...");
    let roc_data = compute_roc(&y_true, &y_scores).expect("Failed to compute ROC");

    println!("  ROC points: {}", roc_data.points.len());
    println!("  AUC (Area Under Curve): {:.4}", roc_data.auc);

    // Interpret AUC
    let auc_interpretation = match roc_data.auc {
        auc if auc >= 0.9 => "Excellent",
        auc if auc >= 0.8 => "Good",
        auc if auc >= 0.7 => "Fair",
        auc if auc >= 0.6 => "Poor",
        _ => "Random",
    };
    println!("  Interpretation: {auc_interpretation}");

    // Step 3: Compute PR curve
    println!("\nStep 3: Computing Precision-Recall curve...");
    let pr_data = compute_pr(&y_true, &y_scores).expect("Failed to compute PR");

    println!("  PR points: {}", pr_data.points.len());
    println!("  Average Precision: {:.4}", pr_data.average_precision);

    // Step 4: Create ROC visualization
    println!("\nStep 4: Creating ROC curve visualization...");
    let roc_curve = RocCurve::new()
        .data(roc_data)
        .color(Rgba::BLUE)
        .diagonal(true) // Show random classifier reference line
        .dimensions(500, 500)
        .build()
        .expect("Failed to build ROC curve");

    let fb_roc = roc_curve.to_framebuffer().expect("Failed to render ROC");
    PngEncoder::write_to_file(&fb_roc, "roc_curve.png").expect("Failed to write PNG");
    println!("  Saved: roc_curve.png");
    println!("  AUC: {:.4}", roc_curve.auc());

    // Step 5: Create PR visualization
    println!("\nStep 5: Creating Precision-Recall curve visualization...");
    let pr_curve = PrCurve::new()
        .data(pr_data)
        .color(Rgba::rgb(0, 128, 0))
        .baseline(true) // Show no-skill reference line
        .dimensions(500, 500)
        .build()
        .expect("Failed to build PR curve");

    let fb_pr = pr_curve.to_framebuffer().expect("Failed to render PR");
    PngEncoder::write_to_file(&fb_pr, "pr_curve.png").expect("Failed to write PNG");
    println!("  Saved: pr_curve.png");
    println!("  Average Precision: {:.4}", pr_curve.average_precision());

    // Step 6: Find optimal threshold
    println!("\nStep 6: Finding optimal operating point...");

    // For ROC: Find point closest to (0, 1) - top-left corner
    let roc_data_ref = compute_roc(&y_true, &y_scores).expect("operation should succeed");
    let optimal_roc = roc_data_ref
        .points
        .iter()
        .min_by(|a, b| {
            let dist_a = a.x.powi(2) + (1.0 - a.y).powi(2);
            let dist_b = b.x.powi(2) + (1.0 - b.y).powi(2);
            dist_a.partial_cmp(&dist_b).expect("operation should succeed")
        })
        .expect("operation should succeed");

    println!("  Optimal ROC point (closest to perfect):");
    println!("    Threshold: {:.3}", optimal_roc.threshold);
    println!("    TPR (Sensitivity): {:.3}", optimal_roc.y);
    println!("    FPR (1 - Specificity): {:.3}", optimal_roc.x);

    // Summary
    println!("\n--- Summary ---");
    println!("ROC AUC:             {:.4} ({})", roc_curve.auc(), auc_interpretation);
    println!("Average Precision:   {:.4}", pr_curve.average_precision());
    println!("Recommended threshold: {:.3}", optimal_roc.threshold);

    println!("\nROC and PR curves successfully generated!");
}

/// Simulate a binary classifier with moderately good performance.
///
/// Returns (`ground_truth_labels`, `prediction_scores`).
fn simulate_binary_classifier() -> (Vec<u8>, Vec<f32>) {
    let mut y_true = Vec::new();
    let mut y_scores = Vec::new();

    // Generate 200 samples with imbalanced classes (30% positive)
    let n_samples = 200;
    let positive_rate = 0.3;

    for i in 0..n_samples {
        // Deterministic "random" assignment
        let is_positive = ((i * 7919 + 104_729) % 100) < (positive_rate * 100.0) as usize;

        if is_positive {
            y_true.push(1);
            // Positive samples: scores skewed higher (mean ~0.7)
            let noise = ((i * 1_103_515_245 + 12345) % 100) as f32 / 100.0;
            let score = 0.5 + noise * 0.45;
            y_scores.push(score.min(0.99));
        } else {
            y_true.push(0);
            // Negative samples: scores skewed lower (mean ~0.3)
            let noise = ((i.wrapping_mul(6_364_136_223) + 1) % 100) as f32 / 100.0;
            let score = 0.1 + noise * 0.5;
            y_scores.push(score.max(0.01));
        }
    }

    (y_true, y_scores)
}
