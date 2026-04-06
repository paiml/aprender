//! Example: Explainability Callback
//!
//! Demonstrates using ExplainabilityCallback with aprender's interpret module
//! to track feature importance during training.

use aprender::primitives::Vector;
use entrenar::train::{ExplainMethod, ExplainabilityCallback};

/// Simple linear model: y = w0*x0 + w1*x1 + w2*x2
fn predict(weights: &[f32], x: &Vector<f32>) -> f32 {
    weights.iter().zip(x.as_slice()).map(|(w, xi)| w * xi).sum()
}

fn main() {
    println!("=== Explainability Callback Example ===\n");

    // Model weights (feature 0 has highest weight)
    let weights = vec![3.0, 1.0, 0.5];

    println!("Model: y = 3.0*x0 + 1.0*x1 + 0.5*x2");
    println!("Expected: feature_0 should be most important\n");

    // Create explainability callback
    let mut explainer = ExplainabilityCallback::new(ExplainMethod::PermutationImportance)
        .with_top_k(3)
        .with_feature_names(vec!["feature_0".into(), "feature_1".into(), "feature_2".into()]);

    println!("Configuration:");
    println!("  Method: {:?}", explainer.method());
    println!("  Top-K: {}", explainer.top_k());
    println!("  Eval samples: {}\n", explainer.eval_samples());

    // Validation data
    let val_x = vec![
        Vector::from_slice(&[1.0, 2.0, 3.0]),
        Vector::from_slice(&[2.0, 1.0, 4.0]),
        Vector::from_slice(&[3.0, 3.0, 1.0]),
        Vector::from_slice(&[4.0, 2.0, 2.0]),
        Vector::from_slice(&[1.5, 4.0, 3.0]),
    ];

    // Ground truth (using the same model)
    let val_y: Vec<f32> = val_x.iter().map(|x| predict(&weights, x)).collect();

    println!("Validation data:");
    for (i, (x, y)) in val_x.iter().zip(&val_y).enumerate() {
        println!("  Sample {}: x={:?}, y={:.1}", i, x.as_slice(), y);
    }
    println!();

    // Simulate training epochs
    println!("Simulating 3 training epochs...\n");

    for epoch in 0..3 {
        // Compute permutation importance
        let importances =
            explainer.compute_permutation_importance(|x| predict(&weights, x), &val_x, &val_y);

        // Record for this epoch
        explainer.record_importances(epoch, importances);

        println!("Epoch {epoch}:");
        let result = explainer.results().last().expect("operation should succeed");
        for (idx, score) in &result.importances {
            let name = explainer.feature_names().map_or("unknown", |n| n[*idx].as_str());
            println!("  {name}: {score:.6}");
        }
        println!();
    }

    // Consistent top features across epochs
    println!("Consistently important features:");
    let consistent = explainer.consistent_top_features();
    for (idx, avg_score) in &consistent {
        let name = explainer.feature_names().map_or("unknown", |n| n[*idx].as_str());
        println!("  {name}: avg_score={avg_score:.6}");
    }

    println!("\n--- Integrated Gradients Demo ---\n");

    // Demo integrated gradients
    let ig_explainer = ExplainabilityCallback::new(ExplainMethod::IntegratedGradients);

    let sample = Vector::from_slice(&[2.0, 1.0, 3.0]);
    let baseline = Vector::from_slice(&[0.0, 0.0, 0.0]);

    let attributions =
        ig_explainer.compute_integrated_gradients(|x| predict(&weights, x), &sample, &baseline);

    println!("Sample: {:?}", sample.as_slice());
    println!("Baseline: {:?}", baseline.as_slice());
    println!("Prediction: {:.2}", predict(&weights, &sample));
    println!("\nIntegrated Gradients attributions:");
    for (idx, (_, attr)) in attributions.iter().enumerate() {
        // Expected: attr ≈ weight * (sample - baseline) = weight * sample
        println!("  feature_{}: {:.4} (expected: {:.4})", idx, attr, weights[idx] * sample[idx]);
    }

    println!("\n--- Saliency Map Demo ---\n");

    // Demo saliency maps
    let saliency_explainer = ExplainabilityCallback::new(ExplainMethod::Saliency);

    let saliency = saliency_explainer.compute_saliency(|x| predict(&weights, x), &sample);

    println!("Saliency (gradients) for sample {:?}:", sample.as_slice());
    for (idx, (_, grad)) in saliency.iter().enumerate() {
        // For linear model, gradient = weight
        println!("  feature_{}: {:.4} (expected: {:.4})", idx, grad, weights[idx]);
    }

    println!("\n=== Example Complete ===");
}
