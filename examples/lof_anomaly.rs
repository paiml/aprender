//! Local Outlier Factor (LOF) anomaly detection example
//!
//! Demonstrates LOF for detecting outliers in varying density regions.
//!
//! Run with:
//! ```bash
//! cargo run --example lof_anomaly
//! ```

use aprender::prelude::*;

#[allow(clippy::needless_range_loop)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Local Outlier Factor (LOF) Anomaly Detection ===\n");

    // Example 1: Basic anomaly detection
    println!("--- Example 1: Basic Anomaly Detection ---");

    let data = Matrix::from_vec(
        10,
        2,
        vec![
            // Normal points clustered around (2, 2)
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2,
            // Two obvious outliers
            10.0, 10.0, -10.0, -10.0,
        ],
    )?;

    println!(
        "Dataset: {} samples, {} features\n",
        data.shape().0,
        data.shape().1
    );

    let mut lof = LocalOutlierFactor::new()
        .with_n_neighbors(5)
        .with_contamination(0.2);

    lof.fit(&data)?;

    let predictions = lof.predict(&data);
    println!("Predictions (1=normal, -1=anomaly):");
    for (i, &pred) in predictions.iter().enumerate() {
        let label = if pred == 1 { "NORMAL" } else { "ANOMALY" };
        println!("  Point {i}: {pred} ({label})");
    }

    let n_anomalies = predictions.iter().filter(|&&p| p == -1).count();
    println!("\nDetected {} anomalies out of {} points", n_anomalies, 10);

    // Example 2: LOF scores interpretation
    println!("\n--- Example 2: LOF Scores Interpretation ---");

    let scores = lof.score_samples(&data);
    println!("\nLOF scores (higher = more anomalous):");
    println!("  LOF ≈ 1: Similar density to neighbors (normal)");
    println!("  LOF >> 1: Lower density than neighbors (outlier)\n");

    for (i, &score) in scores.iter().enumerate() {
        let interpretation = if score < 1.2 {
            "normal"
        } else if score < 2.0 {
            "borderline"
        } else {
            "clear outlier"
        };
        println!("  Point {i}: {score:.3} ({interpretation})");
    }

    // Example 3: Varying density clusters (LOF's key advantage!)
    println!("\n--- Example 3: Varying Density Clusters ---");
    println!("LOF excels at finding outliers in regions with different densities\n");

    let mixed_density = Matrix::from_vec(
        11,
        2,
        vec![
            // Dense cluster: 5 points very close together
            0.0, 0.0, 0.05, 0.05, -0.05, -0.05, 0.0, 0.05, -0.05, 0.0,
            // Sparse cluster: 4 points far apart
            10.0, 10.0, 12.0, 12.0, 11.0, 9.0, 13.0, 11.0, // Outlier between clusters
            5.0, 5.0, // Outlier near sparse cluster (but still anomalous)
            15.0, 8.0,
        ],
    )?;

    let mut lof_varied = LocalOutlierFactor::new()
        .with_n_neighbors(4)
        .with_contamination(0.2);
    lof_varied.fit(&mixed_density)?;

    let varied_scores = lof_varied.score_samples(&mixed_density);
    let varied_preds = lof_varied.predict(&mixed_density);

    println!("Dense cluster (points 0-4):");
    for i in 0..5 {
        println!("  Point {}: LOF={:.3}", i, varied_scores[i]);
    }

    println!("\nSparse cluster (points 5-8):");
    for i in 5..9 {
        println!("  Point {}: LOF={:.3}", i, varied_scores[i]);
    }

    println!("\nOutliers:");
    println!("  Point 9 (between clusters): LOF={:.3}", varied_scores[9]);
    println!("  Point 10 (near sparse): LOF={:.3}", varied_scores[10]);

    let n_detected = varied_preds.iter().filter(|&&p| p == -1).count();
    println!("\nLOF correctly identifies {n_detected} local outliers!");

    // Example 4: Effect of n_neighbors
    println!("\n--- Example 4: Effect of n_neighbors Parameter ---");

    let test_data = Matrix::from_vec(
        8,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 10.0, 10.0, -10.0, -10.0,
        ],
    )?;

    println!("\nFewer neighbors (k=3) - more local focus:");
    let mut lof_few = LocalOutlierFactor::new().with_n_neighbors(3);
    lof_few.fit(&test_data)?;
    let scores_few = lof_few.score_samples(&test_data);
    println!(
        "  Outlier scores: {:.3}, {:.3}",
        scores_few[6], scores_few[7]
    );

    println!("\nMore neighbors (k=5) - broader context:");
    let mut lof_many = LocalOutlierFactor::new().with_n_neighbors(5);
    lof_many.fit(&test_data)?;
    let scores_many = lof_many.score_samples(&test_data);
    println!(
        "  Outlier scores: {:.3}, {:.3}",
        scores_many[6], scores_many[7]
    );

    // Example 5: Contamination parameter
    println!("\n--- Example 5: Contamination Parameter ---");

    println!("\nLow contamination (10%):");
    let mut lof_low = LocalOutlierFactor::new()
        .with_n_neighbors(5)
        .with_contamination(0.1);
    lof_low.fit(&data)?;
    let pred_low = lof_low.predict(&data);
    let anomalies_low = pred_low.iter().filter(|&&p| p == -1).count();
    println!("  Detected {anomalies_low} anomalies");

    println!("\nHigh contamination (30%):");
    let mut lof_high = LocalOutlierFactor::new()
        .with_n_neighbors(5)
        .with_contamination(0.3);
    lof_high.fit(&data)?;
    let pred_high = lof_high.predict(&data);
    let anomalies_high = pred_high.iter().filter(|&&p| p == -1).count();
    println!("  Detected {anomalies_high} anomalies");

    // Example 6: LOF vs Isolation Forest comparison
    println!("\n--- Example 6: LOF vs Isolation Forest ---");
    println!("Testing on varying density dataset:\n");

    // LOF results (already computed above)
    println!("LOF (local density-based):");
    println!("  Handles varying density: ✓");
    println!("  Detected {n_detected} outliers in mixed-density data");

    // Isolation Forest for comparison
    let mut iforest = IsolationForest::new()
        .with_n_estimators(100)
        .with_contamination(0.2)
        .with_random_state(42);
    iforest.fit(&mixed_density)?;
    let iforest_preds = iforest.predict(&mixed_density);
    let iforest_anomalies = iforest_preds.iter().filter(|&&p| p == -1).count();

    println!("\nIsolation Forest (global isolation-based):");
    println!("  Handles varying density: partial");
    println!("  Detected {iforest_anomalies} outliers in mixed-density data");

    println!("\nKey difference:");
    println!("  LOF: Compares density locally → better for varying densities");
    println!("  Isolation Forest: Global isolation → better for overall anomalies");

    // Example 7: Negative Outlier Factor (sklearn compatibility)
    println!("\n--- Example 7: Negative Outlier Factor ---");

    let nof = lof.negative_outlier_factor();
    println!("Negative outlier factor (sklearn compatibility):");
    println!("  NOF = -LOF score");
    println!("  More negative = more anomalous\n");

    for i in 0..3 {
        println!("  Point {}: NOF={:.3}", i, nof[i]);
    }

    // Example 8: Reproducibility
    println!("\n--- Example 8: Reproducibility ---");

    let mut lof1 = LocalOutlierFactor::new().with_n_neighbors(5);
    lof1.fit(&data)?;
    let pred1 = lof1.predict(&data);

    let mut lof2 = LocalOutlierFactor::new().with_n_neighbors(5);
    lof2.fit(&data)?;
    let pred2 = lof2.predict(&data);

    println!("Results are reproducible: {}", pred1 == pred2);
    println!("(Deterministic k-NN produces consistent results)");

    println!("\n=== Example Complete ===");
    println!("\nKey takeaways:");
    println!("✓ LOF detects local outliers based on density deviation");
    println!("✓ Excels at varying density clusters");
    println!("✓ LOF ≈ 1 = normal, LOF >> 1 = outlier");
    println!("✓ n_neighbors controls local vs global context");
    println!("✓ contamination sets expected anomaly proportion");
    println!("✓ Complements Isolation Forest for comprehensive detection");

    Ok(())
}
