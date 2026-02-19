#![allow(clippy::disallowed_methods)]
//! Local Outlier Factor (LOF) anomaly detection example
//!
//! Demonstrates LOF for detecting outliers in varying density regions.
//!
//! Run with:
//! ```bash
//! cargo run --example lof_anomaly
//! ```

use aprender::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Local Outlier Factor (LOF) Anomaly Detection ===\n");

    let data = create_sample_data()?;
    let lof = example_basic_detection(&data)?;
    example_lof_scores(&lof, &data);
    example_varying_density()?;
    example_n_neighbors_effect()?;
    example_contamination_parameter(&data)?;
    example_lof_vs_iforest()?;
    example_negative_outlier_factor(&lof);
    example_reproducibility(&data)?;

    print_takeaways();
    Ok(())
}

fn create_sample_data() -> Result<Matrix<f32>, Box<dyn std::error::Error>> {
    Matrix::from_vec(
        10,
        2,
        vec![
            // Normal points clustered around (2, 2)
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2,
            // Two obvious outliers
            10.0, 10.0, -10.0, -10.0,
        ],
    )
    .map_err(Into::into)
}

fn example_basic_detection(
    data: &Matrix<f32>,
) -> Result<LocalOutlierFactor, Box<dyn std::error::Error>> {
    println!("--- Example 1: Basic Anomaly Detection ---");
    println!(
        "Dataset: {} samples, {} features\n",
        data.shape().0,
        data.shape().1
    );

    let mut lof = LocalOutlierFactor::new()
        .with_n_neighbors(5)
        .with_contamination(0.2);
    lof.fit(data)?;

    let predictions = lof.predict(data);
    println!("Predictions (1=normal, -1=anomaly):");
    for (i, &pred) in predictions.iter().enumerate() {
        let label = if pred == 1 { "NORMAL" } else { "ANOMALY" };
        println!("  Point {i}: {pred} ({label})");
    }

    let n_anomalies = predictions.iter().filter(|&&p| p == -1).count();
    println!("\nDetected {} anomalies out of {} points", n_anomalies, 10);
    Ok(lof)
}

fn example_lof_scores(lof: &LocalOutlierFactor, data: &Matrix<f32>) {
    println!("\n--- Example 2: LOF Scores Interpretation ---");

    let scores = lof.score_samples(data);
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
}

fn example_varying_density() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 3: Varying Density Clusters ---");
    println!("LOF excels at finding outliers in regions with different densities\n");

    let mixed_density = Matrix::from_vec(
        11,
        2,
        vec![
            // Dense cluster: 5 points very close together
            0.0, 0.0, 0.05, 0.05, -0.05, -0.05, 0.0, 0.05, -0.05, 0.0,
            // Sparse cluster: 4 points far apart
            10.0, 10.0, 12.0, 12.0, 11.0, 9.0, 13.0, 11.0, // Outliers
            5.0, 5.0, 15.0, 8.0,
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
    Ok(())
}

fn example_n_neighbors_effect() -> Result<(), Box<dyn std::error::Error>> {
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
    Ok(())
}

fn example_contamination_parameter(data: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 5: Contamination Parameter ---");

    for (label, contamination) in [("Low (10%)", 0.1), ("High (30%)", 0.3)] {
        println!("\n{label} contamination:");
        let mut lof = LocalOutlierFactor::new()
            .with_n_neighbors(5)
            .with_contamination(contamination);
        lof.fit(data)?;
        let preds = lof.predict(data);
        let n_anomalies = preds.iter().filter(|&&p| p == -1).count();
        println!("  Detected {n_anomalies} anomalies");
    }
    Ok(())
}

fn example_lof_vs_iforest() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 6: LOF vs Isolation Forest ---");
    println!("Testing on varying density dataset:\n");

    let mixed_density = Matrix::from_vec(
        11,
        2,
        vec![
            0.0, 0.0, 0.05, 0.05, -0.05, -0.05, 0.0, 0.05, -0.05, 0.0, 10.0, 10.0, 12.0, 12.0,
            11.0, 9.0, 13.0, 11.0, 5.0, 5.0, 15.0, 8.0,
        ],
    )?;

    let mut lof = LocalOutlierFactor::new()
        .with_n_neighbors(4)
        .with_contamination(0.2);
    lof.fit(&mixed_density)?;
    let lof_preds = lof.predict(&mixed_density);
    let lof_anomalies = lof_preds.iter().filter(|&&p| p == -1).count();

    println!("LOF (local density-based):");
    println!("  Handles varying density: ✓");
    println!("  Detected {lof_anomalies} outliers in mixed-density data");

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
    Ok(())
}

fn example_negative_outlier_factor(lof: &LocalOutlierFactor) {
    println!("\n--- Example 7: Negative Outlier Factor ---");

    let nof = lof.negative_outlier_factor();
    println!("Negative outlier factor (sklearn compatibility):");
    println!("  NOF = -LOF score");
    println!("  More negative = more anomalous\n");

    for i in 0..3 {
        println!("  Point {}: NOF={:.3}", i, nof[i]);
    }
}

fn example_reproducibility(data: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 8: Reproducibility ---");

    let mut lof1 = LocalOutlierFactor::new().with_n_neighbors(5);
    lof1.fit(data)?;
    let pred1 = lof1.predict(data);

    let mut lof2 = LocalOutlierFactor::new().with_n_neighbors(5);
    lof2.fit(data)?;
    let pred2 = lof2.predict(data);

    println!("Results are reproducible: {}", pred1 == pred2);
    println!("(Deterministic k-NN produces consistent results)");
    Ok(())
}

fn print_takeaways() {
    println!("\n=== Example Complete ===");
    println!("\nKey takeaways:");
    println!("✓ LOF detects local outliers based on density deviation");
    println!("✓ Excels at varying density clusters");
    println!("✓ LOF ≈ 1 = normal, LOF >> 1 = outlier");
    println!("✓ n_neighbors controls local vs global context");
    println!("✓ contamination sets expected anomaly proportion");
    println!("✓ Complements Isolation Forest for comprehensive detection");
}
