#![allow(clippy::disallowed_methods)]
//! Isolation Forest anomaly detection example
//!
//! Demonstrates using Isolation Forest to detect outliers and anomalies.
//!
//! Run with:
//! ```bash
//! cargo run --example isolation_forest_anomaly
//! ```

use aprender::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Isolation Forest Anomaly Detection Example ===\n");

    let data = create_sample_data()?;

    example_basic_detection(&data)?;
    example_anomaly_scores(&data)?;
    example_contamination_parameter(&data)?;
    example_number_of_trees(&data)?;
    example_fraud_detection()?;
    example_reproducibility(&data)?;
    example_path_length_concept();
    example_max_samples()?;

    print_takeaways();
    Ok(())
}

/// Create sample dataset: 8 normal points + 2 outliers
fn create_sample_data() -> Result<Matrix<f32>, Box<dyn std::error::Error>> {
    Matrix::from_vec(
        10,
        2,
        vec![
            // Normal points clustered around (2, 2)
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2,
            // Two obvious outliers
            10.0, 10.0, // Far outlier
            -10.0, -10.0, // Far outlier
        ],
    )
    .map_err(Into::into)
}

/// Example 1: Basic anomaly detection
fn example_basic_detection(data: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Example 1: Basic Anomaly Detection ---");
    println!(
        "Dataset: {} samples, {} features",
        data.shape().0,
        data.shape().1
    );
    println!("\nData points:");
    for i in 0..data.shape().0 {
        println!(
            "  Point {}: ({:.1}, {:.1})",
            i,
            data.get(i, 0),
            data.get(i, 1)
        );
    }

    let mut iforest = IsolationForest::new()
        .with_n_estimators(100)
        .with_contamination(0.2)
        .with_random_state(42);
    iforest.fit(data)?;

    let predictions = iforest.predict(data);
    println!("\nPredictions (1=normal, -1=anomaly):");
    for (i, &pred) in predictions.iter().enumerate() {
        let label = if pred == 1 { "NORMAL" } else { "ANOMALY" };
        println!("  Point {i}: {pred} ({label})");
    }

    let n_anomalies = predictions.iter().filter(|&&p| p == -1).count();
    println!("\nDetected {} anomalies out of {} points", n_anomalies, 10);
    Ok(())
}

/// Example 2: Anomaly scores
fn example_anomaly_scores(data: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 2: Anomaly Scores ---");

    let mut iforest = IsolationForest::new()
        .with_n_estimators(100)
        .with_contamination(0.2)
        .with_random_state(42);
    iforest.fit(data)?;

    let scores = iforest.score_samples(data);
    println!("\nAnomaly scores (lower = more anomalous):");
    for (i, &score) in scores.iter().enumerate() {
        println!("  Point {i}: {score:.4}");
    }

    let (most_anomalous_idx, &most_anomalous_score) = scores
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .expect("Data should not be empty");
    println!("\nMost anomalous point: {most_anomalous_idx} with score {most_anomalous_score:.4}");
    Ok(())
}

/// Example 3: Effect of contamination parameter
fn example_contamination_parameter(data: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 3: Contamination Parameter ---");

    for (label, contamination) in [
        ("Low (10%)", 0.1),
        ("Medium (20%)", 0.2),
        ("High (30%)", 0.3),
    ] {
        println!("\n{label} contamination:");
        let mut iforest = IsolationForest::new()
            .with_contamination(contamination)
            .with_random_state(42);
        iforest.fit(data)?;
        let predictions = iforest.predict(data);
        let n_anomalies = predictions.iter().filter(|&&p| p == -1).count();
        println!("  Detected {n_anomalies} anomalies");
    }
    Ok(())
}

/// Example 4: Effect of number of trees
fn example_number_of_trees(data: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 4: Number of Trees (Ensemble Size) ---");

    for (label, n_trees) in [("Few trees (10)", 10), ("Many trees (100)", 100)] {
        println!("\n{label}:");
        let mut iforest = IsolationForest::new()
            .with_n_estimators(n_trees)
            .with_random_state(42);
        iforest.fit(data)?;
        let predictions = iforest.predict(data);
        println!("  Predictions: {predictions:?}");
    }
    println!("\nNote: More trees typically provide more stable results");
    Ok(())
}

/// Example 5: Credit card fraud detection scenario
fn example_fraud_detection() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 5: Credit Card Fraud Detection Scenario ---");

    let transactions = Matrix::from_vec(
        12,
        2,
        vec![
            // Normal transactions (small amounts during daytime)
            50.0, 10.0, 75.0, 12.0, 30.0, 14.0, 60.0, 9.0, 45.0, 11.0, 80.0, 13.0, 55.0, 15.0, 70.0,
            16.0, // Suspicious transactions (large amounts at unusual times)
            5000.0, 3.0, 4500.0, 4.0, 3000.0, 2.0, 100.0, 10.0,
        ],
    )?;

    println!("Analyzing 12 credit card transactions...");
    println!("\nTransactions (amount, hour):");
    for i in 0..12 {
        println!(
            "  Transaction {}: ${:.0} at {}:00",
            i,
            transactions.get(i, 0),
            transactions.get(i, 1) as u32
        );
    }

    let mut fraud_detector = IsolationForest::new()
        .with_contamination(0.25)
        .with_n_estimators(100)
        .with_random_state(42);
    fraud_detector.fit(&transactions)?;

    let fraud_predictions = fraud_detector.predict(&transactions);
    let fraud_scores = fraud_detector.score_samples(&transactions);

    println!("\nFraud detection results:");
    for i in 0..12 {
        let status = if fraud_predictions[i] == -1 {
            "FLAGGED"
        } else {
            "OK"
        };
        println!(
            "  Transaction {}: {} (score: {:.4})",
            i, status, fraud_scores[i]
        );
    }

    let n_flagged = fraud_predictions.iter().filter(|&&p| p == -1).count();
    println!("\nFlagged {n_flagged} transactions for review");
    Ok(())
}

/// Example 6: Reproducibility
fn example_reproducibility(data: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 6: Reproducibility ---");

    let mut iforest1 = IsolationForest::new()
        .with_n_estimators(50)
        .with_random_state(123);
    iforest1.fit(data)?;
    let pred1 = iforest1.predict(data);

    let mut iforest2 = IsolationForest::new()
        .with_n_estimators(50)
        .with_random_state(123);
    iforest2.fit(data)?;
    let pred2 = iforest2.predict(data);

    println!("Results are reproducible: {}", pred1 == pred2);
    println!("(Same random seed produces identical results)");
    Ok(())
}

/// Example 7: Understanding isolation path length
fn example_path_length_concept() {
    println!("\n--- Example 7: Isolation Path Length Concept ---");
    println!("\nKey insight: Anomalies are easier to isolate");
    println!("- Normal points: Require more splits to isolate (longer paths)");
    println!("- Anomalies: Can be isolated quickly (shorter paths)");
    println!("\nAnomaly score based on average path length across trees:");
    println!("- Shorter average path → Higher anomaly score");
    println!("- Longer average path → Lower anomaly score (more normal)");
}

/// Example 8: Max samples parameter
fn example_max_samples() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 8: Max Samples (Subsample Size) ---");

    let large_data = Matrix::from_vec(
        20,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2, 1.9,
            1.9, 2.1, 1.8, 2.0, 2.1, 1.95, 2.05, 2.15, 1.85, 2.05, 1.95, 1.9, 2.0, 2.1, 2.0, 10.0,
            10.0, -10.0, -10.0, 9.0, 9.0, -9.0, -9.0,
        ],
    )?;

    println!("\nUsing all samples for each tree:");
    let mut iforest_all = IsolationForest::new()
        .with_max_samples(20)
        .with_random_state(42);
    iforest_all.fit(&large_data)?;

    println!("\nUsing subset (10 samples) for each tree:");
    let mut iforest_subset = IsolationForest::new()
        .with_max_samples(10)
        .with_random_state(42);
    iforest_subset.fit(&large_data)?;

    println!("Both approaches work; subsampling improves efficiency");
    Ok(())
}

fn print_takeaways() {
    println!("\n=== Example Complete ===");
    println!("\nKey takeaways:");
    println!("✓ Isolation Forest detects anomalies without labeled data");
    println!("✓ Contamination parameter controls sensitivity");
    println!("✓ More trees = more stable predictions");
    println!("✓ Works well for fraud detection, outlier detection, quality control");
    println!("✓ Fast: O(n log m) training time");
}
