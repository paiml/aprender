//! Isolation Forest anomaly detection example
//!
//! Demonstrates using Isolation Forest to detect outliers and anomalies.
//!
//! Run with:
//! ```bash
//! cargo run --example isolation_forest_anomaly
//! ```

use aprender::prelude::*;

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Isolation Forest Anomaly Detection Example ===\n");

    // Example 1: Basic anomaly detection
    println!("--- Example 1: Basic Anomaly Detection ---");

    // Create dataset: 8 normal points + 2 clear outliers
    let data = Matrix::from_vec(
        10,
        2,
        vec![
            // Normal points clustered around (2, 2)
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2,
            // Two obvious outliers
            10.0, 10.0, // Far outlier
            -10.0, -10.0, // Far outlier
        ],
    )?;

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

    // Train Isolation Forest with 20% expected contamination
    let mut iforest = IsolationForest::new()
        .with_n_estimators(100)
        .with_contamination(0.2)
        .with_random_state(42);

    iforest.fit(&data)?;

    // Predict anomalies (1 = normal, -1 = anomaly)
    let predictions = iforest.predict(&data);
    println!("\nPredictions (1=normal, -1=anomaly):");
    for (i, &pred) in predictions.iter().enumerate() {
        let label = if pred == 1 { "NORMAL" } else { "ANOMALY" };
        println!("  Point {i}: {pred} ({label})");
    }

    // Count anomalies
    let n_anomalies = predictions.iter().filter(|&&p| p == -1).count();
    println!("\nDetected {} anomalies out of {} points", n_anomalies, 10);

    // Example 2: Anomaly scores
    println!("\n--- Example 2: Anomaly Scores ---");

    let scores = iforest.score_samples(&data);
    println!("\nAnomaly scores (lower = more anomalous):");
    for (i, &score) in scores.iter().enumerate() {
        println!("  Point {i}: {score:.4}");
    }

    // Find most anomalous point
    let (most_anomalous_idx, &most_anomalous_score) = scores
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Example data should be valid"))
        .expect("Example data should be valid");
    println!("\nMost anomalous point: {most_anomalous_idx} with score {most_anomalous_score:.4}");

    // Example 3: Effect of contamination parameter
    println!("\n--- Example 3: Contamination Parameter ---");

    println!("\nLow contamination (10%):");
    let mut iforest_low = IsolationForest::new()
        .with_contamination(0.1)
        .with_random_state(42);
    iforest_low.fit(&data)?;
    let pred_low = iforest_low.predict(&data);
    let anomalies_low = pred_low.iter().filter(|&&p| p == -1).count();
    println!("  Detected {anomalies_low} anomalies");

    println!("\nMedium contamination (20%):");
    let mut iforest_med = IsolationForest::new()
        .with_contamination(0.2)
        .with_random_state(42);
    iforest_med.fit(&data)?;
    let pred_med = iforest_med.predict(&data);
    let anomalies_med = pred_med.iter().filter(|&&p| p == -1).count();
    println!("  Detected {anomalies_med} anomalies");

    println!("\nHigh contamination (30%):");
    let mut iforest_high = IsolationForest::new()
        .with_contamination(0.3)
        .with_random_state(42);
    iforest_high.fit(&data)?;
    let pred_high = iforest_high.predict(&data);
    let anomalies_high = pred_high.iter().filter(|&&p| p == -1).count();
    println!("  Detected {anomalies_high} anomalies");

    // Example 4: Effect of number of trees
    println!("\n--- Example 4: Number of Trees (Ensemble Size) ---");

    println!("\nFew trees (10):");
    let mut iforest_few = IsolationForest::new()
        .with_n_estimators(10)
        .with_random_state(42);
    iforest_few.fit(&data)?;
    let pred_few = iforest_few.predict(&data);
    println!("  Predictions: {pred_few:?}");

    println!("\nMany trees (100):");
    let mut iforest_many = IsolationForest::new()
        .with_n_estimators(100)
        .with_random_state(42);
    iforest_many.fit(&data)?;
    let pred_many = iforest_many.predict(&data);
    println!("  Predictions: {pred_many:?}");

    println!("\nNote: More trees typically provide more stable results");

    // Example 5: Credit card fraud detection scenario
    println!("\n--- Example 5: Credit Card Fraud Detection Scenario ---");

    // Simulate credit card transactions
    // Features: [amount, time_of_day]
    let transactions = Matrix::from_vec(
        12,
        2,
        vec![
            // Normal transactions (small amounts during daytime)
            50.0, 10.0, // $50 at 10am
            75.0, 12.0, // $75 at 12pm
            30.0, 14.0, // $30 at 2pm
            60.0, 9.0, // $60 at 9am
            45.0, 11.0, // $45 at 11am
            80.0, 13.0, // $80 at 1pm
            55.0, 15.0, // $55 at 3pm
            70.0, 16.0, // $70 at 4pm
            // Suspicious transactions (large amounts at unusual times)
            5000.0, 3.0, // $5000 at 3am (FRAUD)
            4500.0, 4.0, // $4500 at 4am (FRAUD)
            3000.0, 2.0, // $3000 at 2am (FRAUD)
            100.0, 10.0, // $100 at 10am (maybe ok)
        ],
    )?;

    println!("Analyzing {} credit card transactions...", 12);
    println!("\nTransactions (amount, hour):");
    for i in 0..12 {
        println!(
            "  Transaction {}: ${:.0} at {}:00",
            i,
            transactions.get(i, 0),
            transactions.get(i, 1) as u32
        );
    }

    // Train fraud detector with 20% expected fraud rate
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

    // Example 6: Reproducibility
    println!("\n--- Example 6: Reproducibility ---");

    let mut iforest1 = IsolationForest::new()
        .with_n_estimators(50)
        .with_random_state(123);
    iforest1.fit(&data)?;
    let pred1 = iforest1.predict(&data);

    let mut iforest2 = IsolationForest::new()
        .with_n_estimators(50)
        .with_random_state(123);
    iforest2.fit(&data)?;
    let pred2 = iforest2.predict(&data);

    println!("Results are reproducible: {}", pred1 == pred2);
    println!("(Same random seed produces identical results)");

    // Example 7: Understanding isolation path length
    println!("\n--- Example 7: Isolation Path Length Concept ---");
    println!("\nKey insight: Anomalies are easier to isolate");
    println!("- Normal points: Require more splits to isolate (longer paths)");
    println!("- Anomalies: Can be isolated quickly (shorter paths)");
    println!("\nAnomaly score based on average path length across trees:");
    println!("- Shorter average path → Higher anomaly score");
    println!("- Longer average path → Lower anomaly score (more normal)");

    // Example 8: Max samples parameter
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

    println!("\n=== Example Complete ===");
    println!("\nKey takeaways:");
    println!("✓ Isolation Forest detects anomalies without labeled data");
    println!("✓ Contamination parameter controls sensitivity");
    println!("✓ More trees = more stable predictions");
    println!("✓ Works well for fraud detection, outlier detection, quality control");
    println!("✓ Fast: O(n log m) training time");

    Ok(())
}
