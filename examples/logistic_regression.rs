//! Logistic Regression Binary Classification Example
//!
//! Demonstrates binary classification using Logistic Regression on a synthetic dataset.
//! Shows probability predictions and class boundaries.
//!
//! Run with: `cargo run --example logistic_regression`

use aprender::prelude::*;

fn main() {
    println!("=== Logistic Regression Binary Classification ===\n");

    // Create synthetic binary classification dataset
    // Class 0: Low values (< 5)
    // Class 1: High values (> 5)
    let x_train = Matrix::from_vec(
        20,
        2,
        vec![
            // Class 0 samples (low values)
            2.0, 1.5, 1.8, 2.2, 2.5, 1.9, 3.0, 2.8, 2.2, 1.7, 3.5, 2.5, 1.5, 1.2, 2.8, 3.2, 2.0,
            2.5, 3.2, 2.7, // Class 1 samples (high values)
            7.0, 8.5, 6.8, 7.2, 8.0, 6.5, 7.5, 8.0, 6.2, 7.8, 8.5, 7.0, 6.5, 6.8, 7.2, 8.2, 6.0,
            7.5, 8.0, 6.8,
        ],
    )
    .expect("Example data should be valid");

    let y_train = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

    // Train Logistic Regression
    println!("Training Logistic Regression...");
    println!("Parameters:");
    println!("  - Learning rate: 0.1");
    println!("  - Max iterations: 1000");
    println!("  - Convergence tolerance: 1e-4");
    println!();

    let mut model = LogisticRegression::new()
        .with_learning_rate(0.1)
        .with_max_iter(1000)
        .with_tolerance(1e-4);

    model.fit(&x_train, &y_train).expect("Failed to fit model");

    // Make predictions
    let predictions = model.predict(&x_train);
    let probabilities = model.predict_proba(&x_train);

    // Calculate accuracy
    let accuracy = model.score(&x_train, &y_train);

    // Print results
    println!("Training Results:");
    println!("{}", "=".repeat(70));
    println!(
        "{:>6} {:>8} {:>12} {:>14} {:>12} {:>10}",
        "Sample", "Feature1", "Feature2", "True Class", "Predicted", "Prob(1)"
    );
    println!("{}", "-".repeat(70));

    for i in 0..20 {
        let match_symbol = if y_train[i] == predictions[i] {
            "✓"
        } else {
            "✗"
        };
        println!(
            "{:>6} {:>8.2} {:>12.2} {:>14} {:>12} {:>9.3} {}",
            i,
            x_train.get(i, 0),
            x_train.get(i, 1),
            y_train[i],
            predictions[i],
            probabilities[i],
            match_symbol
        );
    }

    println!("{}", "=".repeat(70));
    println!("Training Accuracy: {:.1}%", accuracy * 100.0);

    // Test on new samples
    println!("\n\nTesting on new samples:");
    println!("{}", "=".repeat(60));

    let x_test = Matrix::from_vec(
        4,
        2,
        vec![
            // Should predict class 0 (low values)
            2.5, 2.0, // Should predict class 0
            3.0, 2.5, // Boundary case (might go either way)
            5.0, 5.0, // Should predict class 1 (high values)
            7.5, 8.0,
        ],
    )
    .expect("Example data should be valid");

    let test_predictions = model.predict(&x_test);
    let test_probabilities = model.predict_proba(&x_test);

    println!(
        "{:>10} {:>12} {:>14} {:>14}",
        "Feature1", "Feature2", "Predicted", "Prob(1)"
    );
    println!("{}", "-".repeat(60));

    for i in 0..4 {
        let class_name = if test_predictions[i] == 0 {
            "Class 0"
        } else {
            "Class 1"
        };
        println!(
            "{:>10.2} {:>12.2} {:>14} {:>13.3}",
            x_test.get(i, 0),
            x_test.get(i, 1),
            class_name,
            test_probabilities[i]
        );
    }

    println!("\n=== Key Insights ===");
    println!("- Logistic Regression uses sigmoid activation: σ(z) = 1 / (1 + e^(-z))");
    println!("- Outputs probability of class 1 for each sample");
    println!("- Decision boundary at probability = 0.5");
    println!("- Trained using gradient descent with binary cross-entropy loss");
    println!("\n✅ Logistic Regression classification complete!");
}
