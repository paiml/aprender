//! Iris Classification example - Decision Tree
//!
//! Demonstrates Decision Tree classification using simulated iris data.

use aprender::prelude::*;

fn main() {
    println!("Iris Classification - Decision Tree Example");
    println!("============================================\n");

    // Simulated iris-like data
    // Features: [sepal_length, sepal_width, petal_length, petal_width]
    // Three distinct species: Setosa (0), Versicolor (1), Virginica (2)
    let x_train = Matrix::from_vec(
        15,
        4,
        vec![
            // Setosa (class 0)
            5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2, 4.6, 3.1, 1.5, 0.2, 5.0,
            3.6, 1.4, 0.2, // Versicolor (class 1)
            7.0, 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5, 6.9, 3.1, 4.9, 1.5, 5.5, 2.3, 4.0, 1.3, 6.5,
            2.8, 4.6, 1.5, // Virginica (class 2)
            6.3, 3.3, 6.0, 2.5, 5.8, 2.7, 5.1, 1.9, 7.1, 3.0, 5.9, 2.1, 6.3, 2.9, 5.6, 1.8, 6.5,
            3.0, 5.8, 2.2,
        ],
    )
    .expect("Example data should be valid");

    let y_train = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2];

    // Fit Decision Tree with max_depth=5
    println!("Training Decision Tree (max_depth=5)...");
    let mut tree = DecisionTreeClassifier::new().with_max_depth(5);

    tree.fit(&x_train, &y_train)
        .expect("Failed to fit Decision Tree");

    // Make predictions
    let predicted_labels = tree.predict(&x_train);

    // Calculate accuracy
    let accuracy = tree.score(&x_train, &y_train);

    // Print results
    println!("\nClassification Results:");
    println!(
        "{:>6} {:>10} {:>12} {:>10}",
        "Sample", "True", "Predicted", "Match"
    );
    println!("{}", "-".repeat(42));

    for i in 0..15 {
        let match_symbol = if y_train[i] == predicted_labels[i] {
            "✓"
        } else {
            "✗"
        };
        println!(
            "{:>6} {:>10} {:>12} {:>10}",
            i, y_train[i], predicted_labels[i], match_symbol
        );
    }

    println!("\n{}", "=".repeat(42));
    println!("Training Accuracy: {:.1}%", accuracy * 100.0);

    // Test on a few new samples
    println!("\n\nTesting on new samples:");
    println!("{}", "=".repeat(42));

    let x_test = Matrix::from_vec(
        3,
        4,
        vec![
            // Likely Setosa
            5.0, 3.4, 1.5, 0.2, // Likely Versicolor
            6.2, 2.9, 4.3, 1.3, // Likely Virginica
            6.7, 3.1, 5.6, 2.4,
        ],
    )
    .expect("Example data should be valid");

    let test_predictions = tree.predict(&x_test);
    let species = ["Setosa", "Versicolor", "Virginica"];

    println!("\n{:>6} {:>15}", "Sample", "Predicted Species");
    println!("{}", "-".repeat(25));

    for (i, &pred) in test_predictions.iter().enumerate() {
        println!("{:>6} {:>15}", i, species[pred]);
    }

    println!("\n✅ Decision Tree classification complete!");
}
