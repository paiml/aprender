//! Random Forest Iris Classification Example
//!
//! Demonstrates Random Forest ensemble classifier on the classic Iris dataset.
//! Shows how bootstrap sampling and majority voting improve accuracy and stability.

use aprender::primitives::Matrix;
use aprender::tree::{DecisionTreeClassifier, RandomForestClassifier};

fn main() {
    println!("Random Forest - Iris Classification Example");
    println!("============================================\n");

    // Iris dataset (simplified - 3 classes)
    let x = Matrix::from_vec(
        12,
        2,
        vec![
            // Setosa (class 0) - small petals
            1.4, 0.2, 1.3, 0.2, 1.5, 0.2, 1.7, 0.4,
            // Versicolor (class 1) - medium petals
            4.7, 1.4, 4.5, 1.5, 4.9, 1.5, 4.6, 1.3, // Virginica (class 2) - large petals
            6.0, 2.5, 5.9, 2.1, 6.1, 2.3, 5.8, 2.2,
        ],
    )
    .unwrap();
    let y = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

    println!("Dataset:");
    println!("  Samples: 12 (4 per class)");
    println!("  Features: 2 (petal length, petal width)");
    println!("  Classes: 3 (Setosa, Versicolor, Virginica)\n");

    // Example 1: Single Decision Tree
    println!("Example 1: Single Decision Tree");
    println!("--------------------------------");
    single_tree_example(&x, &y);

    // Example 2: Random Forest (5 trees)
    println!("\nExample 2: Random Forest (5 trees)");
    println!("----------------------------------");
    random_forest_example(&x, &y, 5);

    // Example 3: Larger Random Forest (20 trees)
    println!("\nExample 3: Random Forest (20 trees)");
    println!("-----------------------------------");
    random_forest_example(&x, &y, 20);

    println!("\n✅ Random Forest Examples Complete!");
    println!("\nKey Advantages:");
    println!("  • Ensemble learning reduces overfitting");
    println!("  • Bootstrap sampling creates diversity");
    println!("  • Majority voting smooths predictions");
    println!("  • More stable than single trees");
    println!("  • Excellent for real-world classification");
}

fn single_tree_example(x: &Matrix<f32>, y: &[usize]) {
    let mut tree = DecisionTreeClassifier::new().with_max_depth(5);

    tree.fit(x, y).expect("Training failed");

    let predictions = tree.predict(x);
    let accuracy = tree.score(x, y);

    println!("  Max Depth: 5");
    println!("  Training Accuracy: {:.1}%", accuracy * 100.0);
    println!("  Predictions: {:?}", predictions);

    // Show some individual predictions
    println!("\n  Sample Predictions:");
    for i in [0, 4, 8].iter() {
        println!(
            "    Sample {}: True={}, Predicted={}",
            i, y[*i], predictions[*i]
        );
    }
}

fn random_forest_example(x: &Matrix<f32>, y: &[usize], n_trees: usize) {
    let mut rf = RandomForestClassifier::new(n_trees)
        .with_max_depth(5)
        .with_random_state(42);

    rf.fit(x, y).expect("Training failed");

    let predictions = rf.predict(x);
    let accuracy = rf.score(x, y);

    println!("  Number of Trees: {}", n_trees);
    println!("  Max Depth: 5");
    println!("  Random State: 42 (reproducible)");
    println!("  Training Accuracy: {:.1}%", accuracy * 100.0);
    println!("  Predictions: {:?}", predictions);

    // Show voting mechanism
    println!("\n  Sample Predictions:");
    for i in [0, 4, 8].iter() {
        println!(
            "    Sample {}: True={}, Predicted={} (from {} tree votes)",
            i, y[*i], predictions[*i], n_trees
        );
    }

    // Compare to expected
    let errors = predictions
        .iter()
        .zip(y.iter())
        .filter(|(pred, true_label)| pred != true_label)
        .count();

    if errors == 0 {
        println!("\n  ✓ Perfect classification! All samples correctly predicted.");
    } else {
        println!("\n  {} misclassifications", errors);
    }
}
