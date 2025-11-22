//! Model Serialization Example
//!
//! Demonstrates how to save and load trained models using bincode serialization.
//! This is essential for production deployment: train once, serve many times.

use aprender::prelude::*;
use std::fs;
use std::path::Path;

fn main() {
    println!("Model Serialization - Save/Load Example");
    println!("=========================================\n");

    // Example 1: LinearRegression
    println!("Example 1: LinearRegression Serialization");
    println!("------------------------------------------");
    linear_regression_example();

    // Example 2: KMeans
    println!("\nExample 2: KMeans Serialization");
    println!("--------------------------------");
    kmeans_example();

    // Example 3: DecisionTreeClassifier
    println!("\nExample 3: DecisionTreeClassifier Serialization");
    println!("-----------------------------------------------");
    decision_tree_example();

    println!("\n✅ All models successfully saved and loaded!");
    println!("\nUse Cases:");
    println!("  • Production deployment: Train offline, serve online");
    println!("  • Model versioning: Track model evolution");
    println!("  • Reproducibility: Share exact model state");
    println!("  • Performance: Avoid re-training expensive models");
}

fn linear_regression_example() {
    // Train a model
    let x_train = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("Example data should be valid");
    let y_train = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0, 11.0]); // y = 2x + 1

    let mut model = LinearRegression::new();
    model
        .fit(&x_train, &y_train)
        .expect("Example data should be valid");

    println!(
        "  Trained model: y = {:.2}x + {:.2}",
        model.coefficients()[0],
        model.intercept()
    );

    // Save model
    let path = Path::new("/tmp/linear_regression.bin");
    model.save(path).expect("Failed to save model");
    println!("  ✓ Saved to {path:?}");

    // Get file size
    let metadata = fs::metadata(path).expect("Example data should be valid");
    println!("  ✓ File size: {} bytes", metadata.len());

    // Load model
    let loaded_model = LinearRegression::load(path).expect("Failed to load model");
    println!("  ✓ Loaded from {path:?}");

    // Verify predictions match
    let x_test = Matrix::from_vec(1, 1, vec![10.0]).expect("Example data should be valid");
    let original_pred = model.predict(&x_test);
    let loaded_pred = loaded_model.predict(&x_test);

    println!("  Original prediction for x=10: {:.2}", original_pred[0]);
    println!("  Loaded prediction for x=10:   {:.2}", loaded_pred[0]);
    assert!((original_pred[0] - loaded_pred[0]).abs() < 1e-6);
    println!("  ✓ Predictions match!");

    // Cleanup
    fs::remove_file(path).ok();
}

fn kmeans_example() {
    // Create clustered data
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            1.0, 1.0, // Cluster 1
            1.5, 1.5, // Cluster 1
            2.0, 2.0, // Cluster 1
            10.0, 10.0, // Cluster 2
            10.5, 10.5, // Cluster 2
            11.0, 11.0, // Cluster 2
        ],
    )
    .expect("Example data should be valid");

    // Train model
    let mut kmeans = KMeans::new(2).with_random_state(42);
    kmeans.fit(&data).expect("Example data should be valid");

    println!("  Trained KMeans with {} clusters", 2);
    println!("  Inertia: {:.2}", kmeans.inertia());

    // Save model
    let path = Path::new("/tmp/kmeans.bin");
    kmeans.save(path).expect("Failed to save model");
    println!("  ✓ Saved to {path:?}");

    // Get file size
    let metadata = fs::metadata(path).expect("Example data should be valid");
    println!("  ✓ File size: {} bytes", metadata.len());

    // Load model
    let loaded_kmeans = KMeans::load(path).expect("Failed to load model");
    println!("  ✓ Loaded from {path:?}");

    // Verify predictions match
    let test_point = Matrix::from_vec(1, 2, vec![1.2, 1.2]).expect("Example data should be valid");
    let original_cluster = kmeans.predict(&test_point);
    let loaded_cluster = loaded_kmeans.predict(&test_point);

    println!("  Original cluster for (1.2, 1.2): {}", original_cluster[0]);
    println!("  Loaded cluster for (1.2, 1.2):   {}", loaded_cluster[0]);
    assert_eq!(original_cluster, loaded_cluster);
    println!("  ✓ Predictions match!");

    // Cleanup
    fs::remove_file(path).ok();
}

fn decision_tree_example() {
    // Create classification data (Iris-like)
    let x_train = Matrix::from_vec(
        6,
        2,
        vec![
            1.0, 1.0, // class 0
            1.5, 1.5, // class 0
            5.0, 5.0, // class 1
            5.5, 5.5, // class 1
            9.0, 9.0, // class 2
            9.5, 9.5, // class 2
        ],
    )
    .expect("Example data should be valid");
    let y_train = vec![0, 0, 1, 1, 2, 2];

    // Train model
    let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
    tree.fit(&x_train, &y_train)
        .expect("Example data should be valid");

    let accuracy = tree.score(&x_train, &y_train);
    println!("  Trained DecisionTree with max_depth=5");
    println!("  Training accuracy: {:.1}%", accuracy * 100.0);

    // Save model
    let path = Path::new("/tmp/decision_tree.bin");
    tree.save(path).expect("Failed to save model");
    println!("  ✓ Saved to {path:?}");

    // Get file size
    let metadata = fs::metadata(path).expect("Example data should be valid");
    println!("  ✓ File size: {} bytes", metadata.len());

    // Load model
    let loaded_tree = DecisionTreeClassifier::load(path).expect("Failed to load model");
    println!("  ✓ Loaded from {path:?}");

    // Verify predictions match
    let test_data = Matrix::from_vec(1, 2, vec![5.2, 5.2]).expect("Example data should be valid");
    let original_pred = tree.predict(&test_data);
    let loaded_pred = loaded_tree.predict(&test_data);

    println!(
        "  Original prediction for (5.2, 5.2): class {}",
        original_pred[0]
    );
    println!(
        "  Loaded prediction for (5.2, 5.2):   class {}",
        loaded_pred[0]
    );
    assert_eq!(original_pred, loaded_pred);
    println!("  ✓ Predictions match!");

    // Cleanup
    fs::remove_file(path).ok();
}
