//! Data Preprocessing with Scalers Example
//!
//! Demonstrates StandardScaler and MinMaxScaler for feature normalization.
//! Shows when to use each scaler and their impact on ML algorithm performance.

use aprender::classification::KNearestNeighbors;
use aprender::prelude::*;
use aprender::preprocessing::{MinMaxScaler, StandardScaler};
use aprender::primitives::Matrix;

fn main() {
    println!("Data Preprocessing with Scalers");
    println!("================================\n");

    // Example 1: Understanding StandardScaler
    println!("Example 1: StandardScaler (Z-score Normalization)");
    println!("-------------------------------------------------");
    standard_scaler_example();

    // Example 2: Understanding MinMaxScaler
    println!("\nExample 2: MinMaxScaler (Range Normalization)");
    println!("----------------------------------------------");
    minmax_scaler_example();

    // Example 3: Comparing Scalers
    println!("\nExample 3: StandardScaler vs MinMaxScaler");
    println!("-----------------------------------------");
    scaler_comparison();

    // Example 4: Impact on K-NN Classification
    println!("\nExample 4: Impact on K-NN Classification");
    println!("-----------------------------------------");
    scaling_impact_on_knn();

    // Example 5: Custom Range with MinMaxScaler
    println!("\nExample 5: Custom Range Scaling");
    println!("--------------------------------");
    custom_range_example();

    // Example 6: Inverse Transformation
    println!("\nExample 6: Inverse Transformation");
    println!("----------------------------------");
    inverse_transform_example();

    println!("\n✅ Data Preprocessing Examples Complete!");
    println!("\nKey Takeaways:");
    println!("  • StandardScaler: Best for normally distributed data");
    println!("  • MinMaxScaler: Best when you need specific range");
    println!("  • Always fit() on training data only");
    println!("  • Apply same scaler to test data with transform()");
    println!("  • Scaling crucial for distance-based algorithms");
    println!("  • Use inverse_transform() to get original scale back");
}

fn standard_scaler_example() {
    // Data with different scales
    let data = Matrix::from_vec(
        5,
        2,
        vec![
            100.0, 1.0, // Large value, small value
            200.0, 2.0, 300.0, 3.0, 400.0, 4.0, 500.0, 5.0,
        ],
    )
    .expect("Example data should be valid");

    let mut scaler = StandardScaler::new();
    scaler.fit(&data).expect("Example data should be valid");

    let scaled = scaler
        .transform(&data)
        .expect("Example data should be valid");

    println!("  Original Data:");
    println!("    Feature 0: [100, 200, 300, 400, 500]");
    println!("    Feature 1: [1, 2, 3, 4, 5]");

    println!("\n  Computed Statistics:");
    println!("    Mean: {:?}", scaler.mean());
    println!("    Std:  {:?}", scaler.std());

    println!("\n  After StandardScaler:");
    for i in 0..5 {
        println!(
            "    Sample {}: [{:>6.2}, {:>6.2}]",
            i,
            scaled.get(i, 0),
            scaled.get(i, 1)
        );
    }

    println!("\n  What StandardScaler Does:");
    println!("    • Centers data: subtracts mean");
    println!("    • Scales data: divides by standard deviation");
    println!("    • Result: mean=0, std=1 for each feature");
    println!("    • Formula: z = (x - μ) / σ");
}

fn minmax_scaler_example() {
    // Data with different ranges
    let data = Matrix::from_vec(
        5,
        2,
        vec![
            10.0, 100.0, // Different scales
            20.0, 200.0, 30.0, 300.0, 40.0, 400.0, 50.0, 500.0,
        ],
    )
    .expect("Example data should be valid");

    let mut scaler = MinMaxScaler::new(); // Default range [0, 1]
    scaler.fit(&data).expect("Example data should be valid");

    let scaled = scaler
        .transform(&data)
        .expect("Example data should be valid");

    println!("  Original Data:");
    println!("    Feature 0: [10, 20, 30, 40, 50]");
    println!("    Feature 1: [100, 200, 300, 400, 500]");

    println!("\n  Computed Range:");
    println!("    Min: {:?}", scaler.data_min());
    println!("    Max: {:?}", scaler.data_max());

    println!("\n  After MinMaxScaler [0, 1]:");
    for i in 0..5 {
        println!(
            "    Sample {}: [{:>6.2}, {:>6.2}]",
            i,
            scaled.get(i, 0),
            scaled.get(i, 1)
        );
    }

    println!("\n  What MinMaxScaler Does:");
    println!("    • Scales to specific range (default [0, 1])");
    println!("    • Preserves shape of original distribution");
    println!("    • Formula: x' = (x - min) / (max - min)");
    println!("    • All features on same scale");
}

fn scaler_comparison() {
    // Data with outliers
    let data = Matrix::from_vec(
        6,
        1,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 100.0, // Outlier!
        ],
    )
    .expect("Example data should be valid");

    // StandardScaler
    let mut standard = StandardScaler::new();
    standard.fit(&data).expect("Example data should be valid");
    let standard_scaled = standard
        .transform(&data)
        .expect("Example data should be valid");

    // MinMaxScaler
    let mut minmax = MinMaxScaler::new();
    minmax.fit(&data).expect("Example data should be valid");
    let minmax_scaled = minmax
        .transform(&data)
        .expect("Example data should be valid");

    println!("  Data with Outlier: [1, 2, 3, 4, 5, 100]");
    println!(
        "\n  {:>10} {:>15} {:>15}",
        "Original", "StandardScaler", "MinMaxScaler"
    );
    println!("  {}", "-".repeat(42));

    for i in 0..6 {
        println!(
            "  {:>10.1} {:>15.2} {:>15.2}",
            data.get(i, 0),
            standard_scaled.get(i, 0),
            minmax_scaled.get(i, 0)
        );
    }

    println!("\n  Observations:");
    println!("    • StandardScaler: Less affected by outliers");
    println!("      (outlier is ~2.3 std devs from mean)");
    println!("    • MinMaxScaler: Heavily affected by outliers");
    println!("      (outlier compresses other values near 0)");
    println!("\n  When to Use:");
    println!("    • StandardScaler: Normally distributed data, outliers present");
    println!("    • MinMaxScaler: Need specific range, no outliers");
}

fn scaling_impact_on_knn() {
    // Create dataset where features have different scales
    // Feature 0: salary (thousands), Feature 1: age (years)
    let x_train = Matrix::from_vec(
        8,
        2,
        vec![
            50.0, 25.0, // Low salary, young -> Class 0
            55.0, 27.0, 60.0, 30.0, 65.0, 32.0, 80.0, 35.0, // High salary, young -> Class 1
            85.0, 38.0, 90.0, 40.0, 95.0, 42.0,
        ],
    )
    .expect("Example data should be valid");
    let y_train = vec![0, 0, 0, 0, 1, 1, 1, 1];

    let x_test =
        Matrix::from_vec(2, 2, vec![70.0, 33.0, 75.0, 36.0]).expect("Example data should be valid");

    // K-NN without scaling
    let mut knn_unscaled = KNearestNeighbors::new(3);
    knn_unscaled
        .fit(&x_train, &y_train)
        .expect("Example data should be valid");
    let pred_unscaled = knn_unscaled.predict(&x_test);

    // K-NN with StandardScaler
    let mut scaler = StandardScaler::new();
    let x_train_scaled = scaler
        .fit_transform(&x_train)
        .expect("Example data should be valid");
    let x_test_scaled = scaler
        .transform(&x_test)
        .expect("Example data should be valid");

    let mut knn_scaled = KNearestNeighbors::new(3);
    knn_scaled
        .fit(&x_train_scaled, &y_train)
        .expect("Example data should be valid");
    let pred_scaled = knn_scaled.predict(&x_test_scaled);

    println!("  Dataset: Employee classification");
    println!("    Feature 0: Salary (thousands)");
    println!("    Feature 1: Age (years)");
    println!("    Classes: 0=Junior, 1=Senior");

    println!("\n  Feature Scales:");
    println!("    Salary: 50-95 (range=45)");
    println!("    Age:    25-42 (range=17)");
    println!("    → Salary dominates distance calculation!");

    println!("\n  Test Samples:");
    println!("    Sample 0: Salary=70k, Age=33");
    println!("    Sample 1: Salary=75k, Age=36");

    println!("\n  Predictions (K=3):");
    println!("    Without scaling: {pred_unscaled:?}");
    println!("    With scaling:    {pred_scaled:?}");

    println!("\n  Why Scaling Matters:");
    println!("    • K-NN uses Euclidean distance");
    println!("    • Distance dominated by large-scale features");
    println!("    • Age differences (2-3 years) become negligible");
    println!("    • Scaling equalizes feature importance");
}

fn custom_range_example() {
    let data = Matrix::from_vec(5, 1, vec![0.0, 25.0, 50.0, 75.0, 100.0])
        .expect("Example data should be valid");

    // Scale to [-1, 1]
    let mut scaler = MinMaxScaler::new().with_range(-1.0, 1.0);
    scaler.fit(&data).expect("Example data should be valid");
    let scaled = scaler
        .transform(&data)
        .expect("Example data should be valid");

    println!("  Original: [0, 25, 50, 75, 100]");
    println!("  Scaled to [-1, 1]:");
    for i in 0..5 {
        println!("    {:.0} → {:.2}", data.get(i, 0), scaled.get(i, 0));
    }

    println!("\n  Use Cases for Custom Ranges:");
    println!("    • [-1, 1]: Neural network inputs (tanh activation)");
    println!("    • [0, 1]:  Probabilities, image pixels");
    println!("    • [0, 255]: 8-bit image processing");
}

fn inverse_transform_example() {
    let original = Matrix::from_vec(3, 2, vec![10.0, 100.0, 20.0, 200.0, 30.0, 300.0])
        .expect("Example data should be valid");

    // Scale and inverse
    let mut scaler = StandardScaler::new();
    let scaled = scaler
        .fit_transform(&original)
        .expect("Example data should be valid");
    let recovered = scaler
        .inverse_transform(&scaled)
        .expect("Example data should be valid");

    println!("  Original Data:");
    for i in 0..3 {
        println!(
            "    [{:>6.1}, {:>6.1}]",
            original.get(i, 0),
            original.get(i, 1)
        );
    }

    println!("\n  After Scaling:");
    for i in 0..3 {
        println!("    [{:>6.2}, {:>6.2}]", scaled.get(i, 0), scaled.get(i, 1));
    }

    println!("\n  After Inverse Transform:");
    for i in 0..3 {
        println!(
            "    [{:>6.1}, {:>6.1}]",
            recovered.get(i, 0),
            recovered.get(i, 1)
        );
    }

    // Verify recovery
    let mut max_error = 0.0_f32;
    for i in 0..3 {
        for j in 0..2 {
            let error = (original.get(i, j) - recovered.get(i, j)).abs();
            max_error = max_error.max(error);
        }
    }

    println!("\n  Maximum reconstruction error: {max_error:.6}");
    println!("  → Data perfectly recovered! ✓");

    println!("\n  When to Use Inverse Transform:");
    println!("    • Interpreting model coefficients in original scale");
    println!("    • Presenting predictions to users");
    println!("    • Visualizing scaled data");
}
