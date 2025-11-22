//! Decision Tree Regression example
//!
//! Demonstrates Decision Tree regression using simulated housing data,
//! comparing with linear regression and showing max_depth effects.

use aprender::prelude::*;

fn main() {
    println!("Decision Tree Regression Example");
    println!("=================================\n");

    // Simulated housing data with non-linear patterns
    // Features: [sqft, bedrooms, bathrooms, age]
    // Target: price (in thousands)
    // Note: Includes non-linear relationships that trees capture well
    let x_train = Matrix::from_vec(
        20,
        4,
        vec![
            // Small houses
            1000.0, 2.0, 1.0, 50.0, 1100.0, 2.0, 1.0, 45.0, 1200.0, 2.0, 1.0, 40.0, 1300.0, 2.0,
            1.5, 35.0, // Medium houses
            1500.0, 3.0, 2.0, 25.0, 1600.0, 3.0, 2.0, 20.0, 1700.0, 3.0, 2.0, 15.0, 1800.0, 3.0,
            2.0, 10.0, // Large houses (newer = more valuable, non-linear effect)
            2000.0, 4.0, 2.5, 8.0, 2200.0, 4.0, 3.0, 5.0, 2500.0, 5.0, 3.0, 3.0, 2800.0, 5.0, 3.5,
            2.0, // Very large houses
            3000.0, 5.0, 4.0, 1.0, 3200.0, 6.0, 4.0, 1.0, 3500.0, 6.0, 4.5, 2.0, 3800.0, 7.0, 5.0,
            1.0, // Luxury houses (exponential price increase)
            4000.0, 7.0, 5.0, 0.5, 4500.0, 8.0, 6.0, 0.5, 5000.0, 8.0, 6.0, 1.0, 5500.0, 9.0, 7.0,
            0.5,
        ],
    )
    .expect("Example data should be valid");

    // Prices with non-linear patterns
    // Price increases non-linearly with size and age discount
    let y_train = Vector::from_slice(&[
        140.0, 145.0, 150.0, 160.0, // Small houses
        250.0, 265.0, 280.0, 295.0, // Medium houses
        360.0, 410.0, 480.0, 550.0, // Large houses
        650.0, 720.0, 800.0, 920.0, // Very large houses
        1100.0, 1350.0, 1600.0, 1950.0, // Luxury houses
    ]);

    // Test data for predictions
    let x_test = Matrix::from_vec(
        5,
        4,
        vec![
            1150.0, 2.0, 1.0, 42.0, // Small house
            1650.0, 3.0, 2.0, 18.0, // Medium house
            2300.0, 4.0, 3.0, 6.0, // Large house
            3300.0, 6.0, 4.0, 1.5, // Very large house
            4800.0, 8.0, 6.0, 0.8, // Luxury house
        ],
    )
    .expect("Example data should be valid");

    println!("=== Part 1: Decision Tree vs Linear Regression ===\n");

    // Train Decision Tree
    println!("Training Decision Tree Regressor (max_depth=5)...");
    let mut tree = DecisionTreeRegressor::new().with_max_depth(5);
    tree.fit(&x_train, &y_train).expect("Failed to fit tree");

    // Train Linear Regression for comparison
    println!("Training Linear Regression for comparison...");
    let mut linear = LinearRegression::new();
    linear.fit(&x_train, &y_train).expect("Failed to fit LR");

    // Compare performance on training data
    let tree_r2 = tree.score(&x_train, &y_train);
    let linear_r2 = linear.score(&x_train, &y_train);

    println!("\nTraining Performance:");
    println!("  Decision Tree R² Score: {tree_r2:.4}");
    println!("  Linear Regression R²:   {linear_r2:.4}");
    println!("  → Tree advantage:       {:.4}", tree_r2 - linear_r2);

    // Predictions on test data
    let tree_preds = tree.predict(&x_test);
    let linear_preds = linear.predict(&x_test);

    println!("\nTest Predictions Comparison:");
    println!(
        "{:>12} {:>12} {:>12} {:>12}",
        "Sqft", "Tree Pred", "Linear Pred", "Difference"
    );
    println!("{}", "-".repeat(50));

    let sqft_values = [1150.0, 1650.0, 2300.0, 3300.0, 4800.0];
    let tree_slice = tree_preds.as_slice();
    let linear_slice = linear_preds.as_slice();

    for (i, &sqft) in sqft_values.iter().enumerate() {
        let tree_pred = tree_slice[i];
        let linear_pred = linear_slice[i];
        let diff = tree_pred - linear_pred;
        println!("{sqft:>12.0} {tree_pred:>12.0} {linear_pred:>12.0} {diff:>12.0}");
    }

    println!("\n=== Part 2: Effect of max_depth Parameter ===\n");

    // Train trees with different max_depth values
    let depths = [2, 3, 5, 10];
    println!("Comparing different max_depth values:");
    println!(
        "{:>12} {:>12} {:>12} {:>12}",
        "max_depth", "R²", "MSE", "Depth"
    );
    println!("{}", "-".repeat(50));

    for &depth in &depths {
        let mut tree = DecisionTreeRegressor::new().with_max_depth(depth);
        tree.fit(&x_train, &y_train)
            .expect("Example data should be valid");

        let preds = tree.predict(&x_train);
        let r2 = tree.score(&x_train, &y_train);
        let mse_val = mse(&preds, &y_train);

        println!("{depth:>12} {r2:>12.4} {mse_val:>12.2} {depth:>12}");
    }

    println!("\n=== Part 3: Min Samples Parameters ===\n");

    // Demonstrate pruning parameters
    let mut tree_default = DecisionTreeRegressor::new().with_max_depth(10);

    let mut tree_pruned = DecisionTreeRegressor::new()
        .with_max_depth(10)
        .with_min_samples_split(4)
        .with_min_samples_leaf(2);

    tree_default
        .fit(&x_train, &y_train)
        .expect("Example data should be valid");
    tree_pruned
        .fit(&x_train, &y_train)
        .expect("Example data should be valid");

    let r2_default = tree_default.score(&x_train, &y_train);
    let r2_pruned = tree_pruned.score(&x_train, &y_train);

    println!("Pruning parameters prevent overfitting:");
    println!("  Default tree R²:           {r2_default:.4}");
    println!("  Pruned tree R²:            {r2_pruned:.4}");
    println!("  (Pruned: min_split=4, min_leaf=2)");

    println!("\n=== Part 4: Handling Non-Linear Patterns ===\n");

    // Create purely quadratic data to show tree strength
    let x_quad = Matrix::from_vec(
        10,
        1,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .expect("Example data should be valid");

    // y = x²
    let y_quad = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0]);

    let mut tree_quad = DecisionTreeRegressor::new().with_max_depth(4);
    let mut linear_quad = LinearRegression::new();

    tree_quad
        .fit(&x_quad, &y_quad)
        .expect("Example data should be valid");
    linear_quad
        .fit(&x_quad, &y_quad)
        .expect("Example data should be valid");

    let tree_r2_quad = tree_quad.score(&x_quad, &y_quad);
    let linear_r2_quad = linear_quad.score(&x_quad, &y_quad);

    println!("Performance on quadratic data (y = x²):");
    println!("  Decision Tree R²:     {tree_r2_quad:.4}");
    println!("  Linear Regression R²: {linear_r2_quad:.4}");
    println!(
        "  → Tree captures non-linearity {:.1}% better",
        (tree_r2_quad - linear_r2_quad) * 100.0
    );

    // Show predictions
    println!("\nQuadratic pattern predictions:");
    println!("{:>6} {:>8} {:>10} {:>10}", "x", "True y", "Tree", "Linear");
    println!("{}", "-".repeat(36));

    let tree_preds_quad = tree_quad.predict(&x_quad);
    let linear_preds_quad = linear_quad.predict(&x_quad);

    for i in 0..10 {
        let x = (i + 1) as f32;
        let y_true = y_quad.as_slice()[i];
        let tree_pred = tree_preds_quad.as_slice()[i];
        let linear_pred = linear_preds_quad.as_slice()[i];

        println!("{x:>6.0} {y_true:>8.0} {tree_pred:>10.1} {linear_pred:>10.1}");
    }

    println!("\n=== Summary ===\n");
    println!("✅ Decision Trees for Regression:");
    println!("   • Capture non-linear relationships without feature engineering");
    println!("   • Handle complex interactions between features");
    println!("   • max_depth controls model complexity (prevent overfitting)");
    println!("   • min_samples_split/leaf provide additional regularization");
    println!("   • Outperform linear models on non-linear data");
    println!("   • May overfit on small datasets without proper tuning");
    println!("\n✅ When to use Decision Tree Regression:");
    println!("   • Non-linear relationships in data");
    println!("   • Feature interactions are important");
    println!("   • Interpretability is needed (tree structure)");
    println!("   • Sufficient training data (avoid overfitting)");
    println!("\n✅ When to prefer Linear Regression:");
    println!("   • Linear relationships in data");
    println!("   • Small datasets (better generalization)");
    println!("   • Need smooth predictions");
    println!("   • Extrapolation beyond training range");
}
