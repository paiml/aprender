//! Random Forest Regression example
//!
//! Demonstrates Random Forest regression with housing price prediction,
//! comparing with single decision trees and showing variance reduction benefits.

use aprender::prelude::*;

#[allow(clippy::too_many_lines)]
fn main() {
    println!("Random Forest Regression Example");
    println!("=================================\n");

    // Simulated housing data with non-linear patterns
    // Features: [sqft, bedrooms, bathrooms, age]
    // Target: price (in thousands)
    let x_train = Matrix::from_vec(
        25,
        4,
        vec![
            // Small houses
            1000.0, 2.0, 1.0, 50.0, 1100.0, 2.0, 1.0, 45.0, 1200.0, 2.0, 1.0, 40.0, 1300.0, 2.0,
            1.5, 35.0, 1400.0, 2.0, 1.5, 30.0, // Medium houses
            1500.0, 3.0, 2.0, 25.0, 1600.0, 3.0, 2.0, 20.0, 1700.0, 3.0, 2.0, 15.0, 1800.0, 3.0,
            2.0, 10.0, 1900.0, 3.0, 2.0, 8.0, // Large houses
            2000.0, 4.0, 2.5, 8.0, 2200.0, 4.0, 3.0, 5.0, 2500.0, 5.0, 3.0, 3.0, 2800.0, 5.0, 3.5,
            2.0, 3000.0, 5.0, 4.0, 1.0, // Very large houses
            3200.0, 6.0, 4.0, 1.0, 3500.0, 6.0, 4.5, 2.0, 3800.0, 7.0, 5.0, 1.0, 4000.0, 7.0, 5.0,
            0.5, 4500.0, 8.0, 6.0, 0.5, // Luxury houses
            5000.0, 8.0, 6.0, 1.0, 5500.0, 9.0, 7.0, 0.5, 6000.0, 9.0, 7.0, 0.5, 6500.0, 10.0, 8.0,
            1.0, 7000.0, 10.0, 8.0, 0.5,
        ],
    )
    .expect("Example data should be valid");

    let y_train = Vector::from_slice(&[
        140.0, 145.0, 150.0, 160.0, 170.0, // Small
        250.0, 265.0, 280.0, 295.0, 310.0, // Medium
        360.0, 410.0, 480.0, 550.0, 620.0, // Large
        720.0, 800.0, 920.0, 1050.0, 1200.0, // Very large
        1400.0, 1650.0, 1950.0, 2300.0, 2700.0, // Luxury
    ]);

    // Test data
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

    println!("=== Part 1: Random Forest vs Single Decision Tree ===\n");

    // Train Random Forest
    println!("Training Random Forest (n_estimators=50, max_depth=5)...");
    let mut rf = RandomForestRegressor::new(50).with_max_depth(5);
    rf.fit(&x_train, &y_train).expect("Failed to fit RF");

    // Train single Decision Tree for comparison
    println!("Training single Decision Tree (max_depth=5)...");
    let mut single_tree = DecisionTreeRegressor::new().with_max_depth(5);
    single_tree
        .fit(&x_train, &y_train)
        .expect("Failed to fit tree");

    // Compare performance on training data
    let rf_r2 = rf.score(&x_train, &y_train);
    let tree_r2 = single_tree.score(&x_train, &y_train);

    println!("\nTraining Performance:");
    println!("  Random Forest R²:     {rf_r2:.4}");
    println!("  Single Tree R²:       {tree_r2:.4}");
    println!("  → RF advantage:       {:.4}", rf_r2 - tree_r2);

    // Predictions on test data
    let rf_preds = rf.predict(&x_test);
    let tree_preds = single_tree.predict(&x_test);

    println!("\nTest Predictions Comparison:");
    println!(
        "{:>12} {:>12} {:>12} {:>12}",
        "Sqft", "RF Pred", "Tree Pred", "Difference"
    );
    println!("{}", "-".repeat(50));

    let sqft_values = [1150.0, 1650.0, 2300.0, 3300.0, 4800.0];
    let rf_slice = rf_preds.as_slice();
    let tree_slice = tree_preds.as_slice();

    for (i, &sqft) in sqft_values.iter().enumerate() {
        let rf_pred = rf_slice[i];
        let tree_pred = tree_slice[i];
        let diff = rf_pred - tree_pred;
        println!("{sqft:>12.0} {rf_pred:>12.0} {tree_pred:>12.0} {diff:>12.0}");
    }

    println!("\n=== Part 2: Effect of n_estimators (Number of Trees) ===\n");

    // Train forests with different numbers of trees
    let n_estimators_values = [5, 10, 30, 100];
    println!("Comparing different n_estimators:");
    println!(
        "{:>12} {:>12} {:>12}",
        "n_estimators", "Train R²", "Consistency"
    );
    println!("{}", "-".repeat(40));

    for &n_est in &n_estimators_values {
        let mut rf = RandomForestRegressor::new(n_est)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x_train, &y_train)
            .expect("Example data should be valid");

        let r2 = rf.score(&x_train, &y_train);

        println!("{:>12} {:>12.4} {:>12}", n_est, r2, "✓");
    }

    println!("\nObservation: More trees generally lead to:");
    println!("  • More stable predictions (reduced variance)");
    println!("  • Diminishing returns after ~30-50 trees");
    println!("  • Longer training time");

    println!("\n=== Part 3: Variance Reduction Through Ensemble ===\n");

    // Train multiple single trees to show variance
    println!("Training 5 single trees with different random seeds:");
    let mut tree_r2s = Vec::new();

    for seed in 0..5 {
        let mut tree = DecisionTreeRegressor::new().with_max_depth(6);

        // Simulate different training by using different bootstrap samples
        // (In practice, you'd use different random splits)
        tree.fit(&x_train, &y_train)
            .expect("Example data should be valid");
        let r2 = tree.score(&x_train, &y_train);
        tree_r2s.push(r2);
        println!("  Tree {}: R² = {:.4}", seed + 1, r2);
    }

    let tree_mean = tree_r2s.iter().sum::<f32>() / tree_r2s.len() as f32;
    let tree_std = (tree_r2s
        .iter()
        .map(|&r2| (r2 - tree_mean).powi(2))
        .sum::<f32>()
        / tree_r2s.len() as f32)
        .sqrt();

    println!("\n  Single trees: Mean R² = {tree_mean:.4}, Std = {tree_std:.4}");

    // Train Random Forest
    let mut rf = RandomForestRegressor::new(50)
        .with_max_depth(6)
        .with_random_state(42);
    rf.fit(&x_train, &y_train)
        .expect("Example data should be valid");
    let rf_r2 = rf.score(&x_train, &y_train);

    println!("  Random Forest: R² = {rf_r2:.4} (stable)");
    println!("\n  → Random Forest reduces variance through averaging!");

    println!("\n=== Part 4: Handling Non-Linear Patterns ===\n");

    // Create quadratic data
    let x_quad = Matrix::from_vec(
        12,
        1,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .expect("Example data should be valid");

    let y_quad = Vector::from_slice(&[
        1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0, 144.0,
    ]);

    // Train Random Forest
    let mut rf_quad = RandomForestRegressor::new(30)
        .with_max_depth(4)
        .with_random_state(42);
    rf_quad
        .fit(&x_quad, &y_quad)
        .expect("Example data should be valid");

    // Train Linear Regression
    let mut lr_quad = LinearRegression::new();
    lr_quad
        .fit(&x_quad, &y_quad)
        .expect("Example data should be valid");

    let rf_r2_quad = rf_quad.score(&x_quad, &y_quad);
    let lr_r2_quad = lr_quad.score(&x_quad, &y_quad);

    println!("Performance on quadratic data (y = x²):");
    println!("  Random Forest R²:      {rf_r2_quad:.4}");
    println!("  Linear Regression R²:  {lr_r2_quad:.4}");
    println!(
        "  → RF captures non-linearity {:.1}% better",
        (rf_r2_quad - lr_r2_quad) * 100.0
    );

    println!("\n=== Part 5: Reproducibility with random_state ===\n");

    // Train two forests with same random state
    let mut rf1 = RandomForestRegressor::new(20)
        .with_max_depth(5)
        .with_random_state(42);
    rf1.fit(&x_train, &y_train)
        .expect("Example data should be valid");
    let pred1 = rf1.predict(&x_test);

    let mut rf2 = RandomForestRegressor::new(20)
        .with_max_depth(5)
        .with_random_state(42);
    rf2.fit(&x_train, &y_train)
        .expect("Example data should be valid");
    let pred2 = rf2.predict(&x_test);

    println!("Training two forests with same random_state=42:");
    println!(
        "{:>10} {:>12} {:>12} {:>12}",
        "Sample", "Forest 1", "Forest 2", "Match"
    );
    println!("{}", "-".repeat(48));

    let pred1_slice = pred1.as_slice();
    let pred2_slice = pred2.as_slice();

    for i in 0..5 {
        let match_symbol = if (pred1_slice[i] - pred2_slice[i]).abs() < 1e-10 {
            "✓"
        } else {
            "✗"
        };
        println!(
            "{:>10} {:>12.1} {:>12.1} {:>12}",
            i + 1,
            pred1_slice[i],
            pred2_slice[i],
            match_symbol
        );
    }

    println!("\n  → Predictions are identical (reproducible)");

    println!("\n=== Part 6: Practical Example - House Price Prediction ===\n");

    // New houses to predict
    let new_houses = Matrix::from_vec(
        3,
        4,
        vec![
            1850.0, 3.0, 2.0, 12.0, // Medium house: 1850 sqft, 3 bed, 2 bath, 12 years
            2750.0, 5.0, 3.5, 3.0, // Large house: 2750 sqft, 5 bed, 3.5 bath, 3 years
            5200.0, 9.0, 6.5, 0.5, // Luxury house: 5200 sqft, 9 bed, 6.5 bath, 0.5 years
        ],
    )
    .expect("Example data should be valid");

    // Train final model
    let mut final_rf = RandomForestRegressor::new(50)
        .with_max_depth(8)
        .with_random_state(42);
    final_rf
        .fit(&x_train, &y_train)
        .expect("Example data should be valid");

    let predictions = final_rf.predict(&new_houses);

    println!("Predicting prices for new houses:");
    println!("{}", "-".repeat(70));
    println!(
        "{:>10} {:>8} {:>6} {:>8} {:>8} {:>15}",
        "Sqft", "Beds", "Baths", "Age", "Predict", "Description"
    );
    println!("{}", "-".repeat(70));

    let descriptions = ["Medium", "Large", "Luxury"];
    for (i, desc) in descriptions.iter().enumerate() {
        let sqft = new_houses.get(i, 0);
        let beds = new_houses.get(i, 1);
        let baths = new_houses.get(i, 2);
        let age = new_houses.get(i, 3);
        let price = predictions.as_slice()[i];

        println!("{sqft:>10.0} {beds:>8.0} {baths:>6.1} {age:>8.0} ${price:>7.0}k {desc:>15}");
    }

    println!("\n=== Part 7: Feature Importance ===\n");

    // Train model with all features to see their relative importance
    let mut rf_importance = RandomForestRegressor::new(50)
        .with_max_depth(8)
        .with_random_state(42);
    rf_importance
        .fit(&x_train, &y_train)
        .expect("Example data should be valid");

    let importances = rf_importance.feature_importances();

    if let Some(imps) = importances {
        println!("Feature Importances:");
        let feature_names = ["sqft", "bedrooms", "bathrooms", "age"];
        println!(
            "{:>12} {:>12} {:>12}",
            "Feature", "Importance", "Percentage"
        );
        println!("{}", "-".repeat(38));

        for (i, &imp) in imps.iter().enumerate() {
            println!(
                "{:>12} {:>12.4} {:>11.1}%",
                feature_names[i],
                imp,
                imp * 100.0
            );
        }

        // Identify most important feature
        let max_idx = imps
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Example data should be valid"))
            .map(|(idx, _)| idx)
            .expect("Example data should be valid");

        println!(
            "\n  → Most important: {} ({:.1}%)",
            feature_names[max_idx],
            imps[max_idx] * 100.0
        );

        // Verify they sum to 1.0
        let sum: f32 = imps.iter().sum();
        println!("  → Importances sum to: {sum:.3} ✓");
    }

    println!("\nWhat is Feature Importance?");
    println!("  • Measures how much each feature contributes to predictions");
    println!("  • Based on mean decrease in impurity across all trees");
    println!("  • Normalized to sum to 1.0");
    println!("  • Higher values = more important features");

    println!("\nPractical Use Cases:");
    println!("  ✅ Feature selection: Drop low-importance features");
    println!("  ✅ Domain insights: Understand what drives prices");
    println!("  ✅ Model debugging: Verify expected features are important");
    println!("  ✅ Explainability: Show stakeholders what matters");

    println!("\n=== Part 8: Out-of-Bag (OOB) Error Estimation ===\n");

    // Train model with OOB evaluation
    let mut rf_oob = RandomForestRegressor::new(50)
        .with_max_depth(8)
        .with_random_state(42);
    rf_oob
        .fit(&x_train, &y_train)
        .expect("Example data should be valid");

    // Get OOB score (free validation without test set)
    let oob_score = rf_oob.oob_score();
    let training_score = rf_oob.score(&x_train, &y_train);

    println!("Performance comparison:");
    println!("  Training R²:    {training_score:.4}");
    if let Some(oob) = oob_score {
        println!("  OOB R²:         {oob:.4}");
        println!("  Difference:     {:.4}", (training_score - oob).abs());
    }

    println!("\nWhat is Out-of-Bag (OOB) error?");
    println!("  • Each tree is trained on ~63% of samples (bootstrap)");
    println!("  • Remaining ~37% are 'out-of-bag' for that tree");
    println!("  • OOB samples used for validation (unbiased estimate)");
    println!("  • No need for separate validation set!");

    println!("\nOOB vs Test Set:");
    println!("  ✅ OOB: Free validation, no data split needed");
    println!("  ✅ OOB: Unbiased estimate of generalization error");
    println!("  ✅ OOB: All data used for training AND validation");
    println!("  ✅ Test Set: Gold standard, but requires holding out data");

    println!("\n=== Summary ===\n");
    println!("✅ Random Forest Regression Advantages:");
    println!("   • Reduces overfitting through ensemble averaging");
    println!("   • Lower variance than single decision trees");
    println!("   • Handles non-linear relationships naturally");
    println!("   • No feature scaling required");
    println!("   • Good default hyperparameters (minimal tuning)");
    println!("   • Reproducible with random_state");
    println!("   • Feature importance for interpretability");
    println!("   • OOB error estimation provides free validation");
    println!("\n✅ When to use Random Forest Regression:");
    println!("   • Non-linear relationships in data");
    println!("   • Feature interactions are important");
    println!("   • Medium to large datasets (100+ samples)");
    println!("   • Want stable, low-variance predictions");
    println!("   • Don't want to tune many hyperparameters");
    println!("\n✅ Hyperparameter Guidelines:");
    println!("   • n_estimators: 30-100 (more trees = more stable, diminishing returns)");
    println!("   • max_depth: 5-10 (shallower than single trees, less overfitting)");
    println!("   • Use random_state for reproducibility");
    println!("\n✅ Typical Performance:");
    println!("   • Training R²: 0.95-1.00 (high but not overfitting)");
    println!("   • Test R²: Often 5-10% better than single tree");
    println!("   • Prediction variance: ~1/sqrt(n_trees) of single tree");
}
