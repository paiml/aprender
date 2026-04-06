#![allow(clippy::disallowed_methods)]
//! Regularized Regression Example
//!
//! Demonstrates Ridge, Lasso, and ElasticNet regression with:
//! - Feature scaling using StandardScaler
//! - Grid search for optimal hyperparameters
//! - Cross-validation for model evaluation
//! - Comparison of different regularization methods
//!
//! Run with: `cargo run --example regularized_regression`

use aprender::model_selection::{grid_search_alpha, train_test_split, KFold};
use aprender::prelude::*;

#[allow(clippy::too_many_lines)]
fn main() {
    println!("=== Regularized Regression Demo ===\n");

    // Create synthetic dataset with noise
    // y = 3*x1 + 2*x2 - x3 + noise
    let n_samples = 100;
    let mut x_data = Vec::with_capacity(n_samples * 3);
    let mut y_data = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let x1 = (i as f32) / 10.0;
        let x2 = (i as f32) / 20.0 + 5.0;
        let x3 = (i as f32) / 30.0 - 2.0;

        x_data.push(x1);
        x_data.push(x2);
        x_data.push(x3);

        // True relationship: y = 3*x1 + 2*x2 - x3 + small noise
        let noise = ((i as f32 * 0.1).sin()) * 0.5;
        y_data.push(3.0 * x1 + 2.0 * x2 - x3 + noise);
    }

    let x = Matrix::from_vec(n_samples, 3, x_data).expect("Example data should be valid");
    let y = Vector::from_vec(y_data);

    println!("Dataset: {n_samples} samples, 3 features");
    println!("True coefficients: [3.0, 2.0, -1.0]\n");

    // Split into train/test
    let (x_train, x_test, y_train, y_test) =
        train_test_split(&x, &y, 0.2, Some(42)).expect("Example data should be valid");

    println!("Train set: {} samples", x_train.shape().0);
    println!("Test set: {} samples\n", x_test.shape().0);

    // Feature scaling - important for regularized models
    println!("--- Feature Scaling ---");
    let mut scaler = StandardScaler::new();
    scaler.fit(&x_train).expect("Example data should be valid");
    let x_train_scaled = scaler
        .transform(&x_train)
        .expect("Example data should be valid");
    let x_test_scaled = scaler
        .transform(&x_test)
        .expect("Example data should be valid");
    println!("Applied StandardScaler to features\n");

    // 1. Ridge Regression (L2 regularization)
    println!("--- Ridge Regression (L2) ---");
    let alphas = vec![0.001, 0.01, 0.1, 1.0, 10.0];
    let kfold = KFold::new(5).with_random_state(42);

    let ridge_result = grid_search_alpha("ridge", &alphas, &x_train_scaled, &y_train, &kfold, None)
        .expect("Example data should be valid");

    println!("Grid search results:");
    for (alpha, score) in ridge_result.alphas.iter().zip(ridge_result.scores.iter()) {
        println!("  α = {alpha:.3}: R² = {score:.4}");
    }
    println!(
        "Best α = {:.3} (R² = {:.4})",
        ridge_result.best_alpha, ridge_result.best_score
    );

    let mut ridge = Ridge::new(ridge_result.best_alpha);
    ridge
        .fit(&x_train_scaled, &y_train)
        .expect("Example data should be valid");
    let ridge_test_score = ridge.score(&x_test_scaled, &y_test);
    println!("Test R² = {ridge_test_score:.4}\n");

    // 2. Lasso Regression (L1 regularization)
    println!("--- Lasso Regression (L1) ---");
    let lasso_result = grid_search_alpha("lasso", &alphas, &x_train_scaled, &y_train, &kfold, None)
        .expect("Example data should be valid");

    println!("Grid search results:");
    for (alpha, score) in lasso_result.alphas.iter().zip(lasso_result.scores.iter()) {
        println!("  α = {alpha:.3}: R² = {score:.4}");
    }
    println!(
        "Best α = {:.3} (R² = {:.4})",
        lasso_result.best_alpha, lasso_result.best_score
    );

    let mut lasso = Lasso::new(lasso_result.best_alpha);
    lasso
        .fit(&x_train_scaled, &y_train)
        .expect("Example data should be valid");
    let lasso_test_score = lasso.score(&x_test_scaled, &y_test);
    println!("Test R² = {lasso_test_score:.4}\n");

    // 3. ElasticNet Regression (L1 + L2)
    println!("--- ElasticNet Regression (L1 + L2) ---");
    let l1_ratio = 0.5; // 50% L1, 50% L2

    let elastic_result = grid_search_alpha(
        "elastic_net",
        &alphas,
        &x_train_scaled,
        &y_train,
        &kfold,
        Some(l1_ratio),
    )
    .expect("Example data should be valid");

    println!("Grid search results (l1_ratio = {l1_ratio}):");
    for (alpha, score) in elastic_result
        .alphas
        .iter()
        .zip(elastic_result.scores.iter())
    {
        println!("  α = {alpha:.3}: R² = {score:.4}");
    }
    println!(
        "Best α = {:.3} (R² = {:.4})",
        elastic_result.best_alpha, elastic_result.best_score
    );

    let mut elastic = ElasticNet::new(elastic_result.best_alpha, l1_ratio);
    elastic
        .fit(&x_train_scaled, &y_train)
        .expect("Example data should be valid");
    let elastic_test_score = elastic.score(&x_test_scaled, &y_test);
    println!("Test R² = {elastic_test_score:.4}\n");

    // Comparison
    println!("=== Model Comparison ===");
    println!("Ridge:      Test R² = {ridge_test_score:.4}");
    println!("Lasso:      Test R² = {lasso_test_score:.4}");
    println!("ElasticNet: Test R² = {elastic_test_score:.4}");

    // Determine best model
    let best = if ridge_test_score >= lasso_test_score && ridge_test_score >= elastic_test_score {
        "Ridge"
    } else if lasso_test_score >= elastic_test_score {
        "Lasso"
    } else {
        "ElasticNet"
    };
    println!("\nBest model: {best}");

    println!("\n=== Key Insights ===");
    println!("- Ridge (L2): Shrinks coefficients, keeps all features");
    println!("- Lasso (L1): Can drive coefficients to exactly zero (feature selection)");
    println!("- ElasticNet: Combines L1 and L2 benefits");
    println!("- Feature scaling is crucial for regularized models");
    println!("- Grid search finds optimal regularization strength");
}
