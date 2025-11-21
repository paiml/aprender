//! Grid Search Hyperparameter Tuning Example
//!
//! Demonstrates grid search for finding optimal regularization parameters
//! using cross-validation. Shows how to tune Ridge, Lasso, and ElasticNet
//! regression models.

use aprender::linear_model::{ElasticNet, Lasso, Ridge};
use aprender::model_selection::{grid_search_alpha, train_test_split, KFold};
use aprender::prelude::*;
use aprender::primitives::{Matrix, Vector};

fn main() {
    println!("Grid Search Hyperparameter Tuning");
    println!("==================================\n");

    // Generate synthetic regression data
    let (x, y) = generate_regression_data();

    // Split into train and test sets
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, Some(42)).unwrap();

    println!("Dataset:");
    println!("  Training samples: {}", x_train.shape().0);
    println!("  Test samples:     {}", x_test.shape().0);
    println!("  Features:         {}\n", x_train.shape().1);

    // Example 1: Ridge Regression Grid Search
    println!("Example 1: Ridge Regression Alpha Tuning");
    println!("-----------------------------------------");
    ridge_grid_search_example(&x_train, &y_train, &x_test, &y_test);

    // Example 2: Lasso Regression Grid Search
    println!("\nExample 2: Lasso Regression Alpha Tuning");
    println!("-----------------------------------------");
    lasso_grid_search_example(&x_train, &y_train, &x_test, &y_test);

    // Example 3: ElasticNet with L1 Ratio Tuning
    println!("\nExample 3: ElasticNet Alpha and L1 Ratio Tuning");
    println!("------------------------------------------------");
    elasticnet_grid_search_example(&x_train, &y_train, &x_test, &y_test);

    // Example 4: Visualizing Grid Search Results
    println!("\nExample 4: Visualizing Alpha vs Score");
    println!("--------------------------------------");
    visualize_grid_search_results(&x_train, &y_train);

    // Example 5: Default vs Optimized Comparison
    println!("\nExample 5: Default vs Optimized Parameters");
    println!("-------------------------------------------");
    compare_default_vs_optimized(&x_train, &y_train, &x_test, &y_test);

    println!("\n✅ Grid Search Examples Complete!");
    println!("\nKey Takeaways:");
    println!("  • Grid search finds optimal hyperparameters via CV");
    println!("  • Use K-Fold cross-validation to avoid overfitting");
    println!("  • Ridge works well for many correlated features");
    println!("  • Lasso performs feature selection (sparse solutions)");
    println!("  • ElasticNet combines Ridge and Lasso benefits");
    println!("  • Visualize alpha curves to understand behavior");
    println!("  • Always evaluate final model on held-out test set");
}

fn generate_regression_data() -> (Matrix<f32>, Vector<f32>) {
    // Generate simple synthetic regression data
    let n_samples = 100;
    let n_features = 5;

    let mut x_data = Vec::with_capacity(n_samples * n_features);
    let mut y_data = Vec::with_capacity(n_samples);

    // True coefficients (sparse - only first 3 features matter)
    let true_coef = [3.0, -2.0, 1.5, 0.0, 0.0];

    for i in 0..n_samples {
        let mut features = Vec::new();
        for j in 0..n_features {
            // Generate features: simple linear progression with variation
            let val = (i as f32) * 0.1 + (j as f32) + ((i + j) % 5) as f32 * 0.5;
            features.push(val);
        }

        // Compute target with noise
        let mut y_val = 10.0; // Intercept
        for (j, &feat) in features.iter().enumerate() {
            y_val += true_coef[j] * feat;
        }
        // Add moderate noise
        y_val += ((i % 11) as f32 - 5.0) * 0.5;

        x_data.extend(features);
        y_data.push(y_val);
    }

    let x = Matrix::from_vec(n_samples, n_features, x_data).unwrap();
    let y = Vector::from_vec(y_data);

    (x, y)
}

fn ridge_grid_search_example(
    x_train: &Matrix<f32>,
    y_train: &Vector<f32>,
    x_test: &Matrix<f32>,
    y_test: &Vector<f32>,
) {
    // Define alpha grid (regularization strength)
    let alphas = vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0];

    // Setup K-Fold cross-validation
    let kfold = KFold::new(5).with_random_state(42);

    // Run grid search
    let result = grid_search_alpha("ridge", &alphas, x_train, y_train, &kfold, None).unwrap();

    println!("  Alpha Grid: {:?}", alphas);
    println!("\n  Cross-Validation Scores:");
    for (alpha, score) in result.alphas.iter().zip(result.scores.iter()) {
        println!("    α={:<8.3} → R²={:.4}", alpha, score);
    }

    println!("\n  Best Parameters:");
    println!("    α={:.3}", result.best_alpha);
    println!("    CV Score: {:.4}", result.best_score);

    // Train final model with best alpha
    let mut ridge = Ridge::new(result.best_alpha);
    ridge.fit(x_train, y_train).unwrap();

    let test_score = ridge.score(x_test, y_test);
    println!("\n  Test Set Performance:");
    println!("    R²={:.4}", test_score);
}

fn lasso_grid_search_example(
    x_train: &Matrix<f32>,
    y_train: &Vector<f32>,
    x_test: &Matrix<f32>,
    y_test: &Vector<f32>,
) {
    // Lasso needs smaller alphas typically
    let alphas = vec![0.0001, 0.001, 0.01, 0.1, 1.0, 10.0];

    let kfold = KFold::new(5).with_random_state(42);

    let result = grid_search_alpha("lasso", &alphas, x_train, y_train, &kfold, None).unwrap();

    println!("  Alpha Grid: {:?}", alphas);
    println!("\n  Cross-Validation Scores:");
    for (alpha, score) in result.alphas.iter().zip(result.scores.iter()) {
        println!("    α={:<8.4} → R²={:.4}", alpha, score);
    }

    println!("\n  Best Parameters:");
    println!("    α={:.4}", result.best_alpha);
    println!("    CV Score: {:.4}", result.best_score);

    // Train final model and check sparsity
    let mut lasso = Lasso::new(result.best_alpha);
    lasso.fit(x_train, y_train).unwrap();

    let test_score = lasso.score(x_test, y_test);
    let coef = lasso.coefficients();

    // Count non-zero coefficients (feature selection)
    let non_zero = coef.as_slice().iter().filter(|&&c| c.abs() > 1e-6).count();

    println!("\n  Test Set Performance:");
    println!("    R²={:.4}", test_score);
    println!(
        "    Non-zero coefficients: {}/{}  (sparse!)",
        non_zero,
        coef.len()
    );
}

fn elasticnet_grid_search_example(
    x_train: &Matrix<f32>,
    y_train: &Vector<f32>,
    x_test: &Matrix<f32>,
    y_test: &Vector<f32>,
) {
    // ElasticNet combines L1 and L2 penalties
    let alphas = vec![0.001, 0.01, 0.1, 1.0, 10.0];
    let l1_ratios = vec![0.25, 0.5, 0.75]; // Mix of Lasso and Ridge

    let kfold = KFold::new(5).with_random_state(42);

    println!("  Searching over:");
    println!("    α: {:?}", alphas);
    println!("    l1_ratio: {:?}", l1_ratios);

    let mut best_alpha = alphas[0];
    let mut best_l1_ratio = l1_ratios[0];
    let mut best_score = f32::NEG_INFINITY;

    println!("\n  Results:");
    println!("  {:>8} {:>10} {:>10}", "α", "l1_ratio", "CV Score");
    println!("  {}", "-".repeat(30));

    for &l1_ratio in &l1_ratios {
        let result = grid_search_alpha(
            "elastic_net",
            &alphas,
            x_train,
            y_train,
            &kfold,
            Some(l1_ratio),
        )
        .unwrap();

        for (alpha, score) in result.alphas.iter().zip(result.scores.iter()) {
            println!("  {:>8.3} {:>10.2} {:>10.4}", alpha, l1_ratio, score);

            if *score > best_score {
                best_score = *score;
                best_alpha = *alpha;
                best_l1_ratio = l1_ratio;
            }
        }
    }

    println!("\n  Best Parameters:");
    println!("    α={:.3}", best_alpha);
    println!("    l1_ratio={:.2}", best_l1_ratio);
    println!("    CV Score: {:.4}", best_score);

    // Train final model
    let mut enet = ElasticNet::new(best_alpha, best_l1_ratio);
    enet.fit(x_train, y_train).unwrap();

    let test_score = enet.score(x_test, y_test);
    println!("\n  Test Set Performance:");
    println!("    R²={:.4}", test_score);

    println!("\n  l1_ratio Guide:");
    println!("    0.00: Pure Ridge (L2 only)");
    println!("    0.50: Equal mix of Lasso and Ridge");
    println!("    1.00: Pure Lasso (L1 only)");
}

fn visualize_grid_search_results(x_train: &Matrix<f32>, y_train: &Vector<f32>) {
    let alphas = vec![0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0];
    let kfold = KFold::new(5).with_random_state(42);

    println!("  Comparing Ridge vs Lasso Alpha Curves:");
    println!("\n  {:>10} {:>15} {:>15}", "Alpha", "Ridge R²", "Lasso R²");
    println!("  {}", "-".repeat(42));

    let ridge_result = grid_search_alpha("ridge", &alphas, x_train, y_train, &kfold, None).unwrap();
    let lasso_result = grid_search_alpha("lasso", &alphas, x_train, y_train, &kfold, None).unwrap();

    for ((alpha, ridge_score), lasso_score) in alphas
        .iter()
        .zip(ridge_result.scores.iter())
        .zip(lasso_result.scores.iter())
    {
        println!(
            "  {:>10.4} {:>15.4} {:>15.4}",
            alpha, ridge_score, lasso_score
        );
    }

    println!("\n  Observations:");
    println!("    • Ridge: Gradual performance degradation");
    println!("    • Lasso: Sharp drop after optimal α");
    println!("    • Both: Too small α → overfitting");
    println!("    • Both: Too large α → underfitting");
}

fn compare_default_vs_optimized(
    x_train: &Matrix<f32>,
    y_train: &Vector<f32>,
    x_test: &Matrix<f32>,
    y_test: &Vector<f32>,
) {
    // Default Ridge (α=1.0)
    let mut ridge_default = Ridge::new(1.0);
    ridge_default.fit(x_train, y_train).unwrap();
    let default_score = ridge_default.score(x_test, y_test);

    // Optimized Ridge
    let alphas = vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0];
    let kfold = KFold::new(5).with_random_state(42);
    let result = grid_search_alpha("ridge", &alphas, x_train, y_train, &kfold, None).unwrap();

    let mut ridge_optimized = Ridge::new(result.best_alpha);
    ridge_optimized.fit(x_train, y_train).unwrap();
    let optimized_score = ridge_optimized.score(x_test, y_test);

    println!("  Ridge Regression Comparison:");
    println!("\n  Default (α=1.0):");
    println!("    Test R²: {:.4}", default_score);

    println!("\n  Grid Search Optimized (α={:.3}):", result.best_alpha);
    println!("    CV R²:   {:.4}", result.best_score);
    println!("    Test R²: {:.4}", optimized_score);

    let improvement = ((optimized_score - default_score) / default_score.abs()) * 100.0;
    println!("\n  Improvement: {:.2}%", improvement);

    if improvement > 0.0 {
        println!("  → Grid search found better hyperparameters! ✓");
    } else {
        println!("  → Default parameters were already near-optimal");
    }
}
