#![allow(clippy::disallowed_methods)]
//! Bayesian Linear Regression with Analytical Posterior
//!
//! Demonstrates Bayesian linear regression with conjugate Normal-InverseGamma prior,
//! showing uncertainty quantification and comparison with OLS.
//!
//! # Run
//!
//! ```bash
//! cargo run --example bayesian_linear_regression
//! ```

use aprender::bayesian::BayesianLinearRegression;
use aprender::primitives::{Matrix, Vector};

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║ Bayesian Linear Regression with Analytical Posterior          ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Example 1: Simple linear regression with weak prior
    example_1_weak_prior();

    println!("\n{}", "═".repeat(64));

    // Example 2: Linear regression with informative prior
    example_2_informative_prior();

    println!("\n{}", "═".repeat(64));

    // Example 3: Multivariate regression
    example_3_multivariate();
}

/// Example 1: Simple univariate regression with weak (noninformative) prior
fn example_1_weak_prior() {
    println!("EXAMPLE 1: Univariate Regression with Weak Prior");
    println!("{}", "─".repeat(64));

    // Generate synthetic data: y = 2x + noise
    let x = Matrix::from_vec(
        10,
        1,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .expect("Valid matrix dimensions");
    let y = Vector::from_vec(vec![2.1, 3.9, 6.2, 8.1, 9.8, 12.3, 13.9, 16.1, 18.2, 20.0]);

    println!("\n📊 Training data (10 samples):");
    println!("   True relationship: y ≈ 2x + noise");
    for i in 0..5 {
        println!("   x[{}] = {:.1}, y[{}] = {:.1}", i, x.get(i, 0), i, y[i]);
    }
    println!("   ...");

    // Create model with weak prior
    let mut model = BayesianLinearRegression::new(1);

    println!("\n🔧 Model configuration:");
    println!("   Prior: β ~ N(0, 10000I) (weak/noninformative)");
    println!("   Noise prior: σ² ~ InvGamma(0.001, 0.001)");

    // Fit model
    model.fit(&x, &y).expect("Fit should succeed");

    println!("\n📈 Posterior estimates:");
    let beta = model.posterior_mean().expect("Posterior mean should exist");
    let sigma2 = model.noise_variance().expect("Noise variance should exist");

    println!("   β (slope): {:.4}", beta[0]);
    println!("   σ² (noise variance): {sigma2:.4}");

    // Make predictions
    let x_test = Matrix::from_vec(3, 1, vec![11.0, 12.0, 13.0]).expect("Valid test matrix");
    let predictions = model.predict(&x_test).expect("Prediction should succeed");

    println!("\n🔮 Predictions on new data:");
    for i in 0..3 {
        println!(
            "   x = {:.1} → E[y|x] = {:.2}",
            x_test.get(i, 0),
            predictions[i]
        );
    }

    println!("\n✓ With weak prior, posterior ≈ OLS (Maximum Likelihood)");
}

/// Example 2: Regression with informative prior (ridge-like regularization)
fn example_2_informative_prior() {
    println!("EXAMPLE 2: Univariate Regression with Informative Prior");
    println!("{}", "─".repeat(64));

    // Small dataset with potential overfitting
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![2.5, 4.1, 5.8, 8.2, 9.9]);

    println!("\n📊 Training data (5 samples):");
    println!("   Small dataset → regularization helpful");
    for i in 0..5 {
        println!("   x[{}] = {:.1}, y[{}] = {:.1}", i, x.get(i, 0), i, y[i]);
    }

    // Create models: weak prior vs. strong prior
    let mut weak_model = BayesianLinearRegression::new(1);
    let mut strong_model = BayesianLinearRegression::with_prior(
        1,
        vec![1.5], // Prior belief: slope around 1.5
        1.0,       // Moderate precision (variance = 1.0)
        3.0,       // Shape for noise prior
        2.0,       // Scale for noise prior
    )
    .expect("Valid prior parameters");

    weak_model.fit(&x, &y).expect("Fit weak model");
    strong_model.fit(&x, &y).expect("Fit strong model");

    println!("\n📈 Posterior comparison:");
    let beta_weak = weak_model.posterior_mean().expect("Weak posterior exists");
    let beta_strong = strong_model
        .posterior_mean()
        .expect("Strong posterior exists");

    println!("   Weak prior:       β = {:.4}", beta_weak[0]);
    println!(
        "   Informative prior: β = {:.4} (shrunk toward 1.5)",
        beta_strong[0]
    );

    println!("\n💡 Informative prior acts as regularization:");
    println!("   - Shrinks coefficients toward prior mean");
    println!("   - Reduces overfitting on small datasets");
    println!("   - Equivalent to ridge regression (L2 penalty)");
}

/// Example 3: Multivariate regression
fn example_3_multivariate() {
    println!("EXAMPLE 3: Multivariate Regression (2 features)");
    println!("{}", "─".repeat(64));

    // Generate data: y = 2x₁ + 3x₂ + noise
    let x = Matrix::from_vec(
        8,
        2,
        vec![
            1.0, 1.0, // row 0
            2.0, 1.0, // row 1
            3.0, 2.0, // row 2
            4.0, 2.0, // row 3
            5.0, 3.0, // row 4
            6.0, 3.0, // row 5
            7.0, 4.0, // row 6
            8.0, 4.0, // row 7
        ],
    )
    .expect("Valid matrix");

    let y = Vector::from_vec(vec![5.1, 7.2, 11.9, 14.1, 19.2, 21.0, 25.8, 27.9]);

    println!("\n📊 Training data (8 samples, 2 features):");
    println!("   True relationship: y ≈ 2x₁ + 3x₂ + noise");
    println!("   Sample 0: x₁={:.1}, x₂={:.1} → y={:.1}", 1.0, 1.0, 5.1);
    println!("   Sample 1: x₁={:.1}, x₂={:.1} → y={:.1}", 2.0, 1.0, 7.2);
    println!("   ...");

    // Fit model
    let mut model = BayesianLinearRegression::new(2);
    model.fit(&x, &y).expect("Fit multivariate model");

    println!("\n📈 Posterior estimates:");
    let beta = model.posterior_mean().expect("Posterior mean should exist");
    let sigma2 = model.noise_variance().expect("Noise variance should exist");

    println!("   β₁ (coefficient for x₁): {:.4}", beta[0]);
    println!("   β₂ (coefficient for x₂): {:.4}", beta[1]);
    println!("   σ² (noise variance):     {sigma2:.4}");

    // Make predictions
    let x_test = Matrix::from_vec(
        3,
        2,
        vec![
            9.0, 5.0, // Should predict ≈ 2*9 + 3*5 = 33
            10.0, 5.0, // Should predict ≈ 2*10 + 3*5 = 35
            10.0, 6.0, // Should predict ≈ 2*10 + 3*6 = 38
        ],
    )
    .expect("Valid test matrix");

    let predictions = model.predict(&x_test).expect("Predictions should succeed");

    println!("\n🔮 Predictions:");
    for i in 0..3 {
        println!(
            "   x₁={:.1}, x₂={:.1} → E[y|x] = {:.2}",
            x_test.get(i, 0),
            x_test.get(i, 1),
            predictions[i]
        );
    }

    println!("\n✓ Bayesian approach naturally handles multiple features");
}
