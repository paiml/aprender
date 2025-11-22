//! Bayesian Logistic Regression with Laplace Approximation
//!
//! Demonstrates Bayesian logistic regression using the Laplace approximation,
//! showing MAP estimation, uncertainty quantification, and comparison with MLE.
//!
//! # Run
//!
//! ```bash
//! cargo run --example bayesian_logistic_regression
//! ```

use aprender::bayesian::BayesianLogisticRegression;
use aprender::primitives::{Matrix, Vector};

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║ Bayesian Logistic Regression with Laplace Approximation      ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Example 1: Binary classification with weak prior
    example_1_binary_classification();

    println!("\n{}", "═".repeat(64));

    // Example 2: Uncertainty quantification
    example_2_uncertainty_quantification();

    println!("\n{}", "═".repeat(64));

    // Example 3: Effect of prior regularization
    example_3_prior_regularization();
}

/// Example 1: Simple binary classification
fn example_1_binary_classification() {
    println!("EXAMPLE 1: Binary Classification with Weak Prior");
    println!("{}", "─".repeat(64));

    // Generate linearly separable data: y = 1 if x > 0, else 0
    let x = Matrix::from_vec(8, 1, vec![-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0])
        .expect("Valid matrix dimensions");
    let y = Vector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

    println!("\n📊 Training data (8 samples):");
    println!("   Binary classification: y = 1 if x > 0, else 0");
    for i in 0..4 {
        println!("   x[{}] = {:.1}, y[{}] = {:.0}", i, x.get(i, 0), i, y[i]);
    }
    println!("   ...");

    // Create model with weak prior (small precision = weak regularization)
    let mut model = BayesianLogisticRegression::new(0.1);

    println!("\n🔧 Model configuration:");
    println!("   Prior: β ~ N(0, 10I) (weak prior, precision λ = 0.1)");
    println!("   Inference: Laplace approximation (Gaussian at MAP)");
    println!("   Optimization: Gradient ascent for MAP estimation");

    // Fit model
    model.fit(&x, &y).expect("Fit should succeed");

    println!("\n📈 MAP estimate:");
    let beta = model.coefficients_map().expect("MAP estimate should exist");

    println!("   β (coefficient): {:.4}", beta[0]);
    println!("   → Positive coefficient indicates positive correlation");

    // Make predictions
    let x_test =
        Matrix::from_vec(5, 1, vec![-2.0, -1.0, 0.0, 1.0, 2.0]).expect("Valid test matrix");
    let probas = model
        .predict_proba(&x_test)
        .expect("Prediction should succeed");

    println!("\n🔮 Predictions (probability of class 1):");
    for i in 0..5 {
        let x_val = x_test.get(i, 0);
        let proba = probas[i];
        let label = i32::from(proba >= 0.5);
        println!("   x = {x_val:.1} → P(y=1|x) = {proba:.4}, predicted class = {label}");
    }

    println!("\n✓ Model correctly separates classes with smooth probabilities");
}

/// Example 2: Uncertainty quantification with credible intervals
fn example_2_uncertainty_quantification() {
    println!("EXAMPLE 2: Uncertainty Quantification");
    println!("{}", "─".repeat(64));

    // Small dataset to show higher uncertainty
    let x = Matrix::from_vec(6, 1, vec![-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]).expect("Valid matrix");
    let y = Vector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

    println!("\n📊 Training data (6 samples):");
    println!("   Small dataset → higher posterior uncertainty");

    let mut model = BayesianLogisticRegression::new(0.1);
    model.fit(&x, &y).expect("Fit succeeds");

    // Predict with uncertainty
    let x_test = Matrix::from_vec(4, 1, vec![-2.0, -0.5, 0.5, 2.0]).expect("Valid test matrix");

    let probas = model.predict_proba(&x_test).expect("Predictions succeed");
    let (lower, upper) = model
        .predict_proba_interval(&x_test, 0.95)
        .expect("Interval predictions succeed");

    println!("\n🔮 Predictions with 95% credible intervals:");
    println!(
        "   {:>6}  {:>10}  {:>12}  {:>6}",
        "x", "P(y=1|x)", "[95% CI]", "Width"
    );
    println!("   {}", "─".repeat(50));

    for i in 0..4 {
        let x_val = x_test.get(i, 0);
        let width = upper[i] - lower[i];
        println!(
            "   {:>6.1}  {:>10.4}  [{:.4}, {:.4}]  {:.4}",
            x_val, probas[i], lower[i], upper[i], width
        );
    }

    println!("\n💡 Interpretation:");
    println!("   - Credible intervals quantify prediction uncertainty");
    println!("   - Narrower intervals at extreme x values (more certain)");
    println!("   - Wider intervals near decision boundary (less certain)");
    println!("   - All point estimates lie within their credible intervals");
}

/// Example 3: Effect of prior regularization
fn example_3_prior_regularization() {
    println!("EXAMPLE 3: Prior Regularization (Ridge-like)");
    println!("{}", "─".repeat(64));

    // Very small dataset prone to overfitting
    let x = Matrix::from_vec(4, 1, vec![-1.0, -0.3, 0.3, 1.0]).expect("Valid matrix");
    let y = Vector::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

    println!("\n📊 Training data (4 samples):");
    println!("   Tiny dataset → regularization is crucial");

    // Weak prior (low regularization)
    let mut weak_model = BayesianLogisticRegression::new(0.1).with_max_iter(2000); // More iterations for convergence
    weak_model.fit(&x, &y).expect("Fit weak model");

    // Strong prior (high regularization)
    let mut strong_model = BayesianLogisticRegression::new(2.0);
    strong_model.fit(&x, &y).expect("Fit strong model");

    println!("\n📈 MAP estimates:");
    let beta_weak = weak_model
        .coefficients_map()
        .expect("Weak model coefficient");
    let beta_strong = strong_model
        .coefficients_map()
        .expect("Strong model coefficient");

    println!("   Weak prior (λ=0.1):   β = {:.4}", beta_weak[0]);
    println!("   Strong prior (λ=2.0): β = {:.4}", beta_strong[0]);
    println!("   → Strong prior shrinks β toward 0 (regularization)");

    // Compare predictions
    let x_test = Matrix::from_vec(2, 1, vec![-2.0, 2.0]).expect("Valid test matrix");

    let probas_weak = weak_model.predict_proba(&x_test).expect("Weak predictions");
    let probas_strong = strong_model
        .predict_proba(&x_test)
        .expect("Strong predictions");

    println!("\n🔮 Predictions at extreme values:");
    println!(
        "   {:>6}  {:>15}  {:>15}",
        "x", "Weak Prior", "Strong Prior"
    );
    println!("   {}", "─".repeat(40));

    for i in 0..2 {
        let x_val = x_test.get(i, 0);
        println!(
            "   {:>6.1}  {:>15.4}  {:>15.4}",
            x_val, probas_weak[i], probas_strong[i]
        );
    }

    println!("\n💡 Bayesian interpretation:");
    println!("   - Prior precision λ controls regularization strength");
    println!("   - High λ → strong shrinkage toward zero (like ridge)");
    println!("   - Low λ → weak regularization (closer to MLE)");
    println!("   - Optimal λ balances prior belief and data evidence");

    println!("\n✓ Laplace approximation provides fast Bayesian inference");
}
