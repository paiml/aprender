//! Negative Binomial GLM Example
//!
//! Demonstrates the Negative Binomial family in aprender's GLM implementation.
//!
//! **CURRENT LIMITATION (v0.7.0)**: The Negative Binomial implementation uses
//! IRLS with step damping, which converges on simple linear data but may produce
//! suboptimal predictions. Future versions will implement more robust solvers
//! (L-BFGS, Newton-Raphson with line search) for better numerical stability.
//!
//! This example demonstrates the statistical concept and API, showing why
//! Negative Binomial is theoretically correct for overdispersed count data.

use aprender::glm::{Family, GLM};
use aprender::primitives::{Matrix, Vector};

fn main() {
    println!("=== Negative Binomial GLM for Overdispersed Count Data ===\n");

    // Example: Simple count data demonstration
    // X = Day, Y = Count
    // Note: This demonstrates the NB family with simple linear data
    // Real-world overdispersed data may require additional algorithmic improvements
    let days = Matrix::from_vec(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("Valid matrix");

    // Simple count data (gentle linear trend)
    let counts = Vector::from_vec(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

    // Calculate sample statistics to check for overdispersion
    let mean = counts.as_slice().iter().sum::<f32>() / counts.len() as f32;
    let variance = counts
        .as_slice()
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>()
        / (counts.len() - 1) as f32;

    println!("Sample Statistics:");
    println!("  Mean: {mean:.2}");
    println!("  Variance: {variance:.2}");
    println!("  Variance/Mean Ratio: {:.2}", variance / mean);
    println!(
        "  Overdispersion? {}",
        if variance > mean * 1.5 { "YES" } else { "NO" }
    );
    println!();

    // Fit Negative Binomial model with low dispersion
    println!("Fitting Negative Binomial GLM (α = 0.1)...");
    let mut nb_model = GLM::new(Family::NegativeBinomial)
        .with_dispersion(0.1)
        .with_max_iter(5000);

    match nb_model.fit(&days, &counts) {
        Ok(()) => {
            println!("  ✓ Model converged successfully!");
            println!(
                "  Intercept: {:.4}",
                nb_model.intercept().expect("Model fitted")
            );
            println!(
                "  Coefficient: {:.4}",
                nb_model.coefficients().expect("Model fitted")[0]
            );
            println!();

            // Make predictions
            println!("Predictions for each day:");
            let predictions = nb_model.predict(&days).expect("Predictions succeed");
            for (i, (&actual, &pred)) in counts
                .as_slice()
                .iter()
                .zip(predictions.as_slice())
                .enumerate()
            {
                println!(
                    "  Day {}: Actual = {:.0}, Predicted = {:.2}",
                    i + 1,
                    actual,
                    pred
                );
            }
            println!();
        }
        Err(e) => {
            println!("  ✗ Model failed to converge: {e}");
            println!();
        }
    }

    // Compare with different dispersion parameters
    println!("=== Effect of Dispersion Parameter α ===\n");

    for alpha in [0.05, 0.1, 0.2, 0.5] {
        let mut model = GLM::new(Family::NegativeBinomial)
            .with_dispersion(alpha)
            .with_max_iter(5000);

        match model.fit(&days, &counts) {
            Ok(()) => {
                println!("α = {alpha:.1}:");
                println!(
                    "  Intercept: {:.4}, Coefficient: {:.4}",
                    model.intercept().expect("Model fitted"),
                    model.coefficients().expect("Model fitted")[0]
                );

                // Variance function: V(μ) = μ + α*μ²
                let mean_pred = 7.5; // Approximate mean prediction
                let variance_func = mean_pred + alpha * mean_pred * mean_pred;
                println!("  Variance function V(μ) = μ + α*μ² ≈ {variance_func:.2}");
            }
            Err(_) => {
                println!("α = {alpha:.1}: Failed to converge");
            }
        }
    }
    println!();

    // Educational note
    println!("=== Why Negative Binomial? ===");
    println!();
    println!("Poisson Assumption:");
    println!("  - Assumes variance = mean (V(μ) = μ)");
    println!("  - Fails when data is overdispersed (variance >> mean)");
    println!("  - Can lead to underestimated uncertainty");
    println!();
    println!("Negative Binomial Solution:");
    println!("  - Allows variance > mean (V(μ) = μ + α*μ²)");
    println!("  - Dispersion parameter α controls extra variance");
    println!("  - Gamma-Poisson mixture model interpretation");
    println!("  - Provides accurate credible intervals");
    println!();
    println!("References:");
    println!("  - Cameron & Trivedi (2013): Regression Analysis of Count Data");
    println!("  - Hilbe (2011): Negative Binomial Regression");
    println!("  - See notes-poisson.md for detailed explanation");
}
