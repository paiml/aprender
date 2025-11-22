//! Normal-InverseGamma Bayesian Inference Example
//!
//! Demonstrates how to use conjugate priors for normal data with unknown mean and variance.
//! This example shows:
//! 1. Prior specification (noninformative, informative)
//! 2. Sequential Bayesian updating for continuous data
//! 3. Posterior statistics for both mean and variance
//! 4. Real-world scenarios: Manufacturing quality control, medical data analysis

use aprender::prelude::*;

fn main() {
    println!("Normal-InverseGamma Bayesian Inference");
    println!("=======================================\n");

    // Example 1: Manufacturing quality control
    println!("Example 1: Manufacturing Quality Control");
    println!("----------------------------------------");
    manufacturing_example();

    println!("\nExample 2: Medical Data Analysis");
    println!("--------------------------------");
    medical_example();

    println!("\nExample 3: Sequential Learning");
    println!("------------------------------");
    sequential_learning_example();

    println!("\nExample 4: Prior Comparison");
    println!("---------------------------");
    prior_comparison_example();
}

fn manufacturing_example() {
    // Manufacturing: Measure diameter of machined parts (target: 10.0mm)
    // Want to estimate both mean diameter and manufacturing variance
    let part_diameters = vec![
        9.98, 10.02, 9.97, 10.03, 10.01, 9.99, 10.04, 9.96, 10.00, 10.02,
    ];

    // Start with weakly informative prior
    // μ₀ = 10.0 (target diameter), κ₀ = 1.0 (low confidence in prior mean)
    // α₀ = 3.0, β₀ = 0.02 (weakly informative for variance)
    let mut model = NormalInverseGamma::new(10.0, 1.0, 3.0, 0.02).expect("Valid parameters");

    println!("Prior:");
    println!("  μ₀ = {:.4} mm (prior mean)", 10.0);
    println!("  E[σ²] = {:.6} mm² (prior variance)", 0.02 / (3.0 - 1.0));

    // Update with observed measurements
    model.update(&part_diameters);

    let mean_mu = model.posterior_mean_mu();
    let mean_var = model.posterior_mean_variance().expect("Variance exists");
    let std_dev = mean_var.sqrt();

    println!("\nAfter measuring {} parts:", part_diameters.len());
    println!("  E[μ|D] = {mean_mu:.4} mm (posterior mean diameter)");
    println!("  E[σ²|D] = {mean_var:.6} mm² (posterior variance)");
    println!("  E[σ|D] ≈ {std_dev:.4} mm (posterior std deviation)");

    // 95% credible interval for mean
    let (lower, upper) = model
        .credible_interval_mu(0.95)
        .expect("Valid confidence level");
    println!("  95% CI for μ: [{lower:.4}, {upper:.4}] mm");

    println!("\nInterpretation:");
    let target = 10.0;
    if lower <= target && target <= upper {
        println!("  ✓ Process is on-target: {target:.4} mm is within 95% CI");
        println!("    Standard deviation of {std_dev:.4} mm indicates good precision.");
    } else {
        println!("  ⚠ Process may be off-target: {target:.4} mm is outside 95% CI");
        println!("    Recommend process recalibration.");
    }
}

fn medical_example() {
    // Medical: Patient blood pressure readings (systolic BP in mmHg)
    // Compare two patients to detect significant differences

    // Patient A: Stable blood pressure
    let patient_a_readings = vec![118.0, 122.0, 120.0, 119.0, 121.0, 120.0, 118.0, 122.0];

    let mut model_a = NormalInverseGamma::noninformative();
    model_a.update(&patient_a_readings);

    let mean_a = model_a.posterior_mean_mu();
    let (lower_a, upper_a) = model_a
        .credible_interval_mu(0.95)
        .expect("Valid confidence");
    let var_a = model_a.posterior_mean_variance().expect("Variance exists");

    println!("Patient A: {} readings", patient_a_readings.len());
    println!("  Mean BP: {mean_a:.2} mmHg");
    println!("  95% CI: [{lower_a:.2}, {upper_a:.2}] mmHg");
    println!("  Variance: {var_a:.4} mmHg²");

    // Patient B: More variable blood pressure
    let patient_b_readings = vec![135.0, 142.0, 138.0, 145.0, 140.0, 137.0, 143.0, 139.0];

    let mut model_b = NormalInverseGamma::noninformative();
    model_b.update(&patient_b_readings);

    let mean_b = model_b.posterior_mean_mu();
    let (lower_b, upper_b) = model_b
        .credible_interval_mu(0.95)
        .expect("Valid confidence");
    let var_b = model_b.posterior_mean_variance().expect("Variance exists");

    println!("\nPatient B: {} readings", patient_b_readings.len());
    println!("  Mean BP: {mean_b:.2} mmHg");
    println!("  95% CI: [{lower_b:.2}, {upper_b:.2}] mmHg");
    println!("  Variance: {var_b:.4} mmHg²");

    println!("\nConclusion:");
    if lower_b > upper_a {
        println!("  ⚠ Patient B has significantly higher BP than Patient A (95% confidence)");
        println!("    Recommend medical intervention for Patient B.");
    } else if lower_a > upper_b {
        println!("  Patient A has significantly higher BP than Patient B (95% confidence)");
    } else {
        println!("  Credible intervals overlap - BP levels not significantly different");
    }

    // Compare variability
    println!("\nVariability analysis:");
    if var_b > 2.0 * var_a {
        println!(
            "  ⚠ Patient B shows {:.1}x higher BP variability than Patient A",
            var_b / var_a
        );
        println!("    High variability may indicate cardiovascular instability.");
    } else {
        println!("  Both patients show similar BP variability.");
    }
}

fn sequential_learning_example() {
    // Sensor calibration: Temperature sensor readings (°C)
    // True temperature: 25.0°C, sensor has some bias and noise
    let mut model = NormalInverseGamma::noninformative();

    let experiments = vec![
        vec![25.2, 24.8, 25.1, 24.9, 25.0],                   // 5 readings
        vec![25.3, 24.7, 25.2, 24.8, 25.1],                   // 5 more
        vec![25.0, 25.1, 24.9, 25.2, 24.8, 25.0],             // 6 more
        vec![25.1, 24.9, 25.0, 25.2, 24.8, 25.1, 25.0],       // 7 more
        vec![25.0, 25.1, 24.9, 25.0, 25.2, 24.8, 25.1, 25.0], // 8 more
    ];

    println!("Sequential sensor calibration (true temp = 25.0°C):\n");
    println!("  Readings | Mean (°C) | Var(μ) | Var(σ²) | 95% CI Width (°C)");
    println!("  ---------|-----------|--------|---------|------------------");

    let mut total_readings = 0;
    for batch in experiments {
        model.update(&batch);
        total_readings += batch.len();

        let mean = model.posterior_mean_mu();
        let var_mu = model
            .posterior_variance_mu()
            .expect("Variance of mean exists");
        let var_sigma2 = model.posterior_mean_variance().expect("Variance exists");
        let (lower, upper) = model.credible_interval_mu(0.95).expect("Valid confidence");
        let width = upper - lower;

        println!(
            "  {total_readings:8} | {mean:9.4} | {var_mu:6.4} | {var_sigma2:7.4} | {width:16.4}"
        );
    }

    println!("\nObservation: Uncertainty in mean (Var(μ)) and CI width decrease with more data.");
    println!("  Posterior mean converges to true temperature (25.0°C).");
}

fn prior_comparison_example() {
    // Compare different priors with same data
    // Data: Temperature measurements [22.1, 22.5, 22.3, 22.7, 22.4]

    let measurements = vec![22.1, 22.5, 22.3, 22.7, 22.4];

    println!("Effect of prior choice (data: {measurements:?}):\n");

    // Noninformative prior
    let mut noninformative = NormalInverseGamma::noninformative();
    noninformative.update(&measurements);
    let mean_ni = noninformative.posterior_mean_mu();
    let var_ni = noninformative
        .posterior_mean_variance()
        .expect("Variance exists");
    println!("Noninformative Prior NIG(0, 1, 1, 1):");
    println!("  Posterior E[μ] = {mean_ni:.4}°C");
    println!("  Posterior E[σ²] = {var_ni:.6}°C²");

    // Weakly informative prior (prior belief: μ ≈ 22°C, σ² ≈ 1)
    let mut weak = NormalInverseGamma::new(22.0, 1.0, 3.0, 2.0).expect("Valid parameters");
    weak.update(&measurements);
    let mean_weak = weak.posterior_mean_mu();
    let var_weak = weak.posterior_mean_variance().expect("Variance exists");
    println!("\nWeakly Informative Prior NIG(22, 1, 3, 2) [μ ≈ 22, σ² ≈ 1]:");
    println!("  Posterior E[μ] = {mean_weak:.4}°C");
    println!("  Posterior E[σ²] = {var_weak:.6}°C²");

    // Informative prior (strong belief: μ = 20°C, σ² ≈ 0.5)
    let mut informative = NormalInverseGamma::new(20.0, 10.0, 10.0, 5.0).expect("Valid parameters");
    informative.update(&measurements);
    let mean_inf = informative.posterior_mean_mu();
    let var_inf = informative
        .posterior_mean_variance()
        .expect("Variance exists");
    println!("\nInformative Prior NIG(20, 10, 10, 5) [strong μ = 20, σ² ≈ 0.56]:");
    println!("  Posterior E[μ] = {mean_inf:.4}°C");
    println!("  Posterior E[σ²] = {var_inf:.6}°C²");

    println!("\nObservation: Strong priors pull the posterior toward prior beliefs.");
    println!("  Noninformative prior: posterior dominated by data (E[μ] ≈ {mean_ni:.2}°C)");
    println!(
        "  Informative prior: posterior influenced by prior (E[μ] = {mean_inf:.2}°C vs data mean ≈ 22.4°C)"
    );
    println!("  With more data, all priors converge to the same posterior.");
}
