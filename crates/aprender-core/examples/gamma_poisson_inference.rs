#![allow(clippy::disallowed_methods)]
//! Gamma-Poisson Bayesian Inference Example
//!
//! Demonstrates how to use conjugate priors for count data (Poisson-distributed events).
//! This example shows:
//! 1. Prior specification (noninformative, informative)
//! 2. Sequential Bayesian updating
//! 3. Posterior statistics and credible intervals
//! 4. Real-world scenarios: Call center analysis, quality control, server monitoring

use aprender::prelude::*;

fn main() {
    println!("Gamma-Poisson Bayesian Inference");
    println!("=================================\n");

    // Example 1: Call center arrivals
    println!("Example 1: Call Center Analysis");
    println!("--------------------------------");
    call_center_example();

    println!("\nExample 2: Quality Control");
    println!("--------------------------");
    quality_control_example();

    println!("\nExample 3: Sequential Learning");
    println!("------------------------------");
    sequential_learning_example();

    println!("\nExample 4: Prior Comparison");
    println!("---------------------------");
    prior_comparison_example();
}

fn call_center_example() {
    // Analyze hourly call arrival rates at a call center
    // Data: Number of calls received each hour over a 10-hour period
    let hourly_calls = vec![3, 5, 4, 6, 2, 4, 5, 3, 4, 4];

    // Start with noninformative prior
    let mut model = GammaPoisson::noninformative();
    println!("Prior: Gamma({:.3}, {:.3})", model.alpha(), model.beta());
    println!("  Prior mean rate: {:.4}", model.posterior_mean());

    // Update with observed call counts
    model.update(&hourly_calls);

    println!(
        "\nAfter observing {} hours of call data:",
        hourly_calls.len()
    );
    println!(
        "Posterior: Gamma({:.3}, {:.3})",
        model.alpha(),
        model.beta()
    );
    println!(
        "  Posterior mean rate: {:.4} calls/hour",
        model.posterior_mean()
    );
    println!(
        "  Posterior mode: {:.4} calls/hour",
        model.posterior_mode().expect("Mode exists")
    );
    println!("  Posterior variance: {:.6}", model.posterior_variance());

    // 95% credible interval
    let (lower, upper) = model
        .credible_interval(0.95)
        .expect("Valid confidence level");
    println!("  95% credible interval: [{lower:.4}, {upper:.4}] calls/hour");

    // Predictive probability for next hour
    let predicted_rate = model.posterior_predictive();
    println!("  Expected calls next hour: {predicted_rate:.4}");

    println!("\nInterpretation:");
    println!(
        "  The call center receives approximately {:.1} calls per hour.",
        model.posterior_mean()
    );
    println!(
        "  We are 95% confident the true rate is between {lower:.2} and {upper:.2} calls/hour."
    );
}

fn quality_control_example() {
    // Manufacturing quality control: defects per batch
    // Company A: 12 defects over 100 batches (0.12 defects/batch)
    // Company B: 45 defects over 100 batches (0.45 defects/batch)

    // Simulate defect counts for Company A (low defect rate)
    let company_a_defects = vec![
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, // 20 batches
    ];

    let mut model_a = GammaPoisson::noninformative();
    model_a.update(&company_a_defects);
    let mean_a = model_a.posterior_mean();
    let (lower_a, upper_a) = model_a.credible_interval(0.95).expect("Valid confidence");

    println!(
        "Company A: {} defects in {} batches",
        company_a_defects.iter().sum::<u32>(),
        company_a_defects.len()
    );
    println!("  Defect rate: {mean_a:.4} defects/batch");
    println!("  95% credible interval: [{lower_a:.4}, {upper_a:.4}]");

    // Simulate defect counts for Company B (higher defect rate)
    let company_b_defects = vec![
        1, 0, 2, 1, 1, 0, 1, 1, 0, 1, 1, 2, 0, 1, 1, 0, 1, 0, 1, 1, // 20 batches
    ];

    let mut model_b = GammaPoisson::noninformative();
    model_b.update(&company_b_defects);
    let mean_b = model_b.posterior_mean();
    let (lower_b, upper_b) = model_b.credible_interval(0.95).expect("Valid confidence");

    println!(
        "\nCompany B: {} defects in {} batches",
        company_b_defects.iter().sum::<u32>(),
        company_b_defects.len()
    );
    println!("  Defect rate: {mean_b:.4} defects/batch");
    println!("  95% credible interval: [{lower_b:.4}, {upper_b:.4}]");

    println!("\nConclusion:");
    if lower_b > upper_a {
        println!("  ✓ Company B has significantly higher defect rate (95% confidence)");
        println!("    Company A is the better supplier.");
    } else if lower_a > upper_b {
        println!("  ✓ Company A has significantly higher defect rate (95% confidence)");
        println!("    Company B is the better supplier.");
    } else {
        println!("  ⚠ Credible intervals overlap - no clear difference");
        println!("    Consider testing more batches from each company.");
    }
}

fn sequential_learning_example() {
    // Server monitoring: HTTP requests per minute
    // Demonstrates how uncertainty decreases as we collect more data
    let mut model = GammaPoisson::noninformative();

    let experiments = vec![
        vec![8, 12, 10, 11, 9],              // 5 minutes: mean ≈ 10
        vec![9, 11, 10, 12, 8],              // 5 more minutes
        vec![10, 9, 11, 10, 10],             // 5 more minutes
        vec![11, 10, 9, 10, 11, 10, 9],      // 7 more minutes
        vec![10, 11, 10, 9, 10, 11, 10, 10], // 8 more minutes
    ];

    println!("Sequential updates (true rate ≈ 10 requests/min):\n");
    println!("  Minutes | Total Events | Mean  | Variance  | 95% CI Width");
    println!("  --------|--------------|-------|-----------|-------------");

    let mut total_minutes = 0;
    for batch in experiments {
        let batch_u32: Vec<u32> = batch.clone();
        model.update(&batch_u32);
        total_minutes += batch.len();

        let mean = model.posterior_mean();
        let variance = model.posterior_variance();
        let (lower, upper) = model.credible_interval(0.95).expect("Valid confidence");
        let width = upper - lower;

        let total_events: u32 = batch_u32.iter().sum();
        println!(
            "  {total_minutes:7} | {total_events:12} | {mean:.3} | {variance:.7} | {width:.4}"
        );
    }

    println!("\nObservation: Variance and CI width decrease as data accumulates.");
    println!("  The posterior mean converges to the true rate (~10 requests/min).");
}

fn prior_comparison_example() {
    // Compare different priors with same data
    // Data: Event counts [3, 5, 4, 6, 2] over 5 time intervals

    let counts = vec![3, 5, 4, 6, 2];

    println!("Effect of prior choice (data: {counts:?}):\n");

    // Noninformative prior Gamma(0.001, 0.001)
    let mut noninformative = GammaPoisson::noninformative();
    noninformative.update(&counts);
    println!("Noninformative Prior Gamma(0.001, 0.001):");
    println!("  Posterior mean: {:.4}", noninformative.posterior_mean());
    println!(
        "  Posterior: Gamma({:.3}, {:.3})",
        noninformative.alpha(),
        noninformative.beta()
    );

    // Weakly informative prior Gamma(1, 1) [mean = 1]
    let mut weak = GammaPoisson::new(1.0, 1.0).expect("Valid parameters");
    weak.update(&counts);
    println!("\nWeakly Informative Prior Gamma(1, 1) [mean = 1]:");
    println!("  Posterior mean: {:.4}", weak.posterior_mean());
    println!(
        "  Posterior: Gamma({:.3}, {:.3})",
        weak.alpha(),
        weak.beta()
    );

    // Informative prior Gamma(50, 10) [mean = 5, strong belief]
    let mut informative = GammaPoisson::new(50.0, 10.0).expect("Valid parameters");
    informative.update(&counts);
    println!("\nInformative Prior Gamma(50, 10) [mean = 5, strong belief]:");
    println!("  Posterior mean: {:.4}", informative.posterior_mean());
    println!(
        "  Posterior: Gamma({:.3}, {:.3})",
        informative.alpha(),
        informative.beta()
    );

    println!("\nObservation: Strong priors pull the posterior toward prior belief.");
    println!("  Noninformative and weak priors let the data dominate.");
    println!("  With more data, all priors converge to the same posterior.");
}
