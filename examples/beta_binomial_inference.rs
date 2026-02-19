#![allow(clippy::disallowed_methods)]
//! Beta-Binomial Bayesian Inference Example
//!
//! Demonstrates how to use conjugate priors for Bernoulli/Binomial data.
//! This example shows:
//! 1. Prior specification (uniform, Jeffrey's, informative)
//! 2. Sequential Bayesian updating
//! 3. Posterior statistics and credible intervals
//! 4. Real-world scenario: A/B testing

use aprender::prelude::*;

fn main() {
    println!("Beta-Binomial Bayesian Inference");
    println!("=================================\n");

    // Example 1: Coin flip inference
    println!("Example 1: Coin Flip Inference");
    println!("-------------------------------");
    coin_flip_example();

    println!("\nExample 2: A/B Testing");
    println!("----------------------");
    ab_testing_example();

    println!("\nExample 3: Sequential Learning");
    println!("------------------------------");
    sequential_learning_example();

    println!("\nExample 4: Prior Comparison");
    println!("---------------------------");
    prior_comparison_example();
}

fn coin_flip_example() {
    // Start with uniform prior (complete ignorance)
    let mut model = BetaBinomial::uniform();
    println!("Prior: Beta({}, {})", model.alpha(), model.beta());
    println!("  Prior mean: {:.4}", model.posterior_mean());

    // Flip coin 10 times, observe 7 heads
    model.update(7, 10);
    println!("\nAfter observing 7 heads in 10 flips:");
    println!("Posterior: Beta({}, {})", model.alpha(), model.beta());
    println!("  Posterior mean: {:.4}", model.posterior_mean());
    println!(
        "  Posterior mode: {:.4}",
        model.posterior_mode().expect("Mode exists")
    );
    println!("  Posterior variance: {:.6}", model.posterior_variance());

    // 95% credible interval
    let (lower, upper) = model
        .credible_interval(0.95)
        .expect("Valid confidence level");
    println!("  95% credible interval: [{lower:.4}, {upper:.4}]");

    // Predictive probability
    let prob_heads = model.posterior_predictive();
    println!("  Probability of heads on next flip: {prob_heads:.4}");
}

fn ab_testing_example() {
    // A/B test: comparing two website variants

    // Variant A: 120 conversions out of 1000 visitors
    let mut variant_a = BetaBinomial::uniform();
    variant_a.update(120, 1000);
    let mean_a = variant_a.posterior_mean();
    let (lower_a, upper_a) = variant_a.credible_interval(0.95).expect("Valid confidence");

    println!("Variant A: 120 conversions / 1000 visitors");
    println!("  Conversion rate: {:.2}%", mean_a * 100.0);
    println!(
        "  95% credible interval: [{:.2}%, {:.2}%]",
        lower_a * 100.0,
        upper_a * 100.0
    );

    // Variant B: 145 conversions out of 1000 visitors
    let mut variant_b = BetaBinomial::uniform();
    variant_b.update(145, 1000);
    let mean_b = variant_b.posterior_mean();
    let (lower_b, upper_b) = variant_b.credible_interval(0.95).expect("Valid confidence");

    println!("\nVariant B: 145 conversions / 1000 visitors");
    println!("  Conversion rate: {:.2}%", mean_b * 100.0);
    println!(
        "  95% credible interval: [{:.2}%, {:.2}%]",
        lower_b * 100.0,
        upper_b * 100.0
    );

    println!("\nConclusion:");
    if lower_b > upper_a {
        println!("  ✓ Variant B is significantly better (95% confidence)");
    } else if lower_a > upper_b {
        println!("  ✓ Variant A is significantly better (95% confidence)");
    } else {
        println!("  ⚠ No clear winner yet - credible intervals overlap");
        println!("    Consider collecting more data");
    }
}

fn sequential_learning_example() {
    // Demonstrates how uncertainty decreases with more data
    let mut model = BetaBinomial::uniform();

    let experiments = vec![
        (7, 10),   // 70% success rate
        (15, 20),  // 75% success rate
        (23, 30),  // 76.7% success rate
        (31, 40),  // 77.5% success rate
        (77, 100), // 77% success rate
    ];

    println!("Sequential updates (true rate ≈ 77%):\n");
    println!("  Trials | Successes | Mean  | Variance  | 95% CI Width");
    println!("  -------|-----------|-------|-----------|-------------");

    let mut total_trials = 0;
    for (successes, trials) in experiments {
        model.update(successes, trials);
        total_trials += trials;

        let mean = model.posterior_mean();
        let variance = model.posterior_variance();
        let (lower, upper) = model.credible_interval(0.95).expect("Valid confidence");
        let width = upper - lower;

        println!("  {total_trials:6} | {successes:9} | {mean:.3} | {variance:.7} | {width:.4}");
    }

    println!("\nObservation: Variance and CI width decrease as data accumulates.");
}

fn prior_comparison_example() {
    // Compare different priors with same data

    // Data: 7 successes in 10 trials
    let successes = 7;
    let trials = 10;

    println!("Effect of prior choice (data: 7/10 successes):\n");

    // Uniform prior
    let mut uniform = BetaBinomial::uniform();
    uniform.update(successes, trials);
    println!("Uniform Prior Beta(1, 1):");
    println!("  Posterior mean: {:.4}", uniform.posterior_mean());
    println!("  Posterior: Beta({}, {})", uniform.alpha(), uniform.beta());

    // Jeffrey's prior
    let mut jeffreys = BetaBinomial::jeffreys();
    jeffreys.update(successes, trials);
    println!("\nJeffrey's Prior Beta(0.5, 0.5):");
    println!("  Posterior mean: {:.4}", jeffreys.posterior_mean());
    println!(
        "  Posterior: Beta({}, {})",
        jeffreys.alpha(),
        jeffreys.beta()
    );

    // Informative prior (strong belief in 50% success rate)
    let mut informative = BetaBinomial::new(50.0, 50.0).expect("Valid parameters");
    informative.update(successes, trials);
    println!("\nInformative Prior Beta(50, 50) [strong 50% belief]:");
    println!("  Posterior mean: {:.4}", informative.posterior_mean());
    println!(
        "  Posterior: Beta({}, {})",
        informative.alpha(),
        informative.beta()
    );

    println!("\nObservation: Strong priors pull the posterior toward prior belief.");
    println!("  With more data, all priors converge to the same posterior.");
}
