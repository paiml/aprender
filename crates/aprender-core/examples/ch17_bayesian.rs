#![allow(clippy::disallowed_methods)]
//! Chapter 17: Bayesian Methods
//!
//! Demonstrates conjugate prior updates with NormalInverseGamma.
//! Citation: Efron, "Bayesians, Frequentists, and Scientists," arXiv:math/0504499
//! Contract: contracts/apr-book-ch17-v1.yaml (v2 — api_calls enforced)

use aprender::bayesian::{BetaBinomial, NormalInverseGamma};

fn main() {
    // --- BetaBinomial conjugate model ---
    let bb = BetaBinomial::new(2.0, 2.0).expect("valid Beta prior");
    println!("BetaBinomial prior: Beta(2, 2)");
    let _ = bb; // Instantiation contract

    // --- NormalInverseGamma conjugate model ---
    // Prior: mu=0, kappa=1, alpha=3, beta=2
    let mut nig = NormalInverseGamma::new(0.0, 1.0, 3.0, 2.0)
        .expect("valid NIG prior parameters");

    let prior_mu = nig.posterior_mean_mu();
    let prior_var = nig.posterior_mean_variance();
    println!("\nNormalInverseGamma prior:");
    println!("  mu (location):  {prior_mu:.3}");
    println!("  E[variance]:    {:.3}", prior_var.unwrap_or(f32::NAN));

    // Observe data centered around 5.0
    let data: Vec<f32> = vec![4.8, 5.1, 5.3, 4.9, 5.0, 5.2, 4.7, 5.1];
    let data_mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    println!("\nObserved data: {} samples, mean={data_mean:.2}", data.len());

    // Update posterior with data
    nig.update(&data);

    let post_mu = nig.posterior_mean_mu();
    let post_var = nig.posterior_mean_variance();
    println!("\nPosterior after update:");
    println!("  mu (location):  {post_mu:.3}");
    println!("  E[variance]:    {:.3}", post_var.unwrap_or(f32::NAN));

    // Posterior mean should shift toward data mean (5.0)
    let prior_distance = (prior_mu - data_mean).abs();
    let post_distance = (post_mu - data_mean).abs();
    println!("\n  Prior distance to data mean: {prior_distance:.3}");
    println!("  Posterior distance to data mean: {post_distance:.3}");
    assert!(
        post_distance < prior_distance,
        "Posterior must move toward data: {post_distance:.3} < {prior_distance:.3}"
    );

    // Predictive value
    let predictive = nig.posterior_predictive();
    println!("  Posterior predictive: {predictive:.3}");

    println!("\nChapter 17 contracts: PASSED");
}
