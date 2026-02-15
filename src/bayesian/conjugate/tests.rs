//\! Conjugate Tests - Extreme TDD
//\! PMAT-085: Extracted from mod.rs for PMAT file health compliance

use super::*;

// ========== Beta-Binomial Tests ==========

#[test]
fn test_beta_binomial_uniform_prior() {
    let prior = BetaBinomial::uniform();
    assert_eq!(prior.alpha(), 1.0);
    assert_eq!(prior.beta(), 1.0);
    assert!((prior.posterior_mean() - 0.5).abs() < 1e-6);
}

#[test]
fn test_beta_binomial_jeffreys_prior() {
    let prior = BetaBinomial::jeffreys();
    assert_eq!(prior.alpha(), 0.5);
    assert_eq!(prior.beta(), 0.5);
}

#[test]
fn test_beta_binomial_custom_prior() {
    let prior = BetaBinomial::new(2.0, 5.0).expect("Valid parameters");
    assert_eq!(prior.alpha(), 2.0);
    assert_eq!(prior.beta(), 5.0);
}

#[test]
fn test_beta_binomial_invalid_prior() {
    assert!(BetaBinomial::new(0.0, 1.0).is_err());
    assert!(BetaBinomial::new(1.0, -1.0).is_err());
}

#[test]
fn test_beta_binomial_update() {
    let mut model = BetaBinomial::uniform();
    model.update(7, 10);

    // Posterior should be Beta(1+7, 1+3) = Beta(8, 4)
    assert_eq!(model.alpha(), 8.0);
    assert_eq!(model.beta(), 4.0);
}

#[test]
fn test_beta_binomial_posterior_mean() {
    let mut model = BetaBinomial::uniform();
    model.update(7, 10);

    let mean = model.posterior_mean();
    let expected = 8.0 / 12.0; // α/(α+β)
    assert!((mean - expected).abs() < 1e-6);
}

#[test]
fn test_beta_binomial_posterior_mode() {
    let mut model = BetaBinomial::new(2.0, 2.0).expect("Valid parameters");
    model.update(7, 10);

    // Posterior is Beta(9, 5)
    // Mode = (α-1)/(α+β-2) = 8/12
    let mode = model.posterior_mode().expect("Mode should exist");
    let expected = 8.0 / 12.0;
    assert!((mode - expected).abs() < 1e-6);
}

#[test]
fn test_beta_binomial_no_mode_for_uniform() {
    let model = BetaBinomial::uniform();
    // Beta(1, 1) has no unique mode (uniform on [0,1])
    assert!(model.posterior_mode().is_none());
}

#[test]
fn test_beta_binomial_posterior_variance() {
    let mut model = BetaBinomial::uniform();
    model.update(70, 100);

    let variance = model.posterior_variance();

    // More data → lower variance
    assert!(variance < 0.01);
}

#[test]
fn test_beta_binomial_predictive() {
    let mut model = BetaBinomial::uniform();
    model.update(7, 10);

    let prob = model.posterior_predictive();
    let mean = model.posterior_mean();

    // For Beta-Binomial, predictive equals posterior mean
    assert!((prob - mean).abs() < 1e-6);
}

#[test]
fn test_beta_binomial_credible_interval() {
    let mut model = BetaBinomial::uniform();
    model.update(7, 10);

    let (lower, upper) = model
        .credible_interval(0.95)
        .expect("Valid confidence level");

    let mean = model.posterior_mean();

    // Mean should be within interval
    assert!(lower < mean);
    assert!(mean < upper);

    // Bounds should be in [0, 1]
    assert!((0.0..=1.0).contains(&lower));
    assert!((0.0..=1.0).contains(&upper));
}

#[test]
fn test_beta_binomial_credible_interval_90pct() {
    let mut model = BetaBinomial::uniform();
    model.update(7, 10);

    let (lower, upper) = model
        .credible_interval(0.90)
        .expect("Valid confidence level");

    let mean = model.posterior_mean();

    // Mean should be within interval
    assert!(lower < mean);
    assert!(mean < upper);

    // 90% interval should be narrower than 95%
    let (lower_95, upper_95) = model.credible_interval(0.95).unwrap();
    assert!(
        lower >= lower_95,
        "90% lower bound should be >= 95% lower bound"
    );
    assert!(
        upper <= upper_95,
        "90% upper bound should be <= 95% upper bound"
    );
}

#[test]
fn test_beta_binomial_credible_interval_99pct() {
    let mut model = BetaBinomial::uniform();
    model.update(7, 10);

    let (lower, upper) = model
        .credible_interval(0.99)
        .expect("Valid confidence level");

    let mean = model.posterior_mean();

    // Mean should be within interval
    assert!(lower < mean);
    assert!(mean < upper);

    // 99% interval should be wider than 95%
    let (lower_95, upper_95) = model.credible_interval(0.95).unwrap();
    assert!(
        lower <= lower_95,
        "99% lower bound should be <= 95% lower bound"
    );
    assert!(
        upper >= upper_95,
        "99% upper bound should be >= 95% upper bound"
    );
}

#[test]
fn test_beta_binomial_credible_interval_invalid() {
    let model = BetaBinomial::uniform();

    assert!(model.credible_interval(-0.1).is_err());
    assert!(model.credible_interval(1.1).is_err());
}

#[test]
fn test_beta_binomial_sequential_updates() {
    let mut model = BetaBinomial::uniform();

    // First experiment: 7/10 successes
    model.update(7, 10);
    assert_eq!(model.alpha(), 8.0);
    assert_eq!(model.beta(), 4.0);

    // Second experiment: 3/5 successes
    model.update(3, 5);
    assert_eq!(model.alpha(), 11.0);
    assert_eq!(model.beta(), 6.0);
}

#[test]
#[should_panic(expected = "Successes cannot exceed total trials")]
fn test_beta_binomial_invalid_update() {
    let mut model = BetaBinomial::uniform();
    model.update(11, 10); // More successes than trials
}

// Property-based tests would go here using proptest
// Example: Verify posterior mean is always in [0, 1]
// Example: Verify variance decreases with more data

// ========== Gamma-Poisson Tests ==========

#[test]
fn test_gamma_poisson_noninformative_prior() {
    let prior = GammaPoisson::noninformative();
    assert_eq!(prior.alpha(), 0.001);
    assert_eq!(prior.beta(), 0.001);
    assert!((prior.posterior_mean() - 1.0).abs() < 0.01); // α/β = 1
}

#[test]
fn test_gamma_poisson_custom_prior() {
    let prior = GammaPoisson::new(50.0, 10.0).expect("Valid parameters");
    assert_eq!(prior.alpha(), 50.0);
    assert_eq!(prior.beta(), 10.0);
    assert!((prior.posterior_mean() - 5.0).abs() < 0.01); // 50/10 = 5
}

#[test]
fn test_gamma_poisson_invalid_prior() {
    assert!(GammaPoisson::new(0.0, 1.0).is_err());
    assert!(GammaPoisson::new(1.0, -1.0).is_err());
}

#[test]
fn test_gamma_poisson_update() {
    let mut model = GammaPoisson::noninformative();
    model.update(&[3, 5, 4, 6, 2]);

    // Sum = 20, n = 5
    // Posterior should be Gamma(0.001 + 20, 0.001 + 5)
    assert!((model.alpha() - 20.001).abs() < 0.01);
    assert!((model.beta() - 5.001).abs() < 0.01);
}

#[test]
fn test_gamma_poisson_posterior_mean() {
    let mut model = GammaPoisson::noninformative();
    model.update(&[3, 5, 4, 6, 2]);

    let mean = model.posterior_mean();
    let expected = 20.001 / 5.001; // α/β ≈ 4.0
    assert!((mean - expected).abs() < 0.01);
}

#[test]
fn test_gamma_poisson_posterior_mode() {
    let mut model = GammaPoisson::new(2.0, 1.0).expect("Valid parameters");
    model.update(&[3, 5, 4, 6, 2]);

    // Posterior is Gamma(22, 6)
    // Mode = (α-1)/β = 21/6 = 3.5
    let mode = model.posterior_mode().expect("Mode should exist");
    let expected = 21.0 / 6.0;
    assert!((mode - expected).abs() < 0.01);
}

#[test]
fn test_gamma_poisson_no_mode_for_weak_prior() {
    let model = GammaPoisson::noninformative();
    // Gamma(0.001, 0.001) has α < 1, no unique mode
    assert!(model.posterior_mode().is_none());
}

#[test]
fn test_gamma_poisson_posterior_variance() {
    let mut model = GammaPoisson::noninformative();
    model.update(&[3, 5, 4, 6, 2, 8, 7, 9, 1, 0]); // 10 observations

    let variance = model.posterior_variance();

    // More data → lower variance
    assert!(variance < 1.0);
}

#[test]
fn test_gamma_poisson_predictive() {
    let mut model = GammaPoisson::noninformative();
    model.update(&[3, 5, 4, 6, 2]);

    let prob = model.posterior_predictive();
    let mean = model.posterior_mean();

    // For Gamma-Poisson, predictive mean equals posterior mean
    assert!((prob - mean).abs() < 1e-6);
}

#[test]
fn test_gamma_poisson_credible_interval() {
    let mut model = GammaPoisson::noninformative();
    model.update(&[3, 5, 4, 6, 2]);

    let (lower, upper) = model
        .credible_interval(0.95)
        .expect("Valid confidence level");

    let mean = model.posterior_mean();

    // Mean should be within interval
    assert!(lower < mean);
    assert!(mean < upper);

    // Lower bound should be non-negative (rate cannot be negative)
    assert!(lower >= 0.0);
}

#[test]
fn test_gamma_poisson_credible_interval_90pct() {
    let mut model = GammaPoisson::noninformative();
    model.update(&[3, 5, 4, 6, 2]);

    let (lower, upper) = model
        .credible_interval(0.90)
        .expect("Valid confidence level");

    let mean = model.posterior_mean();

    // Mean should be within interval
    assert!(lower < mean);
    assert!(mean < upper);

    // 90% interval should be narrower than 95%
    let (lower_95, upper_95) = model.credible_interval(0.95).unwrap();
    assert!(
        lower >= lower_95,
        "90% lower bound should be >= 95% lower bound"
    );
    assert!(
        upper <= upper_95,
        "90% upper bound should be <= 95% upper bound"
    );
}

#[test]
fn test_gamma_poisson_credible_interval_99pct() {
    let mut model = GammaPoisson::noninformative();
    model.update(&[3, 5, 4, 6, 2]);

    let (lower, upper) = model
        .credible_interval(0.99)
        .expect("Valid confidence level");

    let mean = model.posterior_mean();

    // Mean should be within interval
    assert!(lower < mean);
    assert!(mean < upper);

    // 99% interval should be wider than 95%
    let (lower_95, upper_95) = model.credible_interval(0.95).unwrap();
    assert!(
        lower <= lower_95,
        "99% lower bound should be <= 95% lower bound"
    );
    assert!(
        upper >= upper_95,
        "99% upper bound should be >= 95% upper bound"
    );
}

#[test]
fn test_gamma_poisson_credible_interval_invalid() {
    let model = GammaPoisson::noninformative();

    assert!(model.credible_interval(-0.1).is_err());
    assert!(model.credible_interval(1.1).is_err());
}

#[test]
fn test_gamma_poisson_sequential_updates() {
    let mut model = GammaPoisson::noninformative();

    // First batch: [3, 5, 4] sum=12, n=3
    model.update(&[3, 5, 4]);
    assert!((model.alpha() - 12.001).abs() < 0.01);
    assert!((model.beta() - 3.001).abs() < 0.01);

    // Second batch: [6, 2] sum=8, n=2
    model.update(&[6, 2]);
    assert!((model.alpha() - 20.001).abs() < 0.01);
    assert!((model.beta() - 5.001).abs() < 0.01);
}

#[test]
fn test_gamma_poisson_empty_update() {
    let mut model = GammaPoisson::noninformative();
    let original_alpha = model.alpha();
    let original_beta = model.beta();

    // Empty data should not change parameters
    model.update(&[]);

    assert_eq!(model.alpha(), original_alpha);
    assert_eq!(model.beta(), original_beta);
}

// ========== Normal-InverseGamma Tests ==========

#[test]
fn test_normal_inverse_gamma_noninformative_prior() {
    let prior = NormalInverseGamma::noninformative();
    assert_eq!(prior.mu(), 0.0);
    assert_eq!(prior.kappa(), 0.001);
    assert_eq!(prior.alpha(), 0.001);
    assert_eq!(prior.beta(), 0.001);
}

#[test]
fn test_normal_inverse_gamma_custom_prior() {
    let prior = NormalInverseGamma::new(5.0, 10.0, 5.0, 5.0).expect("Valid parameters");
    assert_eq!(prior.mu(), 5.0);
    assert_eq!(prior.kappa(), 10.0);
    assert_eq!(prior.alpha(), 5.0);
    assert_eq!(prior.beta(), 5.0);
}

#[test]
fn test_normal_inverse_gamma_invalid_prior() {
    assert!(NormalInverseGamma::new(0.0, 0.0, 1.0, 1.0).is_err()); // kappa = 0
    assert!(NormalInverseGamma::new(0.0, 1.0, 0.0, 1.0).is_err()); // alpha = 0
    assert!(NormalInverseGamma::new(0.0, 1.0, 1.0, 0.0).is_err()); // beta = 0
}

#[test]
fn test_normal_inverse_gamma_update() {
    let mut model = NormalInverseGamma::noninformative();
    model.update(&[4.2, 5.8, 6.1, 4.5, 5.0]);

    // Parameters should have been updated
    assert!(model.kappa() > 0.001); // Precision increased
    assert!(model.alpha() > 0.001); // Shape increased
}

#[test]
fn test_normal_inverse_gamma_posterior_mean_mu() {
    let mut model = NormalInverseGamma::noninformative();
    let data = vec![4.2, 5.8, 6.1, 4.5, 5.0];
    model.update(&data);

    let mean_mu = model.posterior_mean_mu();
    let sample_mean = data.iter().sum::<f32>() / data.len() as f32;

    // With weak prior, posterior mean should be close to sample mean
    assert!((mean_mu - sample_mean).abs() < 0.1);
}

include!("tests_part_02.rs");
