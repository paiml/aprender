use super::*;

#[test]
fn test_normal_inverse_gamma_posterior_mean_variance() {
    let mut model = NormalInverseGamma::new(0.0, 1.0, 3.0, 2.0).expect("Valid parameters");
    model.update(&[1.0, 2.0, 3.0, 4.0, 5.0]);

    let mean_var = model.posterior_mean_variance().expect("Alpha > 1");
    assert!(mean_var > 0.0); // Variance is positive
}

#[test]
fn test_normal_inverse_gamma_no_variance_for_weak_prior() {
    let model = NormalInverseGamma::noninformative();
    // Alpha = 0.001 < 1, so mean variance is undefined
    assert!(model.posterior_mean_variance().is_none());
    assert!(model.posterior_variance_mu().is_none());
}

#[test]
fn test_normal_inverse_gamma_posterior_variance_mu() {
    let mut model = NormalInverseGamma::new(0.0, 1.0, 3.0, 2.0).expect("Valid parameters");
    model.update(&[1.0, 2.0, 3.0, 4.0, 5.0]);

    let var_mu = model.posterior_variance_mu().expect("Alpha > 1");
    assert!(var_mu > 0.0); // Positive uncertainty
}

#[test]
fn test_normal_inverse_gamma_predictive() {
    let mut model = NormalInverseGamma::noninformative();
    model.update(&[4.2, 5.8, 6.1, 4.5, 5.0]);

    let pred_mean = model.posterior_predictive();
    let mean_mu = model.posterior_mean_mu();

    // Predictive mean equals posterior mean of μ
    assert!((pred_mean - mean_mu).abs() < 1e-6);
}

#[test]
fn test_normal_inverse_gamma_credible_interval() {
    let mut model = NormalInverseGamma::new(0.0, 1.0, 3.0, 2.0).expect("Valid parameters");
    model.update(&[4.2, 5.8, 6.1, 4.5, 5.0]);

    let (lower, upper) = model
        .credible_interval_mu(0.95)
        .expect("Valid confidence level");

    let mean = model.posterior_mean_mu();

    // Mean should be within interval
    assert!(lower < mean);
    assert!(mean < upper);
}

#[test]
fn test_normal_inverse_gamma_credible_interval_invalid() {
    let model = NormalInverseGamma::new(0.0, 1.0, 3.0, 2.0).expect("Valid parameters");

    assert!(model.credible_interval_mu(-0.1).is_err()); // Invalid confidence
    assert!(model.credible_interval_mu(1.1).is_err()); // Invalid confidence
}

#[test]
fn test_normal_inverse_gamma_credible_interval_requires_alpha_gt_1() {
    let model = NormalInverseGamma::noninformative(); // Alpha = 0.001 < 1

    // Should fail because alpha <= 1
    assert!(model.credible_interval_mu(0.95).is_err());
}

#[test]
fn test_normal_inverse_gamma_sequential_updates() {
    let mut model = NormalInverseGamma::noninformative();

    // First batch
    model.update(&[4.2, 5.8]);
    let kappa_1 = model.kappa();
    let alpha_1 = model.alpha();

    // Second batch
    model.update(&[6.1, 4.5, 5.0]);
    let kappa_2 = model.kappa();
    let alpha_2 = model.alpha();

    // Precision should increase with more data
    assert!(kappa_2 > kappa_1);
    assert!(alpha_2 > alpha_1);
}

#[test]
fn test_normal_inverse_gamma_empty_update() {
    let mut model = NormalInverseGamma::noninformative();
    let original_mu = model.mu();
    let original_kappa = model.kappa();

    // Empty data should not change parameters
    model.update(&[]);

    assert_eq!(model.mu(), original_mu);
    assert_eq!(model.kappa(), original_kappa);
}

#[test]
fn test_normal_inverse_gamma_strong_prior_influence() {
    // Strong prior centered at 10.0
    let mut strong_prior =
        NormalInverseGamma::new(10.0, 100.0, 50.0, 50.0).expect("Valid parameters");

    // Observe data far from prior (mean ≈ 5.0)
    strong_prior.update(&[4.2, 5.8, 6.1, 4.5, 5.0]);

    let posterior_mean = strong_prior.posterior_mean_mu();

    // Posterior should be pulled toward prior (closer to 10.0 than 5.0)
    assert!(posterior_mean > 7.0); // Strong prior dominates
}

// ========== Dirichlet-Multinomial Tests ==========

#[test]
fn test_dirichlet_multinomial_uniform_prior() {
    let prior = DirichletMultinomial::uniform(3);
    assert_eq!(prior.num_categories(), 3);
    assert_eq!(prior.alphas()[0], 1.0);
    assert_eq!(prior.alphas()[1], 1.0);
    assert_eq!(prior.alphas()[2], 1.0);
}

#[test]
#[should_panic(expected = "Must have at least 2 categories")]
fn test_dirichlet_multinomial_requires_2_categories() {
    let _ = DirichletMultinomial::uniform(1);
}

#[test]
fn test_dirichlet_multinomial_custom_prior() {
    let prior = DirichletMultinomial::new(vec![5.0, 3.0, 2.0]).expect("Valid parameters");
    assert_eq!(prior.num_categories(), 3);
    assert_eq!(prior.alphas()[0], 5.0);
    assert_eq!(prior.alphas()[1], 3.0);
    assert_eq!(prior.alphas()[2], 2.0);
}

#[test]
fn test_dirichlet_multinomial_invalid_prior() {
    assert!(DirichletMultinomial::new(vec![0.0, 1.0, 1.0]).is_err()); // Zero alpha
    assert!(DirichletMultinomial::new(vec![-1.0, 1.0, 1.0]).is_err()); // Negative alpha
    assert!(DirichletMultinomial::new(vec![1.0]).is_err()); // Only 1 category
}

#[test]
fn test_dirichlet_multinomial_update() {
    let mut model = DirichletMultinomial::uniform(3);
    model.update(&[10, 5, 3]);

    // Posterior should be Dirichlet(1+10, 1+5, 1+3) = Dirichlet(11, 6, 4)
    assert_eq!(model.alphas()[0], 11.0);
    assert_eq!(model.alphas()[1], 6.0);
    assert_eq!(model.alphas()[2], 4.0);
}

#[test]
#[should_panic(expected = "Counts must match number of categories")]
fn test_dirichlet_multinomial_update_wrong_size() {
    let mut model = DirichletMultinomial::uniform(3);
    model.update(&[10, 5]); // Only 2 counts for 3 categories
}

#[test]
fn test_dirichlet_multinomial_posterior_mean() {
    let mut model = DirichletMultinomial::uniform(3);
    model.update(&[10, 5, 3]);

    let mean = model.posterior_mean();

    // Should be [11/21, 6/21, 4/21]
    assert!((mean[0] - 11.0 / 21.0).abs() < 1e-6);
    assert!((mean[1] - 6.0 / 21.0).abs() < 1e-6);
    assert!((mean[2] - 4.0 / 21.0).abs() < 1e-6);

    // Sum should be 1.0
    assert!((mean.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}

#[test]
fn test_dirichlet_multinomial_posterior_mode() {
    let mut model = DirichletMultinomial::new(vec![2.0, 2.0, 2.0]).expect("Valid parameters");
    model.update(&[10, 5, 3]);

    // Posterior is Dirichlet(12, 7, 5), all > 1
    // Mode = [(12-1)/(24-3), (7-1)/(24-3), (5-1)/(24-3)]
    let mode = model.posterior_mode().expect("Mode should exist");

    assert!((mode[0] - 11.0 / 21.0).abs() < 1e-6);
    assert!((mode[1] - 6.0 / 21.0).abs() < 1e-6);
    assert!((mode[2] - 4.0 / 21.0).abs() < 1e-6);
}

#[test]
fn test_dirichlet_multinomial_no_mode_for_uniform() {
    let model = DirichletMultinomial::uniform(3);
    // Dirichlet(1, 1, 1) has no unique mode
    assert!(model.posterior_mode().is_none());
}

#[test]
fn test_dirichlet_multinomial_posterior_variance() {
    let mut model = DirichletMultinomial::uniform(3);
    model.update(&[10, 5, 3]);

    let variance = model.posterior_variance();

    // All variances should be positive
    assert!(variance[0] > 0.0);
    assert!(variance[1] > 0.0);
    assert!(variance[2] > 0.0);

    // Higher probability categories should have higher variance
    assert!(variance[0] > variance[2]); // Category 0 has highest prob
}

#[test]
fn test_dirichlet_multinomial_predictive() {
    let mut model = DirichletMultinomial::uniform(3);
    model.update(&[10, 5, 3]);

    let pred = model.posterior_predictive();
    let mean = model.posterior_mean();

    // Predictive should equal posterior mean
    for i in 0..3 {
        assert!((pred[i] - mean[i]).abs() < 1e-6);
    }

    // Sum should be 1.0
    assert!((pred.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}

#[test]
fn test_dirichlet_multinomial_credible_intervals() {
    let mut model = DirichletMultinomial::uniform(3);
    model.update(&[10, 5, 3]);

    let intervals = model
        .credible_intervals(0.95)
        .expect("Valid confidence level");
    let mean = model.posterior_mean();

    // Mean should be within interval for each category
    for i in 0..3 {
        assert!(intervals[i].0 < mean[i]);
        assert!(mean[i] < intervals[i].1);

        // Bounds should be in [0, 1]
        assert!((0.0..=1.0).contains(&intervals[i].0));
        assert!((0.0..=1.0).contains(&intervals[i].1));
    }
}

#[test]
fn test_dirichlet_multinomial_credible_intervals_invalid() {
    let model = DirichletMultinomial::uniform(3);

    assert!(model.credible_intervals(-0.1).is_err());
    assert!(model.credible_intervals(1.1).is_err());
}

#[test]
fn test_dirichlet_multinomial_sequential_updates() {
    let mut model = DirichletMultinomial::uniform(3);

    // First batch: [10, 5, 3]
    model.update(&[10, 5, 3]);
    assert_eq!(model.alphas()[0], 11.0);

    // Second batch: [5, 10, 2]
    model.update(&[5, 10, 2]);
    assert_eq!(model.alphas()[0], 16.0); // 11 + 5
    assert_eq!(model.alphas()[1], 16.0); // 6 + 10
    assert_eq!(model.alphas()[2], 6.0); // 4 + 2
}

#[test]
fn test_dirichlet_multinomial_empty_update() {
    let mut model = DirichletMultinomial::uniform(3);
    let original_alphas = model.alphas().to_vec();

    // Empty update (all zeros)
    model.update(&[0, 0, 0]);

    // Alphas should remain unchanged
    assert_eq!(model.alphas()[0], original_alphas[0]);
    assert_eq!(model.alphas()[1], original_alphas[1]);
    assert_eq!(model.alphas()[2], original_alphas[2]);
}

#[test]
fn test_dirichlet_multinomial_many_categories() {
    let k = 10;
    let mut model = DirichletMultinomial::uniform(k);

    // Update with some counts
    let counts = vec![5, 3, 8, 2, 10, 1, 4, 6, 7, 9];
    model.update(&counts);

    let mean = model.posterior_mean();

    // Check properties
    assert_eq!(mean.len(), k);
    assert!((mean.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert!(mean.iter().all(|&p| p > 0.0 && p < 1.0));
}

#[test]
fn test_dirichlet_multinomial_strong_prior_influence() {
    // Strong prior favoring category 0
    let mut strong_prior =
        DirichletMultinomial::new(vec![100.0, 10.0, 10.0]).expect("Valid parameters");

    // Observe data favoring category 1
    strong_prior.update(&[5, 20, 5]);

    let mean = strong_prior.posterior_mean();

    // Category 0 should still have highest probability due to strong prior
    assert!(mean[0] > mean[1]); // Prior dominates
    assert!(mean[0] > mean[2]);
}
