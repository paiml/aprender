// =========================================================================
// FALSIFY-BY: Bayesian conjugate priors contract (aprender bayesian)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-BY-* tests for conjugate priors
//   Why 2: conjugate tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no YAML contract for conjugate priors yet
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Conjugate updates were "obviously correct" (Bayes theorem)
//
// References:
//   - Jaynes (2003) "Probability Theory: The Logic of Science"
//   - Gelman et al. (2013) "Bayesian Data Analysis"
// =========================================================================

use super::*;

/// FALSIFY-BY-001: BetaBinomial posterior mean is α/(α+β)
#[test]
fn falsify_by_001_beta_posterior_mean() {
    let mut model = BetaBinomial::uniform(); // Beta(1,1)
    model.update(7, 10); // Beta(8, 4)

    let mean = model.posterior_mean();
    let expected = 8.0 / 12.0;

    assert!(
        (mean - expected).abs() < 1e-5,
        "FALSIFIED BY-001: mean={mean}, expected {expected}"
    );
}

/// FALSIFY-BY-002: BetaBinomial posterior mean is in [0, 1]
#[test]
fn falsify_by_002_beta_mean_bounded() {
    let mut model = BetaBinomial::uniform();
    model.update(100, 100);

    let mean = model.posterior_mean();
    assert!(
        (0.0..=1.0).contains(&mean),
        "FALSIFIED BY-002: mean={mean}, expected in [0,1]"
    );
}

/// FALSIFY-BY-003: More data reduces posterior variance
#[test]
fn falsify_by_003_beta_more_data_less_variance() {
    let mut few = BetaBinomial::uniform();
    few.update(7, 10);

    let mut many = BetaBinomial::uniform();
    many.update(70, 100);

    assert!(
        many.posterior_variance() < few.posterior_variance(),
        "FALSIFIED BY-003: variance(100 trials)={} >= variance(10 trials)={}",
        many.posterior_variance(),
        few.posterior_variance()
    );
}

/// FALSIFY-BY-004: GammaPoisson posterior mean approaches sample mean
#[test]
fn falsify_by_004_gamma_posterior_mean() {
    let mut model = GammaPoisson::noninformative();
    model.update(&[3, 5, 4, 6, 2]);

    let mean = model.posterior_mean();
    // With noninformative prior, posterior mean ≈ sample mean = 20/5 = 4.0
    assert!(
        (mean - 4.0).abs() < 0.5,
        "FALSIFIED BY-004: GammaPoisson mean={mean}, expected ~4.0"
    );
}

/// FALSIFY-BY-005: DirichletMultinomial posterior probabilities sum to 1
#[test]
fn falsify_by_005_dirichlet_probs_sum_to_one() {
    let mut model = DirichletMultinomial::uniform(3);
    model.update(&[10, 5, 3]);

    let probs = model.posterior_mean();
    let sum: f32 = probs.iter().sum();

    assert!(
        (sum - 1.0).abs() < 1e-5,
        "FALSIFIED BY-005: Dirichlet probs sum={sum}, expected 1.0"
    );
}

/// FALSIFY-BY-006: DirichletMultinomial probabilities are non-negative
#[test]
fn falsify_by_006_dirichlet_probs_nonneg() {
    let mut model = DirichletMultinomial::uniform(4);
    model.update(&[1, 0, 3, 2]);

    let probs = model.posterior_mean();
    for (i, &p) in probs.iter().enumerate() {
        assert!(p >= 0.0, "FALSIFIED BY-006: prob[{i}]={p}, expected >= 0.0");
    }
}

mod by_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-BY-002-prop: BetaBinomial posterior mean in [0, 1]
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_by_002_prop_beta_mean_bounded(
            successes in 0..100u32,
            trials in 1..200u32,
        ) {
            let successes = successes.min(trials);
            let mut model = BetaBinomial::uniform();
            model.update(successes, trials);

            let mean = model.posterior_mean();
            prop_assert!(
                (0.0..=1.0).contains(&mean),
                "FALSIFIED BY-002-prop: mean={} not in [0,1] (successes={}, trials={})",
                mean, successes, trials
            );
        }
    }

    /// FALSIFY-BY-005-prop: Dirichlet probabilities sum to 1
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn falsify_by_005_prop_dirichlet_sum(
            k in 2..=5usize,
            seed in 0..500u32,
        ) {
            let mut model = DirichletMultinomial::uniform(k);
            let counts: Vec<u32> = (0..k).map(|i| ((seed + i as u32) % 20) + 1).collect();
            model.update(&counts);

            let probs = model.posterior_mean();
            let sum: f32 = probs.iter().sum();
            prop_assert!(
                (sum - 1.0).abs() < 1e-4,
                "FALSIFIED BY-005-prop: Dirichlet sum={}, expected ~1.0",
                sum
            );
        }
    }
}
