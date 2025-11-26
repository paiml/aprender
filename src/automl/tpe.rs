//! Tree-structured Parzen Estimator (TPE) optimizer.
//!
//! TPE is a sequential model-based optimization algorithm that models
//! p(x|y) instead of p(y|x), making it more efficient than random search.
//!
//! # References
//!
//! Bergstra et al. (2011). Algorithms for Hyper-Parameter Optimization. NeurIPS.

use crate::automl::params::ParamKey;
use crate::automl::search::{SearchSpace, SearchStrategy, Trial, TrialResult};

/// TPE optimizer configuration.
#[derive(Debug, Clone)]
pub struct TPEConfig {
    /// Quantile for splitting good/bad observations (default: 0.25)
    pub gamma: f32,
    /// Number of candidates to sample per iteration (default: 24)
    pub n_candidates: usize,
    /// Minimum observations before using the model (default: 10)
    pub n_startup_trials: usize,
}

impl Default for TPEConfig {
    fn default() -> Self {
        Self {
            gamma: 0.25,
            n_candidates: 24,
            n_startup_trials: 10,
        }
    }
}

/// Tree-structured Parzen Estimator optimizer.
///
/// More sample-efficient than random search for >10 trials.
///
/// # Example
///
/// ```
/// use aprender::automl::{TPE, SearchSpace, SearchStrategy};
/// use aprender::automl::params::RandomForestParam as RF;
///
/// let space = SearchSpace::new()
///     .add(RF::NEstimators, 10..500)
///     .add(RF::MaxDepth, 2..20);
///
/// let mut tpe = TPE::new(100);
/// let trials = tpe.suggest(&space, 1);
/// ```
#[derive(Debug, Clone)]
pub struct TPE {
    config: TPEConfig,
    n_trials: usize,
    #[allow(dead_code)] // Will be used when TPE model is fully implemented
    history: Vec<TrialResult<GenericParam>>,
    trials_suggested: usize,
    seed: u64,
}

// Placeholder for generic param until we have proper implementation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GenericParam;

impl ParamKey for GenericParam {
    fn name(&self) -> &'static str {
        "generic"
    }
}

impl TPE {
    /// Create TPE optimizer with n iterations.
    #[must_use]
    pub fn new(n_trials: usize) -> Self {
        Self {
            config: TPEConfig::default(),
            n_trials,
            history: Vec::new(),
            trials_suggested: 0,
            seed: 42,
        }
    }

    /// Set random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set gamma (quantile for good/bad split).
    #[must_use]
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.config.gamma = gamma.clamp(0.01, 0.5);
        self
    }

    /// Set number of startup trials (random before model).
    #[must_use]
    pub fn with_startup_trials(mut self, n: usize) -> Self {
        self.config.n_startup_trials = n;
        self
    }

    /// Remaining trials.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.n_trials.saturating_sub(self.trials_suggested)
    }
}

impl<P: ParamKey> SearchStrategy<P> for TPE {
    fn suggest(&mut self, space: &SearchSpace<P>, n: usize) -> Vec<Trial<P>> {
        // TODO: Implement TPE sampling
        // For now, fall back to random sampling
        let n = n.min(self.remaining());
        if n == 0 {
            return Vec::new();
        }

        let mut rng = crate::automl::search::XorShift64::new(
            self.seed.wrapping_add(self.trials_suggested as u64),
        );
        let trials: Vec<Trial<P>> = (0..n).map(|_| space.sample(&mut rng)).collect();
        self.trials_suggested += trials.len();
        trials
    }

    fn update(&mut self, _results: &[TrialResult<P>]) {
        // TODO: Update history for TPE model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automl::params::RandomForestParam as RF;
    use crate::automl::SearchSpace;

    // ==================== UNIT TESTS ====================

    #[test]
    fn test_tpe_config_defaults() {
        let config = TPEConfig::default();
        assert!((config.gamma - 0.25).abs() < 0.01);
        assert_eq!(config.n_candidates, 24);
        assert_eq!(config.n_startup_trials, 10);
    }

    #[test]
    fn test_tpe_creation() {
        let tpe = TPE::new(100);
        assert_eq!(tpe.n_trials, 100);
        assert_eq!(tpe.remaining(), 100);
    }

    #[test]
    fn test_tpe_with_seed() {
        let tpe = TPE::new(50).with_seed(12345);
        assert_eq!(tpe.seed, 12345);
    }

    #[test]
    fn test_tpe_with_gamma() {
        let tpe = TPE::new(50).with_gamma(0.15);
        assert!((tpe.config.gamma - 0.15).abs() < 0.01);
    }

    #[test]
    fn test_tpe_gamma_clamped() {
        let tpe_low = TPE::new(50).with_gamma(0.0);
        assert!(tpe_low.config.gamma >= 0.01);

        let tpe_high = TPE::new(50).with_gamma(1.0);
        assert!(tpe_high.config.gamma <= 0.5);
    }

    #[test]
    fn test_tpe_suggest_respects_budget() {
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..500);

        let mut tpe = TPE::new(5);
        let t1 = tpe.suggest(&space, 3);
        assert_eq!(t1.len(), 3);
        assert_eq!(tpe.remaining(), 2);

        let t2 = tpe.suggest(&space, 10);
        assert_eq!(t2.len(), 2);
        assert_eq!(tpe.remaining(), 0);
    }

    #[test]
    fn test_tpe_deterministic_with_seed() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add(RF::NEstimators, 10..500)
            .add(RF::MaxDepth, 2..20);

        let mut tpe1 = TPE::new(10).with_seed(42);
        let mut tpe2 = TPE::new(10).with_seed(42);

        let t1 = tpe1.suggest(&space, 5);
        let t2 = tpe2.suggest(&space, 5);

        for (a, b) in t1.iter().zip(t2.iter()) {
            assert_eq!(a.get(&RF::NEstimators), b.get(&RF::NEstimators));
        }
    }

    #[test]
    fn test_tpe_empty_when_exhausted() {
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);

        let mut tpe = TPE::new(2);
        let _ = tpe.suggest(&space, 2);
        let empty = tpe.suggest(&space, 1);
        assert!(empty.is_empty());
    }

    // ==================== PROPERTY TESTS ====================
    // TPE-specific properties to verify once implemented
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::automl::params::RandomForestParam as RF;
    use crate::automl::SearchSpace;
    use proptest::prelude::*;

    proptest! {
        /// TPE should always respect budget constraint.
        #[test]
        fn prop_tpe_respects_budget(
            n_trials in 1_usize..50,
            seed in any::<u64>(),
            request in 1_usize..100
        ) {
            let space: SearchSpace<RF> = SearchSpace::new()
                .add(RF::NEstimators, 10..500);

            let mut tpe = TPE::new(n_trials).with_seed(seed);
            let trials = tpe.suggest(&space, request);

            prop_assert!(trials.len() <= n_trials);
            prop_assert!(trials.len() <= request);
        }

        /// Same seed should produce same initial trials.
        #[test]
        fn prop_tpe_deterministic(seed in any::<u64>()) {
            let space: SearchSpace<RF> = SearchSpace::new()
                .add(RF::NEstimators, 10..500);

            let mut tpe1 = TPE::new(10).with_seed(seed);
            let mut tpe2 = TPE::new(10).with_seed(seed);

            let t1 = tpe1.suggest(&space, 5);
            let t2 = tpe2.suggest(&space, 5);

            for (a, b) in t1.iter().zip(t2.iter()) {
                prop_assert_eq!(a.get(&RF::NEstimators), b.get(&RF::NEstimators));
            }
        }

        /// Gamma should always be in valid range.
        #[test]
        fn prop_gamma_clamped(gamma in -1.0_f32..2.0) {
            let tpe = TPE::new(10).with_gamma(gamma);
            prop_assert!(tpe.config.gamma >= 0.01);
            prop_assert!(tpe.config.gamma <= 0.5);
        }
    }
}
