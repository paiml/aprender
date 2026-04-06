//! Hyperband scheduler for efficient hyperparameter search
//!
//! Based on Li et al. (2018) - Hyperband: A Novel Bandit-Based Approach

use std::collections::HashMap;

use super::types::{HyperparameterSpace, ParameterValue};

/// Hyperband scheduler for efficient hyperparameter search
///
/// # Toyota Way: Muda (Waste Elimination)
///
/// Aggressive early stopping eliminates poorly performing configurations,
/// focusing resources on promising candidates.
#[derive(Debug, Clone)]
pub struct HyperbandScheduler {
    /// Maximum iterations per configuration
    pub(crate) max_iter: usize,
    /// Reduction factor (typically 3)
    pub(crate) eta: f64,
    /// Search space
    space: HyperparameterSpace,
}

impl HyperbandScheduler {
    /// Create a new Hyperband scheduler
    pub fn new(space: HyperparameterSpace, max_iter: usize) -> Self {
        Self { max_iter, eta: 3.0, space }
    }

    /// Set reduction factor
    pub fn with_eta(mut self, eta: f64) -> Self {
        self.eta = eta.max(2.0);
        self
    }

    /// Get s_max (number of successive halving brackets)
    pub fn s_max(&self) -> usize {
        (self.max_iter as f64).log(self.eta).floor() as usize
    }

    /// Get total budget B
    pub fn budget(&self) -> usize {
        (self.s_max() + 1) * self.max_iter
    }

    /// Generate bracket configurations
    ///
    /// Returns Vec of (n_configs, n_iterations) for each rung in the bracket
    pub fn bracket(&self, s: usize) -> Vec<(usize, usize)> {
        let s_max = self.s_max();
        if s > s_max {
            return Vec::new();
        }

        let n = ((self.budget() as f64 / self.max_iter as f64)
            * (self.eta.powi(s as i32) / (s + 1) as f64))
            .ceil() as usize;
        let r = self.max_iter / self.eta.powi(s as i32) as usize;

        (0..=s)
            .map(|i| {
                let n_i = (n as f64 / self.eta.powi(i as i32)).floor() as usize;
                let r_i = (r as f64 * self.eta.powi(i as i32)).floor() as usize;
                (n_i.max(1), r_i.max(1))
            })
            .collect()
    }

    /// Generate all configurations for a bracket
    pub fn generate_configs(&self, n: usize) -> Vec<HashMap<String, ParameterValue>> {
        let mut rng = rand::rng();
        (0..n).map(|_| self.space.sample_random(&mut rng)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::hpo::types::ParameterDomain;

    #[test]
    fn test_hyperband_new() {
        let space = HyperparameterSpace::new();
        let hb = HyperbandScheduler::new(space, 81);
        assert_eq!(hb.max_iter, 81);
        assert!((hb.eta - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_hyperband_s_max() {
        let space = HyperparameterSpace::new();
        let hb = HyperbandScheduler::new(space, 81);
        // log_3(81) = 4
        assert_eq!(hb.s_max(), 4);
    }

    #[test]
    fn test_hyperband_budget() {
        let space = HyperparameterSpace::new();
        let hb = HyperbandScheduler::new(space, 81);
        // B = (s_max + 1) * max_iter = 5 * 81 = 405
        assert_eq!(hb.budget(), 405);
    }

    #[test]
    fn test_hyperband_bracket() {
        let space = HyperparameterSpace::new();
        let hb = HyperbandScheduler::new(space, 81);

        // Bracket s=4 should start with most configs and least resources
        let bracket = hb.bracket(4);
        assert!(!bracket.is_empty());

        // First rung should have more configs than last
        let (n_first, r_first) = bracket.first().expect("collection should not be empty");
        let (n_last, r_last) = bracket.last().expect("collection should not be empty");
        assert!(*n_first >= *n_last);
        assert!(*r_first <= *r_last);
    }

    #[test]
    fn test_hyperband_generate_configs() {
        let mut space = HyperparameterSpace::new();
        space.add("lr", ParameterDomain::Continuous { low: 0.0, high: 1.0, log_scale: false });

        let hb = HyperbandScheduler::new(space, 81);
        let configs = hb.generate_configs(10);
        assert_eq!(configs.len(), 10);
    }

    #[test]
    fn test_hyperband_with_eta() {
        let space = HyperparameterSpace::new();
        let hb = HyperbandScheduler::new(space, 81).with_eta(4.0);
        assert!((hb.eta - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_hyperband_bracket_invalid_s() {
        let space = HyperparameterSpace::new();
        let hb = HyperbandScheduler::new(space, 81);
        let bracket = hb.bracket(100); // s > s_max
        assert!(bracket.is_empty());
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_hyperband_bracket_nonempty(max_iter in 9usize..243, eta in 2.0f64..5.0) {
            let space = HyperparameterSpace::new();
            let hb = HyperbandScheduler::new(space, max_iter).with_eta(eta);
            let s_max = hb.s_max();
            for s in 0..=s_max {
                let bracket = hb.bracket(s);
                prop_assert!(!bracket.is_empty());
            }
        }
    }
}
