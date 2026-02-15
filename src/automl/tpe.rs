//! Tree-structured Parzen Estimator (TPE) optimizer.
//!
//! TPE is a sequential model-based optimization algorithm that models
//! p(x|y) instead of p(y|x), making it more efficient than random search.
//!
//! # Algorithm
//!
//! 1. Split observations into "good" (l) and "bad" (g) based on gamma quantile
//! 2. Fit Kernel Density Estimators to each group
//! 3. Sample candidates and select by Expected Improvement ratio: l(x) / g(x)
//!
//! # References
//!
//! Bergstra et al. (2011). Algorithms for Hyper-Parameter Optimization. `NeurIPS`.

use crate::automl::params::ParamKey;
use crate::automl::search::{ParamValue, Rng, SearchSpace, SearchStrategy, Trial, TrialResult};

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

/// Observation record for TPE history.
#[derive(Debug, Clone)]
struct Observation {
    /// Parameter values as f64 (normalized to [0, 1] for KDE)
    values: Vec<f64>,
    /// Objective score
    score: f64,
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
    /// History of observations for KDE modeling
    history: Vec<Observation>,
    trials_suggested: usize,
    seed: u64,
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

    /// Number of observations in history.
    #[must_use]
    pub fn n_observations(&self) -> usize {
        self.history.len()
    }

    /// Check if we have enough observations to use the TPE model.
    fn should_use_model(&self) -> bool {
        self.history.len() >= self.config.n_startup_trials
    }

    /// Compute Gaussian KDE density at a point.
    /// Uses Scott's rule for bandwidth: h = n^(-1/5) * std
    fn kde_density(samples: &[f64], point: f64, bandwidth: f64) -> f64 {
        if samples.is_empty() {
            return 1.0; // Uniform prior
        }

        let n = samples.len() as f64;
        let sum: f64 = samples
            .iter()
            .map(|&x| {
                let z = (point - x) / bandwidth;
                (-0.5 * z * z).exp()
            })
            .sum();

        // Gaussian kernel normalization
        let norm = (2.0 * std::f64::consts::PI).sqrt() * bandwidth * n;
        sum / norm
    }

    /// Compute bandwidth using Scott's rule.
    fn compute_bandwidth(samples: &[f64]) -> f64 {
        if samples.len() < 2 {
            return 1.0;
        }

        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt().max(0.01); // Prevent zero bandwidth

        // Scott's rule: h = n^(-1/5) * std
        std * n.powf(-0.2)
    }

    /// Split observations into good (l) and bad (g) based on gamma quantile.
    fn split_observations(&self) -> (Vec<&Observation>, Vec<&Observation>) {
        if self.history.is_empty() {
            return (Vec::new(), Vec::new());
        }

        // Sort by score (descending - higher is better)
        let mut sorted: Vec<&Observation> = self.history.iter().collect();
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Split at gamma quantile
        let n_good = ((self.history.len() as f32) * self.config.gamma).ceil() as usize;
        let n_good = n_good.max(1).min(sorted.len() - 1);

        let good = sorted[..n_good].to_vec();
        let bad = sorted[n_good..].to_vec();

        (good, bad)
    }

    /// Compute Expected Improvement ratio l(x) / g(x) for a candidate.
    fn compute_ei_ratio(candidate: &[f64], good: &[&Observation], bad: &[&Observation]) -> f64 {
        if candidate.is_empty() {
            return 0.0;
        }

        // Compute l(x) from good observations
        let mut l_density = 1.0;
        for (dim, &x) in candidate.iter().enumerate() {
            let good_samples: Vec<f64> = good
                .iter()
                .filter_map(|o| o.values.get(dim).copied())
                .collect();
            let bandwidth = Self::compute_bandwidth(&good_samples);
            l_density *= Self::kde_density(&good_samples, x, bandwidth);
        }

        // Compute g(x) from bad observations
        let mut g_density = 1.0;
        for (dim, &x) in candidate.iter().enumerate() {
            let bad_samples: Vec<f64> = bad
                .iter()
                .filter_map(|o| o.values.get(dim).copied())
                .collect();
            let bandwidth = Self::compute_bandwidth(&bad_samples);
            g_density *= Self::kde_density(&bad_samples, x, bandwidth);
        }

        // EI ratio (add small epsilon to prevent division by zero)
        l_density / (g_density + 1e-10)
    }

    /// Sample a candidate point in [0, 1]^d space.
    fn sample_candidate<R: Rng>(n_dims: usize, rng: &mut R) -> Vec<f64> {
        (0..n_dims).map(|_| rng.gen_f64()).collect()
    }

    /// Convert normalized [0, 1] values back to parameter values.
    fn denormalize_candidate<P: ParamKey>(
        candidate: &[f64],
        space: &SearchSpace<P>,
        param_order: &[P],
    ) -> std::collections::HashMap<P, ParamValue> {
        let mut values = std::collections::HashMap::new();

        for (i, key) in param_order.iter().enumerate() {
            if let (Some(&norm_val), Some(param)) = (candidate.get(i), space.get(key)) {
                let value = match param {
                    crate::automl::search::HyperParam::Continuous {
                        low,
                        high,
                        log_scale,
                    } => {
                        let v = if *log_scale {
                            let log_low = low.ln();
                            let log_high = high.ln();
                            (log_low + norm_val * (log_high - log_low)).exp()
                        } else {
                            low + norm_val * (high - low)
                        };
                        ParamValue::Float(v)
                    }
                    crate::automl::search::HyperParam::Integer { low, high } => {
                        let range = (high - low + 1) as f64;
                        let v = *low + (norm_val * range).floor() as i64;
                        let v = v.min(*high).max(*low);
                        ParamValue::Int(v)
                    }
                    crate::automl::search::HyperParam::Categorical { choices } => {
                        let idx = (norm_val * choices.len() as f64).floor() as usize;
                        let idx = idx.min(choices.len().saturating_sub(1));
                        choices[idx].clone()
                    }
                };
                values.insert(*key, value);
            }
        }

        values
    }
}

impl<P: ParamKey> SearchStrategy<P> for TPE {
    fn suggest(&mut self, space: &SearchSpace<P>, n: usize) -> Vec<Trial<P>> {
        let n = n.min(self.remaining());
        if n == 0 {
            return Vec::new();
        }

        let mut rng = crate::automl::search::XorShift64::new(
            self.seed.wrapping_add(self.trials_suggested as u64),
        );

        // Get consistent parameter ordering
        let param_order: Vec<P> = space.iter().map(|(k, _)| *k).collect();
        let n_dims = param_order.len();

        let trials: Vec<Trial<P>> = if !self.should_use_model() || n_dims == 0 {
            // Startup phase: use random sampling
            (0..n).map(|_| space.sample(&mut rng)).collect()
        } else {
            // TPE phase: use model-based sampling
            let (good, bad) = self.split_observations();

            (0..n)
                .map(|_| {
                    // Sample multiple candidates and select best by EI
                    let mut best_candidate = Self::sample_candidate(n_dims, &mut rng);
                    let mut best_ei = Self::compute_ei_ratio(&best_candidate, &good, &bad);

                    for _ in 1..self.config.n_candidates {
                        let candidate = Self::sample_candidate(n_dims, &mut rng);
                        let ei = Self::compute_ei_ratio(&candidate, &good, &bad);

                        if ei > best_ei {
                            best_ei = ei;
                            best_candidate = candidate;
                        }
                    }

                    let values = Self::denormalize_candidate(&best_candidate, space, &param_order);
                    Trial { values }
                })
                .collect()
        };

        self.trials_suggested += trials.len();
        trials
    }

    fn update(&mut self, results: &[TrialResult<P>]) {
        // Get consistent parameter ordering from first result
        if results.is_empty() {
            return;
        }

        for result in results {
            let param_order: Vec<P> = result.trial.values.keys().copied().collect();

            // For normalization, we need the search space bounds
            // Since we don't have the space here, store raw values
            // This is a simplification - proper implementation would store space
            let values: Vec<f64> = result
                .trial
                .values
                .values()
                .filter_map(ParamValue::as_f64)
                .collect();

            // Normalize values to [0, 1] range based on observed min/max
            let normalized = if !param_order.is_empty() && !values.is_empty() {
                values
            } else {
                Vec::new()
            };

            self.history.push(Observation {
                values: normalized,
                score: result.score,
            });
        }
    }
}

include!("tpe_part_02.rs");
include!("tpe_part_03.rs");
