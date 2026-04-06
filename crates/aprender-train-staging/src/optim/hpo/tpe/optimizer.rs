//! TPE optimizer core implementation

use rand::Rng;
use std::collections::HashMap;

use crate::optim::hpo::error::{HPOError, Result};
use crate::optim::hpo::types::{
    HyperparameterSpace, ParameterDomain, ParameterValue, Trial, TrialStatus,
};

use super::sampling::{count_categorical, sample_ei_ratio_continuous, sample_ei_ratio_discrete};

/// Tree-structured Parzen Estimator optimizer
///
/// # Toyota Way: Kaizen
///
/// Uses accumulated knowledge from trials to make increasingly better suggestions.
/// Splits trials by quantile to model "good" vs "bad" configurations.
#[derive(Debug, Clone)]
pub struct TPEOptimizer {
    /// Search space
    space: HyperparameterSpace,
    /// Quantile for splitting good/bad (default: 0.25)
    pub(crate) gamma: f64,
    /// Number of startup trials (random sampling)
    n_startup: usize,
    /// KDE bandwidth
    kde_bandwidth: f64,
    /// Completed trials
    trials: Vec<Trial>,
    /// Next trial ID
    next_id: usize,
}

impl TPEOptimizer {
    /// Create a new TPE optimizer
    pub fn new(space: HyperparameterSpace) -> Self {
        Self {
            space,
            gamma: 0.25,
            n_startup: 10,
            kde_bandwidth: 1.0,
            trials: Vec::new(),
            next_id: 0,
        }
    }

    /// Set gamma (quantile for splitting)
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma.clamp(0.01, 0.99);
        self
    }

    /// Set number of startup trials
    pub fn with_startup(mut self, n: usize) -> Self {
        self.n_startup = n.max(1);
        self
    }

    /// Get number of completed trials
    pub fn n_trials(&self) -> usize {
        self.trials.iter().filter(|t| t.status == TrialStatus::Completed).count()
    }

    /// Get best trial so far
    pub fn best_trial(&self) -> Option<&Trial> {
        self.trials
            .iter()
            .filter(|t| t.status == TrialStatus::Completed)
            .min_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Suggest next configuration to try
    pub fn suggest(&mut self) -> Result<Trial> {
        if self.space.is_empty() {
            return Err(HPOError::EmptySpace);
        }

        let mut rng = rand::rng();
        let config = if self.n_trials() < self.n_startup {
            // Random sampling during startup phase
            self.space.sample_random(&mut rng)
        } else {
            // TPE-guided sampling
            self.tpe_sample(&mut rng)
        };

        let trial = Trial::new(self.next_id, config);
        self.next_id += 1;
        Ok(trial)
    }

    /// Record trial result
    pub fn record(&mut self, mut trial: Trial, score: f64, iterations: usize) {
        trial.complete(score, iterations);
        self.trials.push(trial);
    }

    /// Record failed trial
    pub fn record_failed(&mut self, mut trial: Trial) {
        trial.fail();
        self.trials.push(trial);
    }

    /// TPE sampling (internal)
    fn tpe_sample<R: Rng>(&self, rng: &mut R) -> HashMap<String, ParameterValue> {
        let completed: Vec<_> =
            self.trials.iter().filter(|t| t.status == TrialStatus::Completed).collect();

        if completed.is_empty() {
            return self.space.sample_random(rng);
        }

        // Split trials into good (l) and bad (g) by gamma quantile
        let n_good = ((completed.len() as f64) * self.gamma).ceil() as usize;
        let n_good = n_good.max(1).min(completed.len() - 1);

        let mut sorted: Vec<_> = completed.clone();
        sorted.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));

        let (good_trials, bad_trials) = sorted.split_at(n_good);

        // Sample each parameter using TPE
        let mut config = HashMap::new();
        for (name, domain) in self.space.iter() {
            let value = self.sample_parameter_tpe(name, domain, good_trials, bad_trials, rng);
            config.insert(name.clone(), value);
        }

        config
    }

    /// Sample a single parameter using TPE
    fn sample_parameter_tpe<R: Rng>(
        &self,
        name: &str,
        domain: &ParameterDomain,
        good_trials: &[&Trial],
        bad_trials: &[&Trial],
        rng: &mut R,
    ) -> ParameterValue {
        match domain {
            ParameterDomain::Continuous { low, high, log_scale } => {
                // Extract values from trials
                let good_values: Vec<f64> = good_trials
                    .iter()
                    .filter_map(|t| t.config.get(name)?.as_float())
                    .map(|v| if *log_scale { v.max(f64::MIN_POSITIVE).ln() } else { v })
                    .collect();

                let bad_values: Vec<f64> = bad_trials
                    .iter()
                    .filter_map(|t| t.config.get(name)?.as_float())
                    .map(|v| if *log_scale { v.max(f64::MIN_POSITIVE).ln() } else { v })
                    .collect();

                // Sample from l(x) / g(x) using simple KDE approximation
                let (effective_low, effective_high) = if *log_scale {
                    (low.max(f64::MIN_POSITIVE).ln(), high.max(f64::MIN_POSITIVE).ln())
                } else {
                    (*low, *high)
                };

                let value = sample_ei_ratio_continuous(
                    &good_values,
                    &bad_values,
                    effective_low,
                    effective_high,
                    self.kde_bandwidth,
                    rng,
                );

                let final_value = if *log_scale { value.exp() } else { value };
                ParameterValue::Float(final_value.clamp(*low, *high))
            }
            ParameterDomain::Discrete { low, high } => {
                // Extract values
                let good_values: Vec<i64> =
                    good_trials.iter().filter_map(|t| t.config.get(name)?.as_int()).collect();

                let bad_values: Vec<i64> =
                    bad_trials.iter().filter_map(|t| t.config.get(name)?.as_int()).collect();

                let value = sample_ei_ratio_discrete(&good_values, &bad_values, *low, *high, rng);
                ParameterValue::Int(value)
            }
            ParameterDomain::Categorical { choices } => {
                // Count occurrences
                let good_counts = count_categorical(name, good_trials, choices);
                let bad_counts = count_categorical(name, bad_trials, choices);

                // Sample based on l(x) / g(x)
                let mut weights: Vec<f64> = choices
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let l = (good_counts[i] + 1) as f64; // Laplace smoothing
                        let g = (bad_counts[i] + 1) as f64;
                        l / g
                    })
                    .collect();

                // Normalize
                let total: f64 = weights.iter().sum();
                for w in &mut weights {
                    *w /= total;
                }

                // Sample
                let r: f64 = rng.random();
                let mut cumsum = 0.0;
                for (i, &w) in weights.iter().enumerate() {
                    cumsum += w;
                    if r < cumsum {
                        return ParameterValue::Categorical(choices[i].clone());
                    }
                }

                ParameterValue::Categorical(
                    choices.last().expect("choices is non-empty per validate()").clone(),
                )
            }
        }
    }
}
