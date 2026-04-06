//! Searcher and scheduler implementations for classification HPO.
//!
//! Provides concrete searcher/scheduler types used by `ClassifyTuner`.
//!
//! - **Searchers**: `TpeSearcher`, `GridSearcher`, `RandomSearcher`
//! - **Schedulers**: `AshaScheduler`, `MedianScheduler`, `NoScheduler`

use std::collections::HashMap;

use crate::optim::{
    GridSearch, HyperparameterSpace, ParameterValue, TPEOptimizer, Trial, TrialStatus,
};

// ═══════════════════════════════════════════════════════════════════════
// Traits
// ═══════════════════════════════════════════════════════════════════════

/// Search strategy for suggesting hyperparameter configurations.
///
/// Wraps existing TPEOptimizer, GridSearch, and random sampling.
pub trait TuneSearcher {
    /// Suggest the next trial configuration to evaluate.
    fn suggest(&mut self) -> crate::Result<Trial>;

    /// Record a completed trial's score.
    fn record(&mut self, trial: Trial, score: f64, epochs: usize);

    /// Get the best trial so far (lowest score).
    fn best(&self) -> Option<&Trial>;
}

/// Scheduler for deciding whether to stop a trial early.
///
/// Wraps existing HyperbandScheduler logic for ASHA-style pruning.
pub trait TuneScheduler {
    /// Should this trial be stopped early?
    ///
    /// # Arguments
    /// * `trial_id` - Current trial index
    /// * `epoch` - Current epoch (0-indexed)
    /// * `val_loss` - Current validation loss
    fn should_stop(&self, trial_id: usize, epoch: usize, val_loss: f64) -> bool;
}

// ═══════════════════════════════════════════════════════════════════════
// Searcher implementations
// ═══════════════════════════════════════════════════════════════════════

/// TPE-based searcher (Bayesian optimization).
pub struct TpeSearcher {
    optimizer: TPEOptimizer,
}

impl TpeSearcher {
    /// Create a TPE searcher with the given search space.
    pub fn new(space: HyperparameterSpace, n_startup: usize) -> Self {
        let optimizer = TPEOptimizer::new(space).with_startup(n_startup);
        Self { optimizer }
    }
}

impl TuneSearcher for TpeSearcher {
    fn suggest(&mut self) -> crate::Result<Trial> {
        self.optimizer
            .suggest()
            .map_err(|e| crate::Error::ConfigError(format!("TPE suggest failed: {e}")))
    }

    fn record(&mut self, trial: Trial, score: f64, epochs: usize) {
        self.optimizer.record(trial, score, epochs);
    }

    fn best(&self) -> Option<&Trial> {
        self.optimizer.best_trial()
    }
}

/// Grid-based searcher (exhaustive).
pub struct GridSearcher {
    configs: Vec<HashMap<String, ParameterValue>>,
    trials: Vec<Trial>,
    next_idx: usize,
}

impl GridSearcher {
    /// Create a grid searcher with the given search space.
    pub fn new(space: HyperparameterSpace, n_points: usize) -> Self {
        let grid = GridSearch::new(space, n_points);
        let configs = grid.configurations();
        Self { configs, trials: Vec::new(), next_idx: 0 }
    }
}

impl TuneSearcher for GridSearcher {
    fn suggest(&mut self) -> crate::Result<Trial> {
        if self.next_idx >= self.configs.len() {
            return Err(crate::Error::ConfigError(
                "Grid search exhausted all configurations".to_string(),
            ));
        }
        let config = self.configs[self.next_idx].clone();
        let trial = Trial::new(self.next_idx, config);
        self.next_idx += 1;
        Ok(trial)
    }

    fn record(&mut self, trial: Trial, score: f64, epochs: usize) {
        let mut trial = trial;
        trial.complete(score, epochs);
        self.trials.push(trial);
    }

    fn best(&self) -> Option<&Trial> {
        self.trials
            .iter()
            .filter(|t| t.status == TrialStatus::Completed)
            .min_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
    }
}

/// Random searcher (uniform sampling).
pub struct RandomSearcher {
    space: HyperparameterSpace,
    trials: Vec<Trial>,
    next_id: usize,
}

impl RandomSearcher {
    /// Create a random searcher with the given search space.
    pub fn new(space: HyperparameterSpace) -> Self {
        Self { space, trials: Vec::new(), next_id: 0 }
    }
}

impl TuneSearcher for RandomSearcher {
    fn suggest(&mut self) -> crate::Result<Trial> {
        if self.space.is_empty() {
            return Err(crate::Error::ConfigError("Empty search space".to_string()));
        }
        let mut rng = rand::rng();
        let config = self.space.sample_random(&mut rng);
        let trial = Trial::new(self.next_id, config);
        self.next_id += 1;
        Ok(trial)
    }

    fn record(&mut self, trial: Trial, score: f64, epochs: usize) {
        let mut trial = trial;
        trial.complete(score, epochs);
        self.trials.push(trial);
    }

    fn best(&self) -> Option<&Trial> {
        self.trials
            .iter()
            .filter(|t| t.status == TrialStatus::Completed)
            .min_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Scheduler implementations
// ═══════════════════════════════════════════════════════════════════════

/// ASHA-style scheduler: stops trials whose val_loss exceeds the median at the same epoch.
pub struct AshaScheduler {
    /// Grace period: minimum epochs before pruning is eligible.
    grace_period: usize,
    /// Reduction factor (keep top 1/eta at each rung).
    reduction_factor: f64,
    /// Recorded metrics per trial: trial_id → vec of val_loss per epoch.
    history: Vec<Vec<f64>>,
}

impl AshaScheduler {
    /// Create an ASHA scheduler.
    pub fn new(grace_period: usize, reduction_factor: f64) -> Self {
        Self { grace_period, reduction_factor: reduction_factor.max(2.0), history: Vec::new() }
    }

    /// Record a metric for a trial at a given epoch.
    pub fn record_metric(&mut self, trial_id: usize, _epoch: usize, val_loss: f64) {
        while self.history.len() <= trial_id {
            self.history.push(Vec::new());
        }
        self.history[trial_id].push(val_loss);
    }
}

impl TuneScheduler for AshaScheduler {
    fn should_stop(&self, _trial_id: usize, epoch: usize, val_loss: f64) -> bool {
        if epoch < self.grace_period {
            return false;
        }

        // Collect all completed trials' val_loss at this epoch
        let mut losses_at_epoch: Vec<f64> =
            self.history.iter().filter_map(|h| h.get(epoch).copied()).collect();

        if losses_at_epoch.is_empty() {
            return false;
        }

        losses_at_epoch.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Keep top 1/eta — prune if val_loss is above the cutoff
        let keep_fraction = 1.0 / self.reduction_factor;
        let cutoff_idx = ((losses_at_epoch.len() as f64 * keep_fraction).ceil() as usize).max(1);
        if cutoff_idx >= losses_at_epoch.len() {
            return false;
        }
        let cutoff_val = losses_at_epoch[cutoff_idx];
        val_loss > cutoff_val
    }
}

/// Median scheduler: prunes trials whose metric is worse than the median.
pub struct MedianScheduler {
    /// Minimum epochs before pruning.
    n_warmup: usize,
    /// All recorded metrics: trial_id → vec of val_loss per epoch.
    history: Vec<Vec<f64>>,
}

impl MedianScheduler {
    /// Create a median scheduler.
    pub fn new(n_warmup: usize) -> Self {
        Self { n_warmup, history: Vec::new() }
    }

    /// Record a metric for a trial at a given epoch.
    pub fn record_metric(&mut self, trial_id: usize, _epoch: usize, val_loss: f64) {
        while self.history.len() <= trial_id {
            self.history.push(Vec::new());
        }
        self.history[trial_id].push(val_loss);
    }
}

impl TuneScheduler for MedianScheduler {
    fn should_stop(&self, _trial_id: usize, epoch: usize, val_loss: f64) -> bool {
        if epoch < self.n_warmup {
            return false;
        }

        let mut losses: Vec<f64> =
            self.history.iter().filter_map(|h| h.get(epoch).copied()).collect();

        if losses.len() < 2 {
            return false;
        }

        losses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = losses[losses.len() / 2];
        val_loss > median
    }
}

/// No-op scheduler (never prunes).
pub struct NoScheduler;

impl TuneScheduler for NoScheduler {
    fn should_stop(&self, _trial_id: usize, _epoch: usize, _val_loss: f64) -> bool {
        false
    }
}
