//! `AutoTuner` for automatic hyperparameter optimization.
//!
//! Provides high-level API for tuning any model that implements the Estimator trait.

use std::time::{Duration, Instant};

use crate::automl::params::ParamKey;
use crate::automl::search::{SearchSpace, SearchStrategy, Trial, TrialResult};

/// Callback trait for monitoring optimization progress.
pub trait Callback<P: ParamKey> {
    /// Called at the start of optimization.
    fn on_start(&mut self, _space: &SearchSpace<P>) {}

    /// Called before each trial.
    fn on_trial_start(&mut self, _trial_num: usize, _trial: &Trial<P>) {}

    /// Called after each trial with results.
    fn on_trial_end(&mut self, _trial_num: usize, _result: &TrialResult<P>) {}

    /// Called at the end of optimization.
    fn on_end(&mut self, _best: Option<&TrialResult<P>>) {}

    /// Return true to stop optimization early.
    fn should_stop(&self) -> bool {
        false
    }
}

/// Progress logging callback.
#[derive(Debug, Default)]
pub struct ProgressCallback {
    verbose: bool,
}

impl ProgressCallback {
    /// Create verbose progress callback.
    #[must_use]
    pub fn verbose() -> Self {
        Self { verbose: true }
    }
}

impl<P: ParamKey> Callback<P> for ProgressCallback {
    fn on_trial_end(&mut self, trial_num: usize, result: &TrialResult<P>) {
        if self.verbose {
            println!(
                "Trial {:>3}: score={:.4} params={}",
                trial_num, result.score, result.trial
            );
        }
    }

    fn on_end(&mut self, best: Option<&TrialResult<P>>) {
        if self.verbose {
            if let Some(b) = best {
                println!("\nBest: score={:.4} params={}", b.score, b.trial);
            }
        }
    }
}

/// Early stopping callback.
#[derive(Debug)]
pub struct EarlyStopping {
    patience: usize,
    min_delta: f64,
    trials_without_improvement: usize,
    best_score: f64,
}

impl EarlyStopping {
    /// Create early stopping with patience (number of trials without improvement).
    #[must_use]
    pub fn new(patience: usize) -> Self {
        Self {
            patience,
            min_delta: 1e-4,
            trials_without_improvement: 0,
            best_score: f64::NEG_INFINITY,
        }
    }

    /// Set minimum improvement threshold.
    #[must_use]
    pub fn min_delta(mut self, delta: f64) -> Self {
        self.min_delta = delta;
        self
    }
}

impl<P: ParamKey> Callback<P> for EarlyStopping {
    fn on_trial_end(&mut self, _trial_num: usize, result: &TrialResult<P>) {
        if result.score > self.best_score + self.min_delta {
            self.best_score = result.score;
            self.trials_without_improvement = 0;
        } else {
            self.trials_without_improvement += 1;
        }
    }

    fn should_stop(&self) -> bool {
        self.trials_without_improvement >= self.patience
    }
}

/// Time budget constraint.
#[derive(Debug)]
pub struct TimeBudget {
    budget: Duration,
    start: Option<Instant>,
}

impl TimeBudget {
    /// Create time budget in seconds.
    #[must_use]
    pub fn seconds(secs: u64) -> Self {
        Self {
            budget: Duration::from_secs(secs),
            start: None,
        }
    }

    /// Create time budget in minutes.
    #[must_use]
    pub fn minutes(mins: u64) -> Self {
        Self::seconds(mins * 60)
    }

    /// Elapsed time since start.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start.map_or(Duration::ZERO, |s| s.elapsed())
    }

    /// Remaining time.
    #[must_use]
    pub fn remaining(&self) -> Duration {
        self.budget.saturating_sub(self.elapsed())
    }
}

impl<P: ParamKey> Callback<P> for TimeBudget {
    fn on_start(&mut self, _space: &SearchSpace<P>) {
        self.start = Some(Instant::now());
    }

    fn should_stop(&self) -> bool {
        self.elapsed() >= self.budget
    }
}

/// Result of hyperparameter optimization.
#[derive(Debug, Clone)]
pub struct TuneResult<P: ParamKey> {
    /// Best trial found.
    pub best_trial: Trial<P>,
    /// Best score achieved.
    pub best_score: f64,
    /// All trial results.
    pub history: Vec<TrialResult<P>>,
    /// Total optimization time.
    pub elapsed: Duration,
    /// Number of trials run.
    pub n_trials: usize,
}

/// `AutoTuner` for hyperparameter optimization.
///
/// # Example
///
/// ```ignore
/// use aprender::automl::{AutoTuner, RandomSearch, SearchSpace};
/// use aprender::automl::params::RandomForestParam as RF;
///
/// let space = SearchSpace::new()
///     .add(RF::NEstimators, 10..500)
///     .add(RF::MaxDepth, 2..20);
///
/// let result = AutoTuner::new(RandomSearch::new(100))
///     .time_limit_secs(60)
///     .early_stopping(20)
///     .maximize(space, |trial| {
///         let n = trial.get_usize(&RF::NEstimators).unwrap_or(100);
///         let d = trial.get_usize(&RF::MaxDepth).unwrap_or(5);
///         // Return cross-validation score
///         evaluate_model(n, d)
///     });
///
/// println!("Best: {:?}", result.best_trial);
/// ```
#[allow(missing_debug_implementations)]
pub struct AutoTuner<S, P: ParamKey> {
    strategy: S,
    callbacks: Vec<Box<dyn Callback<P>>>,
    _phantom: std::marker::PhantomData<P>,
}

impl<S, P: ParamKey> AutoTuner<S, P>
where
    S: SearchStrategy<P>,
{
    /// Create new tuner with search strategy.
    pub fn new(strategy: S) -> Self {
        Self {
            strategy,
            callbacks: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Add time limit in seconds.
    #[must_use]
    pub fn time_limit_secs(mut self, secs: u64) -> Self {
        self.callbacks.push(Box::new(TimeBudget::seconds(secs)));
        self
    }

    /// Add time limit in minutes.
    #[must_use]
    pub fn time_limit_mins(mut self, mins: u64) -> Self {
        self.callbacks.push(Box::new(TimeBudget::minutes(mins)));
        self
    }

    /// Add early stopping with patience.
    #[must_use]
    pub fn early_stopping(mut self, patience: usize) -> Self {
        self.callbacks.push(Box::new(EarlyStopping::new(patience)));
        self
    }

    /// Add verbose progress logging.
    #[must_use]
    pub fn verbose(mut self) -> Self {
        self.callbacks.push(Box::new(ProgressCallback::verbose()));
        self
    }

    /// Add custom callback.
    #[must_use]
    pub fn callback(mut self, cb: impl Callback<P> + 'static) -> Self {
        self.callbacks.push(Box::new(cb));
        self
    }

    /// Run optimization to maximize objective.
    pub fn maximize<F>(mut self, space: &SearchSpace<P>, mut objective: F) -> TuneResult<P>
    where
        F: FnMut(&Trial<P>) -> f64,
    {
        let start = Instant::now();

        // Notify callbacks of start
        for cb in &mut self.callbacks {
            cb.on_start(space);
        }

        let mut history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_trial: Option<Trial<P>> = None;
        let mut trial_num = 0;

        loop {
            // Check stopping conditions
            if self.callbacks.iter().any(|cb| cb.should_stop()) {
                break;
            }

            // Get next trial(s)
            let trials = self.strategy.suggest(space, 1);
            if trials.is_empty() {
                break;
            }

            let trial = trials.into_iter().next().expect("should have trial");
            trial_num += 1;

            // Notify trial start
            for cb in &mut self.callbacks {
                cb.on_trial_start(trial_num, &trial);
            }

            // Evaluate
            let score = objective(&trial);

            let result = TrialResult {
                trial: trial.clone(),
                score,
                metrics: std::collections::HashMap::new(),
            };

            // Update best
            if score > best_score {
                best_score = score;
                best_trial = Some(trial);
            }

            // Notify trial end
            for cb in &mut self.callbacks {
                cb.on_trial_end(trial_num, &result);
            }

            // Update strategy with result
            self.strategy.update(std::slice::from_ref(&result));

            history.push(result);
        }

        // Notify end
        let best_result = history.iter().max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for cb in &mut self.callbacks {
            cb.on_end(best_result);
        }

        TuneResult {
            best_trial: best_trial.unwrap_or_else(|| Trial {
                values: std::collections::HashMap::new(),
            }),
            best_score,
            history,
            elapsed: start.elapsed(),
            n_trials: trial_num,
        }
    }

    /// Run optimization to minimize objective.
    pub fn minimize<F>(self, space: &SearchSpace<P>, mut objective: F) -> TuneResult<P>
    where
        F: FnMut(&Trial<P>) -> f64,
    {
        self.maximize(space, move |trial| -objective(trial))
    }
}

#[cfg(test)]
#[path = "tuner_tests.rs"]
mod tests;
