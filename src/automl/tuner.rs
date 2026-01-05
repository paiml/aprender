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
mod tests {
    use super::*;
    use crate::automl::params::RandomForestParam as RF;
    use crate::automl::{RandomSearch, SearchSpace};

    #[test]
    fn test_auto_tuner_basic() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add(RF::NEstimators, 10..100)
            .add(RF::MaxDepth, 2..10);

        let result =
            AutoTuner::new(RandomSearch::new(10).with_seed(42)).maximize(&space, |trial| {
                let n = trial.get_usize(&RF::NEstimators).unwrap_or(50);
                let d = trial.get_usize(&RF::MaxDepth).unwrap_or(5);
                // Simple objective: prefer more trees and moderate depth
                (n as f64 / 100.0) + (1.0 - (d as f64 - 5.0).abs() / 5.0)
            });

        assert_eq!(result.n_trials, 10);
        assert!(result.best_score > 0.0);
    }

    #[test]
    fn test_early_stopping() {
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);

        // Constant objective should trigger early stopping
        let result = AutoTuner::new(RandomSearch::new(100))
            .early_stopping(3)
            .maximize(&space, |_| 0.5);

        // Should stop after patience + 1 trials
        assert!(result.n_trials <= 4);
    }

    #[test]
    #[ignore = "Uses thread::sleep - run with cargo test -- --ignored"]
    fn test_time_budget() {
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);

        let result = AutoTuner::new(RandomSearch::new(1000))
            .time_limit_secs(1)
            .maximize(&space, |_| {
                std::thread::sleep(Duration::from_millis(100));
                1.0
            });

        // Should complete within ~1 second
        assert!(result.elapsed.as_secs() <= 2);
        assert!(result.n_trials < 1000);
    }

    #[test]
    fn test_callbacks() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct CountingCallback {
            count: Arc<AtomicUsize>,
        }

        impl<P: ParamKey> Callback<P> for CountingCallback {
            fn on_trial_end(&mut self, _: usize, _: &TrialResult<P>) {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
        }

        let count = Arc::new(AtomicUsize::new(0));
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);

        let _ = AutoTuner::new(RandomSearch::new(5))
            .callback(CountingCallback {
                count: Arc::clone(&count),
            })
            .maximize(&space, |_| 1.0);

        assert_eq!(count.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn test_minimize() {
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);

        let result =
            AutoTuner::new(RandomSearch::new(10).with_seed(42)).minimize(&space, |trial| {
                let n = trial.get_usize(&RF::NEstimators).unwrap_or(50);
                n as f64 // minimize number of estimators
            });

        assert_eq!(result.n_trials, 10);
        // Best score should be negative (since minimize negates)
        assert!(result.best_score < 0.0);
    }

    #[test]
    fn test_time_limit_mins() {
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);

        // Use 0 minutes which should stop immediately after first trial
        let result = AutoTuner::new(RandomSearch::new(1000))
            .time_limit_mins(0)
            .maximize(&space, |_| 1.0);

        // Should complete very quickly
        assert!(result.n_trials <= 1);
    }

    #[test]
    fn test_verbose_callback() {
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);

        // Just verify it doesn't panic
        let result = AutoTuner::new(RandomSearch::new(2))
            .verbose()
            .maximize(&space, |_| 1.0);

        assert_eq!(result.n_trials, 2);
    }

    #[test]
    fn test_early_stopping_min_delta() {
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);

        // Create early stopping with high min_delta - small improvements won't reset counter
        let early_stop = EarlyStopping::new(2).min_delta(0.5);

        let result = AutoTuner::new(RandomSearch::new(100))
            .callback(early_stop)
            .maximize(&space, |trial| {
                // Very small increments
                let n = trial.get_usize(&RF::NEstimators).unwrap_or(50);
                (n as f64) * 0.001
            });

        // Should stop early due to no significant improvement
        assert!(result.n_trials < 100);
    }

    #[test]
    #[ignore = "Uses thread::sleep - run with cargo test -- --ignored"]
    fn test_time_budget_elapsed_remaining() {
        let mut budget = TimeBudget::seconds(10);

        // Before start, elapsed is zero
        assert_eq!(budget.elapsed(), Duration::ZERO);
        assert_eq!(budget.remaining(), Duration::from_secs(10));

        // Simulate start
        let space: SearchSpace<RF> = SearchSpace::new();
        budget.on_start(&space);

        // After start, elapsed > 0
        std::thread::sleep(Duration::from_millis(10));
        assert!(budget.elapsed() > Duration::ZERO);
        assert!(budget.remaining() < Duration::from_secs(10));
    }

    #[test]
    fn test_progress_callback_default() {
        let _cb = ProgressCallback::default();
        // Should not panic - just tests Default impl
    }

    #[test]
    fn test_callback_default_methods() {
        // Test default implementations of Callback trait methods
        struct NoOpCallback;
        impl Callback<RF> for NoOpCallback {}

        let mut cb = NoOpCallback;
        let space: SearchSpace<RF> = SearchSpace::new();
        let trial: Trial<RF> = Trial {
            values: std::collections::HashMap::new(),
        };
        let result: TrialResult<RF> = TrialResult {
            trial: trial.clone(),
            score: 0.5,
            metrics: std::collections::HashMap::new(),
        };

        // Call all default methods - should not panic
        cb.on_start(&space);
        cb.on_trial_start(1, &trial);
        cb.on_trial_end(1, &result);
        cb.on_end(Some(&result));
        assert!(!cb.should_stop());
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_tune_result_clone() {
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);
        let result = AutoTuner::new(RandomSearch::new(3).with_seed(42)).maximize(&space, |_| 0.5);

        let cloned = result.clone();
        assert_eq!(cloned.n_trials, result.n_trials);
        assert!((cloned.best_score - result.best_score).abs() < 1e-9);
        assert_eq!(cloned.history.len(), result.history.len());
    }

    #[test]
    fn test_tune_result_debug() {
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);
        let result = AutoTuner::new(RandomSearch::new(2).with_seed(42)).maximize(&space, |_| 0.5);

        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("TuneResult"));
        assert!(debug_str.contains("best_score"));
    }

    #[test]
    fn test_progress_callback_non_verbose() {
        let mut cb = ProgressCallback::default();
        let trial: Trial<RF> = Trial {
            values: std::collections::HashMap::new(),
        };
        let result: TrialResult<RF> = TrialResult {
            trial: trial.clone(),
            score: 0.5,
            metrics: std::collections::HashMap::new(),
        };

        // Non-verbose should not print
        cb.on_trial_end(1, &result);
        cb.on_end(Some(&result));
        cb.on_end(None::<&TrialResult<RF>>);
        // Just verify no panic
    }

    #[test]
    fn test_progress_callback_verbose_coverage() {
        let mut cb = ProgressCallback::verbose();
        let trial: Trial<RF> = Trial {
            values: std::collections::HashMap::new(),
        };
        let result: TrialResult<RF> = TrialResult {
            trial,
            score: 0.75,
            metrics: std::collections::HashMap::new(),
        };

        // Verbose prints - just verify no panic
        cb.on_trial_end(1, &result);
        cb.on_end(Some(&result));
        cb.on_end(None::<&TrialResult<RF>>); // Cover None path
    }

    #[test]
    fn test_progress_callback_debug() {
        let cb = ProgressCallback::verbose();
        let debug_str = format!("{:?}", cb);
        assert!(debug_str.contains("ProgressCallback"));
        assert!(debug_str.contains("verbose"));
    }

    #[test]
    fn test_early_stopping_debug() {
        let es = EarlyStopping::new(5).min_delta(0.01);
        let debug_str = format!("{:?}", es);
        assert!(debug_str.contains("EarlyStopping"));
        assert!(debug_str.contains("patience"));
    }

    #[test]
    fn test_early_stopping_improvement_resets() {
        let mut es = EarlyStopping::new(3).min_delta(0.1);
        let trial: Trial<RF> = Trial {
            values: std::collections::HashMap::new(),
        };

        // First trial sets baseline
        let result1 = TrialResult {
            trial: trial.clone(),
            score: 0.5,
            metrics: std::collections::HashMap::new(),
        };
        <EarlyStopping as Callback<RF>>::on_trial_end(&mut es, 1, &result1);
        assert!(!<EarlyStopping as Callback<RF>>::should_stop(&es));

        // No improvement
        let result2 = TrialResult {
            trial: trial.clone(),
            score: 0.55, // Less than min_delta improvement
            metrics: std::collections::HashMap::new(),
        };
        <EarlyStopping as Callback<RF>>::on_trial_end(&mut es, 2, &result2);
        assert!(!<EarlyStopping as Callback<RF>>::should_stop(&es));

        // Significant improvement resets counter
        let result3 = TrialResult {
            trial: trial.clone(),
            score: 1.0,
            metrics: std::collections::HashMap::new(),
        };
        <EarlyStopping as Callback<RF>>::on_trial_end(&mut es, 3, &result3);
        assert!(!<EarlyStopping as Callback<RF>>::should_stop(&es));
        assert_eq!(es.trials_without_improvement, 0);
    }

    #[test]
    fn test_time_budget_debug() {
        let tb = TimeBudget::seconds(60);
        let debug_str = format!("{:?}", tb);
        assert!(debug_str.contains("TimeBudget"));
        assert!(debug_str.contains("budget"));
    }

    #[test]
    fn test_time_budget_before_start() {
        let budget = TimeBudget::seconds(10);

        // Before on_start, elapsed should be zero
        assert_eq!(budget.elapsed(), Duration::ZERO);
        assert_eq!(budget.remaining(), Duration::from_secs(10));
        assert!(!<TimeBudget as Callback<RF>>::should_stop(&budget));
    }

    #[test]
    fn test_time_budget_zero_seconds() {
        let budget = TimeBudget::seconds(0);
        assert_eq!(budget.remaining(), Duration::ZERO);
    }

    #[test]
    fn test_time_budget_minutes_conversion() {
        let budget = TimeBudget::minutes(2);
        assert_eq!(budget.budget, Duration::from_secs(120));
    }

    #[test]
    fn test_time_budget_on_start() {
        let mut budget = TimeBudget::seconds(100);
        let space: SearchSpace<RF> = SearchSpace::new();

        assert!(budget.start.is_none());
        budget.on_start(&space);
        assert!(budget.start.is_some());
    }

    #[test]
    fn test_auto_tuner_empty_space() {
        let space: SearchSpace<RF> = SearchSpace::new();

        // Empty search space should complete quickly
        let result = AutoTuner::new(RandomSearch::new(10)).maximize(&space, |_| 1.0);

        // Should still run trials even with empty params
        assert!(result.n_trials <= 10);
    }

    #[test]
    fn test_auto_tuner_single_trial() {
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);

        let result = AutoTuner::new(RandomSearch::new(1)).maximize(&space, |_| 0.5);

        assert_eq!(result.n_trials, 1);
        assert_eq!(result.history.len(), 1);
    }

    #[test]
    fn test_auto_tuner_best_trial_tracking() {
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);

        // Objective increases with estimators
        let result = AutoTuner::new(RandomSearch::new(20).with_seed(42))
            .maximize(&space, |trial| {
                trial.get_usize(&RF::NEstimators).unwrap_or(50) as f64
            });

        // Best trial should have the highest n_estimators
        assert!(result.best_score >= 10.0);
        assert!(result.best_score <= 100.0);
    }

    #[test]
    fn test_auto_tuner_with_multiple_callbacks() {
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);

        // Add multiple callbacks
        let result = AutoTuner::new(RandomSearch::new(5))
            .early_stopping(10)
            .verbose()
            .maximize(&space, |_| 0.5);

        assert_eq!(result.n_trials, 5);
    }

    #[test]
    fn test_callback_on_trial_start() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct TrialStartCounter {
            count: Arc<AtomicUsize>,
        }

        impl<P: ParamKey> Callback<P> for TrialStartCounter {
            fn on_trial_start(&mut self, _trial_num: usize, _trial: &Trial<P>) {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
        }

        let count = Arc::new(AtomicUsize::new(0));
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);

        let _ = AutoTuner::new(RandomSearch::new(5))
            .callback(TrialStartCounter {
                count: Arc::clone(&count),
            })
            .maximize(&space, |_| 1.0);

        assert_eq!(count.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn test_callback_on_start_called() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        struct StartTracker {
            started: Arc<AtomicBool>,
        }

        impl<P: ParamKey> Callback<P> for StartTracker {
            fn on_start(&mut self, _space: &SearchSpace<P>) {
                self.started.store(true, Ordering::SeqCst);
            }
        }

        let started = Arc::new(AtomicBool::new(false));
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);

        let _ = AutoTuner::new(RandomSearch::new(1))
            .callback(StartTracker {
                started: Arc::clone(&started),
            })
            .maximize(&space, |_| 1.0);

        assert!(started.load(Ordering::SeqCst));
    }

    #[test]
    fn test_callback_on_end_called() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        struct EndTracker {
            ended: Arc<AtomicBool>,
        }

        impl<P: ParamKey> Callback<P> for EndTracker {
            fn on_end(&mut self, _best: Option<&TrialResult<P>>) {
                self.ended.store(true, Ordering::SeqCst);
            }
        }

        let ended = Arc::new(AtomicBool::new(false));
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);

        let _ = AutoTuner::new(RandomSearch::new(1))
            .callback(EndTracker {
                ended: Arc::clone(&ended),
            })
            .maximize(&space, |_| 1.0);

        assert!(ended.load(Ordering::SeqCst));
    }

    #[test]
    fn test_stopping_callback() {
        struct ImmediateStop;

        impl<P: ParamKey> Callback<P> for ImmediateStop {
            fn should_stop(&self) -> bool {
                true
            }
        }

        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);

        let result = AutoTuner::new(RandomSearch::new(100))
            .callback(ImmediateStop)
            .maximize(&space, |_| 1.0);

        // Should stop immediately due to callback
        assert_eq!(result.n_trials, 0);
    }

    #[test]
    fn test_tune_result_elapsed_duration() {
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);
        let result = AutoTuner::new(RandomSearch::new(5).with_seed(42)).maximize(&space, |_| 0.5);

        // Elapsed should be non-zero but small
        assert!(result.elapsed.as_nanos() > 0);
    }

    #[test]
    fn test_tune_result_history_scores() {
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);
        let result = AutoTuner::new(RandomSearch::new(5).with_seed(42)).maximize(&space, |_| 0.5);

        // All history entries should have the same score
        for entry in &result.history {
            assert!((entry.score - 0.5).abs() < 1e-9);
        }
    }
}
