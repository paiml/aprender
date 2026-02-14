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

// =========================================================================
// Coverage gap tests (targeting 27 missed lines)
// =========================================================================

#[test]
fn test_maximize_zero_trials_fallback() {
    // Covers the best_trial.unwrap_or_else fallback (line 321-323)
    // when no trials are run at all
    struct InstantStop;
    impl<P: ParamKey> Callback<P> for InstantStop {
        fn should_stop(&self) -> bool {
            true
        }
    }

    let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);
    let result = AutoTuner::new(RandomSearch::new(100))
        .callback(InstantStop)
        .maximize(&space, |_| 1.0);

    // No trials means best_trial falls back to empty
    assert_eq!(result.n_trials, 0);
    assert_eq!(result.best_score, f64::NEG_INFINITY);
    assert!(result.best_trial.values.is_empty());
    assert!(result.history.is_empty());
}

#[test]
fn test_maximize_on_end_with_none_best() {
    // When history is empty, on_end gets None
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    struct NoneChecker {
        got_none: Arc<AtomicBool>,
    }

    impl<P: ParamKey> Callback<P> for NoneChecker {
        fn should_stop(&self) -> bool {
            true // Stop immediately so no trials run
        }
        fn on_end(&mut self, best: Option<&TrialResult<P>>) {
            if best.is_none() {
                self.got_none.store(true, Ordering::SeqCst);
            }
        }
    }

    let got_none = Arc::new(AtomicBool::new(false));
    let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);

    let _ = AutoTuner::new(RandomSearch::new(100))
        .callback(NoneChecker {
            got_none: Arc::clone(&got_none),
        })
        .maximize(&space, |_| 1.0);

    assert!(
        got_none.load(Ordering::SeqCst),
        "on_end should receive None when no trials run"
    );
}

#[test]
fn test_early_stopping_exact_patience_boundary() {
    // Test that exactly patience trials without improvement triggers stop
    let mut es = EarlyStopping::new(3).min_delta(0.0);
    let trial: Trial<RF> = Trial {
        values: std::collections::HashMap::new(),
    };

    // Set initial baseline
    let r0 = TrialResult {
        trial: trial.clone(),
        score: 1.0,
        metrics: std::collections::HashMap::new(),
    };
    <EarlyStopping as Callback<RF>>::on_trial_end(&mut es, 1, &r0);
    assert!(!<EarlyStopping as Callback<RF>>::should_stop(&es));

    // 1 without improvement
    let r1 = TrialResult {
        trial: trial.clone(),
        score: 0.5,
        metrics: std::collections::HashMap::new(),
    };
    <EarlyStopping as Callback<RF>>::on_trial_end(&mut es, 2, &r1);
    assert!(!<EarlyStopping as Callback<RF>>::should_stop(&es));
    assert_eq!(es.trials_without_improvement, 1);

    // 2 without improvement
    <EarlyStopping as Callback<RF>>::on_trial_end(&mut es, 3, &r1);
    assert!(!<EarlyStopping as Callback<RF>>::should_stop(&es));
    assert_eq!(es.trials_without_improvement, 2);

    // 3 without improvement - should now stop
    <EarlyStopping as Callback<RF>>::on_trial_end(&mut es, 4, &r1);
    assert!(<EarlyStopping as Callback<RF>>::should_stop(&es));
    assert_eq!(es.trials_without_improvement, 3);
}

#[test]
fn test_time_budget_should_stop_after_expired() {
    // Start a budget of 0 seconds, then check should_stop
    let mut budget = TimeBudget::seconds(0);
    let space: SearchSpace<RF> = SearchSpace::new();

    <TimeBudget as Callback<RF>>::on_start(&mut budget, &space);

    // With 0-second budget, should_stop immediately
    assert!(<TimeBudget as Callback<RF>>::should_stop(&budget));
}

#[test]
fn test_time_budget_remaining_after_start() {
    let mut budget = TimeBudget::seconds(1000);
    let space: SearchSpace<RF> = SearchSpace::new();

    budget.on_start(&space);
    // Immediately after start, remaining should be close to budget
    let remaining = budget.remaining();
    assert!(remaining.as_secs() >= 999);
}

#[test]
fn test_minimize_selects_lowest() {
    let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);

    // Minimize: n_estimators value (lower is better)
    let result = AutoTuner::new(RandomSearch::new(20).with_seed(42))
        .minimize(&space, |trial| {
            trial.get_usize(&RF::NEstimators).unwrap_or(50) as f64
        });

    assert_eq!(result.n_trials, 20);
    // The best_score is negated internally, so it's negative of the actual minimum
    assert!(result.best_score < 0.0);
}

#[test]
fn test_progress_callback_verbose_on_end_none() {
    // Covers the verbose on_end(None) path (line 56-59)
    let mut cb = ProgressCallback::verbose();
    cb.on_end(None::<&TrialResult<RF>>);
    // Should not panic (None means no best result to print)
}

#[test]
fn test_early_stopping_with_improvement_then_stagnation() {
    let mut es = EarlyStopping::new(2).min_delta(0.1);
    let trial: Trial<RF> = Trial {
        values: std::collections::HashMap::new(),
    };

    // Improving sequence
    for score in [0.1, 0.3, 0.5, 0.7, 0.9] {
        let r = TrialResult {
            trial: trial.clone(),
            score,
            metrics: std::collections::HashMap::new(),
        };
        <EarlyStopping as Callback<RF>>::on_trial_end(&mut es, 1, &r);
        assert!(!<EarlyStopping as Callback<RF>>::should_stop(&es));
    }

    // Stagnation: score drops and doesn't recover
    for _ in 0..3 {
        let r = TrialResult {
            trial: trial.clone(),
            score: 0.85, // Below best (0.9) + min_delta (0.1) = 1.0
            metrics: std::collections::HashMap::new(),
        };
        <EarlyStopping as Callback<RF>>::on_trial_end(&mut es, 1, &r);
    }

    assert!(<EarlyStopping as Callback<RF>>::should_stop(&es));
}

#[test]
fn test_auto_tuner_maximize_with_nan_scores() {
    // NaN scores from the objective should be handled
    let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);

    let mut count = 0;
    let result = AutoTuner::new(RandomSearch::new(5).with_seed(42)).maximize(&space, |_| {
        count += 1;
        if count == 3 {
            f64::NAN
        } else {
            0.5
        }
    });

    assert_eq!(result.n_trials, 5);
}

#[test]
fn test_auto_tuner_maximize_descending_scores() {
    // Test that best tracking works when best appears early
    let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);

    let mut call = 0;
    let result = AutoTuner::new(RandomSearch::new(5).with_seed(42)).maximize(&space, |_| {
        call += 1;
        // First trial has highest score
        if call == 1 {
            1.0
        } else {
            0.1
        }
    });

    assert!((result.best_score - 1.0).abs() < 1e-9);
}

#[test]
fn test_tune_result_best_trial_has_values() {
    let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);

    let result = AutoTuner::new(RandomSearch::new(1).with_seed(42)).maximize(&space, |_| 0.5);

    // Best trial should have actual values
    assert!(!result.best_trial.values.is_empty());
    assert!(result.best_trial.get(&RF::NEstimators).is_some());
}

#[test]
fn test_auto_tuner_time_limit_mins_with_immediate_stop() {
    // Covers time_limit_mins path (line 220-224)
    let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);
    let result = AutoTuner::new(RandomSearch::new(1000))
        .time_limit_mins(0)
        .maximize(&space, |_| 0.5);

    // 0 minutes = stop almost immediately
    assert!(result.n_trials <= 1);
}
