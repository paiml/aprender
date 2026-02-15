
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
    let result = AutoTuner::new(RandomSearch::new(20).with_seed(42)).minimize(&space, |trial| {
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
