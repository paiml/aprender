use std::collections::HashMap;

use super::*;
use crate::optim::{HyperparameterSpace, ParameterDomain, ParameterValue};

#[test]
fn test_tune_config_default() {
    let config = TuneConfig::default();
    assert_eq!(config.budget, 10);
    assert_eq!(config.strategy, TuneStrategy::Tpe);
    assert_eq!(config.scheduler, SchedulerKind::Asha);
    assert!(!config.scout);
    assert_eq!(config.max_epochs, 20);
    assert_eq!(config.num_classes, 5);
    assert_eq!(config.seed, 42);
}

#[test]
fn test_classify_tuner_new() {
    let config = TuneConfig::default();
    let tuner = ClassifyTuner::new(config).expect("valid");
    assert!(tuner.leaderboard.is_empty());
    assert!(!tuner.space.is_empty());
    assert_eq!(tuner.space.len(), 9); // 9 search parameters
}

#[test]
fn test_falsify_tune_001_budget_zero() {
    let config = TuneConfig { budget: 0, ..TuneConfig::default() };
    let result = ClassifyTuner::new(config);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("FALSIFY-TUNE-001"), "Expected FALSIFY-TUNE-001, got: {err}");
}

#[test]
fn test_falsify_tune_004_num_classes_zero() {
    let config = TuneConfig { num_classes: 0, ..TuneConfig::default() };
    let result = ClassifyTuner::new(config);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("FALSIFY-TUNE-004"), "Expected FALSIFY-TUNE-004, got: {err}");
}

#[test]
fn test_tpe_searcher_suggest_and_record() {
    let space = default_classify_search_space();
    let mut searcher = TpeSearcher::new(space, 3);

    // Should be able to suggest multiple trials
    let trial1 = searcher.suggest().expect("valid");
    assert_eq!(trial1.id, 0);
    assert!(!trial1.config.is_empty());

    let trial2 = searcher.suggest().expect("valid");
    assert_eq!(trial2.id, 1);

    // Record results
    searcher.record(trial1, 0.5, 1);
    searcher.record(trial2, 0.3, 1);

    // Best should be trial2 (lower score)
    let best = searcher.best().expect("valid");
    assert_eq!(best.id, 1);
    assert!((best.score - 0.3).abs() < 1e-10);
}

#[test]
fn test_grid_searcher() {
    let mut space = HyperparameterSpace::new();
    space.add("lr", ParameterDomain::Continuous { low: 1e-4, high: 1e-2, log_scale: true });
    space.add(
        "act",
        ParameterDomain::Categorical { choices: vec!["relu".to_string(), "gelu".to_string()] },
    );

    let mut searcher = GridSearcher::new(space, 3);

    // 3 lr * 2 act = 6 configs
    let mut count = 0;
    while searcher.suggest().is_ok() {
        count += 1;
        if count > 10 {
            break; // safety
        }
    }
    assert_eq!(count, 6);
}

#[test]
fn test_random_searcher() {
    let space = default_classify_search_space();
    let mut searcher = RandomSearcher::new(space);

    let t1 = searcher.suggest().expect("valid");
    let t2 = searcher.suggest().expect("valid");
    assert_ne!(t1.id, t2.id);

    searcher.record(t1, 1.0, 1);
    searcher.record(t2, 0.5, 1);

    let best = searcher.best().expect("valid");
    assert!((best.score - 0.5).abs() < 1e-10);
}

#[test]
fn test_no_scheduler_never_stops() {
    let scheduler = NoScheduler;
    assert!(!scheduler.should_stop(0, 0, 100.0));
    assert!(!scheduler.should_stop(0, 100, 100.0));
}

#[test]
fn test_asha_scheduler_grace_period() {
    let scheduler = AshaScheduler::new(2, 3.0);
    // Before grace period, should never stop
    assert!(!scheduler.should_stop(0, 0, 100.0));
    assert!(!scheduler.should_stop(0, 1, 100.0));
}

#[test]
fn test_median_scheduler_warmup() {
    let scheduler = MedianScheduler::new(3);
    assert!(!scheduler.should_stop(0, 0, 100.0));
    assert!(!scheduler.should_stop(0, 2, 100.0));
}

#[test]
fn test_strategy_parse() {
    assert_eq!("tpe".parse::<TuneStrategy>().expect("valid"), TuneStrategy::Tpe);
    assert_eq!("grid".parse::<TuneStrategy>().expect("valid"), TuneStrategy::Grid);
    assert_eq!("random".parse::<TuneStrategy>().expect("valid"), TuneStrategy::Random);
    assert_eq!("bayesian".parse::<TuneStrategy>().expect("valid"), TuneStrategy::Tpe);
    assert!("invalid".parse::<TuneStrategy>().is_err());
}

#[test]
fn test_scheduler_kind_parse() {
    assert_eq!("asha".parse::<SchedulerKind>().expect("valid"), SchedulerKind::Asha);
    assert_eq!("median".parse::<SchedulerKind>().expect("valid"), SchedulerKind::Median);
    assert_eq!("none".parse::<SchedulerKind>().expect("valid"), SchedulerKind::None);
    assert!("invalid".parse::<SchedulerKind>().is_err());
}

#[test]
fn test_extract_trial_params() {
    let mut config = HashMap::new();
    config.insert("learning_rate".to_string(), ParameterValue::Float(3.2e-5));
    config.insert("lora_rank".to_string(), ParameterValue::Int(8)); // 8*4=32
    config.insert("lora_alpha_ratio".to_string(), ParameterValue::Float(1.5));
    config.insert("batch_size".to_string(), ParameterValue::Categorical("64".to_string()));
    config.insert("warmup_fraction".to_string(), ParameterValue::Float(0.08));
    config.insert("gradient_clip_norm".to_string(), ParameterValue::Float(1.5));
    config.insert(
        "class_weights".to_string(),
        ParameterValue::Categorical("sqrt_inverse".to_string()),
    );
    config.insert("target_modules".to_string(), ParameterValue::Categorical("qv".to_string()));
    config.insert("lr_min_ratio".to_string(), ParameterValue::Float(0.01));

    let (lr, rank, alpha, batch, warmup, clip, weights, targets, lr_min) =
        extract_trial_params(&config);

    assert!((lr - 3.2e-5).abs() < 1e-8);
    assert_eq!(rank, 32); // 8 * 4
    assert!((alpha - 48.0).abs() < 1e-3); // 32 * 1.5
    assert_eq!(batch, 64);
    assert!((warmup - 0.08).abs() < 1e-5);
    assert!((clip - 1.5).abs() < 1e-5);
    assert_eq!(weights, "sqrt_inverse");
    assert_eq!(targets, "qv");
    assert!((lr_min - 0.01).abs() < 1e-5);
}

#[test]
fn test_tuner_leaderboard() {
    let config = TuneConfig { budget: 3, ..TuneConfig::default() };
    let mut tuner = ClassifyTuner::new(config).expect("valid");

    // Add trials out of order
    tuner.record_trial(TrialSummary {
        id: 0,
        val_loss: 1.5,
        val_accuracy: 0.70,
        train_loss: 1.0,
        train_accuracy: 0.85,
        epochs_run: 1,
        time_ms: 1000,
        config: HashMap::new(),
        status: "completed".to_string(),
    });
    tuner.record_trial(TrialSummary {
        id: 1,
        val_loss: 0.8,
        val_accuracy: 0.86,
        train_loss: 0.5,
        train_accuracy: 0.92,
        epochs_run: 1,
        time_ms: 1000,
        config: HashMap::new(),
        status: "completed".to_string(),
    });
    tuner.record_trial(TrialSummary {
        id: 2,
        val_loss: 1.2,
        val_accuracy: 0.75,
        train_loss: 0.8,
        train_accuracy: 0.88,
        epochs_run: 1,
        time_ms: 1000,
        config: HashMap::new(),
        status: "completed".to_string(),
    });

    // Leaderboard should be sorted: trial 1 (0.8) < trial 2 (1.2) < trial 0 (1.5)
    assert_eq!(tuner.leaderboard[0].id, 1);
    assert_eq!(tuner.leaderboard[1].id, 2);
    assert_eq!(tuner.leaderboard[2].id, 0);

    let best = tuner.best_trial().expect("valid");
    assert_eq!(best.id, 1);
    assert!((best.val_loss - 0.8).abs() < 1e-10);
}

#[test]
fn test_tuner_into_result() {
    let config = TuneConfig { budget: 2, scout: true, ..TuneConfig::default() };
    let mut tuner = ClassifyTuner::new(config).expect("valid");

    tuner.record_trial(TrialSummary {
        id: 0,
        val_loss: 1.0,
        val_accuracy: 0.75,
        train_loss: 0.8,
        train_accuracy: 0.85,
        epochs_run: 1,
        time_ms: 5000,
        config: HashMap::new(),
        status: "completed".to_string(),
    });

    let result = tuner.into_result(10000);
    assert_eq!(result.strategy, "tpe");
    assert_eq!(result.mode, "scout");
    assert_eq!(result.budget, 2);
    assert_eq!(result.trials.len(), 1);
    assert_eq!(result.best_trial_id, 0);
    assert_eq!(result.total_time_ms, 10000);
}

#[test]
fn test_default_search_space() {
    let space = default_classify_search_space();
    assert_eq!(space.len(), 9);

    // Verify all parameters exist
    assert!(space.get("learning_rate").is_some());
    assert!(space.get("lora_rank").is_some());
    assert!(space.get("lora_alpha_ratio").is_some());
    assert!(space.get("batch_size").is_some());
    assert!(space.get("warmup_fraction").is_some());
    assert!(space.get("gradient_clip_norm").is_some());
    assert!(space.get("class_weights").is_some());
    assert!(space.get("target_modules").is_some());
    assert!(space.get("lr_min_ratio").is_some());
}

#[test]
fn test_build_searcher_tpe() {
    let config = TuneConfig { strategy: TuneStrategy::Tpe, ..TuneConfig::default() };
    let tuner = ClassifyTuner::new(config).expect("valid");
    let mut searcher = tuner.build_searcher();
    let trial = searcher.suggest().expect("valid");
    assert_eq!(trial.id, 0);
}

#[test]
fn test_build_scheduler_scout_uses_none() {
    let config =
        TuneConfig { scout: true, scheduler: SchedulerKind::Asha, ..TuneConfig::default() };
    let tuner = ClassifyTuner::new(config).expect("valid");
    let scheduler = tuner.build_scheduler();
    // Scout mode always uses NoScheduler
    assert!(!scheduler.should_stop(0, 100, 999.0));
}

// ── ASHA scheduler coverage tests ──────────────────────────────

#[test]
fn test_asha_scheduler_prunes_bad_trial() {
    let mut scheduler = AshaScheduler::new(1, 3.0);
    // record_metric pushes to per-trial vec; should_stop indexes by epoch.
    // Record 6 trials at epochs 0 and 1 (2 entries each so get(1) works)
    for trial_id in 0..6 {
        let loss = (trial_id + 1) as f64 * 0.5;
        scheduler.record_metric(trial_id, 0, loss + 1.0); // epoch 0 (filler)
        scheduler.record_metric(trial_id, 1, loss); // epoch 1: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
    }
    // At epoch 1 (past grace period 1), a trial with very high loss should be pruned
    assert!(scheduler.should_stop(6, 1, 10.0));
    // A trial with very low loss should NOT be pruned
    assert!(!scheduler.should_stop(6, 1, 0.1));
}

#[test]
fn test_asha_scheduler_empty_history_no_prune() {
    let scheduler = AshaScheduler::new(1, 3.0);
    // No history recorded — should not prune
    assert!(!scheduler.should_stop(0, 5, 100.0));
}

#[test]
fn test_asha_scheduler_reduction_factor_clamped() {
    // reduction_factor < 2.0 should be clamped to 2.0
    let scheduler = AshaScheduler::new(0, 0.5);
    assert!(!scheduler.should_stop(0, 0, 1.0));
}

#[test]
fn test_asha_scheduler_cutoff_at_boundary() {
    // grace_period=0 so pruning starts immediately at epoch 0
    let mut scheduler = AshaScheduler::new(0, 2.0);
    // 4 trials with losses at epoch 0 to get enough data for pruning
    scheduler.record_metric(0, 0, 1.0);
    scheduler.record_metric(1, 0, 2.0);
    scheduler.record_metric(2, 0, 3.0);
    scheduler.record_metric(3, 0, 4.0);
    // With reduction_factor 2.0, keep top 50%: cutoff_idx=2, cutoff_val=3.0
    // Loss exactly at cutoff should NOT be pruned (not strictly greater)
    assert!(!scheduler.should_stop(4, 0, 3.0));
    // Loss above cutoff should be pruned
    assert!(scheduler.should_stop(4, 0, 3.1));
}

// ── Median scheduler coverage tests ────────────────────────────

#[test]
fn test_median_scheduler_prunes_above_median() {
    let mut scheduler = MedianScheduler::new(0);
    // Record 4 trials at epoch 0: losses 1.0, 2.0, 3.0, 4.0
    for trial_id in 0..4 {
        scheduler.record_metric(trial_id, 0, (trial_id + 1) as f64);
    }
    // Median is losses[2] = 3.0
    // Loss above median should be pruned
    assert!(scheduler.should_stop(4, 0, 3.5));
    // Loss below median should not be pruned
    assert!(!scheduler.should_stop(4, 0, 2.5));
}

#[test]
fn test_median_scheduler_insufficient_history() {
    let mut scheduler = MedianScheduler::new(0);
    // Only 1 trial recorded — need at least 2
    scheduler.record_metric(0, 0, 1.0);
    assert!(!scheduler.should_stop(1, 0, 100.0));
}

#[test]
fn test_median_scheduler_record_multiple_epochs() {
    let mut scheduler = MedianScheduler::new(1);
    // Record trial 0: epoch 0 = 2.0, epoch 1 = 1.5
    scheduler.record_metric(0, 0, 2.0);
    scheduler.record_metric(0, 1, 1.5);
    // Record trial 1: epoch 0 = 1.0, epoch 1 = 0.8
    scheduler.record_metric(1, 0, 1.0);
    scheduler.record_metric(1, 1, 0.8);
    // At epoch 1 (past warmup), median of [1.5, 0.8] = 1.5
    assert!(scheduler.should_stop(2, 1, 2.0));
    assert!(!scheduler.should_stop(2, 1, 0.5));
}

// ── Grid searcher edge cases ───────────────────────────────────

#[test]
fn test_grid_searcher_record_and_best() {
    let mut space = HyperparameterSpace::new();
    space.add("lr", ParameterDomain::Continuous { low: 0.01, high: 0.1, log_scale: false });
    let mut searcher = GridSearcher::new(space, 3);

    let t1 = searcher.suggest().expect("valid");
    let t2 = searcher.suggest().expect("valid");
    let t3 = searcher.suggest().expect("valid");

    searcher.record(t1, 2.0, 5);
    searcher.record(t2, 0.5, 5);
    searcher.record(t3, 1.0, 5);

    let best = searcher.best().expect("valid");
    assert!((best.score - 0.5).abs() < 1e-10);
}

#[test]
fn test_grid_searcher_exhausted() {
    let mut space = HyperparameterSpace::new();
    space.add("act", ParameterDomain::Categorical { choices: vec!["relu".to_string()] });
    let mut searcher = GridSearcher::new(space, 1);

    // Only 1 config: should succeed once, then fail
    assert!(searcher.suggest().is_ok());
    assert!(searcher.suggest().is_err());
}

// ── Random searcher edge cases ─────────────────────────────────

#[test]
fn test_random_searcher_empty_space() {
    let space = HyperparameterSpace::new(); // empty
    let mut searcher = RandomSearcher::new(space);
    assert!(searcher.suggest().is_err());
}

// ── Strategy/Scheduler display ─────────────────────────────────

#[test]
fn test_strategy_display() {
    assert_eq!(format!("{}", TuneStrategy::Tpe), "tpe");
    assert_eq!(format!("{}", TuneStrategy::Grid), "grid");
    assert_eq!(format!("{}", TuneStrategy::Random), "random");
}

// ── ClassifyTuner scheduler variants ───────────────────────────

#[test]
fn test_build_scheduler_median() {
    let config = TuneConfig { scheduler: SchedulerKind::Median, ..TuneConfig::default() };
    let tuner = ClassifyTuner::new(config).expect("valid");
    let scheduler = tuner.build_scheduler();
    // Median scheduler with no history should not prune
    assert!(!scheduler.should_stop(0, 10, 999.0));
}

#[test]
fn test_build_scheduler_none() {
    let config = TuneConfig { scheduler: SchedulerKind::None, ..TuneConfig::default() };
    let tuner = ClassifyTuner::new(config).expect("valid");
    let scheduler = tuner.build_scheduler();
    assert!(!scheduler.should_stop(0, 100, 999.0));
}

#[test]
fn test_build_searcher_grid() {
    let config = TuneConfig { strategy: TuneStrategy::Grid, ..TuneConfig::default() };
    let tuner = ClassifyTuner::new(config).expect("valid");
    let mut searcher = tuner.build_searcher();
    let trial = searcher.suggest().expect("valid");
    assert_eq!(trial.id, 0);
}

#[test]
fn test_build_searcher_random() {
    let config = TuneConfig { strategy: TuneStrategy::Random, ..TuneConfig::default() };
    let tuner = ClassifyTuner::new(config).expect("valid");
    let mut searcher = tuner.build_searcher();
    let trial = searcher.suggest().expect("valid");
    assert_eq!(trial.id, 0);
}

// ── Extract trial params edge cases ────────────────────────────

#[test]
fn test_extract_trial_params_defaults() {
    // Empty config — all defaults
    let config = HashMap::new();
    let (lr, rank, alpha, batch, warmup, clip, weights, targets, lr_min) =
        extract_trial_params(&config);

    assert!((lr - 1e-4).abs() < 1e-8); // default
    assert_eq!(rank, 16); // 4*4=16
    assert_eq!(batch, 32);
    assert!((warmup - 0.1).abs() < 1e-5);
    assert!((clip - 1.0).abs() < 1e-5);
    assert_eq!(weights, "uniform");
    assert_eq!(targets, "qv");
    assert!((lr_min - 0.01).abs() < 1e-5);
    // alpha = rank * alpha_ratio = 16 * 1.0 = 16.0
    assert!((alpha - 16.0).abs() < 1e-3);
}

#[test]
fn test_extract_trial_params_rank_clamping() {
    let mut config = HashMap::new();
    // rank_raw=0 → 0*4=0 → clamped to 4
    config.insert("lora_rank".to_string(), ParameterValue::Int(0));
    let (_, rank, _, _, _, _, _, _, _) = extract_trial_params(&config);
    assert_eq!(rank, 4);

    // rank_raw=20 → 20*4=80 → clamped to 64
    config.insert("lora_rank".to_string(), ParameterValue::Int(20));
    let (_, rank, _, _, _, _, _, _, _) = extract_trial_params(&config);
    assert_eq!(rank, 64);
}

// ── Falsification tests (SPEC-TUNE-2026-001 §7) ───────────────

#[test]
fn test_falsify_tune_002_invalid_strategy() {
    let result = "superbayesian".parse::<TuneStrategy>();
    assert!(result.is_err(), "FALSIFY-TUNE-002: invalid strategy should fail");
    let err = result.unwrap_err();
    assert!(err.contains("Unknown strategy"), "Should contain 'Unknown strategy', got: {err}");
}

#[test]
fn test_falsify_tune_005_invalid_scheduler() {
    let result = "hyperband99".parse::<SchedulerKind>();
    assert!(result.is_err(), "FALSIFY-TUNE-005: invalid scheduler should fail");
}

#[test]
fn test_falsify_empty_leaderboard() {
    let config = TuneConfig { budget: 1, ..TuneConfig::default() };
    let tuner = ClassifyTuner::new(config).expect("valid");
    assert!(tuner.best_trial().is_none(), "Empty leaderboard should return None");
}

#[test]
fn test_falsify_into_result_empty() {
    let config = TuneConfig { budget: 1, ..TuneConfig::default() };
    let tuner = ClassifyTuner::new(config).expect("valid");
    let result = tuner.into_result(0);
    assert_eq!(result.trials.len(), 0);
    assert_eq!(result.best_trial_id, 0); // default when empty
}

#[test]
fn test_falsify_strategy_case_insensitive() {
    assert_eq!("TPE".parse::<TuneStrategy>().expect("valid"), TuneStrategy::Tpe);
    assert_eq!("Grid".parse::<TuneStrategy>().expect("valid"), TuneStrategy::Grid);
    assert_eq!("RANDOM".parse::<TuneStrategy>().expect("valid"), TuneStrategy::Random);
}

#[test]
fn test_falsify_scheduler_case_insensitive() {
    assert_eq!("ASHA".parse::<SchedulerKind>().expect("valid"), SchedulerKind::Asha);
    assert_eq!("Median".parse::<SchedulerKind>().expect("valid"), SchedulerKind::Median);
    assert_eq!("NONE".parse::<SchedulerKind>().expect("valid"), SchedulerKind::None);
}
