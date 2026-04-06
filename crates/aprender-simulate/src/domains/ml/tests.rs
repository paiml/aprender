use super::*;

// -------------------------------------------------------------------------
// RollingStats Tests
// -------------------------------------------------------------------------

#[test]
fn test_rolling_stats_empty() {
    let stats = RollingStats::new(0);
    assert_eq!(stats.mean(), 0.0);
    assert_eq!(stats.variance(), 0.0);
    assert_eq!(stats.std_dev(), 0.0);
}

#[test]
fn test_rolling_stats_single_value() {
    let mut stats = RollingStats::new(0);
    stats.update(5.0);
    assert!((stats.mean() - 5.0).abs() < 1e-10);
    assert_eq!(stats.variance(), 0.0); // n-1 variance with n=1
}

#[test]
fn test_rolling_stats_multiple_values() {
    let mut stats = RollingStats::new(0);
    for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
        stats.update(v);
    }
    assert!((stats.mean() - 5.0).abs() < 1e-10);
    assert!((stats.variance() - 4.571_428_571_428_571).abs() < 1e-10);
}

#[test]
fn test_rolling_stats_z_score() {
    let mut stats = RollingStats::new(0);
    for v in [10.0, 10.0, 10.0, 10.0, 10.0] {
        stats.update(v);
    }
    // All same values => std = 0 => z_score = 0
    assert!((stats.z_score(10.0)).abs() < 1e-10);
}

#[test]
fn test_rolling_stats_windowed() {
    let mut stats = RollingStats::new(3);
    stats.update(1.0);
    stats.update(2.0);
    stats.update(3.0);
    stats.update(4.0); // Window: [2, 3, 4]
    assert_eq!(stats.recent.len(), 3);
    assert_eq!(stats.recent, vec![2.0, 3.0, 4.0]);
}

// -------------------------------------------------------------------------
// AnomalyDetector Tests
// -------------------------------------------------------------------------

#[test]
fn test_anomaly_detector_nan() {
    let mut detector = AnomalyDetector::new(3.0);
    let result = detector.check(f64::NAN, 1.0);
    assert!(matches!(result, Some(TrainingAnomaly::NonFiniteLoss)));
}

#[test]
fn test_anomaly_detector_inf() {
    let mut detector = AnomalyDetector::new(3.0);
    let result = detector.check(f64::INFINITY, 1.0);
    assert!(matches!(result, Some(TrainingAnomaly::NonFiniteLoss)));
}

#[test]
fn test_anomaly_detector_gradient_explosion() {
    let mut detector = AnomalyDetector::new(3.0).with_gradient_explosion_threshold(1e6);
    let result = detector.check(1.0, 1e7);
    assert!(matches!(
        result,
        Some(TrainingAnomaly::GradientExplosion { .. })
    ));
}

#[test]
fn test_anomaly_detector_gradient_vanishing() {
    let mut detector = AnomalyDetector::new(3.0).with_gradient_vanishing_threshold(1e-10);
    let result = detector.check(1.0, 1e-12);
    assert!(matches!(
        result,
        Some(TrainingAnomaly::GradientVanishing { .. })
    ));
}

#[test]
fn test_anomaly_detector_loss_spike() {
    let mut detector = AnomalyDetector::new(3.0).with_warmup(5);

    // Warmup with stable losses
    for _ in 0..10 {
        detector.check(1.0, 1.0);
    }

    // Now introduce a spike
    let result = detector.check(100.0, 1.0);
    assert!(matches!(result, Some(TrainingAnomaly::LossSpike { .. })));
}

#[test]
fn test_anomaly_detector_no_anomaly() {
    let mut detector = AnomalyDetector::new(3.0);
    let result = detector.check(1.0, 1.0);
    assert!(result.is_none());
}

#[test]
fn test_anomaly_detector_count() {
    let mut detector = AnomalyDetector::new(3.0);
    detector.check(f64::NAN, 1.0);
    detector.check(f64::INFINITY, 1.0);
    assert_eq!(detector.anomaly_count(), 2);
}

// -------------------------------------------------------------------------
// TrainingTrajectory Tests
// -------------------------------------------------------------------------

#[test]
fn test_trajectory_empty() {
    let traj = TrainingTrajectory::new();
    assert!(traj.final_state().is_none());
    assert!(traj.best_val_loss().is_none());
    assert!(!traj.converged(0.01));
}

#[test]
fn test_trajectory_best_val_loss() {
    let mut traj = TrainingTrajectory::new();
    let rng = SimRng::new(42);
    let rng_state = rng.save_state();

    traj.push(TrainingState {
        epoch: 0,
        loss: 1.0,
        val_loss: 0.9,
        metrics: TrainingMetrics::default(),
        rng_state: rng_state.clone(),
    });
    traj.push(TrainingState {
        epoch: 1,
        loss: 0.8,
        val_loss: 0.7,
        metrics: TrainingMetrics::default(),
        rng_state: rng_state.clone(),
    });
    traj.push(TrainingState {
        epoch: 2,
        loss: 0.6,
        val_loss: 0.8,
        metrics: TrainingMetrics::default(),
        rng_state: rng_state.clone(),
    });

    assert!((traj.best_val_loss().unwrap_or(0.0) - 0.7).abs() < 1e-10);
}

#[test]
fn test_trajectory_converged() {
    let mut traj = TrainingTrajectory::new();
    let rng = SimRng::new(42);
    let rng_state = rng.save_state();

    for i in 0..15 {
        traj.push(TrainingState {
            epoch: i,
            loss: 0.5, // Constant loss
            val_loss: 0.5,
            metrics: TrainingMetrics::default(),
            rng_state: rng_state.clone(),
        });
    }
    assert!(traj.converged(0.01));
}

// -------------------------------------------------------------------------
// TrainingSimulation Tests
// -------------------------------------------------------------------------

#[test]
fn test_training_simulation_new() {
    let sim = TrainingSimulation::new(42);
    assert_eq!(sim.config().learning_rate, 0.001);
    assert_eq!(sim.trajectory().states.len(), 0);
}

#[test]
fn test_training_simulation_step() {
    let mut sim = TrainingSimulation::new(42);
    let result = sim.step(0.5, 1.0);
    assert!(result.is_ok());
    assert_eq!(sim.trajectory().states.len(), 1);
}

#[test]
fn test_training_simulation_anomaly_stops() {
    let mut sim = TrainingSimulation::new(42);
    let result = sim.step(f64::NAN, 1.0);
    assert!(result.is_err());
}

#[test]
fn test_training_simulation_simulate() {
    let mut sim = TrainingSimulation::new(42);
    let result = sim.simulate(10, |epoch, _rng| {
        // Simulated decreasing loss
        let loss = 1.0 / (epoch as f64 + 1.0);
        let grad_norm = 0.5;
        (loss, grad_norm)
    });
    assert!(result.is_ok());
    assert_eq!(result.unwrap().states.len(), 10);
}

#[test]
fn test_training_simulation_early_stopping() {
    let config = TrainingConfig {
        early_stopping: Some(3),
        ..Default::default()
    };
    let mut sim = TrainingSimulation::with_config(42, config);

    // Loss that doesn't improve
    let result = sim.simulate(100, |_epoch, _rng| (1.0, 1.0));
    assert!(result.is_ok());
    // Should stop early due to no improvement
    assert!(result.unwrap().states.len() < 100);
}

// -------------------------------------------------------------------------
// PredictionSimulation Tests
// -------------------------------------------------------------------------

#[test]
fn test_prediction_simulation_new() {
    let sim = PredictionSimulation::new(42);
    assert_eq!(sim.config().batch_size, 32);
    assert!(sim.history().is_empty());
}

#[test]
fn test_prediction_simulation_predict() {
    let mut sim = PredictionSimulation::new(42);
    let result = sim.predict(&[1.0, 2.0, 3.0], |input| {
        input.iter().map(|x| x * 2.0).collect()
    });
    assert!(result.is_ok());
    let state = result.unwrap();
    assert_eq!(state.output, vec![2.0, 4.0, 6.0]);
    assert_eq!(sim.history().len(), 1);
}

#[test]
fn test_prediction_simulation_batch() {
    let mut sim = PredictionSimulation::new(42);
    let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
    let result = sim.predict_batch(&inputs, |input| vec![input[0] * 2.0]);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 3);
}

#[test]
fn test_prediction_simulation_temperature() {
    let config = InferenceConfig {
        temperature: 0.5,
        ..Default::default()
    };
    let mut sim = PredictionSimulation::with_config(42, config);
    let result = sim.predict(&[1.0, 2.0], |input| input.to_vec());
    assert!(result.is_ok());
    let state = result.unwrap();
    // Temperature scaling divides by temperature
    assert!((state.output[0] - 2.0).abs() < 1e-10);
}

#[test]
fn test_prediction_simulation_top_k() {
    let config = InferenceConfig {
        top_k: 2,
        ..Default::default()
    };
    let mut sim = PredictionSimulation::with_config(42, config);
    let result = sim.predict(&[], |_| vec![0.1, 0.5, 0.3, 0.1]);
    assert!(result.is_ok());
    let state = result.unwrap();
    // Top-2 should keep 0.5 and 0.3, zero out others
    assert!(state.output[0].abs() < 1e-10); // 0.1 zeroed
    assert!((state.output[1] - 0.5).abs() < 1e-10);
    assert!((state.output[2] - 0.3).abs() < 1e-10);
    assert!(state.output[3].abs() < 1e-10); // 0.1 zeroed
}

#[test]
fn test_prediction_simulation_uncertainty() {
    let config = InferenceConfig {
        uncertainty: true,
        ..Default::default()
    };
    let mut sim = PredictionSimulation::with_config(42, config);
    let result = sim.predict(&[], |_| vec![1.0, 2.0, 3.0]);
    assert!(result.is_ok());
    assert!(result.unwrap().uncertainty.is_some());
}

// -------------------------------------------------------------------------
// MultiTurnSimulation Tests
// -------------------------------------------------------------------------

#[test]
fn test_multi_turn_simulation_new() {
    let sim = MultiTurnSimulation::new(42);
    assert!(sim.history().is_empty());
}

#[test]
fn test_multi_turn_simulation_turn() {
    let mut sim = MultiTurnSimulation::new(42);
    let result = sim.turn("Hello", None, |input, _| format!("Response to: {input}"));
    assert!(result.is_ok());
    let turn = result.unwrap();
    assert_eq!(turn.index, 0);
    assert!(turn.output.contains("Hello"));
}

#[test]
fn test_multi_turn_simulation_with_expected() {
    let mut sim = MultiTurnSimulation::new(42);
    let result = sim.turn("What is 2+2?", Some("4"), |_, _| "4".to_string());
    assert!(result.is_ok());
    let turn = result.unwrap();
    assert!(turn.metrics.accuracy.is_some());
    assert!((turn.metrics.accuracy.unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_multi_turn_simulation_history() {
    let mut sim = MultiTurnSimulation::new(42);
    sim.turn("First", None, |_, _| "Response 1".to_string())
        .unwrap();
    sim.turn("Second", None, |_, history| {
        format!("Response 2 (after {} turns)", history.len())
    })
    .unwrap();
    assert_eq!(sim.history().len(), 2);
}

#[test]
fn test_multi_turn_evaluation_minimum_runs() {
    let mut sim = MultiTurnSimulation::new(42);
    let queries = vec![("Hello".to_string(), None)];
    let result = sim.evaluate(&queries, 3, |_, _| "Hi".to_string());
    assert!(result.is_err()); // Should fail with < 5 runs
}

#[test]
fn test_multi_turn_evaluation_success() {
    let mut sim = MultiTurnSimulation::new(42);
    let queries = vec![
        ("Q1".to_string(), Some("A1".to_string())),
        ("Q2".to_string(), Some("A2".to_string())),
    ];
    let result = sim.evaluate(&queries, 5, |_, _| "A1 A2".to_string());
    assert!(result.is_ok());
    let eval = result.unwrap();
    assert_eq!(eval.n_runs, 5);
    assert!(eval.mean_accuracy.is_some());
}

#[test]
fn test_pareto_analysis_dominance() {
    let evals = vec![
        (
            "model_a".to_string(),
            MultiTurnEvaluation {
                mean_accuracy: Some(0.9),
                mean_latency: Some(100.0),
                total_cost: 1.0,
                confidence_interval: 0.95,
                n_runs: 5,
            },
        ),
        (
            "model_b".to_string(),
            MultiTurnEvaluation {
                mean_accuracy: Some(0.8),
                mean_latency: Some(200.0),
                total_cost: 2.0,
                confidence_interval: 0.95,
                n_runs: 5,
            },
        ),
    ];

    let analysis = MultiTurnSimulation::pareto_analysis(&evals);
    // model_a dominates model_b (better in all dimensions)
    assert_eq!(analysis.frontier.len(), 1);
    assert_eq!(analysis.frontier[0].model_id, "model_a");
}

#[test]
fn test_pareto_analysis_no_dominance() {
    let evals = vec![
        (
            "model_a".to_string(),
            MultiTurnEvaluation {
                mean_accuracy: Some(0.9),
                mean_latency: Some(200.0), // Worse latency
                total_cost: 1.0,
                confidence_interval: 0.95,
                n_runs: 5,
            },
        ),
        (
            "model_b".to_string(),
            MultiTurnEvaluation {
                mean_accuracy: Some(0.8),
                mean_latency: Some(100.0), // Better latency
                total_cost: 2.0,
                confidence_interval: 0.95,
                n_runs: 5,
            },
        ),
    ];

    let analysis = MultiTurnSimulation::pareto_analysis(&evals);
    // Neither dominates - both on frontier (trade-off)
    assert_eq!(analysis.frontier.len(), 2);
}

// -------------------------------------------------------------------------
// JidokaMLFeedback Tests
// -------------------------------------------------------------------------

#[test]
fn test_jidoka_feedback_new() {
    let feedback = JidokaMLFeedback::new();
    assert!(feedback.patterns().is_empty());
    assert!(feedback.patches().is_empty());
}

#[test]
fn test_jidoka_feedback_record_anomaly() {
    let mut feedback = JidokaMLFeedback::new();
    let patch = feedback.record_anomaly(TrainingAnomaly::GradientExplosion {
        norm: 1e7,
        threshold: 1e6,
    });
    assert!(patch.is_none()); // First occurrence, no patch yet
    assert_eq!(feedback.patterns().len(), 1);
}

#[test]
fn test_jidoka_feedback_auto_patch() {
    let mut feedback = JidokaMLFeedback::new().with_auto_patch_threshold(2);

    // First occurrence
    feedback.record_anomaly(TrainingAnomaly::GradientExplosion {
        norm: 1e7,
        threshold: 1e6,
    });

    // Second occurrence - should trigger patch
    let patch = feedback.record_anomaly(TrainingAnomaly::GradientExplosion {
        norm: 1e8,
        threshold: 1e6,
    });

    assert!(patch.is_some());
    assert_eq!(patch.unwrap().rule_type, RuleType::GradientClipping);
}

#[test]
fn test_jidoka_feedback_loss_spike_patch() {
    let mut feedback = JidokaMLFeedback::new().with_auto_patch_threshold(1);

    let patch = feedback.record_anomaly(TrainingAnomaly::LossSpike {
        z_score: 5.0,
        loss: 100.0,
    });

    assert!(patch.is_some());
    assert_eq!(patch.unwrap().rule_type, RuleType::LearningRateWarmup);
}

#[test]
fn test_jidoka_feedback_different_anomalies() {
    let mut feedback = JidokaMLFeedback::new();

    feedback.record_anomaly(TrainingAnomaly::GradientExplosion {
        norm: 1e7,
        threshold: 1e6,
    });
    feedback.record_anomaly(TrainingAnomaly::LossSpike {
        z_score: 5.0,
        loss: 100.0,
    });

    assert_eq!(feedback.patterns().len(), 2);
}

#[test]
fn test_jidoka_feedback_reset() {
    let mut feedback = JidokaMLFeedback::new();
    feedback.record_anomaly(TrainingAnomaly::NonFiniteLoss);
    feedback.reset();
    assert!(feedback.patterns().is_empty());
}

// -------------------------------------------------------------------------
// Display and Clone Tests
// -------------------------------------------------------------------------

#[test]
fn test_training_anomaly_display() {
    let anomaly = TrainingAnomaly::NonFiniteLoss;
    let display = format!("{}", anomaly);
    assert!(display.contains("NaN/Inf"));

    let anomaly = TrainingAnomaly::GradientExplosion {
        norm: 1e7,
        threshold: 1e6,
    };
    let display = format!("{}", anomaly);
    assert!(display.contains("explosion"));

    let anomaly = TrainingAnomaly::GradientVanishing {
        norm: 1e-12,
        threshold: 1e-10,
    };
    let display = format!("{}", anomaly);
    assert!(display.contains("vanishing"));

    let anomaly = TrainingAnomaly::LossSpike {
        z_score: 5.0,
        loss: 100.0,
    };
    let display = format!("{}", anomaly);
    assert!(display.contains("spike"));

    let anomaly = TrainingAnomaly::LowConfidence {
        confidence: 0.3,
        threshold: 0.5,
    };
    let display = format!("{}", anomaly);
    assert!(display.contains("confidence"));
}

#[test]
fn test_rolling_stats_reset() {
    let mut stats = RollingStats::new(5);
    stats.update(1.0);
    stats.update(2.0);
    stats.update(3.0);
    stats.reset();
    assert_eq!(stats.mean(), 0.0);
    assert_eq!(stats.variance(), 0.0);
}

#[test]
fn test_rolling_stats_z_score_with_variance() {
    let mut stats = RollingStats::new(0);
    for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
        stats.update(v);
    }
    // Mean = 3, std dev = sqrt(2.5) ≈ 1.58
    let z = stats.z_score(5.0);
    assert!(z > 1.0); // 5 is above mean
}

#[test]
fn test_rolling_stats_clone() {
    let mut stats = RollingStats::new(3);
    stats.update(1.0);
    stats.update(2.0);
    let cloned = stats.clone();
    assert_eq!(cloned.mean(), stats.mean());
}

#[test]
fn test_training_config_clone() {
    let config = TrainingConfig::default();
    let cloned = config.clone();
    assert_eq!(cloned.learning_rate, config.learning_rate);
    assert_eq!(cloned.batch_size, config.batch_size);
}

#[test]
fn test_training_state_clone() {
    let rng = SimRng::new(42);
    let state = TrainingState {
        epoch: 5,
        loss: 0.5,
        val_loss: 0.6,
        metrics: TrainingMetrics::default(),
        rng_state: rng.save_state(),
    };
    let cloned = state.clone();
    assert_eq!(cloned.epoch, state.epoch);
    assert_eq!(cloned.loss, state.loss);
}

#[test]
fn test_training_metrics_clone() {
    let metrics = TrainingMetrics {
        train_loss: 0.5,
        val_loss: 0.6,
        accuracy: Some(0.9),
        gradient_norm: 1.0,
        learning_rate: 0.001,
        params_updated: 1000,
    };
    let cloned = metrics.clone();
    assert_eq!(cloned.accuracy, metrics.accuracy);
}

#[test]
fn test_training_trajectory_clone() {
    let mut traj = TrainingTrajectory::new();
    let rng = SimRng::new(42);
    traj.push(TrainingState {
        epoch: 0,
        loss: 1.0,
        val_loss: 0.9,
        metrics: TrainingMetrics::default(),
        rng_state: rng.save_state(),
    });
    let cloned = traj.clone();
    assert_eq!(cloned.states.len(), 1);
}

#[test]
fn test_anomaly_detector_clone() {
    let detector = AnomalyDetector::new(3.0)
        .with_warmup(10)
        .with_gradient_explosion_threshold(1e6);
    let cloned = detector.clone();
    assert_eq!(cloned.threshold_sigma, detector.threshold_sigma);
}

#[test]
fn test_inference_config_default() {
    let config = InferenceConfig::default();
    assert_eq!(config.batch_size, 32);
    assert!((config.temperature - 1.0).abs() < 1e-10);
    assert_eq!(config.top_k, 0);
}

#[test]
fn test_prediction_state_clone() {
    let state = PredictionState {
        input: vec![1.0, 2.0],
        output: vec![2.0, 4.0],
        uncertainty: Some(0.05),
        latency_us: 100,
        sequence: 0,
    };
    let cloned = state.clone();
    assert_eq!(cloned.sequence, state.sequence);
}

#[test]
fn test_pareto_point_clone() {
    let point = ParetoPoint {
        model_id: "test".to_string(),
        accuracy: 0.9,
        latency: 100.0,
        cost: 1.0,
        dominated_by: vec![],
    };
    let cloned = point.clone();
    assert_eq!(cloned.model_id, "test");
}
