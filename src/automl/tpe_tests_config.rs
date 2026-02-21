
    // ==================== UNIT TESTS ====================

    #[test]
    fn test_tpe_config_defaults() {
        let config = TPEConfig::default();
        assert!((config.gamma - 0.25).abs() < 0.01);
        assert_eq!(config.n_candidates, 24);
        assert_eq!(config.n_startup_trials, 10);
    }

    #[test]
    fn test_tpe_creation() {
        let tpe = TPE::new(100);
        assert_eq!(tpe.n_trials, 100);
        assert_eq!(tpe.remaining(), 100);
    }

    #[test]
    fn test_tpe_with_seed() {
        let tpe = TPE::new(50).with_seed(12345);
        assert_eq!(tpe.seed, 12345);
    }

    #[test]
    fn test_tpe_with_gamma() {
        let tpe = TPE::new(50).with_gamma(0.15);
        assert!((tpe.config.gamma - 0.15).abs() < 0.01);
    }

    #[test]
    fn test_tpe_gamma_clamped() {
        let tpe_low = TPE::new(50).with_gamma(0.0);
        assert!(tpe_low.config.gamma >= 0.01);

        let tpe_high = TPE::new(50).with_gamma(1.0);
        assert!(tpe_high.config.gamma <= 0.5);
    }

    #[test]
    fn test_tpe_suggest_respects_budget() {
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..500);

        let mut tpe = TPE::new(5);
        let t1 = tpe.suggest(&space, 3);
        assert_eq!(t1.len(), 3);
        assert_eq!(tpe.remaining(), 2);

        let t2 = tpe.suggest(&space, 10);
        assert_eq!(t2.len(), 2);
        assert_eq!(tpe.remaining(), 0);
    }

    #[test]
    fn test_tpe_deterministic_with_seed() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add(RF::NEstimators, 10..500)
            .add(RF::MaxDepth, 2..20);

        let mut tpe1 = TPE::new(10).with_seed(42);
        let mut tpe2 = TPE::new(10).with_seed(42);

        let t1 = tpe1.suggest(&space, 5);
        let t2 = tpe2.suggest(&space, 5);

        for (a, b) in t1.iter().zip(t2.iter()) {
            assert_eq!(a.get(&RF::NEstimators), b.get(&RF::NEstimators));
        }
    }

    #[test]
    fn test_tpe_empty_when_exhausted() {
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);

        let mut tpe = TPE::new(2);
        let _ = tpe.suggest(&space, 2);
        let empty = tpe.suggest(&space, 1);
        assert!(empty.is_empty());
    }

    // ==================== TPE MODEL TESTS ====================

    #[test]
    fn test_tpe_update_stores_history() {
        let mut tpe = TPE::new(100);
        assert_eq!(tpe.n_observations(), 0);

        // Create a trial result
        let mut values = std::collections::HashMap::new();
        values.insert(RF::NEstimators, ParamValue::Int(100));
        let trial = Trial { values };

        let result = TrialResult {
            trial,
            score: 0.85,
            metrics: std::collections::HashMap::new(),
        };

        tpe.update(&[result]);
        assert_eq!(tpe.n_observations(), 1);
    }

    #[test]
    fn test_tpe_uses_random_during_startup() {
        let mut tpe = TPE::new(100).with_startup_trials(10);
        assert!(!tpe.should_use_model());

        // Add 9 observations (still below startup)
        for i in 0_i64..9 {
            let mut values = std::collections::HashMap::new();
            values.insert(RF::NEstimators, ParamValue::Int(100 + i));
            let trial = Trial { values };
            let result = TrialResult {
                trial,
                score: i as f64 / 10.0,
                metrics: std::collections::HashMap::new(),
            };
            tpe.update(&[result]);
        }

        assert!(!tpe.should_use_model());

        // Add 10th observation
        let mut values = std::collections::HashMap::new();
        values.insert(RF::NEstimators, ParamValue::Int(200));
        let trial = Trial { values };
        let result = TrialResult {
            trial,
            score: 0.9,
            metrics: std::collections::HashMap::new(),
        };
        tpe.update(&[result]);

        assert!(tpe.should_use_model());
    }

    #[test]
    fn test_kde_density_basic() {
        let samples = vec![0.5];
        let density = TPE::kde_density(&samples, 0.5, 0.1);
        assert!(density > 0.0);

        // Density should be higher at sample points
        let density_at_sample = TPE::kde_density(&samples, 0.5, 0.1);
        let density_far = TPE::kde_density(&samples, 0.0, 0.1);
        assert!(density_at_sample > density_far);
    }

    #[test]
    fn test_kde_density_empty() {
        let samples: Vec<f64> = vec![];
        let density = TPE::kde_density(&samples, 0.5, 0.1);
        assert!((density - 1.0).abs() < 0.001); // Uniform prior
    }

    #[test]
    fn test_bandwidth_computation() {
        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let bandwidth = TPE::compute_bandwidth(&samples);
        assert!(bandwidth > 0.0);
        assert!(bandwidth < 1.0);
    }

    #[test]
    fn test_split_observations() {
        let mut tpe = TPE::new(100).with_gamma(0.25);

        // Add 4 observations with scores 0.1, 0.2, 0.3, 0.4
        for i in 0_i32..4 {
            tpe.history.push(Observation {
                values: vec![f64::from(i) / 4.0],
                score: f64::from(i + 1) / 10.0,
            });
        }

        let (good, bad) = tpe.split_observations();

        // With gamma=0.25, top 25% should be "good"
        // With 4 observations, that's 1 observation
        assert_eq!(good.len(), 1);
        assert_eq!(bad.len(), 3);

        // Best score (0.4) should be in good
        assert!((good[0].score - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_tpe_suggests_after_model_active() {
        let space: SearchSpace<RF> = SearchSpace::new()
            .add(RF::NEstimators, 10..500)
            .add(RF::MaxDepth, 2..20);

        let mut tpe = TPE::new(100).with_startup_trials(5).with_seed(42);

        // Add 5 observations to activate model
        for i in 0_i64..5 {
            let mut values = std::collections::HashMap::new();
            values.insert(RF::NEstimators, ParamValue::Int(100 + i * 50));
            values.insert(RF::MaxDepth, ParamValue::Int(5 + i));
            let trial = Trial { values };
            let result = TrialResult {
                trial,
                score: i as f64 / 5.0,
                metrics: std::collections::HashMap::new(),
            };
            tpe.update(&[result]);
        }

        assert!(tpe.should_use_model());

        // Now suggest should use TPE model
        let trials = tpe.suggest(&space, 3);
        assert_eq!(trials.len(), 3);

        // Verify suggestions are in valid ranges
        for trial in &trials {
            let n = trial
                .get_i64(&RF::NEstimators)
                .expect("should have n_estimators");
            let d = trial.get_i64(&RF::MaxDepth).expect("should have max_depth");
            assert!((10..=499).contains(&n));
            assert!((2..=19).contains(&d));
        }
    }

    // ==================== COVERAGE GAP TESTS ====================

    #[test]
    fn test_bandwidth_single_sample() {
        // Covers compute_bandwidth with < 2 samples (line 150-151)
        let samples = vec![0.5];
        let bw = TPE::compute_bandwidth(&samples);
        assert!(
            (bw - 1.0).abs() < 1e-10,
            "Single sample bandwidth should be 1.0"
        );
    }

    #[test]
    fn test_bandwidth_empty_samples() {
        // Covers compute_bandwidth with 0 samples
        let samples: Vec<f64> = vec![];
        let bw = TPE::compute_bandwidth(&samples);
        assert!(
            (bw - 1.0).abs() < 1e-10,
            "Empty samples bandwidth should be 1.0"
        );
    }

    #[test]
    fn test_bandwidth_identical_samples() {
        // Covers the variance.sqrt().max(0.01) path (line 157)
        let samples = vec![0.5, 0.5, 0.5, 0.5];
        let bw = TPE::compute_bandwidth(&samples);
        assert!(
            bw > 0.0,
            "Bandwidth should be positive even for zero variance"
        );
    }

    #[test]
    fn test_split_observations_empty_history() {
        // Covers split_observations with empty history (line 165-167)
        let tpe = TPE::new(100);
        let (good, bad) = tpe.split_observations();
        assert!(good.is_empty());
        assert!(bad.is_empty());
    }

    #[test]
    fn test_split_observations_single_observation() {
        // Covers n_good.max(1).min(sorted.len() - 1) edge case
        let mut tpe = TPE::new(100).with_gamma(0.25);
        tpe.history.push(Observation {
            values: vec![0.5],
            score: 0.9,
        });
        // With 1 observation: n_good = ceil(1 * 0.25) = 1, but min(sorted.len()-1) = 0
        // So n_good becomes 0 wait no: max(1) first, then min(0) = 0
        // Actually: n_good = max(1, ...).min(sorted.len()-1) = max(1, 1).min(0) = min(1, 0) = 0
        // Let's just verify it doesn't panic
        let (good, bad) = tpe.split_observations();
        // Total should be 1
        assert_eq!(good.len() + bad.len(), 1);
    }

    #[test]
    fn test_split_observations_two_observations() {
        let mut tpe = TPE::new(100).with_gamma(0.25);
        tpe.history.push(Observation {
            values: vec![0.3],
            score: 0.5,
        });
        tpe.history.push(Observation {
            values: vec![0.7],
            score: 0.9,
        });

        let (good, bad) = tpe.split_observations();
        assert_eq!(good.len(), 1);
        assert_eq!(bad.len(), 1);
        // Best score (0.9) should be in good
        assert!((good[0].score - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_ei_ratio_empty_candidate() {
        // Covers compute_ei_ratio with empty candidate (line 189-191)
        let good: Vec<&Observation> = vec![];
        let bad: Vec<&Observation> = vec![];
        let ratio = TPE::compute_ei_ratio(&[], &good, &bad);
        assert!(
            (ratio - 0.0).abs() < 1e-10,
            "Empty candidate should return 0.0"
        );
    }

    #[test]
    fn test_ei_ratio_with_observations() {
        let obs_good = Observation {
            values: vec![0.5],
            score: 0.9,
        };
        let obs_bad = Observation {
            values: vec![0.1],
            score: 0.1,
        };

        let good = vec![&obs_good];
        let bad = vec![&obs_bad];

        // Near good observation should have high EI ratio
        let ratio_near_good = TPE::compute_ei_ratio(&[0.5], &good, &bad);
        let ratio_near_bad = TPE::compute_ei_ratio(&[0.1], &good, &bad);

        assert!(ratio_near_good > 0.0);
        assert!(ratio_near_bad > 0.0);
        assert!(
            ratio_near_good > ratio_near_bad,
            "Point near good ({ratio_near_good}) should have higher EI than near bad ({ratio_near_bad})"
        );
    }

    #[test]
    fn test_tpe_update_empty_results() {
        // Covers update with empty results (line 319-321)
        let mut tpe = TPE::new(100);
        let empty: &[TrialResult<RF>] = &[];
        tpe.update(empty); // Should return early, no panic
        assert_eq!(tpe.n_observations(), 0);
    }

    #[test]
    fn test_tpe_update_with_non_numeric_values() {
        // Covers the case where ParamValue::as_f64 returns None for non-numeric
        let mut tpe = TPE::new(100);

        let mut values = std::collections::HashMap::new();
        values.insert(RF::MaxFeatures, ParamValue::String("sqrt".to_string()));
        let trial = Trial { values };
        let result = TrialResult {
            trial,
            score: 0.8,
            metrics: std::collections::HashMap::new(),
        };

        tpe.update(&[result]);
        assert_eq!(tpe.n_observations(), 1);
        // Non-numeric values are filtered out, so observation has empty values
        assert!(tpe.history[0].values.is_empty());
    }

    #[test]
    fn test_tpe_update_with_bool_values() {
        // Bool values should not convert to f64
        let mut tpe = TPE::new(100);

        let mut values = std::collections::HashMap::new();
        values.insert(RF::Bootstrap, ParamValue::Bool(true));
        let trial = Trial { values };
        let result = TrialResult {
            trial,
            score: 0.7,
            metrics: std::collections::HashMap::new(),
        };

        tpe.update(&[result]);
        assert_eq!(tpe.n_observations(), 1);
    }

    #[test]
    fn test_tpe_config_clone_debug() {
        let config = TPEConfig::default();
        let cloned = config.clone();
        assert!((cloned.gamma - config.gamma).abs() < 0.001);
        assert_eq!(cloned.n_candidates, config.n_candidates);
        assert_eq!(cloned.n_startup_trials, config.n_startup_trials);

        let debug = format!("{config:?}");
        assert!(debug.contains("TPEConfig"));
    }

    #[test]
    fn test_tpe_clone_debug() {
        let tpe = TPE::new(50).with_seed(99);
        let cloned = tpe.clone();
        assert_eq!(cloned.n_trials, 50);
        assert_eq!(cloned.seed, 99);
        assert_eq!(cloned.remaining(), 50);

        let debug = format!("{tpe:?}");
        assert!(debug.contains("TPE"));
    }

    #[test]
    fn test_tpe_model_phase_with_continuous() {
        // Tests the TPE model phase with continuous parameters
        let space: SearchSpace<RF> =
            SearchSpace::new().add_continuous(RF::NEstimators, 10.0, 500.0);

        let mut tpe = TPE::new(100).with_startup_trials(5).with_seed(42);

        // Add enough observations to activate model
        for i in 0..5 {
            let mut values = std::collections::HashMap::new();
            values.insert(RF::NEstimators, ParamValue::Float(100.0 + i as f64 * 50.0));
            let trial = Trial { values };
            let result = TrialResult {
                trial,
                score: i as f64 / 5.0,
                metrics: std::collections::HashMap::new(),
            };
            tpe.update(&[result]);
        }

        assert!(tpe.should_use_model());
        let trials = tpe.suggest(&space, 3);
        assert_eq!(trials.len(), 3);

        for trial in &trials {
            let v = trial.get_f64(&RF::NEstimators).expect("should have value");
            assert!(v >= 10.0 && v <= 500.0, "Continuous value {v} out of range");
        }
    }
