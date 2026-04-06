
    #[test]
    fn test_tpe_model_phase_with_log_scale() {
        // Tests denormalize_candidate with log_scale=true
        let space: SearchSpace<RF> = SearchSpace::new().add_log_scale(
            RF::NEstimators,
            crate::automl::LogScale {
                low: 1e-4,
                high: 1.0,
            },
        );

        let mut tpe = TPE::new(100).with_startup_trials(3).with_seed(42);

        // Activate model
        for i in 0..3 {
            let mut values = std::collections::HashMap::new();
            values.insert(RF::NEstimators, ParamValue::Float(0.001 * (i as f64 + 1.0)));
            let trial = Trial { values };
            let result = TrialResult {
                trial,
                score: i as f64 * 0.3,
                metrics: std::collections::HashMap::new(),
            };
            tpe.update(&[result]);
        }

        assert!(tpe.should_use_model());
        let trials = tpe.suggest(&space, 2);
        assert_eq!(trials.len(), 2);
    }

    #[test]
    fn test_tpe_model_phase_with_categorical() {
        // Tests denormalize_candidate with categorical params
        let space: SearchSpace<RF> =
            SearchSpace::new().add_categorical(RF::MaxFeatures, ["sqrt", "log2", "auto"]);

        let mut tpe = TPE::new(100).with_startup_trials(3).with_seed(42);

        // Add observations with string values
        for i in 0..3 {
            let mut values = std::collections::HashMap::new();
            let choices = ["sqrt", "log2", "auto"];
            values.insert(
                RF::MaxFeatures,
                ParamValue::String(choices[i % 3].to_string()),
            );
            let trial = Trial { values };
            let result = TrialResult {
                trial,
                score: i as f64 * 0.2,
                metrics: std::collections::HashMap::new(),
            };
            tpe.update(&[result]);
        }

        assert!(tpe.should_use_model());
        let trials = tpe.suggest(&space, 3);
        assert_eq!(trials.len(), 3);

        for trial in &trials {
            let v = trial.get(&RF::MaxFeatures).expect("should have value");
            let s = v.as_str().expect("should be string");
            assert!(
                ["sqrt", "log2", "auto"].contains(&s),
                "Categorical value '{s}' not in choices"
            );
        }
    }

    #[test]
    fn test_tpe_zero_remaining() {
        // Covers the n == 0 early return in suggest (line 272-274)
        let space: SearchSpace<RF> = SearchSpace::new().add(RF::NEstimators, 10..100);

        let mut tpe = TPE::new(0);
        let trials = tpe.suggest(&space, 10);
        assert!(trials.is_empty());
    }

    #[test]
    fn test_tpe_with_startup_trials() {
        let tpe = TPE::new(50).with_startup_trials(20);
        assert_eq!(tpe.config.n_startup_trials, 20);
    }

    #[test]
    fn test_kde_density_multiple_samples() {
        let samples = vec![0.2, 0.4, 0.6, 0.8];
        let bw = TPE::compute_bandwidth(&samples);

        // Density at center should be reasonable
        let density_center = TPE::kde_density(&samples, 0.5, bw);
        assert!(density_center > 0.0);

        // Density far away should be lower
        let density_far = TPE::kde_density(&samples, 10.0, bw);
        assert!(density_center > density_far);
    }

    #[test]
    fn test_tpe_suggest_empty_space_with_model() {
        // Empty space with model active: n_dims == 0 forces random sampling
        let space: SearchSpace<RF> = SearchSpace::new();

        let mut tpe = TPE::new(100).with_startup_trials(0).with_seed(42);

        // Even with startup_trials=0, empty space triggers random path
        // because n_dims == 0
        let trials = tpe.suggest(&space, 3);
        assert_eq!(trials.len(), 3);
    }

    #[test]
    fn test_split_observations_with_nan_scores() {
        // NaN scores should be handled by the partial_cmp fallback
        let mut tpe = TPE::new(100).with_gamma(0.5);
        tpe.history.push(Observation {
            values: vec![0.3],
            score: f64::NAN,
        });
        tpe.history.push(Observation {
            values: vec![0.5],
            score: 0.5,
        });
        tpe.history.push(Observation {
            values: vec![0.7],
            score: 0.8,
        });

        // Should not panic on NaN comparisons
        let (good, bad) = tpe.split_observations();
        assert_eq!(good.len() + bad.len(), 3);
    }
