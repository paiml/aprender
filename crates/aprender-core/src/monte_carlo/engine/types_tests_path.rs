
    #[test]
    fn test_simulation_path() {
        let path = SimulationPath::new(
            vec![0.0, 0.5, 1.0],
            vec![100.0, 105.0, 110.0],
            PathMetadata {
                path_id: 0,
                seed: 42,
                is_antithetic: false,
            },
        );

        assert_eq!(path.initial_value(), Some(100.0));
        assert_eq!(path.final_value(), Some(110.0));
        assert!((path.total_return().unwrap() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_period_returns() {
        let path = SimulationPath::new(
            vec![0.0, 0.5, 1.0],
            vec![100.0, 110.0, 121.0],
            PathMetadata {
                path_id: 0,
                seed: 42,
                is_antithetic: false,
            },
        );

        let returns = path.period_returns();
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.1).abs() < 0.001);
        assert!((returns[1] - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_log_returns() {
        let path = SimulationPath::new(
            vec![0.0, 0.5, 1.0],
            vec![100.0, 110.0, 121.0],
            PathMetadata {
                path_id: 0,
                seed: 42,
                is_antithetic: false,
            },
        );

        let log_returns = path.log_returns();
        assert_eq!(log_returns.len(), 2);
        assert!((log_returns[0] - (1.1_f64).ln()).abs() < 0.001);
    }

    #[test]
    fn test_time_horizon_years() {
        let horizon = TimeHorizon::years(1);
        assert_eq!(horizon.n_steps(), 12); // Monthly default
        assert!((horizon.dt() - 1.0 / 12.0).abs() < 0.001);
    }

    #[test]
    fn test_time_horizon_quarters() {
        let horizon = TimeHorizon::quarters(4);
        assert!((horizon.duration - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_time_horizon_daily() {
        let horizon = TimeHorizon::years(1).with_step(TimeStep::Daily);
        assert_eq!(horizon.n_steps(), 252);
    }

    #[test]
    fn test_time_points() {
        let horizon = TimeHorizon::years(1).with_step(TimeStep::Quarterly);
        let points = horizon.time_points();
        assert_eq!(points.len(), 5); // 0, 0.25, 0.5, 0.75, 1.0
        assert!((points[0] - 0.0).abs() < 0.001);
        assert!((points[4] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_percentile_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        assert!((percentile(&values, 0.0) - 1.0).abs() < 0.001);
        assert!((percentile(&values, 0.5) - 5.5).abs() < 0.001);
        assert!((percentile(&values, 1.0) - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_percentile_interpolation() {
        let values = vec![0.0, 1.0];
        assert!((percentile(&values, 0.25) - 0.25).abs() < 0.001);
        assert!((percentile(&values, 0.75) - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_statistics_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = Statistics::from_values(&values);

        assert!((stats.mean - 3.0).abs() < 0.001);
        assert!((stats.min - 1.0).abs() < 0.001);
        assert!((stats.max - 5.0).abs() < 0.001);
        assert_eq!(stats.n, 5);
    }

    #[test]
    fn test_statistics_normal() {
        // Generate roughly normal samples
        let values: Vec<f64> = (0..1000)
            .map(|i| {
                let x = (i as f64 / 1000.0) * 2.0 - 1.0;
                x * x.signum() * (-x.abs()).exp()
            })
            .collect();

        let stats = Statistics::from_values(&values);
        assert!(stats.std > 0.0);
        assert!(stats.skewness.is_finite());
        assert!(stats.kurtosis.is_finite());
    }

    #[test]
    fn test_percentiles_from_values() {
        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let pcts = Percentiles::from_values(&values);

        assert!(pcts.p50 > pcts.p25);
        assert!(pcts.p75 > pcts.p50);
        assert!(pcts.p99 > pcts.p95);
    }

    #[test]
    fn test_budget_max_simulations() {
        assert_eq!(Budget::Simulations(1000).max_simulations(), 1000);
        assert_eq!(
            Budget::Convergence {
                patience: 5,
                min_delta: 0.001,
                max_simulations: 5000
            }
            .max_simulations(),
            5000
        );
    }

    #[test]
    fn test_empty_statistics() {
        let stats = Statistics::from_values(&[]);
        assert_eq!(stats.n, 0);
        assert!((stats.mean - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_path_final_value() {
        let path = SimulationPath::new(
            vec![],
            vec![],
            PathMetadata {
                path_id: 0,
                seed: 0,
                is_antithetic: false,
            },
        );
        assert_eq!(path.final_value(), None);
    }

    #[test]
    fn test_empty_path_initial_value() {
        let path = SimulationPath::new(
            vec![],
            vec![],
            PathMetadata {
                path_id: 0,
                seed: 0,
                is_antithetic: false,
            },
        );
        assert_eq!(path.initial_value(), None);
    }

    #[test]
    fn test_total_return_empty() {
        let path = SimulationPath::new(
            vec![],
            vec![],
            PathMetadata {
                path_id: 0,
                seed: 0,
                is_antithetic: false,
            },
        );
        assert_eq!(path.total_return(), None);
    }

    #[test]
    fn test_total_return_zero_initial() {
        let path = SimulationPath::new(
            vec![0.0, 1.0],
            vec![0.0, 5.0],
            PathMetadata {
                path_id: 0,
                seed: 0,
                is_antithetic: false,
            },
        );
        // initial_value is 0.0, so total_return should be None
        assert_eq!(path.total_return(), None);
    }

    #[test]
    fn test_period_returns_fewer_than_two_values() {
        let path = SimulationPath::new(
            vec![0.0],
            vec![100.0],
            PathMetadata {
                path_id: 0,
                seed: 0,
                is_antithetic: false,
            },
        );
        assert!(path.period_returns().is_empty());
    }

    #[test]
    fn test_period_returns_with_zero_value() {
        let path = SimulationPath::new(
            vec![0.0, 0.5, 1.0],
            vec![0.0, 50.0, 100.0],
            PathMetadata {
                path_id: 0,
                seed: 0,
                is_antithetic: false,
            },
        );
        let returns = path.period_returns();
        // First window [0.0, 50.0] => w[0] is 0.0, filtered out
        // Second window [50.0, 100.0] => (100.0 - 50.0) / 50.0 = 1.0
        assert_eq!(returns.len(), 1);
        assert!((returns[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_log_returns_fewer_than_two_values() {
        let path = SimulationPath::new(
            vec![0.0],
            vec![100.0],
            PathMetadata {
                path_id: 0,
                seed: 0,
                is_antithetic: false,
            },
        );
        assert!(path.log_returns().is_empty());
    }

    #[test]
    fn test_log_returns_with_zero_value() {
        let path = SimulationPath::new(
            vec![0.0, 0.5, 1.0],
            vec![0.0, 50.0, 100.0],
            PathMetadata {
                path_id: 0,
                seed: 0,
                is_antithetic: false,
            },
        );
        let log_ret = path.log_returns();
        // First window [0.0, 50.0] => w[0] is 0.0, filtered out
        // Second window [50.0, 100.0] => ln(100/50) = ln(2)
        assert_eq!(log_ret.len(), 1);
        assert!((log_ret[0] - 2.0_f64.ln()).abs() < 0.001);
    }

    #[test]
    fn test_log_returns_with_negative_value() {
        let path = SimulationPath::new(
            vec![0.0, 0.5],
            vec![100.0, -50.0],
            PathMetadata {
                path_id: 0,
                seed: 0,
                is_antithetic: false,
            },
        );
        let log_ret = path.log_returns();
        // w[1] is negative, so filtered out
        assert!(log_ret.is_empty());
    }

    #[test]
    fn test_time_horizon_months() {
        let horizon = TimeHorizon::months(6);
        // duration = 6/12 = 0.5 years
        assert!((horizon.duration - 0.5).abs() < 0.001);
        // Default step for months() is Weekly
        let n_steps = horizon.n_steps();
        assert_eq!(n_steps, 26); // 0.5 * 52 = 26
    }

    #[test]
    fn test_time_horizon_yearly_step() {
        let horizon = TimeHorizon::years(3).with_step(TimeStep::Yearly);
        assert_eq!(horizon.n_steps(), 3);
        assert!((horizon.dt() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_time_horizon_custom_step() {
        let horizon = TimeHorizon::years(1).with_step(TimeStep::Custom(100.0));
        assert_eq!(horizon.n_steps(), 100);
    }

    #[test]
    fn test_budget_evaluations() {
        assert_eq!(Budget::Evaluations(500).max_simulations(), 500);
    }

    #[test]
    fn test_percentiles_iqr() {
        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let pcts = Percentiles::from_values(&values);
        let iqr = pcts.iqr();
        assert!((iqr - (pcts.p75 - pcts.p25)).abs() < 1e-10);
        assert!(iqr > 0.0);
    }

    #[test]
    fn test_statistics_cv_near_zero_mean() {
        let stats = Statistics {
            mean: 0.0,
            std: 1.0,
            ..Statistics::default()
        };
        assert_eq!(stats.cv(), f64::INFINITY);
    }

    #[test]
    fn test_statistics_cv_normal() {
        let stats = Statistics {
            mean: 10.0,
            std: 2.0,
            ..Statistics::default()
        };
        assert!((stats.cv() - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_statistics_sem_zero_n() {
        let stats = Statistics {
            n: 0,
            std: 1.0,
            ..Statistics::default()
        };
        assert_eq!(stats.sem(), f64::INFINITY);
    }

    #[test]
    fn test_statistics_sem_normal() {
        let stats = Statistics {
            n: 100,
            std: 10.0,
            ..Statistics::default()
        };
        assert!((stats.sem() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_percentile_empty() {
        assert!((percentile(&[], 0.5) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_single_element() {
        assert!((percentile(&[42.0], 0.5) - 42.0).abs() < 1e-10);
        assert!((percentile(&[42.0], 0.0) - 42.0).abs() < 1e-10);
        assert!((percentile(&[42.0], 1.0) - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_statistics_single_value_zero_std() {
        let stats = Statistics::from_values(&[5.0]);
        assert!((stats.mean - 5.0).abs() < 1e-10);
        assert!((stats.std - 0.0).abs() < 1e-10);
        // Skewness and kurtosis should be 0 when std is 0
        assert!((stats.skewness - 0.0).abs() < 1e-10);
        assert!((stats.kurtosis - 0.0).abs() < 1e-10);
        assert_eq!(stats.n, 1);
    }

    #[test]
    fn test_path_metadata_antithetic() {
        let meta = PathMetadata {
            path_id: 5,
            seed: 123,
            is_antithetic: true,
        };
        assert!(meta.is_antithetic);
        assert_eq!(meta.path_id, 5);
        assert_eq!(meta.seed, 123);
    }

    #[test]
    fn test_simulation_path_debug_clone() {
        let path = SimulationPath::new(
            vec![0.0, 1.0],
            vec![100.0, 110.0],
            PathMetadata {
                path_id: 0,
                seed: 42,
                is_antithetic: false,
            },
        );
        let debug_str = format!("{:?}", path);
        assert!(debug_str.contains("SimulationPath"));

        let cloned = path.clone();
        assert_eq!(cloned.values.len(), path.values.len());
    }

    #[test]
    fn test_time_horizon_debug_clone() {
        let horizon = TimeHorizon::years(1);
        let debug_str = format!("{:?}", horizon);
        assert!(debug_str.contains("TimeHorizon"));

        let cloned = horizon.clone();
        assert!((cloned.duration - horizon.duration).abs() < 1e-10);
    }

    #[test]
    fn test_time_step_debug_clone() {
        let step = TimeStep::Daily;
        let debug_str = format!("{:?}", step);
        assert!(debug_str.contains("Daily"));

        let cloned = step;
        let _ = format!("{:?}", cloned);
    }
