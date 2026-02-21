
    #[test]
    fn test_budget_debug_clone() {
        let budget = Budget::Convergence {
            patience: 5,
            min_delta: 0.01,
            max_simulations: 1000,
        };
        let debug_str = format!("{:?}", budget);
        assert!(debug_str.contains("Convergence"));

        let cloned = budget.clone();
        assert_eq!(cloned.max_simulations(), 1000);
    }

    #[test]
    fn test_percentiles_default() {
        let pcts = Percentiles::default();
        assert!((pcts.p50 - 0.0).abs() < 1e-10);
        assert!((pcts.p1 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_statistics_default() {
        let stats = Statistics::default();
        assert_eq!(stats.n, 0);
        assert!((stats.mean - 0.0).abs() < 1e-10);
        assert!((stats.std - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_horizon_time_points_count() {
        let horizon = TimeHorizon::years(1).with_step(TimeStep::Monthly);
        let points = horizon.time_points();
        // n_steps is 12, time_points has n+1 elements
        assert_eq!(points.len(), 13);
    }

    #[test]
    fn test_percentile_clamp_out_of_range() {
        let values = vec![1.0, 2.0, 3.0];
        // p < 0 should clamp to 0
        let p_neg = percentile(&values, -0.5);
        assert!((p_neg - 1.0).abs() < 0.001);
        // p > 1 should clamp to 1
        let p_over = percentile(&values, 1.5);
        assert!((p_over - 3.0).abs() < 0.001);
    }
