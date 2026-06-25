
    // ========================================================================
    // PMAT-125 B4: pct_change / classify_metric pure-logic tests
    // ========================================================================

    #[test]
    fn test_pct_change_positive_increase() {
        assert!((pct_change(100.0, 150.0) - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_pct_change_decrease() {
        assert!((pct_change(200.0, 150.0) - (-25.0)).abs() < 1e-9);
    }

    #[test]
    fn test_pct_change_zero_baseline_returns_zero() {
        assert_eq!(pct_change(0.0, 123.0), 0.0);
        assert_eq!(pct_change(-1.0, 5.0), 0.0);
    }

    #[test]
    fn test_pct_change_no_change() {
        assert_eq!(pct_change(42.0, 42.0), 0.0);
    }

    #[test]
    fn test_classify_metric_latency_regression() {
        let mut regs = Vec::new();
        let mut imps = Vec::new();
        classify_metric(
            "p99", 20.0, 5.0, 10.0, 12.0, "ms", false, &mut regs, &mut imps,
        );
        assert_eq!(regs.len(), 1);
        assert!(imps.is_empty());
        assert!(regs[0].contains("p99"));
        assert!(regs[0].contains("slower"));
        assert!(regs[0].contains("ms"));
    }

    #[test]
    fn test_classify_metric_latency_improvement() {
        let mut regs = Vec::new();
        let mut imps = Vec::new();
        classify_metric(
            "p50", -30.0, 5.0, 10.0, 7.0, "ms", false, &mut regs, &mut imps,
        );
        assert!(regs.is_empty());
        assert_eq!(imps.len(), 1);
        assert!(imps[0].contains("faster"));
    }

    #[test]
    fn test_classify_metric_throughput_regression() {
        let mut regs = Vec::new();
        let mut imps = Vec::new();
        classify_metric(
            "tok/s", -15.0, 5.0, 100.0, 85.0, "tok/s", true, &mut regs, &mut imps,
        );
        assert_eq!(regs.len(), 1);
        assert!(imps.is_empty());
        assert!(regs[0].contains("slower"));
    }

    #[test]
    fn test_classify_metric_throughput_improvement() {
        let mut regs = Vec::new();
        let mut imps = Vec::new();
        classify_metric(
            "tok/s", 25.0, 5.0, 100.0, 125.0, "tok/s", true, &mut regs, &mut imps,
        );
        assert!(regs.is_empty());
        assert_eq!(imps.len(), 1);
        assert!(imps[0].contains("faster"));
    }

    #[test]
    fn test_classify_metric_within_threshold_is_neutral() {
        let mut regs = Vec::new();
        let mut imps = Vec::new();
        classify_metric(
            "p99", 2.0, 5.0, 10.0, 10.2, "ms", false, &mut regs, &mut imps,
        );
        assert!(regs.is_empty());
        assert!(imps.is_empty());
    }

    #[test]
    fn test_classify_metric_exactly_at_threshold_is_neutral() {
        let mut regs = Vec::new();
        let mut imps = Vec::new();
        classify_metric(
            "p99", 5.0, 5.0, 10.0, 10.5, "ms", false, &mut regs, &mut imps,
        );
        assert!(regs.is_empty());
        assert!(imps.is_empty());
    }

    #[test]
    fn test_classify_metric_appends_preserving_existing() {
        let mut regs = vec!["existing-reg".to_string()];
        let mut imps = vec!["existing-imp".to_string()];
        classify_metric(
            "p99", 30.0, 5.0, 10.0, 13.0, "ms", false, &mut regs, &mut imps,
        );
        assert_eq!(regs.len(), 2);
        assert_eq!(regs[0], "existing-reg");
        assert_eq!(imps.len(), 1);
    }
