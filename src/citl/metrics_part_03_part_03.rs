
    // ==================== Coverage: ConvergenceMetrics Default impl ====================

    #[test]
    fn test_convergence_metrics_default() {
        let metrics = ConvergenceMetrics::default();
        assert_eq!(metrics.total_attempts(), 0);
        assert!((metrics.average_iterations() - 0.0).abs() < 0.001);
        assert!((metrics.success_rate() - 0.0).abs() < 0.001);
        assert!(metrics.histogram().is_empty());
    }

    // ==================== Coverage: ConvergenceMetrics all failures ====================

    #[test]
    fn test_convergence_metrics_all_failures() {
        let mut metrics = ConvergenceMetrics::new();
        metrics.record(5, false);
        metrics.record(10, false);

        assert_eq!(metrics.total_attempts(), 2);
        assert!((metrics.success_rate() - 0.0).abs() < 0.001);
        assert!((metrics.average_iterations() - 7.5).abs() < 0.001);
    }

    // ==================== Coverage: ConvergenceMetrics all successes ====================

    #[test]
    fn test_convergence_metrics_all_successes() {
        let mut metrics = ConvergenceMetrics::new();
        metrics.record(1, true);
        metrics.record(2, true);

        assert_eq!(metrics.total_attempts(), 2);
        assert!((metrics.success_rate() - 1.0).abs() < 0.001);
        assert!((metrics.average_iterations() - 1.5).abs() < 0.001);
    }

    // ==================== Coverage: MetricsSummary to_report with empty errors ====================

    #[test]
    fn test_metrics_summary_to_report_no_errors() {
        let summary = MetricsSummary {
            total_fix_attempts: 0,
            fix_success_rate: 0.0,
            total_compilations: 0,
            avg_compilation_time_ms: 0.0,
            most_common_errors: vec![],
            avg_iterations_to_fix: 0.0,
            convergence_rate: 0.0,
            session_duration: Duration::from_secs(0),
        };

        let report = summary.to_report();
        assert!(report.contains("Fix Attempts: 0"));
        assert!(report.contains("Compilations: 0"));
        assert!(report.contains("Convergence: 0.0%"));
        // Should NOT contain "Most Common Errors" section
        assert!(!report.contains("Most Common Errors"));
    }

    // ==================== Coverage: MetricsSummary to_report all sections ====================

    #[test]
    fn test_metrics_summary_to_report_all_sections() {
        let summary = MetricsSummary {
            total_fix_attempts: 50,
            fix_success_rate: 0.5,
            total_compilations: 100,
            avg_compilation_time_ms: 25.0,
            most_common_errors: vec![
                ("E0308".to_string(), 20),
                ("E0382".to_string(), 15),
                ("E0277".to_string(), 10),
            ],
            avg_iterations_to_fix: 3.5,
            convergence_rate: 0.65,
            session_duration: Duration::from_secs(300),
        };

        let report = summary.to_report();
        assert!(report.contains("CITL Metrics Summary"));
        assert!(report.contains("Fix Attempts: 50"));
        assert!(report.contains("50.0%"));
        assert!(report.contains("Compilations: 100"));
        assert!(report.contains("25.0ms"));
        assert!(report.contains("65.0%"));
        assert!(report.contains("3.5"));
        assert!(report.contains("E0308: 20"));
        assert!(report.contains("E0382: 15"));
        assert!(report.contains("E0277: 10"));
        assert!(report.contains("Session Duration"));
    }

    // ==================== Coverage: MetricsSummary Debug impl ====================

    #[test]
    fn test_metrics_summary_debug() {
        let summary = MetricsSummary {
            total_fix_attempts: 1,
            fix_success_rate: 1.0,
            total_compilations: 1,
            avg_compilation_time_ms: 10.0,
            most_common_errors: vec![],
            avg_iterations_to_fix: 1.0,
            convergence_rate: 1.0,
            session_duration: Duration::from_secs(1),
        };
        let debug_str = format!("{:?}", summary);
        assert!(debug_str.contains("MetricsSummary"));
    }

    // ==================== Coverage: MetricsSummary Clone impl ====================

    #[test]
    fn test_metrics_summary_clone() {
        let summary = MetricsSummary {
            total_fix_attempts: 5,
            fix_success_rate: 0.8,
            total_compilations: 10,
            avg_compilation_time_ms: 42.0,
            most_common_errors: vec![("E0308".to_string(), 3)],
            avg_iterations_to_fix: 2.0,
            convergence_rate: 0.9,
            session_duration: Duration::from_secs(60),
        };
        let cloned = summary.clone();
        assert_eq!(cloned.total_fix_attempts, 5);
        assert!((cloned.fix_success_rate - 0.8).abs() < 0.001);
        assert_eq!(cloned.most_common_errors.len(), 1);
    }

    // ==================== Coverage: MetricsTracker combined workflow ====================

    #[test]
    fn test_metrics_tracker_full_workflow() {
        let mut tracker = MetricsTracker::new();

        // Record multiple fix attempts
        tracker.record_fix_attempt(true, "E0308");
        tracker.record_fix_attempt(false, "E0308");
        tracker.record_fix_attempt(true, "E0382");

        // Record pattern usage
        tracker.record_pattern_use(0, true);
        tracker.record_pattern_use(0, false);
        tracker.record_pattern_use(1, true);

        // Record compilations
        tracker.record_compilation_time(Duration::from_millis(50));
        tracker.record_compilation_time(Duration::from_millis(150));

        // Record convergence
        tracker.record_convergence(1, true);
        tracker.record_convergence(5, false);

        // Check all accessors
        assert_eq!(tracker.fix_attempts().total(), 3);
        assert_eq!(tracker.fix_attempts().successes(), 2);
        assert_eq!(tracker.fix_attempts().failures(), 1);
        assert_eq!(tracker.pattern_usage().total_patterns_used(), 2);
        assert_eq!(tracker.compilation_times().count(), 2);
        assert_eq!(tracker.convergence().total_attempts(), 2);
        assert_eq!(tracker.error_frequencies().total(), 3);
        assert_eq!(tracker.error_frequencies().unique_errors(), 2);

        // Get summary
        let summary = tracker.summary();
        assert_eq!(summary.total_fix_attempts, 3);
        assert!(summary.fix_success_rate > 0.6);
        assert_eq!(summary.total_compilations, 2);

        // Reset and verify
        tracker.reset();
        assert_eq!(tracker.fix_attempts().total(), 0);
        assert_eq!(tracker.compilation_times().count(), 0);
        assert_eq!(tracker.convergence().total_attempts(), 0);
        assert_eq!(tracker.error_frequencies().total(), 0);
        assert_eq!(tracker.pattern_usage().total_patterns_used(), 0);
    }

    // ==================== Coverage: FixAttemptMetrics Clone impl ====================

    #[test]
    fn test_fix_attempt_metrics_clone() {
        let mut metrics = FixAttemptMetrics::new();
        metrics.record(true);
        metrics.record(false);
        let cloned = metrics.clone();
        assert_eq!(cloned.successes(), 1);
        assert_eq!(cloned.failures(), 1);
    }

    // ==================== Coverage: PatternUsageMetrics Clone impl ====================

    #[test]
    fn test_pattern_usage_metrics_clone() {
        let mut metrics = PatternUsageMetrics::new();
        metrics.record(0, true);
        let cloned = metrics.clone();
        assert_eq!(cloned.usage_count(0), 1);
        assert_eq!(cloned.total_patterns_used(), 1);
    }

    // ==================== Coverage: CompilationTimeMetrics Clone impl ====================

    #[test]
    fn test_compilation_time_metrics_clone() {
        let mut metrics = CompilationTimeMetrics::new();
        metrics.record(Duration::from_millis(100));
        let cloned = metrics.clone();
        assert_eq!(cloned.count(), 1);
        assert_eq!(cloned.min_time(), Some(Duration::from_millis(100)));
    }

    // ==================== Coverage: ErrorFrequencyMetrics Clone impl ====================

    #[test]
    fn test_error_frequency_metrics_clone() {
        let mut metrics = ErrorFrequencyMetrics::new();
        metrics.record("E0308");
        metrics.record("E0308");
        let cloned = metrics.clone();
        assert_eq!(cloned.count("E0308"), 2);
        assert_eq!(cloned.total(), 2);
    }

    // ==================== Coverage: ConvergenceMetrics Clone impl ====================

    #[test]
    fn test_convergence_metrics_clone() {
        let mut metrics = ConvergenceMetrics::new();
        metrics.record(3, true);
        let cloned = metrics.clone();
        assert_eq!(cloned.total_attempts(), 1);
        assert_eq!(*cloned.histogram().get(&3).unwrap_or(&0), 1);
    }
