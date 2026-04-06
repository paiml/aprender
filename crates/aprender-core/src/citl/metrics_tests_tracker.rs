
    // ==================== MetricsTracker Tests ====================

    #[test]
    fn test_metrics_tracker_new() {
        let tracker = MetricsTracker::new();
        assert_eq!(tracker.fix_attempts().total(), 0);
        assert_eq!(tracker.compilation_times().count(), 0);
    }

    #[test]
    fn test_metrics_tracker_record_fix_attempt() {
        let mut tracker = MetricsTracker::new();

        tracker.record_fix_attempt(true, "E0308");
        tracker.record_fix_attempt(true, "E0308");
        tracker.record_fix_attempt(false, "E0382");

        assert_eq!(tracker.fix_attempts().total(), 3);
        assert_eq!(tracker.fix_attempts().successes(), 2);
        assert_eq!(tracker.fix_attempts().failures(), 1);
    }

    #[test]
    fn test_metrics_tracker_record_pattern_use() {
        let mut tracker = MetricsTracker::new();

        tracker.record_pattern_use(0, true);
        tracker.record_pattern_use(0, true);
        tracker.record_pattern_use(1, false);

        assert_eq!(tracker.pattern_usage().usage_count(0), 2);
        assert_eq!(tracker.pattern_usage().usage_count(1), 1);
        assert!((tracker.pattern_usage().pattern_success_rate(0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_metrics_tracker_record_compilation_time() {
        let mut tracker = MetricsTracker::new();

        tracker.record_compilation_time(Duration::from_millis(100));
        tracker.record_compilation_time(Duration::from_millis(200));
        tracker.record_compilation_time(Duration::from_millis(300));

        assert_eq!(tracker.compilation_times().count(), 3);
        assert!((tracker.compilation_times().average_ms() - 200.0).abs() < 0.1);
    }

    #[test]
    fn test_metrics_tracker_record_convergence() {
        let mut tracker = MetricsTracker::new();

        tracker.record_convergence(2, true);
        tracker.record_convergence(3, true);
        tracker.record_convergence(5, false);

        assert_eq!(tracker.convergence().total_attempts(), 3);
        assert!((tracker.convergence().success_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_metrics_tracker_summary() {
        let mut tracker = MetricsTracker::new();

        tracker.record_fix_attempt(true, "E0308");
        tracker.record_fix_attempt(true, "E0308");
        tracker.record_fix_attempt(false, "E0382");
        tracker.record_compilation_time(Duration::from_millis(100));
        tracker.record_convergence(2, true);

        let summary = tracker.summary();
        assert_eq!(summary.total_fix_attempts, 3);
        assert!((summary.fix_success_rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_metrics_tracker_reset() {
        let mut tracker = MetricsTracker::new();

        tracker.record_fix_attempt(true, "E0308");
        tracker.record_compilation_time(Duration::from_millis(100));

        tracker.reset();

        assert_eq!(tracker.fix_attempts().total(), 0);
        assert_eq!(tracker.compilation_times().count(), 0);
    }

    // ==================== FixAttemptMetrics Tests ====================

    #[test]
    fn test_fix_attempt_metrics_new() {
        let metrics = FixAttemptMetrics::new();
        assert_eq!(metrics.total(), 0);
        assert_eq!(metrics.successes(), 0);
        assert_eq!(metrics.failures(), 0);
    }

    #[test]
    fn test_fix_attempt_metrics_success_rate() {
        let mut metrics = FixAttemptMetrics::new();
        metrics.record(true);
        metrics.record(true);
        metrics.record(false);
        metrics.record(false);

        assert!((metrics.success_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_fix_attempt_metrics_empty_success_rate() {
        let metrics = FixAttemptMetrics::new();
        assert!((metrics.success_rate() - 0.0).abs() < 0.001);
    }

    // ==================== PatternUsageMetrics Tests ====================

    #[test]
    fn test_pattern_usage_metrics_new() {
        let metrics = PatternUsageMetrics::new();
        assert_eq!(metrics.total_patterns_used(), 0);
        assert_eq!(metrics.usage_count(0), 0);
    }

    #[test]
    fn test_pattern_usage_metrics_record() {
        let mut metrics = PatternUsageMetrics::new();
        metrics.record(0, true);
        metrics.record(0, false);
        metrics.record(1, true);

        assert_eq!(metrics.usage_count(0), 2);
        assert_eq!(metrics.usage_count(1), 1);
        assert_eq!(metrics.total_patterns_used(), 2);
    }

    #[test]
    fn test_pattern_usage_metrics_most_used() {
        let mut metrics = PatternUsageMetrics::new();
        metrics.record(0, true);
        metrics.record(0, true);
        metrics.record(0, true);
        metrics.record(1, true);
        metrics.record(2, true);
        metrics.record(2, true);

        let most_used = metrics.most_used(2);
        assert_eq!(most_used.len(), 2);
        assert_eq!(most_used[0], (0, 3));
        assert_eq!(most_used[1], (2, 2));
    }

    // ==================== CompilationTimeMetrics Tests ====================

    #[test]
    fn test_compilation_time_metrics_new() {
        let metrics = CompilationTimeMetrics::new();
        assert_eq!(metrics.count(), 0);
        assert!(metrics.min_time().is_none());
        assert!(metrics.max_time().is_none());
    }

    #[test]
    fn test_compilation_time_metrics_record() {
        let mut metrics = CompilationTimeMetrics::new();
        metrics.record(Duration::from_millis(50));
        metrics.record(Duration::from_millis(100));
        metrics.record(Duration::from_millis(150));

        assert_eq!(metrics.count(), 3);
        assert_eq!(metrics.min_time(), Some(Duration::from_millis(50)));
        assert_eq!(metrics.max_time(), Some(Duration::from_millis(150)));
        assert!((metrics.average_ms() - 100.0).abs() < 0.1);
    }

    // ==================== ErrorFrequencyMetrics Tests ====================

    #[test]
    fn test_error_frequency_metrics_new() {
        let metrics = ErrorFrequencyMetrics::new();
        assert_eq!(metrics.total(), 0);
        assert_eq!(metrics.unique_errors(), 0);
    }

    #[test]
    fn test_error_frequency_metrics_record() {
        let mut metrics = ErrorFrequencyMetrics::new();
        metrics.record("E0308");
        metrics.record("E0308");
        metrics.record("E0382");

        assert_eq!(metrics.count("E0308"), 2);
        assert_eq!(metrics.count("E0382"), 1);
        assert_eq!(metrics.total(), 3);
        assert_eq!(metrics.unique_errors(), 2);
    }

    #[test]
    fn test_error_frequency_metrics_top_n() {
        let mut metrics = ErrorFrequencyMetrics::new();
        metrics.record("E0308");
        metrics.record("E0308");
        metrics.record("E0308");
        metrics.record("E0382");
        metrics.record("E0277");
        metrics.record("E0277");

        let top = metrics.top_n(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0], ("E0308".to_string(), 3));
        assert_eq!(top[1], ("E0277".to_string(), 2));
    }

    // ==================== ConvergenceMetrics Tests ====================

    #[test]
    fn test_convergence_metrics_new() {
        let metrics = ConvergenceMetrics::new();
        assert_eq!(metrics.total_attempts(), 0);
        assert!((metrics.average_iterations() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_convergence_metrics_record() {
        let mut metrics = ConvergenceMetrics::new();
        metrics.record(1, true);
        metrics.record(3, true);
        metrics.record(5, false);

        assert_eq!(metrics.total_attempts(), 3);
        assert!((metrics.average_iterations() - 3.0).abs() < 0.001);
        assert!((metrics.success_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_convergence_metrics_histogram() {
        let mut metrics = ConvergenceMetrics::new();
        metrics.record(1, true);
        metrics.record(1, true);
        metrics.record(2, true);
        metrics.record(3, false);

        let hist = metrics.histogram();
        assert_eq!(*hist.get(&1).unwrap_or(&0), 2);
        assert_eq!(*hist.get(&2).unwrap_or(&0), 1);
        assert_eq!(*hist.get(&3).unwrap_or(&0), 1);
    }

    // ==================== MetricsSummary Tests ====================

    #[test]
    fn test_metrics_summary_to_report() {
        let summary = MetricsSummary {
            total_fix_attempts: 100,
            fix_success_rate: 0.75,
            total_compilations: 200,
            avg_compilation_time_ms: 50.5,
            most_common_errors: vec![("E0308".to_string(), 50), ("E0382".to_string(), 30)],
            avg_iterations_to_fix: 2.5,
            convergence_rate: 0.8,
            session_duration: Duration::from_secs(120),
        };

        let report = summary.to_report();
        assert!(report.contains("Fix Attempts: 100"));
        assert!(report.contains("75.0%"));
        assert!(report.contains("E0308"));
    }

    // ==================== Coverage: MetricsTracker Default impl ====================

    #[test]
    fn test_metrics_tracker_default() {
        let tracker = MetricsTracker::default();
        assert_eq!(tracker.fix_attempts().total(), 0);
        assert_eq!(tracker.compilation_times().count(), 0);
        assert_eq!(tracker.convergence().total_attempts(), 0);
        assert_eq!(tracker.error_frequencies().total(), 0);
        assert_eq!(tracker.pattern_usage().total_patterns_used(), 0);
    }

    // ==================== Coverage: session_duration ====================

    #[test]
    fn test_metrics_tracker_session_duration() {
        let tracker = MetricsTracker::new();
        let duration = tracker.session_duration();
        // Session just started, duration should be very short
        assert!(duration.as_secs() < 10);
    }

    // ==================== Coverage: FixAttemptMetrics Default impl ====================

    #[test]
    fn test_fix_attempt_metrics_default() {
        let metrics = FixAttemptMetrics::default();
        assert_eq!(metrics.total(), 0);
        assert_eq!(metrics.successes(), 0);
        assert_eq!(metrics.failures(), 0);
        assert!((metrics.success_rate() - 0.0).abs() < 0.001);
    }

    // ==================== Coverage: FixAttemptMetrics all failures ====================

    #[test]
    fn test_fix_attempt_metrics_all_failures() {
        let mut metrics = FixAttemptMetrics::new();
        metrics.record(false);
        metrics.record(false);
        metrics.record(false);
        assert_eq!(metrics.successes(), 0);
        assert_eq!(metrics.failures(), 3);
        assert!((metrics.success_rate() - 0.0).abs() < 0.001);
    }

    // ==================== Coverage: FixAttemptMetrics all successes ====================

    #[test]
    fn test_fix_attempt_metrics_all_successes() {
        let mut metrics = FixAttemptMetrics::new();
        metrics.record(true);
        metrics.record(true);
        assert_eq!(metrics.successes(), 2);
        assert_eq!(metrics.failures(), 0);
        assert!((metrics.success_rate() - 1.0).abs() < 0.001);
    }

    // ==================== Coverage: PatternUsageMetrics Default impl ====================

    #[test]
    fn test_pattern_usage_metrics_default() {
        let metrics = PatternUsageMetrics::default();
        assert_eq!(metrics.total_patterns_used(), 0);
        assert_eq!(metrics.usage_count(0), 0);
        assert!((metrics.pattern_success_rate(0) - 0.0).abs() < 0.001);
    }

    // ==================== Coverage: PatternUsageMetrics success_rate for unused pattern ====================

    #[test]
    fn test_pattern_usage_metrics_unused_pattern_rate() {
        let metrics = PatternUsageMetrics::new();
        // Unused pattern should return 0.0 success rate
        assert!((metrics.pattern_success_rate(42) - 0.0).abs() < 0.001);
    }

    // ==================== Coverage: PatternUsageMetrics with 0% success rate ====================

    #[test]
    fn test_pattern_usage_metrics_zero_success_rate() {
        let mut metrics = PatternUsageMetrics::new();
        metrics.record(0, false);
        metrics.record(0, false);
        assert!((metrics.pattern_success_rate(0) - 0.0).abs() < 0.001);
    }

    // ==================== Coverage: PatternUsageMetrics most_used empty ====================

    #[test]
    fn test_pattern_usage_metrics_most_used_empty() {
        let metrics = PatternUsageMetrics::new();
        let most = metrics.most_used(5);
        assert!(most.is_empty());
    }

    // ==================== Coverage: CompilationTimeMetrics Default impl ====================

    #[test]
    fn test_compilation_time_metrics_default() {
        let metrics = CompilationTimeMetrics::default();
        assert_eq!(metrics.count(), 0);
        assert!((metrics.average_ms() - 0.0).abs() < 0.001);
        assert!(metrics.min_time().is_none());
        assert!(metrics.max_time().is_none());
        assert_eq!(metrics.total_time(), Duration::ZERO);
    }

    // ==================== Coverage: CompilationTimeMetrics min/max boundary ====================

    #[test]
    fn test_compilation_time_metrics_min_max_single() {
        let mut metrics = CompilationTimeMetrics::new();
        metrics.record(Duration::from_millis(100));

        assert_eq!(metrics.min_time(), Some(Duration::from_millis(100)));
        assert_eq!(metrics.max_time(), Some(Duration::from_millis(100)));
    }

    // ==================== Coverage: CompilationTimeMetrics min does not update for larger value ====================

    #[test]
    fn test_compilation_time_metrics_min_no_update_larger() {
        let mut metrics = CompilationTimeMetrics::new();
        metrics.record(Duration::from_millis(50));
        metrics.record(Duration::from_millis(100)); // larger, should not update min
        metrics.record(Duration::from_millis(75));

        assert_eq!(metrics.min_time(), Some(Duration::from_millis(50)));
    }

    // ==================== Coverage: CompilationTimeMetrics max does not update for smaller value ====================

    #[test]
    fn test_compilation_time_metrics_max_no_update_smaller() {
        let mut metrics = CompilationTimeMetrics::new();
        metrics.record(Duration::from_millis(200));
        metrics.record(Duration::from_millis(100)); // smaller, should not update max
        metrics.record(Duration::from_millis(150));

        assert_eq!(metrics.max_time(), Some(Duration::from_millis(200)));
    }

    // ==================== Coverage: CompilationTimeMetrics total_time ====================

    #[test]
    fn test_compilation_time_metrics_total_time() {
        let mut metrics = CompilationTimeMetrics::new();
        metrics.record(Duration::from_millis(100));
        metrics.record(Duration::from_millis(200));

        assert_eq!(metrics.total_time(), Duration::from_millis(300));
    }

    // ==================== Coverage: ErrorFrequencyMetrics Default impl ====================

    #[test]
    fn test_error_frequency_metrics_default() {
        let metrics = ErrorFrequencyMetrics::default();
        assert_eq!(metrics.total(), 0);
        assert_eq!(metrics.unique_errors(), 0);
        assert_eq!(metrics.count("E0308"), 0);
    }

    // ==================== Coverage: ErrorFrequencyMetrics top_n with fewer entries ====================

    #[test]
    fn test_error_frequency_metrics_top_n_fewer_than_n() {
        let mut metrics = ErrorFrequencyMetrics::new();
        metrics.record("E0308");
        let top = metrics.top_n(5);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0], ("E0308".to_string(), 1));
    }
