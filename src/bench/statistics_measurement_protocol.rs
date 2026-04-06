
// ============================================================================
// Tests (T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // MeasurementProtocol Tests
    // =========================================================================

    #[test]
    fn test_measurement_protocol_default() {
        let protocol = MeasurementProtocol::default();
        assert_eq!(protocol.latency_samples, 100);
        assert_eq!(
            protocol.latency_percentiles,
            vec![50.0, 90.0, 95.0, 99.0, 99.9]
        );
        assert_eq!(protocol.throughput_duration, Duration::from_secs(60));
        assert_eq!(protocol.throughput_ramp_up, Duration::from_secs(10));
        assert_eq!(protocol.memory_samples, 10);
        assert_eq!(protocol.memory_interval, Duration::from_secs(1));
    }

    #[test]
    fn test_measurement_protocol_new() {
        let protocol = MeasurementProtocol::new();
        assert_eq!(protocol.latency_samples, 100);
    }

    #[test]
    fn test_measurement_protocol_with_latency_samples() {
        let protocol = MeasurementProtocol::new().with_latency_samples(200);
        assert_eq!(protocol.latency_samples, 200);
    }

    #[test]
    fn test_measurement_protocol_with_percentiles() {
        let protocol = MeasurementProtocol::new().with_percentiles(vec![50.0, 99.0]);
        assert_eq!(protocol.latency_percentiles, vec![50.0, 99.0]);
    }

    #[test]
    fn test_measurement_protocol_with_throughput_duration() {
        let protocol =
            MeasurementProtocol::new().with_throughput_duration(Duration::from_secs(120));
        assert_eq!(protocol.throughput_duration, Duration::from_secs(120));
    }

    #[test]
    fn test_measurement_protocol_with_memory_samples() {
        let protocol = MeasurementProtocol::new().with_memory_samples(20);
        assert_eq!(protocol.memory_samples, 20);
    }

    #[test]
    fn test_measurement_protocol_builder_chain() {
        let protocol = MeasurementProtocol::new()
            .with_latency_samples(50)
            .with_percentiles(vec![90.0, 99.0])
            .with_throughput_duration(Duration::from_secs(30))
            .with_memory_samples(5);

        assert_eq!(protocol.latency_samples, 50);
        assert_eq!(protocol.latency_percentiles, vec![90.0, 99.0]);
        assert_eq!(protocol.throughput_duration, Duration::from_secs(30));
        assert_eq!(protocol.memory_samples, 5);
    }

    // =========================================================================
    // LatencyStatistics Tests
    // =========================================================================

    #[test]
    fn test_latency_statistics_from_samples_uniform() {
        let samples: Vec<Duration> = (1..=10).map(|i| Duration::from_millis(i * 10)).collect();

        let stats = LatencyStatistics::from_samples(&samples);

        assert_eq!(stats.samples, 10);
        assert_eq!(stats.min, Duration::from_millis(10));
        assert_eq!(stats.max, Duration::from_millis(100));
        // Mean = (10+20+...+100)/10 = 550/10 = 55ms
        assert_eq!(stats.mean.as_millis(), 55);
    }

    #[test]
    fn test_latency_statistics_from_samples_single() {
        let samples = vec![Duration::from_millis(100)];
        let stats = LatencyStatistics::from_samples(&samples);

        assert_eq!(stats.samples, 1);
        assert_eq!(stats.min, Duration::from_millis(100));
        assert_eq!(stats.max, Duration::from_millis(100));
        assert_eq!(stats.mean, Duration::from_millis(100));
        assert_eq!(stats.p50, Duration::from_millis(100));
    }

    #[test]
    fn test_latency_statistics_percentiles() {
        // 100 samples from 1ms to 100ms
        let samples: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();

        let stats = LatencyStatistics::from_samples(&samples);

        // p50 should be around 50ms
        assert!(stats.p50.as_millis() >= 49 && stats.p50.as_millis() <= 51);
        // p90 should be around 90ms
        assert!(stats.p90.as_millis() >= 89 && stats.p90.as_millis() <= 91);
        // p99 should be around 99ms
        assert!(stats.p99.as_millis() >= 98 && stats.p99.as_millis() <= 100);
    }

    #[test]
    #[should_panic(expected = "samples must not be empty")]
    fn test_latency_statistics_empty_samples_panics() {
        let samples: Vec<Duration> = vec![];
        let _ = LatencyStatistics::from_samples(&samples);
    }

    // =========================================================================
    // RegressionDetector Tests
    // =========================================================================

    #[test]
    fn test_regression_detector_default() {
        let detector = RegressionDetector::default();
        assert!((detector.warning_threshold - 0.02).abs() < f64::EPSILON);
        assert!((detector.failure_threshold - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_regression_detector_clone() {
        let detector = RegressionDetector::default();
        let cloned = detector.clone();
        assert!((detector.warning_threshold - cloned.warning_threshold).abs() < f64::EPSILON);
        assert!((detector.failure_threshold - cloned.failure_threshold).abs() < f64::EPSILON);
    }

    // =========================================================================
    // WelchTTestResult Tests
    // =========================================================================

    #[test]
    fn test_welch_t_test_result_fields() {
        // Create a result directly
        let result = WelchTTestResult {
            t_statistic: 2.5,
            degrees_of_freedom: 18.0,
            p_value: 0.02,
            significant: true,
        };
        assert!((result.t_statistic - 2.5).abs() < f64::EPSILON);
        assert!((result.degrees_of_freedom - 18.0).abs() < f64::EPSILON);
        assert!((result.p_value - 0.02).abs() < f64::EPSILON);
        assert!(result.significant);
    }

    #[test]
    fn test_welch_t_test_same_samples() {
        // Same distributions should have high p-value (not significant)
        let a: Vec<f64> = (1..=20).map(|i| 100.0 + i as f64).collect();
        let b: Vec<f64> = (1..=20).map(|i| 100.0 + i as f64).collect();

        let result = welch_t_test(&a, &b, 0.05);
        // p-value should be > 0.05 (not significant)
        assert!(result.p_value > 0.05);
        assert!(!result.significant);
    }

    #[test]
    fn test_welch_t_test_different_samples() {
        // Clearly different distributions with some variance
        let a: Vec<f64> = (1..=30).map(|i| 100.0 + (i as f64 % 5.0)).collect();
        let b: Vec<f64> = (1..=30).map(|i| 200.0 + (i as f64 % 5.0)).collect();

        let result = welch_t_test(&a, &b, 0.05);
        // Should be significant difference (100 unit difference with small variance)
        assert!(result.significant);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_welch_t_test_different_alpha() {
        // Same samples, test with different alpha values
        let a: Vec<f64> = (1..=20).map(|i| 100.0 + i as f64).collect();
        let b: Vec<f64> = (1..=20).map(|i| 110.0 + i as f64).collect();

        let result_05 = welch_t_test(&a, &b, 0.05);
        let result_01 = welch_t_test(&a, &b, 0.01);

        // p-value should be the same, but significance may differ
        assert!((result_05.p_value - result_01.p_value).abs() < 0.001);
    }

    // =========================================================================
    // Statistical Function Tests
    // =========================================================================

    #[test]
    fn test_gamma_ln_positive() {
        // gamma(1) = 1, ln(1) = 0
        let result = gamma_ln(1.0);
        assert!(result.abs() < 0.01);

        // gamma(2) = 1, ln(1) = 0
        let result = gamma_ln(2.0);
        assert!(result.abs() < 0.01);

        // gamma(5) = 24, ln(24) ≈ 3.178
        let result = gamma_ln(5.0);
        assert!((result - 3.178).abs() < 0.1);
    }

    #[test]
    fn test_gamma_ln_negative_or_zero() {
        assert!(gamma_ln(0.0).is_infinite());
        assert!(gamma_ln(-1.0).is_infinite());
    }
}
