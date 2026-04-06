//! Field access, default values, and boundary tests for stress testing types.

use super::*;

#[test]
fn test_frame_profile_all_fields() {
    // Coverage for all FrameProfile fields including memory_bytes
    let profile = FrameProfile {
        cycle: 10,
        duration_ms: 50,
        memory_bytes: 4096,
        tests_passed: 8,
        tests_failed: 2,
        input_seed: 999,
        input_size: 128,
    };

    assert_eq!(profile.cycle, 10);
    assert_eq!(profile.duration_ms, 50);
    assert_eq!(profile.memory_bytes, 4096);
    assert_eq!(profile.tests_passed, 8);
    assert_eq!(profile.tests_failed, 2);
    assert_eq!(profile.input_seed, 999);
    assert_eq!(profile.input_size, 128);
}

#[test]
fn test_stress_report_default_values() {
    // Coverage for StressReport default derive
    let report = StressReport::default();
    assert!(report.frames.is_empty());
    assert_eq!(report.cycles_completed, 0);
    assert_eq!(report.total_passed, 0);
    assert_eq!(report.total_failed, 0);
    assert!(report.anomalies.is_empty());
}

#[test]
fn test_performance_result_all_fields() {
    // Coverage for all PerformanceResult fields
    let result = PerformanceResult {
        passed: true,
        max_frame_ms: 75,
        mean_frame_ms: 50.5,
        variance: 0.12,
        pass_rate: 0.98,
        violations: vec!["violation1".to_string(), "violation2".to_string()],
    };

    assert!(result.passed);
    assert_eq!(result.max_frame_ms, 75);
    assert!((result.mean_frame_ms - 50.5).abs() < 0.001);
    assert!((result.variance - 0.12).abs() < 0.001);
    assert!((result.pass_rate - 0.98).abs() < 0.001);
    assert_eq!(result.violations.len(), 2);
}

#[test]
fn test_performance_thresholds_all_fields() {
    // Coverage for all PerformanceThresholds fields
    let thresholds = PerformanceThresholds {
        max_frame_time_ms: 150,
        max_memory_bytes: 1024 * 1024,
        max_timing_variance: 0.3,
        max_failure_rate: 0.1,
    };

    assert_eq!(thresholds.max_frame_time_ms, 150);
    assert_eq!(thresholds.max_memory_bytes, 1024 * 1024);
    assert!((thresholds.max_timing_variance - 0.3).abs() < 0.001);
    assert!((thresholds.max_failure_rate - 0.1).abs() < 0.001);
}

#[test]
fn test_anomaly_all_fields() {
    // Coverage for all Anomaly fields
    let anomaly = Anomaly {
        cycle: 5,
        kind: AnomalyKind::TimingSpike,
        description: "Timing spike detected at cycle 5".to_string(),
    };

    assert_eq!(anomaly.cycle, 5);
    assert_eq!(anomaly.kind, AnomalyKind::TimingSpike);
    assert!(anomaly.description.contains("cycle 5"));
}

#[test]
fn test_stress_config_all_fields() {
    // Coverage for all StressConfig fields
    let config = StressConfig {
        cycles: 200,
        interval_ms: 50,
        seed: 12345,
        min_input_size: 32,
        max_input_size: 1024,
        thresholds: PerformanceThresholds {
            max_frame_time_ms: 200,
            max_memory_bytes: 128 * 1024 * 1024,
            max_timing_variance: 0.25,
            max_failure_rate: 0.02,
        },
    };

    assert_eq!(config.cycles, 200);
    assert_eq!(config.interval_ms, 50);
    assert_eq!(config.seed, 12345);
    assert_eq!(config.min_input_size, 32);
    assert_eq!(config.max_input_size, 1024);
    assert_eq!(config.thresholds.max_frame_time_ms, 200);
}

#[test]
fn test_stress_report_with_timing_variance() {
    // Coverage for timing_variance calculation with actual variance
    let mut report = StressReport::default();

    // Add frames with known variance pattern
    for i in 0..5 {
        report.add_frame(FrameProfile {
            cycle: i,
            duration_ms: if i % 2 == 0 { 50 } else { 150 },
            tests_passed: 1,
            tests_failed: 0,
            ..Default::default()
        });
    }

    let variance = report.timing_variance();
    // Variance should be non-zero with alternating 50/150ms timings
    assert!(variance > 0.0);
}

#[test]
fn test_stress_report_anomalies_field() {
    // Coverage for anomalies field in StressReport
    let mut report = StressReport::default();

    // Manually add an anomaly
    report.anomalies.push(Anomaly {
        cycle: 0,
        kind: AnomalyKind::HighMemory,
        description: "Memory exceeded".to_string(),
    });

    assert_eq!(report.anomalies.len(), 1);
    assert_eq!(report.anomalies[0].kind, AnomalyKind::HighMemory);
}

#[test]
fn test_anomaly_kind_copy_semantics() {
    // Coverage for AnomalyKind Copy trait
    let kind1 = AnomalyKind::NonDeterministic;
    let kind2 = kind1; // Copy
    let kind3 = kind1; // Another copy

    assert_eq!(kind1, kind2);
    assert_eq!(kind2, kind3);
}

#[test]
fn test_performance_result_with_empty_violations() {
    // Coverage for PerformanceResult with no violations
    let result = PerformanceResult {
        passed: true,
        max_frame_ms: 50,
        mean_frame_ms: 40.0,
        variance: 0.05,
        pass_rate: 1.0,
        violations: Vec::new(),
    };

    assert!(result.passed);
    assert!(result.violations.is_empty());
}
