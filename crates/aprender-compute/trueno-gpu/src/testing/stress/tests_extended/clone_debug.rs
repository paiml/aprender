//! Clone and Debug derive coverage tests for stress testing types.

use super::*;

#[test]
fn test_frame_profile_clone() {
    // Coverage for FrameProfile Clone derive
    let profile = FrameProfile {
        cycle: 5,
        duration_ms: 100,
        memory_bytes: 1024,
        tests_passed: 10,
        tests_failed: 2,
        input_seed: 12345,
        input_size: 256,
    };
    let cloned = profile.clone();
    assert_eq!(profile.cycle, cloned.cycle);
    assert_eq!(profile.duration_ms, cloned.duration_ms);
    assert_eq!(profile.memory_bytes, cloned.memory_bytes);
    assert_eq!(profile.tests_passed, cloned.tests_passed);
    assert_eq!(profile.tests_failed, cloned.tests_failed);
    assert_eq!(profile.input_seed, cloned.input_seed);
    assert_eq!(profile.input_size, cloned.input_size);
}

#[test]
fn test_stress_report_clone() {
    // Coverage for StressReport Clone derive
    let mut report = StressReport::default();
    report.add_frame(FrameProfile {
        cycle: 0,
        duration_ms: 50,
        tests_passed: 5,
        tests_failed: 1,
        ..Default::default()
    });
    report.anomalies.push(Anomaly {
        cycle: 0,
        kind: AnomalyKind::SlowFrame,
        description: "Test anomaly".to_string(),
    });

    let cloned = report.clone();
    assert_eq!(report.cycles_completed, cloned.cycles_completed);
    assert_eq!(report.total_passed, cloned.total_passed);
    assert_eq!(report.total_failed, cloned.total_failed);
    assert_eq!(report.frames.len(), cloned.frames.len());
    assert_eq!(report.anomalies.len(), cloned.anomalies.len());
}

#[test]
fn test_anomaly_clone() {
    // Coverage for Anomaly Clone derive
    let anomaly = Anomaly {
        cycle: 42,
        kind: AnomalyKind::HighMemory,
        description: "Memory exceeded threshold".to_string(),
    };
    let cloned = anomaly.clone();
    assert_eq!(anomaly.cycle, cloned.cycle);
    assert_eq!(anomaly.kind, cloned.kind);
    assert_eq!(anomaly.description, cloned.description);
}

#[test]
fn test_performance_thresholds_clone() {
    // Coverage for PerformanceThresholds Clone derive
    let thresholds = PerformanceThresholds {
        max_frame_time_ms: 50,
        max_memory_bytes: 1024,
        max_timing_variance: 0.1,
        max_failure_rate: 0.05,
    };
    let cloned = thresholds.clone();
    assert_eq!(thresholds.max_frame_time_ms, cloned.max_frame_time_ms);
    assert_eq!(thresholds.max_memory_bytes, cloned.max_memory_bytes);
    assert!((thresholds.max_timing_variance - cloned.max_timing_variance).abs() < 0.001);
    assert!((thresholds.max_failure_rate - cloned.max_failure_rate).abs() < 0.001);
}

#[test]
fn test_performance_result_clone() {
    // Coverage for PerformanceResult Clone derive
    let result = PerformanceResult {
        passed: false,
        max_frame_ms: 100,
        mean_frame_ms: 75.5,
        variance: 0.15,
        pass_rate: 0.95,
        violations: vec!["Test violation".to_string()],
    };
    let cloned = result.clone();
    assert_eq!(result.passed, cloned.passed);
    assert_eq!(result.max_frame_ms, cloned.max_frame_ms);
    assert!((result.mean_frame_ms - cloned.mean_frame_ms).abs() < 0.001);
    assert!((result.variance - cloned.variance).abs() < 0.001);
    assert!((result.pass_rate - cloned.pass_rate).abs() < 0.001);
    assert_eq!(result.violations.len(), cloned.violations.len());
}

#[test]
fn test_stress_rng_clone() {
    // Coverage for StressRng Clone derive
    let mut rng = StressRng::new(42);
    rng.next_u32(); // Advance state

    let cloned = rng.clone();

    // Both should produce same sequence from this point
    assert_eq!(rng.next_u32(), cloned.clone().next_u32());
}

#[test]
fn test_stress_config_clone() {
    // Coverage for StressConfig Clone derive
    let config = StressConfig {
        cycles: 50,
        interval_ms: 200,
        seed: 12345,
        min_input_size: 128,
        max_input_size: 1024,
        thresholds: PerformanceThresholds::default(),
    };
    let cloned = config.clone();
    assert_eq!(config.cycles, cloned.cycles);
    assert_eq!(config.interval_ms, cloned.interval_ms);
    assert_eq!(config.seed, cloned.seed);
    assert_eq!(config.min_input_size, cloned.min_input_size);
    assert_eq!(config.max_input_size, cloned.max_input_size);
}

#[test]
fn test_multiple_anomaly_kinds() {
    // Coverage for all AnomalyKind variants and Anomaly debug
    let anomalies = vec![
        Anomaly { cycle: 0, kind: AnomalyKind::SlowFrame, description: "Slow".to_string() },
        Anomaly { cycle: 1, kind: AnomalyKind::HighMemory, description: "High memory".to_string() },
        Anomaly {
            cycle: 2,
            kind: AnomalyKind::TestFailure,
            description: "Test failed".to_string(),
        },
        Anomaly { cycle: 3, kind: AnomalyKind::TimingSpike, description: "Spike".to_string() },
        Anomaly {
            cycle: 4,
            kind: AnomalyKind::NonDeterministic,
            description: "Non-deterministic".to_string(),
        },
    ];

    // Test Debug impl
    for a in &anomalies {
        let debug_str = format!("{:?}", a);
        assert!(debug_str.contains("Anomaly"));
    }

    // Verify all kinds are distinct
    assert_eq!(anomalies.len(), 5);
}

#[test]
fn test_stress_report_debug() {
    // Coverage for StressReport Debug derive
    let mut report = StressReport::default();
    report.add_frame(FrameProfile::default());

    let debug_str = format!("{:?}", report);
    assert!(debug_str.contains("StressReport"));
    assert!(debug_str.contains("frames"));
}

#[test]
fn test_frame_profile_debug() {
    // Coverage for FrameProfile Debug derive
    let profile = FrameProfile { cycle: 1, duration_ms: 100, ..Default::default() };

    let debug_str = format!("{:?}", profile);
    assert!(debug_str.contains("FrameProfile"));
    assert!(debug_str.contains("cycle"));
}

#[test]
fn test_performance_thresholds_debug() {
    // Coverage for PerformanceThresholds Debug derive
    let thresholds = PerformanceThresholds::default();
    let debug_str = format!("{:?}", thresholds);
    assert!(debug_str.contains("PerformanceThresholds"));
}

#[test]
fn test_performance_result_debug() {
    // Coverage for PerformanceResult Debug derive
    let result = PerformanceResult {
        passed: true,
        max_frame_ms: 50,
        mean_frame_ms: 40.0,
        variance: 0.1,
        pass_rate: 1.0,
        violations: vec![],
    };
    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("PerformanceResult"));
}

#[test]
fn test_stress_rng_debug() {
    // Coverage for StressRng Debug derive
    let rng = StressRng::new(42);
    let debug_str = format!("{:?}", rng);
    assert!(debug_str.contains("StressRng"));
}

#[test]
fn test_stress_config_debug() {
    // Coverage for StressConfig Debug derive
    let config = StressConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("StressConfig"));
}

#[test]
fn test_anomaly_kind_debug() {
    // Coverage for AnomalyKind Debug/Copy derives
    let kind = AnomalyKind::SlowFrame;
    let copied = kind; // Test Copy
    let debug_str = format!("{:?}", copied);
    assert!(debug_str.contains("SlowFrame"));
}
