//! Core tests for the stress testing framework.

use std::time::Duration;

use super::*;

#[test]
fn test_stress_rng_deterministic() {
    let mut rng1 = StressRng::new(42);
    let mut rng2 = StressRng::new(42);

    for _ in 0..100 {
        assert_eq!(rng1.next_u32(), rng2.next_u32());
    }
}

#[test]
fn test_stress_rng_gen_range() {
    let mut rng = StressRng::new(12345);
    for _ in 0..1000 {
        let val = rng.gen_range_u32(10, 100);
        assert!(val >= 10 && val < 100);
    }
}

#[test]
fn test_stress_rng_gen_f32() {
    let mut rng = StressRng::new(99999);
    for _ in 0..1000 {
        let val = rng.gen_f32();
        assert!((0.0..1.0).contains(&val));
    }
}

#[test]
fn test_frame_profile_default() {
    let profile = FrameProfile::default();
    assert_eq!(profile.cycle, 0);
    assert_eq!(profile.duration_ms, 0);
}

#[test]
fn test_stress_report_metrics() {
    let mut report = StressReport::default();

    report.add_frame(FrameProfile {
        cycle: 0,
        duration_ms: 100,
        tests_passed: 5,
        tests_failed: 0,
        ..Default::default()
    });

    report.add_frame(FrameProfile {
        cycle: 1,
        duration_ms: 120,
        tests_passed: 5,
        tests_failed: 0,
        ..Default::default()
    });

    assert_eq!(report.cycles_completed, 2);
    assert_eq!(report.total_passed, 10);
    assert_eq!(report.total_failed, 0);
    assert_eq!(report.mean_frame_time_ms(), 110.0);
    assert_eq!(report.max_frame_time_ms(), 120);
    assert!((report.pass_rate() - 1.0).abs() < 0.001);
}

#[test]
fn test_stress_report_variance() {
    let mut report = StressReport::default();

    // Add frames with same duration - variance should be 0
    for i in 0..10 {
        report.add_frame(FrameProfile {
            cycle: i,
            duration_ms: 100,
            tests_passed: 1,
            tests_failed: 0,
            ..Default::default()
        });
    }

    assert!((report.timing_variance()).abs() < 0.001);
}

#[test]
fn test_performance_thresholds_default() {
    let thresholds = PerformanceThresholds::default();
    assert_eq!(thresholds.max_frame_time_ms, 100);
    assert_eq!(thresholds.max_memory_bytes, 64 * 1024 * 1024);
    assert!((thresholds.max_timing_variance - 0.2).abs() < 0.001);
}

#[test]
fn test_verify_performance_pass() {
    let mut report = StressReport::default();
    for i in 0..10 {
        report.add_frame(FrameProfile {
            cycle: i,
            duration_ms: 50,
            tests_passed: 5,
            tests_failed: 0,
            ..Default::default()
        });
    }

    let result = verify_performance(&report, &PerformanceThresholds::default());
    assert!(result.passed);
    assert!(result.violations.is_empty());
}

#[test]
fn test_verify_performance_fail_slow() {
    let mut report = StressReport::default();
    report.add_frame(FrameProfile {
        cycle: 0,
        duration_ms: 200, // Exceeds 100ms threshold
        tests_passed: 5,
        tests_failed: 0,
        ..Default::default()
    });

    let result = verify_performance(&report, &PerformanceThresholds::default());
    assert!(!result.passed);
    assert_eq!(result.violations.len(), 1);
    assert!(result.violations[0].contains("Max frame time"));
}

#[test]
fn test_stress_runner_generate_input() {
    let config =
        StressConfig { min_input_size: 100, max_input_size: 200, seed: 42, ..Default::default() };

    let mut runner = StressTestRunner::new(config);
    let (seed1, input1) = runner.generate_input();
    let (seed2, input2) = runner.generate_input();

    // Different inputs each time
    assert_ne!(seed1, seed2);
    assert!(input1.len() >= 100 && input1.len() < 200);
    assert!(input2.len() >= 100 && input2.len() < 200);
}

#[test]
fn test_stress_runner_run_cycle() {
    let config = StressConfig { cycles: 1, seed: 42, ..Default::default() };

    let mut runner = StressTestRunner::new(config);
    let profile = runner.run_cycle(0, |input| {
        // Simple test: count positive values
        let positive = input.iter().filter(|&&v| v > 0.5).count() as u32;
        (positive, 0)
    });

    assert_eq!(profile.cycle, 0);
    assert!(profile.tests_passed > 0);
    assert_eq!(profile.tests_failed, 0);
}

#[test]
fn test_anomaly_detection() {
    let config = StressConfig {
        cycles: 1,
        seed: 42,
        thresholds: PerformanceThresholds {
            max_frame_time_ms: 1, // Very low threshold
            ..Default::default()
        },
        ..Default::default()
    };

    let mut runner = StressTestRunner::new(config);

    // This will likely exceed 1ms
    runner.run_cycle(0, |input| {
        std::thread::sleep(Duration::from_millis(5));
        (input.len() as u32, 0)
    });

    let report = runner.report();
    assert!(!report.anomalies.is_empty());
    assert_eq!(report.anomalies[0].kind, AnomalyKind::SlowFrame);
}

// ========================================================================
// EDGE CASE COVERAGE TESTS - TRUENO-SPEC-014
// ========================================================================

#[test]
fn test_stress_report_empty_frames() {
    // Coverage for line 54: when frames is empty
    let report = StressReport::default();
    assert_eq!(report.mean_frame_time_ms(), 0.0);
    assert_eq!(report.max_frame_time_ms(), 0);
}

#[test]
fn test_stress_report_single_frame_variance() {
    // Coverage for timing_variance with < 2 frames
    let mut report = StressReport::default();
    report.add_frame(FrameProfile {
        cycle: 0,
        duration_ms: 100,
        tests_passed: 1,
        tests_failed: 0,
        ..Default::default()
    });
    assert_eq!(report.timing_variance(), 0.0);
}

#[test]
fn test_stress_report_zero_mean_variance() {
    // Coverage for line 68: when mean is 0.0
    let mut report = StressReport::default();
    report.add_frame(FrameProfile {
        cycle: 0,
        duration_ms: 0, // Zero duration
        tests_passed: 1,
        tests_failed: 0,
        ..Default::default()
    });
    report.add_frame(FrameProfile {
        cycle: 1,
        duration_ms: 0, // Zero duration
        tests_passed: 1,
        tests_failed: 0,
        ..Default::default()
    });
    assert_eq!(report.timing_variance(), 0.0);
}

#[test]
fn test_stress_report_pass_rate_no_tests() {
    // Coverage for line 90: when total is 0
    let mut report = StressReport::default();
    report.add_frame(FrameProfile {
        cycle: 0,
        duration_ms: 100,
        tests_passed: 0, // No tests
        tests_failed: 0, // No tests
        ..Default::default()
    });
    assert_eq!(report.pass_rate(), 1.0);
}

// ========================================================================
// ADDITIONAL COVERAGE TESTS - TARGETING 95%+ COVERAGE
// ========================================================================

#[test]
fn test_stress_rng_next_u64() {
    // Coverage for next_u64 (lines 253-257)
    let mut rng = StressRng::new(42);
    let val1 = rng.next_u64();
    let val2 = rng.next_u64();

    // Should produce different values
    assert_ne!(val1, val2);

    // Deterministic check
    let mut rng2 = StressRng::new(42);
    assert_eq!(val1, rng2.next_u64());
}

#[test]
fn test_stress_rng_gen_range_edge_cases() {
    // Coverage for gen_range_u32 edge case when max <= min (line 267)
    let mut rng = StressRng::new(42);

    // When max == min, should return min
    assert_eq!(rng.gen_range_u32(50, 50), 50);

    // When max < min, should return min
    assert_eq!(rng.gen_range_u32(100, 50), 100);
}

#[test]
fn test_test_failure_anomaly_detection() {
    // Coverage for test failure anomaly (lines 370-376)
    let config = StressConfig { cycles: 1, seed: 42, ..Default::default() };

    let mut runner = StressTestRunner::new(config);

    // Run cycle that reports test failures
    runner.run_cycle(0, |_input| {
        (3, 2) // 3 passed, 2 failed
    });

    let report = runner.report();
    assert_eq!(report.total_passed, 3);
    assert_eq!(report.total_failed, 2);

    // Should have a TestFailure anomaly
    let failure_anomalies: Vec<_> =
        report.anomalies.iter().filter(|a| a.kind == AnomalyKind::TestFailure).collect();
    assert_eq!(failure_anomalies.len(), 1);
    assert!(failure_anomalies[0].description.contains("2 tests failed"));
}

#[test]
fn test_verify_performance_fail_variance() {
    // Coverage for timing variance violation (lines 194-199)
    let mut report = StressReport::default();

    // Add frames with high variance in timing
    report.add_frame(FrameProfile {
        cycle: 0,
        duration_ms: 10,
        tests_passed: 1,
        tests_failed: 0,
        ..Default::default()
    });
    report.add_frame(FrameProfile {
        cycle: 1,
        duration_ms: 100, // 10x difference - high variance
        tests_passed: 1,
        tests_failed: 0,
        ..Default::default()
    });

    let thresholds = PerformanceThresholds {
        max_frame_time_ms: 200,    // Pass frame time check
        max_timing_variance: 0.01, // Very strict - will fail
        max_failure_rate: 0.5,     // Pass failure rate
        ..Default::default()
    };

    let result = verify_performance(&report, &thresholds);
    assert!(!result.passed);
    assert!(result.violations.iter().any(|v| v.contains("variance")));
}

#[test]
fn test_verify_performance_fail_pass_rate() {
    // Coverage for pass rate violation (lines 201-207)
    let mut report = StressReport::default();

    // Add frames with high failure rate
    report.add_frame(FrameProfile {
        cycle: 0,
        duration_ms: 50,
        tests_passed: 1,
        tests_failed: 9, // 90% failure rate
        ..Default::default()
    });

    let thresholds = PerformanceThresholds {
        max_frame_time_ms: 200,
        max_timing_variance: 1.0,
        max_failure_rate: 0.05, // 5% max - will fail
        ..Default::default()
    };

    let result = verify_performance(&report, &thresholds);
    assert!(!result.passed);
    assert!(result.violations.iter().any(|v| v.contains("Pass rate")));
}

#[test]
fn test_runner_verify_method() {
    // Coverage for StressTestRunner::verify (lines 409-412)
    let config = StressConfig {
        cycles: 1,
        seed: 42,
        thresholds: PerformanceThresholds::default(),
        ..Default::default()
    };

    let mut runner = StressTestRunner::new(config);
    runner.run_cycle(0, |_input| (5, 0));

    let result = runner.verify();
    assert!(result.passed);
    assert_eq!(result.pass_rate, 1.0);
}

#[test]
fn test_run_all_cycles() {
    // Coverage for run_all (lines 383-400)
    let config = StressConfig {
        cycles: 3,
        interval_ms: 1, // Very short interval for fast test
        seed: 42,
        min_input_size: 10,
        max_input_size: 20,
        thresholds: PerformanceThresholds::default(),
    };

    let mut runner = StressTestRunner::new(config);
    let report = runner.run_all(|input| {
        // Simple test function
        let sum: f32 = input.iter().sum();
        if sum > 0.0 {
            (1, 0)
        } else {
            (0, 1)
        }
    });

    assert_eq!(report.cycles_completed, 3);
    assert_eq!(report.frames.len(), 3);
    assert!(report.total_passed >= 1);
}

#[test]
fn test_run_all_with_slow_test() {
    // Coverage for run_all when test takes longer than interval
    // (lines 394-396 - the checked_sub branch returning None)
    let config = StressConfig {
        cycles: 2,
        interval_ms: 1, // 1ms interval
        seed: 42,
        min_input_size: 10,
        max_input_size: 20,
        thresholds: PerformanceThresholds {
            max_frame_time_ms: 1000, // High threshold to avoid anomaly
            ..Default::default()
        },
    };

    let mut runner = StressTestRunner::new(config);
    let report = runner.run_all(|_input| {
        // Sleep longer than interval to trigger the checked_sub branch
        std::thread::sleep(Duration::from_millis(5));
        (1, 0)
    });

    assert_eq!(report.cycles_completed, 2);
}

#[test]
fn test_anomaly_kinds_equality() {
    // Coverage for AnomalyKind PartialEq/Eq
    assert_eq!(AnomalyKind::SlowFrame, AnomalyKind::SlowFrame);
    assert_ne!(AnomalyKind::SlowFrame, AnomalyKind::HighMemory);
    assert_ne!(AnomalyKind::TestFailure, AnomalyKind::TimingSpike);
    assert_ne!(AnomalyKind::NonDeterministic, AnomalyKind::HighMemory);
}

#[test]
fn test_stress_config_default() {
    // Coverage for StressConfig default values
    let config = StressConfig::default();
    assert_eq!(config.cycles, 100);
    assert_eq!(config.interval_ms, 100);
    assert_eq!(config.seed, 42);
    assert_eq!(config.min_input_size, 64);
    assert_eq!(config.max_input_size, 512);
}
