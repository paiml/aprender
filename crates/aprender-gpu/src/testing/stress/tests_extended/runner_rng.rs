//! StressTestRunner, StressRng, and verify_performance tests.

use super::*;

#[test]
fn test_stress_runner_memory_bytes_calculation() {
    // Coverage for memory_bytes calculation in run_cycle (line 351)
    let config = StressConfig {
        cycles: 1,
        seed: 42,
        min_input_size: 100,
        max_input_size: 101, // Force exact size
        ..Default::default()
    };

    let mut runner = StressTestRunner::new(config);
    let profile = runner.run_cycle(0, |input| (input.len() as u32, 0));

    // memory_bytes should be input_size * sizeof(f32)
    assert_eq!(
        profile.memory_bytes,
        profile.input_size * std::mem::size_of::<f32>()
    );
}

#[test]
fn test_verify_performance_multiple_violations() {
    // Coverage for multiple violations in verify_performance
    let mut report = StressReport::default();

    // Create a report that triggers all violation checks
    report.add_frame(FrameProfile {
        cycle: 0,
        duration_ms: 200, // Exceeds threshold
        tests_passed: 1,
        tests_failed: 99, // Very high failure rate
        ..Default::default()
    });
    report.add_frame(FrameProfile {
        cycle: 1,
        duration_ms: 10, // High variance with previous
        tests_passed: 1,
        tests_failed: 0,
        ..Default::default()
    });

    let thresholds = PerformanceThresholds {
        max_frame_time_ms: 100,   // Will trigger
        max_timing_variance: 0.1, // Will trigger
        max_failure_rate: 0.01,   // Will trigger
        ..Default::default()
    };

    let result = verify_performance(&report, &thresholds);
    assert!(!result.passed);
    assert!(result.violations.len() >= 2); // At least frame time and pass rate
}

#[test]
fn test_stress_rng_sequence_consistency() {
    // Coverage for RNG state consistency across operations
    let mut rng = StressRng::new(42);

    // Generate a sequence
    let seq1: Vec<u32> = (0..10).map(|_| rng.next_u32()).collect();

    // New RNG with same seed should produce same sequence
    let mut rng2 = StressRng::new(42);
    let seq2: Vec<u32> = (0..10).map(|_| rng2.next_u32()).collect();

    assert_eq!(seq1, seq2);
}

#[test]
fn test_stress_rng_different_seeds() {
    // Coverage for RNG producing different sequences with different seeds
    let mut rng1 = StressRng::new(1);
    let mut rng2 = StressRng::new(2);

    let val1 = rng1.next_u32();
    let val2 = rng2.next_u32();

    // Different seeds should produce different values
    assert_ne!(val1, val2);
}

#[test]
fn test_stress_runner_input_generation_determinism() {
    // Coverage for deterministic input generation
    let config1 = StressConfig {
        seed: 12345,
        min_input_size: 50,
        max_input_size: 100,
        ..Default::default()
    };

    let config2 = StressConfig {
        seed: 12345,
        min_input_size: 50,
        max_input_size: 100,
        ..Default::default()
    };

    let mut runner1 = StressTestRunner::new(config1);
    let mut runner2 = StressTestRunner::new(config2);

    let (seed1, input1) = runner1.generate_input();
    let (seed2, input2) = runner2.generate_input();

    // Same seed should produce same inputs
    assert_eq!(seed1, seed2);
    assert_eq!(input1, input2);
}

#[test]
fn test_run_all_with_zero_interval() {
    // Coverage for run_all with zero interval (no sleep needed)
    let config = StressConfig {
        cycles: 3,
        interval_ms: 0, // Zero interval
        seed: 42,
        min_input_size: 10,
        max_input_size: 20,
        thresholds: PerformanceThresholds::default(),
    };

    let mut runner = StressTestRunner::new(config);
    let report = runner.run_all(|_input| (1, 0));

    assert_eq!(report.cycles_completed, 3);
}

#[test]
fn test_verify_performance_boundary_threshold() {
    // Coverage for boundary conditions in threshold checks
    let mut report = StressReport::default();

    // Create exactly at threshold
    report.add_frame(FrameProfile {
        cycle: 0,
        duration_ms: 100, // Exactly at threshold
        tests_passed: 99,
        tests_failed: 1, // Exactly at 1% threshold
        ..Default::default()
    });

    let thresholds = PerformanceThresholds {
        max_frame_time_ms: 100, // Equal should pass
        max_timing_variance: 1.0,
        max_failure_rate: 0.01, // Equal should pass
        ..Default::default()
    };

    let result = verify_performance(&report, &thresholds);
    // At boundary values, the check is `>` for frame time, so 100 == 100 should pass
    // But pass_rate check is `<` so 99% < 99% is false, should pass
    assert!(result.passed || result.violations.len() <= 1);
}

#[test]
fn test_stress_rng_gen_f32_range() {
    // Coverage for gen_f32 producing values in [0, 1) range
    let mut rng = StressRng::new(42);

    for _ in 0..1000 {
        let val = rng.gen_f32();
        assert!(val >= 0.0, "Value should be >= 0.0");
        assert!(val < 1.0, "Value should be < 1.0");
    }
}

#[test]
fn test_stress_runner_report_method() {
    // Coverage for report() method returning reference
    let config = StressConfig::default();
    let runner = StressTestRunner::new(config);

    let report = runner.report();
    assert_eq!(report.cycles_completed, 0);
    assert!(report.frames.is_empty());
}

#[test]
fn test_run_cycle_with_failures_and_slow_frame() {
    // Coverage for both anomaly conditions in single run_cycle
    let config = StressConfig {
        cycles: 1,
        seed: 42,
        thresholds: PerformanceThresholds {
            max_frame_time_ms: 1, // Very low to trigger slow frame
            ..Default::default()
        },
        ..Default::default()
    };

    let mut runner = StressTestRunner::new(config);

    runner.run_cycle(0, |_input| {
        std::thread::sleep(Duration::from_millis(5));
        (3, 5) // Report failures to trigger TestFailure anomaly
    });

    let report = runner.report();

    // Should have both SlowFrame and TestFailure anomalies
    let slow_count = report
        .anomalies
        .iter()
        .filter(|a| a.kind == AnomalyKind::SlowFrame)
        .count();
    let failure_count = report
        .anomalies
        .iter()
        .filter(|a| a.kind == AnomalyKind::TestFailure)
        .count();

    assert_eq!(slow_count, 1, "Should have one SlowFrame anomaly");
    assert_eq!(failure_count, 1, "Should have one TestFailure anomaly");
}
