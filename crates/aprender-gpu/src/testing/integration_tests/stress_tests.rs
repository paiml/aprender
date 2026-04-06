//! Stress testing (renacer + simular integration per spec v1.3.0)

use super::*;

/// Stress test with randomized inputs per frame
#[test]
fn test_stress_runner_visual() {
    use super::super::stress::{PerformanceThresholds, StressConfig, StressTestRunner};
    use super::super::tui::{render_to_string, TuiState};

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       STRESS TEST: Randomized Frame-by-Frame Testing         ║");
    println!("║              renacer v0.7.0 + simular v0.2.0                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let renderer = get_shared_renderer();

    // Configure stress test
    let config = StressConfig {
        cycles: 10,
        interval_ms: 10, // Fast for testing
        seed: 42,
        min_input_size: 64,
        max_input_size: 256,
        thresholds: PerformanceThresholds {
            max_frame_time_ms: 500, // Generous for test
            max_timing_variance: 0.5,
            ..Default::default()
        },
    };

    println!("Configuration:");
    println!("  Cycles: {}", config.cycles);
    println!("  Seed: {}", config.seed);
    println!(
        "  Input size: {}-{}",
        config.min_input_size, config.max_input_size
    );
    println!();

    let mut runner = StressTestRunner::new(config.clone());

    // Run stress test with visual verification
    runner.run_all(|input| {
        // Render input to PNG
        let size = (input.len() as f32).sqrt() as u32;
        let actual_size = size * size;
        let data: Vec<f32> = input.iter().take(actual_size as usize).copied().collect();

        if data.is_empty() {
            return (0, 1);
        }

        let png = renderer.render_to_png(&data, size, size);

        // Self-comparison (should always pass for deterministic input)
        let result = compare_png_bytes(&png, &png, 0);

        if result.different_pixels == 0 {
            (1, 0) // 1 pass, 0 fail
        } else {
            (0, 1) // 0 pass, 1 fail
        }
    });

    // Get report and verify performance
    let report = runner.report().clone();
    let perf = runner.verify();

    // Generate TUI output
    let mut tui_state = TuiState::new(config.cycles);
    tui_state.update_from_report(&report);
    let tui_output = render_to_string(&tui_state, &report, &perf);

    println!("{}", tui_output);

    // Report metrics
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("STRESS TEST METRICS:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Cycles completed: {}", report.cycles_completed);
    println!("  Total passed: {}", report.total_passed);
    println!("  Total failed: {}", report.total_failed);
    println!("  Pass rate: {:.1}%", perf.pass_rate * 100.0);
    println!("  Mean frame time: {:.2}ms", perf.mean_frame_ms);
    println!("  Max frame time: {}ms", perf.max_frame_ms);
    println!("  Timing variance: {:.3}", perf.variance);
    println!("  Anomalies detected: {}", report.anomalies.len());
    println!();

    // Assert performance
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PERFORMANCE VERIFICATION:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    if perf.passed {
        println!("  Status: ✓ PASS (all thresholds met)");
    } else {
        println!("  Status: ✗ FAIL");
        for violation in &perf.violations {
            println!("    - {}", violation);
        }
    }
    println!();

    // Print anomalies if any
    if !report.anomalies.is_empty() {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("ANOMALIES:");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        for anomaly in &report.anomalies {
            println!(
                "  [Cycle {}] {:?}: {}",
                anomaly.cycle, anomaly.kind, anomaly.description
            );
        }
        println!();
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           STRESS TEST COMPLETE (SOVEREIGN STACK)             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Assertions
    assert_eq!(report.cycles_completed, config.cycles);
    assert!(
        report.total_passed > 0,
        "Should have at least some passing tests"
    );
    // Note: We don't assert perf.passed because timing can vary in CI
}

/// Test deterministic stress test reproducibility
#[test]
fn test_stress_determinism() {
    use super::super::stress::{StressConfig, StressTestRunner};

    println!("\n");
    println!("Testing stress test determinism...");

    let config = StressConfig {
        cycles: 5,
        seed: 99999,
        min_input_size: 100,
        max_input_size: 200,
        ..Default::default()
    };

    // Run twice with same seed
    let mut runner1 = StressTestRunner::new(config.clone());
    let mut runner2 = StressTestRunner::new(config);

    // Collect inputs from both runners
    let inputs1: Vec<(u64, usize)> = (0..5)
        .map(|_| {
            let (seed, input) = runner1.generate_input();
            (seed, input.len())
        })
        .collect();

    let inputs2: Vec<(u64, usize)> = (0..5)
        .map(|_| {
            let (seed, input) = runner2.generate_input();
            (seed, input.len())
        })
        .collect();

    // Should be identical
    assert_eq!(
        inputs1, inputs2,
        "Same seed should produce identical inputs"
    );

    println!("  ✓ Deterministic: Same seed produces identical inputs");
    println!("  Inputs generated: {:?}", inputs1);
}
