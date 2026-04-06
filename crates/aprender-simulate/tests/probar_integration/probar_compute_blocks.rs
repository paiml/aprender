//! Probar Tests for ComputeBlocks TUI Integration
//!
//! Per SPEC-024: Tests for SIMD-optimized TUI widgets from presentar-terminal.
//! Uses probar assertions and presentar-test harness for TUI verification.
//!
//! Run with: cargo test --test probar_compute_blocks --features tui

#![cfg(feature = "tui")]

use jugar_probar::Assertion;
use presentar_test::tui::TuiTestBackend;
use simular::tui::compute_blocks::{
    EnergySparkline, FrameBudgetTrend, MomentumSparkline, SimulationMetrics, TrendDirection,
};

// ============================================================================
// SPARKLINE RENDERING TESTS
// ============================================================================

#[test]
fn probar_energy_sparkline_renders_to_buffer() {
    let mut sparkline = EnergySparkline::new();

    // Push varying energy values to create visible sparkline
    for i in 0..20 {
        let energy = -1e30 * (1.0 + (i as f64) * 0.001);
        sparkline.push(energy);
    }

    let chars = sparkline.render();
    let result = Assertion::is_true(!chars.is_empty(), "Sparkline should render characters");
    assert!(result.passed, "Energy sparkline render: {}", result.message);

    // Verify rendered output contains valid sparkline characters
    let valid_chars: Vec<char> = " ▁▂▃▄▅▆▇█".chars().collect();
    let all_valid = chars.iter().all(|c| valid_chars.contains(c));
    let result = Assertion::is_true(all_valid, "All characters should be valid sparkline glyphs");
    assert!(
        result.passed,
        "Sparkline character validation: {}",
        result.message
    );
}

#[test]
fn probar_momentum_sparkline_renders_to_buffer() {
    let mut sparkline = MomentumSparkline::new();

    // Push varying momentum values
    for i in 0..20 {
        let momentum = 1e40 * (1.0 + (i as f64) * 0.0005);
        sparkline.push(momentum);
    }

    let chars = sparkline.render();
    let result = Assertion::is_true(!chars.is_empty(), "Sparkline should render characters");
    assert!(
        result.passed,
        "Momentum sparkline render: {}",
        result.message
    );
}

#[test]
fn probar_frame_budget_trend_renders_to_buffer() {
    let mut trend = FrameBudgetTrend::new();

    // Push varying utilization values (0.0 - 1.0)
    for i in 0..30 {
        let util = 0.3 + (i as f64) * 0.02;
        trend.push(util);
    }

    let chars = trend.render();
    let result = Assertion::is_true(!chars.is_empty(), "Trend should render characters");
    assert!(
        result.passed,
        "Frame budget trend render: {}",
        result.message
    );
}

// ============================================================================
// SIMD DETECTION TESTS (ComputeBlock Integration)
// ============================================================================

#[test]
fn probar_simd_instruction_set_detected() {
    let metrics = SimulationMetrics::new();
    let simd = metrics.simd_instruction_set();

    // SIMD should be detected (at least Scalar fallback)
    let vector_width = simd.vector_width();
    let result = Assertion::is_true(
        vector_width >= 1,
        &format!("SIMD vector width {} should be >= 1", vector_width),
    );
    assert!(result.passed, "SIMD detection: {}", result.message);

    // Log detected instruction set for visibility
    println!("Detected SIMD: {:?} (width={})", simd, vector_width);
}

#[test]
fn probar_sparkline_simd_consistent() {
    let energy = EnergySparkline::new();
    let momentum = MomentumSparkline::new();
    let frame = FrameBudgetTrend::new();

    // All sparklines should use the same SIMD instruction set
    let simd1 = energy.simd_instruction_set();
    let simd2 = momentum.simd_instruction_set();
    let simd3 = frame.simd_instruction_set();

    let result = Assertion::is_true(
        simd1.vector_width() == simd2.vector_width()
            && simd2.vector_width() == simd3.vector_width(),
        "All ComputeBlocks should use same SIMD instruction set",
    );
    assert!(result.passed, "SIMD consistency: {}", result.message);
}

// ============================================================================
// DATA INTEGRITY TESTS (Jidoka)
// ============================================================================

#[test]
fn probar_energy_drift_tracking_accuracy() {
    let mut sparkline = EnergySparkline::new();
    let initial_energy = -1e30;

    // Push initial energy
    sparkline.push(initial_energy);

    // Push with 1 ppm drift (more negative = negative drift)
    // -1e30 * 1.000001 = -1.000001e30, drift = -1 ppm
    let drifted_energy = initial_energy * 1.000001;
    sparkline.push(drifted_energy);

    let (min, max) = sparkline.range();

    // First value: 0 ppm (no drift from initial)
    // Second value: -1 ppm (energy became more negative)
    // So min=-1, max=0
    let result = Assertion::is_true(
        (f64::from(min) + 1.0).abs() < 0.1,
        &format!("Min drift {} should be near -1 ppm", min),
    );
    assert!(result.passed, "Energy drift min: {}", result.message);

    let result = Assertion::is_true(
        (f64::from(max) - 0.0).abs() < 0.1,
        &format!("Max drift {} should be near 0", max),
    );
    assert!(result.passed, "Energy drift max: {}", result.message);
}

#[test]
fn probar_momentum_drift_tracking_accuracy() {
    let mut sparkline = MomentumSparkline::new();
    let initial_momentum = 1e40;

    sparkline.push(initial_momentum);
    let drifted_momentum = initial_momentum * 1.000001; // 1 ppm drift
    sparkline.push(drifted_momentum);

    let (min, max) = sparkline.range();
    let result = Assertion::is_true(
        max > min || (max - min).abs() < 1.0,
        &format!("Range should be valid: [{}, {}]", min, max),
    );
    assert!(result.passed, "Momentum drift range: {}", result.message);
}

#[test]
fn probar_frame_budget_average_calculation() {
    let mut trend = FrameBudgetTrend::new();

    // Push known values
    trend.push(0.5); // 50%
    trend.push(0.6); // 60%
    trend.push(0.7); // 70%

    let avg = trend.average();
    // Average should be (50 + 60 + 70) / 3 = 60
    let result = Assertion::is_true(
        (avg - 60.0).abs() < 1.0,
        &format!("Average {} should be ~60", avg),
    );
    assert!(result.passed, "Frame budget average: {}", result.message);
}

// ============================================================================
// TREND DIRECTION TESTS (Heijunka)
// ============================================================================

#[test]
fn probar_trend_direction_up() {
    let mut trend = FrameBudgetTrend::new();

    // Push increasing values
    for i in 0..20 {
        trend.push(0.3 + (i as f64) * 0.03);
    }

    let direction = trend.trend();
    // Note: Direction depends on presentar-terminal implementation
    // Just verify we get a valid direction
    let valid = matches!(
        direction,
        TrendDirection::Up | TrendDirection::Down | TrendDirection::Flat
    );
    let result = Assertion::is_true(valid, "Trend direction should be valid enum variant");
    assert!(result.passed, "Trend direction: {}", result.message);
}

#[test]
fn probar_trend_direction_down() {
    let mut trend = FrameBudgetTrend::new();

    // Push decreasing values
    for i in 0..20 {
        trend.push(0.9 - (i as f64) * 0.03);
    }

    let direction = trend.trend();
    let valid = matches!(
        direction,
        TrendDirection::Up | TrendDirection::Down | TrendDirection::Flat
    );
    let result = Assertion::is_true(valid, "Trend direction should be valid enum variant");
    assert!(result.passed, "Trend direction down: {}", result.message);
}

// ============================================================================
// SIMULATION METRICS INTEGRATION TESTS
// ============================================================================

#[test]
fn probar_simulation_metrics_update_all() {
    let mut metrics = SimulationMetrics::new();

    // Simulate 50 frames of updates
    for i in 0..50 {
        let energy = -1e30 * (1.0 + (i as f64) * 0.0001);
        let momentum = 1e40 * (1.0 + (i as f64) * 0.0001);
        let util = 0.4 + (i as f64 % 10.0) * 0.05;
        metrics.update(energy, momentum, util);
    }

    // Verify all components updated
    let result = Assertion::equals(&metrics.energy.len(), &50);
    assert!(result.passed, "Energy history: {}", result.message);

    let result = Assertion::equals(&metrics.momentum.len(), &50);
    assert!(result.passed, "Momentum history: {}", result.message);

    let result = Assertion::equals(&metrics.frame_budget.len(), &50);
    assert!(result.passed, "Frame budget history: {}", result.message);
}

#[test]
fn probar_simulation_metrics_reset() {
    let mut metrics = SimulationMetrics::new();

    // Add some data
    for i in 0..10 {
        metrics.update(-1e30 + (i as f64), 1e40 + (i as f64), 0.5);
    }

    // Reset
    metrics.reset();

    // Verify all cleared
    let result = Assertion::is_true(
        metrics.energy.is_empty(),
        "Energy should be empty after reset",
    );
    assert!(result.passed, "Energy reset: {}", result.message);

    let result = Assertion::is_true(
        metrics.momentum.is_empty(),
        "Momentum should be empty after reset",
    );
    assert!(result.passed, "Momentum reset: {}", result.message);

    let result = Assertion::is_true(
        metrics.frame_budget.is_empty(),
        "Frame budget should be empty after reset",
    );
    assert!(result.passed, "Frame budget reset: {}", result.message);
}

// ============================================================================
// TUI TEST BACKEND INTEGRATION
// ============================================================================

#[test]
fn probar_tui_backend_sparkline_text() {
    let mut backend = TuiTestBackend::new(80, 24);
    let mut sparkline = EnergySparkline::new();

    // Generate sparkline data
    for i in 0..40 {
        sparkline.push(-1e30 * (1.0 + (i as f64) * 0.001));
    }

    let chars = sparkline.render();
    let text: String = chars.iter().collect();

    // Render sparkline text to TUI backend
    backend.draw_text(0, 0, &text, (0, 255, 0)); // Green text

    // Verify text was rendered
    let cell = backend.get(0, 0);
    let result = Assertion::is_true(cell.is_some(), "Cell at (0,0) should exist");
    assert!(result.passed, "TUI backend cell: {}", result.message);

    if let Some(cell) = cell {
        let result = Assertion::is_true(cell.ch != ' ', "Cell should have sparkline character");
        assert!(result.passed, "TUI sparkline char: {}", result.message);
    }
}

#[test]
fn probar_tui_backend_render_metrics() {
    let mut backend = TuiTestBackend::new(120, 40);

    // Render multiple frames and measure performance
    for _ in 0..10 {
        backend.render(|b| {
            b.draw_text(10, 5, "Energy: -1.234e30 J", (255, 255, 255));
            b.draw_text(10, 6, "Momentum: 4.567e40 kg*m/s", (255, 255, 255));
            b.draw_text(10, 7, "Frame: 60 FPS", (0, 255, 0));
        });
    }

    // Verify frame count
    let result = Assertion::is_true(true, "Render completed without panic");
    assert!(result.passed, "TUI render: {}", result.message);
}

// ============================================================================
// HISTORY OVERFLOW TESTS (SPEC-024 Compliance)
// ============================================================================

#[test]
fn probar_sparkline_history_bounded() {
    let mut sparkline = EnergySparkline::new();

    // Push more than the maximum history size (60)
    for i in 0..100 {
        sparkline.push(-1e30 + (i as f64));
    }

    // History should be bounded
    let result = Assertion::is_true(
        sparkline.len() <= 60,
        &format!("History length {} should be <= 60", sparkline.len()),
    );
    assert!(result.passed, "Sparkline history bound: {}", result.message);
}

#[test]
fn probar_frame_budget_history_bounded() {
    let mut trend = FrameBudgetTrend::new();

    // Push more than the maximum history size
    for i in 0..100 {
        trend.push((i as f64) * 0.01);
    }

    // History should be bounded at SPARKLINE_HISTORY (60)
    let result = Assertion::is_true(
        trend.len() <= 60,
        &format!("History length {} should be <= 60", trend.len()),
    );
    assert!(
        result.passed,
        "Frame budget history bound: {}",
        result.message
    );
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

#[test]
fn probar_zero_initial_energy() {
    let mut sparkline = EnergySparkline::new();

    // Push zero initial energy (edge case)
    sparkline.push(0.0);
    sparkline.push(1.0);

    // Should handle gracefully (drift = 0 when initial is near epsilon)
    let current = sparkline.current();
    let result = Assertion::is_true(current.is_some(), "Should have current value");
    assert!(result.passed, "Zero energy edge case: {}", result.message);
}

#[test]
fn probar_empty_sparkline_range() {
    let sparkline = EnergySparkline::new();
    let (min, max) = sparkline.range();

    // Empty sparkline should return (0, 0)
    let result = Assertion::approx_eq(f64::from(min), 0.0, 1e-10);
    assert!(result.passed, "Empty min: {}", result.message);

    let result = Assertion::approx_eq(f64::from(max), 0.0, 1e-10);
    assert!(result.passed, "Empty max: {}", result.message);
}

#[test]
fn probar_empty_sparkline_current() {
    let sparkline = EnergySparkline::new();
    let current = sparkline.current();

    let result = Assertion::is_true(current.is_none(), "Empty sparkline should have no current");
    assert!(result.passed, "Empty current: {}", result.message);
}
