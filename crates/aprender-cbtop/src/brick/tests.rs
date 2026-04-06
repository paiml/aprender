//! Tests for brick module types, scoring, and profiling.

use super::*;

#[test]
fn test_brick_budget_uniform() {
    let budget = BrickBudget::uniform(16);
    assert_eq!(budget.collect_ms, 16);
    assert_eq!(budget.layout_ms, 16);
    assert_eq!(budget.render_ms, 16);
    assert_eq!(budget.total_ms(), 48);
}

#[test]
fn test_brick_budget_60fps() {
    let budget = BrickBudget::FRAME_60FPS;
    assert_eq!(budget.total_ms(), 16);
}

#[test]
fn test_brick_verification_new() {
    let v = BrickVerification::new();
    assert!(v.is_valid());
    assert_eq!(v.score(), 1.0);
}

#[test]
fn test_brick_verification_pass_fail() {
    let mut v = BrickVerification::new();
    v.add_pass(BrickAssertion::MinWidth(10));
    v.add_pass(BrickAssertion::MinHeight(5));
    v.add_fail(BrickAssertion::MaxRenderTimeMs(16), "took 20ms");

    assert!(!v.is_valid());
    assert_eq!(v.passed.len(), 2);
    assert_eq!(v.failed.len(), 1);
    assert!((v.score() - 0.666).abs() < 0.01);
}

#[test]
fn test_constraints_constrain() {
    let constraints = Constraints::new(10.0, 100.0, 5.0, 50.0);

    // Within bounds
    let size = constraints.constrain(Size::new(50.0, 25.0));
    assert_eq!(size.width, 50.0);
    assert_eq!(size.height, 25.0);

    // Below minimum
    let size = constraints.constrain(Size::new(5.0, 2.0));
    assert_eq!(size.width, 10.0);
    assert_eq!(size.height, 5.0);

    // Above maximum
    let size = constraints.constrain(Size::new(200.0, 100.0));
    assert_eq!(size.width, 100.0);
    assert_eq!(size.height, 50.0);
}

#[test]
fn test_color_constants() {
    assert_eq!(Color::BLACK.r, 0);
    assert_eq!(Color::WHITE.r, 255);
    assert_eq!(Color::ANDON_GREEN.g, 200);
}

// ========================================================================
// BrickScore Tests (F501-F505 Falsification Criteria)
// ========================================================================

/// F501: Performance score accurate
#[test]
fn f501_performance_score_accurate() {
    // 100% of theoretical = 40 points
    assert_eq!(BrickScore::score_performance(100.0, 100.0), 40);

    // 50% of theoretical = 20 points
    assert_eq!(BrickScore::score_performance(50.0, 100.0), 20);

    // 25% of theoretical = 10 points
    assert_eq!(BrickScore::score_performance(25.0, 100.0), 10);

    // Above theoretical caps at 40
    assert_eq!(BrickScore::score_performance(200.0, 100.0), 40);

    // Zero theoretical = 0 points
    assert_eq!(BrickScore::score_performance(100.0, 0.0), 0);
}

/// F502: Efficiency score reflects backend (via speedup scoring)
#[test]
fn f502_efficiency_reflects_backend() {
    // 1x speedup = 0 points (no improvement)
    assert_eq!(BrickScore::score_speedup(1.0), 0);

    // 2x speedup = 5 points (log2(2) * 5 = 5)
    assert_eq!(BrickScore::score_speedup(2.0), 5);

    // 4x speedup = 10 points (log2(4) * 5 = 10)
    assert_eq!(BrickScore::score_speedup(4.0), 10);

    // 8x speedup = 15 points (log2(8) * 5 = 15)
    assert_eq!(BrickScore::score_speedup(8.0), 15);

    // 16x speedup = 20 points (capped)
    assert_eq!(BrickScore::score_speedup(16.0), 20);

    // >16x speedup still caps at 20
    assert_eq!(BrickScore::score_speedup(64.0), 20);
}

/// F503: Correctness detects failures (via grade system)
#[test]
fn f503_correctness_detects_failures() {
    // Perfect score = Grade A
    let perfect = BrickScore::perfect();
    assert_eq!(perfect.grade(), BrickGrade::A);
    assert_eq!(perfect.total(), 100);

    // Zero correctness drops grade significantly
    let no_correctness = BrickScore::new(40, 25, 0, 15);
    assert_eq!(no_correctness.total(), 80);
    assert_eq!(no_correctness.grade(), BrickGrade::B);

    // Zero score = Grade F
    let zero = BrickScore::zero();
    assert_eq!(zero.grade(), BrickGrade::F);
    assert_eq!(zero.total(), 0);
}

/// F504: Stability detects variance (CV scoring)
#[test]
fn f504_stability_detects_variance() {
    // CV < 5% = 8 points (excellent stability)
    assert_eq!(BrickScore::score_cv(4.9), 8);
    assert_eq!(BrickScore::score_cv(0.0), 8);

    // 5% <= CV < 10% = 4 points (acceptable stability)
    assert_eq!(BrickScore::score_cv(5.0), 4);
    assert_eq!(BrickScore::score_cv(9.9), 4);

    // CV >= 10% = 0 points (poor stability)
    assert_eq!(BrickScore::score_cv(10.0), 0);
    assert_eq!(BrickScore::score_cv(50.0), 0);
}

/// F505: Total score is sum of components
#[test]
fn f505_total_is_sum_of_components() {
    let score = BrickScore::new(38, 22, 20, 14);
    assert_eq!(
        score.total(),
        score.performance + score.efficiency + score.correctness + score.stability
    );
    assert_eq!(score.total(), 38 + 22 + 20 + 14);
    assert_eq!(score.total(), 94);

    // Verify clamping at max values
    let over_max = BrickScore::new(50, 30, 25, 20);
    assert_eq!(over_max.performance, 40);
    assert_eq!(over_max.efficiency, 25);
    assert_eq!(over_max.correctness, 20);
    assert_eq!(over_max.stability, 15);
    assert_eq!(over_max.total(), 100);
}

#[test]
fn test_brick_grade_ordering() {
    assert!(BrickGrade::A > BrickGrade::B);
    assert!(BrickGrade::B > BrickGrade::C);
    assert!(BrickGrade::C > BrickGrade::D);
    assert!(BrickGrade::D > BrickGrade::F);
}

#[test]
fn test_brick_grade_colors() {
    assert_eq!(BrickGrade::A.color(), Color::ANDON_GREEN);
    assert_eq!(BrickGrade::B.color(), Color::ANDON_GREEN);
    assert_eq!(BrickGrade::C.color(), Color::ANDON_YELLOW);
    assert_eq!(BrickGrade::D.color(), Color::ANDON_RED);
    assert_eq!(BrickGrade::F.color(), Color::ANDON_RED);
}

#[test]
fn test_brick_score_percentages() {
    let score = BrickScore::new(20, 12, 10, 7);
    assert!((score.performance_pct() - 0.5).abs() < 0.01);
    assert!((score.efficiency_pct() - 0.48).abs() < 0.01);
    assert!((score.correctness_pct() - 0.5).abs() < 0.01);
    assert!((score.stability_pct() - 0.4666).abs() < 0.01);
}

#[test]
fn test_render_bar() {
    let bar = BrickScore::render_bar(20, 40, 10);
    assert_eq!(bar.chars().filter(|c| *c == '█').count(), 5);
    assert_eq!(bar.chars().filter(|c| *c == '░').count(), 5);

    let full = BrickScore::render_bar(40, 40, 10);
    assert_eq!(full.chars().filter(|c| *c == '█').count(), 10);

    let empty = BrickScore::render_bar(0, 40, 10);
    assert_eq!(empty.chars().filter(|c| *c == '░').count(), 10);
}

#[test]
fn test_brick_score_display() {
    let score = BrickScore::new(38, 22, 20, 14);
    let display = format!("{}", score);
    assert!(display.contains("94/100"));
    assert!(display.contains("Excellent"));
}

#[test]
fn test_kernel_trace_checksum() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let trace = KernelTrace::new("test_kernel", 0, 0, "CPU")
        .with_input_checksum(&data)
        .with_output_checksum(&data);
    assert_eq!(trace.input_checksum, trace.output_checksum);
    assert_ne!(trace.input_checksum, 0);
}

#[test]
fn test_divergence_report_matched() {
    let report = DivergenceReport::matched(10);
    assert!(report.matched);
    assert_eq!(report.kernels_compared, 10);
    assert!(report.first_divergent_kernel.is_none());
}

#[test]
fn test_divergence_report_diverged() {
    let cpu_trace = KernelTrace::new("rope_neox", 0, 1, "CPU")
        .with_input_checksum(&[1.0, 2.0, 3.0])
        .with_output_checksum(&[4.0, 5.0, 6.0]);
    let gpu_trace = KernelTrace::new("rope_neox", 0, 1, "CUDA")
        .with_input_checksum(&[1.0, 2.0, 3.0])
        .with_output_checksum(&[7.0, 8.0, 9.0]); // Different output!

    let report = DivergenceReport::diverged(cpu_trace, gpu_trace, 5);
    assert!(!report.matched);
    assert_eq!(report.kernels_compared, 5);
    assert!(report.first_divergent_kernel.is_some());
    assert!(report.diagnosis.contains("DIVERGENCE"));
}

#[test]
fn test_brick_profiler_basic() {
    let mut profiler = BrickProfiler::new("test_run");

    let trace = KernelTrace::new("matmul", 0, 0, "CPU")
        .with_input_checksum(&[1.0, 2.0])
        .with_output_checksum(&[3.0, 4.0]);
    profiler.add_trace(trace);

    assert_eq!(profiler.traces.len(), 1);
    assert!(!profiler.is_diverged());
}

#[test]
fn test_brick_profiler_detect_divergence() {
    let mut cpu_profiler = BrickProfiler::new("cpu_run");
    let mut gpu_profiler = BrickProfiler::new("gpu_run");

    // Same inputs, same outputs = match
    cpu_profiler.add_trace(
        KernelTrace::new("rope", 0, 1, "CPU")
            .with_input_checksum(&[1.0, 2.0])
            .with_output_checksum(&[3.0, 4.0]),
    );
    gpu_profiler.add_trace(
        KernelTrace::new("rope", 0, 1, "CUDA")
            .with_input_checksum(&[1.0, 2.0])
            .with_output_checksum(&[3.0, 4.0]),
    );

    let report = cpu_profiler.compare(&gpu_profiler);
    assert!(report.matched);

    // Add divergent kernel
    cpu_profiler
        .add_trace(KernelTrace::new("rmsnorm", 1, 1, "CPU").with_output_checksum(&[5.0, 6.0]));
    gpu_profiler.add_trace(
        KernelTrace::new("rmsnorm", 1, 1, "CUDA").with_output_checksum(&[7.0, 8.0]), // Different!
    );

    let report = cpu_profiler.compare(&gpu_profiler);
    assert!(!report.matched);
    assert!(report.diagnosis.contains("rmsnorm"));
}
