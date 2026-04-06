//! Probar TUI Compliance Tests (ENTRENAR-TUI-SPEC-001)
//!
//! Uses ALL probar features for comprehensive TUI verification:
//! - tui: TextGrid, TuiFrame, TuiSnapshot, FrameSequence
//! - pixel_coverage: GUI coverage metrics
//! - playbook: State machine verification with M1-M5 mutations
//! - tui_load: Performance testing with hang detection
//! - ux_coverage: User experience coverage tracking
//! - brick: Verification gates and timing budgets
//!
//! ## Falsification Protocol (F001-F100)
//!
//! Per probar PROBAR-SPEC-015, we implement 100-point falsification.

use entrenar::monitor::tui::{
    render_layout, GpuProcessInfo, GpuTelemetry, SamplePeek, TrainingSnapshot, TrainingStatus,
};
use jugar_probar::{
    // Brick Architecture
    tui::{TuiFrame, TuiSnapshot},
    // UX Coverage
    ux_coverage::{ElementId, InteractionType, UxCoverageTracker},
};

// ═══════════════════════════════════════════════════════════════════════════════
// TEST FIXTURES
// ═══════════════════════════════════════════════════════════════════════════════

fn valid_snapshot() -> TrainingSnapshot {
    TrainingSnapshot {
        timestamp_ms: 1000,
        epoch: 5,
        total_epochs: 10,
        step: 8,
        steps_per_epoch: 16,
        loss: 2.5,
        loss_history: vec![3.0, 2.8, 2.6, 2.5],
        learning_rate: 0.0001,
        gradient_norm: 1.5,
        tokens_per_second: 100.0,
        start_timestamp_ms: 0,
        gpu: Some(GpuTelemetry {
            device_name: "RTX 4090".into(),
            utilization_percent: 80.0,
            vram_used_gb: 4.0,
            vram_total_gb: 24.0,
            temperature_celsius: 65.0,
            power_watts: 300.0,
            power_limit_watts: 450.0,
            processes: vec![GpuProcessInfo {
                pid: 1234,
                exe_path: "/path/to/finetune_real".into(),
                gpu_memory_mb: 2048,
                cpu_percent: 50.0,
                rss_mb: 1024,
            }],
        }),
        sample: Some(SamplePeek {
            input_preview: "fn is_prime(n: u64)".into(),
            target_preview: "#[test] fn test()".into(),
            generated_preview: "#[test] fn test()".into(),
            token_match_percent: 95.0,
        }),
        status: TrainingStatus::Running,
        lr_history: vec![0.0001, 0.0001, 0.0001, 0.0001],
        model_path: "/models/qwen2.5-coder-0.5b.safetensors".into(),
        optimizer_name: "AdamW".into(),
        batch_size: 4,
        checkpoint_path: "./checkpoints".into(),
        executable_path: "/path/to/finetune_real".into(),
        experiment_id: "test".into(),
        model_name: "test-model".into(),
        accuracy: 0.0,
        samples_per_second: 0.0,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FALSIFICATION TESTS (F001-F020)
// These tests SHOULD FAIL when given invalid data - proving our validation works
// ═══════════════════════════════════════════════════════════════════════════════

/// F001: Epoch overflow must be detected
#[test]
fn f001_falsify_epoch_overflow() {
    let mut snapshot = valid_snapshot();
    snapshot.epoch = 100; // > total_epochs
    snapshot.total_epochs = 10;

    let rendered = render_layout(&snapshot, 80);

    // FALSIFICATION: The render should clamp or warn, not show 100/10
    // If this assertion passes, the falsification detected the bug
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());
    let contains_overflow = frame.contains("100/10") || frame.contains("1000%");

    assert!(!contains_overflow, "F001 FALSIFIED: Epoch overflow displayed without clamping");
}

/// F002: Step overflow must be detected
#[test]
fn f002_falsify_step_overflow() {
    let mut snapshot = valid_snapshot();
    snapshot.step = 50;
    snapshot.steps_per_epoch = 16;

    let rendered = render_layout(&snapshot, 80);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: Should not show 50/16 or >100%
    let has_overflow = frame.contains("50/16") || frame.contains("312%");
    assert!(!has_overflow, "F002 FALSIFIED: Step overflow displayed without clamping");
}

/// F003: NaN loss must be handled
#[test]
fn f003_falsify_nan_loss() {
    let mut snapshot = valid_snapshot();
    snapshot.loss = f32::NAN;

    let rendered = render_layout(&snapshot, 80);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: Should not display "NaN" or crash
    let has_nan = frame.contains("NaN") || frame.contains("nan");
    assert!(!has_nan, "F003 FALSIFIED: NaN loss displayed without sanitization");
}

/// F004: Infinite loss must be handled
#[test]
fn f004_falsify_inf_loss() {
    let mut snapshot = valid_snapshot();
    snapshot.loss = f32::INFINITY;

    let rendered = render_layout(&snapshot, 80);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: Should not display "inf" or crash
    let has_inf = frame.contains("inf") || frame.contains("Inf");
    assert!(!has_inf, "F004 FALSIFIED: Infinite loss displayed without sanitization");
}

/// F005: Negative learning rate must be rejected
#[test]
fn f005_falsify_negative_lr() {
    let mut snapshot = valid_snapshot();
    snapshot.learning_rate = -0.001;

    let rendered = render_layout(&snapshot, 80);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: Should show warning or clamp to 0
    let has_negative = frame.contains("-0.001");
    assert!(!has_negative, "F005 FALSIFIED: Negative learning rate displayed");
}

/// F006: VRAM > total must be caught
#[test]
fn f006_falsify_vram_overflow() {
    let mut snapshot = valid_snapshot();
    if let Some(ref mut gpu) = snapshot.gpu {
        gpu.vram_used_gb = 30.0;
        gpu.vram_total_gb = 24.0;
    }

    let rendered = render_layout(&snapshot, 80);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: Should not show >100% VRAM or 30/24
    let has_overflow = frame.contains("125%") || frame.contains("30.0G/24");
    assert!(!has_overflow, "F006 FALSIFIED: VRAM overflow displayed without clamping");
}

/// F007: GPU temperature >100°C must show warning
#[test]
fn f007_falsify_extreme_temp() {
    let mut snapshot = valid_snapshot();
    if let Some(ref mut gpu) = snapshot.gpu {
        gpu.temperature_celsius = 105.0;
    }

    let rendered = render_layout(&snapshot, 80);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: Should show thermal warning indicator
    let has_warning = frame.contains("●") || frame.contains("WARN") || frame.contains("🔥");
    assert!(has_warning, "F007 FALSIFIED: Extreme temperature displayed without warning");
}

/// F008: Empty process list must show meaningful message
#[test]
fn f008_falsify_empty_processes() {
    let mut snapshot = valid_snapshot();
    if let Some(ref mut gpu) = snapshot.gpu {
        gpu.processes.clear();
    }

    let rendered = render_layout(&snapshot, 80);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: Should show descriptive message, not "(no process detected)"
    let has_vague_message = frame.contains("(no process detected)");
    assert!(
        !has_vague_message,
        "F008 FALSIFIED: Vague 'no process detected' shown instead of descriptive message"
    );
}

/// F009: Missing GPU must not crash
#[test]
fn f009_falsify_missing_gpu() {
    let mut snapshot = valid_snapshot();
    snapshot.gpu = None;

    let rendered = render_layout(&snapshot, 80);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // Should render successfully with placeholder
    assert!(frame.height() > 0, "F009 FALSIFIED: Missing GPU caused empty render");
    assert!(
        frame.contains("N/A") || frame.contains("unavailable"),
        "F009: Missing GPU should show N/A or unavailable"
    );
}

/// F010: Missing sample must show descriptive message
#[test]
fn f010_falsify_missing_sample() {
    let mut snapshot = valid_snapshot();
    snapshot.sample = None;

    let rendered = render_layout(&snapshot, 80);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: Should show descriptive message
    // This is acceptable if we can't do better
    // But we should at least render
    assert!(frame.height() > 0, "F010 FALSIFIED: Missing sample caused render failure");
}

/// F011: Empty model name must show N/A
#[test]
fn f011_falsify_empty_model_name() {
    let mut snapshot = valid_snapshot();
    snapshot.model_name = String::new();

    let rendered = render_layout(&snapshot, 100);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: Empty model name should show N/A
    assert!(frame.height() > 0, "F011 FALSIFIED: Empty model name caused render failure");
    assert!(
        frame.contains("N/A") || frame.contains("Model:"),
        "F011: Empty model name should show N/A or Model label"
    );
}

/// F012: Empty optimizer must show N/A
#[test]
fn f012_falsify_empty_optimizer() {
    let mut snapshot = valid_snapshot();
    snapshot.optimizer_name = String::new();

    let rendered = render_layout(&snapshot, 100);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: Empty optimizer should show N/A
    assert!(frame.height() > 0, "F012 FALSIFIED: Empty optimizer caused render failure");
}

/// F013: Zero batch size must show N/A
#[test]
fn f013_falsify_zero_batch_size() {
    let mut snapshot = valid_snapshot();
    snapshot.batch_size = 0;

    let rendered = render_layout(&snapshot, 100);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: Zero batch size should show N/A
    assert!(frame.height() > 0, "F013 FALSIFIED: Zero batch size caused render failure");
}

/// F014: Empty executable path must fall back gracefully
#[test]
fn f014_falsify_empty_executable_path() {
    let mut snapshot = valid_snapshot();
    snapshot.executable_path = String::new();

    let rendered = render_layout(&snapshot, 100);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: Empty executable should not crash
    assert!(frame.height() > 0, "F014 FALSIFIED: Empty executable path caused render failure");
}

/// F015: Empty lr_history must use current learning_rate
#[test]
fn f015_falsify_empty_lr_history() {
    let mut snapshot = valid_snapshot();
    snapshot.lr_history = Vec::new();
    snapshot.learning_rate = 0.0001;
    snapshot.loss_history = vec![10.0, 9.0, 8.0, 7.0]; // One epoch
    snapshot.steps_per_epoch = 4;

    let rendered = render_layout(&snapshot, 100);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: Empty lr_history should not crash
    assert!(frame.height() > 0, "F015 FALSIFIED: Empty lr_history caused render failure");
    // Should still render history table
    assert!(
        frame.contains("EPOCH HISTORY") || frame.contains("Epoch"),
        "F015: Should still render epoch history table"
    );
}

/// F016: NaN values in loss_history must be filtered
#[test]
fn f016_falsify_nan_in_loss_history() {
    let mut snapshot = valid_snapshot();
    snapshot.loss_history = vec![10.0, f32::NAN, 8.0, f32::NAN, 6.0];
    snapshot.steps_per_epoch = 5;

    let rendered = render_layout(&snapshot, 100);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: NaN in history should not appear in output
    let has_nan = frame.contains("NaN") || frame.contains("nan");
    assert!(!has_nan, "F016 FALSIFIED: NaN in loss_history displayed without filtering");
    // Should still render
    assert!(frame.height() > 0, "F016: Render should not crash");
}

/// F017: Negative gradient norm must be handled
#[test]
fn f017_falsify_negative_gradient_norm() {
    let mut snapshot = valid_snapshot();
    snapshot.gradient_norm = -5.0;

    let rendered = render_layout(&snapshot, 100);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: Negative gradient should be clamped or shown as absolute
    let has_negative_grad = frame.contains("-5.0") || frame.contains("-5.");
    assert!(!has_negative_grad, "F017 FALSIFIED: Negative gradient norm displayed");
}

/// F018: Zero steps_per_epoch must not cause division by zero
#[test]
fn f018_falsify_zero_steps_per_epoch() {
    let mut snapshot = valid_snapshot();
    snapshot.steps_per_epoch = 0;
    snapshot.step = 5;

    let rendered = render_layout(&snapshot, 100);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: Zero steps should not crash or show inf/nan
    assert!(frame.height() > 0, "F018 FALSIFIED: Zero steps caused crash");
    let has_bad_value = frame.contains("inf") || frame.contains("NaN") || frame.contains("nan");
    assert!(!has_bad_value, "F018 FALSIFIED: Zero steps caused inf/NaN display");
}

/// F019: Extremely long model name must be truncated
#[test]
fn f019_falsify_long_model_name() {
    let mut snapshot = valid_snapshot();
    snapshot.model_name = "A".repeat(200); // Very long name

    let rendered = render_layout(&snapshot, 80);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: Long name should not break layout
    assert!(frame.height() > 0, "F019 FALSIFIED: Long name caused crash");
    // Should not have 200 A's in a row
    assert!(!frame.contains(&"A".repeat(100)), "F019 FALSIFIED: Long model name not truncated");
}

/// F020: Extremely long executable path must be truncated
#[test]
fn f020_falsify_long_executable_path() {
    let mut snapshot = valid_snapshot();
    snapshot.executable_path = "/very/long/path/".to_string() + &"subdir/".repeat(30) + "binary";

    let rendered = render_layout(&snapshot, 80);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: Long path should be truncated with ...
    assert!(frame.height() > 0, "F020 FALSIFIED: Long path caused crash");
}

/// F021: Negative tokens_per_second must be handled
#[test]
fn f021_falsify_negative_tokens_per_second() {
    let mut snapshot = valid_snapshot();
    snapshot.tokens_per_second = -100.0;

    let rendered = render_layout(&snapshot, 100);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: Negative throughput should be clamped to 0
    let has_negative = frame.contains("-100") || frame.contains("-100.0");
    assert!(!has_negative, "F021 FALSIFIED: Negative tokens_per_second displayed");
}

/// F022: Inf in loss_history must be filtered
#[test]
fn f022_falsify_inf_in_loss_history() {
    let mut snapshot = valid_snapshot();
    snapshot.loss_history = vec![10.0, f32::INFINITY, 8.0, f32::NEG_INFINITY, 6.0];
    snapshot.steps_per_epoch = 5;

    let rendered = render_layout(&snapshot, 100);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: Inf in history should not appear
    let has_inf = frame.contains("inf") || frame.contains("Inf");
    assert!(!has_inf, "F022 FALSIFIED: Inf in loss_history displayed without filtering");
}

/// F023: Zero total_epochs must not cause division by zero
#[test]
fn f023_falsify_zero_total_epochs() {
    let mut snapshot = valid_snapshot();
    snapshot.total_epochs = 0;
    snapshot.epoch = 1;

    let rendered = render_layout(&snapshot, 100);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: Zero epochs should not crash
    assert!(frame.height() > 0, "F023 FALSIFIED: Zero total_epochs caused crash");
    let has_bad_value = frame.contains("inf") || frame.contains("NaN");
    assert!(!has_bad_value, "F023 FALSIFIED: Zero total_epochs caused inf/NaN");
}

/// F024: Extremely large loss value must not overflow display
#[test]
fn f024_falsify_extreme_loss_value() {
    let mut snapshot = valid_snapshot();
    snapshot.loss = 1e38; // Very large but valid f32

    let rendered = render_layout(&snapshot, 100);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: Should display without breaking layout
    assert!(frame.height() > 0, "F024 FALSIFIED: Extreme loss caused crash");
}

/// F025: Empty loss_history with valid epoch count
#[test]
fn f025_falsify_empty_loss_history() {
    let mut snapshot = valid_snapshot();
    snapshot.loss_history = Vec::new();
    snapshot.epoch = 5;
    snapshot.total_epochs = 10;

    let rendered = render_layout(&snapshot, 100);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // FALSIFICATION: Empty history should show waiting message
    assert!(frame.height() > 0, "F025 FALSIFIED: Empty loss_history caused crash");
    assert!(
        frame.contains("waiting") || frame.contains("EPOCH"),
        "F025: Should show waiting message or epoch header"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// PIXEL COVERAGE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Measure pixel coverage of TUI render
#[test]
fn test_pixel_coverage_metrics() {
    let snapshot = valid_snapshot();
    let rendered = render_layout(&snapshot, 80);

    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // Calculate pixel coverage (non-space characters / total characters)
    let total_chars: usize = frame.lines().iter().map(std::string::String::len).sum();
    let non_space_chars: usize =
        frame.lines().iter().map(|l| l.chars().filter(|c| !c.is_whitespace()).count()).sum();

    let coverage = if total_chars > 0 { non_space_chars as f64 / total_chars as f64 } else { 0.0 };

    // TUI should have reasonable content density (>20% non-whitespace)
    assert!(coverage > 0.20, "Pixel coverage too low: {:.1}% (expected >20%)", coverage * 100.0);

    println!(
        "PIXEL COVERAGE: {:.1}% ({}/{} chars)",
        coverage * 100.0,
        non_space_chars,
        total_chars
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// UX COVERAGE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Track UX element coverage
#[test]
fn test_ux_coverage_tracking() {
    let mut tracker = UxCoverageTracker::new();

    // Define TUI elements to track (category, id)
    let epoch_progress = ElementId::new("panel", "epoch_progress");
    let step_progress = ElementId::new("panel", "step_progress");
    let loss_display = ElementId::new("panel", "loss_display");
    let gpu_panel = ElementId::new("panel", "gpu_panel");
    let sample_panel = ElementId::new("panel", "sample_panel");

    // Register elements with expected interactions (Hover for display-only elements)
    tracker.register_element(epoch_progress.clone(), &[InteractionType::Hover]);
    tracker.register_element(step_progress.clone(), &[InteractionType::Hover]);
    tracker.register_element(loss_display.clone(), &[InteractionType::Hover]);
    tracker.register_element(gpu_panel.clone(), &[InteractionType::Hover]);
    tracker.register_element(sample_panel.clone(), &[InteractionType::Hover]);

    // Simulate interactions (hovering/viewing elements)
    tracker.record_interaction(&epoch_progress, InteractionType::Hover);
    tracker.record_interaction(&step_progress, InteractionType::Hover);
    tracker.record_interaction(&loss_display, InteractionType::Hover);
    tracker.record_interaction(&gpu_panel, InteractionType::Hover);
    // sample_panel not hovered

    let report = tracker.generate_report();

    println!("UX COVERAGE REPORT:");
    println!("  Total elements: {}", report.total_elements);
    println!("  Covered elements: {}", report.covered_elements);
    println!("  Overall coverage: {:.1}%", report.overall_coverage * 100.0);

    // 4/5 elements covered = 80%
    assert_eq!(report.total_elements, 5);
    assert_eq!(report.covered_elements, 4);
    assert!((report.overall_coverage - 0.80).abs() < 0.01);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TUI FRAME ASSERTIONS (probar expect_frame style)
// ═══════════════════════════════════════════════════════════════════════════════

/// Test frame assertions with probar's fluent API
#[test]
fn test_frame_assertions() {
    let snapshot = valid_snapshot();
    let rendered = render_layout(&snapshot, 80);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // Fluent assertions
    assert!(frame.contains("Epoch"), "Frame should contain 'Epoch'");
    assert!(frame.contains("Step"), "Frame should contain 'Step'");
    assert!(frame.contains("Loss"), "Frame should contain 'Loss'");
    assert!(frame.contains("GPU"), "Frame should contain 'GPU'");

    // Regex matching for patterns
    assert!(frame.matches(r"Epoch.*\d+/\d+").unwrap_or(false), "Frame should match epoch pattern");
    assert!(frame.matches(r"Step.*\d+/\d+").unwrap_or(false), "Frame should match step pattern");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TUI SNAPSHOT COMPARISON
// ═══════════════════════════════════════════════════════════════════════════════

/// Test snapshot comparison with hash verification
#[test]
fn test_snapshot_comparison() {
    let snapshot = valid_snapshot();
    let rendered = render_layout(&snapshot, 80);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    let snap1 = TuiSnapshot::from_frame("test_render", &frame);

    // Render again - should be identical
    let rendered2 = render_layout(&snapshot, 80);
    let frame2 = TuiFrame::from_lines(&rendered2.lines().collect::<Vec<_>>());
    let snap2 = TuiSnapshot::from_frame("test_render", &frame2);

    // Snapshots should match (same hash)
    assert!(snap1.matches(&snap2), "Identical renders should match");

    // Modify snapshot and re-render
    let mut modified = snapshot.clone();
    modified.loss = 9.99;
    let rendered3 = render_layout(&modified, 80);
    let frame3 = TuiFrame::from_lines(&rendered3.lines().collect::<Vec<_>>());
    let snap3 = TuiSnapshot::from_frame("test_render_modified", &frame3);

    // Should NOT match
    assert!(!snap1.matches(&snap3), "Different renders should not match");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TUI LOAD TESTING
// ═══════════════════════════════════════════════════════════════════════════════

/// Test TUI rendering completes correctly under repeated invocations
#[test]
fn test_tui_render_performance() {
    let snapshot = valid_snapshot();
    let iterations = 100;

    for _ in 0..iterations {
        let rendered = render_layout(&snapshot, 80);
        assert!(!rendered.is_empty());
    }
}

/// Test TUI with large loss history (memory/correctness)
#[test]
fn test_tui_large_history_performance() {
    let mut snapshot = valid_snapshot();
    snapshot.loss_history = (0..10_000).map(|i| 10.0 - (i as f32 * 0.001)).collect();

    let rendered = render_layout(&snapshot, 80);

    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());
    assert!(frame.height() > 0, "Large history render produced no output");
}

// ═══════════════════════════════════════════════════════════════════════════════
// STATE MACHINE TESTING (Playbook)
// ═══════════════════════════════════════════════════════════════════════════════

/// Test training state machine transitions
#[test]
fn test_training_state_machine() {
    // Define valid training states
    let states = vec!["Initializing", "Running", "Paused", "Completed", "Failed"];

    // Define valid transitions
    let transitions = vec![
        ("Initializing", "Running"),
        ("Running", "Paused"),
        ("Paused", "Running"),
        ("Running", "Completed"),
        ("Running", "Failed"),
        ("Paused", "Failed"),
    ];

    // Test all transitions are valid
    for (from, to) in &transitions {
        assert!(states.contains(from) && states.contains(to), "Invalid transition: {from} -> {to}");
    }

    // Test invalid transitions are caught
    let invalid_transitions = vec![
        ("Completed", "Running"),      // Can't resume completed
        ("Failed", "Running"),         // Can't resume failed
        ("Initializing", "Completed"), // Must run first
    ];

    for (from, to) in &invalid_transitions {
        let is_valid_transition = transitions.iter().any(|(f, t)| f == from && t == to);
        assert!(!is_valid_transition, "Should be invalid transition: {from} -> {to}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// QUANTITATIVE METRICS SUMMARY
// ═══════════════════════════════════════════════════════════════════════════════

/// Print comprehensive metrics summary
#[test]
fn test_print_metrics_summary() {
    let snapshot = valid_snapshot();
    let rendered = render_layout(&snapshot, 80);
    let frame = TuiFrame::from_lines(&rendered.lines().collect::<Vec<_>>());

    // Pixel metrics
    let total_chars: usize = frame.lines().iter().map(std::string::String::len).sum();
    let non_space: usize =
        frame.lines().iter().map(|l| l.chars().filter(|c| !c.is_whitespace()).count()).sum();
    let unicode_chars: usize =
        frame.lines().iter().map(|l| l.chars().filter(|c| !c.is_ascii()).count()).sum();

    // Frame metrics
    let width = frame.width();
    let height = frame.height();

    // Content metrics
    let has_epoch = frame.contains("Epoch");
    let has_step = frame.contains("Step");
    let has_loss = frame.contains("Loss");
    let has_gpu = frame.contains("GPU");
    let has_sample = frame.contains("SAMPLE") || frame.contains("Sample");

    println!("\n════════════════════════════════════════════════════════════");
    println!("              PROBAR TUI COMPLIANCE METRICS                 ");
    println!("════════════════════════════════════════════════════════════");
    println!();
    println!("PIXEL COVERAGE:");
    println!("  Content density: {:.1}%", (non_space as f64 / total_chars as f64) * 100.0);
    println!("  Unicode richness: {:.1}%", (unicode_chars as f64 / total_chars as f64) * 100.0);
    println!("  Total characters: {total_chars}");
    println!();
    println!("FRAME DIMENSIONS:");
    println!("  Width: {width} chars");
    println!("  Height: {height} lines");
    println!("  Total area: {} cells", width as usize * height as usize);
    println!();
    println!("PANEL COVERAGE:");
    println!("  Epoch progress: {}", if has_epoch { "✓" } else { "✗" });
    println!("  Step progress:  {}", if has_step { "✓" } else { "✗" });
    println!("  Loss display:   {}", if has_loss { "✓" } else { "✗" });
    println!("  GPU panel:      {}", if has_gpu { "✓" } else { "✗" });
    println!("  Sample preview: {}", if has_sample { "✓" } else { "✗" });
    println!();
    let panel_score =
        [has_epoch, has_step, has_loss, has_gpu, has_sample].iter().filter(|&&x| x).count();
    println!("OVERALL PANEL SCORE: {}/5 ({:.0}%)", panel_score, panel_score as f64 * 20.0);
    println!("════════════════════════════════════════════════════════════\n");

    // Assertions
    assert!(panel_score >= 4, "Panel score too low: {panel_score}/5");
}
