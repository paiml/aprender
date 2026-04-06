//! TUI Snapshot Tests (ENT-140, ENT-144, ENT-145)
//!
//! Uses probar's TuiSnapshot system for golden file testing and insta for additional snapshots.
//! Golden snapshots are stored in tests/snapshots/ (insta) and tests/probar_snapshots/ (probar).
//!
//! # Running Tests
//!
//! ```bash
//! cargo test --test tui_snapshot_test
//!
//! # Update insta snapshots when intentionally changing TUI
//! cargo insta review
//!
//! # Update probar snapshots
//! PROBAR_UPDATE_SNAPSHOTS=1 cargo test --test tui_snapshot_test
//! ```

use entrenar::monitor::tui::{
    render_layout_colored, ColorMode, GpuProcessInfo, GpuTelemetry, SamplePeek, TrainingSnapshot,
    TrainingStatus,
};
use jugar_probar::tui::{FrameSequence, SnapshotManager, TuiFrame, TuiSnapshot};
use std::path::Path;

/// Snapshot directory for probar-based tests
const PROBAR_SNAPSHOT_DIR: &str = "tests/probar_snapshots";

/// Create a mock training snapshot for testing
fn mock_snapshot(epoch: usize, step: usize, loss: f32) -> TrainingSnapshot {
    TrainingSnapshot {
        timestamp_ms: 1700000000000,
        epoch,
        total_epochs: 15,
        step,
        steps_per_epoch: 16,
        loss,
        loss_history: vec![6.9, 6.5, 6.0, 5.5, 5.0, loss],
        learning_rate: 6e-4,
        gradient_norm: 2.5,
        tokens_per_second: 30.3,
        start_timestamp_ms: 1700000000000 - 60_000,
        gpu: Some(GpuTelemetry {
            device_name: "NVIDIA GeForce RTX 4090".to_string(),
            utilization_percent: 35.0,
            vram_used_gb: 2.1,
            vram_total_gb: 24.0,
            temperature_celsius: 45.0,
            power_watts: 110.0,
            power_limit_watts: 480.0,
            processes: vec![],
        }),
        sample: Some(SamplePeek {
            input_preview: "fn is_prime(n: u64) -> bool { ... }".to_string(),
            target_preview: "#[test] fn test_is_prime() { ... }".to_string(),
            generated_preview: "(training...)".to_string(),
            token_match_percent: 0.0,
        }),
        status: TrainingStatus::Running,
        experiment_id: "test-experiment".to_string(),
        model_name: "Qwen2.5-Coder-0.5B".to_string(),
        lr_history: vec![6e-4; 6],
        model_path: "/models/qwen2.5-coder-0.5b.safetensors".to_string(),
        optimizer_name: "AdamW".to_string(),
        batch_size: 4,
        checkpoint_path: "./experiments/test/checkpoints".to_string(),
        executable_path: "/usr/bin/finetune_real".to_string(),
        accuracy: 0.0,
        samples_per_second: 0.0,
    }
}

/// Convert rendered TUI output to a TuiFrame for probar testing
fn rendered_to_frame(rendered: &str) -> TuiFrame {
    let lines: Vec<&str> = rendered.lines().collect();
    TuiFrame::from_lines(&lines)
}

/// Check if we should update snapshots (via environment variable)
fn should_update_snapshots() -> bool {
    std::env::var("PROBAR_UPDATE_SNAPSHOTS").is_ok()
}

// ═══════════════════════════════════════════════════════════════════════════════
// INSTA SNAPSHOT TESTS (existing)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_tui_initial_state_snapshot() {
    let snapshot = mock_snapshot(1, 1, 6.9);
    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    insta::assert_snapshot!("tui_initial_state", rendered);
}

#[test]
fn test_tui_mid_training_snapshot() {
    let snapshot = mock_snapshot(8, 10, 4.2);
    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    insta::assert_snapshot!("tui_mid_training", rendered);
}

#[test]
fn test_tui_final_state_snapshot() {
    let mut snapshot = mock_snapshot(15, 16, 2.1);
    snapshot.status = TrainingStatus::Completed;
    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    insta::assert_snapshot!("tui_final_state", rendered);
}

#[test]
fn test_tui_error_state_snapshot() {
    let mut snapshot = mock_snapshot(5, 8, 15.0);
    snapshot.status = TrainingStatus::Failed("CUDA OOM".to_string());
    snapshot.gradient_norm = 1000.0; // Exploding gradient
    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    insta::assert_snapshot!("tui_error_state", rendered);
}

#[test]
fn test_tui_no_gpu_snapshot() {
    let mut snapshot = mock_snapshot(3, 5, 5.5);
    snapshot.gpu = None; // CPU-only training
    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    insta::assert_snapshot!("tui_no_gpu", rendered);
}

#[test]
fn test_tui_no_sample_snapshot() {
    let mut snapshot = mock_snapshot(2, 3, 6.2);
    snapshot.sample = None; // Before first sample
    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    insta::assert_snapshot!("tui_no_sample", rendered);
}

// Frame sequence test for progress animation (ENT-144)
#[test]
fn test_tui_progress_sequence() {
    let mut frames = Vec::new();

    // Capture frames at different progress points
    for step in [1, 4, 8, 12, 16] {
        let loss = 6.9 - (step as f32 * 0.3);
        let snapshot = mock_snapshot(1, step, loss);
        let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        frames.push(rendered);
    }

    // Verify frame count
    assert_eq!(frames.len(), 5);

    // Verify frames are different (progress is visible)
    for i in 1..frames.len() {
        assert_ne!(frames[i - 1], frames[i], "Frame {} should differ from frame {}", i - 1, i);
    }
}

// Test different color modes
#[test]
fn test_tui_color_modes() {
    let snapshot = mock_snapshot(5, 8, 4.5);

    // Test each color mode
    let mono = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let color16 = render_layout_colored(&snapshot, 80, ColorMode::Color16);
    let color256 = render_layout_colored(&snapshot, 80, ColorMode::Color256);

    // Color modes should produce different output
    assert_ne!(mono.len(), color256.len(), "Color output should have ANSI codes");

    // Mono should have no ANSI escape sequences
    assert!(!mono.contains("\x1b["), "Mono mode should not contain ANSI codes");

    // Color modes should have ANSI escape sequences
    assert!(
        color16.contains("\x1b[") || color256.contains("\x1b["),
        "Color modes should contain ANSI codes"
    );
}

// Test different terminal widths
#[test]
fn test_tui_width_adaptation() {
    let snapshot = mock_snapshot(5, 8, 4.5);

    let narrow = render_layout_colored(&snapshot, 60, ColorMode::Mono);
    let standard = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let wide = render_layout_colored(&snapshot, 120, ColorMode::Mono);

    // All renders should produce non-empty output
    assert!(!narrow.is_empty(), "Narrow render should not be empty");
    assert!(!standard.is_empty(), "Standard render should not be empty");
    assert!(!wide.is_empty(), "Wide render should not be empty");

    // All should contain core elements
    assert!(standard.contains("ENTRENAR"), "Should contain header");
    assert!(standard.contains("Loss"), "Should contain loss info");
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROBAR TUI SNAPSHOT TESTS (ENT-140)
// ═══════════════════════════════════════════════════════════════════════════════

/// Test using probar's TuiSnapshot for single frame capture
#[test]
fn test_probar_tui_snapshot_basic() {
    let snapshot = mock_snapshot(5, 8, 4.5);
    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let frame = rendered_to_frame(&rendered);

    // Create TuiSnapshot from frame
    let tui_snap = TuiSnapshot::from_frame("basic_training", &frame);

    // Verify basic properties
    assert_eq!(tui_snap.name, "basic_training");
    assert!(!tui_snap.hash.is_empty());
    assert!(tui_snap.height > 0);
    assert!(tui_snap.width > 0);

    // Verify content contains expected elements
    let frame_text = frame.as_text();
    assert!(frame_text.contains("ENTRENAR"));
    assert!(frame_text.contains("Loss History"));
    assert!(frame_text.contains("GPU"));
}

/// Test probar's SnapshotManager for golden file comparison
#[test]
fn test_probar_snapshot_manager() {
    let manager = SnapshotManager::new(Path::new(PROBAR_SNAPSHOT_DIR))
        .with_update_mode(should_update_snapshots());

    let snapshot = mock_snapshot(1, 1, 6.9);
    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let frame = rendered_to_frame(&rendered);

    // This will create the snapshot on first run, then compare on subsequent runs
    let result = manager.assert_snapshot("probar_initial_state", &frame);

    // Handle the result based on whether the snapshot exists
    match result {
        Ok(()) => {} // Snapshot matched or was created
        Err(e) => {
            // Only fail if the snapshot exists and doesn't match (not in update mode)
            assert!(
                !manager.exists("probar_initial_state") || should_update_snapshots(),
                "Snapshot mismatch - run with PROBAR_UPDATE_SNAPSHOTS=1 to update: {e}"
            );
            // Otherwise the error was creating a new snapshot, which is fine
        }
    }
}

/// Test probar's FrameSequence for animation testing
#[test]
fn test_probar_frame_sequence() {
    let mut sequence = FrameSequence::new("training_progress");

    // Generate a sequence of training frames
    for step in [1, 4, 8, 12, 16] {
        let loss = 6.9 - (step as f32 * 0.3);
        let snapshot = mock_snapshot(1, step, loss);
        let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        let frame = rendered_to_frame(&rendered);
        sequence.add_frame(&frame);
    }

    // Verify sequence properties
    assert_eq!(sequence.len(), 5);
    assert!(!sequence.is_empty());
    assert!(sequence.first().is_some());
    assert!(sequence.last().is_some());

    // Verify frames are different (progress is visible)
    let first = sequence.first().expect("collection should not be empty");
    let last = sequence.last().expect("collection should not be empty");
    assert!(!first.matches(last), "First and last frames should differ");

    // Verify intermediate frames exist
    assert!(sequence.frame_at(2).is_some());
}

// ═══════════════════════════════════════════════════════════════════════════════
// EDGE CASE TESTS (ENT-145)
// ═══════════════════════════════════════════════════════════════════════════════

/// Edge case: Progress >100% (step exceeds steps_per_epoch)
#[test]
fn test_edge_case_progress_over_100_percent() {
    let mut snapshot = mock_snapshot(1, 20, 4.0); // step=20 > steps_per_epoch=16
    snapshot.step = 20;
    snapshot.steps_per_epoch = 16;

    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let frame = rendered_to_frame(&rendered);

    // Should not panic and should clamp to 100%
    assert!(frame.as_text().contains("100%") || frame.as_text().contains("16/16"));

    // Verify with probar TuiSnapshot
    let tui_snap = TuiSnapshot::from_frame("over_100_percent", &frame)
        .with_metadata("edge_case", "progress_overflow");
    assert!(!tui_snap.content.is_empty());
}

/// Edge case: Epoch exceeds total_epochs
#[test]
fn test_edge_case_epoch_over_100_percent() {
    let mut snapshot = mock_snapshot(18, 16, 2.0); // epoch=18 > total_epochs=15
    snapshot.epoch = 18;
    snapshot.total_epochs = 15;

    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let frame = rendered_to_frame(&rendered);

    // Should not panic
    assert!(frame.as_text().contains("Epoch"));

    let tui_snap = TuiSnapshot::from_frame("epoch_overflow", &frame)
        .with_metadata("edge_case", "epoch_overflow");
    assert!(!tui_snap.content.is_empty());
}

/// Edge case: Missing GPU data (None)
#[test]
fn test_edge_case_missing_gpu_data() {
    let mut snapshot = mock_snapshot(5, 8, 4.5);
    snapshot.gpu = None;

    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let frame = rendered_to_frame(&rendered);

    // Should show N/A or unavailable message
    let text = frame.as_text();
    assert!(
        text.contains("N/A") || text.contains("unavailable") || text.contains("no GPU"),
        "Should indicate GPU is unavailable"
    );

    // Verify with probar
    let tui_snap =
        TuiSnapshot::from_frame("no_gpu", &frame).with_metadata("edge_case", "missing_gpu");
    assert!(!tui_snap.content.is_empty());
}

/// Edge case: Missing sample data (None)
#[test]
fn test_edge_case_missing_sample_data() {
    let mut snapshot = mock_snapshot(1, 1, 6.9);
    snapshot.sample = None;

    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let frame = rendered_to_frame(&rendered);

    // Should render training info even without sample
    let text = frame.as_text();
    assert!(
        text.contains("Loss History") || text.contains("Config"),
        "Should show training progress even without sample"
    );

    let tui_snap =
        TuiSnapshot::from_frame("no_sample", &frame).with_metadata("edge_case", "missing_sample");
    assert!(!tui_snap.content.is_empty());
}

/// Edge case: Empty loss history
#[test]
fn test_edge_case_empty_loss_history() {
    let mut snapshot = mock_snapshot(1, 1, 0.0);
    snapshot.loss_history = vec![];
    snapshot.loss = 0.0;

    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let frame = rendered_to_frame(&rendered);

    // Should not panic and should show "waiting for data" or similar
    let text = frame.as_text();
    assert!(
        text.contains("waiting") || text.contains("LOSS"),
        "Should handle empty loss history gracefully"
    );

    let tui_snap = TuiSnapshot::from_frame("empty_loss_history", &frame)
        .with_metadata("edge_case", "empty_loss_history");
    assert!(!tui_snap.content.is_empty());
}

/// Edge case: Zero tokens per second
#[test]
fn test_edge_case_zero_tokens_per_second() {
    let mut snapshot = mock_snapshot(1, 1, 6.9);
    snapshot.tokens_per_second = 0.0;

    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let frame = rendered_to_frame(&rendered);

    // Should not show "0 tok/s" prominently or should hide it
    // The header should still render correctly
    assert!(frame.as_text().contains("ENTRENAR"));

    let tui_snap = TuiSnapshot::from_frame("zero_tps", &frame)
        .with_metadata("edge_case", "zero_tokens_per_second");
    assert!(!tui_snap.content.is_empty());
}

/// Edge case: Extremely high loss value
#[test]
fn test_edge_case_high_loss() {
    let mut snapshot = mock_snapshot(1, 1, 999999.0);
    snapshot.loss = 999999.0;
    snapshot.loss_history = vec![999999.0, 999998.0, 999997.0];

    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let frame = rendered_to_frame(&rendered);

    // Should render without panicking
    assert!(frame.as_text().contains("Loss"));

    let tui_snap = TuiSnapshot::from_frame("high_loss", &frame)
        .with_metadata("edge_case", "extreme_high_loss");
    assert!(!tui_snap.content.is_empty());
}

/// Edge case: Exploding gradient (high gradient norm)
#[test]
fn test_edge_case_exploding_gradient() {
    let mut snapshot = mock_snapshot(5, 8, 15.0);
    snapshot.gradient_norm = 1e10; // Exploding gradient

    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let frame = rendered_to_frame(&rendered);

    // Should render gradient info
    assert!(frame.as_text().contains("Grad"));

    let tui_snap = TuiSnapshot::from_frame("exploding_gradient", &frame)
        .with_metadata("edge_case", "exploding_gradient");
    assert!(!tui_snap.content.is_empty());
}

/// Edge case: Vanishing gradient (very small gradient norm)
#[test]
fn test_edge_case_vanishing_gradient() {
    let mut snapshot = mock_snapshot(5, 8, 0.001);
    snapshot.gradient_norm = 1e-10; // Vanishing gradient

    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let frame = rendered_to_frame(&rendered);

    // Should render gradient info
    assert!(frame.as_text().contains("Grad"));

    let tui_snap = TuiSnapshot::from_frame("vanishing_gradient", &frame)
        .with_metadata("edge_case", "vanishing_gradient");
    assert!(!tui_snap.content.is_empty());
}

/// Edge case: GPU at thermal throttling temperature
#[test]
fn test_edge_case_gpu_thermal_throttling() {
    let mut snapshot = mock_snapshot(5, 8, 4.5);
    snapshot.gpu = Some(GpuTelemetry {
        device_name: "NVIDIA GeForce RTX 4090".to_string(),
        utilization_percent: 100.0,
        vram_used_gb: 23.5,
        vram_total_gb: 24.0,
        temperature_celsius: 95.0, // Thermal throttling
        power_watts: 450.0,
        power_limit_watts: 450.0,
        processes: vec![],
    });

    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let frame = rendered_to_frame(&rendered);

    // Should show temperature
    assert!(frame.as_text().contains("95"));

    let tui_snap = TuiSnapshot::from_frame("gpu_thermal_throttling", &frame)
        .with_metadata("edge_case", "thermal_throttling");
    assert!(!tui_snap.content.is_empty());
}

/// Edge case: GPU at 100% utilization and VRAM
#[test]
fn test_edge_case_gpu_maxed_out() {
    let mut snapshot = mock_snapshot(5, 8, 4.5);
    snapshot.gpu = Some(GpuTelemetry {
        device_name: "NVIDIA GeForce RTX 4090".to_string(),
        utilization_percent: 100.0,
        vram_used_gb: 24.0,
        vram_total_gb: 24.0, // 100% VRAM
        temperature_celsius: 83.0,
        power_watts: 450.0,
        power_limit_watts: 450.0,
        processes: vec![GpuProcessInfo {
            pid: 12345,
            exe_path: "/usr/bin/python3".to_string(),
            gpu_memory_mb: 24000,
            cpu_percent: 95.0,
            rss_mb: 8000,
        }],
    });

    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let frame = rendered_to_frame(&rendered);

    // Should show 100% values
    assert!(frame.as_text().contains("100%"));

    let tui_snap = TuiSnapshot::from_frame("gpu_maxed_out", &frame)
        .with_metadata("edge_case", "gpu_100_percent");
    assert!(!tui_snap.content.is_empty());
}

/// Edge case: Very narrow terminal (minimum width)
#[test]
fn test_edge_case_narrow_terminal() {
    let snapshot = mock_snapshot(5, 8, 4.5);
    let rendered = render_layout_colored(&snapshot, 40, ColorMode::Mono); // Very narrow
    let frame = rendered_to_frame(&rendered);

    // Should not panic
    assert!(!frame.as_text().is_empty());

    let tui_snap = TuiSnapshot::from_frame("narrow_terminal", &frame)
        .with_metadata("edge_case", "narrow_width");
    assert!(!tui_snap.content.is_empty());
}

/// Edge case: Very wide terminal
#[test]
fn test_edge_case_wide_terminal() {
    let snapshot = mock_snapshot(5, 8, 4.5);
    let rendered = render_layout_colored(&snapshot, 200, ColorMode::Mono); // Very wide
    let frame = rendered_to_frame(&rendered);

    // Should not panic
    assert!(!frame.as_text().is_empty());

    let tui_snap =
        TuiSnapshot::from_frame("wide_terminal", &frame).with_metadata("edge_case", "wide_width");
    assert!(!tui_snap.content.is_empty());
}

/// Edge case: Zero epochs/steps
#[test]
fn test_edge_case_zero_epochs() {
    let mut snapshot = mock_snapshot(0, 0, 0.0);
    snapshot.epoch = 0;
    snapshot.total_epochs = 0;
    snapshot.step = 0;
    snapshot.steps_per_epoch = 0;

    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let frame = rendered_to_frame(&rendered);

    // Should not panic (division by zero protection)
    assert!(!frame.as_text().is_empty());

    let tui_snap =
        TuiSnapshot::from_frame("zero_epochs", &frame).with_metadata("edge_case", "zero_epochs");
    assert!(!tui_snap.content.is_empty());
}

/// Edge case: Initializing status
#[test]
fn test_edge_case_initializing_status() {
    let mut snapshot = mock_snapshot(0, 0, 0.0);
    snapshot.status = TrainingStatus::Initializing;
    snapshot.tokens_per_second = 0.0;

    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let frame = rendered_to_frame(&rendered);

    // Should show Init status
    let text = frame.as_text();
    assert!(
        text.contains("Init") || text.contains("Initializing"),
        "Should show initializing status"
    );

    let tui_snap = TuiSnapshot::from_frame("initializing", &frame)
        .with_metadata("edge_case", "initializing_status");
    assert!(!tui_snap.content.is_empty());
}

/// Edge case: Paused status
#[test]
fn test_edge_case_paused_status() {
    let mut snapshot = mock_snapshot(5, 8, 4.5);
    snapshot.status = TrainingStatus::Paused;

    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let frame = rendered_to_frame(&rendered);

    // Should show Paused status
    assert!(frame.as_text().contains("Paused"), "Should show paused status");

    let tui_snap =
        TuiSnapshot::from_frame("paused", &frame).with_metadata("edge_case", "paused_status");
    assert!(!tui_snap.content.is_empty());
}

/// Edge case: Very long model/experiment names
#[test]
fn test_edge_case_long_names() {
    let mut snapshot = mock_snapshot(5, 8, 4.5);
    snapshot.model_name = "a".repeat(100);
    snapshot.experiment_id = "b".repeat(100);
    if let Some(ref mut gpu) = snapshot.gpu {
        gpu.device_name = "NVIDIA Super Ultra Mega Long Device Name 4090 Ti Super".to_string();
    }

    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let frame = rendered_to_frame(&rendered);

    // Should handle truncation gracefully
    assert!(!frame.as_text().is_empty());

    let tui_snap =
        TuiSnapshot::from_frame("long_names", &frame).with_metadata("edge_case", "long_names");
    assert!(!tui_snap.content.is_empty());
}

/// Edge case: Token match at exactly 100%
#[test]
fn test_edge_case_perfect_token_match() {
    let mut snapshot = mock_snapshot(15, 16, 0.01);
    snapshot.sample = Some(SamplePeek {
        input_preview: "fn add(a: i32, b: i32) -> i32 { a + b }".to_string(),
        target_preview: "#[test] fn test_add() { assert_eq!(add(1, 2), 3); }".to_string(),
        generated_preview: "#[test] fn test_add() { assert_eq!(add(1, 2), 3); }".to_string(),
        token_match_percent: 100.0,
    });

    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let frame = rendered_to_frame(&rendered);

    // Should show 100% match
    assert!(frame.as_text().contains("100%"), "Should show 100% token match");

    let tui_snap = TuiSnapshot::from_frame("perfect_match", &frame)
        .with_metadata("edge_case", "100_percent_match");
    assert!(!tui_snap.content.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROBAR FRAME COMPARISON TESTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Test frame diff functionality
#[test]
fn test_probar_frame_diff() {
    let snapshot1 = mock_snapshot(1, 1, 6.9);
    let snapshot2 = mock_snapshot(1, 8, 5.0);

    let rendered1 = render_layout_colored(&snapshot1, 80, ColorMode::Mono);
    let rendered2 = render_layout_colored(&snapshot2, 80, ColorMode::Mono);

    let frame1 = rendered_to_frame(&rendered1);
    let frame2 = rendered_to_frame(&rendered2);

    // Frames should differ
    assert!(!frame1.is_identical(&frame2), "Different snapshots should produce different frames");

    // Get the diff
    let diff = frame1.diff(&frame2);
    assert!(!diff.is_identical, "Diff should indicate frames are different");
    assert!(!diff.changed_lines.is_empty(), "Should have changed lines");
}

/// Test frame content assertions
#[test]
fn test_probar_frame_content_assertions() {
    let snapshot = mock_snapshot(5, 8, 4.5);
    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let frame = rendered_to_frame(&rendered);

    // Test contains
    assert!(frame.contains("ENTRENAR"), "Should contain ENTRENAR");
    assert!(frame.contains("Loss"), "Should contain Loss");
    assert!(frame.contains("GPU"), "Should contain GPU");
    assert!(frame.contains("Config"), "Should contain Config");

    // Test regex matching
    assert!(
        frame.matches(r"Epoch.*\d+/\d+").expect("operation should succeed"),
        "Should match epoch pattern"
    );
    assert!(
        frame.matches(r"Step.*\d+/\d+").expect("operation should succeed"),
        "Should match step pattern"
    );
}

/// Test TuiSnapshot hash consistency
#[test]
fn test_probar_snapshot_hash_consistency() {
    let snapshot = mock_snapshot(5, 8, 4.5);
    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let frame = rendered_to_frame(&rendered);

    // Create two snapshots from the same frame
    let tui_snap1 = TuiSnapshot::from_frame("test1", &frame);
    let tui_snap2 = TuiSnapshot::from_frame("test2", &frame);

    // Hashes should match (content is identical)
    assert_eq!(tui_snap1.hash, tui_snap2.hash, "Same content should produce same hash");
    assert!(tui_snap1.matches(&tui_snap2), "Snapshots with same content should match");
}

/// Test TuiSnapshot metadata
#[test]
fn test_probar_snapshot_metadata() {
    let snapshot = mock_snapshot(5, 8, 4.5);
    let rendered = render_layout_colored(&snapshot, 80, ColorMode::Mono);
    let frame = rendered_to_frame(&rendered);

    let tui_snap = TuiSnapshot::from_frame("test", &frame)
        .with_metadata("version", "0.5.6")
        .with_metadata("test_type", "unit")
        .with_metadata("color_mode", "mono");

    assert_eq!(tui_snap.metadata.get("version"), Some(&"0.5.6".to_string()));
    assert_eq!(tui_snap.metadata.get("test_type"), Some(&"unit".to_string()));
    assert_eq!(tui_snap.metadata.get("color_mode"), Some(&"mono".to_string()));
}
