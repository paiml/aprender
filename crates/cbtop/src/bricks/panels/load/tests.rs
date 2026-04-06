use super::*;
use presentar_core::RecordingCanvas;

#[test]
fn test_load_control_brick_name() {
    let panel = LoadControlPanelBrick::new();
    assert_eq!(panel.brick_name(), "load_control_panel");
}

#[test]
fn test_load_control_paint_stopped() {
    let panel = LoadControlPanelBrick::new();
    let mut canvas = RecordingCanvas::new();

    panel.paint(&mut canvas, 80.0, 24.0);

    // Should draw header, controls, and help text
    assert!(!canvas.is_empty());
    assert!(canvas.command_count() >= 10);
}

#[test]
fn test_load_control_paint_running() {
    let mut panel = LoadControlPanelBrick::new();
    panel.toggle_running();

    let mut canvas = RecordingCanvas::new();
    panel.paint(&mut canvas, 80.0, 24.0);

    // Should draw RUNNING status
    assert!(!canvas.is_empty());
}

#[test]
fn test_load_control_paint_with_stats() {
    let mut panel = LoadControlPanelBrick::new();
    panel.toggle_running();
    panel.update_stats(LoadStats {
        iterations: 1000,
        elapsed_ms: 5000,
        ops_per_sec: 200.0,
        throughput_gbs: 1.5,
        avg_latency_us: 500.0,
        p99_latency_us: 1200.0,
    });

    let mut canvas = RecordingCanvas::new();
    panel.paint(&mut canvas, 80.0, 24.0);

    // Should draw stats section
    assert!(canvas.command_count() >= 15);
}

#[test]
fn test_load_control_paint_with_error() {
    let mut panel = LoadControlPanelBrick::new();
    panel.set_error("GPU not available".to_string());

    let mut canvas = RecordingCanvas::new();
    panel.paint(&mut canvas, 80.0, 24.0);

    // Should draw error message
    assert!(!canvas.is_empty());
}

#[test]
fn test_load_control_paint_with_score() {
    let mut panel = LoadControlPanelBrick::new();
    let score = BrickScore::new(38, 22, 20, 14);
    panel.update_score(score, 27.92);

    let mut canvas = RecordingCanvas::new();
    panel.paint(&mut canvas, 80.0, 24.0);

    // Should draw ComputeBrick score section
    assert!(canvas.command_count() >= 15);
}

#[test]
fn test_load_control_paint_different_selections() {
    let mut panel = LoadControlPanelBrick::new();

    // Test with each menu item selected
    for item in 0..5 {
        panel.selected_item = item;
        let mut canvas = RecordingCanvas::new();
        panel.paint(&mut canvas, 80.0, 24.0);
        assert!(!canvas.is_empty());
    }
}

#[test]
fn test_load_control_has_assertions() {
    let panel = LoadControlPanelBrick::new();
    assert!(!panel.assertions().is_empty());
}

#[test]
fn test_backend_cycling() {
    let mut panel = LoadControlPanelBrick::new();
    assert_eq!(panel.backend, ComputeBackend::Auto);

    panel.next_backend();
    assert_eq!(panel.backend, ComputeBackend::CpuScalar);

    panel.next_backend();
    assert_eq!(panel.backend, ComputeBackend::CpuSimd);

    panel.prev_backend();
    assert_eq!(panel.backend, ComputeBackend::CpuScalar);
}

#[test]
fn test_workload_cycling() {
    let mut panel = LoadControlPanelBrick::new();
    assert_eq!(panel.workload, WorkloadType::Gemm);

    panel.next_workload();
    assert_eq!(panel.workload, WorkloadType::Softmax);

    panel.prev_workload();
    assert_eq!(panel.workload, WorkloadType::Gemm);
}

#[test]
fn test_intensity_bounds() {
    let mut panel = LoadControlPanelBrick::new();
    panel.intensity = 0.0;

    panel.decrease_intensity();
    assert_eq!(panel.intensity, 0.0);

    panel.intensity = 100.0;
    panel.increase_intensity();
    assert_eq!(panel.intensity, 100.0);
}

#[test]
fn test_size_bounds() {
    let mut panel = LoadControlPanelBrick::new();
    panel.problem_size = 64;

    panel.decrease_size();
    assert_eq!(panel.problem_size, 64);

    panel.problem_size = 65536;
    panel.increase_size();
    assert_eq!(panel.problem_size, 65536);
}

#[test]
fn test_toggle_running() {
    let mut panel = LoadControlPanelBrick::new();
    assert!(!panel.is_running);

    panel.toggle_running();
    assert!(panel.is_running);

    panel.toggle_running();
    assert!(!panel.is_running);
}

#[test]
fn test_navigation() {
    let mut panel = LoadControlPanelBrick::new();
    assert_eq!(panel.selected_item, 0);

    panel.next_item();
    assert_eq!(panel.selected_item, 1);

    panel.prev_item();
    assert_eq!(panel.selected_item, 0);

    panel.prev_item();
    assert_eq!(panel.selected_item, 4);
}

#[test]
fn test_handle_keys() {
    let mut panel = LoadControlPanelBrick::new();

    // Test backend selection
    panel.selected_item = 0;
    panel.handle_right();
    assert_eq!(panel.backend, ComputeBackend::CpuScalar);

    // Test intensity
    panel.selected_item = 2;
    panel.intensity = 50.0;
    panel.handle_right();
    assert_eq!(panel.intensity, 55.0);

    // Test enter on start/stop
    panel.selected_item = 4;
    assert!(!panel.is_running);
    panel.handle_enter();
    assert!(panel.is_running);
}

#[test]
fn test_set_error() {
    let mut panel = LoadControlPanelBrick::new();
    panel.is_running = true;

    panel.set_error("GPU not available".to_string());

    assert!(!panel.is_running);
    assert_eq!(panel.error, Some("GPU not available".to_string()));
}

#[test]
fn test_update_stats() {
    let mut panel = LoadControlPanelBrick::new();

    let stats = LoadStats {
        iterations: 1000,
        elapsed_ms: 5000,
        ops_per_sec: 200.0,
        throughput_gbs: 1.5,
        avg_latency_us: 500.0,
        p99_latency_us: 1200.0,
    };

    panel.update_stats(stats.clone());

    assert_eq!(panel.stats.iterations, 1000);
    assert_eq!(panel.stats.ops_per_sec, 200.0);
    assert_eq!(panel.stats.throughput_gbs, 1.5);
}

#[test]
fn test_format_size() {
    assert_eq!(LoadControlPanelBrick::format_size(512), "512");
    assert_eq!(LoadControlPanelBrick::format_size(1024), "1K");
    assert_eq!(LoadControlPanelBrick::format_size(4096), "4K");
    assert_eq!(LoadControlPanelBrick::format_size(65536), "64K");
}

#[test]
fn test_backend_names() {
    assert_eq!(ComputeBackend::Auto.name(), "Auto");
    assert_eq!(ComputeBackend::GpuCuda.name(), "GPU (CUDA)");
}

#[test]
fn test_workload_names() {
    assert_eq!(WorkloadType::Gemm.name(), "GEMM (Matrix Multiply)");
    assert_eq!(WorkloadType::Gemm.short_name(), "GEMM");
}

#[test]
fn test_update_score() {
    let mut panel = LoadControlPanelBrick::new();
    assert!(panel.brick_score.is_none());
    assert_eq!(panel.gflops, 0.0);

    let score = BrickScore::new(38, 22, 20, 14);
    panel.update_score(score, 27.92);

    assert!(panel.brick_score.is_some());
    assert_eq!(panel.brick_score.unwrap().total(), 94);
    assert!((panel.gflops - 27.92).abs() < 0.01);
}

#[test]
fn test_score_rendering() {
    let bar = LoadControlPanelBrick::render_score_bar(20, 40, 10);
    assert_eq!(bar.chars().filter(|c| *c == '█').count(), 5);
    assert_eq!(bar.chars().filter(|c| *c == '░').count(), 5);
}

#[test]
fn test_new_panel_has_no_score() {
    let panel = LoadControlPanelBrick::new();
    assert!(panel.brick_score.is_none());
    assert_eq!(panel.gflops, 0.0);
}
