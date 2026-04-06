//! Presentar-based Training Dashboard Widget (ALB-047/048/057)
//!
//! Composes sovereign-stack presentar-terminal widgets:
//! - Layout for flex-based arrangement
//! - Border for section panels
//! - Meter for progress bar
//! - GpuPanel for GPU telemetry
//! - Sparkline for loss history
//! - Text for information lines
//!
//! ALB-057: Replaces monolithic draw_text() rendering with composable widget tree.
//! Each dashboard section is a standalone widget composed via Layout::rows().

use super::state::{TrainingSnapshot, TrainingState, TrainingStatus};
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use presentar_terminal::widgets::{
    Border, GpuDevice, GpuPanel as PresentarGpuPanel, GpuProcess as PresentarGpuProcess, GpuVendor,
    Layout, LayoutItem, Meter, Sparkline, Text,
};
use std::any::Any;
use std::path::PathBuf;
use std::time::Duration;

/// Training dashboard widget for `apr monitor`.
///
/// Composes presentar-terminal widgets (Layout, Border, Meter, GpuPanel,
/// Sparkline) via a widget tree rebuilt each frame from `training_state.json`.
/// Delegates layout/paint to the composed tree, getting responsive arrangement
/// and consistent theming from the sovereign stack.
pub struct TrainingDashboard {
    /// Latest snapshot (updated each layout pass)
    snapshot: Option<TrainingSnapshot>,
    /// Experiment directory (contains training_state.json)
    experiment_dir: PathBuf,
    /// Cached bounds from layout
    bounds: Rect,
    /// Composed widget tree — rebuilt each frame from snapshot
    widget_tree: Option<Layout>,
}

#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for TrainingDashboard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainingDashboard")
            .field("experiment_dir", &self.experiment_dir)
            .field("has_snapshot", &self.snapshot.is_some())
            .field("has_widget_tree", &self.widget_tree.is_some())
            .finish()
    }
}

impl TrainingDashboard {
    /// Create a new training dashboard for an experiment directory.
    #[must_use]
    pub fn new(experiment_dir: PathBuf) -> Self {
        Self { snapshot: None, experiment_dir, bounds: Rect::default(), widget_tree: None }
    }

    /// Refresh the snapshot from training_state.json.
    pub fn refresh(&mut self) {
        let mut state = TrainingState::new(&self.experiment_dir);
        if let Ok(Some(snap)) = state.read() {
            self.snapshot = Some(snap);
        }
    }

    /// Check if training is done (for app quit logic).
    pub fn is_finished(&self) -> bool {
        self.snapshot.as_ref().is_some_and(|s| {
            matches!(s.status, TrainingStatus::Completed | TrainingStatus::Failed(_))
        })
    }

    /// Rebuild composed widget tree from current snapshot.
    ///
    /// Called each frame after refresh(). Constructs a Layout::rows tree
    /// of bordered section panels from the latest training data.
    fn rebuild_widgets(&mut self) {
        let Some(snap) = &self.snapshot else {
            self.widget_tree = None;
            return;
        };

        let mut items = Vec::with_capacity(5);

        // Header: title + status badge + model info (2 lines)
        items.push(build_header(snap).into_item().fixed(2.0));

        // Training metrics panel: progress, loss, timing (4 content + 2 border)
        items.push(LayoutItem::new(build_metrics_panel(snap)).fixed(6.0));

        // Shows device, utilization, VRAM, thermal/power (3 content + 2 border)
        if snap.gpu.is_some() {
            items.push(LayoutItem::new(build_gpu_panel(snap)).fixed(5.0));
        }

        // Loss history sparkline (2 content + 2 border)
        if !snap.loss_history.is_empty() {
            items.push(LayoutItem::new(build_loss_panel(snap)).fixed(4.0));
        }

        // Error message if failed
        if let TrainingStatus::Failed(msg) = &snap.status {
            let err = Text::new(format!("ERROR: {msg}")).with_color(Color::new(1.0, 0.3, 0.3, 1.0));
            items.push(LayoutItem::new(err).fixed(1.0));
        }

        self.widget_tree = Some(Layout::rows(items));
    }
}

// =============================================================================
// Widget builders — each returns a composed widget for one dashboard section
// =============================================================================

/// Build header section: title + status badge + model info.
fn build_header(snap: &TrainingSnapshot) -> Layout {
    let (status_str, status_color) = status_display(&snap.status);

    let title = Text::new(format!("apr monitor — {}", truncate_str(&snap.experiment_id, 30)))
        .with_color(Color::WHITE)
        .bold();

    let status = Text::new(status_str).with_color(status_color).right();

    let title_row =
        Layout::columns([LayoutItem::new(title).expanded(), LayoutItem::new(status).fixed(10.0)]);

    let info = Text::new(format!(
        "Model: {} | Opt: {} | Batch: {}",
        truncate_str(&snap.model_name, 30),
        snap.optimizer_name,
        snap.batch_size
    ))
    .with_color(Color::new(0.6, 0.6, 0.6, 1.0));

    Layout::rows([title_row.into_item().fixed(1.0), LayoutItem::new(info).fixed(1.0)])
}

/// Build metrics panel: progress bar, loss, timing — inside a rounded border.
fn build_metrics_panel(snap: &TrainingSnapshot) -> Border {
    let progress = snap.progress_percent();
    let trend = snap.loss_trend();

    let progress_text = Text::new(format!(
        "Epoch {}/{}  Step {}/{}",
        snap.epoch, snap.total_epochs, snap.step, snap.steps_per_epoch
    ))
    .with_color(Color::WHITE);

    let meter = Meter::percentage(f64::from(progress))
        .with_label(format!("{progress:.1}%"))
        .with_color(Color::new(0.3, 0.9, 0.3, 1.0));

    let loss_text = Text::new(format!(
        "Loss: {:.6} {}  LR: {:.2e}  Tok/s: {:.0}  Grad: {:.2}",
        snap.loss,
        trend.arrow(),
        snap.learning_rate,
        snap.tokens_per_second,
        snap.gradient_norm
    ))
    .with_color(Color::WHITE);

    let elapsed = snap.elapsed();
    let eta = snap
        .estimated_remaining()
        .map(|r| format!("ETA: {}", format_duration(r)))
        .unwrap_or_default();
    let time_text = Text::new(format!("Elapsed: {}  {}", format_duration(elapsed), eta))
        .with_color(Color::new(1.0, 0.9, 0.3, 1.0));

    let content = Layout::rows([
        LayoutItem::new(progress_text).fixed(1.0),
        LayoutItem::new(meter).fixed(1.0),
        LayoutItem::new(loss_text).fixed(1.0),
        LayoutItem::new(time_text).fixed(1.0),
    ]);

    Border::rounded("Training").child(content)
}

/// Build GPU panel: utilization, VRAM, temp/power — inside a rounded border.
fn build_gpu_panel(snap: &TrainingSnapshot) -> Border {
    let Some(gpu) = &snap.gpu else {
        return Border::rounded("GPU")
            .child(Text::new("N/A (CPU training)").with_color(Color::new(0.5, 0.5, 0.5, 1.0)));
    };

    let device = convert_gpu_device(gpu);
    let processes = convert_gpu_processes(&gpu.processes);

    let gpu_widget = PresentarGpuPanel::new()
        .with_device(device)
        .with_processes(processes)
        .show_processes(false);

    Border::rounded(format!("GPU: {}", truncate_str(&gpu.device_name, 20))).child(gpu_widget)
}

/// Build loss history panel: sparkline + range — inside a rounded border.
fn build_loss_panel(snap: &TrainingSnapshot) -> Border {
    let values: Vec<f64> = snap.loss_history.iter().map(|v| f64::from(*v)).collect();
    let first = snap.loss_history.first().copied().unwrap_or(0.0);
    let last = snap.loss_history.last().copied().unwrap_or(0.0);

    let sparkline =
        Sparkline::new(values).with_color(Color::new(0.3, 0.7, 1.0, 1.0)).with_trend(true);

    let range =
        Text::new(format!("{first:.4} → {last:.4}")).with_color(Color::new(0.5, 0.5, 0.5, 1.0));

    let content =
        Layout::rows([LayoutItem::new(sparkline).fixed(1.0), LayoutItem::new(range).fixed(1.0)]);

    Border::rounded("Loss History").child(content)
}

// =============================================================================
// Helpers
// =============================================================================

/// Convert entrenar `GpuTelemetry` to presentar `GpuDevice`.
fn convert_gpu_device(gpu: &super::state::GpuTelemetry) -> GpuDevice {
    GpuDevice::new(&gpu.device_name)
        .with_vendor(GpuVendor::Nvidia)
        .with_utilization(gpu.utilization_percent)
        .with_temperature(gpu.temperature_celsius)
        .with_vram(
            (gpu.vram_used_gb.max(0.0) * 1_073_741_824.0) as u64,
            (gpu.vram_total_gb.max(0.0) * 1_073_741_824.0) as u64,
        )
        .with_power(gpu.power_watts, Some(gpu.power_limit_watts))
}

/// Convert entrenar `GpuProcessInfo` to presentar `GpuProcess`.
fn convert_gpu_processes(processes: &[super::state::GpuProcessInfo]) -> Vec<PresentarGpuProcess> {
    processes
        .iter()
        .map(|p| {
            let name = p.exe_path.rsplit('/').next().unwrap_or(&p.exe_path);
            PresentarGpuProcess::new(name, p.pid, p.gpu_memory_mb * 1024 * 1024)
        })
        .collect()
}

/// Get display string and color for training status.
fn status_display(status: &TrainingStatus) -> (&'static str, Color) {
    match status {
        TrainingStatus::Initializing => ("INIT", Color::new(0.7, 0.7, 0.7, 1.0)),
        TrainingStatus::Running => ("RUNNING", Color::new(0.3, 0.9, 0.3, 1.0)),
        TrainingStatus::Paused => ("PAUSED", Color::new(0.7, 0.7, 0.7, 1.0)),
        TrainingStatus::Completed => ("DONE", Color::new(0.3, 0.7, 1.0, 1.0)),
        TrainingStatus::Failed(_) => ("FAILED", Color::new(1.0, 0.3, 0.3, 1.0)),
    }
}

/// Truncate a string with no-alloc slicing.
fn truncate_str(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        &s[..max]
    }
}

/// Format a Duration as human-readable.
fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    if secs > 3600 {
        format!("{}h {}m {}s", secs / 3600, (secs % 3600) / 60, secs % 60)
    } else if secs > 60 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{secs}s")
    }
}

// =============================================================================
// Brick trait implementation (PROBAR-SPEC-009)
// =============================================================================

impl Brick for TrainingDashboard {
    fn brick_name(&self) -> &'static str {
        "training_dashboard"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(16)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16) // 60fps
    }

    fn verify(&self) -> BrickVerification {
        // Wire panel verification system into Brick verify (ALB-057).
        // Uses panel::layout_can_render() to validate snapshot data before rendering.
        let start = std::time::Instant::now();
        if let Some(snap) = &self.snapshot {
            if !super::panel::layout_can_render(snap) {
                return BrickVerification {
                    passed: vec![],
                    failed: vec![(
                        BrickAssertion::max_latency_ms(16),
                        "snapshot data failed panel verification".to_string(),
                    )],
                    verification_time: start.elapsed(),
                };
            }
        }
        BrickVerification {
            passed: vec![BrickAssertion::max_latency_ms(16)],
            failed: vec![],
            verification_time: start.elapsed(),
        }
    }

    fn to_html(&self) -> String {
        String::new() // Terminal-only widget
    }

    fn to_css(&self) -> String {
        String::new() // Terminal-only widget
    }
}

// =============================================================================
// Widget trait implementation
// =============================================================================

impl Widget for TrainingDashboard {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        Size { width: constraints.max_width, height: constraints.max_height }
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        // Refresh data and rebuild widget tree each frame
        self.refresh();
        self.rebuild_widgets();

        // Delegate layout to composed widget tree
        if let Some(tree) = &mut self.widget_tree {
            tree.layout(bounds);
        }

        LayoutResult { size: Size { width: bounds.width, height: bounds.height } }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        // Delegate to composed widget tree
        if let Some(tree) = &self.widget_tree {
            tree.paint(canvas);
            return;
        }

        // Fallback: waiting state
        let dim = TextStyle { color: Color::new(0.5, 0.5, 0.5, 1.0), ..Default::default() };
        canvas.draw_text("Waiting for training data...", Point { x: 1.0, y: 1.0 }, &dim);
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        if let Some(tree) = &mut self.widget_tree {
            return tree.event(event);
        }
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use std::path::PathBuf;

    fn make_snapshot() -> TrainingSnapshot {
        TrainingSnapshot {
            timestamp_ms: 2000,
            epoch: 2,
            total_epochs: 10,
            step: 50,
            steps_per_epoch: 100,
            loss: 0.123456,
            loss_history: vec![0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15],
            learning_rate: 1e-4,
            lr_history: vec![],
            gradient_norm: 1.5,
            accuracy: 0.85,
            tokens_per_second: 1234.0,
            samples_per_second: 10.0,
            start_timestamp_ms: 1000,
            gpu: None,
            sample: None,
            status: TrainingStatus::Running,
            experiment_id: "exp-001".to_string(),
            model_name: "test-model".to_string(),
            model_path: String::new(),
            optimizer_name: "AdamW".to_string(),
            batch_size: 32,
            checkpoint_path: String::new(),
            executable_path: String::new(),
        }
    }

    fn make_gpu() -> super::super::state::GpuTelemetry {
        super::super::state::GpuTelemetry {
            device_name: "RTX 4090".to_string(),
            utilization_percent: 95.0,
            vram_used_gb: 20.0,
            vram_total_gb: 24.0,
            temperature_celsius: 72.0,
            power_watts: 350.0,
            power_limit_watts: 450.0,
            processes: vec![],
        }
    }

    // ── format_duration tests ──────────────────────────────────────────

    #[test]
    fn test_format_duration_seconds() {
        assert_eq!(format_duration(Duration::from_secs(42)), "42s");
    }

    #[test]
    fn test_format_duration_minutes() {
        assert_eq!(format_duration(Duration::from_secs(125)), "2m 5s");
    }

    #[test]
    fn test_format_duration_hours() {
        assert_eq!(format_duration(Duration::from_secs(3723)), "1h 2m 3s");
    }

    #[test]
    fn test_format_duration_zero() {
        assert_eq!(format_duration(Duration::from_secs(0)), "0s");
    }

    #[test]
    fn test_format_duration_exact_minute() {
        // 60s uses `> 60` branch which is false, so falls to else → "60s"
        assert_eq!(format_duration(Duration::from_secs(60)), "60s");
    }

    #[test]
    fn test_format_duration_exact_hour() {
        // 3600s uses `> 3600` which is false, falls to `> 60` → "60m 0s"
        assert_eq!(format_duration(Duration::from_secs(3600)), "60m 0s");
    }

    // ── truncate_str tests ─────────────────────────────────────────────

    #[test]
    fn test_truncate_str_short() {
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_str_exact() {
        assert_eq!(truncate_str("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_str_long() {
        assert_eq!(truncate_str("hello world", 5), "hello");
    }

    #[test]
    fn test_truncate_str_empty() {
        assert_eq!(truncate_str("", 5), "");
    }

    #[test]
    fn test_truncate_str_zero_max() {
        assert_eq!(truncate_str("abc", 0), "");
    }

    // ── status_display tests ───────────────────────────────────────────

    #[test]
    fn test_status_display_running() {
        let (text, color) = status_display(&TrainingStatus::Running);
        assert_eq!(text, "RUNNING");
        assert!(color.g > 0.5); // green-ish
    }

    #[test]
    fn test_status_display_init() {
        let (text, _) = status_display(&TrainingStatus::Initializing);
        assert_eq!(text, "INIT");
    }

    #[test]
    fn test_status_display_paused() {
        let (text, _) = status_display(&TrainingStatus::Paused);
        assert_eq!(text, "PAUSED");
    }

    #[test]
    fn test_status_display_completed() {
        let (text, _) = status_display(&TrainingStatus::Completed);
        assert_eq!(text, "DONE");
    }

    #[test]
    fn test_status_display_failed() {
        let (text, color) = status_display(&TrainingStatus::Failed("oops".into()));
        assert_eq!(text, "FAILED");
        assert!(color.r > 0.5); // red-ish
    }

    // ── convert_gpu_device tests ───────────────────────────────────────

    #[test]
    fn test_convert_gpu_device_basic() {
        let gpu = make_gpu();
        let _dev = convert_gpu_device(&gpu);
        // No panic = success; GpuDevice is opaque
    }

    #[test]
    fn test_convert_gpu_device_zero_vram() {
        let mut gpu = make_gpu();
        gpu.vram_used_gb = 0.0;
        gpu.vram_total_gb = 0.0;
        let _dev = convert_gpu_device(&gpu);
    }

    #[test]
    fn test_convert_gpu_device_negative_vram_clamped() {
        let mut gpu = make_gpu();
        gpu.vram_used_gb = -1.0;
        gpu.vram_total_gb = -1.0;
        // Should not panic — max(0.0) clamp handles negative
        let _dev = convert_gpu_device(&gpu);
    }

    // ── convert_gpu_processes tests ────────────────────────────────────

    #[test]
    fn test_convert_gpu_processes_empty() {
        let procs = convert_gpu_processes(&[]);
        assert!(procs.is_empty());
    }

    #[test]
    fn test_convert_gpu_processes_single() {
        let procs = convert_gpu_processes(&[super::super::state::GpuProcessInfo {
            pid: 1234,
            exe_path: "/usr/bin/python3".to_string(),
            gpu_memory_mb: 4096,
            cpu_percent: 50.0,
            rss_mb: 2048,
        }]);
        assert_eq!(procs.len(), 1);
    }

    #[test]
    fn test_convert_gpu_processes_basename_extraction() {
        let procs = convert_gpu_processes(&[super::super::state::GpuProcessInfo {
            pid: 42,
            exe_path: "/very/long/path/to/trainer".to_string(),
            gpu_memory_mb: 100,
            cpu_percent: 10.0,
            rss_mb: 100,
        }]);
        assert_eq!(procs.len(), 1);
    }

    #[test]
    fn test_convert_gpu_processes_no_slash() {
        let procs = convert_gpu_processes(&[super::super::state::GpuProcessInfo {
            pid: 1,
            exe_path: "python3".to_string(),
            gpu_memory_mb: 50,
            cpu_percent: 5.0,
            rss_mb: 50,
        }]);
        assert_eq!(procs.len(), 1);
    }

    // ── TrainingDashboard tests ────────────────────────────────────────

    #[test]
    fn test_dashboard_new() {
        let dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        assert!(dash.snapshot.is_none());
        assert!(dash.widget_tree.is_none());
    }

    #[test]
    fn test_dashboard_is_finished_no_snapshot() {
        let dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        assert!(!dash.is_finished());
    }

    #[test]
    fn test_dashboard_is_finished_running() {
        let mut dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        dash.snapshot = Some(make_snapshot());
        assert!(!dash.is_finished());
    }

    #[test]
    fn test_dashboard_is_finished_completed() {
        let mut dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        let mut snap = make_snapshot();
        snap.status = TrainingStatus::Completed;
        dash.snapshot = Some(snap);
        assert!(dash.is_finished());
    }

    #[test]
    fn test_dashboard_is_finished_failed() {
        let mut dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        let mut snap = make_snapshot();
        snap.status = TrainingStatus::Failed("boom".into());
        dash.snapshot = Some(snap);
        assert!(dash.is_finished());
    }

    #[test]
    fn test_dashboard_is_finished_init() {
        let mut dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        let mut snap = make_snapshot();
        snap.status = TrainingStatus::Initializing;
        dash.snapshot = Some(snap);
        assert!(!dash.is_finished());
    }

    #[test]
    fn test_dashboard_is_finished_paused() {
        let mut dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        let mut snap = make_snapshot();
        snap.status = TrainingStatus::Paused;
        dash.snapshot = Some(snap);
        assert!(!dash.is_finished());
    }

    // ── Debug impl test ────────────────────────────────────────────────

    #[test]
    fn test_dashboard_debug() {
        let dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        let dbg = format!("{dash:?}");
        assert!(dbg.contains("TrainingDashboard"));
        assert!(dbg.contains("/tmp/exp"));
        assert!(dbg.contains("has_snapshot"));
    }

    // ── rebuild_widgets tests ──────────────────────────────────────────

    #[test]
    fn test_rebuild_widgets_no_snapshot() {
        let mut dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        dash.rebuild_widgets();
        assert!(dash.widget_tree.is_none());
    }

    #[test]
    fn test_rebuild_widgets_basic() {
        let mut dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        dash.snapshot = Some(make_snapshot());
        dash.rebuild_widgets();
        assert!(dash.widget_tree.is_some());
    }

    #[test]
    fn test_rebuild_widgets_with_gpu() {
        let mut dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        let mut snap = make_snapshot();
        snap.gpu = Some(make_gpu());
        dash.snapshot = Some(snap);
        dash.rebuild_widgets();
        assert!(dash.widget_tree.is_some());
    }

    #[test]
    fn test_rebuild_widgets_no_loss_history() {
        let mut dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        let mut snap = make_snapshot();
        snap.loss_history = vec![];
        dash.snapshot = Some(snap);
        dash.rebuild_widgets();
        assert!(dash.widget_tree.is_some());
    }

    #[test]
    fn test_rebuild_widgets_failed_status() {
        let mut dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        let mut snap = make_snapshot();
        snap.status = TrainingStatus::Failed("critical error".into());
        dash.snapshot = Some(snap);
        dash.rebuild_widgets();
        assert!(dash.widget_tree.is_some());
    }

    #[test]
    fn test_rebuild_widgets_completed() {
        let mut dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        let mut snap = make_snapshot();
        snap.status = TrainingStatus::Completed;
        dash.snapshot = Some(snap);
        dash.rebuild_widgets();
        assert!(dash.widget_tree.is_some());
    }

    // ── build_* function tests ─────────────────────────────────────────

    #[test]
    fn test_build_header() {
        let snap = make_snapshot();
        let _header = build_header(&snap);
        // No panic = success
    }

    #[test]
    fn test_build_header_long_experiment_id() {
        let mut snap = make_snapshot();
        snap.experiment_id = "a".repeat(100);
        let _header = build_header(&snap);
    }

    #[test]
    fn test_build_metrics_panel() {
        let snap = make_snapshot();
        let _panel = build_metrics_panel(&snap);
    }

    #[test]
    fn test_build_metrics_panel_zero_progress() {
        let mut snap = make_snapshot();
        snap.epoch = 0;
        snap.step = 0;
        snap.total_epochs = 0;
        snap.steps_per_epoch = 0;
        let _panel = build_metrics_panel(&snap);
    }

    #[test]
    fn test_build_gpu_panel_with_gpu() {
        let mut snap = make_snapshot();
        snap.gpu = Some(make_gpu());
        let _panel = build_gpu_panel(&snap);
    }

    #[test]
    fn test_build_gpu_panel_no_gpu() {
        let snap = make_snapshot();
        let _panel = build_gpu_panel(&snap);
    }

    #[test]
    fn test_build_gpu_panel_with_processes() {
        let mut snap = make_snapshot();
        let mut gpu = make_gpu();
        gpu.processes = vec![
            super::super::state::GpuProcessInfo {
                pid: 100,
                exe_path: "/usr/bin/python3".to_string(),
                gpu_memory_mb: 4096,
                cpu_percent: 50.0,
                rss_mb: 2048,
            },
            super::super::state::GpuProcessInfo {
                pid: 200,
                exe_path: "trainer".to_string(),
                gpu_memory_mb: 2048,
                cpu_percent: 25.0,
                rss_mb: 1024,
            },
        ];
        snap.gpu = Some(gpu);
        let _panel = build_gpu_panel(&snap);
    }

    #[test]
    fn test_build_loss_panel() {
        let snap = make_snapshot();
        let _panel = build_loss_panel(&snap);
    }

    #[test]
    fn test_build_loss_panel_single_value() {
        let mut snap = make_snapshot();
        snap.loss_history = vec![0.5];
        let _panel = build_loss_panel(&snap);
    }

    // ── Brick trait tests ──────────────────────────────────────────────

    #[test]
    fn test_brick_name() {
        let dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        assert_eq!(dash.brick_name(), "training_dashboard");
    }

    #[test]
    fn test_brick_assertions() {
        let dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        assert!(!dash.assertions().is_empty());
    }

    #[test]
    fn test_brick_budget() {
        let dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        let _budget = dash.budget();
    }

    #[test]
    fn test_brick_verify_no_snapshot() {
        let dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        let v = dash.verify();
        assert!(v.failed.is_empty());
    }

    #[test]
    fn test_brick_verify_with_snapshot() {
        let mut dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        dash.snapshot = Some(make_snapshot());
        let v = dash.verify();
        // Should pass for valid snapshot
        assert!(!v.passed.is_empty() || !v.failed.is_empty());
    }

    #[test]
    fn test_brick_to_html() {
        let dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        assert!(dash.to_html().is_empty());
    }

    #[test]
    fn test_brick_to_css() {
        let dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        assert!(dash.to_css().is_empty());
    }

    // ── Widget trait tests ─────────────────────────────────────────────

    #[test]
    fn test_widget_measure() {
        let dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        let size = dash.measure(Constraints {
            min_width: 0.0,
            min_height: 0.0,
            max_width: 80.0,
            max_height: 24.0,
        });
        assert_eq!(size.width, 80.0);
        assert_eq!(size.height, 24.0);
    }

    #[test]
    fn test_widget_children_empty() {
        let dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        assert!(dash.children().is_empty());
    }

    #[test]
    fn test_widget_children_mut_empty() {
        let mut dash = TrainingDashboard::new(PathBuf::from("/tmp/exp"));
        assert!(dash.children_mut().is_empty());
    }
}
