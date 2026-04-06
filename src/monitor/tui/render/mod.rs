//! TUI Rendering - Clean, Simple, Labeled Layout
//!
//! Design: Tufte data-ink ratio, clear labels, no chartjunk

mod bars;
mod charts;
mod epoch;
mod format;

pub use bars::{
    build_block_bar, build_colored_block_bar, pct_color, render_sparkline, trend_arrow,
};
pub use charts::{
    render_braille_chart, render_config_panel, render_gauge, render_history_table,
    render_sample_panel, BrailleChart,
};
pub use epoch::{compute_epoch_summaries, EpochSummary};
pub use format::{format_bytes, format_duration, format_lr};

use super::color::{ColorMode, Styled, TrainingPalette};
use super::state::{TrainingSnapshot, TrainingStatus};

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN LAYOUT - Simple, Clean, Labeled
// ═══════════════════════════════════════════════════════════════════════════════

pub fn render_layout(snapshot: &TrainingSnapshot, width: usize) -> String {
    render_layout_colored(snapshot, width, ColorMode::detect())
}

pub fn render_layout_colored(
    snapshot: &TrainingSnapshot,
    width: usize,
    color_mode: ColorMode,
) -> String {
    let mut lines: Vec<String> = Vec::new();
    let w = width.max(80);

    render_header(&mut lines, snapshot, w, color_mode);
    render_progress(&mut lines, snapshot, w, color_mode);
    render_metrics(&mut lines, snapshot, color_mode);
    render_loss_sparkline(&mut lines, snapshot, w, color_mode);
    render_gpu_section(&mut lines, snapshot, w, color_mode);
    render_epoch_table(&mut lines, snapshot, w, color_mode);
    render_config_footer(&mut lines, snapshot, w, color_mode);

    lines.join("\n")
}

fn render_header(
    lines: &mut Vec<String>,
    snapshot: &TrainingSnapshot,
    w: usize,
    color_mode: ColorMode,
) {
    let (status_icon, status_text, status_color) = match &snapshot.status {
        TrainingStatus::Initializing => ("\u{25D0}", "Init", TrainingPalette::INFO),
        TrainingStatus::Running => ("\u{25CF}", "Running", TrainingPalette::SUCCESS),
        TrainingStatus::Paused => ("\u{25D0}", "Paused", TrainingPalette::WARNING),
        TrainingStatus::Completed => ("\u{25CB}", "Done", TrainingPalette::PRIMARY),
        TrainingStatus::Failed(_) => ("\u{25CF}", "FAIL", TrainingPalette::ERROR),
    };

    let elapsed = format_duration(snapshot.elapsed());
    let tps = snapshot.tokens_per_second.max(0.0);

    let model_display = if snapshot.model_name.is_empty() {
        "N/A".to_string()
    } else if snapshot.model_name.len() > 30 {
        format!("{}...", &snapshot.model_name[..27])
    } else {
        snapshot.model_name.clone()
    };

    let header = format!(
        "ENTRENAR  {}  {} {}  {:.0} tok/s",
        Styled::new(&format!("{status_icon} {status_text}"), color_mode).fg(status_color),
        Styled::new(&elapsed, color_mode).fg((150, 150, 150)),
        Styled::new(&model_display, color_mode).fg((180, 180, 220)),
        tps
    );
    lines.push(format!("\u{2550}{:\u{2550}<w$}\u{2550}", "", w = w - 2));
    lines.push(header);
    lines.push(format!("\u{2500}{:\u{2500}<w$}\u{2500}", "", w = w - 2));
}

fn render_progress(
    lines: &mut Vec<String>,
    snapshot: &TrainingSnapshot,
    _w: usize,
    color_mode: ColorMode,
) {
    let epoch = snapshot.epoch.min(snapshot.total_epochs);
    let epoch_pct = if snapshot.total_epochs > 0 {
        // Percentage computed as f64 for precision, then narrowed after clamping
        let pct = (epoch as f64 / snapshot.total_epochs as f64 * 100.0).clamp(0.0, 100.0);
        pct as f32
    } else {
        0.0
    };

    let step = snapshot.step.min(snapshot.steps_per_epoch);
    let step_pct = if snapshot.steps_per_epoch > 0 {
        let pct = (step as f64 / snapshot.steps_per_epoch as f64 * 100.0).clamp(0.0, 100.0);
        pct as f32
    } else {
        0.0
    };

    let bar_w = 20;
    let epoch_bar = build_colored_block_bar(epoch_pct, bar_w, color_mode);
    let step_bar = build_colored_block_bar(step_pct, bar_w, color_mode);

    lines.push(format!(
        "Epoch {:>2}/{:<2} {} {:>3.0}%    Step {:>2}/{:<2} {} {:>3.0}%",
        epoch,
        snapshot.total_epochs,
        epoch_bar,
        epoch_pct,
        step,
        snapshot.steps_per_epoch,
        step_bar,
        step_pct
    ));
}

fn render_metrics(lines: &mut Vec<String>, snapshot: &TrainingSnapshot, color_mode: ColorMode) {
    let loss_str =
        if snapshot.loss.is_finite() { format!("{:.4}", snapshot.loss) } else { "???".to_string() };
    let loss_color = if snapshot.loss.is_finite() {
        pct_color((snapshot.loss * 10.0).min(100.0))
    } else {
        (255, 64, 64)
    };

    let best = snapshot
        .loss_history
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::INFINITY, f32::min);
    let best_str = if best.is_finite() { format!("{best:.4}") } else { "---".to_string() };

    let grad = snapshot.gradient_norm.max(0.0);
    let eta = snapshot.estimated_remaining().map_or("--:--:--".to_string(), format_duration);

    lines.push(format!(
        "Loss {} {}  Best {}  LR {}  Grad {:.2}  ETA {}",
        Styled::new(&loss_str, color_mode).fg(loss_color),
        trend_arrow(&snapshot.loss_history),
        Styled::new(&best_str, color_mode).fg((100, 200, 100)),
        format_lr(snapshot.learning_rate),
        grad,
        eta
    ));
}

fn render_loss_sparkline(
    lines: &mut Vec<String>,
    snapshot: &TrainingSnapshot,
    w: usize,
    color_mode: ColorMode,
) {
    if snapshot.loss_history.is_empty() {
        return;
    }

    let spark_w = w.saturating_sub(20);
    let sparkline = render_sparkline(&snapshot.loss_history, spark_w, color_mode);

    let valid: Vec<f32> = snapshot.loss_history.iter().copied().filter(|v| v.is_finite()).collect();
    let (min_l, max_l) = if valid.is_empty() {
        (0.0, 0.0)
    } else {
        (
            valid.iter().copied().fold(f32::INFINITY, f32::min),
            valid.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        )
    };

    lines.push(format!("\u{2500}{:\u{2500}<w$}\u{2500}", "", w = w - 2));
    lines.push(format!("Loss History: {sparkline} [{min_l:.2} - {max_l:.2}]"));
}

fn render_gpu_section(
    lines: &mut Vec<String>,
    snapshot: &TrainingSnapshot,
    w: usize,
    color_mode: ColorMode,
) {
    lines.push(format!("\u{2500}{:\u{2500}<w$}\u{2500}", "", w = w - 2));
    if let Some(gpu) = &snapshot.gpu {
        let util_bar = build_colored_block_bar(gpu.utilization_percent, 15, color_mode);
        let vram_pct = gpu.vram_percent().min(100.0);
        let vram_bar = build_colored_block_bar(vram_pct, 15, color_mode);

        let temp_color = if gpu.temperature_celsius > 80.0 {
            TrainingPalette::ERROR
        } else if gpu.temperature_celsius > 70.0 {
            TrainingPalette::WARNING
        } else {
            TrainingPalette::SUCCESS
        };

        lines.push(format!(
            "GPU: {}  Util {} {:>3.0}%  Temp {}  Power {:.0}W",
            gpu.device_name.chars().take(20).collect::<String>(),
            util_bar,
            gpu.utilization_percent,
            Styled::new(&format!("{:.0}\u{00B0}C", gpu.temperature_celsius), color_mode)
                .fg(temp_color),
            gpu.power_watts
        ));

        lines.push(format!(
            "VRAM: {} {:>3.0}%  {:.1}G / {:.0}G",
            vram_bar,
            vram_pct,
            gpu.vram_used_gb.min(gpu.vram_total_gb),
            gpu.vram_total_gb
        ));
    } else {
        lines.push("GPU: N/A".to_string());
    }
}

fn render_epoch_table(
    lines: &mut Vec<String>,
    snapshot: &TrainingSnapshot,
    w: usize,
    color_mode: ColorMode,
) {
    lines.push(format!("\u{2500}{:\u{2500}<w$}\u{2500}", "", w = w - 2));
    lines.push(format!(
        "{:>5}  {:>8}  {:>8}  {:>8}  {:>10}  {:>5}",
        Styled::new("Epoch", color_mode).fg((150, 150, 150)),
        Styled::new("Loss", color_mode).fg((150, 150, 150)),
        Styled::new("Min", color_mode).fg((150, 150, 150)),
        Styled::new("Max", color_mode).fg((150, 150, 150)),
        Styled::new("LR", color_mode).fg((150, 150, 150)),
        Styled::new("Trend", color_mode).fg((150, 150, 150)),
    ));

    let summaries = compute_epoch_summaries(snapshot);
    if summaries.is_empty() {
        lines.push("  (waiting for epoch data...)".to_string());
        return;
    }

    let max_rows = 6;
    let start_idx = summaries.len().saturating_sub(max_rows);

    for (i, summary) in summaries.iter().skip(start_idx).enumerate() {
        let trend = epoch_trend_arrow(i, start_idx, &summaries, color_mode);
        let loss_color = pct_color((summary.avg_loss * 8.0).min(100.0));

        lines.push(format!(
            "{:>5}  {}  {:>8.4}  {:>8.4}  {:>10}  {:>5}",
            summary.epoch,
            Styled::new(&format!("{:>8.4}", summary.avg_loss), color_mode).fg(loss_color),
            summary.min_loss,
            summary.max_loss,
            format_lr(summary.lr),
            trend
        ));
    }

    if start_idx > 0 {
        lines.push(format!(
            "  ... {} earlier epochs",
            Styled::new(&format!("{start_idx}"), color_mode).fg((100, 100, 100))
        ));
    }
}

fn epoch_trend_arrow(
    i: usize,
    start_idx: usize,
    summaries: &[EpochSummary],
    color_mode: ColorMode,
) -> String {
    if i == 0 && start_idx == 0 {
        return " ".to_string();
    }
    let prev_idx = if i > 0 { start_idx + i - 1 } else { start_idx.saturating_sub(1) };
    if let Some(prev) = summaries.get(prev_idx) {
        let current = &summaries[start_idx + i];
        let change = (current.avg_loss - prev.avg_loss) / prev.avg_loss.abs().max(0.001);
        if change < -0.02 {
            Styled::new("\u{2193}", color_mode).fg((100, 255, 100)).to_string()
        } else if change > 0.02 {
            Styled::new("\u{2191}", color_mode).fg((255, 100, 100)).to_string()
        } else {
            Styled::new("\u{2192}", color_mode).fg((150, 150, 150)).to_string()
        }
    } else {
        " ".to_string()
    }
}

fn render_config_footer(
    lines: &mut Vec<String>,
    snapshot: &TrainingSnapshot,
    w: usize,
    color_mode: ColorMode,
) {
    lines.push(format!("\u{2500}{:\u{2500}<w$}\u{2500}", "", w = w - 2));

    let opt = if snapshot.optimizer_name.is_empty() { "N/A" } else { &snapshot.optimizer_name };
    let batch = if snapshot.batch_size > 0 {
        format!("{}", snapshot.batch_size)
    } else {
        "N/A".to_string()
    };

    lines.push(format!(
        "Config: {}  Batch: {}  Checkpoint: {}",
        Styled::new(opt, color_mode).fg((150, 255, 150)),
        batch,
        if snapshot.checkpoint_path.is_empty() { "N/A" } else { &snapshot.checkpoint_path }
    ));

    lines.push(format!("\u{2550}{:\u{2550}<w$}\u{2550}", "", w = w - 2));
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_renders() {
        let snapshot = TrainingSnapshot {
            epoch: 5,
            total_epochs: 10,
            step: 8,
            steps_per_epoch: 16,
            loss: 2.5,
            loss_history: vec![5.0, 4.5, 4.0, 3.5, 3.0, 2.8, 2.6, 2.5],
            learning_rate: 0.0001,
            gradient_norm: 1.5,
            tokens_per_second: 100.0,
            model_name: "TestModel".to_string(),
            optimizer_name: "AdamW".to_string(),
            batch_size: 4,
            ..Default::default()
        };

        let layout = render_layout(&snapshot, 80);
        assert!(layout.contains("ENTRENAR"));
        assert!(layout.contains("Epoch"));
        assert!(layout.contains("Loss"));
        assert!(layout.contains("Step"));
    }

    #[test]
    fn test_render_header_initializing() {
        let snapshot = TrainingSnapshot {
            status: TrainingStatus::Initializing,
            model_name: "TestModel".to_string(),
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        assert!(layout.contains("Init"));
    }

    #[test]
    fn test_render_header_running() {
        let snapshot = TrainingSnapshot {
            status: TrainingStatus::Running,
            model_name: "TestModel".to_string(),
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        assert!(layout.contains("Running"));
    }

    #[test]
    fn test_render_header_paused() {
        let snapshot = TrainingSnapshot {
            status: TrainingStatus::Paused,
            model_name: "TestModel".to_string(),
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        assert!(layout.contains("Paused"));
    }

    #[test]
    fn test_render_header_completed() {
        let snapshot = TrainingSnapshot {
            status: TrainingStatus::Completed,
            model_name: "TestModel".to_string(),
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        assert!(layout.contains("Done"));
    }

    #[test]
    fn test_render_header_failed_arm() {
        let status = TrainingStatus::Failed("out of memory".to_string());
        match &status {
            TrainingStatus::Failed(_) => {}
            _ => unreachable!(),
        }
        let snapshot =
            TrainingSnapshot { status, model_name: "TestModel".to_string(), ..Default::default() };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        assert!(layout.contains("FAIL"));
    }

    // ── Additional coverage tests ──

    #[test]
    fn test_render_layout_colored_mono() {
        let snapshot = TrainingSnapshot {
            epoch: 2,
            total_epochs: 5,
            step: 3,
            steps_per_epoch: 10,
            loss: 1.5,
            loss_history: vec![3.0, 2.5, 2.0, 1.5],
            learning_rate: 0.0001,
            gradient_norm: 0.5,
            tokens_per_second: 50.0,
            model_name: "TestModel".to_string(),
            optimizer_name: "SGD".to_string(),
            batch_size: 8,
            status: TrainingStatus::Running,
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 100, ColorMode::Mono);
        assert!(layout.contains("ENTRENAR"));
        assert!(layout.contains("Running"));
        assert!(layout.contains("Epoch"));
        assert!(layout.contains("Step"));
        assert!(layout.contains("Loss"));
        assert!(layout.contains("Config"));
    }

    #[test]
    fn test_render_layout_with_gpu() {
        let snapshot = TrainingSnapshot {
            epoch: 1,
            total_epochs: 3,
            step: 5,
            steps_per_epoch: 20,
            loss: 2.0,
            loss_history: vec![3.0, 2.5, 2.0],
            learning_rate: 0.001,
            gradient_norm: 1.0,
            tokens_per_second: 200.0,
            model_name: "GPUModel".to_string(),
            status: TrainingStatus::Running,
            gpu: Some(super::super::state::GpuTelemetry {
                device_name: "RTX 4090".to_string(),
                utilization_percent: 95.0,
                vram_used_gb: 20.0,
                vram_total_gb: 24.0,
                temperature_celsius: 72.0,
                power_watts: 350.0,
                power_limit_watts: 400.0,
                processes: vec![],
            }),
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 100, ColorMode::Mono);
        assert!(layout.contains("GPU:"));
        assert!(layout.contains("RTX 4090"));
        assert!(layout.contains("VRAM:"));
    }

    #[test]
    fn test_render_layout_no_gpu() {
        let snapshot = TrainingSnapshot {
            epoch: 1,
            total_epochs: 1,
            step: 1,
            steps_per_epoch: 1,
            loss: 1.0,
            model_name: "NoGPU".to_string(),
            status: TrainingStatus::Running,
            gpu: None,
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        assert!(layout.contains("GPU: N/A"));
    }

    #[test]
    fn test_render_layout_empty_model_name() {
        let snapshot = TrainingSnapshot {
            model_name: String::new(),
            status: TrainingStatus::Running,
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        assert!(layout.contains("N/A"));
    }

    #[test]
    fn test_render_layout_long_model_name() {
        let snapshot = TrainingSnapshot {
            model_name: "A".repeat(50),
            status: TrainingStatus::Running,
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        // Long model names should be truncated with ...
        assert!(layout.contains("..."));
    }

    #[test]
    fn test_render_progress_zero_epochs() {
        let snapshot = TrainingSnapshot {
            epoch: 0,
            total_epochs: 0,
            step: 0,
            steps_per_epoch: 0,
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        assert!(layout.contains("Epoch"));
    }

    #[test]
    fn test_render_metrics_nan_loss() {
        let snapshot = TrainingSnapshot {
            loss: f32::NAN,
            learning_rate: 0.001,
            status: TrainingStatus::Running,
            model_name: "test".to_string(),
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        assert!(layout.contains("???"));
    }

    #[test]
    fn test_render_metrics_infinite_loss() {
        let snapshot = TrainingSnapshot {
            loss: f32::INFINITY,
            learning_rate: 0.001,
            status: TrainingStatus::Running,
            model_name: "test".to_string(),
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        assert!(layout.contains("???"));
    }

    #[test]
    fn test_render_loss_sparkline_empty() {
        let snapshot = TrainingSnapshot {
            loss_history: vec![],
            status: TrainingStatus::Running,
            model_name: "test".to_string(),
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        // Empty loss_history should not produce a "Loss History" section
        assert!(!layout.contains("Loss History"));
    }

    #[test]
    fn test_render_epoch_table_no_data() {
        let snapshot = TrainingSnapshot {
            loss_history: vec![],
            steps_per_epoch: 0,
            status: TrainingStatus::Running,
            model_name: "test".to_string(),
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        assert!(layout.contains("waiting for epoch data"));
    }

    #[test]
    fn test_render_epoch_table_with_epochs() {
        let snapshot = TrainingSnapshot {
            steps_per_epoch: 3,
            loss_history: vec![5.0, 4.0, 3.0, 2.5, 2.0, 1.5],
            learning_rate: 0.001,
            tokens_per_second: 100.0,
            status: TrainingStatus::Running,
            model_name: "test".to_string(),
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        assert!(layout.contains("Epoch"));
        assert!(layout.contains("Loss"));
    }

    #[test]
    fn test_render_epoch_table_many_epochs_truncation() {
        // 20 epochs, max_rows=6 means some should be hidden
        let snapshot = TrainingSnapshot {
            steps_per_epoch: 1,
            loss_history: (0..20).map(|i| 10.0 - (i as f32 * 0.5)).collect(),
            learning_rate: 0.001,
            tokens_per_second: 100.0,
            status: TrainingStatus::Running,
            model_name: "test".to_string(),
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        assert!(layout.contains("earlier epochs"));
    }

    #[test]
    fn test_render_config_footer_empty_optimizer() {
        let snapshot = TrainingSnapshot {
            optimizer_name: String::new(),
            batch_size: 0,
            checkpoint_path: String::new(),
            status: TrainingStatus::Running,
            model_name: "test".to_string(),
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        assert!(layout.contains("Config:"));
    }

    #[test]
    fn test_render_config_footer_with_values() {
        let snapshot = TrainingSnapshot {
            optimizer_name: "AdamW".to_string(),
            batch_size: 16,
            checkpoint_path: "/tmp/checkpoints".to_string(),
            status: TrainingStatus::Running,
            model_name: "test".to_string(),
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 100, ColorMode::Mono);
        assert!(layout.contains("AdamW"));
        assert!(layout.contains("16"));
        assert!(layout.contains("/tmp/checkpoints"));
    }

    #[test]
    fn test_render_gpu_temp_warning() {
        let snapshot = TrainingSnapshot {
            gpu: Some(super::super::state::GpuTelemetry {
                device_name: "RTX 4090".to_string(),
                utilization_percent: 90.0,
                vram_used_gb: 20.0,
                vram_total_gb: 24.0,
                temperature_celsius: 75.0, // Warning range
                power_watts: 300.0,
                power_limit_watts: 400.0,
                processes: vec![],
            }),
            status: TrainingStatus::Running,
            model_name: "test".to_string(),
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 100, ColorMode::Mono);
        assert!(layout.contains("75"));
    }

    #[test]
    fn test_render_gpu_temp_error() {
        let snapshot = TrainingSnapshot {
            gpu: Some(super::super::state::GpuTelemetry {
                device_name: "RTX 4090".to_string(),
                utilization_percent: 90.0,
                vram_used_gb: 20.0,
                vram_total_gb: 24.0,
                temperature_celsius: 85.0, // Error range (>80)
                power_watts: 300.0,
                power_limit_watts: 400.0,
                processes: vec![],
            }),
            status: TrainingStatus::Running,
            model_name: "test".to_string(),
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 100, ColorMode::Mono);
        assert!(layout.contains("85"));
    }

    #[test]
    fn test_epoch_trend_arrow_first_epoch_no_start_idx() {
        let result = epoch_trend_arrow(0, 0, &[], ColorMode::Mono);
        assert_eq!(result, " ");
    }

    #[test]
    fn test_epoch_trend_arrow_with_start_idx() {
        let summaries = vec![
            EpochSummary {
                epoch: 1,
                avg_loss: 5.0,
                min_loss: 4.0,
                max_loss: 6.0,
                end_loss: 4.5,
                avg_grad: 1.0,
                lr: 0.001,
                tokens_per_sec: 100.0,
            },
            EpochSummary {
                epoch: 2,
                avg_loss: 3.0,
                min_loss: 2.5,
                max_loss: 3.5,
                end_loss: 2.8,
                avg_grad: 0.8,
                lr: 0.001,
                tokens_per_sec: 100.0,
            },
        ];
        // i=0, start_idx=1 -> should compare with prev
        let result = epoch_trend_arrow(0, 1, &summaries, ColorMode::Mono);
        // summaries[1] vs summaries[0]: 3.0 vs 5.0 -> decreasing
        assert!(result.contains("\u{2193}")); // down arrow
    }

    #[test]
    fn test_epoch_trend_arrow_no_prev() {
        let summaries: Vec<EpochSummary> = vec![];
        // i=0, start_idx=1, but no element at prev_idx -> space
        let result = epoch_trend_arrow(0, 1, &summaries, ColorMode::Mono);
        assert_eq!(result, " ");
    }

    #[test]
    fn test_render_layout_minimum_width() {
        // Width less than 80 should be clamped to 80
        let snapshot = TrainingSnapshot {
            model_name: "test".to_string(),
            status: TrainingStatus::Running,
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 40, ColorMode::Mono);
        // Should not panic and should contain basic elements
        assert!(layout.contains("ENTRENAR"));
    }

    #[test]
    fn test_render_with_truecolor() {
        let snapshot = TrainingSnapshot {
            epoch: 2,
            total_epochs: 5,
            step: 3,
            steps_per_epoch: 10,
            loss: 1.5,
            loss_history: vec![3.0, 2.5, 2.0, 1.5],
            learning_rate: 0.0001,
            gradient_norm: 0.5,
            tokens_per_second: 50.0,
            model_name: "TestModel".to_string(),
            status: TrainingStatus::Running,
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 100, ColorMode::TrueColor);
        // Should contain ANSI color codes
        assert!(layout.contains("\x1b["));
    }

    #[test]
    fn test_render_best_loss_all_nan() {
        let snapshot = TrainingSnapshot {
            loss: 1.0,
            loss_history: vec![f32::NAN, f32::NAN, f32::NAN],
            learning_rate: 0.001,
            model_name: "test".to_string(),
            status: TrainingStatus::Running,
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        assert!(layout.contains("Best ---"));
    }

    #[test]
    fn test_render_eta_no_tps() {
        let snapshot = TrainingSnapshot {
            tokens_per_second: 0.0,
            learning_rate: 0.001,
            model_name: "test".to_string(),
            status: TrainingStatus::Running,
            ..Default::default()
        };
        let layout = render_layout_colored(&snapshot, 80, ColorMode::Mono);
        assert!(layout.contains("ETA --:--:--"));
    }
}
