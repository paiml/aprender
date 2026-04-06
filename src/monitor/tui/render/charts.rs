//! Chart widgets: gauge, braille chart, sample panel, config panel, history table.

use super::super::color::{ColorMode, Styled};
use super::super::state::{SamplePeek, TrainingSnapshot};
use super::bars::{build_block_bar, render_sparkline};
use super::epoch::{compute_epoch_summaries, EpochSummary};
use super::format::format_lr;

pub fn render_gauge(value: f32, max: f32, width: usize, label: &str) -> String {
    let percent = if max > 0.0 { value / max * 100.0 } else { 0.0 };
    let bar = build_block_bar(percent, width.saturating_sub(label.len() + 8));
    format!("{label}{bar} {percent:>5.1}%")
}

pub struct BrailleChart {
    width: usize,
    height: usize,
    data: Vec<f32>,
    color_mode: ColorMode,
}

impl BrailleChart {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, data: Vec::new(), color_mode: ColorMode::detect() }
    }

    pub fn color_mode(mut self, mode: ColorMode) -> Self {
        self.color_mode = mode;
        self
    }

    pub fn data(mut self, data: Vec<f32>) -> Self {
        self.data = data;
        self
    }

    #[allow(dead_code)]
    pub fn bounds(self, _min: f32, _max: f32) -> Self {
        self
    }

    pub fn log_scale(self, _enabled: bool) -> Self {
        self
    }

    pub fn render(&self) -> String {
        if self.data.is_empty() {
            return " ".repeat(self.width).repeat(self.height);
        }
        let mut lines = Vec::new();
        for row in 0..self.height {
            let start = (row * self.data.len()) / self.height;
            let end = ((row + 1) * self.data.len()) / self.height;
            let slice = if end > start {
                &self.data[start..end]
            } else if start < self.data.len() {
                &self.data[start..=start]
            } else {
                &[]
            };
            lines.push(render_sparkline(slice, self.width, self.color_mode));
        }
        lines.join("\n")
    }
}

pub fn render_braille_chart(data: &[f32], width: usize, height: usize, _log_scale: bool) -> String {
    BrailleChart::new(width, height).data(data.to_vec()).render()
}

pub fn render_sample_panel(
    _sample: Option<&SamplePeek>,
    _width: usize,
    _color_mode: ColorMode,
) -> String {
    String::new()
}

pub fn render_config_panel(
    snapshot: &TrainingSnapshot,
    width: usize,
    color_mode: ColorMode,
) -> String {
    let mut lines = Vec::new();

    let model_name = if snapshot.model_name.is_empty() { "N/A" } else { &snapshot.model_name };
    let model_display: String = model_name.chars().take(width - 8).collect();
    lines.push(Styled::new(&model_display, color_mode).fg((180, 180, 255)).to_string());

    let opt = if snapshot.optimizer_name.is_empty() { "N/A" } else { &snapshot.optimizer_name };
    let batch = if snapshot.batch_size > 0 {
        format!("batch:{}", snapshot.batch_size)
    } else {
        "N/A".to_string()
    };
    lines.push(format!("{opt}  {batch}"));

    lines.join("\n")
}

pub fn render_history_table(
    snapshot: &TrainingSnapshot,
    width: usize,
    max_rows: usize,
    color_mode: ColorMode,
) -> String {
    let mut lines = Vec::new();

    let header = format!(
        "{:>5} {:>8} {:>8} {:>8} {:>10} {:>10} {:>5}",
        "Epoch", "Loss", "Min", "Max", "LR", "Tok/s", "Trend"
    );
    lines.push(Styled::new(&header, color_mode).fg((150, 150, 150)).to_string());
    lines.push("\u{2500}".repeat(width.min(70)));

    let summaries = compute_epoch_summaries(snapshot);
    if summaries.is_empty() {
        lines.push("(waiting for epoch data...)".to_string());
        return lines.join("\n");
    }

    let start_idx = summaries.len().saturating_sub(max_rows);
    for (i, summary) in summaries.iter().skip(start_idx).enumerate() {
        let trend = history_trend(i, start_idx, summary, &summaries, color_mode);

        let row = format!(
            "{:>5} {:>8.3} {:>8.3} {:>8.3} {:>10} {:>10.1} {}",
            summary.epoch,
            summary.avg_loss,
            summary.min_loss,
            summary.max_loss,
            format_lr(summary.lr),
            summary.tokens_per_sec,
            Styled::new(trend.0, color_mode).fg(trend.1)
        );
        lines.push(row);
    }

    if start_idx > 0 {
        lines.push(format!("  \u{2191} {start_idx} more epochs above"));
    }

    lines.join("\n")
}

fn history_trend<'a>(
    i: usize,
    start_idx: usize,
    summary: &EpochSummary,
    summaries: &[EpochSummary],
    _color_mode: ColorMode,
) -> (&'a str, (u8, u8, u8)) {
    if i > 0 || start_idx > 0 {
        let prev_idx = if i > 0 { start_idx + i - 1 } else { start_idx.saturating_sub(1) };
        if let Some(prev) = summaries.get(prev_idx) {
            let change = (summary.avg_loss - prev.avg_loss) / prev.avg_loss.abs().max(0.001);
            if change < -0.02 {
                ("\u{2193}", (100, 255, 100))
            } else if change > 0.02 {
                ("\u{2191}", (255, 100, 100))
            } else {
                ("\u{2192}", (150, 150, 150))
            }
        } else {
            ("", (150, 150, 150))
        }
    } else {
        ("", (150, 150, 150))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_history_table_render() {
        let snapshot = TrainingSnapshot {
            steps_per_epoch: 4,
            loss_history: vec![10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5],
            tokens_per_second: 100.0,
            learning_rate: 0.0001,
            gradient_norm: 2.5,
            ..Default::default()
        };

        let table = render_history_table(&snapshot, 80, 10, ColorMode::Mono);
        assert!(table.contains("Epoch"));
        assert!(table.contains("Loss"));
    }

    #[test]
    fn test_history_table_empty() {
        let snapshot = TrainingSnapshot::default();
        let table = render_history_table(&snapshot, 80, 10, ColorMode::Mono);
        assert!(table.contains("waiting for epoch data"));
    }

    // ── render_gauge tests ─────────────────────────────────────────

    #[test]
    fn test_render_gauge_zero() {
        let result = render_gauge(0.0, 100.0, 30, "GPU: ");
        assert!(result.contains("0.0%"));
        assert!(result.starts_with("GPU: "));
    }

    #[test]
    fn test_render_gauge_full() {
        let result = render_gauge(100.0, 100.0, 30, "");
        assert!(result.contains("100.0%"));
    }

    #[test]
    fn test_render_gauge_zero_max() {
        let result = render_gauge(50.0, 0.0, 30, "");
        assert!(result.contains("0.0%"));
    }

    #[test]
    fn test_render_gauge_half() {
        let result = render_gauge(50.0, 100.0, 30, "VRAM: ");
        assert!(result.contains("50.0%"));
    }

    // ── BrailleChart tests ─────────────────────────────────────────

    #[test]
    fn test_braille_chart_empty_data() {
        let chart = BrailleChart::new(10, 3).data(Vec::new()).render();
        // Empty data should return spaces
        assert!(chart.chars().all(|c| c == ' '));
    }

    #[test]
    fn test_braille_chart_with_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let chart = BrailleChart::new(10, 2).data(data).render();
        assert!(!chart.is_empty());
    }

    #[test]
    fn test_braille_chart_color_mode() {
        let data = vec![1.0, 2.0, 3.0];
        let chart = BrailleChart::new(10, 2).color_mode(ColorMode::Mono).data(data).render();
        assert!(!chart.is_empty());
    }

    #[test]
    fn test_braille_chart_log_scale_noop() {
        let data = vec![1.0, 10.0, 100.0];
        let chart = BrailleChart::new(10, 2).log_scale(true).data(data).render();
        assert!(!chart.is_empty());
    }

    #[test]
    fn test_braille_chart_bounds_noop() {
        let data = vec![1.0, 5.0, 10.0];
        let chart = BrailleChart::new(10, 2).bounds(0.0, 10.0).data(data).render();
        assert!(!chart.is_empty());
    }

    #[test]
    fn test_braille_chart_single_datapoint() {
        let data = vec![5.0];
        let chart = BrailleChart::new(10, 2).data(data).render();
        assert!(!chart.is_empty());
    }

    #[test]
    fn test_render_braille_chart_function() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = render_braille_chart(&data, 10, 2, false);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_render_braille_chart_empty() {
        let result = render_braille_chart(&[], 10, 2, false);
        assert!(result.chars().all(|c| c == ' '));
    }

    // ── render_sample_panel tests ──────────────────────────────────

    #[test]
    fn test_render_sample_panel_none() {
        let result = render_sample_panel(None, 80, ColorMode::Mono);
        assert!(result.is_empty());
    }

    #[test]
    fn test_render_sample_panel_some() {
        let sample = SamplePeek {
            input_preview: "fn hello()".to_string(),
            target_preview: "fn test_hello()".to_string(),
            generated_preview: "fn test_hello()".to_string(),
            token_match_percent: 100.0,
        };
        let result = render_sample_panel(Some(&sample), 80, ColorMode::Mono);
        // Current implementation returns empty string
        assert!(result.is_empty());
    }

    // ── render_config_panel tests ──────────────────────────────────

    #[test]
    fn test_render_config_panel_defaults() {
        let snapshot = TrainingSnapshot::default();
        let result = render_config_panel(&snapshot, 80, ColorMode::Mono);
        assert!(result.contains("N/A")); // empty model_name and optimizer
    }

    #[test]
    fn test_render_config_panel_with_values() {
        let snapshot = TrainingSnapshot {
            model_name: "Qwen2.5-Coder-0.5B".to_string(),
            optimizer_name: "AdamW".to_string(),
            batch_size: 4,
            ..Default::default()
        };
        let result = render_config_panel(&snapshot, 80, ColorMode::Mono);
        assert!(result.contains("Qwen2.5-Coder-0.5B"));
        assert!(result.contains("AdamW"));
        assert!(result.contains("batch:4"));
    }

    #[test]
    fn test_render_config_panel_zero_batch() {
        let snapshot = TrainingSnapshot {
            model_name: "model".to_string(),
            optimizer_name: "SGD".to_string(),
            batch_size: 0,
            ..Default::default()
        };
        let result = render_config_panel(&snapshot, 80, ColorMode::Mono);
        assert!(result.contains("N/A")); // batch_size 0 shows N/A
    }

    #[test]
    fn test_render_config_panel_long_model_name_truncated() {
        let snapshot = TrainingSnapshot { model_name: "A".repeat(200), ..Default::default() };
        let result = render_config_panel(&snapshot, 30, ColorMode::Mono);
        // Model name should be truncated to fit within width - 8
        let first_line = result.lines().next().unwrap_or("");
        assert!(first_line.len() <= 30);
    }

    // ── render_history_table advanced tests ─────────────────────────

    #[test]
    fn test_history_table_multiple_epochs_with_trend() {
        let snapshot = TrainingSnapshot {
            steps_per_epoch: 2,
            loss_history: vec![10.0, 9.0, 5.0, 4.0, 2.0, 1.0],
            lr_history: vec![0.001, 0.001, 0.0005, 0.0005, 0.0001, 0.0001],
            tokens_per_second: 500.0,
            ..Default::default()
        };
        let table = render_history_table(&snapshot, 80, 10, ColorMode::Mono);
        // Should have header + separator + 3 epochs
        let lines: Vec<&str> = table.lines().collect();
        assert!(lines.len() >= 4); // header + sep + at least 2 data rows
    }

    #[test]
    fn test_history_table_max_rows_truncation() {
        let snapshot = TrainingSnapshot {
            steps_per_epoch: 1,
            loss_history: vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            tokens_per_second: 100.0,
            learning_rate: 0.001,
            ..Default::default()
        };
        let table = render_history_table(&snapshot, 80, 3, ColorMode::Mono);
        // Should show "more epochs above" message
        assert!(table.contains("more epochs above"));
    }

    #[test]
    fn test_history_table_single_epoch() {
        let snapshot = TrainingSnapshot {
            steps_per_epoch: 3,
            loss_history: vec![5.0, 4.0, 3.0],
            tokens_per_second: 200.0,
            learning_rate: 0.001,
            ..Default::default()
        };
        let table = render_history_table(&snapshot, 80, 10, ColorMode::Mono);
        assert!(table.contains("Epoch"));
    }

    // ── history_trend tests ────────────────────────────────────────

    #[test]
    fn test_history_trend_first_epoch() {
        let summary = EpochSummary {
            epoch: 1,
            avg_loss: 5.0,
            min_loss: 4.0,
            max_loss: 6.0,
            end_loss: 4.5,
            avg_grad: 1.0,
            lr: 0.001,
            tokens_per_sec: 100.0,
        };
        let summaries = vec![summary.clone()];
        let (arrow, _color) = history_trend(0, 0, &summary, &summaries, ColorMode::Mono);
        assert_eq!(arrow, ""); // first epoch, no trend
    }

    #[test]
    fn test_history_trend_decreasing() {
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
        let (arrow, color) = history_trend(1, 0, &summaries[1], &summaries, ColorMode::Mono);
        assert_eq!(arrow, "\u{2193}"); // down arrow for decreasing loss
        assert_eq!(color, (100, 255, 100)); // green
    }

    #[test]
    fn test_history_trend_increasing() {
        let summaries = vec![
            EpochSummary {
                epoch: 1,
                avg_loss: 3.0,
                min_loss: 2.5,
                max_loss: 3.5,
                end_loss: 2.8,
                avg_grad: 1.0,
                lr: 0.001,
                tokens_per_sec: 100.0,
            },
            EpochSummary {
                epoch: 2,
                avg_loss: 5.0,
                min_loss: 4.0,
                max_loss: 6.0,
                end_loss: 4.5,
                avg_grad: 0.8,
                lr: 0.001,
                tokens_per_sec: 100.0,
            },
        ];
        let (arrow, color) = history_trend(1, 0, &summaries[1], &summaries, ColorMode::Mono);
        assert_eq!(arrow, "\u{2191}"); // up arrow for increasing loss
        assert_eq!(color, (255, 100, 100)); // red
    }

    #[test]
    fn test_history_trend_stable() {
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
                avg_loss: 5.01,
                min_loss: 4.0,
                max_loss: 6.0,
                end_loss: 4.5,
                avg_grad: 0.8,
                lr: 0.001,
                tokens_per_sec: 100.0,
            },
        ];
        let (arrow, color) = history_trend(1, 0, &summaries[1], &summaries, ColorMode::Mono);
        assert_eq!(arrow, "\u{2192}"); // right arrow for stable
        assert_eq!(color, (150, 150, 150)); // grey
    }
}
