//! Formatting helper methods for the cbtop TUI.
//!
//! Contains bar generators, sparkline renderers, number formatters,
//! and the status bar renderer.

use presentar_core::{Canvas, Point, TextStyle};
use presentar_terminal::direct::DirectTerminalCanvas;
use presentar_terminal::Theme;

use crate::app::hardware::LoadMetrics;
use crate::app::CbtopApp;
use crate::config::ComputeBackend;

impl CbtopApp {
    pub(super) fn make_bar(value: f64, max: f64, width: usize) -> String {
        let filled = ((value / max) * width as f64).round() as usize;
        let empty = width.saturating_sub(filled);
        format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
    }

    /// PMAT-012 UI-02: Mini bar for per-core CPU display (no brackets, compact)
    pub(super) fn make_mini_bar(value: f64, max: f64, width: usize) -> String {
        let filled = ((value / max) * width as f64).round() as usize;
        let empty = width.saturating_sub(filled);
        format!("{}{}", "▓".repeat(filled), "░".repeat(empty))
    }

    pub(super) fn make_sparkline(data: &[f64], width: usize) -> String {
        const CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        if data.is_empty() {
            return " ".repeat(width);
        }

        let max = data
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(1.0);
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min).min(0.0);
        let range = (max - min).max(1.0);

        // Sample data to fit width
        let step = data.len().max(1) as f64 / width as f64;
        let mut result = String::with_capacity(width);

        for i in 0..width {
            let idx = (i as f64 * step) as usize;
            if idx < data.len() {
                let normalized = (data[idx] - min) / range;
                let char_idx = (normalized * 7.0).round() as usize;
                result.push(CHARS[char_idx.min(7)]);
            } else {
                result.push(' ');
            }
        }

        result
    }

    /// PMAT-012 UI-05: Braille graph for higher resolution sparklines
    /// Uses Unicode braille patterns (U+2800-U+28FF) - each character encodes 2 columns x 4 rows
    pub(super) fn make_braille_sparkline(data: &[f64], width: usize) -> String {
        if data.is_empty() {
            return " ".repeat(width);
        }

        let max = data
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(1.0);
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min).min(0.0);
        let range = (max - min).max(1.0);

        // Braille dot positions (column 1: bits 0,1,2,6; column 2: bits 3,4,5,7)
        // Dots are numbered: 1,2,3,7 (left col), 4,5,6,8 (right col)
        // Bit mapping: dot1=0x01, dot2=0x02, dot3=0x04, dot4=0x08,
        //              dot5=0x10, dot6=0x20, dot7=0x40, dot8=0x80
        const COL1_DOTS: [u8; 4] = [0x40, 0x04, 0x02, 0x01]; // bottom to top
        const COL2_DOTS: [u8; 4] = [0x80, 0x20, 0x10, 0x08]; // bottom to top

        let mut result = String::with_capacity(width);
        let points_per_char = 2;
        let step = data.len().max(1) as f64 / (width * points_per_char) as f64;

        for i in 0..width {
            let mut pattern: u8 = 0;

            // Left column (first data point)
            let idx1 = ((i * 2) as f64 * step) as usize;
            if idx1 < data.len() {
                let normalized = (data[idx1] - min) / range;
                let dots = (normalized * 4.0).round() as usize;
                for d in 0..dots.min(4) {
                    pattern |= COL1_DOTS[d];
                }
            }

            // Right column (second data point)
            let idx2 = ((i * 2 + 1) as f64 * step) as usize;
            if idx2 < data.len() {
                let normalized = (data[idx2] - min) / range;
                let dots = (normalized * 4.0).round() as usize;
                for d in 0..dots.min(4) {
                    pattern |= COL2_DOTS[d];
                }
            }

            // Braille base is U+2800
            result.push(char::from_u32(0x2800 + pattern as u32).unwrap_or(' '));
        }

        result
    }

    pub(super) fn format_number(n: u64) -> String {
        if n >= 1_000_000_000 {
            format!("{:.2}B", n as f64 / 1_000_000_000.0)
        } else if n >= 1_000_000 {
            format!("{:.2}M", n as f64 / 1_000_000.0)
        } else if n >= 1_000 {
            format!("{:.2}K", n as f64 / 1_000.0)
        } else {
            n.to_string()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_status_bar(
        canvas: &mut DirectTerminalCanvas,
        width: u16,
        height: u16,
        is_running: bool,
        intensity: f64,
        backend: ComputeBackend,
        show_fps: bool,
        frame_avg: f64,
        metrics: &LoadMetrics,
        theme: &Theme,
    ) {
        let y = height as f32 - 1.0;
        let dim_style = TextStyle {
            color: theme.dim,
            ..Default::default()
        };

        // Load status
        let status = if is_running { "●RUN" } else { "○OFF" };
        let status_color = if is_running {
            theme.cpu.sample(0.0)
        } else {
            theme.dim
        };
        canvas.draw_text(
            &format!(" {} ", status),
            Point::new(0.0, y),
            &TextStyle {
                color: status_color,
                ..Default::default()
            },
        );

        // Backend
        canvas.draw_text(&format!("│ {:?} ", backend), Point::new(6.0, y), &dim_style);

        // Intensity
        canvas.draw_text(
            &format!("│ Int:{:.0}% ", intensity * 100.0),
            Point::new(16.0, y),
            &dim_style,
        );

        // Real metrics in status bar
        canvas.draw_text(
            &format!(
                "│ {:.0} brick/s │ {:.1}μs ",
                metrics.bricks_per_second, metrics.avg_latency_us
            ),
            Point::new(28.0, y),
            &TextStyle {
                color: theme.foreground,
                ..Default::default()
            },
        );

        // PMAT-012 UI-06: GFLOP/s in status bar
        let gflops = metrics.ops_per_second / 1_000_000_000.0;
        canvas.draw_text(
            &format!("│ {:.2} GFLOP/s ", gflops),
            Point::new(55.0, y),
            &TextStyle {
                color: theme.cpu.sample(0.3),
                ..Default::default()
            },
        );

        // FPS
        if show_fps || frame_avg > 0.0 {
            let fps = if frame_avg > 0.0 {
                1000.0 / frame_avg
            } else {
                0.0
            };
            canvas.draw_text(
                &format!("│ {:.0} FPS", fps),
                Point::new(width as f32 - 10.0, y),
                &dim_style,
            );
        }
    }
}
