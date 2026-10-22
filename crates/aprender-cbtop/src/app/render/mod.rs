//! Rendering methods for the cbtop TUI.

mod formatters;
mod panels;

use presentar_core::{Canvas, Point, Rect, TextStyle};
use presentar_terminal::direct::DirectTerminalCanvas;
use presentar_terminal::Theme;

use super::hardware::{HardwareInfo, LoadMetrics};
use super::panels::ActivePanel;
use super::CbtopApp;
use crate::error::CbtopError;

impl CbtopApp {
    /// Render the UI
    #[allow(clippy::wildcard_enum_match_arm)]
    pub(super) fn render(&mut self) -> Result<(), CbtopError> {
        use presentar_terminal::direct::CellBuffer;

        // Resize buffer if needed
        let (width, height) = crossterm::terminal::size()?;
        if self.buffer.width() != width || self.buffer.height() != height {
            self.buffer = CellBuffer::new(width, height);
        }

        // Extract state needed for rendering
        let active_panel = self.active_panel;
        let is_running = self.is_running;
        let intensity = self.intensity;
        let backend = self.backend;
        let show_fps = self.config.show_fps;
        let problem_size = self.problem_size;
        let _frame_count = self.frame_count;
        let cpu_data: Vec<f64> = self.cpu_history.iter().copied().collect();
        let bricks_data: Vec<f64> = self.bricks_history.iter().copied().collect();
        let cpu_avg = self.cpu_history.mean();
        let frame_avg = self.frame_times.mean();
        let hardware = self.hardware.clone();
        let metrics = self.load_metrics.clone();

        {
            let mut canvas = DirectTerminalCanvas::new(&mut self.buffer);

            // Background
            canvas.fill_rect(
                Rect::new(0.0, 0.0, width as f32, height as f32),
                self.theme.background,
            );

            // Title bar with hardware info
            Self::render_title_bar(&mut canvas, width, active_panel, &hardware, &self.theme);

            // Main content - REAL metrics display
            Self::render_main_content(
                &mut canvas,
                width,
                height,
                is_running,
                &metrics,
                &cpu_data,
                &bricks_data,
                cpu_avg,
                problem_size,
                &hardware,
                &self.theme,
            );

            // Status bar with real metrics
            Self::render_status_bar(
                &mut canvas,
                width,
                height,
                is_running,
                intensity,
                backend,
                show_fps,
                frame_avg,
                &metrics,
                &self.theme,
            );
        }

        // Flush to terminal
        let mut output = Vec::with_capacity(16384);
        self.renderer
            .flush(&mut self.buffer, &mut output)
            .map_err(|e| CbtopError::Render(e.to_string()))?;
        std::io::Write::write_all(&mut std::io::stdout(), &output)?;

        Ok(())
    }

    pub(super) fn render_title_bar(
        canvas: &mut DirectTerminalCanvas,
        width: u16,
        active_panel: ActivePanel,
        hardware: &HardwareInfo,
        theme: &Theme,
    ) {
        let title_style = TextStyle {
            color: theme.foreground,
            ..Default::default()
        };

        // Title with hardware info
        let hw_info = format!(
            " cbtop │ {} ({} cores, {}) │ {:.0}GB RAM ",
            hardware.cpu_model.chars().take(30).collect::<String>(),
            hardware.cpu_cores,
            hardware.simd_type,
            hardware.memory_gb,
        );
        canvas.draw_text(&hw_info, Point::new(0.0, 0.0), &title_style);

        // GPU info if available
        if let Some(ref gpu) = hardware.gpu_name {
            let gpu_style = TextStyle {
                color: theme.cpu.sample(0.5),
                ..Default::default()
            };
            canvas.draw_text(
                &format!("│ GPU: {} ", gpu.chars().take(25).collect::<String>()),
                Point::new(hw_info.len() as f32, 0.0),
                &gpu_style,
            );
        }

        // PMAT-012 UI-10: Panel navigation tab bar
        Self::render_tab_bar(canvas, width, active_panel, theme);
    }

    /// Render panel navigation tab bar (UI-10)
    pub(super) fn render_tab_bar(
        canvas: &mut DirectTerminalCanvas,
        width: u16,
        active_panel: ActivePanel,
        theme: &Theme,
    ) {
        let dim_style = TextStyle {
            color: theme.dim,
            ..Default::default()
        };
        let active_style = TextStyle {
            color: theme.foreground,
            ..Default::default()
        };
        let highlight_style = TextStyle {
            color: theme.cpu.sample(0.2),
            ..Default::default()
        };

        // Build tab bar string with highlighting
        let mut x: f32 = 1.0;
        canvas.draw_text("│", Point::new(0.0, 1.0), &dim_style);

        for panel in ActivePanel::all() {
            let key = panel.key_number();
            let title = panel.title();
            let is_active = *panel == active_panel;

            if is_active {
                // Active panel: highlighted with brackets
                canvas.draw_text("[", Point::new(x, 1.0), &highlight_style);
                x += 1.0;
                canvas.draw_text(&format!("{}", key), Point::new(x, 1.0), &highlight_style);
                x += 1.0;
                canvas.draw_text(":", Point::new(x, 1.0), &highlight_style);
                x += 1.0;
                canvas.draw_text(title, Point::new(x, 1.0), &active_style);
                x += title.len() as f32;
                canvas.draw_text("]", Point::new(x, 1.0), &highlight_style);
                x += 1.0;
            } else {
                // Inactive panel: dimmed
                canvas.draw_text(&format!(" {}:", key), Point::new(x, 1.0), &dim_style);
                x += 3.0;
                canvas.draw_text(title, Point::new(x, 1.0), &dim_style);
                x += title.len() as f32;
            }

            // Separator between panels (except last)
            if *panel != ActivePanel::Help {
                canvas.draw_text(" ", Point::new(x, 1.0), &dim_style);
                x += 1.0;
            }
        }

        // Fill remaining width with spaces and close
        let remaining = (width as f32 - x - 1.0).max(0.0) as usize;
        if remaining > 0 {
            canvas.draw_text(&" ".repeat(remaining), Point::new(x, 1.0), &dim_style);
        }
        canvas.draw_text("│", Point::new(width as f32 - 1.0, 1.0), &dim_style);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_main_content(
        canvas: &mut DirectTerminalCanvas,
        width: u16,
        height: u16,
        is_running: bool,
        metrics: &LoadMetrics,
        cpu_data: &[f64],
        bricks_data: &[f64],
        cpu_avg: f64,
        problem_size: usize,
        hardware: &HardwareInfo,
        theme: &Theme,
    ) {
        let dim_style = TextStyle {
            color: theme.dim,
            ..Default::default()
        };
        let bright_style = TextStyle {
            color: theme.foreground,
            ..Default::default()
        };
        let accent_style = TextStyle {
            color: theme.cpu.sample(0.3),
            ..Default::default()
        };

        // PMAT-012 UI-01: Responsive width boxes
        let box_width = (width as usize).saturating_sub(2).max(40);
        let inner_width = box_width.saturating_sub(2);
        let bar_width = inner_width.saturating_sub(22).max(10);

        // Header line 2: Load status
        let status = if is_running {
            "● RUNNING"
        } else {
            "○ STOPPED"
        };
        let status_color = if is_running {
            theme.cpu.sample(0.0)
        } else {
            theme.dim
        };
        canvas.draw_text(
            &format!(" Load: {} ", status),
            Point::new(0.0, 2.0),
            &TextStyle {
                color: status_color,
                ..Default::default()
            },
        );

        // Metrics box with responsive width
        let box_top = format!(
            "┌─ Real-Time Metrics {}┐",
            "─".repeat(inner_width.saturating_sub(20))
        );
        canvas.draw_text(&box_top, Point::new(1.0, 3.0), &dim_style);

        // CPU Usage (REAL from /proc/stat) with color gradient
        let cpu_bar = Self::make_bar(cpu_avg, 100.0, bar_width);
        canvas.draw_text("│ CPU Usage:     ", Point::new(1.0, 4.0), &dim_style);
        canvas.draw_text(
            &cpu_bar,
            Point::new(17.0, 4.0),
            &TextStyle {
                color: theme.cpu_color(cpu_avg),
                ..Default::default()
            },
        );
        let cpu_val_x = 17.0 + bar_width as f32 + 1.0;
        canvas.draw_text(
            &format!("{:5.1}%", cpu_avg),
            Point::new(cpu_val_x, 4.0),
            &bright_style,
        );
        canvas.draw_text(" │", Point::new(box_width as f32, 4.0), &dim_style);

        // Bricks/Second with color gradient based on rate
        let bps = metrics.bricks_per_second;
        let bps_normalized = (bps / 10000.0).min(1.0) * 100.0;
        let bps_bar = Self::make_bar(bps.min(10000.0), 10000.0, bar_width);
        canvas.draw_text("│ Bricks/sec:    ", Point::new(1.0, 5.0), &dim_style);
        canvas.draw_text(
            &bps_bar,
            Point::new(17.0, 5.0),
            &TextStyle {
                color: theme.cpu_color(bps_normalized),
                ..Default::default()
            },
        );
        canvas.draw_text(
            &format!("{:>7.0}", bps),
            Point::new(cpu_val_x, 5.0),
            &bright_style,
        );
        canvas.draw_text(" │", Point::new(box_width as f32, 5.0), &dim_style);

        // Total Bricks executed
        let total_str = format!(
            "{:>width$}",
            Self::format_number(metrics.total_bricks),
            width = inner_width.saturating_sub(17)
        );
        canvas.draw_text("│ Total Bricks:  ", Point::new(1.0, 6.0), &dim_style);
        canvas.draw_text(&total_str, Point::new(17.0, 6.0), &bright_style);
        canvas.draw_text(" │", Point::new(box_width as f32, 6.0), &dim_style);

        // Avg Latency
        let latency_str = format!(
            "{:>width$.1} μs",
            metrics.avg_latency_us,
            width = inner_width.saturating_sub(20)
        );
        canvas.draw_text("│ Avg Latency:   ", Point::new(1.0, 7.0), &dim_style);
        canvas.draw_text(&latency_str, Point::new(17.0, 7.0), &bright_style);
        canvas.draw_text(" │", Point::new(box_width as f32, 7.0), &dim_style);

        // Problem size
        let size_str = format!(
            "{:>width$} elements",
            Self::format_number(problem_size as u64),
            width = inner_width.saturating_sub(26)
        );
        canvas.draw_text("│ Problem Size:  ", Point::new(1.0, 8.0), &dim_style);
        canvas.draw_text(&size_str, Point::new(17.0, 8.0), &bright_style);
        canvas.draw_text(" │", Point::new(box_width as f32, 8.0), &dim_style);

        // Throughput
        let gflops = metrics.ops_per_second / 1_000_000_000.0;
        let gbps = metrics.bytes_per_second / 1_073_741_824.0;
        canvas.draw_text("│ Throughput:    ", Point::new(1.0, 9.0), &dim_style);
        canvas.draw_text(
            &format!("{:>10.2} GFLOP/s │ {:>10.2} GB/s", gflops, gbps),
            Point::new(17.0, 9.0),
            &accent_style,
        );
        canvas.draw_text(" │", Point::new(box_width as f32, 9.0), &dim_style);

        let box_bottom = format!("└{}┘", "─".repeat(inner_width));
        canvas.draw_text(&box_bottom, Point::new(1.0, 10.0), &dim_style);

        // PMAT-012 UI-02: Per-core CPU bars
        let mut current_y = 12.0_f32;
        if !metrics.per_core_usage.is_empty() && height > 20 {
            Self::render_per_core_cpu(
                canvas,
                &mut current_y,
                height,
                metrics,
                inner_width,
                box_width,
                theme,
                &dim_style,
            );
        } else {
            Self::render_hardware_box(
                canvas,
                &mut current_y,
                hardware,
                inner_width,
                box_width,
                theme,
                &dim_style,
            );
        }

        // PMAT-012 UI-04: Memory breakdown box
        if metrics.memory.total_kb > 0 && current_y < height as f32 - 12.0 {
            Self::render_memory_box(
                canvas,
                &mut current_y,
                metrics,
                inner_width,
                box_width,
                bar_width,
                theme,
                &dim_style,
                &bright_style,
            );
        }

        // PMAT-012 F405: GPU panel when NVIDIA present
        if hardware.gpu_name.is_some() && current_y < height as f32 - 10.0 {
            Self::render_gpu_box(
                canvas,
                &mut current_y,
                hardware,
                inner_width,
                box_width,
                theme,
                &dim_style,
            );
        }

        // PMAT-012 UI-07 P2: Network TX/RX panel
        if (metrics.network.rx_rate > 0.0 || metrics.network.tx_rate > 0.0)
            && current_y < height as f32 - 8.0
        {
            Self::render_network_box(
                canvas,
                &mut current_y,
                metrics,
                inner_width,
                box_width,
                &dim_style,
            );
        }

        // PMAT-012 UI-08 P2: Disk per-mount panel
        if !metrics.disks.is_empty() && current_y < height as f32 - 6.0 {
            Self::render_disk_box(
                canvas,
                &mut current_y,
                height,
                metrics,
                inner_width,
                box_width,
                theme,
                &dim_style,
            );
        }

        // PMAT-012 UI-09: Sparklines
        let sparkline_width = (width as usize).saturating_sub(6).max(10);
        Self::render_sparklines(
            canvas,
            &mut current_y,
            height,
            cpu_data,
            bricks_data,
            inner_width,
            box_width,
            sparkline_width,
            theme,
            &dim_style,
            &accent_style,
        );

        // Controls reminder
        if height > 10 {
            canvas.draw_text(
                " [Space] Toggle load  [+/-] Intensity  [b] Backend  [w] Workload  [r] Reset  [q] Quit ",
                Point::new(1.0, height as f32 - 3.0),
                &dim_style,
            );
        }
    }
}
