//! Hardware panel rendering methods for the cbtop TUI.
//!
//! Contains per-core CPU, hardware info, memory, GPU, network, disk,
//! and sparkline panel rendering.

use presentar_core::{Canvas, Point, TextStyle};
use presentar_terminal::direct::DirectTerminalCanvas;
use presentar_terminal::Theme;

use crate::app::hardware::{
    DiskMetrics, HardwareInfo, LoadMetrics, MemoryBreakdown, NetworkMetrics,
};
use crate::app::CbtopApp;

impl CbtopApp {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::app) fn render_per_core_cpu(
        canvas: &mut DirectTerminalCanvas,
        current_y: &mut f32,
        height: u16,
        metrics: &LoadMetrics,
        inner_width: usize,
        box_width: usize,
        theme: &Theme,
        dim_style: &TextStyle,
    ) {
        let core_box_top =
            format!("┌─ Per-Core CPU {}┐", "─".repeat(inner_width.saturating_sub(15)));
        canvas.draw_text(&core_box_top, Point::new(1.0, *current_y), dim_style);
        *current_y += 1.0;

        // Render cores in rows of 4 for compact display
        let cores_per_row = 4.min(metrics.per_core_usage.len());
        let mini_bar_width = (inner_width / cores_per_row).saturating_sub(10).max(5);

        for (i, chunk) in metrics.per_core_usage.chunks(cores_per_row).enumerate() {
            let mut row = String::from("│ ");
            for (j, &usage) in chunk.iter().enumerate() {
                let core_num = i * cores_per_row + j;
                let mini_bar = Self::make_mini_bar(usage, 100.0, mini_bar_width);
                row.push_str(&format!("C{:02}:{} ", core_num, mini_bar));
            }
            // Pad to box width
            while row.len() < box_width {
                row.push(' ');
            }
            row.push('│');

            // Draw with per-core color gradient
            canvas.draw_text(&row[..3], Point::new(1.0, *current_y), dim_style);
            let mut x_pos = 4.0;
            for &usage in chunk.iter() {
                let bar_str = Self::make_mini_bar(usage, 100.0, mini_bar_width);
                canvas.draw_text(
                    &bar_str,
                    Point::new(x_pos + 4.0, *current_y),
                    &TextStyle { color: theme.cpu_color(usage), ..Default::default() },
                );
                x_pos += (mini_bar_width + 10) as f32;
            }
            canvas.draw_text("│", Point::new(box_width as f32, *current_y), dim_style);
            *current_y += 1.0;

            if *current_y > height as f32 - 10.0 {
                break; // Don't overflow screen
            }
        }

        let core_box_bottom = format!("└{}┘", "─".repeat(inner_width));
        canvas.draw_text(&core_box_bottom, Point::new(1.0, *current_y), dim_style);
        *current_y += 2.0;
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::app) fn render_hardware_box(
        canvas: &mut DirectTerminalCanvas,
        current_y: &mut f32,
        hardware: &HardwareInfo,
        inner_width: usize,
        box_width: usize,
        _theme: &Theme,
        dim_style: &TextStyle,
    ) {
        let hw_box_top = format!("┌─ Hardware {}┐", "─".repeat(inner_width.saturating_sub(12)));
        canvas.draw_text(&hw_box_top, Point::new(1.0, *current_y), dim_style);
        *current_y += 1.0;

        let cpu_info = format!(
            "│ CPU:  {:width$} │",
            hardware.cpu_model.chars().take(inner_width.saturating_sub(10)).collect::<String>(),
            width = inner_width.saturating_sub(8)
        );
        canvas.draw_text(&cpu_info, Point::new(1.0, *current_y), dim_style);
        *current_y += 1.0;

        canvas.draw_text(
            &format!("│ Cores: {} │ SIMD: {} ", hardware.cpu_cores, hardware.simd_type),
            Point::new(1.0, *current_y),
            dim_style,
        );
        canvas.draw_text("│", Point::new(box_width as f32, *current_y), dim_style);
        *current_y += 1.0;

        if let Some(ref gpu) = hardware.gpu_name {
            let gpu_str = format!(
                "│ GPU:  {:width$} │",
                gpu.chars().take(inner_width.saturating_sub(10)).collect::<String>(),
                width = inner_width.saturating_sub(8)
            );
            canvas.draw_text(&gpu_str, Point::new(1.0, *current_y), dim_style);
        } else {
            canvas.draw_text("│ GPU:  Not detected ", Point::new(1.0, *current_y), dim_style);
            canvas.draw_text("│", Point::new(box_width as f32, *current_y), dim_style);
        }
        *current_y += 1.0;

        canvas.draw_text(
            &format!("│ RAM:  {:.1} GB ", hardware.memory_gb),
            Point::new(1.0, *current_y),
            dim_style,
        );
        canvas.draw_text("│", Point::new(box_width as f32, *current_y), dim_style);
        *current_y += 1.0;

        let hw_box_bottom = format!("└{}┘", "─".repeat(inner_width));
        canvas.draw_text(&hw_box_bottom, Point::new(1.0, *current_y), dim_style);
        *current_y += 2.0;
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::app) fn render_memory_box(
        canvas: &mut DirectTerminalCanvas,
        current_y: &mut f32,
        metrics: &LoadMetrics,
        inner_width: usize,
        box_width: usize,
        bar_width: usize,
        theme: &Theme,
        dim_style: &TextStyle,
        bright_style: &TextStyle,
    ) {
        let mem_box_top = format!("┌─ Memory {}┐", "─".repeat(inner_width.saturating_sub(11)));
        canvas.draw_text(&mem_box_top, Point::new(1.0, *current_y), dim_style);
        *current_y += 1.0;

        // Memory usage bar with color gradient
        let mem_pct = metrics.memory.usage_percent();
        let mem_bar = Self::make_bar(mem_pct, 100.0, bar_width);
        canvas.draw_text("│ Used:    ", Point::new(1.0, *current_y), dim_style);
        canvas.draw_text(
            &mem_bar,
            Point::new(11.0, *current_y),
            &TextStyle { color: theme.memory_color(mem_pct), ..Default::default() },
        );
        let mem_val_x = 11.0 + bar_width as f32 + 1.0;
        canvas.draw_text(
            &format!("{:5.1}%", mem_pct),
            Point::new(mem_val_x, *current_y),
            bright_style,
        );
        canvas.draw_text(" │", Point::new(box_width as f32, *current_y), dim_style);
        *current_y += 1.0;

        // Memory breakdown values
        let total_str = MemoryBreakdown::format_kb(metrics.memory.total_kb);
        let used_str = MemoryBreakdown::format_kb(metrics.memory.used_kb);
        let cached_str = MemoryBreakdown::format_kb(metrics.memory.cached_kb);
        let buffers_str = MemoryBreakdown::format_kb(metrics.memory.buffers_kb);

        canvas.draw_text(
            &format!(
                "│ Total: {:>8}  Used: {:>8}  Cache: {:>8}  Buf: {:>6} ",
                total_str, used_str, cached_str, buffers_str
            ),
            Point::new(1.0, *current_y),
            dim_style,
        );
        canvas.draw_text("│", Point::new(box_width as f32, *current_y), dim_style);
        *current_y += 1.0;

        let mem_box_bottom = format!("└{}┘", "─".repeat(inner_width));
        canvas.draw_text(&mem_box_bottom, Point::new(1.0, *current_y), dim_style);
        *current_y += 2.0;
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::app) fn render_gpu_box(
        canvas: &mut DirectTerminalCanvas,
        current_y: &mut f32,
        hardware: &HardwareInfo,
        inner_width: usize,
        box_width: usize,
        theme: &Theme,
        dim_style: &TextStyle,
    ) {
        let gpu_box_top = format!("┌─ GPU {}┐", "─".repeat(inner_width.saturating_sub(7)));
        canvas.draw_text(&gpu_box_top, Point::new(1.0, *current_y), dim_style);
        *current_y += 1.0;

        // GPU name
        if let Some(ref gpu) = hardware.gpu_name {
            let gpu_str = format!(
                "│ {:width$} │",
                gpu.chars().take(inner_width.saturating_sub(4)).collect::<String>(),
                width = inner_width.saturating_sub(2)
            );
            canvas.draw_text(
                &gpu_str,
                Point::new(1.0, *current_y),
                &TextStyle { color: theme.gpu_color(50.0), ..Default::default() },
            );
        }
        *current_y += 1.0;

        // GPU status hint
        canvas.draw_text(
            "│ Status: Ready for CUDA workloads ",
            Point::new(1.0, *current_y),
            dim_style,
        );
        canvas.draw_text("│", Point::new(box_width as f32, *current_y), dim_style);
        *current_y += 1.0;

        let gpu_box_bottom = format!("└{}┘", "─".repeat(inner_width));
        canvas.draw_text(&gpu_box_bottom, Point::new(1.0, *current_y), dim_style);
        *current_y += 2.0;
    }

    pub(in crate::app) fn render_network_box(
        canvas: &mut DirectTerminalCanvas,
        current_y: &mut f32,
        metrics: &LoadMetrics,
        inner_width: usize,
        box_width: usize,
        dim_style: &TextStyle,
    ) {
        let net_box_top = format!("┌─ Network {}┐", "─".repeat(inner_width.saturating_sub(11)));
        canvas.draw_text(&net_box_top, Point::new(1.0, *current_y), dim_style);
        *current_y += 1.0;

        let rx_str = NetworkMetrics::format_rate(metrics.network.rx_rate);
        let tx_str = NetworkMetrics::format_rate(metrics.network.tx_rate);
        canvas.draw_text(
            &format!("│ ↓ RX: {:>12}  ↑ TX: {:>12} ", rx_str, tx_str),
            Point::new(1.0, *current_y),
            dim_style,
        );
        canvas.draw_text("│", Point::new(box_width as f32, *current_y), dim_style);
        *current_y += 1.0;

        let net_box_bottom = format!("└{}┘", "─".repeat(inner_width));
        canvas.draw_text(&net_box_bottom, Point::new(1.0, *current_y), dim_style);
        *current_y += 2.0;
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::app) fn render_disk_box(
        canvas: &mut DirectTerminalCanvas,
        current_y: &mut f32,
        height: u16,
        metrics: &LoadMetrics,
        inner_width: usize,
        box_width: usize,
        theme: &Theme,
        dim_style: &TextStyle,
    ) {
        let disk_box_top = format!("┌─ Disks {}┐", "─".repeat(inner_width.saturating_sub(10)));
        canvas.draw_text(&disk_box_top, Point::new(1.0, *current_y), dim_style);
        *current_y += 1.0;

        for disk in metrics.disks.iter().take(3) {
            let used_str = DiskMetrics::format_bytes(disk.used_bytes);
            let total_str = DiskMetrics::format_bytes(disk.total_bytes);
            let mount_short: String = disk.mount.chars().take(15).collect();
            let disk_bar = Self::make_mini_bar(disk.usage_percent, 100.0, 10);

            canvas.draw_text(
                &format!("│ {:15} ", mount_short),
                Point::new(1.0, *current_y),
                dim_style,
            );
            canvas.draw_text(
                &disk_bar,
                Point::new(18.0, *current_y),
                &TextStyle { color: theme.memory_color(disk.usage_percent), ..Default::default() },
            );
            canvas.draw_text(
                &format!(" {:>6}/{:>6} ({:.0}%)", used_str, total_str, disk.usage_percent),
                Point::new(29.0, *current_y),
                dim_style,
            );
            canvas.draw_text("│", Point::new(box_width as f32, *current_y), dim_style);
            *current_y += 1.0;

            if *current_y > height as f32 - 5.0 {
                break;
            }
        }

        let disk_box_bottom = format!("└{}┘", "─".repeat(inner_width));
        canvas.draw_text(&disk_box_bottom, Point::new(1.0, *current_y), dim_style);
        *current_y += 2.0;
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::app) fn render_sparklines(
        canvas: &mut DirectTerminalCanvas,
        current_y: &mut f32,
        height: u16,
        cpu_data: &[f64],
        bricks_data: &[f64],
        inner_width: usize,
        box_width: usize,
        sparkline_width: usize,
        theme: &Theme,
        dim_style: &TextStyle,
        accent_style: &TextStyle,
    ) {
        // PMAT-012 UI-05: Braille graphs for higher resolution sparklines
        if !cpu_data.is_empty() && *current_y < height as f32 - 8.0 {
            let spark_box_top =
                format!("┌─ CPU History (braille) {}┐", "─".repeat(inner_width.saturating_sub(23)));
            canvas.draw_text(&spark_box_top, Point::new(1.0, *current_y), dim_style);
            *current_y += 1.0;

            // Use braille for higher resolution (2x data density)
            let braille_line = Self::make_braille_sparkline(cpu_data, sparkline_width);
            canvas.draw_text("│ ", Point::new(1.0, *current_y), dim_style);
            canvas.draw_text(
                &braille_line,
                Point::new(3.0, *current_y),
                &TextStyle { color: theme.cpu.sample(0.3), ..Default::default() },
            );
            canvas.draw_text(" │", Point::new(box_width as f32, *current_y), dim_style);
            *current_y += 1.0;

            let spark_box_bottom = format!("└{}┘", "─".repeat(inner_width));
            canvas.draw_text(&spark_box_bottom, Point::new(1.0, *current_y), dim_style);
            *current_y += 2.0;
        }

        // Bricks/sec braille sparkline
        if !bricks_data.is_empty() && *current_y < height as f32 - 5.0 {
            let brick_box_top =
                format!("┌─ Bricks/sec (braille) {}┐", "─".repeat(inner_width.saturating_sub(24)));
            canvas.draw_text(&brick_box_top, Point::new(1.0, *current_y), dim_style);
            *current_y += 1.0;

            let braille_line = Self::make_braille_sparkline(bricks_data, sparkline_width);
            canvas.draw_text("│ ", Point::new(1.0, *current_y), dim_style);
            canvas.draw_text(&braille_line, Point::new(3.0, *current_y), accent_style);
            canvas.draw_text(" │", Point::new(box_width as f32, *current_y), dim_style);
            *current_y += 1.0;

            let brick_box_bottom = format!("└{}┘", "─".repeat(inner_width));
            canvas.draw_text(&brick_box_bottom, Point::new(1.0, *current_y), dim_style);
        }
    }
}
