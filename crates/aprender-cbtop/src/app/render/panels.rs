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
        let core_box_top = format!(
            "┌─ Per-Core CPU {}┐",
            "─".repeat(inner_width.saturating_sub(15))
        );
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
                    &TextStyle {
                        color: theme.cpu_color(usage),
                        ..Default::default()
                    },
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
        let hw_box_top = format!(
            "┌─ Hardware {}┐",
            "─".repeat(inner_width.saturating_sub(12))
        );
        canvas.draw_text(&hw_box_top, Point::new(1.0, *current_y), dim_style);
        *current_y += 1.0;

        let cpu_info = format!(
            "│ CPU:  {:width$} │",
            hardware
                .cpu_model
                .chars()
                .take(inner_width.saturating_sub(10))
                .collect::<String>(),
            width = inner_width.saturating_sub(8)
        );
        canvas.draw_text(&cpu_info, Point::new(1.0, *current_y), dim_style);
        *current_y += 1.0;

        canvas.draw_text(
            &format!(
                "│ Cores: {} │ SIMD: {} ",
                hardware.cpu_cores, hardware.simd_type
            ),
            Point::new(1.0, *current_y),
            dim_style,
        );
        canvas.draw_text("│", Point::new(box_width as f32, *current_y), dim_style);
        *current_y += 1.0;

        if let Some(ref gpu) = hardware.gpu_name {
            let gpu_str = format!(
                "│ GPU:  {:width$} │",
                gpu.chars()
                    .take(inner_width.saturating_sub(10))
                    .collect::<String>(),
                width = inner_width.saturating_sub(8)
            );
            canvas.draw_text(&gpu_str, Point::new(1.0, *current_y), dim_style);
        } else {
            canvas.draw_text(
                "│ GPU:  Not detected ",
                Point::new(1.0, *current_y),
                dim_style,
            );
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
            &TextStyle {
                color: theme.memory_color(mem_pct),
                ..Default::default()
            },
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
                gpu.chars()
                    .take(inner_width.saturating_sub(4))
                    .collect::<String>(),
                width = inner_width.saturating_sub(2)
            );
            canvas.draw_text(
                &gpu_str,
                Point::new(1.0, *current_y),
                &TextStyle {
                    color: theme.gpu_color(50.0),
                    ..Default::default()
                },
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
                &TextStyle {
                    color: theme.memory_color(disk.usage_percent),
                    ..Default::default()
                },
            );
            canvas.draw_text(
                &format!(
                    " {:>6}/{:>6} ({:.0}%)",
                    used_str, total_str, disk.usage_percent
                ),
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
            let spark_box_top = format!(
                "┌─ CPU History (braille) {}┐",
                "─".repeat(inner_width.saturating_sub(23))
            );
            canvas.draw_text(&spark_box_top, Point::new(1.0, *current_y), dim_style);
            *current_y += 1.0;

            // Use braille for higher resolution (2x data density)
            let braille_line = Self::make_braille_sparkline(cpu_data, sparkline_width);
            canvas.draw_text("│ ", Point::new(1.0, *current_y), dim_style);
            canvas.draw_text(
                &braille_line,
                Point::new(3.0, *current_y),
                &TextStyle {
                    color: theme.cpu.sample(0.3),
                    ..Default::default()
                },
            );
            canvas.draw_text(" │", Point::new(box_width as f32, *current_y), dim_style);
            *current_y += 1.0;

            let spark_box_bottom = format!("└{}┘", "─".repeat(inner_width));
            canvas.draw_text(&spark_box_bottom, Point::new(1.0, *current_y), dim_style);
            *current_y += 2.0;
        }

        // Bricks/sec braille sparkline
        if !bricks_data.is_empty() && *current_y < height as f32 - 5.0 {
            let brick_box_top = format!(
                "┌─ Bricks/sec (braille) {}┐",
                "─".repeat(inner_width.saturating_sub(24))
            );
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

#[cfg(test)]
mod render_sparklines_tests {
    use super::*;
    use presentar_terminal::direct::CellBuffer;

    const W: u16 = 80;
    const H: u16 = 40;

    fn row_text(buf: &CellBuffer, y: u16) -> String {
        (0..W)
            .map(|x| buf.get(x, y).map_or(" ", |c| c.symbol.as_str()))
            .collect()
    }

    /// Run `render_sparklines` on a blank buffer; returns (buffer, final current_y).
    fn run(start_y: f32, height: u16, cpu: &[f64], bricks: &[f64]) -> (CellBuffer, f32) {
        let mut buf = CellBuffer::new(W, H);
        let theme = Theme::default();
        let style = TextStyle::default();
        let mut y = start_y;
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buf);
            CbtopApp::render_sparklines(
                &mut canvas,
                &mut y,
                height,
                cpu,
                bricks,
                60,
                62,
                40,
                &theme,
                &style,
                &style,
            );
        }
        (buf, y)
    }

    #[test]
    fn empty_data_draws_nothing_and_keeps_y() {
        let (buf, y) = run(2.0, H, &[], &[]);
        assert_eq!(y, 2.0);
        for row in 0..H {
            assert!(!row_text(&buf, row).contains('┌'), "row {row} drew a box");
        }
    }

    #[test]
    fn cpu_box_draws_header_and_advances_four() {
        let (buf, y) = run(2.0, H, &[0.5, 0.9], &[]);
        assert_eq!(y, 6.0);
        assert!(row_text(&buf, 2).contains("CPU History (braille)"));
        assert!(row_text(&buf, 3).contains('│'));
        assert!(row_text(&buf, 4).contains('└'));
        assert!(!row_text(&buf, 6).contains("Bricks"));
    }

    #[test]
    fn cpu_box_skipped_when_too_close_to_bottom() {
        // threshold is current_y < height - 8
        let (_, y) = run(32.0, H, &[0.5], &[]);
        assert_eq!(y, 32.0);
        let (buf, y) = run(31.0, H, &[0.5], &[]);
        assert_eq!(y, 35.0);
        assert!(row_text(&buf, 31).contains("CPU History"));
    }

    #[test]
    fn bricks_box_has_own_threshold_and_no_trailing_advance() {
        // threshold is current_y < height - 5
        let (_, y) = run(35.0, H, &[], &[1.0]);
        assert_eq!(y, 35.0);
        let (buf, y) = run(34.0, H, &[], &[1.0]);
        assert_eq!(y, 36.0);
        assert!(row_text(&buf, 34).contains("Bricks/sec (braille)"));
        assert!(row_text(&buf, 36).contains('└'));
    }

    #[test]
    fn both_boxes_stack_cpu_then_bricks() {
        let (buf, y) = run(2.0, H, &[0.5], &[2.0]);
        assert!(row_text(&buf, 2).contains("CPU History"));
        assert!(row_text(&buf, 6).contains("Bricks/sec"));
        assert_eq!(y, 8.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use presentar_terminal::direct::CellBuffer;

    const W: u16 = 60;
    const EXPECTED_ROW: &str = " │ /data          ▓▓▓▓▓░░░░░    5.0G/ 10.0G (50%) │";

    fn disk(mount: &str, used: u64, total: u64, pct: f64) -> DiskMetrics {
        DiskMetrics {
            mount: mount.to_string(),
            total_bytes: total,
            used_bytes: used,
            usage_percent: pct,
        }
    }

    /// Render the disk box into a fresh buffer; returns (rows, final y).
    fn render(disks: Vec<DiskMetrics>, height: u16) -> (Vec<String>, f32) {
        let mut buf = CellBuffer::new(W, 12);
        let metrics = LoadMetrics {
            disks,
            ..Default::default()
        };
        let mut y = 0.0_f32;
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buf);
            CbtopApp::render_disk_box(
                &mut canvas,
                &mut y,
                height,
                &metrics,
                40,
                50,
                &Theme::default(),
                &TextStyle::default(),
            );
        }
        let rows = (0..12)
            .map(|r| {
                (0..W)
                    .filter_map(|c| buf.get(c, r).map(|cell| cell.symbol.to_string()))
                    .collect::<String>()
            })
            .collect();
        (rows, y)
    }

    #[test]
    fn disk_box_draws_frame_row_and_advances_y() {
        let (rows, y) = render(vec![disk("/data", 5 << 30, 10 << 30, 50.0)], 40);
        assert!(rows[0].contains("┌─ Disks "), "{:?}", rows[0]);
        assert!(rows[1].contains("/data"), "{:?}", rows[1]);
        assert!(rows[1].contains("(50%)"), "{:?}", rows[1]);
        assert!(rows[1].contains("▓▓▓▓▓░░░░░"), "{:?}", rows[1]);
        assert!(
            rows[2].contains('└') && rows[2].contains('┘'),
            "{:?}",
            rows[2]
        );
        // top 1 + one disk row 1 + bottom 1 + 1 blank line.
        assert!((y - 4.0).abs() < f32::EPSILON, "y={y}");
    }

    #[test]
    fn disk_box_without_disks_is_just_the_frame() {
        let (rows, y) = render(Vec::new(), 40);
        assert!(rows[0].contains("Disks"));
        assert!(rows[1].contains('└'));
        assert!((y - 3.0).abs() < f32::EPSILON, "y={y}");
    }

    #[test]
    fn disk_box_shows_at_most_three_disks() {
        let disks = (0..5)
            .map(|i| disk(&format!("/m{i}"), 1 << 30, 2 << 30, 50.0))
            .collect();
        let (rows, y) = render(disks, 40);
        assert!(rows[3].contains("/m2"), "{:?}", rows[3]);
        assert!(!rows.iter().any(|r| r.contains("/m3")));
        assert!(rows[4].contains('└'), "{:?}", rows[4]);
        assert!((y - 6.0).abs() < f32::EPSILON, "y={y}");
    }

    #[test]
    fn disk_box_truncates_mount_to_fifteen_chars() {
        let (rows, _) = render(vec![disk("/abcdefghijklmnopqrst", 1, 2, 50.0)], 40);
        assert!(rows[1].contains("/abcdefghijklmn"), "{:?}", rows[1]);
        assert!(!rows[1].contains("/abcdefghijklmno"), "{:?}", rows[1]);
    }

    #[test]
    fn disk_box_stops_rows_when_height_is_exhausted() {
        // height 7 => break once y > 2.0: the second row is the last drawn.
        let disks = (0..3)
            .map(|i| disk(&format!("/m{i}"), 1 << 30, 2 << 30, 50.0))
            .collect();
        let (rows, y) = render(disks, 7);
        assert!(rows[2].contains("/m1"), "{:?}", rows[2]);
        assert!(!rows.iter().any(|r| r.contains("/m2")));
        assert!(rows[3].contains('└'), "{:?}", rows[3]);
        assert!((y - 5.0).abs() < f32::EPSILON, "y={y}");
    }

    #[test]
    fn disk_box_row_layout_is_exact() {
        // Pins used/total text, bar width (10 cells), the right border column and
        // the frame width: any drift in one of them changes the whole row.
        let (rows, _) = render(vec![disk("/data", 5 << 30, 10 << 30, 50.0)], 40);
        assert_eq!(rows[1].trim_end(), EXPECTED_ROW, "{:?}", rows[1]);
        assert_eq!(
            rows[0].trim_end(),
            format!(" ┌─ Disks {}┐", "─".repeat(30)),
            "{:?}",
            rows[0]
        );
        assert_eq!(
            rows[2].trim_end(),
            format!(" └{}┘", "─".repeat(40)),
            "{:?}",
            rows[2]
        );
    }

    #[test]
    fn disk_box_bar_uses_memory_colour_for_usage() {
        let theme = Theme::default();
        for pct in [10.0_f64, 95.0] {
            let mut buf = CellBuffer::new(W, 12);
            let metrics = LoadMetrics {
                disks: vec![disk("/d", 1, 2, pct)],
                ..Default::default()
            };
            let mut y = 0.0_f32;
            {
                let mut canvas = DirectTerminalCanvas::new(&mut buf);
                CbtopApp::render_disk_box(
                    &mut canvas,
                    &mut y,
                    40,
                    &metrics,
                    40,
                    50,
                    &theme,
                    &TextStyle::default(),
                );
            }
            let want = theme.memory_color(pct);
            let got = buf.get(19, 1).expect("bar cell").fg;
            assert_eq!(got, want, "bar colour at {pct}%");
        }
        assert_ne!(theme.memory_color(10.0), theme.memory_color(95.0));
    }
}

#[cfg(test)]
mod hardware_box_tests {
    use super::*;
    use presentar_terminal::direct::CellBuffer;

    const W: u16 = 60;

    fn hw(gpu: Option<&str>, cpu: &str) -> HardwareInfo {
        HardwareInfo {
            cpu_model: cpu.to_string(),
            cpu_cores: 8,
            simd_type: "AVX2",
            gpu_name: gpu.map(str::to_string),
            memory_gb: 31.5,
        }
    }

    /// Draw the box into a fresh buffer; returns its rows (trailing blanks trimmed) and the final y.
    fn draw(hardware: &HardwareInfo, inner: usize) -> (Vec<String>, f32) {
        let mut buf = CellBuffer::new(W, 12);
        let mut y = 0.0_f32;
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buf);
            CbtopApp::render_hardware_box(
                &mut canvas,
                &mut y,
                hardware,
                inner,
                inner + 2,
                &Theme::tokyo_night(),
                &TextStyle::default(),
            );
        }
        let rows = (0..12u16)
            .map(|r| {
                let line: String = (0..W)
                    .map(|c| {
                        buf.get(c, r)
                            .map_or(" ", |cell| cell.symbol.as_str())
                            .to_string()
                    })
                    .collect();
                line.trim_end().to_string()
            })
            .collect();
        (rows, y)
    }

    #[test]
    fn box_rows_with_gpu_and_cursor_advance() {
        let (rows, y) = draw(&hw(Some("RTX 4090"), "Test CPU"), 40);
        assert!(
            rows[0].trim_start().starts_with("┌─ Hardware ") && rows[0].ends_with('┐'),
            "{:?}",
            rows[0]
        );
        assert!(rows[1].contains("CPU:  Test CPU") && rows[1].ends_with('│'));
        assert!(rows[2].contains("Cores: 8 │ SIMD: AVX2"));
        assert!(rows[3].contains("GPU:  RTX 4090") && !rows[3].contains("Not detected"));
        assert!(rows[4].contains("RAM:  31.5 GB"));
        assert!(rows[5].trim_start().starts_with('└') && rows[5].ends_with('┘'));
        // 5 content rows plus 2 trailing blank rows
        assert_eq!(y, 7.0);
    }

    #[test]
    fn missing_gpu_prints_not_detected_with_closing_bar_at_box_width() {
        let (rows, _) = draw(&hw(None, "Test CPU"), 40);
        assert!(rows[3].contains("GPU:  Not detected"));
        assert!(!rows[3].contains("RTX"));
        // the closing bar sits at x = box_width (42)
        assert_eq!(rows[3].chars().nth(42), Some('│'), "{:?}", rows[3]);
    }

    #[test]
    fn long_cpu_and_gpu_names_are_truncated_to_inner_width() {
        let long = "X".repeat(100);
        let (rows, _) = draw(&hw(Some(&long), &long), 30);
        assert_eq!(rows[1].matches('X').count(), 20);
        assert_eq!(rows[3].matches('X').count(), 20);
    }

    #[test]
    fn tiny_inner_width_saturates_instead_of_underflowing() {
        let (rows, y) = draw(&hw(Some("GPU"), "CPU"), 3);
        assert!(rows[0].trim_start().starts_with("┌─ Hardware "));
        // take(0): only the "CPU:" label survives, the model name is dropped
        assert_eq!(rows[1].matches("CPU").count(), 1);
        assert_eq!(y, 7.0);
    }
}
