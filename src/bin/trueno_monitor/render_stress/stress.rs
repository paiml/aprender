//! Stress test tab rendering (idle screen and running visualization).

use presentar_core::{Canvas, Color, Point, Rect, TextStyle};
use presentar_terminal::direct::DirectTerminalCanvas;
use presentar_terminal::Theme;
use trueno_gpu::monitor::{ComputeDevice, PressureLevel};

use crate::render::{color_dark_gray, color_white, make_bar, make_sparkline};
use crate::{App, StressTestVerdict};

fn color_red() -> Color {
    Color::new(1.0, 0.3, 0.3, 1.0)
}

fn color_green() -> Color {
    Color::new(0.3, 1.0, 0.3, 1.0)
}

fn color_yellow() -> Color {
    Color::new(1.0, 1.0, 0.3, 1.0)
}

fn color_cyan() -> Color {
    Color::new(0.3, 1.0, 1.0, 1.0)
}

fn color_magenta() -> Color {
    Color::new(1.0, 0.3, 1.0, 1.0)
}

fn color_black() -> Color {
    Color::new(0.0, 0.0, 0.0, 1.0)
}

fn color_orange() -> Color {
    Color::new(1.0, 0.65, 0.0, 1.0)
}

pub(crate) fn render_stress_tab(
    canvas: &mut DirectTerminalCanvas,
    app: &App,
    width: u16,
    height: u16,
    theme: &Theme,
) {
    if app.stress_running {
        render_stress_running(canvas, app, width, height, theme);
    } else {
        render_stress_idle(canvas, app, width, height, theme);
    }
}

fn render_stress_idle(
    canvas: &mut DirectTerminalCanvas,
    app: &App,
    width: u16,
    _height: u16,
    theme: &Theme,
) {
    let w = width as usize;
    let dim = TextStyle { color: color_dark_gray(), ..Default::default() };
    let white = TextStyle { color: color_white(), ..Default::default() };
    let cyan_bold = TextStyle { color: color_cyan(), ..Default::default() };
    let yellow_style = TextStyle { color: color_yellow(), ..Default::default() };
    let magenta = TextStyle { color: color_magenta(), ..Default::default() };
    let green_bold = TextStyle { color: color_green(), ..Default::default() };

    let mut y: f32 = 3.0;

    // Box top
    let top = format!(
        "\u{250c}\u{2500} Stress Test Mode (TRUENO-SPEC-025) {}\u{2510}",
        "\u{2500}".repeat(w.saturating_sub(39))
    );
    canvas.draw_text(&top, Point::new(0.0, y), &dim);
    y += 1.0;

    // Status: IDLE
    canvas.draw_text("\u{2502}", Point::new(0.0, y), &dim);
    y += 1.0;
    canvas.draw_text("\u{2502}  Status: ", Point::new(0.0, y), &dim);
    canvas.draw_text("IDLE", Point::new(12.0, y), &dim);
    y += 1.0;

    canvas.draw_text("\u{2502}", Point::new(0.0, y), &dim);
    y += 1.0;

    // Hardware Detected
    canvas.draw_text("\u{2502}  Hardware Detected:", Point::new(0.0, y), &cyan_bold);
    y += 1.0;
    canvas.draw_text("\u{2502}", Point::new(0.0, y), &dim);
    y += 1.0;

    // CPU
    canvas.draw_text("\u{2502}    CPU: ", Point::new(0.0, y), &dim);
    canvas.draw_text(app.cpu.device_name(), Point::new(10.0, y), &white);
    let cores_x = 10.0 + app.cpu.device_name().len() as f32;
    canvas.draw_text(&format!(" ({} cores)", num_cpus::get()), Point::new(cores_x, y), &dim);
    y += 1.0;

    // GPUs
    for (i, gpu) in app.gpus.iter().enumerate() {
        canvas.draw_text(&format!("\u{2502}    GPU{}: ", i), Point::new(0.0, y), &dim);
        let name_x = 4.0 + format!("GPU{}: ", i).len() as f32;
        canvas.draw_text(&gpu.info.name, Point::new(name_x, y), &magenta);
        y += 1.0;
    }

    canvas.draw_text("\u{2502}", Point::new(0.0, y), &dim);
    y += 1.0;

    // Stress Test Will
    canvas.draw_text("\u{2502}  Stress Test Will:", Point::new(0.0, y), &cyan_bold);
    y += 1.0;
    canvas.draw_text("\u{2502}", Point::new(0.0, y), &dim);
    y += 1.0;

    canvas.draw_text("\u{2502}    ", Point::new(0.0, y), &dim);
    canvas.draw_text("CPU:", Point::new(6.0, y), &yellow_style);
    canvas.draw_text(
        &format!(" {} threads doing FP math (sin/cos/sqrt)", num_cpus::get()),
        Point::new(10.0, y),
        &TextStyle { color: theme.foreground, ..Default::default() },
    );
    y += 1.0;

    canvas.draw_text("\u{2502}    ", Point::new(0.0, y), &dim);
    canvas.draw_text("MEM:", Point::new(6.0, y), &yellow_style);
    canvas.draw_text(
        " Allocate 512MB, touch every page",
        Point::new(10.0, y),
        &TextStyle { color: theme.foreground, ..Default::default() },
    );
    y += 1.0;

    for (i, gpu) in app.gpus.iter().enumerate() {
        canvas.draw_text("\u{2502}    ", Point::new(0.0, y), &dim);
        canvas.draw_text(&format!("GPU{}:", i), Point::new(6.0, y), &yellow_style);
        canvas.draw_text(
            &format!(" {} - H2D/D2H transfers (1M x f32)", gpu.info.name),
            Point::new(6.0 + format!("GPU{}:", i).len() as f32, y),
            &TextStyle { color: theme.foreground, ..Default::default() },
        );
        y += 1.0;
    }

    // Stress test report if available
    if let Some(ref report) = app.stress_report {
        let verdict_color = match report.verdict {
            StressTestVerdict::Pass => color_green(),
            StressTestVerdict::PassWithNotes => color_yellow(),
            StressTestVerdict::Fail => color_red(),
        };

        canvas.draw_text("\u{2502}", Point::new(0.0, y), &dim);
        y += 1.0;
        canvas.draw_text("\u{2502}  Stress Test Report (renacer):", Point::new(0.0, y), &cyan_bold);
        y += 1.0;
        canvas.draw_text("\u{2502}", Point::new(0.0, y), &dim);
        y += 1.0;

        canvas.draw_text("\u{2502}    Verdict: ", Point::new(0.0, y), &dim);
        canvas.draw_text(
            &format!("{}", report.verdict),
            Point::new(15.0, y),
            &TextStyle { color: verdict_color, ..Default::default() },
        );
        y += 1.0;

        canvas.draw_text("\u{2502}    Duration: ", Point::new(0.0, y), &dim);
        canvas.draw_text(&format!("{}s", report.duration_secs), Point::new(16.0, y), &white);
        let workers_x = 16.0 + format!("{}s", report.duration_secs).len() as f32;
        canvas.draw_text(" | Workers: ", Point::new(workers_x, y), &dim);
        let count_x = workers_x + 12.0;
        canvas.draw_text(
            &format!("{} CPU + {} GPU", report.cpu_workers, report.gpu_workers),
            Point::new(count_x, y),
            &white,
        );
        y += 1.0;

        canvas.draw_text("\u{2502}", Point::new(0.0, y), &dim);
        y += 1.0;

        canvas.draw_text("\u{2502}    Peak CPU: ", Point::new(0.0, y), &dim);
        canvas.draw_text(
            &format!(
                "{:.2} M ops/sec ({:.1}%)",
                report.peak_cpu_ops as f64 / 1_000_000.0,
                report.peak_cpu_util,
            ),
            Point::new(16.0, y),
            &TextStyle { color: color_cyan(), ..Default::default() },
        );
        y += 1.0;

        canvas.draw_text("\u{2502}    Peak MEM: ", Point::new(0.0, y), &dim);
        canvas.draw_text(
            &format!(
                "{:.2} M pages/sec ({:.1}%)",
                report.peak_mem_ops as f64 / 1_000_000.0,
                report.peak_ram_util,
            ),
            Point::new(16.0, y),
            &TextStyle { color: color_magenta(), ..Default::default() },
        );
        y += 1.0;

        if report.peak_gpu_ops > 0 {
            canvas.draw_text("\u{2502}    Peak GPU: ", Point::new(0.0, y), &dim);
            canvas.draw_text(
                &format!(
                    "{:.2} G xfers/sec ({:.1}% VRAM)",
                    report.peak_gpu_ops as f64 / 1_000_000_000.0,
                    report.peak_vram_util,
                ),
                Point::new(16.0, y),
                &TextStyle { color: color_yellow(), ..Default::default() },
            );
            y += 1.0;
        }

        // Recommendations
        if !report.recommendations.is_empty() {
            canvas.draw_text("\u{2502}", Point::new(0.0, y), &dim);
            y += 1.0;
            canvas.draw_text("\u{2502}  Recommendations:", Point::new(0.0, y), &yellow_style);
            y += 1.0;

            for rec in &report.recommendations {
                canvas.draw_text("\u{2502}    \u{2022} ", Point::new(0.0, y), &dim);
                canvas.draw_text(rec, Point::new(8.0, y), &dim);
                y += 1.0;
            }
        }
    }

    canvas.draw_text("\u{2502}", Point::new(0.0, y), &dim);
    y += 1.0;
    canvas.draw_text("\u{2502}", Point::new(0.0, y), &dim);
    y += 1.0;

    canvas.draw_text(
        "\u{2502}  >>> Press 's' to START stress test <<<",
        Point::new(0.0, y),
        &green_bold,
    );
    y += 1.0;

    // Bottom border
    let bottom = format!("\u{2514}{}\u{2518}", "\u{2500}".repeat(w.saturating_sub(2)));
    canvas.draw_text(&bottom, Point::new(0.0, y), &dim);
}

fn render_stress_running(
    canvas: &mut DirectTerminalCanvas,
    app: &App,
    width: u16,
    _height: u16,
    _theme: &Theme,
) {
    let w = width as usize;
    let bar_width = w.saturating_sub(30).max(10);
    let dim = TextStyle { color: color_dark_gray(), ..Default::default() };
    let has_gpu = !app.gpu_workers.is_empty();
    let mut y: f32 = 3.0;

    // Status bar
    let elapsed = app.stress_start.map(|s| s.elapsed().as_secs()).unwrap_or(0);
    let status_text = format!(
        " STRESS TEST RUNNING | {}s | {} CPU + {} GPU workers | 's' to STOP ",
        elapsed,
        app.cpu_workers.len(),
        app.gpu_workers.len(),
    );
    canvas.fill_rect(Rect::new(0.0, y, width as f32, 1.0), color_green());
    canvas.draw_text(
        &status_text,
        Point::new(1.0, y),
        &TextStyle { color: color_black(), ..Default::default() },
    );
    y += 1.0;

    // CPU ops gauge
    let cpu_mops = app.cpu_ops_per_sec as f64 / 1_000_000.0;
    let cpu_title = format!(
        "CPU: {:.1} M ops/sec (peak: {:.1})",
        cpu_mops,
        app.peak_cpu_ops as f64 / 1_000_000.0,
    );
    draw_gauge_box(canvas, &cpu_title, cpu_mops, 100.0, bar_width, w, &mut y, color_cyan());

    // CPU ops sparkline
    draw_sparkline_box(canvas, "CPU History", &app.cpu_ops_history, w, &mut y, color_cyan());

    // Memory ops gauge
    let mem_mops = app.mem_ops_per_sec as f64 / 1_000_000.0;
    let mem_title = format!(
        "MEM: {:.1} M pages/sec (peak: {:.1})",
        mem_mops,
        app.peak_mem_ops as f64 / 1_000_000.0,
    );
    draw_gauge_box(canvas, &mem_title, mem_mops, 10.0, bar_width, w, &mut y, color_magenta());

    // Memory ops sparkline
    draw_sparkline_box(canvas, "Memory History", &app.mem_ops_history, w, &mut y, color_magenta());

    if has_gpu {
        // GPU ops gauge
        let gpu_gops = app.gpu_ops_per_sec as f64 / 1_000_000_000.0;
        let gpu_title = format!(
            "GPU: {:.2} G transfers/sec (peak: {:.2})",
            gpu_gops,
            app.peak_gpu_ops as f64 / 1_000_000_000.0,
        );
        draw_gauge_box(canvas, &gpu_title, gpu_gops, 10.0, bar_width, w, &mut y, color_yellow());

        // GPU ops sparkline
        draw_sparkline_box(canvas, "GPU History", &app.gpu_ops_history, w, &mut y, color_yellow());

        // Per-GPU VRAM history sparklines
        for (i, vram_history) in app.gpu_vram_history.iter().enumerate() {
            let gpu_name = app.gpus.get(i).map(|g| g.info.name.as_str()).unwrap_or("GPU");
            let title = format!("GPU{} VRAM % ({})", i, gpu_name);
            draw_sparkline_box_with_max(
                canvas,
                &title,
                vram_history,
                100,
                w,
                &mut y,
                color_magenta(),
            );
        }
    }

    // Live stats (System Impact)
    let cpu_util = app.cpu.compute_utilization().unwrap_or(0.0);
    let mem_pct_used = app.memory.ram_usage_percent();

    let top = format!(
        "\u{250c}\u{2500} System Impact {}\u{2510}",
        "\u{2500}".repeat(w.saturating_sub(18))
    );
    canvas.draw_text(&top, Point::new(0.0, y), &dim);
    y += 1.0;

    // System line
    canvas.draw_text("\u{2502} System: ", Point::new(0.0, y), &dim);
    let cpu_util_color = if cpu_util > 90.0 { color_red() } else { color_green() };
    canvas.draw_text(
        &format!("CPU {:.0}%", cpu_util),
        Point::new(10.0, y),
        &TextStyle { color: cpu_util_color, ..Default::default() },
    );
    let sep1_x = 10.0 + format!("CPU {:.0}%", cpu_util).len() as f32;
    canvas.draw_text(" | ", Point::new(sep1_x, y), &dim);
    let ram_color = if mem_pct_used > 80.0 { color_red() } else { color_green() };
    let ram_x = sep1_x + 3.0;
    canvas.draw_text(
        &format!("RAM {:.0}%", mem_pct_used),
        Point::new(ram_x, y),
        &TextStyle { color: ram_color, ..Default::default() },
    );
    let sep2_x = ram_x + format!("RAM {:.0}%", mem_pct_used).len() as f32;
    canvas.draw_text(" | Pressure: ", Point::new(sep2_x, y), &dim);
    let pressure_color = match app.memory.pressure_level {
        PressureLevel::Ok => color_green(),
        PressureLevel::Elevated => color_yellow(),
        PressureLevel::Warning => color_orange(),
        PressureLevel::Critical => color_red(),
    };
    let pressure_x = sep2_x + 13.0;
    canvas.draw_text(
        &format!("{}", app.memory.pressure_level),
        Point::new(pressure_x, y),
        &TextStyle { color: pressure_color, ..Default::default() },
    );
    y += 1.0;

    // GPU VRAM stats
    for (i, gpu) in app.gpus.iter().enumerate() {
        canvas.draw_text(&format!("\u{2502} GPU{} VRAM: ", i), Point::new(0.0, y), &dim);
        let vram_x = 2.0 + format!("GPU{} VRAM: ", i).len() as f32;
        let vram_color = if gpu.vram_percent > 90.0 { color_red() } else { color_green() };
        canvas.draw_text(
            &format!("{:.1}%", gpu.vram_percent),
            Point::new(vram_x, y),
            &TextStyle { color: vram_color, ..Default::default() },
        );
        let detail_x = vram_x + format!("{:.1}%", gpu.vram_percent).len() as f32;
        canvas.draw_text(
            &format!(" ({:.1}/{:.1} GB)", gpu.vram_used_gb, gpu.vram_total_gb),
            Point::new(detail_x, y),
            &dim,
        );
        y += 1.0;
    }

    let bottom = format!("\u{2514}{}\u{2518}", "\u{2500}".repeat(w.saturating_sub(2)));
    canvas.draw_text(&bottom, Point::new(0.0, y), &dim);
}

// ---------------------------------------------------------------------------
// Drawing helpers for stress tab
// ---------------------------------------------------------------------------

fn draw_gauge_box(
    canvas: &mut DirectTerminalCanvas,
    title: &str,
    value: f64,
    scale_max: f64,
    bar_width: usize,
    width: usize,
    y: &mut f32,
    color: Color,
) {
    let dim = TextStyle { color: color_dark_gray(), ..Default::default() };
    let bar_style = TextStyle { color, ..Default::default() };

    let top = format!(
        "\u{250c}\u{2500} {} {}\u{2510}",
        title,
        "\u{2500}".repeat(width.saturating_sub(title.len() + 5))
    );
    canvas.draw_text(&top, Point::new(0.0, *y), &dim);
    *y += 1.0;

    let pct = ((value / scale_max) * 100.0).min(100.0);
    let bar = make_bar(pct, 100.0, bar_width);
    canvas.draw_text("\u{2502} ", Point::new(0.0, *y), &dim);
    canvas.draw_text(&bar, Point::new(2.0, *y), &bar_style);
    *y += 1.0;

    let bottom = format!("\u{2514}{}\u{2518}", "\u{2500}".repeat(width.saturating_sub(2)));
    canvas.draw_text(&bottom, Point::new(0.0, *y), &dim);
    *y += 1.0;
}

fn draw_sparkline_box(
    canvas: &mut DirectTerminalCanvas,
    title: &str,
    data: &[u64],
    width: usize,
    y: &mut f32,
    color: Color,
) {
    let max = data.iter().copied().max().unwrap_or(1).max(1);
    draw_sparkline_box_with_max(canvas, title, data, max, width, y, color);
}

fn draw_sparkline_box_with_max(
    canvas: &mut DirectTerminalCanvas,
    title: &str,
    data: &[u64],
    max: u64,
    width: usize,
    y: &mut f32,
    color: Color,
) {
    let dim = TextStyle { color: color_dark_gray(), ..Default::default() };
    let spark_style = TextStyle { color, ..Default::default() };

    let top = format!(
        "\u{250c}\u{2500} {} {}\u{2510}",
        title,
        "\u{2500}".repeat(width.saturating_sub(title.len() + 5))
    );
    canvas.draw_text(&top, Point::new(0.0, *y), &dim);
    *y += 1.0;

    let spark = make_sparkline(data, max);
    canvas.draw_text("\u{2502} ", Point::new(0.0, *y), &dim);
    canvas.draw_text(&spark, Point::new(2.0, *y), &spark_style);
    *y += 1.0;

    let bottom = format!("\u{2514}{}\u{2518}", "\u{2500}".repeat(width.saturating_sub(2)));
    canvas.draw_text(&bottom, Point::new(0.0, *y), &dim);
    *y += 1.0;
}
