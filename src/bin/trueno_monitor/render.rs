//! Main UI rendering: tabs, compute tab, memory tab.

use presentar_core::{Canvas, Color, Point, Rect, TextStyle};
use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};
use presentar_terminal::Theme;
use trueno_gpu::monitor::{ComputeDevice, PressureLevel};

use super::render_stress;
use super::{App, GpuState};

// ---------------------------------------------------------------------------
// Color helpers
// ---------------------------------------------------------------------------

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

pub(crate) fn color_white() -> Color {
    Color::new(1.0, 1.0, 1.0, 1.0)
}

pub(crate) fn color_dark_gray() -> Color {
    Color::new(0.5, 0.5, 0.5, 1.0)
}

fn color_blue() -> Color {
    Color::new(0.3, 0.5, 1.0, 1.0)
}

fn color_orange() -> Color {
    Color::new(1.0, 0.65, 0.0, 1.0)
}

fn color_black() -> Color {
    Color::new(0.0, 0.0, 0.0, 1.0)
}

/// Map a percentage to a traffic-light color (green/yellow/red).
pub(crate) fn pct_color(pct: f64) -> Color {
    if pct > 90.0 {
        color_red()
    } else if pct > 70.0 {
        color_yellow()
    } else {
        color_green()
    }
}

/// Map a `PressureLevel` to a display color.
fn pressure_color(level: &PressureLevel) -> Color {
    match level {
        PressureLevel::Ok => color_green(),
        PressureLevel::Elevated => color_yellow(),
        PressureLevel::Warning => color_orange(),
        PressureLevel::Critical => color_red(),
    }
}

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------

pub(crate) fn make_bar(value: f64, max: f64, width: usize) -> String {
    let ratio = (value / max).clamp(0.0, 1.0);
    let filled = (ratio * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    format!("[{}{}] {:.1}%", "\u{2588}".repeat(filled), "\u{2591}".repeat(empty), value)
}

pub(crate) fn make_sparkline(data: &[u64], max: u64) -> String {
    const BLOCKS: [char; 8] = [
        '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}',
        '\u{2588}',
    ];
    let max = max.max(1);
    data.iter()
        .map(|&v| {
            let idx = ((v as f64 / max as f64) * 7.0).round() as usize;
            BLOCKS[idx.min(7)]
        })
        .collect()
}

/// Draw a bordered box top: `\u{250c}\u{2500} Title \u{2500}\u{2500}\u{2500}\u{2510}`
fn draw_box_top(
    canvas: &mut DirectTerminalCanvas,
    x: f32,
    y: f32,
    width: usize,
    title: &str,
    color: Color,
) {
    let style = TextStyle { color, ..Default::default() };
    let inner = width.saturating_sub(title.len() + 4);
    let line = format!("\u{250c}\u{2500} {} {}\u{2510}", title, "\u{2500}".repeat(inner));
    canvas.draw_text(&line, Point::new(x, y), &style);
}

/// Draw a bordered box bottom: `\u{2514}\u{2500}\u{2500}\u{2500}\u{2518}`
fn draw_box_bottom(canvas: &mut DirectTerminalCanvas, x: f32, y: f32, width: usize, color: Color) {
    let style = TextStyle { color, ..Default::default() };
    let line = format!("\u{2514}{}\u{2518}", "\u{2500}".repeat(width.saturating_sub(2)));
    canvas.draw_text(&line, Point::new(x, y), &style);
}

// ---------------------------------------------------------------------------
// Main UI entry point
// ---------------------------------------------------------------------------

pub(crate) fn ui(
    buffer: &mut CellBuffer,
    renderer: &mut DiffRenderer,
    theme: &Theme,
    app: &App,
) -> Result<(), Box<dyn std::error::Error>> {
    let (width, height) = (buffer.width(), buffer.height());

    {
        let mut canvas = DirectTerminalCanvas::new(buffer);

        // Clear background
        canvas.fill_rect(Rect::new(0.0, 0.0, width as f32, height as f32), theme.background);

        // Header with tabs (rows 0-2)
        render_header(&mut canvas, app, width, theme);

        // Content based on selected tab (row 3 to height-3)
        match app.selected_tab {
            0 => render_compute_tab(&mut canvas, app, width, height, theme),
            1 => render_memory_tab(&mut canvas, app, width, height, theme),
            2 => render_stress::render_dataflow_tab(&mut canvas, app, width, height, theme),
            3 => render_stress::render_stress_tab(&mut canvas, app, width, height, theme),
            4.. => {}
        }

        // Footer (last 3 rows)
        render_footer(&mut canvas, app, width, height, theme);

        // Help overlay
        if app.show_help {
            render_stress::render_help_overlay(&mut canvas, width, height, theme);
        }
    }

    // Flush to terminal
    let mut output = Vec::with_capacity(16384);
    renderer.flush(buffer, &mut output).ok();
    std::io::Write::write_all(&mut std::io::stdout(), &output)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

fn render_header(canvas: &mut DirectTerminalCanvas, app: &App, width: u16, theme: &Theme) {
    let dim_style = TextStyle { color: theme.dim, ..Default::default() };
    let bright_style = TextStyle { color: theme.foreground, ..Default::default() };
    let highlight_style = TextStyle { color: color_cyan(), ..Default::default() };

    // Top border
    let gpu_count = app.gpus.len();
    let title = if gpu_count > 0 {
        format!(" TRUENO Monitor v0.10.1 | {} GPU(s) ", gpu_count)
    } else {
        " TRUENO Monitor v0.10.1 | No CUDA GPU ".to_string()
    };
    let border_fill = (width as usize).saturating_sub(title.len() + 4);
    let top_line = format!("\u{250c}\u{2500} {}{}\u{2510}", title, "\u{2500}".repeat(border_fill));
    canvas.draw_text(&top_line, Point::new(0.0, 0.0), &dim_style);

    // Tab bar
    let titles = ["Compute", "Memory", "Data Flow", "Stress Test"];
    let mut x: f32 = 1.0;
    canvas.draw_text("\u{2502}", Point::new(0.0, 1.0), &dim_style);

    for (i, tab_title) in titles.iter().enumerate() {
        if i == app.selected_tab {
            canvas.draw_text("[", Point::new(x, 1.0), &highlight_style);
            x += 1.0;
            canvas.draw_text(tab_title, Point::new(x, 1.0), &bright_style);
            x += tab_title.len() as f32;
            canvas.draw_text("]", Point::new(x, 1.0), &highlight_style);
            x += 1.0;
        } else {
            canvas.draw_text(&format!(" {} ", tab_title), Point::new(x, 1.0), &dim_style);
            x += tab_title.len() as f32 + 2.0;
        }
    }

    // Fill rest and close
    let remaining = (width as f32 - x - 1.0).max(0.0) as usize;
    if remaining > 0 {
        canvas.draw_text(&" ".repeat(remaining), Point::new(x, 1.0), &dim_style);
    }
    canvas.draw_text("\u{2502}", Point::new(width as f32 - 1.0, 1.0), &dim_style);

    // Bottom border of header
    let bottom =
        format!("\u{2514}{}\u{2518}", "\u{2500}".repeat((width as usize).saturating_sub(2)));
    canvas.draw_text(&bottom, Point::new(0.0, 2.0), &dim_style);
}

// ---------------------------------------------------------------------------
// Footer
// ---------------------------------------------------------------------------

fn render_footer(
    canvas: &mut DirectTerminalCanvas,
    app: &App,
    width: u16,
    height: u16,
    _theme: &Theme,
) {
    let dim_style = TextStyle { color: color_dark_gray(), ..Default::default() };

    let footer_y = height as f32 - 3.0;

    // Top border
    let top = format!("\u{250c}{}\u{2510}", "\u{2500}".repeat((width as usize).saturating_sub(2)));
    canvas.draw_text(&top, Point::new(0.0, footer_y), &dim_style);

    // Content
    let help_text = if app.stress_running {
        " q:Quit  Tab:Switch  s:Stop Stress  ?:Help  |  STRESS TEST RUNNING "
    } else {
        " q:Quit  Tab:Switch  s:Start Stress  ?:Help  |  Refresh: 100ms "
    };
    let padding = (width as usize).saturating_sub(help_text.len() + 4);
    let content = format!("\u{2502} {}{} \u{2502}", help_text, " ".repeat(padding));
    canvas.draw_text(&content, Point::new(0.0, footer_y + 1.0), &dim_style);

    // Bottom border
    let bottom =
        format!("\u{2514}{}\u{2518}", "\u{2500}".repeat((width as usize).saturating_sub(2)));
    canvas.draw_text(&bottom, Point::new(0.0, footer_y + 2.0), &dim_style);
}

// ---------------------------------------------------------------------------
// Compute Tab
// ---------------------------------------------------------------------------

fn render_compute_tab(
    canvas: &mut DirectTerminalCanvas,
    app: &App,
    width: u16,
    _height: u16,
    theme: &Theme,
) {
    let w = width as usize;
    let bar_width = w.saturating_sub(30).max(10);
    let mut y: f32 = 3.0;

    // Stress banner
    if app.stress_running {
        let elapsed = app.stress_start.map(|s| s.elapsed().as_secs()).unwrap_or(0);
        let banner = format!(
            " STRESS TEST ACTIVE | {}s | CPU: {:.1}M ops/s | MEM: {:.1}M pg/s | GPU: {:.2}G xf/s ",
            elapsed,
            app.cpu_ops_per_sec as f64 / 1_000_000.0,
            app.mem_ops_per_sec as f64 / 1_000_000.0,
            app.gpu_ops_per_sec as f64 / 1_000_000_000.0,
        );
        let banner_style = TextStyle { color: color_black(), ..Default::default() };
        // Fill banner background
        canvas.fill_rect(Rect::new(0.0, y, width as f32, 1.0), color_yellow());
        canvas.draw_text(&banner, Point::new(1.0, y), &banner_style);
        y += 1.0;
    }

    // CPU utilization gauge
    let cpu_pct = app.cpu.compute_utilization().unwrap_or(0.0);
    let cpu_title = if app.stress_running {
        format!("CPU Utilization [STRESS: {} workers]", app.cpu_workers.len())
    } else {
        "CPU Utilization".to_string()
    };
    draw_box_top(canvas, 0.0, y, w, &cpu_title, color_dark_gray());
    y += 1.0;
    let bar = make_bar(cpu_pct, 100.0, bar_width);
    let bar_content_style = TextStyle { color: pct_color(cpu_pct), ..Default::default() };
    canvas.draw_text(
        "\u{2502} ",
        Point::new(0.0, y),
        &TextStyle { color: color_dark_gray(), ..Default::default() },
    );
    canvas.draw_text(&bar, Point::new(2.0, y), &bar_content_style);
    canvas.draw_text(
        " \u{2502}",
        Point::new(w as f32 - 2.0, y),
        &TextStyle { color: color_dark_gray(), ..Default::default() },
    );
    y += 1.0;
    draw_box_bottom(canvas, 0.0, y, w, color_dark_gray());
    y += 1.0;

    // CPU sparkline
    draw_box_top(canvas, 0.0, y, w, "CPU History (60s)", color_dark_gray());
    y += 1.0;
    let spark = make_sparkline(&app.cpu_history, 100);
    let spark_style = TextStyle { color: color_cyan(), ..Default::default() };
    canvas.draw_text(
        "\u{2502} ",
        Point::new(0.0, y),
        &TextStyle { color: color_dark_gray(), ..Default::default() },
    );
    canvas.draw_text(&spark, Point::new(2.0, y), &spark_style);
    y += 1.0;
    draw_box_bottom(canvas, 0.0, y, w, color_dark_gray());
    y += 1.0;

    // GPU gauges
    let stress_active = app.stress_running && !app.gpu_workers.is_empty();
    for (i, gpu) in app.gpus.iter().enumerate() {
        y = render_gpu_gauge(canvas, gpu, i, stress_active, w, bar_width, y);
    }

    // Device info
    render_device_info(canvas, app, w, y, theme);
}

/// Render a single GPU VRAM gauge. Returns the next y position.
fn render_gpu_gauge(
    canvas: &mut DirectTerminalCanvas,
    gpu: &GpuState,
    gpu_idx: usize,
    stress_active: bool,
    width: usize,
    bar_width: usize,
    mut y: f32,
) -> f32 {
    let gpu_color = pct_color(gpu.vram_percent);
    let title = if stress_active {
        format!("GPU {} [STRESS] {}", gpu_idx, gpu.info.name)
    } else {
        format!("GPU {} VRAM: {}", gpu_idx, gpu.info.name)
    };
    let _label =
        format!("{:.1} / {:.1} GB ({:.1}%)", gpu.vram_used_gb, gpu.vram_total_gb, gpu.vram_percent);

    draw_box_top(canvas, 0.0, y, width, &title, color_dark_gray());
    y += 1.0;

    let bar = make_bar(gpu.vram_percent, 100.0, bar_width);
    let bar_style = TextStyle { color: gpu_color, ..Default::default() };
    canvas.draw_text(
        "\u{2502} ",
        Point::new(0.0, y),
        &TextStyle { color: color_dark_gray(), ..Default::default() },
    );
    canvas.draw_text(&bar, Point::new(2.0, y), &bar_style);
    y += 1.0;
    draw_box_bottom(canvas, 0.0, y, width, color_dark_gray());
    y += 1.0;

    y
}

/// Render device info panel
fn render_device_info(
    canvas: &mut DirectTerminalCanvas,
    app: &App,
    width: usize,
    mut y: f32,
    _theme: &Theme,
) {
    let dim = TextStyle { color: color_dark_gray(), ..Default::default() };
    let white = TextStyle { color: color_white(), ..Default::default() };
    let cyan = TextStyle { color: color_cyan(), ..Default::default() };

    draw_box_top(canvas, 0.0, y, width, "Device Info", color_dark_gray());
    y += 1.0;

    // CPU name
    // CPU name line
    canvas.draw_text("\u{2502} ", Point::new(0.0, y), &dim);
    canvas.draw_text("CPU: ", Point::new(2.0, y), &dim);
    canvas.draw_text(app.cpu.device_name(), Point::new(7.0, y), &white);
    y += 1.0;

    // CPU details
    let clock = app.cpu.compute_clock_mhz().unwrap_or(0);
    let temp = app.cpu.compute_temperature_c().unwrap_or(0.0);
    let cores = app.cpu.compute_unit_count();
    let temp_color = if temp > 80.0 { color_red() } else { color_green() };
    let temp_style = TextStyle { color: temp_color, ..Default::default() };

    canvas.draw_text("\u{2502} ", Point::new(0.0, y), &dim);
    canvas.draw_text("Cores: ", Point::new(2.0, y), &dim);
    canvas.draw_text(&format!("{}", cores), Point::new(9.0, y), &cyan);
    canvas.draw_text("  Clock: ", Point::new(9.0 + format!("{}", cores).len() as f32, y), &dim);
    let clock_x = 9.0 + format!("{}", cores).len() as f32 + 9.0;
    canvas.draw_text(&format!("{} MHz", clock), Point::new(clock_x, y), &cyan);
    let temp_x = clock_x + format!("{} MHz", clock).len() as f32 + 2.0;
    canvas.draw_text("Temp: ", Point::new(temp_x, y), &dim);
    canvas.draw_text(&format!("{:.0}C", temp), Point::new(temp_x + 6.0, y), &temp_style);
    y += 1.0;

    // GPU info
    for (i, gpu) in app.gpus.iter().enumerate() {
        let magenta = TextStyle { color: color_magenta(), ..Default::default() };
        canvas.draw_text("\u{2502} ", Point::new(0.0, y), &dim);
        canvas.draw_text(&format!("GPU{}: ", i), Point::new(2.0, y), &dim);
        let name_x = 2.0 + format!("GPU{}: ", i).len() as f32;
        canvas.draw_text(&gpu.info.name, Point::new(name_x, y), &magenta);
        let mem_x = name_x + gpu.info.name.len() as f32;
        canvas.draw_text(
            &format!(" ({:.1} GB)", gpu.info.total_memory_gb()),
            Point::new(mem_x, y),
            &dim,
        );
        y += 1.0;
    }

    draw_box_bottom(canvas, 0.0, y, width, color_dark_gray());
}

// ---------------------------------------------------------------------------
// Memory Tab
// ---------------------------------------------------------------------------

fn render_memory_tab(
    canvas: &mut DirectTerminalCanvas,
    app: &App,
    width: u16,
    _height: u16,
    _theme: &Theme,
) {
    let w = width as usize;
    let bar_width = w.saturating_sub(30).max(10);
    let mut y: f32 = 3.0;

    // Stress banner
    if app.stress_running {
        let banner = format!(
            " STRESS: Allocating 512MB + {} GPU buffers | RAM pressure: {} ",
            app.gpu_workers.len() * 8,
            app.memory.pressure_level,
        );
        let banner_style = TextStyle { color: color_black(), ..Default::default() };
        canvas.fill_rect(Rect::new(0.0, y, width as f32, 1.0), color_yellow());
        canvas.draw_text(&banner, Point::new(1.0, y), &banner_style);
        y += 1.0;
    }

    // RAM gauge
    let ram_pct = app.memory.ram_usage_percent();
    let ram_color = pressure_color(&app.memory.pressure_level);
    let ram_used_gb = app.memory.ram_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let ram_total_gb = app.memory.ram_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let ram_title = if app.stress_running { "RAM [STRESS ACTIVE]" } else { "RAM" };

    draw_box_top(canvas, 0.0, y, w, ram_title, color_dark_gray());
    y += 1.0;
    let bar_text = format!(
        "[{}{}] {:.1} / {:.1} GB ({:.1}%)",
        "\u{2588}".repeat(((ram_pct / 100.0) * bar_width as f64).round() as usize),
        "\u{2591}".repeat(
            bar_width.saturating_sub(((ram_pct / 100.0) * bar_width as f64).round() as usize)
        ),
        ram_used_gb,
        ram_total_gb,
        ram_pct,
    );
    let bar_style = TextStyle { color: ram_color, ..Default::default() };
    let dim = TextStyle { color: color_dark_gray(), ..Default::default() };
    canvas.draw_text("\u{2502} ", Point::new(0.0, y), &dim);
    canvas.draw_text(&bar_text, Point::new(2.0, y), &bar_style);
    y += 1.0;
    draw_box_bottom(canvas, 0.0, y, w, color_dark_gray());
    y += 1.0;

    // SWAP gauge
    let swap_pct = app.memory.swap_usage_percent();
    let swap_used_gb = app.memory.swap_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let swap_total_gb = app.memory.swap_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let swap_color = if swap_pct > 50.0 { color_yellow() } else { color_blue() };

    draw_box_top(canvas, 0.0, y, w, "SWAP", color_dark_gray());
    y += 1.0;
    let swap_bar = format!(
        "[{}{}] {:.1} / {:.1} GB ({:.1}%)",
        "\u{2588}".repeat(((swap_pct / 100.0) * bar_width as f64).round() as usize),
        "\u{2591}".repeat(
            bar_width.saturating_sub(((swap_pct / 100.0) * bar_width as f64).round() as usize)
        ),
        swap_used_gb,
        swap_total_gb,
        swap_pct,
    );
    let swap_style = TextStyle { color: swap_color, ..Default::default() };
    canvas.draw_text("\u{2502} ", Point::new(0.0, y), &dim);
    canvas.draw_text(&swap_bar, Point::new(2.0, y), &swap_style);
    y += 1.0;
    draw_box_bottom(canvas, 0.0, y, w, color_dark_gray());
    y += 1.0;

    // GPU VRAM gauges
    let stress_active = app.stress_running && !app.gpu_workers.is_empty();
    for (i, gpu) in app.gpus.iter().enumerate() {
        let vram_color = pct_color(gpu.vram_percent);
        let title = if stress_active {
            format!("VRAM {} [STRESS]", i)
        } else {
            format!("VRAM {} [{}]", i, gpu.info.name)
        };

        draw_box_top(canvas, 0.0, y, w, &title, color_dark_gray());
        y += 1.0;
        let vram_bar =
            format!(
                "[{}{}] {:.1} / {:.1} GB ({:.1}%)",
                "\u{2588}".repeat(((gpu.vram_percent / 100.0) * bar_width as f64).round() as usize),
                "\u{2591}".repeat(bar_width.saturating_sub(
                    ((gpu.vram_percent / 100.0) * bar_width as f64).round() as usize
                )),
                gpu.vram_used_gb,
                gpu.vram_total_gb,
                gpu.vram_percent,
            );
        let vram_style = TextStyle { color: vram_color, ..Default::default() };
        canvas.draw_text("\u{2502} ", Point::new(0.0, y), &dim);
        canvas.draw_text(&vram_bar, Point::new(2.0, y), &vram_style);
        y += 1.0;
        draw_box_bottom(canvas, 0.0, y, w, color_dark_gray());
        y += 1.0;
    }

    // Memory sparkline
    draw_box_top(canvas, 0.0, y, w, "Memory History (60s)", color_dark_gray());
    y += 1.0;
    let spark = make_sparkline(&app.mem_history, 100);
    let spark_style = TextStyle { color: color_magenta(), ..Default::default() };
    canvas.draw_text("\u{2502} ", Point::new(0.0, y), &dim);
    canvas.draw_text(&spark, Point::new(2.0, y), &spark_style);
    y += 1.0;
    draw_box_bottom(canvas, 0.0, y, w, color_dark_gray());
    y += 1.0;

    // Pressure info
    render_pressure_info(canvas, app, w, y);
}

fn render_pressure_info(canvas: &mut DirectTerminalCanvas, app: &App, width: usize, mut y: f32) {
    let dim = TextStyle { color: color_dark_gray(), ..Default::default() };
    let cyan = TextStyle { color: color_cyan(), ..Default::default() };

    let (label, color, desc) = match app.memory.pressure_level {
        PressureLevel::Ok => ("OK", color_green(), ">= 50% available"),
        PressureLevel::Elevated => ("ELEVATED", color_yellow(), "30-50% available"),
        PressureLevel::Warning => ("WARNING", color_orange(), "15-30% available"),
        PressureLevel::Critical => ("CRITICAL", color_red(), "< 15% available"),
    };

    draw_box_top(canvas, 0.0, y, width, "Memory Pressure (LAMBDA-0002)", color_dark_gray());
    y += 1.0;

    canvas.draw_text("\u{2502} ", Point::new(0.0, y), &dim);
    canvas.draw_text("Pressure Level: ", Point::new(2.0, y), &dim);
    let label_style = TextStyle { color, ..Default::default() };
    canvas.draw_text(label, Point::new(18.0, y), &label_style);
    let desc_x = 18.0 + label.len() as f32;
    canvas.draw_text(&format!(" ({})", desc), Point::new(desc_x, y), &dim);
    y += 1.0;

    canvas.draw_text("\u{2502} ", Point::new(0.0, y), &dim);
    canvas.draw_text("Safe Parallel Jobs: ", Point::new(2.0, y), &dim);
    canvas.draw_text(&format!("{}", app.memory.safe_parallel_jobs), Point::new(22.0, y), &cyan);
    y += 1.0;

    draw_box_bottom(canvas, 0.0, y, width, color_dark_gray());
}
