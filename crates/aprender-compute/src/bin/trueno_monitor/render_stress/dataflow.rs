//! Data flow tab rendering (PCIe bandwidth monitoring).

use presentar_core::{Canvas, Color, Point, TextStyle};
use presentar_terminal::direct::DirectTerminalCanvas;
use presentar_terminal::Theme;

use crate::App;

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

fn color_white() -> Color {
    Color::new(1.0, 1.0, 1.0, 1.0)
}

fn color_dark_gray() -> Color {
    Color::new(0.5, 0.5, 0.5, 1.0)
}

pub(crate) fn render_dataflow_tab(
    canvas: &mut DirectTerminalCanvas,
    app: &App,
    width: u16,
    _height: u16,
    theme: &Theme,
) {
    let w = width as usize;
    let dim = TextStyle { color: color_dark_gray(), ..Default::default() };
    let mut y: f32 = 3.0;

    let title = if app.stress_running && !app.gpu_workers.is_empty() {
        "Data Flow Monitor [STRESS: H2D/D2H ACTIVE]"
    } else {
        "Data Flow Monitor"
    };

    let top = format!(
        "\u{250c}\u{2500} {} {}\u{2510}",
        title,
        "\u{2500}".repeat(w.saturating_sub(title.len() + 5))
    );
    canvas.draw_text(&top, Point::new(0.0, y), &dim);
    y += 1.0;

    if app.gpus.is_empty() {
        // No GPU - show message
        let lines = [
            "",
            "  No CUDA GPU Detected",
            "",
            "  To enable GPU monitoring:",
            "",
            "    1. Install NVIDIA CUDA drivers",
            "    2. Build with: cargo build --features tui-monitor,cuda",
            "    3. Run: cargo run --bin trueno-monitor --features tui-monitor,cuda",
            "",
            "  PCIe bandwidth and transfer tracking require CUDA hardware.",
        ];

        let yellow_bold = TextStyle { color: color_yellow(), ..Default::default() };

        for (i, line) in lines.iter().enumerate() {
            let padding = w.saturating_sub(line.len() + 4);
            let row = format!("\u{2502} {}{} \u{2502}", line, " ".repeat(padding));
            let style = if i == 1 {
                &yellow_bold
            } else if i == 3 || i == 9 {
                &dim
            } else {
                &TextStyle { color: theme.foreground, ..Default::default() }
            };
            canvas.draw_text(&row, Point::new(0.0, y), style);
            y += 1.0;
        }
    } else {
        // Real GPU data
        let cyan_bold = TextStyle { color: color_cyan(), ..Default::default() };
        let fg = TextStyle { color: theme.foreground, ..Default::default() };

        canvas.draw_text("\u{2502}", Point::new(0.0, y), &dim);
        y += 1.0;

        canvas.draw_text("\u{2502}  PCIe Data Flow", Point::new(0.0, y), &cyan_bold);
        y += 1.0;

        canvas.draw_text("\u{2502}", Point::new(0.0, y), &dim);
        y += 1.0;

        for (i, gpu) in app.gpus.iter().enumerate() {
            // GPU name
            canvas.draw_text(&format!("\u{2502}  GPU {}: ", i), Point::new(0.0, y), &dim);
            let name_x = 2.0 + format!("GPU {}: ", i).len() as f32;
            canvas.draw_text(
                &gpu.info.name,
                Point::new(name_x, y),
                &TextStyle { color: color_magenta(), ..Default::default() },
            );
            y += 1.0;

            // VRAM
            canvas.draw_text("\u{2502}    VRAM: ", Point::new(0.0, y), &dim);
            canvas.draw_text(
                &format!("{:.1} GB", gpu.vram_used_gb),
                Point::new(12.0, y),
                &TextStyle { color: color_cyan(), ..Default::default() },
            );
            let slash_x = 12.0 + format!("{:.1} GB", gpu.vram_used_gb).len() as f32;
            canvas.draw_text(" / ", Point::new(slash_x, y), &dim);
            let total_x = slash_x + 3.0;
            canvas.draw_text(
                &format!("{:.1} GB", gpu.vram_total_gb),
                Point::new(total_x, y),
                &TextStyle { color: color_white(), ..Default::default() },
            );
            let paren_x = total_x + format!("{:.1} GB", gpu.vram_total_gb).len() as f32;
            canvas.draw_text(" (", Point::new(paren_x, y), &dim);
            let pct_x = paren_x + 2.0;
            let vram_pct_color = if gpu.vram_percent > 80.0 { color_red() } else { color_green() };
            canvas.draw_text(
                &format!("{:.1}%", gpu.vram_percent),
                Point::new(pct_x, y),
                &TextStyle { color: vram_pct_color, ..Default::default() },
            );
            let close_x = pct_x + format!("{:.1}%", gpu.vram_percent).len() as f32;
            canvas.draw_text(")", Point::new(close_x, y), &dim);
            y += 1.0;

            // PCIe info
            if app.stress_running && !app.gpu_workers.is_empty() {
                let bandwidth_gbps =
                    (app.gpu_ops_per_sec as f64 * 4.0) / (1024.0 * 1024.0 * 1024.0);
                canvas.draw_text("\u{2502}    PCIe: ", Point::new(0.0, y), &dim);
                canvas.draw_text(
                    "STRESS ACTIVE",
                    Point::new(12.0, y),
                    &TextStyle { color: color_yellow(), ..Default::default() },
                );
                canvas.draw_text(" - ", Point::new(25.0, y), &dim);
                canvas.draw_text(
                    &format!("{:.2} GB/s", bandwidth_gbps),
                    Point::new(28.0, y),
                    &TextStyle { color: color_cyan(), ..Default::default() },
                );
                let actual_x = 28.0 + format!("{:.2} GB/s", bandwidth_gbps).len() as f32;
                canvas.draw_text(" actual", Point::new(actual_x, y), &dim);
            } else {
                canvas.draw_text("\u{2502}    PCIe: ", Point::new(0.0, y), &dim);
                canvas.draw_text(
                    "Gen4 x16",
                    Point::new(12.0, y),
                    &TextStyle { color: color_green(), ..Default::default() },
                );
                canvas.draw_text(" (", Point::new(20.0, y), &dim);
                canvas.draw_text(
                    "31.5 GB/s",
                    Point::new(22.0, y),
                    &TextStyle { color: color_cyan(), ..Default::default() },
                );
                canvas.draw_text(" theoretical)", Point::new(31.0, y), &dim);
            }
            y += 1.0;

            canvas.draw_text("\u{2502}", Point::new(0.0, y), &dim);
            y += 1.0;
        }

        canvas.draw_text("\u{2502}  Active Transfers", Point::new(0.0, y), &cyan_bold);
        y += 1.0;

        canvas.draw_text("\u{2502}", Point::new(0.0, y), &dim);
        y += 1.0;

        if app.stress_running && !app.gpu_workers.is_empty() {
            canvas.draw_text("\u{2502}    ", Point::new(0.0, y), &dim);
            canvas.draw_text(
                "H2D: ",
                Point::new(6.0, y),
                &TextStyle { color: color_green(), ..Default::default() },
            );
            canvas.draw_text("1M x f32 (4MB) per iteration", Point::new(11.0, y), &fg);
            y += 1.0;

            canvas.draw_text("\u{2502}    ", Point::new(0.0, y), &dim);
            canvas.draw_text(
                "D2H: ",
                Point::new(6.0, y),
                &TextStyle { color: color_yellow(), ..Default::default() },
            );
            canvas.draw_text("1M x f32 (4MB) per iteration", Point::new(11.0, y), &fg);
            y += 1.0;

            canvas.draw_text("\u{2502}    ", Point::new(0.0, y), &dim);
            canvas.draw_text(
                &format!(
                    "Throughput: {:.2} G elements/sec",
                    app.gpu_ops_per_sec as f64 / 1_000_000_000.0,
                ),
                Point::new(6.0, y),
                &cyan_bold,
            );
            y += 1.0;
        } else {
            canvas.draw_text("\u{2502}    No active kernel transfers", Point::new(0.0, y), &dim);
            y += 1.0;
        }
    }

    // Bottom border
    let bottom = format!("\u{2514}{}\u{2518}", "\u{2500}".repeat(w.saturating_sub(2)));
    canvas.draw_text(&bottom, Point::new(0.0, y), &dim);
}
