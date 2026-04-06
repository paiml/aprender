//! Stress test, data flow, and help overlay rendering.

mod dataflow;
mod stress;

pub(crate) use dataflow::render_dataflow_tab;
pub(crate) use stress::render_stress_tab;

use presentar_core::{Canvas, Color, Point, Rect, TextStyle};
use presentar_terminal::direct::DirectTerminalCanvas;
use presentar_terminal::Theme;

use super::render::{color_dark_gray, color_white};

fn color_dark_bg() -> Color {
    Color::new(0.2, 0.2, 0.2, 1.0)
}

pub(crate) fn render_help_overlay(
    canvas: &mut DirectTerminalCanvas,
    width: u16,
    height: u16,
    theme: &Theme,
) {
    // Calculate centered rectangle (50% x 60%)
    let overlay_w = (width as f32 * 0.50).round() as usize;
    let overlay_h = (height as f32 * 0.60).round() as usize;
    let start_x = ((width as usize).saturating_sub(overlay_w)) / 2;
    let start_y = ((height as usize).saturating_sub(overlay_h)) / 2;

    // Fill background
    canvas.fill_rect(
        Rect::new(start_x as f32, start_y as f32, overlay_w as f32, overlay_h as f32),
        color_dark_bg(),
    );

    let dim = TextStyle { color: color_dark_gray(), ..Default::default() };
    let bright = TextStyle { color: color_white(), ..Default::default() };
    let x = start_x as f32;
    let mut y = start_y as f32;

    // Top border
    let top =
        format!("\u{250c}\u{2500} Help {}\u{2510}", "\u{2500}".repeat(overlay_w.saturating_sub(9)));
    canvas.draw_text(&top, Point::new(x, y), &dim);
    y += 1.0;

    let lines = [
        "",
        "Keyboard Controls",
        "",
        "  q        Quit",
        "  Tab      Next tab",
        "  Shift+Tab Previous tab",
        "  s        Toggle stress test",
        "  ?        Toggle this help",
        "",
        "Tabs",
        "",
        "  Compute    CPU/GPU utilization",
        "  Memory     RAM/SWAP/VRAM usage",
        "  Data Flow  PCIe bandwidth",
        "  Stress     Stress test controls",
        "",
        "Press ? to close",
    ];

    for line in &lines {
        let padding = overlay_w.saturating_sub(line.len() + 4);
        let row = format!("\u{2502} {}{} \u{2502}", line, " ".repeat(padding));
        let style = if *line == "Keyboard Controls" || *line == "Tabs" {
            &bright
        } else if *line == "Press ? to close" {
            &dim
        } else {
            &TextStyle { color: theme.foreground, ..Default::default() }
        };
        canvas.draw_text(&row, Point::new(x, y), style);
        y += 1.0;
    }

    // Fill remaining rows
    while y < (start_y + overlay_h) as f32 - 1.0 {
        let empty_padding = overlay_w.saturating_sub(4);
        let row = format!("\u{2502} {} \u{2502}", " ".repeat(empty_padding));
        canvas.draw_text(&row, Point::new(x, y), &dim);
        y += 1.0;
    }

    // Bottom border
    let bottom = format!("\u{2514}{}\u{2518}", "\u{2500}".repeat(overlay_w.saturating_sub(2)));
    canvas.draw_text(&bottom, Point::new(x, y), &dim);
}
