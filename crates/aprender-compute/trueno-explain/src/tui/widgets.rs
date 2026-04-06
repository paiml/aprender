//! Sidebar widget rendering for the TUI dashboard.
//!
//! Contains render functions for registers, memory, roofline, bug hunting,
//! and muda (waste) warning panels using presentar canvas drawing.

use presentar_core::{Canvas, Color, Point, TextStyle};
use presentar_terminal::direct::DirectTerminalCanvas;

use crate::analyzer::MudaType;
use crate::ptx::BugSeverity;

use super::TuiApp;

/// Color constants for sidebar rendering.
const COLOR_GREEN: Color = Color { r: 0.3, g: 1.0, b: 0.3, a: 1.0 };
const COLOR_YELLOW: Color = Color { r: 1.0, g: 1.0, b: 0.3, a: 1.0 };
const COLOR_RED: Color = Color { r: 1.0, g: 0.3, b: 0.3, a: 1.0 };
const COLOR_BLUE: Color = Color { r: 0.3, g: 0.5, b: 1.0, a: 1.0 };
const COLOR_TEXT: Color = Color { r: 0.8, g: 0.8, b: 0.8, a: 1.0 };

/// Draw a box border top line: `┌─ Title ───┐`
fn draw_box_top(
    canvas: &mut DirectTerminalCanvas<'_>,
    title: &str,
    x: f32,
    y: f32,
    width: u16,
    color: Color,
) {
    let style = TextStyle { color, ..Default::default() };
    let inner = (width as usize).saturating_sub(2);
    let title_part = format!("┌─ {} ", title);
    let fill_len = inner.saturating_sub(title_part.len().saturating_sub(1));
    let line = format!("{}{}┐", title_part, "─".repeat(fill_len));
    canvas.draw_text(&line, Point::new(x, y), &style);
}

/// Draw a box border bottom line: `└───────────┘`
fn draw_box_bottom(
    canvas: &mut DirectTerminalCanvas<'_>,
    x: f32,
    y: f32,
    width: u16,
    color: Color,
) {
    let style = TextStyle { color, ..Default::default() };
    let inner = (width as usize).saturating_sub(2);
    let line = format!("└{}┘", "─".repeat(inner));
    canvas.draw_text(&line, Point::new(x, y), &style);
}

/// Draw a bordered text line: `│ text...   │` padded to width.
fn draw_box_line(
    canvas: &mut DirectTerminalCanvas<'_>,
    text: &str,
    x: f32,
    y: f32,
    width: u16,
    text_color: Color,
    border_color: Color,
) {
    let border_style = TextStyle { color: border_color, ..Default::default() };
    let text_style = TextStyle { color: text_color, ..Default::default() };
    let inner = (width as usize).saturating_sub(4); // "│ " + " │"
    let padded: String = if text.len() > inner {
        text.chars().take(inner).collect()
    } else {
        format!("{:<width$}", text, width = inner)
    };
    canvas.draw_text("│ ", Point::new(x, y), &border_style);
    canvas.draw_text(&padded, Point::new(x + 2.0, y), &text_style);
    canvas.draw_text(" │", Point::new(x + 2.0 + inner as f32, y), &border_style);
}

/// Render the full sidebar onto the canvas.
pub(super) fn render_sidebar(
    canvas: &mut DirectTerminalCanvas<'_>,
    app: &TuiApp,
    x: f32,
    y: f32,
    width: u16,
    height: u16,
) {
    // Section heights (matching original layout constraints)
    let reg_h: u16 = 8;
    let mem_h: u16 = 6;
    let roof_h: u16 = 5;
    let bug_h: u16 = 6;
    let warn_h: u16 = height.saturating_sub(reg_h + mem_h + roof_h + bug_h);

    let mut cy = y;

    render_register_panel(canvas, app, x, cy, width, reg_h);
    cy += f32::from(reg_h);

    render_memory_panel(canvas, app, x, cy, width, mem_h);
    cy += f32::from(mem_h);

    render_roofline_panel(canvas, app, x, cy, width, roof_h);
    cy += f32::from(roof_h);

    render_bugs_panel(canvas, app, x, cy, width, bug_h);
    cy += f32::from(bug_h);

    render_warnings_panel(canvas, app, x, cy, width, warn_h);
}

fn render_register_panel(
    canvas: &mut DirectTerminalCanvas<'_>,
    app: &TuiApp,
    x: f32,
    y: f32,
    width: u16,
    _height: u16,
) {
    let regs = &app.report.registers;
    let total = regs.total();
    let occupancy = app.report.estimated_occupancy;

    let status_color = if total < 64 {
        COLOR_GREEN
    } else if total < 128 {
        COLOR_YELLOW
    } else {
        COLOR_RED
    };

    draw_box_top(canvas, "Registers", x, y, width, status_color);
    draw_box_line(
        canvas,
        &format!(".f32: {:3} / 255", regs.f32_regs),
        x,
        y + 1.0,
        width,
        COLOR_TEXT,
        status_color,
    );
    draw_box_line(
        canvas,
        &format!(".b32: {:3} / 255", regs.b32_regs),
        x,
        y + 2.0,
        width,
        COLOR_TEXT,
        status_color,
    );
    draw_box_line(
        canvas,
        &format!(".b64: {:3} / 255", regs.b64_regs),
        x,
        y + 3.0,
        width,
        COLOR_TEXT,
        status_color,
    );
    draw_box_line(
        canvas,
        &format!(".pred: {:2} / 8", regs.pred_regs),
        x,
        y + 4.0,
        width,
        COLOR_TEXT,
        status_color,
    );

    // Total + occupancy line with status color for the value
    let occ_text = format!("Total: {} -> {:.0}% occ", total, occupancy * 100.0);
    draw_box_line(canvas, &occ_text, x, y + 5.0, width, status_color, status_color);

    // Empty line + bottom border
    draw_box_line(canvas, "", x, y + 6.0, width, COLOR_TEXT, status_color);
    draw_box_bottom(canvas, x, y + 7.0, width, status_color);
}

fn render_memory_panel(
    canvas: &mut DirectTerminalCanvas<'_>,
    app: &TuiApp,
    x: f32,
    y: f32,
    width: u16,
    _height: u16,
) {
    let mem = &app.report.memory;
    let coal_pct = mem.coalesced_ratio * 100.0;

    let status_color = if coal_pct >= 90.0 {
        COLOR_GREEN
    } else if coal_pct >= 70.0 {
        COLOR_YELLOW
    } else {
        COLOR_RED
    };

    draw_box_top(canvas, "Memory", x, y, width, status_color);
    draw_box_line(
        canvas,
        &format!("Global ld: {}", mem.global_loads),
        x,
        y + 1.0,
        width,
        COLOR_TEXT,
        status_color,
    );
    draw_box_line(
        canvas,
        &format!("Global st: {}", mem.global_stores),
        x,
        y + 2.0,
        width,
        COLOR_TEXT,
        status_color,
    );

    let coal_text = format!("Coalesced: {:.1}%", coal_pct);
    draw_box_line(canvas, &coal_text, x, y + 3.0, width, status_color, status_color);

    draw_box_line(canvas, "", x, y + 4.0, width, COLOR_TEXT, status_color);
    draw_box_bottom(canvas, x, y + 5.0, width, status_color);
}

fn render_roofline_panel(
    canvas: &mut DirectTerminalCanvas<'_>,
    app: &TuiApp,
    x: f32,
    y: f32,
    width: u16,
    _height: u16,
) {
    let roof = &app.report.roofline;

    let (bound_text, bound_color) = if roof.memory_bound {
        ("Memory-bound", COLOR_YELLOW)
    } else {
        ("Compute-bound", COLOR_GREEN)
    };

    draw_box_top(canvas, "Roofline", x, y, width, COLOR_BLUE);
    draw_box_line(
        canvas,
        &format!("AI: {:.2} FLOP/B", roof.arithmetic_intensity),
        x,
        y + 1.0,
        width,
        COLOR_TEXT,
        COLOR_BLUE,
    );
    draw_box_line(
        canvas,
        &format!("Bottleneck: {}", bound_text),
        x,
        y + 2.0,
        width,
        bound_color,
        COLOR_BLUE,
    );
    draw_box_line(canvas, "", x, y + 3.0, width, COLOR_TEXT, COLOR_BLUE);
    draw_box_bottom(canvas, x, y + 4.0, width, COLOR_BLUE);
}

fn render_bugs_panel(
    canvas: &mut DirectTerminalCanvas<'_>,
    app: &TuiApp,
    x: f32,
    y: f32,
    width: u16,
    _height: u16,
) {
    let bug_report = &app.bug_report;
    let critical = bug_report.count_by_severity(BugSeverity::Critical);
    let high = bug_report.count_by_severity(BugSeverity::High);
    let medium = bug_report.count_by_severity(BugSeverity::Medium);

    let status_color = if critical > 0 {
        COLOR_RED
    } else if high > 0 {
        COLOR_YELLOW
    } else {
        COLOR_GREEN
    };

    draw_box_top(canvas, "Bug Hunt", x, y, width, status_color);

    if bug_report.bugs.is_empty() {
        draw_box_line(canvas, "No bugs detected", x, y + 1.0, width, COLOR_GREEN, status_color);
        draw_box_line(canvas, "", x, y + 2.0, width, COLOR_TEXT, status_color);
        draw_box_line(canvas, "", x, y + 3.0, width, COLOR_TEXT, status_color);
    } else {
        let c_color = if critical > 0 { COLOR_RED } else { COLOR_GREEN };
        let h_color = if high > 0 { COLOR_YELLOW } else { COLOR_GREEN };
        let m_color = if medium > 0 { COLOR_BLUE } else { COLOR_GREEN };

        draw_box_line(
            canvas,
            &format!("P0 Critical: {}", critical),
            x,
            y + 1.0,
            width,
            c_color,
            status_color,
        );
        draw_box_line(
            canvas,
            &format!("P1 High: {}", high),
            x,
            y + 2.0,
            width,
            h_color,
            status_color,
        );
        draw_box_line(
            canvas,
            &format!("P2 Medium: {}", medium),
            x,
            y + 3.0,
            width,
            m_color,
            status_color,
        );
    }

    draw_box_line(canvas, "", x, y + 4.0, width, COLOR_TEXT, status_color);
    draw_box_bottom(canvas, x, y + 5.0, width, status_color);
}

fn render_warnings_panel(
    canvas: &mut DirectTerminalCanvas<'_>,
    app: &TuiApp,
    x: f32,
    y: f32,
    width: u16,
    height: u16,
) {
    let border_color = if app.report.warnings.is_empty() { COLOR_GREEN } else { COLOR_YELLOW };

    draw_box_top(canvas, "Muda (Waste)", x, y, width, border_color);

    if app.report.warnings.is_empty() {
        draw_box_line(canvas, "No Muda detected", x, y + 1.0, width, COLOR_GREEN, border_color);
        // Fill remaining lines
        for row in 2..height.saturating_sub(1) {
            draw_box_line(canvas, "", x, y + f32::from(row), width, COLOR_TEXT, border_color);
        }
    } else {
        let max_items = (height as usize).saturating_sub(2);
        for (i, w) in app.report.warnings.iter().take(max_items).enumerate() {
            let icon = match w.muda_type {
                MudaType::Transport => "! ",
                MudaType::Waiting => "~ ",
                MudaType::Overprocessing => "@ ",
            };
            let text = format!("{}{}", icon, w.description);
            draw_box_line(canvas, &text, x, y + 1.0 + i as f32, width, COLOR_YELLOW, border_color);
        }
        // Fill remaining lines
        let used = app.report.warnings.len().min(max_items);
        for row in (used + 1)..height.saturating_sub(1) as usize {
            draw_box_line(canvas, "", x, y + row as f32, width, COLOR_TEXT, border_color);
        }
    }

    draw_box_bottom(canvas, x, y + f32::from(height - 1), width, border_color);
}
