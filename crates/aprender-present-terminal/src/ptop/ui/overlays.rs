//! Overlay dialogs and status bar for ptop.
//!
//! This module handles:
//! - Help overlay (`?` key)
//! - Signal confirmation dialog (kill/term process)
//! - Filter input overlay (`/` key)
//! - Status bar with navigation hints

use crate::direct::DirectTerminalCanvas;
use crate::ptop::app::App;
use crate::ptop::config::{PanelType, SignalType};
use crate::Border;
use presentar_core::{Canvas, Color, Point, Rect, TextStyle, Widget};

use super::colors::{FOCUS_ACCENT_COLOR, STATUS_BAR_BG};

// =============================================================================
// HELP OVERLAY
// =============================================================================

/// Help overlay dimensions
pub const HELP_POPUP_WIDTH: f32 = 55.0;
pub const HELP_POPUP_HEIGHT: f32 = 27.0;

/// Help content lines: (key, description, is_section_header)
pub const HELP_LINES: &[(&str, &str, bool)] = &[
    ("", "-- General --", true),
    ("q, Esc, Ctrl+C", "Quit", false),
    ("h, ?", "Toggle help", false),
    ("", "-- Panel Navigation --", true),
    ("Tab", "Focus next panel", false),
    ("Shift+Tab", "Focus previous panel", false),
    ("hjkl", "Vim-style focus navigation", false),
    ("Enter, z", "Explode/zoom focused panel", false),
    ("", "-- Process List --", true),
    ("j/k, ↑/↓", "Navigate processes", false),
    ("PgUp/PgDn", "Page up/down", false),
    ("g/G", "Go to top/bottom", false),
    ("c/m/p", "Sort by CPU/Memory/PID", false),
    ("s", "Cycle sort column", false),
    ("r", "Reverse sort", false),
    ("/, f", "Filter processes", false),
    ("Delete", "Clear filter", false),
    ("", "-- Signals --", true),
    ("x", "SIGTERM (graceful stop)", false),
    ("X", "SIGKILL (force kill)", false),
    ("", "-- Panels --", true),
    ("1-5", "Toggle panels", false),
    ("0", "Reset panels", false),
];

/// Draw the help overlay (activated by `?` or `h`).
pub fn draw_help_overlay(canvas: &mut DirectTerminalCanvas<'_>, w: f32, h: f32) {
    let popup_w = HELP_POPUP_WIDTH;
    let popup_h = HELP_POPUP_HEIGHT;
    let px = (w - popup_w) / 2.0;
    let py = (h - popup_h) / 2.0;

    // Clear background
    draw_popup_background(
        canvas,
        px,
        py,
        popup_w,
        popup_h,
        Color::new(0.1, 0.1, 0.15, 1.0),
    );

    // Border
    let mut border = Border::new()
        .with_title(" Help ")
        .with_style(crate::BorderStyle::Double)
        .with_color(Color::new(0.3, 0.8, 0.9, 1.0));
    border.layout(Rect::new(px, py, popup_w, popup_h));
    border.paint(canvas);

    // Styles
    let text_style = TextStyle {
        color: Color::new(0.9, 0.9, 0.9, 1.0),
        ..Default::default()
    };
    let key_style = TextStyle {
        color: Color::new(0.3, 0.8, 0.9, 1.0),
        ..Default::default()
    };
    let section_style = TextStyle {
        color: Color::new(0.8, 0.8, 0.2, 1.0),
        ..Default::default()
    };

    // Draw help lines
    for (i, (key, desc, is_section)) in HELP_LINES.iter().enumerate() {
        let y = py + 1.0 + i as f32;
        if *is_section {
            canvas.draw_text(desc, Point::new(px + 2.0, y), &section_style);
        } else {
            canvas.draw_text(&format!("{key:>14}"), Point::new(px + 2.0, y), &key_style);
            canvas.draw_text(desc, Point::new(px + 18.0, y), &text_style);
        }
    }
}

// =============================================================================
// SIGNAL DIALOG
// =============================================================================

/// Signal dialog dimensions
pub const SIGNAL_POPUP_WIDTH: f32 = 50.0;
pub const SIGNAL_POPUP_HEIGHT: f32 = 7.0;

/// Get border color for signal type
#[must_use]
pub fn signal_border_color(signal: SignalType) -> Color {
    match signal {
        SignalType::Kill => Color::new(1.0, 0.3, 0.3, 1.0), // Red
        SignalType::Term => Color::new(1.0, 0.8, 0.2, 1.0), // Yellow
        SignalType::Stop => Color::new(0.8, 0.4, 1.0, 1.0), // Purple
        _ => Color::new(0.3, 0.8, 0.9, 1.0),                // Cyan
    }
}

/// Draw signal confirmation dialog.
pub fn draw_signal_dialog(app: &App, canvas: &mut DirectTerminalCanvas<'_>, w: f32, h: f32) {
    let Some((pid, ref name, signal)) = app.pending_signal else {
        return;
    };

    let popup_w = SIGNAL_POPUP_WIDTH;
    let popup_h = SIGNAL_POPUP_HEIGHT;
    let px = (w - popup_w) / 2.0;
    let py = (h - popup_h) / 2.0;

    // Clear background
    draw_popup_background(
        canvas,
        px,
        py,
        popup_w,
        popup_h,
        Color::new(0.15, 0.1, 0.1, 1.0),
    );

    // Border
    let border_color = signal_border_color(signal);
    let mut border = Border::new()
        .with_title(format!(" Send SIG{} ", signal.name()))
        .with_style(crate::BorderStyle::Double)
        .with_color(border_color);
    border.layout(Rect::new(px, py, popup_w, popup_h));
    border.paint(canvas);

    // Styles
    let text_style = TextStyle {
        color: Color::new(0.9, 0.9, 0.9, 1.0),
        ..Default::default()
    };
    let warning_style = TextStyle {
        color: border_color,
        ..Default::default()
    };
    let hint_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };

    // Truncate process name if too long
    let display_name = truncate_name(name, 25);

    // Dialog content
    canvas.draw_text(
        &format!("Process: {} (PID {})", display_name, pid),
        Point::new(px + 2.0, py + 1.0),
        &text_style,
    );
    canvas.draw_text(
        &format!("Signal: {} - {}", signal.name(), signal.description()),
        Point::new(px + 2.0, py + 2.0),
        &warning_style,
    );
    canvas.draw_text(
        "Send signal? [Y]es / [n]o / [Esc] cancel",
        Point::new(px + 2.0, py + 4.0),
        &text_style,
    );
    canvas.draw_text(
        "x=TERM  K=KILL  H=HUP  i=INT  p=STOP",
        Point::new(px + 2.0, py + 5.0),
        &hint_style,
    );
}

// =============================================================================
// FILTER OVERLAY
// =============================================================================

/// Filter popup dimensions
pub const FILTER_POPUP_WIDTH: f32 = 45.0;
pub const FILTER_POPUP_HEIGHT: f32 = 3.0;

/// Draw filter input overlay.
pub fn draw_filter_overlay(app: &App, canvas: &mut DirectTerminalCanvas<'_>, w: f32, h: f32) {
    let popup_w = FILTER_POPUP_WIDTH;
    let popup_h = FILTER_POPUP_HEIGHT;
    let px = (w - popup_w) / 2.0;
    let py = (h - popup_h) / 2.0;

    let mut border = Border::new()
        .with_title(" Filter Processes ")
        .with_style(crate::BorderStyle::Rounded)
        .with_color(Color::new(0.3, 0.8, 0.9, 1.0));
    border.layout(Rect::new(px, py, popup_w, popup_h));
    border.paint(canvas);

    let filter_display = format!("{}_", app.filter);
    canvas.draw_text(
        &filter_display,
        Point::new(px + 2.0, py + 1.0),
        &TextStyle {
            color: Color::new(1.0, 1.0, 1.0, 1.0),
            ..Default::default()
        },
    );
}

// =============================================================================
// STATUS BAR
// =============================================================================

/// Get panel display name
#[must_use]
pub fn panel_name(panel: PanelType) -> &'static str {
    match panel {
        PanelType::Cpu => "CPU",
        PanelType::Memory => "Memory",
        PanelType::Disk => "Disk",
        PanelType::Network => "Network",
        PanelType::Process => "Process",
        PanelType::Gpu => "GPU",
        PanelType::Battery => "Battery",
        PanelType::Sensors => "Sensors",
        PanelType::Files => "Files",
        PanelType::Connections => "Connections",
        PanelType::Psi => "PSI",
        PanelType::Containers => "Containers",
    }
}

/// Navigation hints for normal view
pub const NORMAL_HINTS: &str = " [Tab]Panel  [Enter]Explode  [↑↓]Row  [/]Filter  [?]Help  [q]Quit ";
/// Navigation hints for exploded view
pub const EXPLODED_HINTS: &str = " [Esc]Exit  [↑↓]Row  [←→]Col  [?]Help  [q]Quit ";

/// Draw status bar with navigation hints.
pub fn draw_status_bar(app: &App, canvas: &mut DirectTerminalCanvas<'_>, w: f32, h: f32) {
    let y = h - 1.0;

    // Styles
    let bracket_style = TextStyle {
        color: Color::new(0.5, 0.5, 0.5, 1.0),
        ..Default::default()
    };
    let key_style = TextStyle {
        color: FOCUS_ACCENT_COLOR,
        ..Default::default()
    };
    let action_style = TextStyle {
        color: Color::new(0.7, 0.7, 0.7, 1.0),
        ..Default::default()
    };
    let focus_indicator_style = TextStyle {
        color: FOCUS_ACCENT_COLOR,
        ..Default::default()
    };

    // Draw background bar
    canvas.fill_rect(Rect::new(0.0, y, w, 1.0), STATUS_BAR_BG);

    // Navigation hints
    let hints = if app.exploded_panel.is_some() {
        EXPLODED_HINTS
    } else {
        NORMAL_HINTS
    };

    // Draw hints with bracket highlighting
    let x_end =
        draw_hints_with_brackets(canvas, hints, y, &bracket_style, &key_style, &action_style);

    // Focus indicator on right
    if let Some(panel) = app.focused_panel {
        let name = panel_name(panel);
        let focus_text = format!("► {name} ");
        let focus_x = w - focus_text.chars().count() as f32 - 1.0;
        if focus_x > x_end {
            canvas.draw_text(&focus_text, Point::new(focus_x, y), &focus_indicator_style);
        }
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Draw popup background (clear area)
fn draw_popup_background(
    canvas: &mut DirectTerminalCanvas<'_>,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    color: Color,
) {
    for row in 0..h as u16 {
        let spaces: String = (0..w as usize).map(|_| ' ').collect();
        canvas.draw_text(
            &spaces,
            Point::new(x, y + row as f32),
            &TextStyle {
                color,
                ..Default::default()
            },
        );
    }
}

/// Truncate name with ellipsis if too long
#[must_use]
pub fn truncate_name(name: &str, max_len: usize) -> String {
    if name.len() > max_len {
        format!("{}...", &name[..max_len.saturating_sub(3)])
    } else {
        name.to_string()
    }
}

/// Draw hints string with bracket-key highlighting, returns x position after drawing
fn draw_hints_with_brackets(
    canvas: &mut DirectTerminalCanvas<'_>,
    hints: &str,
    y: f32,
    bracket_style: &TextStyle,
    key_style: &TextStyle,
    action_style: &TextStyle,
) -> f32 {
    let mut x = 0.0;
    let mut in_bracket = false;
    for ch in hints.chars() {
        let style = if ch == '[' {
            in_bracket = true;
            bracket_style
        } else if ch == ']' {
            in_bracket = false;
            bracket_style
        } else if in_bracket {
            key_style
        } else {
            action_style
        };
        canvas.draw_text(&ch.to_string(), Point::new(x, y), style);
        x += 1.0;
    }
    x
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Constants tests
    // =========================================================================

    #[test]
    fn test_help_popup_dimensions() {
        assert_eq!(HELP_POPUP_WIDTH, 55.0);
        assert_eq!(HELP_POPUP_HEIGHT, 27.0);
    }

    #[test]
    fn test_signal_popup_dimensions() {
        assert_eq!(SIGNAL_POPUP_WIDTH, 50.0);
        assert_eq!(SIGNAL_POPUP_HEIGHT, 7.0);
    }

    #[test]
    fn test_filter_popup_dimensions() {
        assert_eq!(FILTER_POPUP_WIDTH, 45.0);
        assert_eq!(FILTER_POPUP_HEIGHT, 3.0);
    }

    #[test]
    fn test_help_lines_count() {
        // 23 help lines
        assert_eq!(HELP_LINES.len(), 23);
    }

    #[test]
    fn test_help_lines_has_sections() {
        let sections: Vec<_> = HELP_LINES
            .iter()
            .filter(|(_, _, is_sect)| *is_sect)
            .collect();
        assert_eq!(sections.len(), 5, "Should have 5 section headers");
    }

    #[test]
    fn test_help_lines_has_keybindings() {
        let keybindings: Vec<_> = HELP_LINES
            .iter()
            .filter(|(_, _, is_sect)| !*is_sect)
            .collect();
        assert_eq!(keybindings.len(), 18, "Should have 18 keybindings");
    }

    #[test]
    fn test_normal_hints_contains_tab() {
        assert!(NORMAL_HINTS.contains("[Tab]"));
    }

    #[test]
    fn test_normal_hints_contains_quit() {
        assert!(NORMAL_HINTS.contains("[q]Quit"));
    }

    #[test]
    fn test_normal_hints_contains_help() {
        assert!(NORMAL_HINTS.contains("[?]Help"));
    }

    #[test]
    fn test_exploded_hints_contains_esc() {
        assert!(EXPLODED_HINTS.contains("[Esc]Exit"));
    }

    #[test]
    fn test_exploded_hints_differs_from_normal() {
        assert_ne!(NORMAL_HINTS, EXPLODED_HINTS);
    }

    // =========================================================================
    // signal_border_color tests
    // =========================================================================

    #[test]
    fn test_signal_border_color_kill_is_red() {
        let color = signal_border_color(SignalType::Kill);
        assert!(color.r > 0.9, "Kill should be red");
        assert!(color.g < 0.5);
    }

    #[test]
    fn test_signal_border_color_term_is_yellow() {
        let color = signal_border_color(SignalType::Term);
        assert!(color.r > 0.9, "Term should have high red");
        assert!(color.g > 0.7, "Term should have high green (yellow)");
    }

    #[test]
    fn test_signal_border_color_stop_is_purple() {
        let color = signal_border_color(SignalType::Stop);
        assert!(color.b > 0.9, "Stop should have high blue");
        assert!(color.r > 0.7, "Stop should have medium-high red (purple)");
    }

    #[test]
    fn test_signal_border_color_hup_is_cyan() {
        let color = signal_border_color(SignalType::Hup);
        assert!(color.b > 0.8, "Hup should be cyan");
        assert!(color.g > 0.7, "Hup should have green");
    }

    #[test]
    fn test_signal_border_color_int_is_cyan() {
        let color = signal_border_color(SignalType::Int);
        assert!(color.b > 0.8);
    }

    // =========================================================================
    // panel_name tests
    // =========================================================================

    #[test]
    fn test_panel_name_cpu() {
        assert_eq!(panel_name(PanelType::Cpu), "CPU");
    }

    #[test]
    fn test_panel_name_memory() {
        assert_eq!(panel_name(PanelType::Memory), "Memory");
    }

    #[test]
    fn test_panel_name_disk() {
        assert_eq!(panel_name(PanelType::Disk), "Disk");
    }

    #[test]
    fn test_panel_name_network() {
        assert_eq!(panel_name(PanelType::Network), "Network");
    }

    #[test]
    fn test_panel_name_process() {
        assert_eq!(panel_name(PanelType::Process), "Process");
    }

    #[test]
    fn test_panel_name_gpu() {
        assert_eq!(panel_name(PanelType::Gpu), "GPU");
    }

    #[test]
    fn test_panel_name_battery() {
        assert_eq!(panel_name(PanelType::Battery), "Battery");
    }

    #[test]
    fn test_panel_name_sensors() {
        assert_eq!(panel_name(PanelType::Sensors), "Sensors");
    }

    #[test]
    fn test_panel_name_files() {
        assert_eq!(panel_name(PanelType::Files), "Files");
    }

    #[test]
    fn test_panel_name_connections() {
        assert_eq!(panel_name(PanelType::Connections), "Connections");
    }

    #[test]
    fn test_panel_name_psi() {
        assert_eq!(panel_name(PanelType::Psi), "PSI");
    }

    #[test]
    fn test_panel_name_containers() {
        assert_eq!(panel_name(PanelType::Containers), "Containers");
    }

    // =========================================================================
    // truncate_name tests
    // =========================================================================

    #[test]
    fn test_truncate_name_short() {
        assert_eq!(truncate_name("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_name_exact() {
        assert_eq!(truncate_name("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_name_long() {
        assert_eq!(truncate_name("hello world", 8), "hello...");
    }

    #[test]
    fn test_truncate_name_very_long() {
        assert_eq!(
            truncate_name("this is a very long process name", 15),
            "this is a ve..."
        );
    }

    #[test]
    fn test_truncate_name_empty() {
        assert_eq!(truncate_name("", 10), "");
    }

    #[test]
    fn test_truncate_name_max_len_3() {
        // Edge case: max_len = 3, so nothing left after "..."
        assert_eq!(truncate_name("hello", 3), "...");
    }

    #[test]
    fn test_truncate_name_max_len_0() {
        // Edge case: max_len = 0
        assert_eq!(truncate_name("hello", 0), "...");
    }

    // =========================================================================
    // Help line content tests
    // =========================================================================

    #[test]
    fn test_help_lines_quit_exists() {
        let quit_line = HELP_LINES.iter().find(|(k, _, _)| k.contains("q"));
        assert!(quit_line.is_some(), "Should have quit keybinding");
    }

    #[test]
    fn test_help_lines_tab_exists() {
        let tab_line = HELP_LINES.iter().find(|(k, _, _)| k.contains("Tab"));
        assert!(tab_line.is_some(), "Should have Tab keybinding");
    }

    #[test]
    fn test_help_lines_filter_exists() {
        let filter_line = HELP_LINES.iter().find(|(_, d, _)| d.contains("Filter"));
        assert!(filter_line.is_some(), "Should have filter keybinding");
    }

    #[test]
    fn test_help_lines_signal_term_exists() {
        let term_line = HELP_LINES.iter().find(|(_, d, _)| d.contains("SIGTERM"));
        assert!(term_line.is_some(), "Should have SIGTERM keybinding");
    }

    #[test]
    fn test_help_lines_signal_kill_exists() {
        let kill_line = HELP_LINES.iter().find(|(_, d, _)| d.contains("SIGKILL"));
        assert!(kill_line.is_some(), "Should have SIGKILL keybinding");
    }

    #[test]
    fn test_help_lines_vim_navigation() {
        let vim_line = HELP_LINES.iter().find(|(_, d, _)| d.contains("Vim"));
        assert!(vim_line.is_some(), "Should have Vim navigation");
    }

    // =========================================================================
    // Popup centering tests
    // =========================================================================

    #[test]
    fn test_help_popup_center_calculation_80x24() {
        let w = 80.0;
        let h = 24.0;
        let px = (w - HELP_POPUP_WIDTH) / 2.0;
        let py = (h - HELP_POPUP_HEIGHT) / 2.0;
        assert_eq!(px, 12.5);
        assert!(py < 0.0, "Help popup is taller than 24 rows");
    }

    #[test]
    fn test_help_popup_center_calculation_120x40() {
        let w = 120.0;
        let h = 40.0;
        let px = (w - HELP_POPUP_WIDTH) / 2.0;
        let py = (h - HELP_POPUP_HEIGHT) / 2.0;
        assert_eq!(px, 32.5);
        assert_eq!(py, 6.5);
    }

    #[test]
    fn test_signal_popup_center_calculation_80x24() {
        let w = 80.0;
        let h = 24.0;
        let px = (w - SIGNAL_POPUP_WIDTH) / 2.0;
        let py = (h - SIGNAL_POPUP_HEIGHT) / 2.0;
        assert_eq!(px, 15.0);
        assert_eq!(py, 8.5);
    }

    #[test]
    fn test_filter_popup_center_calculation_80x24() {
        let w = 80.0;
        let h = 24.0;
        let px = (w - FILTER_POPUP_WIDTH) / 2.0;
        let py = (h - FILTER_POPUP_HEIGHT) / 2.0;
        assert_eq!(px, 17.5);
        assert_eq!(py, 10.5);
    }

    // =========================================================================
    // Color validity tests
    // =========================================================================

    #[test]
    fn test_signal_colors_have_valid_alpha() {
        for signal in [
            SignalType::Kill,
            SignalType::Term,
            SignalType::Stop,
            SignalType::Hup,
            SignalType::Int,
        ] {
            let color = signal_border_color(signal);
            assert_eq!(color.a, 1.0, "Signal {:?} should have full alpha", signal);
        }
    }

    #[test]
    fn test_signal_colors_in_valid_range() {
        for signal in [
            SignalType::Kill,
            SignalType::Term,
            SignalType::Stop,
            SignalType::Hup,
            SignalType::Int,
        ] {
            let color = signal_border_color(signal);
            assert!(color.r >= 0.0 && color.r <= 1.0);
            assert!(color.g >= 0.0 && color.g <= 1.0);
            assert!(color.b >= 0.0 && color.b <= 1.0);
        }
    }
}
