//! UI layout and rendering for renacer visualize (ttop-identical pattern)
//!
//! Uses ratatui for terminal rendering with btop-style panels.

use super::app::VisualizeApp;
use super::panels;
use super::theme::borders;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
    Frame,
};

/// Main draw function - renders all visible panels
pub fn draw(f: &mut Frame, app: &mut VisualizeApp) {
    let size = f.area();

    // Guard against zero-sized terminal (e.g., pseudo-TTY with no dimensions)
    if size.width == 0 || size.height == 0 {
        return;
    }

    // Calculate layout based on visible panels
    let top_panel_count = app.count_top_panels();

    if top_panel_count == 0 && !app.panels.process_syscalls {
        // Nothing to show
        draw_empty_state(f, size);
        return;
    }

    // Two-row layout: top panels and bottom process table
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(if app.panels.process_syscalls {
            vec![Constraint::Percentage(45), Constraint::Percentage(55)]
        } else {
            vec![Constraint::Percentage(100)]
        })
        .split(size);

    // Draw top panels
    if top_panel_count > 0 {
        draw_top_panels(f, app, main_chunks[0]);
    }

    // Draw process table (bottom)
    if app.panels.process_syscalls && main_chunks.len() > 1 {
        panels::process_syscalls::draw(f, app, main_chunks[1]);
    }

    // Draw overlays
    if app.show_help {
        draw_help_overlay(f, size);
    }

    if app.show_filter_input {
        draw_filter_input(f, app, size);
    }

    // Draw FPS overlay if enabled
    if app.config.show_fps {
        draw_fps_overlay(f, app, size);
    }

    // Draw trace complete message
    if app.trace_complete {
        draw_trace_complete_overlay(f, size);
    }
}

/// Draw top panels in adaptive grid
fn draw_top_panels(f: &mut Frame, app: &mut VisualizeApp, area: Rect) {
    let panel_count = app.count_top_panels();
    if panel_count == 0 {
        return;
    }

    // Determine grid layout based on panel count
    let constraints: Vec<Constraint> = match panel_count {
        1 => vec![Constraint::Percentage(100)],
        2 => vec![Constraint::Percentage(50), Constraint::Percentage(50)],
        3 => {
            vec![Constraint::Percentage(34), Constraint::Percentage(33), Constraint::Percentage(33)]
        }
        4 => vec![
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ],
        _ => vec![
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
        ],
    };

    let chunks =
        Layout::default().direction(Direction::Horizontal).constraints(constraints).split(area);

    let mut chunk_idx = 0;

    // Draw each enabled panel
    if app.panels.syscall_heatmap && chunk_idx < chunks.len() {
        panels::syscall_heatmap::draw(f, app, chunks[chunk_idx]);
        chunk_idx += 1;
    }

    if app.panels.anomaly_timeline && chunk_idx < chunks.len() {
        panels::anomaly_timeline::draw(f, app, chunks[chunk_idx]);
        chunk_idx += 1;
    }

    if app.panels.ml_scatter && chunk_idx < chunks.len() {
        panels::ml_scatter::draw(f, app, chunks[chunk_idx]);
        chunk_idx += 1;
    }

    if app.panels.trace_waterfall && chunk_idx < chunks.len() {
        panels::trace_waterfall::draw(f, app, chunks[chunk_idx]);
        chunk_idx += 1;
    }

    if app.panels.stats_summary && chunk_idx < chunks.len() {
        panels::stats_summary::draw(f, app, chunks[chunk_idx]);
    }
}

/// Draw empty state message
fn draw_empty_state(f: &mut Frame, area: Rect) {
    let text = vec![
        Line::from("No panels visible"),
        Line::from(""),
        Line::from("Press 1-6 to toggle panels, ? for help"),
    ];
    let paragraph = Paragraph::new(text).style(Style::default().fg(Color::DarkGray)).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(borders::STYLE)
            .title(" renacer visualize "),
    );
    f.render_widget(paragraph, area);
}

/// Draw help overlay
fn draw_help_overlay(f: &mut Frame, area: Rect) {
    // Calculate centered popup area
    let popup_area = centered_rect(60, 70, area);

    // Clear background
    f.render_widget(Clear, popup_area);

    let help_text = vec![
        Line::from(Span::styled(
            "Keyboard Shortcuts",
            Style::default().add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled("Navigation", Style::default().add_modifier(Modifier::UNDERLINED))),
        Line::from("  j/↓     Move down"),
        Line::from("  k/↑     Move up"),
        Line::from("  g       Go to top"),
        Line::from("  G       Go to bottom"),
        Line::from("  PgUp    Page up"),
        Line::from("  PgDn    Page down"),
        Line::from(""),
        Line::from(Span::styled("Panels", Style::default().add_modifier(Modifier::UNDERLINED))),
        Line::from("  1       Toggle Syscall Heatmap"),
        Line::from("  2       Toggle Anomaly Timeline"),
        Line::from("  3       Toggle ML Clusters"),
        Line::from("  4       Toggle Trace Waterfall"),
        Line::from("  5       Toggle Process Syscalls"),
        Line::from("  6       Toggle Stats Summary"),
        Line::from("  0       Reset all panels"),
        Line::from(""),
        Line::from(Span::styled(
            "Sorting & Filtering",
            Style::default().add_modifier(Modifier::UNDERLINED),
        )),
        Line::from("  s/Tab   Cycle sort column"),
        Line::from("  r       Reverse sort order"),
        Line::from("  f/      Open filter"),
        Line::from("  Del     Clear filter"),
        Line::from(""),
        Line::from(Span::styled("General", Style::default().add_modifier(Modifier::UNDERLINED))),
        Line::from("  ?/F1    Toggle this help"),
        Line::from("  q       Quit"),
        Line::from("  Esc     Close overlay"),
    ];

    let help = Paragraph::new(help_text)
        .style(Style::default().fg(Color::White))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(borders::STYLE)
                .border_style(Style::default().fg(borders::HELP))
                .title(" Help "),
        )
        .wrap(Wrap { trim: false });

    f.render_widget(help, popup_area);
}

/// Draw filter input overlay
fn draw_filter_input(f: &mut Frame, app: &VisualizeApp, area: Rect) {
    let popup_area = centered_rect(40, 10, area);

    // Clear background
    f.render_widget(Clear, popup_area);

    let filter_text = vec![
        Line::from("Filter processes:"),
        Line::from(""),
        Line::from(Span::styled(format!(">{}_", app.filter), Style::default().fg(Color::Yellow))),
        Line::from(""),
        Line::from(Span::styled(
            "Enter to confirm, Esc to cancel",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let filter = Paragraph::new(filter_text).style(Style::default().fg(Color::White)).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(borders::STYLE)
            .border_style(Style::default().fg(Color::Yellow))
            .title(" Filter "),
    );

    f.render_widget(filter, popup_area);
}

/// Draw FPS overlay
fn draw_fps_overlay(f: &mut Frame, app: &VisualizeApp, area: Rect) {
    let fps_text = format!(
        " Frame: {} | Avg: {}us | Max: {}us ",
        app.frame_id, app.avg_frame_time_us, app.max_frame_time_us
    );

    let text_len = fps_text.len();

    let fps = Paragraph::new(fps_text).style(Style::default().fg(Color::DarkGray).bg(Color::Black));

    // Position in top-right corner
    let fps_area = Rect {
        x: area.width.saturating_sub(text_len as u16 + 2),
        y: 0,
        width: text_len as u16 + 2,
        height: 1,
    };

    f.render_widget(fps, fps_area);
}

/// Draw trace complete banner at bottom
fn draw_trace_complete_overlay(f: &mut Frame, area: Rect) {
    let text = " TRACE COMPLETE - Press 'q' to exit ";
    let text_len = text.len() as u16;

    let banner = Paragraph::new(text)
        .style(Style::default().fg(Color::Black).bg(Color::Green).add_modifier(Modifier::BOLD));

    // Center at bottom of screen
    let x = area.width.saturating_sub(text_len) / 2;
    let banner_area = Rect { x, y: area.height.saturating_sub(1), width: text_len, height: 1 };

    f.render_widget(banner, banner_area);
}

/// Create a centered rectangle
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visualize::app::VisualizeApp;
    use crate::visualize::VisualizeConfig;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn buffer_to_string(buffer: &ratatui::buffer::Buffer) -> String {
        let mut s = String::new();
        for y in 0..buffer.area.height {
            for x in 0..buffer.area.width {
                s.push(buffer[(x, y)].symbol().chars().next().unwrap_or(' '));
            }
            s.push('\n');
        }
        s
    }

    #[test]
    fn test_centered_rect() {
        let area = Rect::new(0, 0, 100, 50);
        let centered = centered_rect(50, 50, area);

        // Should be roughly centered
        assert!(centered.x > 0);
        assert!(centered.y > 0);
        assert!(centered.width < 100);
        assert!(centered.height < 50);
    }

    #[test]
    fn test_draw_empty_state() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal
            .draw(|f| {
                draw_empty_state(f, f.area());
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content = buffer_to_string(buffer);
        assert!(content.contains("No panels visible"));
    }

    #[test]
    fn test_draw_help_overlay() {
        let backend = TestBackend::new(100, 40);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal
            .draw(|f| {
                draw_help_overlay(f, f.area());
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content = buffer_to_string(buffer);
        assert!(content.contains("Keyboard Shortcuts"));
        assert!(content.contains("Navigation"));
    }

    #[test]
    fn test_draw_filter_input() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = VisualizeApp::new(VisualizeConfig::default());

        terminal
            .draw(|f| {
                draw_filter_input(f, &app, f.area());
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content = buffer_to_string(buffer);
        assert!(content.contains("Filter"));
    }

    #[test]
    fn test_draw_fps_overlay() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = VisualizeApp::new(VisualizeConfig::default());

        terminal
            .draw(|f| {
                draw_fps_overlay(f, &app, f.area());
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content = buffer_to_string(buffer);
        assert!(content.contains("Frame"));
    }

    #[test]
    fn test_draw_trace_complete_overlay() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal
            .draw(|f| {
                draw_trace_complete_overlay(f, f.area());
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content = buffer_to_string(buffer);
        assert!(content.contains("TRACE COMPLETE"));
    }

    #[test]
    fn test_draw_with_zero_size() {
        let backend = TestBackend::new(0, 0);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = VisualizeApp::new(VisualizeConfig::default());

        // Should not panic with zero-sized terminal
        let _ = terminal.draw(|f| {
            draw(f, &mut app);
        });
    }

    #[test]
    fn test_draw_with_all_panels_disabled() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = VisualizeApp::new(VisualizeConfig::default());
        app.panels.syscall_heatmap = false;
        app.panels.anomaly_timeline = false;
        app.panels.ml_scatter = false;
        app.panels.trace_waterfall = false;
        app.panels.stats_summary = false;
        app.panels.process_syscalls = false;

        terminal
            .draw(|f| {
                draw(f, &mut app);
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content = buffer_to_string(buffer);
        assert!(content.contains("No panels visible"));
    }

    #[test]
    fn test_draw_with_help_visible() {
        let backend = TestBackend::new(100, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = VisualizeApp::new(VisualizeConfig::default());
        app.show_help = true;

        terminal
            .draw(|f| {
                draw(f, &mut app);
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content = buffer_to_string(buffer);
        assert!(content.contains("Keyboard Shortcuts"));
    }

    #[test]
    fn test_draw_with_filter_visible() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = VisualizeApp::new(VisualizeConfig::default());
        app.show_filter_input = true;
        app.filter = "read".to_string();

        terminal
            .draw(|f| {
                draw(f, &mut app);
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content = buffer_to_string(buffer);
        assert!(content.contains("Filter"));
    }

    #[test]
    fn test_draw_with_fps_visible() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = VisualizeApp::new(VisualizeConfig { show_fps: true, ..Default::default() });

        terminal
            .draw(|f| {
                draw(f, &mut app);
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content = buffer_to_string(buffer);
        assert!(content.contains("Frame"));
    }

    #[test]
    fn test_draw_with_trace_complete() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = VisualizeApp::new(VisualizeConfig::default());
        app.trace_complete = true;

        terminal
            .draw(|f| {
                draw(f, &mut app);
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content = buffer_to_string(buffer);
        assert!(content.contains("TRACE COMPLETE"));
    }

    #[test]
    fn test_draw_top_panels_one() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = VisualizeApp::new(VisualizeConfig::default());
        app.panels.syscall_heatmap = true;
        app.panels.anomaly_timeline = false;
        app.panels.ml_scatter = false;
        app.panels.trace_waterfall = false;
        app.panels.stats_summary = false;

        terminal
            .draw(|f| {
                draw_top_panels(f, &mut app, f.area());
            })
            .unwrap();
    }

    #[test]
    fn test_draw_top_panels_two() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = VisualizeApp::new(VisualizeConfig::default());
        app.panels.syscall_heatmap = true;
        app.panels.anomaly_timeline = true;
        app.panels.ml_scatter = false;
        app.panels.trace_waterfall = false;
        app.panels.stats_summary = false;

        terminal
            .draw(|f| {
                draw_top_panels(f, &mut app, f.area());
            })
            .unwrap();
    }

    #[test]
    fn test_draw_top_panels_three() {
        let backend = TestBackend::new(120, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = VisualizeApp::new(VisualizeConfig::default());
        app.panels.syscall_heatmap = true;
        app.panels.anomaly_timeline = true;
        app.panels.ml_scatter = true;
        app.panels.trace_waterfall = false;
        app.panels.stats_summary = false;

        terminal
            .draw(|f| {
                draw_top_panels(f, &mut app, f.area());
            })
            .unwrap();
    }

    #[test]
    fn test_draw_top_panels_four() {
        let backend = TestBackend::new(120, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = VisualizeApp::new(VisualizeConfig::default());
        app.panels.syscall_heatmap = true;
        app.panels.anomaly_timeline = true;
        app.panels.ml_scatter = true;
        app.panels.trace_waterfall = true;
        app.panels.stats_summary = false;

        terminal
            .draw(|f| {
                draw_top_panels(f, &mut app, f.area());
            })
            .unwrap();
    }

    #[test]
    fn test_draw_top_panels_five() {
        let backend = TestBackend::new(150, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = VisualizeApp::new(VisualizeConfig::default());
        app.panels.syscall_heatmap = true;
        app.panels.anomaly_timeline = true;
        app.panels.ml_scatter = true;
        app.panels.trace_waterfall = true;
        app.panels.stats_summary = true;

        terminal
            .draw(|f| {
                draw_top_panels(f, &mut app, f.area());
            })
            .unwrap();
    }

    #[test]
    fn test_draw_top_panels_zero() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = VisualizeApp::new(VisualizeConfig::default());
        app.panels.syscall_heatmap = false;
        app.panels.anomaly_timeline = false;
        app.panels.ml_scatter = false;
        app.panels.trace_waterfall = false;
        app.panels.stats_summary = false;

        terminal
            .draw(|f| {
                draw_top_panels(f, &mut app, f.area());
            })
            .unwrap();
    }

    #[test]
    fn test_draw_with_process_syscalls() {
        let backend = TestBackend::new(80, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = VisualizeApp::new(VisualizeConfig::default());
        app.panels.process_syscalls = true;

        terminal
            .draw(|f| {
                draw(f, &mut app);
            })
            .unwrap();
    }

    #[test]
    fn test_centered_rect_edge_cases() {
        // Very small area
        let area = Rect::new(0, 0, 10, 10);
        let centered = centered_rect(80, 80, area);
        assert!(centered.width > 0);
        assert!(centered.height > 0);

        // Zero percent
        let centered = centered_rect(0, 0, area);
        assert_eq!(centered.width, 0);
        assert_eq!(centered.height, 0);

        // 100 percent
        let centered = centered_rect(100, 100, area);
        assert_eq!(centered.x, 0);
        assert_eq!(centered.y, 0);
    }
}
