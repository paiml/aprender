//! Stats summary panel - aggregate statistics

use crate::visualize::app::VisualizeApp;
use crate::visualize::theme::{borders, format_duration_us, format_rate};
use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

/// Draw the stats summary panel
pub fn draw(f: &mut Frame, app: &VisualizeApp, area: Rect) {
    let header = " Stats ";

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(borders::STYLE)
        .border_style(Style::default().fg(borders::STATS_SUMMARY))
        .title(Span::styled(
            header,
            Style::default().fg(borders::STATS_SUMMARY).add_modifier(Modifier::BOLD),
        ));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height < 4 {
        return;
    }

    // Calculate statistics
    let avg_latency = app.latency_history.avg();
    let min_latency = app.latency_history.min();
    let max_latency = app.latency_history.max();
    let latency_stddev = app.latency_history.stddev();

    let avg_zscore = app.anomaly_history.avg();

    // Build stats display
    let stats = vec![
        Line::from(vec![
            Span::styled("Total Calls: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", app.total_syscalls), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Call Rate:   ", Style::default().fg(Color::DarkGray)),
            Span::styled(format_rate(app.syscall_rate), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::styled("Errors:      ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}", app.total_errors),
                Style::default().fg(if app.total_errors > 0 { Color::Red } else { Color::Green }),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Latency Avg: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format_duration_us(avg_latency as u64), Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("Latency Min: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format_duration_us(min_latency as u64), Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::styled("Latency Max: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format_duration_us(max_latency as u64), Style::default().fg(Color::Red)),
        ]),
        Line::from(vec![
            Span::styled("Latency σ:   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format_duration_us(latency_stddev as u64),
                Style::default().fg(Color::Magenta),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Anomalies:   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}", app.anomaly_count),
                Style::default().fg(if app.anomaly_count > 0 {
                    Color::Yellow
                } else {
                    Color::Green
                }),
            ),
        ]),
        Line::from(vec![
            Span::styled("Avg Z-Score: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.2}σ", avg_zscore), Style::default().fg(Color::Cyan)),
        ]),
    ];

    let paragraph = Paragraph::new(stats);
    f.render_widget(paragraph, inner);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visualize::VisualizeConfig;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    #[test]
    fn test_draw_stats_summary() {
        let backend = TestBackend::new(40, 15);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = VisualizeApp::new(VisualizeConfig::default());

        terminal
            .draw(|f| {
                draw(f, &app, f.area());
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content = buffer_to_string(buffer);
        assert!(content.contains("Stats"));
        assert!(content.contains("Total Calls"));
    }

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
}
