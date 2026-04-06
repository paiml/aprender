//! Metrics panel - counters, gauges, histograms (Sprint 56)
//!
//! Displays real-time metrics collected during tracing:
//! - Counters: monotonically increasing values (syscall counts, errors)
//! - Gauges: point-in-time values (active connections, queue depth)
//! - Histograms: distribution summaries (latency percentiles)
//!
//! # Toyota Way Principle: Genchi Genbutsu
//!
//! Go and see: metrics provide direct visibility into system behavior.

use crate::visualize::app::VisualizeApp;
use crate::visualize::theme::{borders, format_duration_us};
use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

/// Draw the metrics panel
pub fn draw(f: &mut Frame, app: &VisualizeApp, area: Rect) {
    let header = " Metrics ";

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(borders::STYLE)
        .border_style(Style::default().fg(borders::METRICS))
        .title(Span::styled(
            header,
            Style::default().fg(borders::METRICS).add_modifier(Modifier::BOLD),
        ));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height < 4 {
        return;
    }

    // Build metrics display
    let mut lines: Vec<Line> = Vec::new();

    // Section: Counters
    lines.push(Line::from(Span::styled(
        "─── Counters ───",
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
    )));

    lines.push(Line::from(vec![
        Span::styled("syscall_total:    ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{}", app.total_syscalls), Style::default().fg(Color::White)),
    ]));

    lines.push(Line::from(vec![
        Span::styled("error_total:      ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{}", app.total_errors),
            Style::default().fg(if app.total_errors > 0 { Color::Red } else { Color::Green }),
        ),
    ]));

    lines.push(Line::from(vec![
        Span::styled("anomaly_total:    ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{}", app.anomaly_count),
            Style::default().fg(if app.anomaly_count > 0 { Color::Yellow } else { Color::Green }),
        ),
    ]));

    lines.push(Line::from(""));

    // Section: Gauges
    lines.push(Line::from(Span::styled(
        "─── Gauges ───",
        Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
    )));

    lines.push(Line::from(vec![
        Span::styled("syscall_rate:     ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{:.1}/s", app.syscall_rate), Style::default().fg(Color::Yellow)),
    ]));

    lines.push(Line::from(vec![
        Span::styled("avg_latency:      ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format_duration_us(app.latency_history.avg() as u64),
            Style::default().fg(Color::Cyan),
        ),
    ]));

    lines.push(Line::from(vec![
        Span::styled("avg_zscore:       ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{:.2}σ", app.anomaly_history.avg()),
            Style::default().fg(Color::Magenta),
        ),
    ]));

    lines.push(Line::from(""));

    // Section: Histograms (latency distribution)
    lines.push(Line::from(Span::styled(
        "─── Latency Distribution ───",
        Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
    )));

    let min_latency = app.latency_history.min();
    let max_latency = app.latency_history.max();
    let stddev = app.latency_history.stddev();

    lines.push(Line::from(vec![
        Span::styled("min:              ", Style::default().fg(Color::DarkGray)),
        Span::styled(format_duration_us(min_latency as u64), Style::default().fg(Color::Green)),
    ]));

    lines.push(Line::from(vec![
        Span::styled("max:              ", Style::default().fg(Color::DarkGray)),
        Span::styled(format_duration_us(max_latency as u64), Style::default().fg(Color::Red)),
    ]));

    lines.push(Line::from(vec![
        Span::styled("stddev:           ", Style::default().fg(Color::DarkGray)),
        Span::styled(format_duration_us(stddev as u64), Style::default().fg(Color::Yellow)),
    ]));

    // Visual sparkline of recent latencies
    if !app.latency_history.is_empty() {
        lines.push(Line::from(""));
        let values: Vec<f64> = app.latency_history.iter().copied().collect();
        let spark =
            crate::visualize::theme::sparkline(&values, inner.width.saturating_sub(2) as usize);
        lines.push(Line::from(Span::styled(spark, Style::default().fg(Color::Cyan))));
    }

    let paragraph = Paragraph::new(lines);
    f.render_widget(paragraph, inner);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visualize::VisualizeConfig;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    #[test]
    fn test_draw_metrics_panel() {
        let backend = TestBackend::new(40, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = VisualizeApp::new(VisualizeConfig::default());

        terminal
            .draw(|f| {
                draw(f, &app, f.area());
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content = buffer_to_string(buffer);
        assert!(content.contains("Metrics"));
        assert!(content.contains("Counters"));
        assert!(content.contains("Gauges"));
    }

    #[test]
    fn test_draw_metrics_small_area() {
        let backend = TestBackend::new(20, 3);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = VisualizeApp::new(VisualizeConfig::default());

        // Should not panic with small area
        terminal
            .draw(|f| {
                draw(f, &app, f.area());
            })
            .unwrap();
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
