//! Trace waterfall panel - OTLP span Gantt chart

use crate::visualize::app::VisualizeApp;
use crate::visualize::theme::borders;
use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

/// Draw the trace waterfall panel
pub fn draw(f: &mut Frame, app: &VisualizeApp, area: Rect) {
    // Build header
    let header = " Trace Waterfall ";

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(borders::STYLE)
        .border_style(Style::default().fg(borders::TRACE_WATERFALL))
        .title(Span::styled(
            header,
            Style::default().fg(borders::TRACE_WATERFALL).add_modifier(Modifier::BOLD),
        ));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height < 2 {
        return;
    }

    // Placeholder - would render OTLP spans as Gantt bars
    // In full implementation, this would:
    // 1. Receive spans from OtlpExporter
    // 2. Calculate time range and scale
    // 3. Render horizontal bars proportional to duration
    // 4. Indent child spans
    // 5. Highlight critical path

    let placeholder = if app.config.otlp_endpoint.is_some() {
        vec![
            Line::from(Span::styled("Waiting for spans...", Style::default().fg(Color::DarkGray))),
            Line::from(""),
            Line::from(Span::styled(
                "Spans will appear as they are exported via OTLP",
                Style::default().fg(Color::DarkGray),
            )),
        ]
    } else {
        vec![
            Line::from(Span::styled("OTLP not configured", Style::default().fg(Color::DarkGray))),
            Line::from(""),
            Line::from(Span::styled(
                "Use --otlp-endpoint to enable span export",
                Style::default().fg(Color::DarkGray),
            )),
        ]
    };

    let paragraph = Paragraph::new(placeholder);
    f.render_widget(paragraph, inner);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visualize::VisualizeConfig;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    #[test]
    fn test_draw_trace_waterfall() {
        let backend = TestBackend::new(60, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = VisualizeApp::new(VisualizeConfig::default());

        terminal
            .draw(|f| {
                draw(f, &app, f.area());
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content = buffer_to_string(buffer);
        assert!(content.contains("Trace Waterfall"));
    }

    #[test]
    fn test_draw_trace_waterfall_small_area() {
        let backend = TestBackend::new(20, 3);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = VisualizeApp::new(VisualizeConfig::default());

        // Should not panic with small area (height < 2 after border)
        terminal
            .draw(|f| {
                draw(f, &app, f.area());
            })
            .unwrap();
    }

    #[test]
    fn test_draw_trace_waterfall_very_small() {
        let backend = TestBackend::new(10, 2);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = VisualizeApp::new(VisualizeConfig::default());

        // Should handle very small area gracefully
        terminal
            .draw(|f| {
                draw(f, &app, f.area());
            })
            .unwrap();
    }

    #[test]
    fn test_draw_trace_waterfall_with_otlp() {
        let backend = TestBackend::new(60, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        let config = VisualizeConfig {
            otlp_endpoint: Some("http://localhost:4317".to_string()),
            ..Default::default()
        };
        let app = VisualizeApp::new(config);

        terminal
            .draw(|f| {
                draw(f, &app, f.area());
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content = buffer_to_string(buffer);
        assert!(content.contains("Trace Waterfall"));
        assert!(content.contains("Waiting"));
    }

    #[test]
    fn test_draw_trace_waterfall_no_otlp() {
        let backend = TestBackend::new(60, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = VisualizeApp::new(VisualizeConfig::default());

        terminal
            .draw(|f| {
                draw(f, &app, f.area());
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content = buffer_to_string(buffer);
        assert!(content.contains("OTLP not configured"));
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
