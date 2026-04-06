//! Syscall heatmap panel - category activity over time

use crate::visualize::app::{SyscallCategory, VisualizeApp};
use crate::visualize::theme::{borders, format_rate, graph, sparkline};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

/// Draw the syscall heatmap panel
pub fn draw(f: &mut Frame, app: &VisualizeApp, area: Rect) {
    // Build header with stats
    let header = format!(
        " Syscalls {} │ {} errors │ top: {} ",
        format_rate(app.syscall_rate),
        app.total_errors,
        app.syscall_counts.iter().max_by_key(|(_, v)| *v).map_or("-", |(k, _)| k.as_str())
    );

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(borders::STYLE)
        .border_style(Style::default().fg(borders::SYSCALL_HEATMAP))
        .title(Span::styled(
            header,
            Style::default().fg(borders::SYSCALL_HEATMAP).add_modifier(Modifier::BOLD),
        ));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height < 2 {
        return;
    }

    // Render category sparklines
    let categories = [
        (SyscallCategory::File, graph::SYSCALL_FILE),
        (SyscallCategory::Network, graph::SYSCALL_NET),
        (SyscallCategory::Memory, graph::SYSCALL_MEM),
        (SyscallCategory::Process, graph::SYSCALL_PROC),
        (SyscallCategory::Other, graph::SYSCALL_OTHER),
    ];

    let constraints: Vec<Constraint> = categories.iter().map(|_| Constraint::Min(1)).collect();
    let rows =
        Layout::default().direction(Direction::Vertical).constraints(constraints).split(inner);

    for (i, (category, color)) in categories.iter().enumerate() {
        if i >= rows.len() {
            break;
        }

        let row = rows[i];
        if row.height < 1 {
            continue;
        }

        // Get history for this category
        #[allow(clippy::redundant_closure_for_method_calls)]
        let history: Vec<f64> =
            app.category_history.get(category).map(|buf| buf.to_vec()).unwrap_or_default();

        // Get current rate
        let rate = app.category_rates.get(category).copied().unwrap_or(0.0);

        // Build sparkline
        let spark = sparkline(&history, row.width.saturating_sub(15) as usize);

        // Build line: "cat   ▃▄▅▆▇█▇▆   123/s"
        let line = Line::from(vec![
            Span::styled(format!("{:5} ", category.name()), Style::default().fg(*color)),
            Span::styled(spark, Style::default().fg(*color)),
            Span::styled(format!(" {:>6}", format_rate(rate)), Style::default().fg(Color::White)),
        ]);

        let paragraph = Paragraph::new(line);
        f.render_widget(paragraph, row);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visualize::VisualizeConfig;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    #[test]
    fn test_draw_syscall_heatmap() {
        let backend = TestBackend::new(80, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = VisualizeApp::new(VisualizeConfig::default());

        terminal
            .draw(|f| {
                draw(f, &app, f.area());
            })
            .unwrap();

        // Verify panel rendered
        let buffer = terminal.backend().buffer();
        let content = buffer_to_string(buffer);
        assert!(content.contains("Syscalls"));
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
