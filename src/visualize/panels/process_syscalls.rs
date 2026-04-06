//! Process syscalls panel - per-process syscall breakdown with sparklines

use crate::visualize::app::{SortColumn, VisualizeApp};
use crate::visualize::theme::{borders, format_rate, percent_color, sparkline};
use ratatui::{
    layout::{Constraint, Rect},
    style::{Color, Modifier, Style},
    text::Span,
    widgets::{Block, Borders, Row, Scrollbar, ScrollbarOrientation, ScrollbarState, Table},
    Frame,
};

/// Draw the process syscalls panel
pub fn draw(f: &mut Frame, app: &VisualizeApp, area: Rect) {
    // Build header with stats
    let header = format!(
        " Process Syscalls {} processes │ {} calls/s ",
        app.processes.len(),
        format_rate(app.syscall_rate)
    );

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(borders::STYLE)
        .border_style(Style::default().fg(borders::PROCESS_SYSCALLS))
        .title(Span::styled(
            header,
            Style::default().fg(borders::PROCESS_SYSCALLS).add_modifier(Modifier::BOLD),
        ));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height < 2 {
        return;
    }

    // Build header row with sort indicators
    let sort_indicator = |col: SortColumn| {
        if col == app.sort_column {
            if app.sort_descending {
                " ▼"
            } else {
                " ▲"
            }
        } else {
            ""
        }
    };

    let header_cells = [
        format!("PID{}", sort_indicator(SortColumn::Pid)),
        format!("Process{}", sort_indicator(SortColumn::Name)),
        format!("CPU%{}", sort_indicator(SortColumn::Cpu)),
        format!("Calls/s{}", sort_indicator(SortColumn::Calls)),
        "Top Syscall".to_string(),
        "Trend".to_string(),
        format!("Errors{}", sort_indicator(SortColumn::Errors)),
    ];

    let header = Row::new(
        header_cells
            .iter()
            .map(|h| Span::styled(h.as_str(), Style::default().add_modifier(Modifier::BOLD))),
    )
    .height(1);

    // Get sorted/filtered processes
    let processes = app.sorted_processes();

    let rows: Vec<Row> = processes
        .iter()
        .enumerate()
        .map(|(i, proc)| {
            let is_selected = i == app.process_selected;
            let cpu_color = percent_color(proc.cpu_percent);
            let trend = sparkline(&proc.history, 8);

            let style = if is_selected {
                Style::default().bg(Color::Rgb(40, 40, 60)).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };

            Row::new(vec![
                Span::styled(format!("{:>6}", proc.pid), style.fg(Color::White)),
                Span::styled(truncate(&proc.name, 15), style.fg(Color::Cyan)),
                Span::styled(format!("{:>5.1}", proc.cpu_percent), style.fg(cpu_color)),
                Span::styled(
                    format!("{:>8}", format_rate(proc.calls_per_sec)),
                    style.fg(Color::Yellow),
                ),
                Span::styled(truncate(&proc.top_syscall, 12), style.fg(Color::White)),
                Span::styled(trend, style.fg(Color::Green)),
                Span::styled(
                    format!("{:>6}", proc.error_count),
                    style.fg(if proc.error_count > 0 { Color::Red } else { Color::DarkGray }),
                ),
            ])
            .style(style)
        })
        .collect();

    let widths = [
        Constraint::Length(7),  // PID
        Constraint::Length(16), // Process
        Constraint::Length(6),  // CPU%
        Constraint::Length(9),  // Calls/s
        Constraint::Length(13), // Top Syscall
        Constraint::Length(9),  // Trend
        Constraint::Length(7),  // Errors
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .column_spacing(1)
        .row_highlight_style(Style::default().bg(Color::Rgb(40, 40, 60)));

    f.render_widget(table, inner);

    // Render scrollbar if needed
    if processes.len() > inner.height.saturating_sub(1) as usize {
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight);
        let mut scrollbar_state =
            ScrollbarState::new(processes.len()).position(app.process_selected);
        f.render_stateful_widget(scrollbar, inner, &mut scrollbar_state);
    }
}

/// Truncate string to max length with ellipsis
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        format!("{:<width$}", s, width = max_len)
    } else {
        format!("{}…", &s[..max_len - 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visualize::app::ProcessSyscallStats;
    use crate::visualize::VisualizeConfig;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    #[test]
    fn test_draw_process_syscalls() {
        let backend = TestBackend::new(100, 15);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = VisualizeApp::new(VisualizeConfig::default());

        // Add test processes
        app.processes = vec![
            ProcessSyscallStats {
                pid: 1234,
                name: "nginx".to_string(),
                cpu_percent: 12.3,
                calls_per_sec: 4521.0,
                error_count: 0,
                top_syscall: "read".to_string(),
                history: vec![0.3, 0.5, 0.6, 0.4, 0.3, 0.5],
            },
            ProcessSyscallStats {
                pid: 5678,
                name: "postgres".to_string(),
                cpu_percent: 8.7,
                calls_per_sec: 3102.0,
                error_count: 12,
                top_syscall: "futex".to_string(),
                history: vec![0.4, 0.5, 0.6, 0.7, 0.6],
            },
        ];

        terminal
            .draw(|f| {
                draw(f, &app, f.area());
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content = buffer_to_string(buffer);
        assert!(content.contains("Process Syscalls"));
        assert!(content.contains("nginx"));
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 10), "short     ");
        assert_eq!(truncate("this is a long string", 10), "this is a…");
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
