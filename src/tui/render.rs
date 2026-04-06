//! TUI rendering - Draw functions for all UI components.
//!
//! All rendering is tested with 100% probador coverage using ratatui's `TestBackend`.

use super::model::{Alert, App, Focus, NodeState, Selection};
use super::widgets::{
    format_duration, latency_color, progress_bar, state_color, success_color, truncate,
    utilization_color,
};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

/// Main draw function - renders the entire UI.
pub fn draw_ui(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Header
            Constraint::Min(10),   // Main content
            Constraint::Length(3), // Task detail
            Constraint::Length(4), // Alerts (if shown)
        ])
        .split(f.area());

    render_header(f, chunks[0], app);
    render_main(f, chunks[1], app);
    render_detail(f, chunks[2], app);

    if app.show_alerts {
        render_alerts(f, chunks[3], app);
    }

    if app.show_help {
        render_help_overlay(f, f.area());
    }
}

/// Render the header bar.
fn render_header(f: &mut Frame, area: Rect, app: &App) {
    let title =
        format!(" REPARTIR JOB FLOW | Nodes: {} | Tasks: {} ", app.nodes.len(), app.queue.total());

    let help_text = "[q]uit [r]efresh [?]help";

    let title_len = title.len();
    let help_len = help_text.len();
    let available = area.width as usize;

    let spans = if title_len + help_len + 2 <= available {
        let padding = available - title_len - help_len;
        vec![
            Span::styled(title, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(" ".repeat(padding)),
            Span::styled(help_text, Style::default().fg(Color::DarkGray)),
        ]
    } else {
        vec![Span::styled(title, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))]
    };

    let para = Paragraph::new(Line::from(spans));
    f.render_widget(para, area);
}

/// Render the main content area (cluster + queue).
fn render_main(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    render_cluster(f, chunks[0], app);
    render_queue_and_completions(f, chunks[1], app);
}

/// Render the cluster panel with node statuses.
fn render_cluster(f: &mut Frame, area: Rect, app: &App) {
    let is_focused = app.focus == Focus::Cluster;
    let border_color = if is_focused { Color::Yellow } else { Color::White };

    let block = Block::default()
        .title(" CLUSTER ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if app.nodes.is_empty() {
        let msg = Paragraph::new("No nodes connected").style(Style::default().fg(Color::DarkGray));
        f.render_widget(msg, inner);
        return;
    }

    // Calculate layout for nodes
    let node_height = 6u16;
    let max_nodes = (inner.height / node_height) as usize;
    let visible_nodes = app.nodes.len().min(max_nodes);

    let constraints: Vec<Constraint> =
        (0..visible_nodes).map(|_| Constraint::Length(node_height)).collect();

    let node_areas =
        Layout::default().direction(Direction::Vertical).constraints(constraints).split(inner);

    for (i, node) in app.nodes.iter().take(visible_nodes).enumerate() {
        let is_selected = matches!(&app.selected, Selection::Node(idx) if *idx == i);
        render_node_status(f, node_areas[i], node, is_selected);
    }
}

/// Render a single node status.
fn render_node_status(f: &mut Frame, area: Rect, node: &super::model::NodeStatus, selected: bool) {
    let state_indicator = match node.state {
        NodeState::Online => "●",
        NodeState::Suspected => "◐",
        NodeState::Offline => "○",
    };

    let backends_str: String =
        node.backends.iter().map(|b| format!("{}", b.backend_type)).collect::<Vec<_>>().join("+");

    let backend_display = if backends_str.is_empty() { "CPU".to_string() } else { backends_str };

    let border_style = if selected {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(state_color(node.state == NodeState::Online))
    };

    let block = Block::default().borders(Borders::ALL).border_style(border_style);

    let inner = block.inner(area);
    f.render_widget(block, area);

    let lines = vec![
        Line::from(vec![
            Span::styled(
                state_indicator,
                Style::default().fg(state_color(node.state == NodeState::Online)),
            ),
            Span::raw(" "),
            Span::styled(truncate(&node.name, 15), Style::default().add_modifier(Modifier::BOLD)),
            Span::raw("  "),
            Span::styled(format!("[{backend_display}]"), Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::raw("  CPU: "),
            Span::styled(
                progress_bar(node.cpu_pct, 8),
                Style::default().fg(utilization_color(node.cpu_pct)),
            ),
            Span::raw(format!(" {:.0}%", node.cpu_pct)),
        ]),
        Line::from(vec![
            Span::raw("  Mem: "),
            Span::styled(
                progress_bar(node.mem_pct, 8),
                Style::default().fg(utilization_color(node.mem_pct)),
            ),
            Span::raw(format!(" {:.0}%", node.mem_pct)),
        ]),
        Line::from(vec![
            Span::raw(format!("  Tasks: {} running  ", node.running_tasks)),
            Span::styled(
                format!("{}ms", node.latency_ms),
                Style::default().fg(latency_color(node.latency_ms)),
            ),
        ]),
    ];

    let para = Paragraph::new(lines);
    f.render_widget(para, inner);
}

/// Render queue and completions panel.
fn render_queue_and_completions(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(area);

    render_queue(f, chunks[0], app);
    render_completions(f, chunks[1], app);
}

/// Render the task queue panel.
fn render_queue(f: &mut Frame, area: Rect, app: &App) {
    let is_focused = app.focus == Focus::Queue;
    let border_color = if is_focused { Color::Yellow } else { Color::White };

    let block = Block::default()
        .title(" TASK QUEUE ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let queue = &app.queue;
    let util_pct = queue.utilization_pct();

    let lines = vec![
        Line::from(vec![Span::raw(format!(
            "Pending: {}    In-Flight: {}",
            queue.pending, queue.in_flight
        ))]),
        Line::from(vec![
            Span::styled(
                progress_bar(util_pct, 20),
                Style::default().fg(utilization_color(util_pct)),
            ),
            Span::raw(format!(" {util_pct:.0}%")),
        ]),
        Line::from(Span::raw("")),
        Line::from(vec![
            Span::raw("Priority: "),
            Span::styled(format!("High:{}", queue.high_priority), Style::default().fg(Color::Red)),
            Span::raw(" "),
            Span::styled(
                format!("Normal:{}", queue.normal_priority),
                Style::default().fg(Color::Yellow),
            ),
            Span::raw(" "),
            Span::styled(format!("Low:{}", queue.low_priority), Style::default().fg(Color::Green)),
        ]),
    ];

    let para = Paragraph::new(lines);
    f.render_widget(para, inner);
}

/// Render the completions panel.
fn render_completions(f: &mut Frame, area: Rect, app: &App) {
    let is_focused = app.focus == Focus::Completions;
    let border_color = if is_focused { Color::Yellow } else { Color::White };

    let block = Block::default()
        .title(" RECENT COMPLETIONS ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if app.completions.is_empty() {
        let msg = Paragraph::new("No completions yet").style(Style::default().fg(Color::DarkGray));
        f.render_widget(msg, inner);
        return;
    }

    let max_items = inner.height as usize;
    let items: Vec<ListItem> = app
        .completions
        .iter()
        .take(max_items)
        .enumerate()
        .map(|(i, c)| {
            let is_selected = matches!(&app.selected, Selection::Completion(idx) if *idx == i);
            let style =
                if is_selected { Style::default().bg(Color::DarkGray) } else { Style::default() };

            let indicator = if c.success { "✓" } else { "✗" };
            let indicator_color = success_color(c.success);

            let duration_or_error = if c.success {
                format_duration(c.duration)
            } else {
                c.error.clone().unwrap_or_else(|| "ERROR".to_string())
            };

            let line = Line::from(vec![
                Span::styled(indicator, Style::default().fg(indicator_color)),
                Span::raw(" "),
                Span::styled(format!("{}", c.backend), Style::default().fg(Color::Cyan)),
                Span::raw(" "),
                Span::raw(truncate(&c.node_name, 10)),
                Span::raw(" "),
                Span::styled(duration_or_error, Style::default().fg(Color::White)),
            ]);

            ListItem::new(line).style(style)
        })
        .collect();

    let list = List::new(items);
    f.render_widget(list, inner);
}

/// Render the detail panel.
fn render_detail(f: &mut Frame, area: Rect, app: &App) {
    let block = Block::default().title(" DETAIL ").borders(Borders::ALL);

    let inner = block.inner(area);
    f.render_widget(block, area);

    let content = match &app.selected {
        Selection::Node(idx) if *idx < app.nodes.len() => {
            let node = &app.nodes[*idx];
            format!(
                "Node: {} | Endpoint: {} | Backends: {} | Tasks: {}",
                node.name,
                node.endpoint,
                node.backends.len(),
                node.running_tasks
            )
        }
        Selection::Completion(idx) if *idx < app.completions.len() => {
            let c = &app.completions[*idx];
            format!(
                "Task: {} | Backend: {} | Node: {} | Duration: {}",
                c.task_id,
                c.backend,
                c.node_name,
                format_duration(c.duration)
            )
        }
        _ => "Select a node or task for details".to_string(),
    };

    let para = Paragraph::new(content).style(Style::default().fg(Color::White));
    f.render_widget(para, inner);
}

/// Render the alerts panel.
fn render_alerts(f: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .title(" ALERTS ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if app.alerts.is_empty() {
        let msg = Paragraph::new("No active alerts").style(Style::default().fg(Color::Green));
        f.render_widget(msg, inner);
        return;
    }

    let max_alerts = inner.height as usize;
    let lines: Vec<Line> = app
        .alerts
        .iter()
        .take(max_alerts)
        .map(|alert| {
            let color = match alert {
                Alert::MemoryPressure { pct, .. } if *pct > 95.0 => Color::Red,
                Alert::MemoryPressure { .. } | Alert::WorkImbalance { .. } => Color::Yellow,
                Alert::NodeSuspected { .. }
                | Alert::TaskTimeout { .. }
                | Alert::BackendError { .. } => Color::Red,
            };

            Line::from(vec![
                Span::styled("⚠ ", Style::default().fg(color)),
                Span::styled(alert.message(), Style::default().fg(color)),
            ])
        })
        .collect();

    let para = Paragraph::new(lines);
    f.render_widget(para, inner);
}

/// Render help overlay.
fn render_help_overlay(f: &mut Frame, area: Rect) {
    let popup_area = centered_rect(60, 50, area);

    let block = Block::default()
        .title(" HELP ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .style(Style::default().bg(Color::Black));

    let inner = block.inner(popup_area);
    f.render_widget(block, popup_area);

    let help_text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  q      ", Style::default().fg(Color::Cyan)),
            Span::raw("Quit"),
        ]),
        Line::from(vec![
            Span::styled("  r      ", Style::default().fg(Color::Cyan)),
            Span::raw("Refresh"),
        ]),
        Line::from(vec![
            Span::styled("  ↑/↓    ", Style::default().fg(Color::Cyan)),
            Span::raw("Navigate"),
        ]),
        Line::from(vec![
            Span::styled("  Tab    ", Style::default().fg(Color::Cyan)),
            Span::raw("Switch focus"),
        ]),
        Line::from(vec![
            Span::styled("  a      ", Style::default().fg(Color::Cyan)),
            Span::raw("Toggle alerts"),
        ]),
        Line::from(vec![
            Span::styled("  ?      ", Style::default().fg(Color::Cyan)),
            Span::raw("Toggle help"),
        ]),
        Line::from(""),
        Line::from(Span::styled("  Press any key to close", Style::default().fg(Color::DarkGray))),
    ];

    let para = Paragraph::new(help_text);
    f.render_widget(para, inner);
}

/// Create a centered rectangle.
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
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    clippy::panic
)]
mod tests {
    use super::*;
    use crate::task::TaskId;
    use crate::tui::model::{BackendStatus, BackendType, CompletionRecord, NodeStatus, TaskQueue};
    use jugar_probar::tui::{expect_frame, TuiFrame};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use std::time::Duration;

    /// Creates a test application with sample node data.
    fn create_test_app() -> App {
        let mut app = App::new();

        // Add nodes
        let mut node1 = NodeStatus::new("linux-rtx4090", "127.0.0.1:9000".parse().unwrap());
        node1.cpu_pct = 58.0;
        node1.mem_pct = 42.0;
        node1.running_tasks = 3;
        node1.latency_ms = 2;
        node1.add_backend(BackendStatus {
            backend_type: BackendType::Cuda,
            device_name: "RTX 4090".to_string(),
            utilization: 82.0,
            memory_pct: 42.0,
            temperature: Some(65.0),
        });
        app.add_node(node1);

        let mut node2 = NodeStatus::new("mac-pro-xeon", "192.168.50.100:9000".parse().unwrap());
        node2.cpu_pct = 76.0;
        node2.mem_pct = 54.0;
        node2.running_tasks = 5;
        node2.latency_ms = 45;
        node2.add_backend(BackendStatus {
            backend_type: BackendType::Metal,
            device_name: "W5700X #0".to_string(),
            utilization: 61.0,
            memory_pct: 54.0,
            temperature: Some(58.0),
        });
        node2.add_backend(BackendStatus {
            backend_type: BackendType::Metal,
            device_name: "W5700X #1".to_string(),
            utilization: 38.0,
            memory_pct: 92.0,
            temperature: Some(62.0),
        });
        app.add_node(node2);

        // Add queue stats
        app.queue = TaskQueue {
            pending: 42,
            in_flight: 8,
            high_priority: 12,
            normal_priority: 28,
            low_priority: 2,
        };

        // Add completions
        app.add_completion(CompletionRecord::success(
            TaskId::new(),
            BackendType::Cuda,
            "linux",
            Duration::from_millis(45),
        ));
        app.add_completion(CompletionRecord::success(
            TaskId::new(),
            BackendType::Metal,
            "mac-pro",
            Duration::from_millis(120),
        ));
        app.add_completion(CompletionRecord::failure(
            TaskId::new(),
            BackendType::Cuda,
            "linux",
            "TIMEOUT",
        ));

        app
    }

    // =========================================================================
    // Header Tests
    // =========================================================================

    #[test]
    fn test_header_renders_title() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame).to_contain_text("REPARTIR JOB FLOW").unwrap();
    }

    #[test]
    fn test_header_shows_counts() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame)
            .to_contain_text("Nodes: 2")
            .unwrap()
            .to_contain_text("Tasks: 50")
            .unwrap();
    }

    // =========================================================================
    // Cluster Panel Tests
    // =========================================================================

    #[test]
    fn test_cluster_renders_node_names() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame)
            .to_contain_text("linux-rtx4090")
            .unwrap()
            .to_contain_text("mac-pro-xeon")
            .unwrap();
    }

    #[test]
    fn test_cluster_renders_backend_types() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame).to_contain_text("CUDA").unwrap().to_contain_text("Metal").unwrap();
    }

    #[test]
    fn test_cluster_renders_cpu_gauge() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame).to_contain_text("CPU:").unwrap();
    }

    #[test]
    fn test_cluster_renders_task_count() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame)
            .to_contain_text("3 running")
            .unwrap()
            .to_contain_text("5 running")
            .unwrap();
    }

    #[test]
    fn test_cluster_empty_shows_message() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = App::new();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame).to_contain_text("No nodes").unwrap();
    }

    // =========================================================================
    // Queue Panel Tests
    // =========================================================================

    #[test]
    fn test_queue_renders_pending() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame).to_contain_text("Pending: 42").unwrap();
    }

    #[test]
    fn test_queue_renders_in_flight() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame).to_contain_text("In-Flight: 8").unwrap();
    }

    #[test]
    fn test_queue_renders_priority() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame)
            .to_contain_text("High:")
            .unwrap()
            .to_contain_text("Normal:")
            .unwrap()
            .to_contain_text("Low:")
            .unwrap();
    }

    // =========================================================================
    // Completions Panel Tests
    // =========================================================================

    #[test]
    fn test_completions_renders_success() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        // Check for success indicator
        expect_frame(&frame).to_contain_text("45ms").unwrap();
    }

    #[test]
    fn test_completions_renders_failure() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame).to_contain_text("TIMEOUT").unwrap();
    }

    #[test]
    fn test_completions_empty_shows_message() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new();
        app.add_node(NodeStatus::new("node-1", "127.0.0.1:9000".parse().unwrap()));

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame).to_contain_text("No completions").unwrap();
    }

    // =========================================================================
    // Alert Panel Tests
    // =========================================================================

    #[test]
    fn test_alerts_renders_when_present() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = create_test_app();

        // Add high memory node to trigger alert
        let mut node = NodeStatus::new("high-mem", "127.0.0.1:9002".parse().unwrap());
        node.mem_pct = 95.0;
        app.add_node(node);
        app.tick(); // Generate alerts

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame).to_contain_text("memory pressure").unwrap();
    }

    #[test]
    fn test_alerts_empty_shows_no_alerts() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = App::new();

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame).to_contain_text("No active alerts").unwrap();
    }

    // =========================================================================
    // Help Overlay Tests
    // =========================================================================

    #[test]
    fn test_help_overlay_renders() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = create_test_app();
        app.show_help = true;

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame)
            .to_contain_text("HELP")
            .unwrap()
            .to_contain_text("Quit")
            .unwrap()
            .to_contain_text("Refresh")
            .unwrap();
    }

    // =========================================================================
    // Layout Tests
    // =========================================================================

    #[test]
    fn test_layout_80x24() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        // Should not panic
        terminal.draw(|f| draw_ui(f, &app)).unwrap();
    }

    #[test]
    fn test_layout_120x40() {
        let backend = TestBackend::new(120, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        // Should not panic
        terminal.draw(|f| draw_ui(f, &app)).unwrap();
    }

    #[test]
    fn test_layout_minimum_40x10() {
        let backend = TestBackend::new(40, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = create_test_app();

        // Should not panic even at tiny size
        terminal.draw(|f| draw_ui(f, &app)).unwrap();
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_handles_nan_utilization() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new();

        let mut node = NodeStatus::new("nan-node", "127.0.0.1:9000".parse().unwrap());
        node.cpu_pct = f64::NAN;
        node.mem_pct = f64::NAN;
        app.add_node(node);

        // Should not panic
        terminal.draw(|f| draw_ui(f, &app)).unwrap();
    }

    #[test]
    fn test_handles_long_node_name() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new();

        let node = NodeStatus::new(
            "very-long-node-name-that-might-overflow",
            "127.0.0.1:9000".parse().unwrap(),
        );
        app.add_node(node);

        // Should not panic
        terminal.draw(|f| draw_ui(f, &app)).unwrap();
    }

    #[test]
    fn test_handles_many_nodes() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new();

        for i in 0..20 {
            let node = NodeStatus::new(
                &format!("node-{}", i),
                format!("127.0.0.1:{}", 9000 + i).parse().unwrap(),
            );
            app.add_node(node);
        }

        // Should not panic
        terminal.draw(|f| draw_ui(f, &app)).unwrap();
    }

    #[test]
    fn test_handles_many_completions() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = create_test_app();

        for _ in 0..200 {
            app.add_completion(CompletionRecord::success(
                TaskId::new(),
                BackendType::Cpu,
                "node",
                Duration::from_millis(10),
            ));
        }

        // Should not panic
        terminal.draw(|f| draw_ui(f, &app)).unwrap();
    }

    #[test]
    fn test_focus_highlighting() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = create_test_app();

        // Default focus is Cluster
        assert_eq!(app.focus, Focus::Cluster);
        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        // Cycle focus
        app.cycle_focus();
        assert_eq!(app.focus, Focus::Queue);
        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        app.cycle_focus();
        assert_eq!(app.focus, Focus::Completions);
        terminal.draw(|f| draw_ui(f, &app)).unwrap();
    }

    #[test]
    fn test_selection_rendering() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = create_test_app();

        app.selected = Selection::Node(0);
        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        app.selected = Selection::Completion(0);
        app.focus = Focus::Completions;
        terminal.draw(|f| draw_ui(f, &app)).unwrap();
    }

    #[test]
    fn test_detail_panel_node_selected() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = create_test_app();

        app.selected = Selection::Node(0);
        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame).to_contain_text("Node: linux-rtx4090").unwrap();
    }

    #[test]
    fn test_detail_panel_completion_selected() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = create_test_app();

        app.selected = Selection::Completion(0);
        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame).to_contain_text("Task:").unwrap();
    }

    #[test]
    fn test_detail_panel_no_selection() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = create_test_app();

        app.selected = Selection::None;
        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame).to_contain_text("Select a node").unwrap();
    }

    #[test]
    fn test_completions_focused() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = create_test_app();

        app.focus = Focus::Completions;
        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        // Should render without panic with focused completions
    }

    #[test]
    fn test_completions_with_selection_highlight() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = create_test_app();

        app.focus = Focus::Completions;
        app.selected = Selection::Completion(1);
        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        // Should render with selection highlight
    }

    #[test]
    fn test_alerts_hidden() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = create_test_app();

        app.show_alerts = false;
        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        // Should render without alerts panel
    }

    #[test]
    fn test_node_offline_rendering() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new();

        let mut node = NodeStatus::new("offline-node", "127.0.0.1:9000".parse().unwrap());
        node.state = NodeState::Offline;
        app.add_node(node);

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        // Should render offline indicator
    }

    #[test]
    fn test_node_suspected_rendering() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new();

        let mut node = NodeStatus::new("suspected-node", "127.0.0.1:9000".parse().unwrap());
        node.state = NodeState::Suspected;
        app.add_node(node);

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        // Should render suspected indicator
    }

    #[test]
    fn test_node_without_backends() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new();

        let node = NodeStatus::new("cpu-only", "127.0.0.1:9000".parse().unwrap());
        app.add_node(node);

        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
        expect_frame(&frame).to_contain_text("CPU").unwrap();
    }

    #[test]
    fn test_queue_focused() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = create_test_app();

        app.focus = Focus::Queue;
        terminal.draw(|f| draw_ui(f, &app)).unwrap();

        // Should render with queue focused
    }
}
