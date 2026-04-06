//! TUI rendering with 100% test coverage
//!
//! Probar: Visual feedback makes state visible

use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Widget},
    Frame,
};

use super::app::CalculatorApp;
use super::keypad::{Keypad, KeypadWidget};

/// Renders the calculator UI to the frame
pub fn render(app: &CalculatorApp, frame: &mut Frame) {
    let area = frame.area();
    let ui = CalculatorUI::new(app);
    frame.render_widget(ui, area);
}

/// Calculator UI widget
pub struct CalculatorUI<'a> {
    app: &'a CalculatorApp,
    keypad: Keypad,
}

impl<'a> CalculatorUI<'a> {
    /// Creates a new calculator UI widget
    #[must_use]
    pub fn new(app: &'a CalculatorApp) -> Self {
        Self {
            app,
            keypad: Keypad::new(),
        }
    }

    /// Creates the main horizontal layout (main + keypad + help sidebar)
    fn create_horizontal_layout(&self, area: Rect) -> Vec<Rect> {
        Layout::default()
            .direction(Direction::Horizontal)
            .margin(1)
            .constraints([
                Constraint::Min(35),    // Main calculator area
                Constraint::Length(22), // Keypad
                Constraint::Length(22), // Help sidebar
            ])
            .split(area)
            .to_vec()
    }

    /// Renders the keypad area
    fn render_keypad(&self, area: Rect, buf: &mut Buffer) {
        let widget = KeypadWidget::new(&self.keypad);
        widget.render(area, buf);
    }

    /// Creates the main layout chunks
    fn create_layout(&self, area: Rect) -> Vec<Rect> {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Input
                Constraint::Length(3), // Result
                Constraint::Min(5),    // History
                Constraint::Length(5), // Anomaly status
            ])
            .split(area)
            .to_vec()
    }

    /// Renders the help sidebar
    fn render_help_sidebar(&self, area: Rect, buf: &mut Buffer) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(10),   // Shortcuts
                Constraint::Length(3), // Operators
                Constraint::Length(2), // Probar badge
            ])
            .split(area);

        // Shortcuts panel
        let shortcuts: Vec<ListItem> = HELP_SHORTCUTS
            .iter()
            .map(|(key, desc)| {
                ListItem::new(Line::from(vec![
                    Span::styled(format!("{:>7}", key), Style::default().fg(Color::Yellow)),
                    Span::raw(" "),
                    Span::styled(*desc, Style::default().fg(Color::Gray)),
                ]))
            })
            .collect();

        let shortcuts_list = List::new(shortcuts).block(
            Block::default()
                .title(" Help ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
        shortcuts_list.render(chunks[0], buf);

        // Operators
        let ops = Paragraph::new(Span::styled(
            HELP_OPERATORS,
            Style::default().fg(Color::Cyan),
        ))
        .block(
            Block::default()
                .borders(Borders::LEFT | Borders::RIGHT)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
        ops.render(chunks[1], buf);

        // Probar badge
        let badge = Paragraph::new(Span::styled(
            PROBAR_BADGE,
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::ITALIC),
        ))
        .block(
            Block::default()
                .borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
        badge.render(chunks[2], buf);
    }

    /// Renders the input area
    fn render_input(&self, area: Rect, buf: &mut Buffer) {
        let input_text = self.app.input();
        let cursor_pos = self.app.cursor();

        // Create spans with cursor highlighting
        let (before, after) = input_text.split_at(cursor_pos.min(input_text.len()));
        let cursor_char = after.chars().next().unwrap_or(' ');
        let after_cursor = if after.len() > 1 {
            &after[cursor_char.len_utf8()..]
        } else {
            ""
        };

        let spans = vec![
            Span::raw(before),
            Span::styled(
                cursor_char.to_string(),
                Style::default().bg(Color::White).fg(Color::Black),
            ),
            Span::raw(after_cursor),
        ];

        let paragraph = Paragraph::new(Line::from(spans)).block(
            Block::default()
                .title(" Expression ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan)),
        );

        paragraph.render(area, buf);
    }

    /// Renders the result area
    fn render_result(&self, area: Rect, buf: &mut Buffer) {
        let result_text = self.app.result_display();

        let style = if result_text.starts_with("Error") {
            Style::default().fg(Color::Red)
        } else {
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD)
        };

        let paragraph = Paragraph::new(Span::styled(result_text, style)).block(
            Block::default()
                .title(" Result ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Yellow)),
        );

        paragraph.render(area, buf);
    }

    /// Renders the history area
    fn render_history(&self, area: Rect, buf: &mut Buffer) {
        let history = self.app.history();

        let items: Vec<ListItem> = history
            .iter_rev()
            .take(10)
            .map(|entry| {
                ListItem::new(Line::from(vec![
                    Span::styled(&entry.expression, Style::default().fg(Color::Gray)),
                    Span::raw(" = "),
                    Span::styled(
                        format!("{}", entry.result),
                        Style::default().fg(Color::Cyan),
                    ),
                ]))
            })
            .collect();

        let list = List::new(items).block(
            Block::default()
                .title(" History (newest first) ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Blue)),
        );

        list.render(area, buf);
    }

    /// Renders the Anomaly status area
    fn render_jidoka_status(&self, area: Rect, buf: &mut Buffer) {
        let status_lines = self.app.jidoka_status();

        let items: Vec<ListItem> = status_lines
            .iter()
            .map(|line| {
                let style = if line.starts_with('✓') {
                    Style::default().fg(Color::Green)
                } else if line.starts_with('✗') {
                    Style::default().fg(Color::Red)
                } else {
                    Style::default().fg(Color::Gray)
                };
                ListItem::new(Span::styled(line.as_str(), style))
            })
            .collect();

        let list = List::new(items).block(
            Block::default()
                .title(" Anomaly Status ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Magenta)),
        );

        list.render(area, buf);
    }
}

impl Widget for CalculatorUI<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Render main border with demo title
        Block::default()
            .title(DEMO_TITLE)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::White))
            .render(area, buf);

        // Split into main area, keypad, and help sidebar
        let h_chunks = self.create_horizontal_layout(area);

        // Main calculator area with keypad and help
        if h_chunks.len() >= 3 {
            let main_area = h_chunks[0];
            let keypad_area = h_chunks[1];
            let help_area = h_chunks[2];

            let chunks = self.create_layout(main_area);

            // Render each section
            if chunks.len() >= 4 {
                self.render_input(chunks[0], buf);
                self.render_result(chunks[1], buf);
                self.render_history(chunks[2], buf);
                self.render_jidoka_status(chunks[3], buf);
            }

            // Render keypad
            self.render_keypad(keypad_area, buf);

            // Render help sidebar
            self.render_help_sidebar(help_area, buf);
        }
    }
}

/// Demo title for the calculator
pub const DEMO_TITLE: &str = " Showcase Calculator - 100% Test Coverage Demo ";

/// Help text for the calculator (compact, for sidebar)
pub const HELP_SHORTCUTS: &[(&str, &str)] = &[
    ("Enter", "Evaluate"),
    ("Esc", "Clear"),
    ("↑", "Recall"),
    ("←/→", "Move cursor"),
    ("Ctrl+C", "Quit"),
    ("Ctrl+L", "Clear all"),
    ("?", "Toggle help"),
];

/// Operators help
pub const HELP_OPERATORS: &str = "Ops: + - * / % ^  ( )";

/// Probar branding shown in demo
pub const PROBAR_BADGE: &str = "Probar - paiml.com/probar";

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn create_test_terminal() -> Terminal<TestBackend> {
        let backend = TestBackend::new(80, 24);
        Terminal::new(backend).unwrap()
    }

    // ===== CalculatorUI tests =====

    #[test]
    fn test_calculator_ui_new() {
        let app = CalculatorApp::new();
        let ui = CalculatorUI::new(&app);
        // Verify it creates without panic
        let _ = format!("{:p}", ui.app);
    }

    #[test]
    fn test_calculator_ui_create_layout() {
        let app = CalculatorApp::new();
        let ui = CalculatorUI::new(&app);
        let area = Rect::new(0, 0, 80, 24);
        let chunks = ui.create_layout(area);
        assert_eq!(chunks.len(), 4);
    }

    #[test]
    fn test_calculator_ui_render() {
        let app = CalculatorApp::new();
        let mut terminal = create_test_terminal();

        terminal
            .draw(|frame| {
                render(&app, frame);
            })
            .unwrap();
    }

    #[test]
    fn test_render_with_input() {
        let mut app = CalculatorApp::new();
        app.set_input("2 + 3");
        let mut terminal = create_test_terminal();

        terminal
            .draw(|frame| {
                render(&app, frame);
            })
            .unwrap();

        // Verify buffer contains input
        let buffer = terminal.backend().buffer();
        let content: String = buffer.content().iter().map(|c| c.symbol()).collect();
        assert!(content.contains("2 + 3"));
    }

    #[test]
    fn test_render_with_result() {
        let mut app = CalculatorApp::new();
        app.set_input("2 + 3");
        app.evaluate();
        let mut terminal = create_test_terminal();

        terminal
            .draw(|frame| {
                render(&app, frame);
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content: String = buffer.content().iter().map(|c| c.symbol()).collect();
        assert!(content.contains('5'));
    }

    #[test]
    fn test_render_with_error() {
        let mut app = CalculatorApp::new();
        app.set_input("1 / 0");
        app.evaluate();
        let mut terminal = create_test_terminal();

        terminal
            .draw(|frame| {
                render(&app, frame);
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content: String = buffer.content().iter().map(|c| c.symbol()).collect();
        assert!(content.contains("Error"));
    }

    #[test]
    fn test_render_with_history() {
        let mut app = CalculatorApp::new();
        app.set_input("1 + 1");
        app.evaluate();
        app.set_input("2 + 2");
        app.evaluate();
        let mut terminal = create_test_terminal();

        terminal
            .draw(|frame| {
                render(&app, frame);
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content: String = buffer.content().iter().map(|c| c.symbol()).collect();
        // Should show history entries
        assert!(content.contains("1 + 1"));
    }

    #[test]
    fn test_render_jidoka_success() {
        let mut app = CalculatorApp::new();
        app.set_input("5 + 5");
        app.evaluate();
        let mut terminal = create_test_terminal();

        terminal
            .draw(|frame| {
                render(&app, frame);
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content: String = buffer.content().iter().map(|c| c.symbol()).collect();
        assert!(content.contains("✓"));
    }

    #[test]
    fn test_render_cursor_position() {
        let mut app = CalculatorApp::new();
        app.set_input("12345");
        app.set_cursor(2); // After "12"
        let mut terminal = create_test_terminal();

        terminal
            .draw(|frame| {
                render(&app, frame);
            })
            .unwrap();

        // Just verify it renders without panic
        let _ = terminal.backend().buffer();
    }

    #[test]
    fn test_render_cursor_at_end() {
        let mut app = CalculatorApp::new();
        app.set_input("abc");
        // cursor is at end by default
        let mut terminal = create_test_terminal();

        terminal
            .draw(|frame| {
                render(&app, frame);
            })
            .unwrap();
    }

    #[test]
    fn test_render_empty_input() {
        let app = CalculatorApp::new();
        let mut terminal = create_test_terminal();

        terminal
            .draw(|frame| {
                render(&app, frame);
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content: String = buffer.content().iter().map(|c| c.symbol()).collect();
        assert!(content.contains("Expression"));
    }

    #[test]
    fn test_render_small_terminal() {
        let app = CalculatorApp::new();
        let backend = TestBackend::new(20, 10);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal
            .draw(|frame| {
                render(&app, frame);
            })
            .unwrap();
    }

    // ===== Demo/Help panel tests (PMAT-CALC-006) =====

    #[test]
    fn test_demo_title_constant() {
        assert!(DEMO_TITLE.contains("Showcase Calculator"));
        assert!(DEMO_TITLE.contains("100% Test Coverage"));
        assert!(DEMO_TITLE.contains("Demo"));
    }

    #[test]
    fn test_help_shortcuts_contains_essential_keys() {
        let keys: Vec<&str> = HELP_SHORTCUTS.iter().map(|(k, _)| *k).collect();
        assert!(keys.contains(&"Enter"));
        assert!(keys.contains(&"Esc"));
        assert!(keys.contains(&"Ctrl+C"));
    }

    #[test]
    fn test_help_shortcuts_has_descriptions() {
        for (key, desc) in HELP_SHORTCUTS {
            assert!(!key.is_empty(), "Key should not be empty");
            assert!(!desc.is_empty(), "Description should not be empty");
        }
    }

    #[test]
    fn test_help_shortcuts_count() {
        assert!(
            HELP_SHORTCUTS.len() >= 5,
            "Should have at least 5 shortcuts"
        );
    }

    #[test]
    fn test_help_operators_contains_all_ops() {
        assert!(HELP_OPERATORS.contains('+'));
        assert!(HELP_OPERATORS.contains('-'));
        assert!(HELP_OPERATORS.contains('*'));
        assert!(HELP_OPERATORS.contains('/'));
        assert!(HELP_OPERATORS.contains('%'));
        assert!(HELP_OPERATORS.contains('^'));
        assert!(HELP_OPERATORS.contains('('));
        assert!(HELP_OPERATORS.contains(')'));
    }

    #[test]
    fn test_probar_badge_contains_branding() {
        assert!(PROBAR_BADGE.contains("Probar"));
        assert!(PROBAR_BADGE.contains("paiml"));
    }

    #[test]
    fn test_render_shows_demo_title() {
        let app = CalculatorApp::new();
        let mut terminal = create_test_terminal();

        terminal
            .draw(|frame| {
                render(&app, frame);
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content: String = buffer.content().iter().map(|c| c.symbol()).collect();
        assert!(content.contains("Showcase") || content.contains("Calculator"));
    }

    #[test]
    fn test_render_shows_help_panel() {
        let app = CalculatorApp::new();
        let backend = TestBackend::new(100, 30); // Wider terminal for help panel
        let mut terminal = Terminal::new(backend).unwrap();

        terminal
            .draw(|frame| {
                render(&app, frame);
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content: String = buffer.content().iter().map(|c| c.symbol()).collect();
        // Should contain help-related content
        assert!(content.contains("Enter") || content.contains("Help") || content.contains("Esc"));
    }

    #[test]
    fn test_render_shows_probar_badge() {
        let app = CalculatorApp::new();
        let backend = TestBackend::new(100, 30);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal
            .draw(|frame| {
                render(&app, frame);
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content: String = buffer.content().iter().map(|c| c.symbol()).collect();
        // Should show Probar reference
        assert!(content.contains("Probar") || content.contains("paiml"));
    }

    #[test]
    fn test_render_help_sidebar_directly() {
        let app = CalculatorApp::new();
        let ui = CalculatorUI::new(&app);
        let area = Rect::new(0, 0, 22, 20);
        let mut buf = Buffer::empty(Rect::new(0, 0, 80, 24));

        ui.render_help_sidebar(area, &mut buf);

        let content: String = buf.content().iter().map(|c| c.symbol()).collect();
        assert!(content.contains("Help"));
        assert!(content.contains("Enter"));
        assert!(content.contains("Esc"));
    }

    #[test]
    fn test_create_horizontal_layout() {
        let app = CalculatorApp::new();
        let ui = CalculatorUI::new(&app);
        let area = Rect::new(0, 0, 100, 30);
        let chunks = ui.create_horizontal_layout(area);
        assert_eq!(chunks.len(), 3);
        // Keypad should be 22 wide
        assert_eq!(chunks[1].width, 22);
        // Help sidebar should be 22 wide
        assert_eq!(chunks[2].width, 22);
    }

    // ===== Widget implementation tests =====

    #[test]
    fn test_widget_render_direct() {
        let app = CalculatorApp::new();
        let ui = CalculatorUI::new(&app);
        let area = Rect::new(0, 0, 80, 24);
        let mut buf = Buffer::empty(area);

        ui.render(area, &mut buf);

        // Verify some content was rendered
        let content: String = buf.content().iter().map(|c| c.symbol()).collect();
        assert!(content.contains("Calculator"));
    }

    #[test]
    fn test_render_sections_individually() {
        let app = CalculatorApp::new();
        let ui = CalculatorUI::new(&app);

        let input_area = Rect::new(0, 0, 40, 3);
        let mut buf = Buffer::empty(Rect::new(0, 0, 80, 24));
        ui.render_input(input_area, &mut buf);

        let result_area = Rect::new(0, 3, 40, 3);
        ui.render_result(result_area, &mut buf);

        let history_area = Rect::new(0, 6, 40, 10);
        ui.render_history(history_area, &mut buf);

        let jidoka_area = Rect::new(0, 16, 40, 5);
        ui.render_jidoka_status(jidoka_area, &mut buf);
    }

    #[test]
    fn test_render_history_many_entries() {
        let mut app = CalculatorApp::new();
        for i in 1..=20 {
            app.set_input(&format!("{i} + {i}"));
            app.evaluate();
        }
        let mut terminal = create_test_terminal();

        terminal
            .draw(|frame| {
                render(&app, frame);
            })
            .unwrap();

        // Should only show last 10
        let buffer = terminal.backend().buffer();
        let content: String = buffer.content().iter().map(|c| c.symbol()).collect();
        assert!(content.contains("20 + 20")); // Most recent
    }

    // ===== Keypad integration tests (PMAT-CALC-007) =====

    #[test]
    fn test_calculator_ui_has_keypad() {
        let app = CalculatorApp::new();
        let ui = CalculatorUI::new(&app);
        // Verify keypad is initialized
        assert_eq!(ui.keypad.button_count(), 20);
    }

    #[test]
    fn test_render_keypad_directly() {
        let app = CalculatorApp::new();
        let ui = CalculatorUI::new(&app);
        let area = Rect::new(0, 0, 22, 12);
        let mut buf = Buffer::empty(Rect::new(0, 0, 80, 24));

        ui.render_keypad(area, &mut buf);

        let content: String = buf.content().iter().map(|c| c.symbol()).collect();
        assert!(content.contains("Keypad"));
        assert!(content.contains("[7]"));
    }

    #[test]
    fn test_render_shows_keypad_in_full_layout() {
        let app = CalculatorApp::new();
        let backend = TestBackend::new(120, 30); // Wide enough for all panels
        let mut terminal = Terminal::new(backend).unwrap();

        terminal
            .draw(|frame| {
                render(&app, frame);
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content: String = buffer.content().iter().map(|c| c.symbol()).collect();
        // Should show keypad
        assert!(content.contains("Keypad"));
        // Should have some button labels
        assert!(content.contains("[7]") || content.contains("[+]") || content.contains("[=]"));
    }

    #[test]
    fn test_keypad_shows_operators() {
        let app = CalculatorApp::new();
        let backend = TestBackend::new(120, 30);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal
            .draw(|frame| {
                render(&app, frame);
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content: String = buf_to_string(buffer);
        // Keypad should show operator buttons
        assert!(content.contains("[/]") || content.contains("[*]") || content.contains("[-]"));
    }

    #[test]
    fn test_full_layout_three_columns() {
        let app = CalculatorApp::new();
        let ui = CalculatorUI::new(&app);
        let area = Rect::new(0, 0, 120, 30);
        let mut buf = Buffer::empty(area);

        ui.render(area, &mut buf);

        let content: String = buf.content().iter().map(|c| c.symbol()).collect();
        // Should have all three areas
        assert!(content.contains("Expression")); // Main area
        assert!(content.contains("Keypad")); // Keypad
        assert!(content.contains("Help")); // Help sidebar
    }

    fn buf_to_string(buffer: &Buffer) -> String {
        buffer.content().iter().map(|c| c.symbol()).collect()
    }
}
