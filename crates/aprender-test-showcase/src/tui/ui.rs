//! TUI rendering with 100% test coverage
//!
//! Probar: Visual feedback makes state visible
//!
//! Uses presentar-terminal-compatible TextBuffer rendering
//! instead of ratatui Frame/Widget pattern.

use super::app::CalculatorApp;
use super::keypad::{Keypad, KeypadWidget, Rect, TextBuffer};

/// Renders the calculator UI into a TextBuffer
pub fn render_to_buffer(app: &CalculatorApp, width: u16, height: u16) -> TextBuffer {
    let area = Rect::new(0, 0, width, height);
    let mut buf = TextBuffer::empty(area);
    let ui = CalculatorUI::new(app);
    ui.render(area, &mut buf);
    buf
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

    /// Render the full calculator UI to a TextBuffer
    pub fn render(&self, area: Rect, buf: &mut TextBuffer) {
        // Title bar
        buf.write_str(area.x + 1, area.y, DEMO_TITLE);

        // Layout: main (left), keypad (center), help (right)
        let margin = 1u16;
        let keypad_w = 22u16;
        let help_w = 22u16;
        let main_w = area.width.saturating_sub(margin * 2 + keypad_w + help_w);

        let content_x = area.x + margin;
        let content_y = area.y + margin;

        // Main area (4 sections stacked)
        let input_h = 3u16;
        let result_h = 3u16;
        let jidoka_h = 5u16;
        let history_h = area
            .height
            .saturating_sub(margin * 2 + input_h + result_h + jidoka_h);

        self.render_input(Rect::new(content_x, content_y, main_w, input_h), buf);
        self.render_result(
            Rect::new(content_x, content_y + input_h, main_w, result_h),
            buf,
        );
        self.render_history(
            Rect::new(content_x, content_y + input_h + result_h, main_w, history_h),
            buf,
        );
        self.render_jidoka_status(
            Rect::new(
                content_x,
                content_y + input_h + result_h + history_h,
                main_w,
                jidoka_h,
            ),
            buf,
        );

        // Keypad
        let keypad_x = content_x + main_w;
        self.render_keypad(
            Rect::new(
                keypad_x,
                content_y,
                keypad_w,
                area.height.saturating_sub(margin * 2),
            ),
            buf,
        );

        // Help sidebar
        let help_x = keypad_x + keypad_w;
        self.render_help_sidebar(
            Rect::new(
                help_x,
                content_y,
                help_w,
                area.height.saturating_sub(margin * 2),
            ),
            buf,
        );
    }

    /// Renders the keypad area
    fn render_keypad(&self, area: Rect, buf: &mut TextBuffer) {
        let widget = KeypadWidget::new(&self.keypad);
        widget.render(area, buf);
    }

    /// Renders the help sidebar
    fn render_help_sidebar(&self, area: Rect, buf: &mut TextBuffer) {
        buf.write_str(area.x + 1, area.y, " Help ");

        let mut y = area.y + 1;
        for (key, desc) in HELP_SHORTCUTS {
            let line = format!("{:>7} {}", key, desc);
            buf.write_str(area.x + 1, y, &line);
            y += 1;
        }

        // Operators
        buf.write_str(area.x + 1, y + 1, HELP_OPERATORS);

        // Probar badge
        buf.write_str(area.x + 1, y + 3, PROBAR_BADGE);
    }

    /// Renders the input area
    fn render_input(&self, area: Rect, buf: &mut TextBuffer) {
        buf.write_str(area.x + 1, area.y, " Expression ");
        let input_text = self.app.input();
        buf.write_str(area.x + 1, area.y + 1, input_text);
    }

    /// Renders the result area
    fn render_result(&self, area: Rect, buf: &mut TextBuffer) {
        buf.write_str(area.x + 1, area.y, " Result ");
        let result_text = self.app.result_display();
        buf.write_str(area.x + 1, area.y + 1, &result_text);
    }

    /// Renders the history area
    fn render_history(&self, area: Rect, buf: &mut TextBuffer) {
        buf.write_str(area.x + 1, area.y, " History (newest first) ");
        let history = self.app.history();

        let mut y = area.y + 1;
        for entry in history.iter_rev().take(10) {
            let line = format!("{} = {}", entry.expression, entry.result);
            buf.write_str(area.x + 1, y, &line);
            y += 1;
        }
    }

    /// Renders the Anomaly status area
    fn render_jidoka_status(&self, area: Rect, buf: &mut TextBuffer) {
        buf.write_str(area.x + 1, area.y, " Anomaly Status ");
        let status_lines = self.app.jidoka_status();

        let mut y = area.y + 1;
        for line in &status_lines {
            buf.write_str(area.x + 1, y, line);
            y += 1;
        }
    }
}

/// Demo title for the calculator
pub const DEMO_TITLE: &str = " Showcase Calculator - 100% Test Coverage Demo ";

/// Help text for the calculator (compact, for sidebar)
pub const HELP_SHORTCUTS: &[(&str, &str)] = &[
    ("Enter", "Evaluate"),
    ("Esc", "Clear"),
    ("Up", "Recall"),
    ("Left/Right", "Move cursor"),
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

    // ===== CalculatorUI tests =====

    #[test]
    fn test_calculator_ui_new() {
        let app = CalculatorApp::new();
        let ui = CalculatorUI::new(&app);
        // Verify it creates without panic
        let _ = format!("{:p}", ui.app);
    }

    #[test]
    fn test_calculator_ui_render() {
        let app = CalculatorApp::new();
        let buf = render_to_buffer(&app, 80, 24);
        // Just verify it renders without panic
        let _ = buf.to_string_content();
    }

    #[test]
    fn test_render_with_input() {
        let mut app = CalculatorApp::new();
        app.set_input("2 + 3");
        let buf = render_to_buffer(&app, 80, 24);

        let content = buf.to_string_content();
        assert!(content.contains("2 + 3"));
    }

    #[test]
    fn test_render_with_result() {
        let mut app = CalculatorApp::new();
        app.set_input("2 + 3");
        app.evaluate();
        let buf = render_to_buffer(&app, 80, 24);

        let content = buf.to_string_content();
        assert!(content.contains('5'));
    }

    #[test]
    fn test_render_with_error() {
        let mut app = CalculatorApp::new();
        app.set_input("1 / 0");
        app.evaluate();
        let buf = render_to_buffer(&app, 80, 24);

        let content = buf.to_string_content();
        assert!(content.contains("Error"));
    }

    #[test]
    fn test_render_with_history() {
        let mut app = CalculatorApp::new();
        app.set_input("1 + 1");
        app.evaluate();
        app.set_input("2 + 2");
        app.evaluate();
        let buf = render_to_buffer(&app, 80, 24);

        let content = buf.to_string_content();
        // Should show history entries
        assert!(content.contains("1 + 1"));
    }

    #[test]
    fn test_render_jidoka_success() {
        let mut app = CalculatorApp::new();
        app.set_input("5 + 5");
        app.evaluate();
        let buf = render_to_buffer(&app, 80, 24);

        let content = buf.to_string_content();
        // Jidoka status should be present (checkmark or status text)
        assert!(
            content.contains('\u{2713}') || content.contains("Anomaly"),
            "Should contain Anomaly status section"
        );
    }

    #[test]
    fn test_render_cursor_position() {
        let mut app = CalculatorApp::new();
        app.set_input("12345");
        app.set_cursor(2); // After "12"
        let buf = render_to_buffer(&app, 80, 24);

        // Just verify it renders without panic
        let _ = buf.to_string_content();
    }

    #[test]
    fn test_render_cursor_at_end() {
        let mut app = CalculatorApp::new();
        app.set_input("abc");
        // cursor is at end by default
        let buf = render_to_buffer(&app, 80, 24);
        let _ = buf.to_string_content();
    }

    #[test]
    fn test_render_empty_input() {
        let app = CalculatorApp::new();
        let buf = render_to_buffer(&app, 80, 24);

        let content = buf.to_string_content();
        assert!(content.contains("Expression"));
    }

    #[test]
    fn test_render_small_terminal() {
        let app = CalculatorApp::new();
        let buf = render_to_buffer(&app, 20, 10);
        let _ = buf.to_string_content();
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
        let buf = render_to_buffer(&app, 80, 24);

        let content = buf.to_string_content();
        assert!(content.contains("Showcase") || content.contains("Calculator"));
    }

    #[test]
    fn test_render_shows_help_panel() {
        let app = CalculatorApp::new();
        let buf = render_to_buffer(&app, 100, 30);

        let content = buf.to_string_content();
        assert!(content.contains("Enter") || content.contains("Help") || content.contains("Esc"));
    }

    #[test]
    fn test_render_shows_probar_badge() {
        let app = CalculatorApp::new();
        let buf = render_to_buffer(&app, 100, 30);

        let content = buf.to_string_content();
        assert!(content.contains("Probar") || content.contains("paiml"));
    }

    #[test]
    fn test_render_help_sidebar_directly() {
        let app = CalculatorApp::new();
        let ui = CalculatorUI::new(&app);
        let area = Rect::new(0, 0, 22, 20);
        let mut buf = TextBuffer::empty(Rect::new(0, 0, 80, 24));

        ui.render_help_sidebar(area, &mut buf);

        let content = buf.to_string_content();
        assert!(content.contains("Help"));
        assert!(content.contains("Enter"));
        assert!(content.contains("Esc"));
    }

    // ===== Widget implementation tests =====

    #[test]
    fn test_widget_render_direct() {
        let app = CalculatorApp::new();
        let ui = CalculatorUI::new(&app);
        let area = Rect::new(0, 0, 80, 24);
        let mut buf = TextBuffer::empty(area);

        ui.render(area, &mut buf);

        // Verify some content was rendered
        let content = buf.to_string_content();
        assert!(content.contains("Calculator"));
    }

    #[test]
    fn test_render_sections_individually() {
        let app = CalculatorApp::new();
        let ui = CalculatorUI::new(&app);

        let mut buf = TextBuffer::empty(Rect::new(0, 0, 80, 24));

        let input_area = Rect::new(0, 0, 40, 3);
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
        let buf = render_to_buffer(&app, 80, 24);

        let content = buf.to_string_content();
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
        let mut buf = TextBuffer::empty(Rect::new(0, 0, 80, 24));

        ui.render_keypad(area, &mut buf);

        let content = buf.to_string_content();
        assert!(content.contains("Keypad"));
        assert!(content.contains("[7]"));
    }

    #[test]
    fn test_render_shows_keypad_in_full_layout() {
        let app = CalculatorApp::new();
        let buf = render_to_buffer(&app, 120, 30);

        let content = buf.to_string_content();
        // Should show keypad
        assert!(content.contains("Keypad"));
        // Should have some button labels
        assert!(content.contains("[7]") || content.contains("[+]") || content.contains("[=]"));
    }

    #[test]
    fn test_keypad_shows_operators() {
        let app = CalculatorApp::new();
        let buf = render_to_buffer(&app, 120, 30);

        let content = buf.to_string_content();
        // Keypad should show operator buttons
        assert!(content.contains("[/]") || content.contains("[*]") || content.contains("[-]"));
    }

    #[test]
    fn test_full_layout_three_columns() {
        let app = CalculatorApp::new();
        let ui = CalculatorUI::new(&app);
        let area = Rect::new(0, 0, 120, 30);
        let mut buf = TextBuffer::empty(area);

        ui.render(area, &mut buf);

        let content = buf.to_string_content();
        // Should have all three areas
        assert!(content.contains("Expression")); // Main area
        assert!(content.contains("Keypad")); // Keypad
        assert!(content.contains("Help")); // Help sidebar
    }
}
