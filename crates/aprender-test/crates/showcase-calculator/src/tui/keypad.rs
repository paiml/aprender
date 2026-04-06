//! Numerical Keypad for TUI Calculator
//!
//! Probar: Visual feedback - Visual buttons make calculator state obvious
//!
//! This module provides an interactive numerical keypad that can be:
//! - Clicked with mouse (TUI mouse events)
//! - Highlighted when corresponding key is pressed
//! - Used for visual demonstration of calculator operation

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::Span,
    widgets::{Block, Borders, Widget},
};

/// A single keypad button
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeypadButton {
    /// The character/symbol on the button
    pub label: char,
    /// Whether the button is currently pressed/highlighted
    pub pressed: bool,
    /// The action this button performs
    pub action: ButtonAction,
}

/// Actions that keypad buttons can perform
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ButtonAction {
    /// Insert a digit (0-9)
    Digit(u8),
    /// Insert a decimal point
    Decimal,
    /// Insert an operator
    Operator(char),
    /// Evaluate the expression
    Equals,
    /// Clear the input
    Clear,
    /// Open parenthesis
    OpenParen,
    /// Close parenthesis
    CloseParen,
}

impl KeypadButton {
    /// Creates a new digit button
    #[must_use]
    pub fn digit(d: u8) -> Self {
        Self {
            label: char::from_digit(d as u32, 10).unwrap_or('?'),
            pressed: false,
            action: ButtonAction::Digit(d),
        }
    }

    /// Creates a new operator button
    #[must_use]
    pub fn operator(op: char) -> Self {
        Self {
            label: op,
            pressed: false,
            action: ButtonAction::Operator(op),
        }
    }

    /// Creates the decimal point button
    #[must_use]
    pub fn decimal() -> Self {
        Self {
            label: '.',
            pressed: false,
            action: ButtonAction::Decimal,
        }
    }

    /// Creates the equals button
    #[must_use]
    pub fn equals() -> Self {
        Self {
            label: '=',
            pressed: false,
            action: ButtonAction::Equals,
        }
    }

    /// Creates the clear button
    #[must_use]
    pub fn clear() -> Self {
        Self {
            label: 'C',
            pressed: false,
            action: ButtonAction::Clear,
        }
    }

    /// Creates the open parenthesis button
    #[must_use]
    pub fn open_paren() -> Self {
        Self {
            label: '(',
            pressed: false,
            action: ButtonAction::OpenParen,
        }
    }

    /// Creates the close parenthesis button
    #[must_use]
    pub fn close_paren() -> Self {
        Self {
            label: ')',
            pressed: false,
            action: ButtonAction::CloseParen,
        }
    }

    /// Sets the pressed state
    pub fn set_pressed(&mut self, pressed: bool) {
        self.pressed = pressed;
    }

    /// Returns the character to insert for this button
    #[must_use]
    pub fn to_char(&self) -> Option<char> {
        match self.action {
            ButtonAction::Digit(d) => char::from_digit(d as u32, 10),
            ButtonAction::Decimal => Some('.'),
            ButtonAction::Operator(op) => Some(op),
            ButtonAction::OpenParen => Some('('),
            ButtonAction::CloseParen => Some(')'),
            ButtonAction::Equals | ButtonAction::Clear => None,
        }
    }
}

/// The keypad layout - a 5x4 grid of buttons
/// ```text
/// [ 7 ] [ 8 ] [ 9 ] [ / ]
/// [ 4 ] [ 5 ] [ 6 ] [ * ]
/// [ 1 ] [ 2 ] [ 3 ] [ - ]
/// [ 0 ] [ . ] [ = ] [ + ]
/// [ C ] [ ( ] [ ) ] [ ^ ]
/// ```
#[derive(Debug, Clone)]
pub struct Keypad {
    /// Buttons in row-major order (5 rows x 4 cols)
    buttons: Vec<KeypadButton>,
    /// Number of columns
    cols: usize,
    /// Number of rows
    rows: usize,
}

impl Default for Keypad {
    fn default() -> Self {
        Self::new()
    }
}

impl Keypad {
    /// Creates a new standard calculator keypad
    #[must_use]
    pub fn new() -> Self {
        let buttons = vec![
            // Row 1: 7 8 9 /
            KeypadButton::digit(7),
            KeypadButton::digit(8),
            KeypadButton::digit(9),
            KeypadButton::operator('/'),
            // Row 2: 4 5 6 *
            KeypadButton::digit(4),
            KeypadButton::digit(5),
            KeypadButton::digit(6),
            KeypadButton::operator('*'),
            // Row 3: 1 2 3 -
            KeypadButton::digit(1),
            KeypadButton::digit(2),
            KeypadButton::digit(3),
            KeypadButton::operator('-'),
            // Row 4: 0 . = +
            KeypadButton::digit(0),
            KeypadButton::decimal(),
            KeypadButton::equals(),
            KeypadButton::operator('+'),
            // Row 5: C ( ) ^
            KeypadButton::clear(),
            KeypadButton::open_paren(),
            KeypadButton::close_paren(),
            KeypadButton::operator('^'),
        ];

        Self {
            buttons,
            cols: 4,
            rows: 5,
        }
    }

    /// Returns the number of buttons
    #[must_use]
    pub fn button_count(&self) -> usize {
        self.buttons.len()
    }

    /// Returns the grid dimensions (rows, cols)
    #[must_use]
    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Gets a button by index
    #[must_use]
    pub fn get_button(&self, index: usize) -> Option<&KeypadButton> {
        self.buttons.get(index)
    }

    /// Gets a mutable button by index
    pub fn get_button_mut(&mut self, index: usize) -> Option<&mut KeypadButton> {
        self.buttons.get_mut(index)
    }

    /// Gets a button by row and column
    #[must_use]
    pub fn get_button_at(&self, row: usize, col: usize) -> Option<&KeypadButton> {
        if row < self.rows && col < self.cols {
            self.buttons.get(row * self.cols + col)
        } else {
            None
        }
    }

    /// Finds a button by its label character
    #[must_use]
    pub fn find_button_by_label(&self, label: char) -> Option<usize> {
        self.buttons.iter().position(|b| b.label == label)
    }

    /// Finds a button by the character it would insert
    #[must_use]
    pub fn find_button_by_char(&self, ch: char) -> Option<usize> {
        self.buttons.iter().position(|b| b.to_char() == Some(ch))
    }

    /// Sets a button as pressed by index
    pub fn press_button(&mut self, index: usize) {
        if let Some(btn) = self.buttons.get_mut(index) {
            btn.set_pressed(true);
        }
    }

    /// Releases all buttons
    pub fn release_all(&mut self) {
        for btn in &mut self.buttons {
            btn.set_pressed(false);
        }
    }

    /// Highlights the button corresponding to a character
    pub fn highlight_char(&mut self, ch: char) {
        self.release_all();
        if let Some(idx) = self.find_button_by_char(ch) {
            self.press_button(idx);
        }
    }

    /// Returns an iterator over all buttons
    pub fn buttons(&self) -> impl Iterator<Item = &KeypadButton> {
        self.buttons.iter()
    }

    /// Returns an iterator over buttons with their (row, col) positions
    pub fn buttons_with_positions(&self) -> impl Iterator<Item = ((usize, usize), &KeypadButton)> {
        self.buttons.iter().enumerate().map(move |(i, btn)| {
            let row = i / self.cols;
            let col = i % self.cols;
            ((row, col), btn)
        })
    }

    /// Converts a click position to button index
    #[must_use]
    pub fn hit_test(&self, area: Rect, x: u16, y: u16) -> Option<usize> {
        if x < area.x || y < area.y || x >= area.x + area.width || y >= area.y + area.height {
            return None;
        }

        let rel_x = x - area.x;
        let rel_y = y - area.y;

        // Account for border (1 char on each side)
        if rel_x == 0 || rel_y == 0 || rel_x >= area.width - 1 || rel_y >= area.height - 1 {
            return None;
        }

        let inner_x = rel_x - 1;
        let inner_y = rel_y - 1;

        let btn_width = (area.width - 2) / self.cols as u16;
        let btn_height = (area.height - 2) / self.rows as u16;

        if btn_width == 0 || btn_height == 0 {
            return None;
        }

        let col = (inner_x / btn_width) as usize;
        let row = (inner_y / btn_height) as usize;

        if row < self.rows && col < self.cols {
            Some(row * self.cols + col)
        } else {
            None
        }
    }
}

/// Keypad widget for rendering
#[derive(Debug)]
pub struct KeypadWidget<'a> {
    keypad: &'a Keypad,
}

impl<'a> KeypadWidget<'a> {
    /// Creates a new keypad widget
    #[must_use]
    pub fn new(keypad: &'a Keypad) -> Self {
        Self { keypad }
    }
}

impl Widget for KeypadWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Draw border
        Block::default()
            .title(" Keypad ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .render(area, buf);

        // Calculate inner area
        let inner = Rect {
            x: area.x + 1,
            y: area.y + 1,
            width: area.width.saturating_sub(2),
            height: area.height.saturating_sub(2),
        };

        if inner.width < 4 || inner.height < 5 {
            return; // Too small to render
        }

        let btn_width = inner.width / self.keypad.cols as u16;
        let btn_height = inner.height / self.keypad.rows as u16;

        for ((row, col), btn) in self.keypad.buttons_with_positions() {
            let x = inner.x + (col as u16 * btn_width);
            let y = inner.y + (row as u16 * btn_height);

            // Button style based on pressed state
            let style = if btn.pressed {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                match btn.action {
                    ButtonAction::Digit(_) => Style::default().fg(Color::White),
                    ButtonAction::Operator(_) => Style::default().fg(Color::Yellow),
                    ButtonAction::Equals => Style::default().fg(Color::Green),
                    ButtonAction::Clear => Style::default().fg(Color::Red),
                    _ => Style::default().fg(Color::Cyan),
                }
            };

            // Render button label centered
            if btn_width >= 3 {
                let label = format!("[{}]", btn.label);
                let label_x = x + (btn_width.saturating_sub(label.len() as u16)) / 2;
                let label_y = y + btn_height / 2;

                if label_y < inner.y + inner.height && label_x < inner.x + inner.width {
                    buf.set_span(label_x, label_y, &Span::styled(label, style), btn_width);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== KeypadButton tests =====

    #[test]
    fn test_digit_button_creation() {
        for d in 0..=9 {
            let btn = KeypadButton::digit(d);
            assert_eq!(btn.label, char::from_digit(d as u32, 10).unwrap());
            assert!(!btn.pressed);
            assert_eq!(btn.action, ButtonAction::Digit(d));
        }
    }

    #[test]
    fn test_operator_button_creation() {
        for op in ['+', '-', '*', '/', '^'] {
            let btn = KeypadButton::operator(op);
            assert_eq!(btn.label, op);
            assert!(!btn.pressed);
            assert_eq!(btn.action, ButtonAction::Operator(op));
        }
    }

    #[test]
    fn test_decimal_button() {
        let btn = KeypadButton::decimal();
        assert_eq!(btn.label, '.');
        assert_eq!(btn.action, ButtonAction::Decimal);
    }

    #[test]
    fn test_equals_button() {
        let btn = KeypadButton::equals();
        assert_eq!(btn.label, '=');
        assert_eq!(btn.action, ButtonAction::Equals);
    }

    #[test]
    fn test_clear_button() {
        let btn = KeypadButton::clear();
        assert_eq!(btn.label, 'C');
        assert_eq!(btn.action, ButtonAction::Clear);
    }

    #[test]
    fn test_paren_buttons() {
        let open = KeypadButton::open_paren();
        assert_eq!(open.label, '(');
        assert_eq!(open.action, ButtonAction::OpenParen);

        let close = KeypadButton::close_paren();
        assert_eq!(close.label, ')');
        assert_eq!(close.action, ButtonAction::CloseParen);
    }

    #[test]
    fn test_button_pressed_state() {
        let mut btn = KeypadButton::digit(5);
        assert!(!btn.pressed);
        btn.set_pressed(true);
        assert!(btn.pressed);
        btn.set_pressed(false);
        assert!(!btn.pressed);
    }

    #[test]
    fn test_button_to_char() {
        assert_eq!(KeypadButton::digit(5).to_char(), Some('5'));
        assert_eq!(KeypadButton::decimal().to_char(), Some('.'));
        assert_eq!(KeypadButton::operator('+').to_char(), Some('+'));
        assert_eq!(KeypadButton::open_paren().to_char(), Some('('));
        assert_eq!(KeypadButton::close_paren().to_char(), Some(')'));
        assert_eq!(KeypadButton::equals().to_char(), None);
        assert_eq!(KeypadButton::clear().to_char(), None);
    }

    #[test]
    fn test_button_clone() {
        let btn = KeypadButton::digit(7);
        let cloned = btn.clone();
        assert_eq!(btn, cloned);
    }

    #[test]
    fn test_button_debug() {
        let btn = KeypadButton::digit(9);
        let debug = format!("{:?}", btn);
        assert!(debug.contains("KeypadButton"));
    }

    // ===== ButtonAction tests =====

    #[test]
    fn test_button_action_copy() {
        let action = ButtonAction::Digit(5);
        let copied = action;
        assert_eq!(action, copied);
    }

    #[test]
    fn test_button_action_debug() {
        let action = ButtonAction::Operator('+');
        let debug = format!("{:?}", action);
        assert!(debug.contains("Operator"));
    }

    // ===== Keypad tests =====

    #[test]
    fn test_keypad_new() {
        let keypad = Keypad::new();
        assert_eq!(keypad.button_count(), 20); // 5 rows x 4 cols
    }

    #[test]
    fn test_keypad_default() {
        let keypad = Keypad::default();
        assert_eq!(keypad.button_count(), 20);
    }

    #[test]
    fn test_keypad_dimensions() {
        let keypad = Keypad::new();
        assert_eq!(keypad.dimensions(), (5, 4));
    }

    #[test]
    fn test_keypad_get_button() {
        let keypad = Keypad::new();
        // First button should be 7
        let btn = keypad.get_button(0).unwrap();
        assert_eq!(btn.label, '7');
    }

    #[test]
    fn test_keypad_get_button_out_of_bounds() {
        let keypad = Keypad::new();
        assert!(keypad.get_button(100).is_none());
    }

    #[test]
    fn test_keypad_get_button_at() {
        let keypad = Keypad::new();
        // Row 0, Col 0 = 7
        assert_eq!(keypad.get_button_at(0, 0).unwrap().label, '7');
        // Row 0, Col 3 = /
        assert_eq!(keypad.get_button_at(0, 3).unwrap().label, '/');
        // Row 4, Col 0 = C
        assert_eq!(keypad.get_button_at(4, 0).unwrap().label, 'C');
    }

    #[test]
    fn test_keypad_get_button_at_out_of_bounds() {
        let keypad = Keypad::new();
        assert!(keypad.get_button_at(10, 10).is_none());
    }

    #[test]
    fn test_keypad_find_by_label() {
        let keypad = Keypad::new();
        assert_eq!(keypad.find_button_by_label('7'), Some(0));
        assert_eq!(keypad.find_button_by_label('0'), Some(12));
        assert_eq!(keypad.find_button_by_label('='), Some(14));
        assert_eq!(keypad.find_button_by_label('X'), None);
    }

    #[test]
    fn test_keypad_find_by_char() {
        let keypad = Keypad::new();
        assert_eq!(keypad.find_button_by_char('5'), Some(5));
        assert_eq!(keypad.find_button_by_char('+'), Some(15));
        assert_eq!(keypad.find_button_by_char('.'), Some(13));
    }

    #[test]
    fn test_keypad_press_button() {
        let mut keypad = Keypad::new();
        keypad.press_button(0);
        assert!(keypad.get_button(0).unwrap().pressed);
        assert!(!keypad.get_button(1).unwrap().pressed);
    }

    #[test]
    fn test_keypad_release_all() {
        let mut keypad = Keypad::new();
        keypad.press_button(0);
        keypad.press_button(5);
        keypad.release_all();
        for btn in keypad.buttons() {
            assert!(!btn.pressed);
        }
    }

    #[test]
    fn test_keypad_highlight_char() {
        let mut keypad = Keypad::new();
        keypad.highlight_char('5');
        assert!(keypad.get_button(5).unwrap().pressed);
        // Other buttons should not be pressed
        assert!(!keypad.get_button(0).unwrap().pressed);
    }

    #[test]
    fn test_keypad_buttons_iterator() {
        let keypad = Keypad::new();
        let count = keypad.buttons().count();
        assert_eq!(count, 20);
    }

    #[test]
    fn test_keypad_buttons_with_positions() {
        let keypad = Keypad::new();
        let positions: Vec<_> = keypad.buttons_with_positions().collect();
        assert_eq!(positions.len(), 20);
        assert_eq!(positions[0].0, (0, 0)); // First button at row 0, col 0
        assert_eq!(positions[19].0, (4, 3)); // Last button at row 4, col 3
    }

    #[test]
    fn test_keypad_hit_test_inside() {
        let keypad = Keypad::new();
        let area = Rect::new(0, 0, 22, 12); // Big enough for 4x5 grid

        // Click in center should hit a button
        let result = keypad.hit_test(area, 10, 5);
        assert!(result.is_some());
    }

    #[test]
    fn test_keypad_hit_test_outside() {
        let keypad = Keypad::new();
        let area = Rect::new(10, 10, 22, 12);

        // Click outside area
        assert!(keypad.hit_test(area, 0, 0).is_none());
        assert!(keypad.hit_test(area, 100, 100).is_none());
    }

    #[test]
    fn test_keypad_hit_test_border() {
        let keypad = Keypad::new();
        let area = Rect::new(0, 0, 22, 12);

        // Click on border (first row/col)
        assert!(keypad.hit_test(area, 0, 0).is_none());
    }

    #[test]
    fn test_keypad_get_button_mut() {
        let mut keypad = Keypad::new();
        if let Some(btn) = keypad.get_button_mut(0) {
            btn.set_pressed(true);
        }
        assert!(keypad.get_button(0).unwrap().pressed);
    }

    #[test]
    fn test_keypad_clone() {
        let keypad = Keypad::new();
        let cloned = keypad.clone();
        assert_eq!(keypad.button_count(), cloned.button_count());
    }

    #[test]
    fn test_keypad_debug() {
        let keypad = Keypad::new();
        let debug = format!("{:?}", keypad);
        assert!(debug.contains("Keypad"));
    }

    // ===== Keypad layout verification =====

    #[test]
    fn test_keypad_row_1() {
        let keypad = Keypad::new();
        assert_eq!(keypad.get_button_at(0, 0).unwrap().label, '7');
        assert_eq!(keypad.get_button_at(0, 1).unwrap().label, '8');
        assert_eq!(keypad.get_button_at(0, 2).unwrap().label, '9');
        assert_eq!(keypad.get_button_at(0, 3).unwrap().label, '/');
    }

    #[test]
    fn test_keypad_row_2() {
        let keypad = Keypad::new();
        assert_eq!(keypad.get_button_at(1, 0).unwrap().label, '4');
        assert_eq!(keypad.get_button_at(1, 1).unwrap().label, '5');
        assert_eq!(keypad.get_button_at(1, 2).unwrap().label, '6');
        assert_eq!(keypad.get_button_at(1, 3).unwrap().label, '*');
    }

    #[test]
    fn test_keypad_row_3() {
        let keypad = Keypad::new();
        assert_eq!(keypad.get_button_at(2, 0).unwrap().label, '1');
        assert_eq!(keypad.get_button_at(2, 1).unwrap().label, '2');
        assert_eq!(keypad.get_button_at(2, 2).unwrap().label, '3');
        assert_eq!(keypad.get_button_at(2, 3).unwrap().label, '-');
    }

    #[test]
    fn test_keypad_row_4() {
        let keypad = Keypad::new();
        assert_eq!(keypad.get_button_at(3, 0).unwrap().label, '0');
        assert_eq!(keypad.get_button_at(3, 1).unwrap().label, '.');
        assert_eq!(keypad.get_button_at(3, 2).unwrap().label, '=');
        assert_eq!(keypad.get_button_at(3, 3).unwrap().label, '+');
    }

    #[test]
    fn test_keypad_row_5() {
        let keypad = Keypad::new();
        assert_eq!(keypad.get_button_at(4, 0).unwrap().label, 'C');
        assert_eq!(keypad.get_button_at(4, 1).unwrap().label, '(');
        assert_eq!(keypad.get_button_at(4, 2).unwrap().label, ')');
        assert_eq!(keypad.get_button_at(4, 3).unwrap().label, '^');
    }

    // ===== KeypadWidget tests =====

    #[test]
    fn test_keypad_widget_new() {
        let keypad = Keypad::new();
        let widget = KeypadWidget::new(&keypad);
        // Just verify it creates without panic
        let _ = widget;
    }

    #[test]
    fn test_keypad_widget_render() {
        let keypad = Keypad::new();
        let widget = KeypadWidget::new(&keypad);
        let area = Rect::new(0, 0, 22, 12);
        let mut buf = Buffer::empty(area);

        widget.render(area, &mut buf);

        let content: String = buf.content().iter().map(|c| c.symbol()).collect();
        assert!(content.contains("Keypad"));
        assert!(content.contains("[7]"));
        assert!(content.contains("[+]"));
    }

    #[test]
    fn test_keypad_widget_render_small() {
        let keypad = Keypad::new();
        let widget = KeypadWidget::new(&keypad);
        let area = Rect::new(0, 0, 5, 5); // Too small
        let mut buf = Buffer::empty(area);

        // Should not panic, just render border
        widget.render(area, &mut buf);
    }

    #[test]
    fn test_keypad_widget_render_pressed() {
        let mut keypad = Keypad::new();
        keypad.press_button(0); // Press '7'
        let widget = KeypadWidget::new(&keypad);
        let area = Rect::new(0, 0, 22, 12);
        let mut buf = Buffer::empty(area);

        widget.render(area, &mut buf);
        // Pressed button should still render
        let content: String = buf.content().iter().map(|c| c.symbol()).collect();
        assert!(content.contains("[7]"));
    }

    // ===== Property-based tests =====

    #[test]
    fn prop_all_digits_have_buttons() {
        let keypad = Keypad::new();
        for d in 0..=9 {
            let ch = char::from_digit(d, 10).unwrap();
            assert!(
                keypad.find_button_by_char(ch).is_some(),
                "Missing button for digit {d}"
            );
        }
    }

    #[test]
    fn prop_all_operators_have_buttons() {
        let keypad = Keypad::new();
        for op in ['+', '-', '*', '/', '^'] {
            assert!(
                keypad.find_button_by_char(op).is_some(),
                "Missing button for operator {op}"
            );
        }
    }

    #[test]
    fn prop_button_char_roundtrip() {
        let keypad = Keypad::new();
        for btn in keypad.buttons() {
            if let Some(ch) = btn.to_char() {
                // Should be able to find it back
                let found = keypad.find_button_by_char(ch);
                assert!(found.is_some(), "Cannot find button for char '{ch}'");
            }
        }
    }

    #[test]
    fn prop_press_release_idempotent() {
        let mut keypad = Keypad::new();
        keypad.press_button(5);
        keypad.press_button(5); // Press again
        assert!(keypad.get_button(5).unwrap().pressed);

        keypad.release_all();
        keypad.release_all(); // Release again
        for btn in keypad.buttons() {
            assert!(!btn.pressed);
        }
    }

    #[test]
    fn prop_highlight_releases_others() {
        let mut keypad = Keypad::new();
        keypad.press_button(0);
        keypad.press_button(5);
        keypad.press_button(10);

        keypad.highlight_char('1'); // Should release all and press only '1'

        let pressed_count = keypad.buttons().filter(|b| b.pressed).count();
        assert_eq!(pressed_count, 1);
    }
}
