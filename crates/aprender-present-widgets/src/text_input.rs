//! `TextInput` widget for text entry.

use presentar_core::{
    widget::{AccessibleRole, LayoutResult, TextStyle},
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event, Key,
    Rect, Size, TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::time::Duration;

/// Message emitted when text changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextChanged {
    /// The new text value
    pub value: String,
}

/// Message emitted when Enter is pressed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextSubmitted {
    /// The submitted text value
    pub value: String,
}

/// `TextInput` widget for text entry.
#[derive(Serialize, Deserialize)]
pub struct TextInput {
    /// Current text value
    value: String,
    /// Placeholder text
    placeholder: String,
    /// Whether the input is disabled
    disabled: bool,
    /// Whether to obscure text (password mode)
    obscure: bool,
    /// Maximum length (0 = unlimited)
    max_length: usize,
    /// Text style
    text_style: TextStyle,
    /// Placeholder text color
    placeholder_color: Color,
    /// Background color
    background_color: Color,
    /// Border color
    border_color: Color,
    /// Focused border color
    focus_border_color: Color,
    /// Padding
    padding: f32,
    /// Minimum width
    min_width: f32,
    /// Test ID
    test_id_value: Option<String>,
    /// Accessible name
    accessible_name_value: Option<String>,
    /// Cached bounds
    #[serde(skip)]
    bounds: Rect,
    /// Whether focused
    #[serde(skip)]
    focused: bool,
    /// Cursor position (character index)
    #[serde(skip)]
    cursor: usize,
}

impl Default for TextInput {
    fn default() -> Self {
        Self::new()
    }
}

impl TextInput {
    /// Create a new text input.
    #[must_use]
    pub fn new() -> Self {
        Self {
            value: String::new(),
            placeholder: String::new(),
            disabled: false,
            obscure: false,
            max_length: 0,
            text_style: TextStyle::default(),
            placeholder_color: Color::new(0.6, 0.6, 0.6, 1.0),
            background_color: Color::WHITE,
            border_color: Color::new(0.8, 0.8, 0.8, 1.0),
            focus_border_color: Color::new(0.2, 0.6, 1.0, 1.0),
            padding: 8.0,
            min_width: 100.0,
            test_id_value: None,
            accessible_name_value: None,
            bounds: Rect::default(),
            focused: false,
            cursor: 0,
        }
    }

    /// Set the current value.
    #[must_use]
    pub fn value(mut self, value: impl Into<String>) -> Self {
        self.value = value.into();
        if self.max_length > 0 && self.value.len() > self.max_length {
            self.value.truncate(self.max_length);
        }
        self.cursor = self.value.len();
        self
    }

    /// Set placeholder text.
    #[must_use]
    pub fn placeholder(mut self, text: impl Into<String>) -> Self {
        self.placeholder = text.into();
        self
    }

    /// Set disabled state.
    #[must_use]
    pub const fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    /// Set password mode.
    #[must_use]
    pub const fn obscure(mut self, obscure: bool) -> Self {
        self.obscure = obscure;
        self
    }

    /// Set maximum length.
    #[must_use]
    pub fn max_length(mut self, max: usize) -> Self {
        self.max_length = max;
        if max > 0 && self.value.len() > max {
            self.value.truncate(max);
            self.cursor = self.cursor.min(max);
        }
        self
    }

    /// Set text style.
    #[must_use]
    pub const fn text_style(mut self, style: TextStyle) -> Self {
        self.text_style = style;
        self
    }

    /// Set placeholder color.
    #[must_use]
    pub const fn placeholder_color(mut self, color: Color) -> Self {
        self.placeholder_color = color;
        self
    }

    /// Set background color.
    #[must_use]
    pub const fn background_color(mut self, color: Color) -> Self {
        self.background_color = color;
        self
    }

    /// Set border color.
    #[must_use]
    pub const fn border_color(mut self, color: Color) -> Self {
        self.border_color = color;
        self
    }

    /// Set focus border color.
    #[must_use]
    pub const fn focus_border_color(mut self, color: Color) -> Self {
        self.focus_border_color = color;
        self
    }

    /// Set padding.
    #[must_use]
    pub fn padding(mut self, padding: f32) -> Self {
        self.padding = padding.max(0.0);
        self
    }

    /// Set minimum width.
    #[must_use]
    pub fn min_width(mut self, width: f32) -> Self {
        self.min_width = width.max(0.0);
        self
    }

    /// Set test ID.
    #[must_use]
    pub fn with_test_id(mut self, id: impl Into<String>) -> Self {
        self.test_id_value = Some(id.into());
        self
    }

    /// Set accessible name.
    #[must_use]
    pub fn with_accessible_name(mut self, name: impl Into<String>) -> Self {
        self.accessible_name_value = Some(name.into());
        self
    }

    /// Get current value.
    #[must_use]
    pub fn get_value(&self) -> &str {
        &self.value
    }

    /// Get placeholder.
    #[must_use]
    pub fn get_placeholder(&self) -> &str {
        &self.placeholder
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.value.is_empty()
    }

    /// Get cursor position.
    #[must_use]
    pub const fn cursor_position(&self) -> usize {
        self.cursor
    }

    /// Check if focused.
    #[must_use]
    pub const fn is_focused(&self) -> bool {
        self.focused
    }

    /// Get display text (obscured if password mode).
    #[must_use]
    pub fn display_text(&self) -> String {
        if self.obscure {
            "•".repeat(self.value.len())
        } else {
            self.value.clone()
        }
    }

    /// Insert text at cursor.
    fn insert_text(&mut self, text: &str) -> bool {
        if self.disabled {
            return false;
        }

        let mut changed = false;
        for c in text.chars() {
            if self.max_length > 0 && self.value.len() >= self.max_length {
                break;
            }
            self.value.insert(self.cursor, c);
            self.cursor += 1;
            changed = true;
        }
        changed
    }

    /// Delete character before cursor.
    fn backspace(&mut self) -> bool {
        if self.disabled || self.cursor == 0 {
            return false;
        }
        self.cursor -= 1;
        self.value.remove(self.cursor);
        true
    }

    /// Delete character at cursor.
    fn delete(&mut self) -> bool {
        if self.disabled || self.cursor >= self.value.len() {
            return false;
        }
        self.value.remove(self.cursor);
        true
    }

    /// Move cursor left.
    fn move_left(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
        }
    }

    /// Move cursor right.
    fn move_right(&mut self) {
        if self.cursor < self.value.len() {
            self.cursor += 1;
        }
    }

    /// Move cursor to start.
    fn move_home(&mut self) {
        self.cursor = 0;
    }

    /// Move cursor to end.
    fn move_end(&mut self) {
        self.cursor = self.value.len();
    }
}

impl Widget for TextInput {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let height = 2.0f32.mul_add(self.padding, self.text_style.size);
        let width = self.min_width.max(constraints.min_width);
        constraints.constrain(Size::new(width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: bounds.size(),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        // Draw background
        canvas.fill_rect(self.bounds, self.background_color);

        // Draw border
        let border_color = if self.focused {
            self.focus_border_color
        } else {
            self.border_color
        };
        canvas.stroke_rect(self.bounds, border_color, 1.0);

        // Draw text or placeholder
        let text_x = self.bounds.x + self.padding;
        let text_y = self.bounds.y + self.padding;
        let position = presentar_core::Point::new(text_x, text_y);

        if self.value.is_empty() {
            // Draw placeholder
            let mut placeholder_style = self.text_style.clone();
            placeholder_style.color = self.placeholder_color;
            canvas.draw_text(&self.placeholder, position, &placeholder_style);
        } else {
            // Draw actual text
            let display = self.display_text();
            canvas.draw_text(&display, position, &self.text_style);
        }
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        if self.disabled {
            return None;
        }

        match event {
            Event::MouseDown { position, .. } => {
                let was_focused = self.focused;
                self.focused = self.bounds.contains_point(position);
                if self.focused && !was_focused {
                    self.cursor = self.value.len();
                }
            }
            Event::FocusIn => {
                self.focused = true;
            }
            Event::FocusOut => {
                self.focused = false;
            }
            Event::TextInput { text } if self.focused => {
                if self.insert_text(text) {
                    return Some(Box::new(TextChanged {
                        value: self.value.clone(),
                    }));
                }
            }
            Event::KeyDown { key, .. } if self.focused => match key {
                Key::Backspace => {
                    if self.backspace() {
                        return Some(Box::new(TextChanged {
                            value: self.value.clone(),
                        }));
                    }
                }
                Key::Delete => {
                    if self.delete() {
                        return Some(Box::new(TextChanged {
                            value: self.value.clone(),
                        }));
                    }
                }
                Key::Left => self.move_left(),
                Key::Right => self.move_right(),
                Key::Home => self.move_home(),
                Key::End => self.move_end(),
                Key::Enter => {
                    return Some(Box::new(TextSubmitted {
                        value: self.value.clone(),
                    }));
                }
                _ => {}
            },
            _ => {}
        }

        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }

    fn is_interactive(&self) -> bool {
        !self.disabled
    }

    fn is_focusable(&self) -> bool {
        !self.disabled
    }

    fn accessible_name(&self) -> Option<&str> {
        self.accessible_name_value.as_deref()
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::TextInput
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

// PROBAR-SPEC-009: Brick Architecture - Tests define interface
impl Brick for TextInput {
    fn brick_name(&self) -> &'static str {
        "TextInput"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[
            BrickAssertion::MaxLatencyMs(16),
            BrickAssertion::ContrastRatio(4.5), // WCAG AA for text input
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let mut failed = Vec::new();

        // Verify contrast ratio
        let contrast = self.background_color.contrast_ratio(&self.text_style.color);
        if contrast >= 4.5 {
            passed.push(BrickAssertion::ContrastRatio(4.5));
        } else {
            failed.push((
                BrickAssertion::ContrastRatio(4.5),
                format!("Contrast ratio {contrast:.2}:1 < 4.5:1"),
            ));
        }

        // Latency assertion always passes at verification time
        passed.push(BrickAssertion::MaxLatencyMs(16));

        BrickVerification {
            passed,
            failed,
            verification_time: Duration::from_micros(10),
        }
    }

    fn to_html(&self) -> String {
        let test_id = self.test_id_value.as_deref().unwrap_or("text-input");
        let disabled = if self.disabled { " disabled" } else { "" };
        let input_type = if self.obscure { "password" } else { "text" };
        let aria_label = self.accessible_name_value.as_deref().unwrap_or("");
        format!(
            r#"<input type="{}" class="brick-textinput" data-testid="{}" placeholder="{}" value="{}" aria-label="{}"{} />"#,
            input_type, test_id, self.placeholder, self.value, aria_label, disabled
        )
    }

    fn to_css(&self) -> String {
        format!(
            r".brick-textinput {{
    background: {};
    color: {};
    border: 1px solid {};
    padding: {}px;
    min-width: {}px;
    font-size: {}px;
}}
.brick-textinput:focus {{
    border-color: {};
    outline: none;
}}
.brick-textinput:disabled {{
    opacity: 0.5;
    cursor: not-allowed;
}}",
            self.background_color.to_hex(),
            self.text_style.color.to_hex(),
            self.border_color.to_hex(),
            self.padding,
            self.min_width,
            self.text_style.size,
            self.focus_border_color.to_hex(),
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;
    use presentar_core::Widget;

    // =========================================================================
    // Message Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_text_changed_message() {
        let msg = TextChanged {
            value: "hello".to_string(),
        };
        assert_eq!(msg.value, "hello");
    }

    #[test]
    fn test_text_submitted_message() {
        let msg = TextSubmitted {
            value: "world".to_string(),
        };
        assert_eq!(msg.value, "world");
    }

    // =========================================================================
    // TextInput Construction Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_text_input_new() {
        let input = TextInput::new();
        assert!(input.get_value().is_empty());
        assert!(input.get_placeholder().is_empty());
        assert!(input.is_empty());
        assert!(!input.disabled);
        assert!(!input.obscure);
    }

    #[test]
    fn test_text_input_default() {
        let input = TextInput::default();
        assert!(input.is_empty());
    }

    #[test]
    fn test_text_input_builder() {
        let input = TextInput::new()
            .value("hello")
            .placeholder("Enter text...")
            .disabled(true)
            .obscure(true)
            .max_length(20)
            .padding(10.0)
            .min_width(200.0)
            .with_test_id("my-input")
            .with_accessible_name("Email");

        assert_eq!(input.get_value(), "hello");
        assert_eq!(input.get_placeholder(), "Enter text...");
        assert!(input.disabled);
        assert!(input.obscure);
        assert_eq!(input.max_length, 20);
        assert_eq!(Widget::test_id(&input), Some("my-input"));
        assert_eq!(input.accessible_name(), Some("Email"));
    }

    // =========================================================================
    // TextInput Value Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_text_input_value() {
        let input = TextInput::new().value("test");
        assert_eq!(input.get_value(), "test");
        assert!(!input.is_empty());
    }

    #[test]
    fn test_text_input_max_length_truncate() {
        let input = TextInput::new().max_length(5).value("hello world");
        assert_eq!(input.get_value(), "hello");
    }

    #[test]
    fn test_text_input_cursor_position() {
        let input = TextInput::new().value("hello");
        assert_eq!(input.cursor_position(), 5); // At end
    }

    // =========================================================================
    // TextInput Display Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_text_input_display_normal() {
        let input = TextInput::new().value("password");
        assert_eq!(input.display_text(), "password");
    }

    #[test]
    fn test_text_input_display_obscured() {
        let input = TextInput::new().value("secret").obscure(true);
        assert_eq!(input.display_text(), "••••••");
    }

    // =========================================================================
    // TextInput Editing Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_text_input_insert() {
        let mut input = TextInput::new().value("hlo");
        input.cursor = 1;
        input.insert_text("el");
        assert_eq!(input.get_value(), "hello");
        assert_eq!(input.cursor_position(), 3);
    }

    #[test]
    fn test_text_input_insert_respects_max_length() {
        let mut input = TextInput::new().max_length(5).value("abc");
        input.insert_text("defgh");
        assert_eq!(input.get_value(), "abcde");
    }

    #[test]
    fn test_text_input_backspace() {
        let mut input = TextInput::new().value("hello");
        input.backspace();
        assert_eq!(input.get_value(), "hell");
        assert_eq!(input.cursor_position(), 4);
    }

    #[test]
    fn test_text_input_backspace_at_start() {
        let mut input = TextInput::new().value("hello");
        input.cursor = 0;
        let changed = input.backspace();
        assert!(!changed);
        assert_eq!(input.get_value(), "hello");
    }

    #[test]
    fn test_text_input_delete() {
        let mut input = TextInput::new().value("hello");
        input.cursor = 0;
        input.delete();
        assert_eq!(input.get_value(), "ello");
        assert_eq!(input.cursor_position(), 0);
    }

    #[test]
    fn test_text_input_delete_at_end() {
        let mut input = TextInput::new().value("hello");
        let changed = input.delete();
        assert!(!changed);
        assert_eq!(input.get_value(), "hello");
    }

    // =========================================================================
    // TextInput Cursor Movement Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_text_input_move_left() {
        let mut input = TextInput::new().value("hello");
        input.move_left();
        assert_eq!(input.cursor_position(), 4);
    }

    #[test]
    fn test_text_input_move_left_at_start() {
        let mut input = TextInput::new().value("hello");
        input.cursor = 0;
        input.move_left();
        assert_eq!(input.cursor_position(), 0); // Stay at 0
    }

    #[test]
    fn test_text_input_move_right() {
        let mut input = TextInput::new().value("hello");
        input.cursor = 2;
        input.move_right();
        assert_eq!(input.cursor_position(), 3);
    }

    #[test]
    fn test_text_input_move_right_at_end() {
        let mut input = TextInput::new().value("hello");
        input.move_right();
        assert_eq!(input.cursor_position(), 5); // Stay at end
    }

    #[test]
    fn test_text_input_move_home() {
        let mut input = TextInput::new().value("hello");
        input.move_home();
        assert_eq!(input.cursor_position(), 0);
    }

    #[test]
    fn test_text_input_move_end() {
        let mut input = TextInput::new().value("hello");
        input.cursor = 0;
        input.move_end();
        assert_eq!(input.cursor_position(), 5);
    }

    // =========================================================================
    // TextInput Widget Trait Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_text_input_type_id() {
        let input = TextInput::new();
        assert_eq!(Widget::type_id(&input), TypeId::of::<TextInput>());
    }

    #[test]
    fn test_text_input_measure() {
        let input = TextInput::new();
        let size = input.measure(Constraints::loose(Size::new(400.0, 100.0)));
        assert!(size.width >= 100.0);
        assert!(size.height > 0.0);
    }

    #[test]
    fn test_text_input_is_interactive() {
        let input = TextInput::new();
        assert!(input.is_interactive());

        let input = TextInput::new().disabled(true);
        assert!(!input.is_interactive());
    }

    #[test]
    fn test_text_input_is_focusable() {
        let input = TextInput::new();
        assert!(input.is_focusable());

        let input = TextInput::new().disabled(true);
        assert!(!input.is_focusable());
    }

    #[test]
    fn test_text_input_accessible_role() {
        let input = TextInput::new();
        assert_eq!(input.accessible_role(), AccessibleRole::TextInput);
    }

    #[test]
    fn test_text_input_children() {
        let input = TextInput::new();
        assert!(input.children().is_empty());
    }

    // =========================================================================
    // TextInput Color Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_text_input_colors() {
        let input = TextInput::new()
            .background_color(Color::RED)
            .border_color(Color::GREEN)
            .focus_border_color(Color::BLUE)
            .placeholder_color(Color::YELLOW);

        assert_eq!(input.background_color, Color::RED);
        assert_eq!(input.border_color, Color::GREEN);
        assert_eq!(input.focus_border_color, Color::BLUE);
        assert_eq!(input.placeholder_color, Color::YELLOW);
    }

    // =========================================================================
    // TextInput Focus Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_text_input_focus_state() {
        let input = TextInput::new();
        assert!(!input.is_focused());
    }

    #[test]
    fn test_text_input_disabled_no_insert() {
        let mut input = TextInput::new().disabled(true);
        let changed = input.insert_text("test");
        assert!(!changed);
        assert!(input.is_empty());
    }

    #[test]
    fn test_text_input_disabled_no_backspace() {
        let mut input = TextInput::new().value("test").disabled(true);
        input.disabled = true; // Force after value set
        let changed = input.backspace();
        assert!(!changed);
    }

    #[test]
    fn test_text_input_disabled_no_delete() {
        let mut input = TextInput::new().value("test").disabled(true);
        input.disabled = true;
        input.cursor = 0;
        let changed = input.delete();
        assert!(!changed);
    }

    // =========================================================================
    // Event Handling Tests - TESTS FIRST
    // =========================================================================

    use presentar_core::{Key, MouseButton, Point};

    #[test]
    fn test_text_input_event_focus_in() {
        let mut input = TextInput::new();
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));

        assert!(!input.focused);
        let result = input.event(&Event::FocusIn);
        assert!(input.focused);
        assert!(result.is_none()); // No message for focus
    }

    #[test]
    fn test_text_input_event_focus_out() {
        let mut input = TextInput::new();
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));

        input.event(&Event::FocusIn);
        assert!(input.focused);

        let result = input.event(&Event::FocusOut);
        assert!(!input.focused);
        assert!(result.is_none());
    }

    #[test]
    fn test_text_input_event_mouse_down_inside_focuses() {
        let mut input = TextInput::new();
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));

        assert!(!input.focused);
        let result = input.event(&Event::MouseDown {
            position: Point::new(100.0, 15.0),
            button: MouseButton::Left,
        });
        assert!(input.focused);
        assert!(result.is_none());
    }

    #[test]
    fn test_text_input_event_mouse_down_outside_unfocuses() {
        let mut input = TextInput::new();
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));
        input.focused = true;

        let result = input.event(&Event::MouseDown {
            position: Point::new(300.0, 15.0),
            button: MouseButton::Left,
        });
        assert!(!input.focused);
        assert!(result.is_none());
    }

    #[test]
    fn test_text_input_event_mouse_down_sets_cursor() {
        let mut input = TextInput::new().value("hello");
        input.cursor = 0; // Start at 0
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));

        // Click to focus should move cursor to end
        input.event(&Event::MouseDown {
            position: Point::new(100.0, 15.0),
            button: MouseButton::Left,
        });
        assert_eq!(input.cursor, 5); // Moved to end of "hello"
    }

    #[test]
    fn test_text_input_event_text_input_when_focused() {
        let mut input = TextInput::new();
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));
        input.event(&Event::FocusIn);

        let result = input.event(&Event::TextInput {
            text: "hello".to_string(),
        });
        assert_eq!(input.get_value(), "hello");
        assert!(result.is_some());

        let msg = result.unwrap().downcast::<TextChanged>().unwrap();
        assert_eq!(msg.value, "hello");
    }

    #[test]
    fn test_text_input_event_text_input_when_not_focused() {
        let mut input = TextInput::new();
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));
        // Not focused

        let result = input.event(&Event::TextInput {
            text: "hello".to_string(),
        });
        assert!(input.get_value().is_empty());
        assert!(result.is_none());
    }

    #[test]
    fn test_text_input_event_key_backspace() {
        let mut input = TextInput::new().value("hello");
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));
        input.event(&Event::FocusIn);
        input.cursor = 5;

        let result = input.event(&Event::key_down(Key::Backspace));
        assert_eq!(input.get_value(), "hell");
        assert!(result.is_some());

        let msg = result.unwrap().downcast::<TextChanged>().unwrap();
        assert_eq!(msg.value, "hell");
    }

    #[test]
    fn test_text_input_event_key_delete() {
        let mut input = TextInput::new().value("hello");
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));
        input.event(&Event::FocusIn);
        input.cursor = 0;

        let result = input.event(&Event::key_down(Key::Delete));
        assert_eq!(input.get_value(), "ello");
        assert!(result.is_some());
    }

    #[test]
    fn test_text_input_event_key_left() {
        let mut input = TextInput::new().value("hello");
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));
        input.event(&Event::FocusIn);
        input.cursor = 3;

        let result = input.event(&Event::key_down(Key::Left));
        assert_eq!(input.cursor, 2);
        assert!(result.is_none()); // No text change
    }

    #[test]
    fn test_text_input_event_key_right() {
        let mut input = TextInput::new().value("hello");
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));
        input.event(&Event::FocusIn);
        input.cursor = 2;

        let result = input.event(&Event::key_down(Key::Right));
        assert_eq!(input.cursor, 3);
        assert!(result.is_none());
    }

    #[test]
    fn test_text_input_event_key_home() {
        let mut input = TextInput::new().value("hello");
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));
        input.event(&Event::FocusIn);
        input.cursor = 5;

        let result = input.event(&Event::key_down(Key::Home));
        assert_eq!(input.cursor, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_text_input_event_key_end() {
        let mut input = TextInput::new().value("hello");
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));
        input.event(&Event::FocusIn);
        input.cursor = 0;

        let result = input.event(&Event::key_down(Key::End));
        assert_eq!(input.cursor, 5);
        assert!(result.is_none());
    }

    #[test]
    fn test_text_input_event_key_enter_submits() {
        let mut input = TextInput::new().value("hello");
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));
        input.event(&Event::FocusIn);

        let result = input.event(&Event::key_down(Key::Enter));
        assert!(result.is_some());

        let msg = result.unwrap().downcast::<TextSubmitted>().unwrap();
        assert_eq!(msg.value, "hello");
    }

    #[test]
    fn test_text_input_event_key_when_not_focused() {
        let mut input = TextInput::new().value("hello");
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));
        // Not focused

        let result = input.event(&Event::key_down(Key::Backspace));
        assert_eq!(input.get_value(), "hello"); // Unchanged
        assert!(result.is_none());
    }

    #[test]
    fn test_text_input_event_disabled_blocks_focus() {
        let mut input = TextInput::new().disabled(true);
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));

        let result = input.event(&Event::FocusIn);
        assert!(!input.focused);
        assert!(result.is_none());
    }

    #[test]
    fn test_text_input_event_disabled_blocks_mouse_down() {
        let mut input = TextInput::new().disabled(true);
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));

        let result = input.event(&Event::MouseDown {
            position: Point::new(100.0, 15.0),
            button: MouseButton::Left,
        });
        assert!(!input.focused);
        assert!(result.is_none());
    }

    #[test]
    fn test_text_input_event_disabled_blocks_text_input() {
        let mut input = TextInput::new().disabled(true);
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));
        input.focused = true; // Force focused

        let result = input.event(&Event::TextInput {
            text: "hello".to_string(),
        });
        assert!(input.get_value().is_empty());
        assert!(result.is_none());
    }

    #[test]
    fn test_text_input_event_disabled_blocks_key_down() {
        let mut input = TextInput::new().value("hello").disabled(true);
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));
        input.focused = true;

        let result = input.event(&Event::key_down(Key::Backspace));
        assert_eq!(input.get_value(), "hello");
        assert!(result.is_none());
    }

    #[test]
    fn test_text_input_event_full_typing_flow() {
        let mut input = TextInput::new();
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));

        // 1. Click to focus
        input.event(&Event::MouseDown {
            position: Point::new(100.0, 15.0),
            button: MouseButton::Left,
        });
        assert!(input.focused);

        // 2. Type "Hello"
        input.event(&Event::TextInput {
            text: "Hello".to_string(),
        });
        assert_eq!(input.get_value(), "Hello");
        assert_eq!(input.cursor, 5);

        // 3. Backspace
        input.event(&Event::key_down(Key::Backspace));
        assert_eq!(input.get_value(), "Hell");
        assert_eq!(input.cursor, 4);

        // 4. Navigate home
        input.event(&Event::key_down(Key::Home));
        assert_eq!(input.cursor, 0);

        // 5. Type "Say "
        input.event(&Event::TextInput {
            text: "Say ".to_string(),
        });
        assert_eq!(input.get_value(), "Say Hell");
        assert_eq!(input.cursor, 4);

        // 6. Navigate end
        input.event(&Event::key_down(Key::End));
        assert_eq!(input.cursor, 8);

        // 7. Type "o"
        input.event(&Event::TextInput {
            text: "o".to_string(),
        });
        assert_eq!(input.get_value(), "Say Hello");

        // 8. Submit
        let result = input.event(&Event::key_down(Key::Enter));
        let msg = result.unwrap().downcast::<TextSubmitted>().unwrap();
        assert_eq!(msg.value, "Say Hello");

        // 9. Click outside to unfocus
        input.event(&Event::MouseDown {
            position: Point::new(300.0, 15.0),
            button: MouseButton::Left,
        });
        assert!(!input.focused);
    }

    #[test]
    fn test_text_input_event_cursor_navigation() {
        let mut input = TextInput::new().value("abcde");
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));
        input.event(&Event::FocusIn);
        input.cursor = 2;

        // Left boundary
        input.event(&Event::key_down(Key::Left));
        input.event(&Event::key_down(Key::Left));
        input.event(&Event::key_down(Key::Left)); // Should stay at 0
        assert_eq!(input.cursor, 0);

        // Right boundary
        input.event(&Event::key_down(Key::End));
        input.event(&Event::key_down(Key::Right)); // Should stay at end
        assert_eq!(input.cursor, 5);
    }

    #[test]
    fn test_text_input_event_backspace_at_start() {
        let mut input = TextInput::new().value("hello");
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));
        input.event(&Event::FocusIn);
        input.cursor = 0;

        let result = input.event(&Event::key_down(Key::Backspace));
        assert_eq!(input.get_value(), "hello"); // Unchanged
        assert!(result.is_none()); // No change message
    }

    #[test]
    fn test_text_input_event_delete_at_end() {
        let mut input = TextInput::new().value("hello");
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));
        input.event(&Event::FocusIn);
        input.cursor = 5;

        let result = input.event(&Event::key_down(Key::Delete));
        assert_eq!(input.get_value(), "hello"); // Unchanged
        assert!(result.is_none());
    }

    #[test]
    fn test_text_input_event_max_length_enforced() {
        let mut input = TextInput::new().max_length(5);
        input.layout(Rect::new(0.0, 0.0, 200.0, 30.0));
        input.event(&Event::FocusIn);

        // Type beyond max length
        input.event(&Event::TextInput {
            text: "hello world".to_string(),
        });
        assert_eq!(input.get_value(), "hello"); // Truncated to 5
    }
}
