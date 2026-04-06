//! Button widget for user interactions.

use presentar_core::{
    widget::{AccessibleRole, FontWeight, LayoutResult, TextStyle},
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints,
    CornerRadius, Event, MouseButton, Point, Rect, Size, TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::time::Duration;

/// Button widget with label and click handling.
#[derive(Clone, Serialize, Deserialize)]
pub struct Button {
    /// Button label
    label: String,
    /// Background color (normal state)
    background: Color,
    /// Background color (hover state)
    background_hover: Color,
    /// Background color (pressed state)
    background_pressed: Color,
    /// Text color
    text_color: Color,
    /// Corner radius
    corner_radius: CornerRadius,
    /// Padding
    padding: f32,
    /// Font size
    font_size: f32,
    /// Whether button is disabled
    disabled: bool,
    /// Test ID
    test_id_value: Option<String>,
    /// Accessible name (overrides label)
    accessible_name: Option<String>,
    /// Current hover state
    #[serde(skip)]
    hovered: bool,
    /// Current pressed state
    #[serde(skip)]
    pressed: bool,
    /// Cached bounds
    #[serde(skip)]
    bounds: Rect,
}

/// Message emitted when button is clicked.
#[derive(Debug, Clone)]
pub struct ButtonClicked;

impl Button {
    /// Create a new button with label.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            background: Color::from_hex("#6366f1").unwrap_or(Color::BLACK),
            background_hover: Color::from_hex("#4f46e5").unwrap_or(Color::BLACK),
            background_pressed: Color::from_hex("#4338ca").unwrap_or(Color::BLACK),
            text_color: Color::WHITE,
            corner_radius: CornerRadius::uniform(4.0),
            padding: 12.0,
            font_size: 14.0,
            disabled: false,
            test_id_value: None,
            accessible_name: None,
            hovered: false,
            pressed: false,
            bounds: Rect::default(),
        }
    }

    /// Set background color.
    #[must_use]
    pub const fn background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    /// Set hover background color.
    #[must_use]
    pub const fn background_hover(mut self, color: Color) -> Self {
        self.background_hover = color;
        self
    }

    /// Set pressed background color.
    #[must_use]
    pub const fn background_pressed(mut self, color: Color) -> Self {
        self.background_pressed = color;
        self
    }

    /// Set text color.
    #[must_use]
    pub const fn text_color(mut self, color: Color) -> Self {
        self.text_color = color;
        self
    }

    /// Set corner radius.
    #[must_use]
    pub const fn corner_radius(mut self, radius: CornerRadius) -> Self {
        self.corner_radius = radius;
        self
    }

    /// Set padding.
    #[must_use]
    pub const fn padding(mut self, padding: f32) -> Self {
        self.padding = padding;
        self
    }

    /// Set font size.
    #[must_use]
    pub const fn font_size(mut self, size: f32) -> Self {
        self.font_size = size;
        self
    }

    /// Set disabled state.
    #[must_use]
    pub const fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
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
        self.accessible_name = Some(name.into());
        self
    }

    /// Get the current background color based on state.
    fn current_background(&self) -> Color {
        if self.disabled {
            // Desaturated version
            let gray = (self.background.r + self.background.g + self.background.b) / 3.0;
            Color::rgb(gray, gray, gray)
        } else if self.pressed {
            self.background_pressed
        } else if self.hovered {
            self.background_hover
        } else {
            self.background
        }
    }

    /// Estimate text size.
    fn estimate_text_size(&self) -> Size {
        let char_width = self.font_size * 0.6;
        let width = self.label.len() as f32 * char_width;
        let height = self.font_size * 1.2;
        Size::new(width, height)
    }
}

impl Widget for Button {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let text_size = self.estimate_text_size();
        let size = Size::new(
            self.padding.mul_add(2.0, text_size.width),
            self.padding.mul_add(2.0, text_size.height),
        );
        constraints.constrain(size)
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: bounds.size(),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        // Draw background
        canvas.fill_rect(self.bounds, self.current_background());

        // Draw text centered
        let text_size = self.estimate_text_size();
        let text_pos = Point::new(
            self.bounds.x + (self.bounds.width - text_size.width) / 2.0,
            self.bounds.y + (self.bounds.height - text_size.height) / 2.0,
        );

        let style = TextStyle {
            size: self.font_size,
            color: if self.disabled {
                Color::rgb(0.7, 0.7, 0.7)
            } else {
                self.text_color
            },
            weight: FontWeight::Medium,
            ..Default::default()
        };

        canvas.draw_text(&self.label, text_pos, &style);
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        if self.disabled {
            return None;
        }

        match event {
            Event::MouseEnter => {
                self.hovered = true;
                None
            }
            Event::MouseLeave => {
                self.hovered = false;
                self.pressed = false;
                None
            }
            Event::MouseDown {
                position,
                button: MouseButton::Left,
            } => {
                if self.bounds.contains_point(position) {
                    self.pressed = true;
                }
                None
            }
            Event::MouseUp {
                position,
                button: MouseButton::Left,
            } => {
                let was_pressed = self.pressed;
                self.pressed = false;

                if was_pressed && self.bounds.contains_point(position) {
                    Some(Box::new(ButtonClicked))
                } else {
                    None
                }
            }
            Event::KeyDown {
                key: presentar_core::Key::Enter | presentar_core::Key::Space,
                ..
            } => {
                self.pressed = true;
                None
            }
            Event::KeyUp {
                key: presentar_core::Key::Enter | presentar_core::Key::Space,
                ..
            } => {
                self.pressed = false;
                Some(Box::new(ButtonClicked))
            }
            _ => None,
        }
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
        self.accessible_name.as_deref().or(Some(&self.label))
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Button
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

// PROBAR-SPEC-009: Brick Architecture - Tests define interface
impl Brick for Button {
    fn brick_name(&self) -> &'static str {
        "Button"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        // Button must have visible text and adequate contrast
        &[
            BrickAssertion::TextVisible,
            BrickAssertion::ContrastRatio(4.5), // WCAG AA
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16) // 60fps
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let mut failed = Vec::new();

        // Verify text contrast
        let bg = self.current_background();
        let contrast = bg.contrast_ratio(&self.text_color);
        if contrast >= 4.5 {
            passed.push(BrickAssertion::ContrastRatio(4.5));
        } else {
            failed.push((
                BrickAssertion::ContrastRatio(4.5),
                format!("Contrast ratio {contrast:.2}:1 < 4.5:1"),
            ));
        }

        // Text visibility
        if self.label.is_empty() {
            failed.push((BrickAssertion::TextVisible, "Button has no label".into()));
        } else {
            passed.push(BrickAssertion::TextVisible);
        }

        BrickVerification {
            passed,
            failed,
            verification_time: Duration::from_micros(10),
        }
    }

    fn to_html(&self) -> String {
        let disabled = if self.disabled { " disabled" } else { "" };
        let test_id = self.test_id_value.as_deref().unwrap_or("button");
        format!(
            r#"<button class="brick-button" data-testid="{}" aria-label="{}"{}>{}</button>"#,
            test_id,
            self.accessible_name.as_deref().unwrap_or(&self.label),
            disabled,
            self.label
        )
    }

    fn to_css(&self) -> String {
        format!(
            r".brick-button {{
    background: {};
    color: {};
    padding: {}px;
    font-size: {}px;
    border: none;
    border-radius: {}px;
    cursor: pointer;
}}
.brick-button:hover {{ background: {}; }}
.brick-button:active {{ background: {}; }}
.brick-button:disabled {{ opacity: 0.5; cursor: not-allowed; }}",
            self.background.to_hex(),
            self.text_color.to_hex(),
            self.padding,
            self.font_size,
            self.corner_radius.top_left,
            self.background_hover.to_hex(),
            self.background_pressed.to_hex(),
        )
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;
    use presentar_core::draw::DrawCommand;
    use presentar_core::{RecordingCanvas, Widget};

    #[test]
    fn test_button_new() {
        let b = Button::new("Click me");
        assert_eq!(b.label, "Click me");
        assert!(!b.disabled);
    }

    #[test]
    fn test_button_builder() {
        let b = Button::new("Test")
            .padding(20.0)
            .font_size(18.0)
            .disabled(true)
            .with_test_id("my-button");

        assert_eq!(b.padding, 20.0);
        assert_eq!(b.font_size, 18.0);
        assert!(b.disabled);
        assert_eq!(Widget::test_id(&b), Some("my-button"));
    }

    #[test]
    fn test_button_accessible() {
        let b = Button::new("OK");
        assert_eq!(Widget::accessible_name(&b), Some("OK"));
        assert_eq!(Widget::accessible_role(&b), AccessibleRole::Button);
        assert!(Widget::is_focusable(&b));
    }

    #[test]
    fn test_button_disabled_not_focusable() {
        let b = Button::new("OK").disabled(true);
        assert!(!Widget::is_focusable(&b));
        assert!(!Widget::is_interactive(&b));
    }

    #[test]
    fn test_button_measure() {
        let b = Button::new("Test");
        let size = b.measure(Constraints::loose(Size::new(1000.0, 1000.0)));
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);
    }

    // ===== Paint Tests =====

    #[test]
    fn test_button_paint_draws_background() {
        let mut button = Button::new("Click");
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        let mut canvas = RecordingCanvas::new();
        button.paint(&mut canvas);

        // Should have at least 2 commands: background rect + text
        assert!(canvas.command_count() >= 2);

        // First command should be the background rect
        match &canvas.commands()[0] {
            DrawCommand::Rect { bounds, style, .. } => {
                assert_eq!(bounds.width, 100.0);
                assert_eq!(bounds.height, 40.0);
                assert!(style.fill.is_some());
            }
            _ => panic!("Expected Rect command for background"),
        }
    }

    #[test]
    fn test_button_paint_draws_text() {
        let mut button = Button::new("Hello");
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        let mut canvas = RecordingCanvas::new();
        button.paint(&mut canvas);

        // Should have text command
        let has_text = canvas
            .commands()
            .iter()
            .any(|cmd| matches!(cmd, DrawCommand::Text { content, .. } if content == "Hello"));
        assert!(has_text, "Should draw button label text");
    }

    #[test]
    fn test_button_paint_disabled_uses_gray() {
        let mut button = Button::new("Disabled").disabled(true);
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        let mut canvas = RecordingCanvas::new();
        button.paint(&mut canvas);

        // Check text color is gray (disabled)
        let text_cmd = canvas
            .commands()
            .iter()
            .find(|cmd| matches!(cmd, DrawCommand::Text { .. }));

        if let Some(DrawCommand::Text { style, .. }) = text_cmd {
            // Disabled text should be grayish
            assert!(style.color.r > 0.5 && style.color.g > 0.5 && style.color.b > 0.5);
        } else {
            panic!("Expected Text command");
        }
    }

    #[test]
    fn test_button_paint_hovered_uses_hover_color() {
        let mut button = Button::new("Hover")
            .background(Color::RED)
            .background_hover(Color::BLUE);
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        // Simulate hover
        button.event(&Event::MouseEnter);

        let mut canvas = RecordingCanvas::new();
        button.paint(&mut canvas);

        // Background should use hover color
        match &canvas.commands()[0] {
            DrawCommand::Rect { style, .. } => {
                assert_eq!(style.fill, Some(Color::BLUE));
            }
            _ => panic!("Expected Rect command"),
        }
    }

    #[test]
    fn test_button_paint_pressed_uses_pressed_color() {
        let mut button = Button::new("Press")
            .background(Color::RED)
            .background_pressed(Color::GREEN);
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        // Simulate press
        button.event(&Event::MouseEnter);
        button.event(&Event::MouseDown {
            position: Point::new(50.0, 20.0),
            button: MouseButton::Left,
        });

        let mut canvas = RecordingCanvas::new();
        button.paint(&mut canvas);

        // Background should use pressed color
        match &canvas.commands()[0] {
            DrawCommand::Rect { style, .. } => {
                assert_eq!(style.fill, Some(Color::GREEN));
            }
            _ => panic!("Expected Rect command"),
        }
    }

    #[test]
    fn test_button_paint_text_centered() {
        let mut button = Button::new("X");
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        let mut canvas = RecordingCanvas::new();
        button.paint(&mut canvas);

        // Text should be roughly centered
        let text_cmd = canvas
            .commands()
            .iter()
            .find(|cmd| matches!(cmd, DrawCommand::Text { .. }));

        if let Some(DrawCommand::Text { position, .. }) = text_cmd {
            // Text should be somewhere in the middle, not at edge
            assert!(position.x > 10.0 && position.x < 90.0);
            assert!(position.y > 5.0 && position.y < 35.0);
        } else {
            panic!("Expected Text command");
        }
    }

    #[test]
    fn test_button_paint_custom_colors() {
        let mut button = Button::new("Custom")
            .background(Color::rgb(1.0, 0.0, 0.0))
            .text_color(Color::rgb(0.0, 1.0, 0.0));
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        let mut canvas = RecordingCanvas::new();
        button.paint(&mut canvas);

        // Check background color
        match &canvas.commands()[0] {
            DrawCommand::Rect { style, .. } => {
                let fill = style.fill.unwrap();
                assert!((fill.r - 1.0).abs() < 0.01);
                assert!(fill.g < 0.01);
                assert!(fill.b < 0.01);
            }
            _ => panic!("Expected Rect command"),
        }

        // Check text color
        let text_cmd = canvas
            .commands()
            .iter()
            .find(|cmd| matches!(cmd, DrawCommand::Text { .. }));
        if let Some(DrawCommand::Text { style, .. }) = text_cmd {
            assert!(style.color.r < 0.01);
            assert!((style.color.g - 1.0).abs() < 0.01);
            assert!(style.color.b < 0.01);
        }
    }

    // ===== Event Handling Tests =====

    use presentar_core::Key;

    #[test]
    fn test_button_event_mouse_enter_sets_hovered() {
        let mut button = Button::new("Test");
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        assert!(!button.hovered);
        let result = button.event(&Event::MouseEnter);
        assert!(button.hovered);
        assert!(result.is_none()); // No message emitted
    }

    #[test]
    fn test_button_event_mouse_leave_clears_hovered() {
        let mut button = Button::new("Test");
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        button.event(&Event::MouseEnter);
        assert!(button.hovered);

        let result = button.event(&Event::MouseLeave);
        assert!(!button.hovered);
        assert!(result.is_none());
    }

    #[test]
    fn test_button_event_mouse_leave_clears_pressed() {
        let mut button = Button::new("Test");
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        // Enter and press
        button.event(&Event::MouseEnter);
        button.event(&Event::MouseDown {
            position: Point::new(50.0, 20.0),
            button: MouseButton::Left,
        });
        assert!(button.pressed);

        // Leave should clear pressed
        button.event(&Event::MouseLeave);
        assert!(!button.pressed);
        assert!(!button.hovered);
    }

    #[test]
    fn test_button_event_mouse_down_sets_pressed() {
        let mut button = Button::new("Test");
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        assert!(!button.pressed);
        let result = button.event(&Event::MouseDown {
            position: Point::new(50.0, 20.0),
            button: MouseButton::Left,
        });
        assert!(button.pressed);
        assert!(result.is_none()); // MouseDown doesn't emit click
    }

    #[test]
    fn test_button_event_mouse_down_outside_bounds_no_press() {
        let mut button = Button::new("Test");
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        let result = button.event(&Event::MouseDown {
            position: Point::new(150.0, 20.0), // Outside bounds
            button: MouseButton::Left,
        });
        assert!(!button.pressed);
        assert!(result.is_none());
    }

    #[test]
    fn test_button_event_mouse_down_right_button_no_press() {
        let mut button = Button::new("Test");
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        let result = button.event(&Event::MouseDown {
            position: Point::new(50.0, 20.0),
            button: MouseButton::Right,
        });
        assert!(!button.pressed);
        assert!(result.is_none());
    }

    #[test]
    fn test_button_event_mouse_up_emits_clicked() {
        let mut button = Button::new("Test");
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        // Press down
        button.event(&Event::MouseDown {
            position: Point::new(50.0, 20.0),
            button: MouseButton::Left,
        });
        assert!(button.pressed);

        // Release inside bounds
        let result = button.event(&Event::MouseUp {
            position: Point::new(50.0, 20.0),
            button: MouseButton::Left,
        });
        assert!(!button.pressed);
        assert!(result.is_some());

        // Verify it's a ButtonClicked message
        let _msg: Box<ButtonClicked> = result.unwrap().downcast::<ButtonClicked>().unwrap();
    }

    #[test]
    fn test_button_event_mouse_up_outside_bounds_no_click() {
        let mut button = Button::new("Test");
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        // Press down inside
        button.event(&Event::MouseDown {
            position: Point::new(50.0, 20.0),
            button: MouseButton::Left,
        });
        assert!(button.pressed);

        // Release outside bounds
        let result = button.event(&Event::MouseUp {
            position: Point::new(150.0, 20.0),
            button: MouseButton::Left,
        });
        assert!(!button.pressed);
        assert!(result.is_none()); // No click emitted
    }

    #[test]
    fn test_button_event_mouse_up_without_prior_press_no_click() {
        let mut button = Button::new("Test");
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        // Mouse up without prior press
        let result = button.event(&Event::MouseUp {
            position: Point::new(50.0, 20.0),
            button: MouseButton::Left,
        });
        assert!(result.is_none());
    }

    #[test]
    fn test_button_event_mouse_up_right_button_no_effect() {
        let mut button = Button::new("Test");
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        // Press with left button
        button.event(&Event::MouseDown {
            position: Point::new(50.0, 20.0),
            button: MouseButton::Left,
        });

        // Release with right button (should not trigger click)
        let result = button.event(&Event::MouseUp {
            position: Point::new(50.0, 20.0),
            button: MouseButton::Right,
        });
        assert!(button.pressed); // Still pressed
        assert!(result.is_none());
    }

    #[test]
    fn test_button_event_key_down_enter_sets_pressed() {
        let mut button = Button::new("Test");
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        let result = button.event(&Event::key_down(Key::Enter));
        assert!(button.pressed);
        assert!(result.is_none()); // KeyDown doesn't emit click
    }

    #[test]
    fn test_button_event_key_down_space_sets_pressed() {
        let mut button = Button::new("Test");
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        let result = button.event(&Event::key_down(Key::Space));
        assert!(button.pressed);
        assert!(result.is_none());
    }

    #[test]
    fn test_button_event_key_up_enter_emits_clicked() {
        let mut button = Button::new("Test");
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        // Key down first
        button.event(&Event::key_down(Key::Enter));
        assert!(button.pressed);

        // Key up emits click
        let result = button.event(&Event::key_up(Key::Enter));
        assert!(!button.pressed);
        assert!(result.is_some());
    }

    #[test]
    fn test_button_event_key_up_space_emits_clicked() {
        let mut button = Button::new("Test");
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        button.event(&Event::key_down(Key::Space));
        let result = button.event(&Event::key_up(Key::Space));
        assert!(!button.pressed);
        assert!(result.is_some());
    }

    #[test]
    fn test_button_event_key_other_no_effect() {
        let mut button = Button::new("Test");
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        let result = button.event(&Event::key_down(Key::Escape));
        assert!(!button.pressed);
        assert!(result.is_none());
    }

    #[test]
    fn test_button_event_disabled_blocks_mouse_enter() {
        let mut button = Button::new("Test").disabled(true);
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        let result = button.event(&Event::MouseEnter);
        assert!(!button.hovered);
        assert!(result.is_none());
    }

    #[test]
    fn test_button_event_disabled_blocks_mouse_down() {
        let mut button = Button::new("Test").disabled(true);
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        let result = button.event(&Event::MouseDown {
            position: Point::new(50.0, 20.0),
            button: MouseButton::Left,
        });
        assert!(!button.pressed);
        assert!(result.is_none());
    }

    #[test]
    fn test_button_event_disabled_blocks_key_down() {
        let mut button = Button::new("Test").disabled(true);
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        let result = button.event(&Event::key_down(Key::Enter));
        assert!(!button.pressed);
        assert!(result.is_none());
    }

    #[test]
    fn test_button_event_disabled_blocks_key_up() {
        let mut button = Button::new("Test").disabled(true);
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        let result = button.event(&Event::key_up(Key::Enter));
        assert!(result.is_none());
    }

    #[test]
    fn test_button_click_full_interaction_flow() {
        let mut button = Button::new("Submit");
        button.layout(Rect::new(10.0, 10.0, 100.0, 40.0));

        // Full click flow: enter -> down -> up -> leave
        button.event(&Event::MouseEnter);
        assert!(button.hovered);
        assert!(!button.pressed);

        button.event(&Event::MouseDown {
            position: Point::new(50.0, 25.0),
            button: MouseButton::Left,
        });
        assert!(button.hovered);
        assert!(button.pressed);

        let result = button.event(&Event::MouseUp {
            position: Point::new(50.0, 25.0),
            button: MouseButton::Left,
        });
        assert!(button.hovered);
        assert!(!button.pressed);
        assert!(result.is_some()); // Click emitted

        button.event(&Event::MouseLeave);
        assert!(!button.hovered);
        assert!(!button.pressed);
    }

    #[test]
    fn test_button_drag_out_and_release_no_click() {
        let mut button = Button::new("Drag");
        button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

        // Press inside
        button.event(&Event::MouseEnter);
        button.event(&Event::MouseDown {
            position: Point::new(50.0, 20.0),
            button: MouseButton::Left,
        });
        assert!(button.pressed);

        // Leave while pressed
        button.event(&Event::MouseLeave);
        assert!(!button.pressed); // Cleared by leave

        // Release outside
        let result = button.event(&Event::MouseUp {
            position: Point::new(150.0, 20.0),
            button: MouseButton::Left,
        });
        assert!(result.is_none()); // No click
    }

    #[test]
    fn test_button_event_bounds_edge_cases() {
        let mut button = Button::new("Edge");
        button.layout(Rect::new(10.0, 20.0, 100.0, 40.0));

        // Click at top-left corner (inside)
        button.event(&Event::MouseDown {
            position: Point::new(10.0, 20.0),
            button: MouseButton::Left,
        });
        assert!(button.pressed);
        button.pressed = false;

        // Click at bottom-right corner (inside, at edge)
        button.event(&Event::MouseDown {
            position: Point::new(109.9, 59.9),
            button: MouseButton::Left,
        });
        assert!(button.pressed);
        button.pressed = false;

        // Click just outside right edge
        button.event(&Event::MouseDown {
            position: Point::new(111.0, 30.0),
            button: MouseButton::Left,
        });
        assert!(!button.pressed);
    }
}
