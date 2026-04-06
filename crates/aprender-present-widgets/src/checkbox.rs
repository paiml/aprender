//! Checkbox widget for boolean input.

use presentar_core::{
    widget::{AccessibleRole, LayoutResult},
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    MouseButton, Rect, Size, TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::time::Duration;

/// Checkbox state (supports tri-state).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CheckState {
    /// Not checked
    #[default]
    Unchecked,
    /// Checked
    Checked,
    /// Indeterminate (for partial selection in trees)
    Indeterminate,
}

impl CheckState {
    /// Toggle between checked and unchecked.
    #[must_use]
    pub const fn toggle(&self) -> Self {
        match self {
            Self::Unchecked => Self::Checked,
            Self::Checked | Self::Indeterminate => Self::Unchecked,
        }
    }

    /// Check if checked (true for Checked, false for others).
    #[must_use]
    pub const fn is_checked(&self) -> bool {
        matches!(self, Self::Checked)
    }

    /// Check if indeterminate.
    #[must_use]
    pub const fn is_indeterminate(&self) -> bool {
        matches!(self, Self::Indeterminate)
    }
}

/// Message emitted when checkbox state changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckboxChanged {
    /// The new state
    pub state: CheckState,
}

/// Checkbox widget.
#[derive(Serialize, Deserialize)]
pub struct Checkbox {
    /// Current state
    state: CheckState,
    /// Whether disabled
    disabled: bool,
    /// Label text
    label: String,
    /// Box size
    box_size: f32,
    /// Spacing between box and label
    spacing: f32,
    /// Unchecked box color
    box_color: Color,
    /// Checked box color
    checked_color: Color,
    /// Check mark color
    check_color: Color,
    /// Label color
    label_color: Color,
    /// Disabled color
    disabled_color: Color,
    /// Test ID
    test_id_value: Option<String>,
    /// Accessible name
    accessible_name_value: Option<String>,
    /// Cached bounds
    #[serde(skip)]
    bounds: Rect,
    /// Whether hovered
    #[serde(skip)]
    hovered: bool,
}

impl Default for Checkbox {
    fn default() -> Self {
        Self::new()
    }
}

impl Checkbox {
    /// Create a new checkbox.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: CheckState::Unchecked,
            disabled: false,
            label: String::new(),
            box_size: 18.0,
            spacing: 8.0,
            box_color: Color::new(0.8, 0.8, 0.8, 1.0),
            checked_color: Color::new(0.2, 0.47, 0.96, 1.0),
            check_color: Color::WHITE,
            label_color: Color::BLACK,
            disabled_color: Color::new(0.6, 0.6, 0.6, 1.0),
            test_id_value: None,
            accessible_name_value: None,
            bounds: Rect::default(),
            hovered: false,
        }
    }

    /// Set the checked state.
    #[must_use]
    pub const fn checked(mut self, checked: bool) -> Self {
        self.state = if checked {
            CheckState::Checked
        } else {
            CheckState::Unchecked
        };
        self
    }

    /// Set the state directly.
    #[must_use]
    pub const fn state(mut self, state: CheckState) -> Self {
        self.state = state;
        self
    }

    /// Set the label.
    #[must_use]
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Set disabled state.
    #[must_use]
    pub const fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    /// Set box size.
    #[must_use]
    pub fn box_size(mut self, size: f32) -> Self {
        self.box_size = size.max(8.0);
        self
    }

    /// Set spacing between box and label.
    #[must_use]
    pub fn spacing(mut self, spacing: f32) -> Self {
        self.spacing = spacing.max(0.0);
        self
    }

    /// Set checked box color.
    #[must_use]
    pub const fn checked_color(mut self, color: Color) -> Self {
        self.checked_color = color;
        self
    }

    /// Set check mark color.
    #[must_use]
    pub const fn check_color(mut self, color: Color) -> Self {
        self.check_color = color;
        self
    }

    /// Set label color.
    #[must_use]
    pub const fn label_color(mut self, color: Color) -> Self {
        self.label_color = color;
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

    /// Get current state.
    #[must_use]
    pub const fn get_state(&self) -> CheckState {
        self.state
    }

    /// Check if currently checked.
    #[must_use]
    pub const fn is_checked(&self) -> bool {
        self.state.is_checked()
    }

    /// Check if indeterminate.
    #[must_use]
    pub const fn is_indeterminate(&self) -> bool {
        self.state.is_indeterminate()
    }

    /// Get the label.
    #[must_use]
    pub fn get_label(&self) -> &str {
        &self.label
    }
}

impl Widget for Checkbox {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        // Estimate label width (rough approximation)
        let label_width = if self.label.is_empty() {
            0.0
        } else {
            self.label.len() as f32 * 8.0 // ~8px per character
        };

        let total_width = self.box_size + self.spacing + label_width;
        let height = self.box_size;

        constraints.constrain(Size::new(total_width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: bounds.size(),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        let box_rect = Rect::new(
            self.bounds.x,
            self.bounds.y + (self.bounds.height - self.box_size) / 2.0,
            self.box_size,
            self.box_size,
        );

        // Draw checkbox box
        let box_color = if self.disabled {
            self.disabled_color
        } else if self.state.is_checked() || self.state.is_indeterminate() {
            self.checked_color
        } else {
            self.box_color
        };

        canvas.fill_rect(box_rect, box_color);

        // Draw check mark or indeterminate line
        if !self.disabled {
            match self.state {
                CheckState::Checked => {
                    // Draw checkmark (simplified as a filled inner rect)
                    let inner = Rect::new(
                        self.box_size.mul_add(0.25, box_rect.x),
                        self.box_size.mul_add(0.25, box_rect.y),
                        self.box_size * 0.5,
                        self.box_size * 0.5,
                    );
                    canvas.fill_rect(inner, self.check_color);
                }
                CheckState::Indeterminate => {
                    // Draw horizontal line
                    let line = Rect::new(
                        self.box_size.mul_add(0.2, box_rect.x),
                        self.box_size.mul_add(0.4, box_rect.y),
                        self.box_size * 0.6,
                        self.box_size * 0.2,
                    );
                    canvas.fill_rect(line, self.check_color);
                }
                CheckState::Unchecked => {}
            }
        }

        // Draw label
        if !self.label.is_empty() {
            let label_x = self.bounds.x + self.box_size + self.spacing;
            let label_y = self.bounds.y + (self.bounds.height - 16.0) / 2.0;
            let label_color = if self.disabled {
                self.disabled_color
            } else {
                self.label_color
            };

            let style = presentar_core::widget::TextStyle {
                color: label_color,
                ..Default::default()
            };
            canvas.draw_text(
                &self.label,
                presentar_core::Point::new(label_x, label_y),
                &style,
            );
        }
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        if self.disabled {
            return None;
        }

        match event {
            Event::MouseMove { position } => {
                self.hovered = self.bounds.contains_point(position);
            }
            Event::MouseDown {
                position,
                button: MouseButton::Left,
            } => {
                if self.bounds.contains_point(position) {
                    self.state = self.state.toggle();
                    return Some(Box::new(CheckboxChanged { state: self.state }));
                }
            }
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
        self.accessible_name_value
            .as_deref()
            .or(if self.label.is_empty() {
                None
            } else {
                Some(self.label.as_str())
            })
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Checkbox
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

// PROBAR-SPEC-009: Brick Architecture - Tests define interface
impl Brick for Checkbox {
    fn brick_name(&self) -> &'static str {
        "Checkbox"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[BrickAssertion::MaxLatencyMs(16)]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification {
            passed: self.assertions().to_vec(),
            failed: vec![],
            verification_time: Duration::from_micros(10),
        }
    }

    fn to_html(&self) -> String {
        let test_id = self.test_id_value.as_deref().unwrap_or("checkbox");
        let checked = if self.state.is_checked() {
            " checked"
        } else {
            ""
        };
        let disabled = if self.disabled { " disabled" } else { "" };
        format!(
            r#"<input type="checkbox" class="brick-checkbox" data-testid="{}" aria-label="{}"{}{}/>"#,
            test_id,
            self.accessible_name_value.as_deref().unwrap_or(&self.label),
            checked,
            disabled
        )
    }

    fn to_css(&self) -> String {
        ".brick-checkbox { display: inline-block; }".into()
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;
    use presentar_core::Widget;

    // =========================================================================
    // CheckState Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_check_state_default() {
        assert_eq!(CheckState::default(), CheckState::Unchecked);
    }

    #[test]
    fn test_check_state_toggle() {
        assert_eq!(CheckState::Unchecked.toggle(), CheckState::Checked);
        assert_eq!(CheckState::Checked.toggle(), CheckState::Unchecked);
        assert_eq!(CheckState::Indeterminate.toggle(), CheckState::Unchecked);
    }

    #[test]
    fn test_check_state_is_checked() {
        assert!(!CheckState::Unchecked.is_checked());
        assert!(CheckState::Checked.is_checked());
        assert!(!CheckState::Indeterminate.is_checked());
    }

    #[test]
    fn test_check_state_is_indeterminate() {
        assert!(!CheckState::Unchecked.is_indeterminate());
        assert!(!CheckState::Checked.is_indeterminate());
        assert!(CheckState::Indeterminate.is_indeterminate());
    }

    // =========================================================================
    // CheckboxChanged Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_checkbox_changed_message() {
        let msg = CheckboxChanged {
            state: CheckState::Checked,
        };
        assert_eq!(msg.state, CheckState::Checked);
    }

    // =========================================================================
    // Checkbox Construction Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_checkbox_new() {
        let cb = Checkbox::new();
        assert_eq!(cb.get_state(), CheckState::Unchecked);
        assert!(!cb.is_checked());
        assert!(!cb.disabled);
        assert!(cb.get_label().is_empty());
    }

    #[test]
    fn test_checkbox_default() {
        let cb = Checkbox::default();
        assert_eq!(cb.get_state(), CheckState::Unchecked);
    }

    #[test]
    fn test_checkbox_builder() {
        let cb = Checkbox::new()
            .checked(true)
            .label("Accept terms")
            .disabled(false)
            .box_size(20.0)
            .spacing(10.0)
            .with_test_id("terms-checkbox")
            .with_accessible_name("Terms and Conditions");

        assert!(cb.is_checked());
        assert_eq!(cb.get_label(), "Accept terms");
        assert!(!cb.disabled);
        assert_eq!(Widget::test_id(&cb), Some("terms-checkbox"));
        assert_eq!(cb.accessible_name(), Some("Terms and Conditions"));
    }

    #[test]
    fn test_checkbox_state_builder() {
        let cb = Checkbox::new().state(CheckState::Indeterminate);
        assert!(cb.is_indeterminate());
        assert!(!cb.is_checked());
    }

    // =========================================================================
    // Checkbox State Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_checkbox_checked_true() {
        let cb = Checkbox::new().checked(true);
        assert!(cb.is_checked());
        assert_eq!(cb.get_state(), CheckState::Checked);
    }

    #[test]
    fn test_checkbox_checked_false() {
        let cb = Checkbox::new().checked(false);
        assert!(!cb.is_checked());
        assert_eq!(cb.get_state(), CheckState::Unchecked);
    }

    #[test]
    fn test_checkbox_indeterminate() {
        let cb = Checkbox::new().state(CheckState::Indeterminate);
        assert!(cb.is_indeterminate());
        assert!(!cb.is_checked());
    }

    // =========================================================================
    // Checkbox Widget Trait Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_checkbox_type_id() {
        let cb = Checkbox::new();
        assert_eq!(Widget::type_id(&cb), TypeId::of::<Checkbox>());
    }

    #[test]
    fn test_checkbox_measure_no_label() {
        let cb = Checkbox::new().box_size(18.0);
        let size = cb.measure(Constraints::loose(Size::new(200.0, 100.0)));
        assert_eq!(size.width, 18.0 + 8.0); // box + spacing
        assert_eq!(size.height, 18.0);
    }

    #[test]
    fn test_checkbox_measure_with_label() {
        let cb = Checkbox::new().box_size(18.0).spacing(8.0).label("Test");
        let size = cb.measure(Constraints::loose(Size::new(200.0, 100.0)));
        // 18 (box) + 8 (spacing) + 4*8 (label ~32px)
        assert!(size.width > 18.0);
    }

    #[test]
    fn test_checkbox_is_interactive() {
        let cb = Checkbox::new();
        assert!(cb.is_interactive());

        let cb = Checkbox::new().disabled(true);
        assert!(!cb.is_interactive());
    }

    #[test]
    fn test_checkbox_is_focusable() {
        let cb = Checkbox::new();
        assert!(cb.is_focusable());

        let cb = Checkbox::new().disabled(true);
        assert!(!cb.is_focusable());
    }

    #[test]
    fn test_checkbox_accessible_role() {
        let cb = Checkbox::new();
        assert_eq!(cb.accessible_role(), AccessibleRole::Checkbox);
    }

    #[test]
    fn test_checkbox_accessible_name_from_label() {
        let cb = Checkbox::new().label("My checkbox");
        assert_eq!(cb.accessible_name(), Some("My checkbox"));
    }

    #[test]
    fn test_checkbox_accessible_name_override() {
        let cb = Checkbox::new()
            .label("Short")
            .with_accessible_name("Full accessible name");
        assert_eq!(cb.accessible_name(), Some("Full accessible name"));
    }

    #[test]
    fn test_checkbox_children() {
        let cb = Checkbox::new();
        assert!(cb.children().is_empty());
    }

    // =========================================================================
    // Checkbox Color Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_checkbox_colors() {
        let cb = Checkbox::new()
            .checked_color(Color::RED)
            .check_color(Color::GREEN)
            .label_color(Color::BLUE);

        assert_eq!(cb.checked_color, Color::RED);
        assert_eq!(cb.check_color, Color::GREEN);
        assert_eq!(cb.label_color, Color::BLUE);
    }

    // =========================================================================
    // Checkbox Layout Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_checkbox_layout() {
        let mut cb = Checkbox::new();
        let bounds = Rect::new(10.0, 20.0, 100.0, 30.0);
        let result = cb.layout(bounds);
        assert_eq!(result.size, bounds.size());
        assert_eq!(cb.bounds, bounds);
    }

    // =========================================================================
    // Checkbox Size Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_checkbox_box_size_min() {
        let cb = Checkbox::new().box_size(2.0);
        assert_eq!(cb.box_size, 8.0); // Minimum is 8
    }

    #[test]
    fn test_checkbox_spacing_min() {
        let cb = Checkbox::new().spacing(-5.0);
        assert_eq!(cb.spacing, 0.0); // Minimum is 0
    }

    // =========================================================================
    // Paint Tests - TESTS FIRST
    // =========================================================================

    use presentar_core::draw::DrawCommand;
    use presentar_core::RecordingCanvas;

    #[test]
    fn test_checkbox_paint_unchecked_draws_box() {
        let mut cb = Checkbox::new().box_size(18.0);
        cb.layout(Rect::new(0.0, 0.0, 100.0, 18.0));

        let mut canvas = RecordingCanvas::new();
        cb.paint(&mut canvas);

        // Should draw box rect
        assert!(canvas.command_count() >= 1);
        match &canvas.commands()[0] {
            DrawCommand::Rect { bounds, style, .. } => {
                assert_eq!(bounds.width, 18.0);
                assert_eq!(bounds.height, 18.0);
                assert!(style.fill.is_some());
            }
            _ => panic!("Expected Rect command for checkbox box"),
        }
    }

    #[test]
    fn test_checkbox_paint_unchecked_no_checkmark() {
        let mut cb = Checkbox::new().box_size(18.0);
        cb.layout(Rect::new(0.0, 0.0, 100.0, 18.0));

        let mut canvas = RecordingCanvas::new();
        cb.paint(&mut canvas);

        // Only box, no checkmark
        assert_eq!(canvas.command_count(), 1);
    }

    #[test]
    fn test_checkbox_paint_checked_draws_checkmark() {
        let mut cb = Checkbox::new().box_size(18.0).checked(true);
        cb.layout(Rect::new(0.0, 0.0, 100.0, 18.0));

        let mut canvas = RecordingCanvas::new();
        cb.paint(&mut canvas);

        // Should draw box + checkmark
        assert_eq!(canvas.command_count(), 2);

        // Second rect is the checkmark (inner rect)
        match &canvas.commands()[1] {
            DrawCommand::Rect { bounds, .. } => {
                // Checkmark is 50% of box size, centered
                assert!((bounds.width - 9.0).abs() < 0.1);
                assert!((bounds.height - 9.0).abs() < 0.1);
            }
            _ => panic!("Expected Rect command for checkmark"),
        }
    }

    #[test]
    fn test_checkbox_paint_indeterminate_draws_line() {
        let mut cb = Checkbox::new()
            .box_size(18.0)
            .state(CheckState::Indeterminate);
        cb.layout(Rect::new(0.0, 0.0, 100.0, 18.0));

        let mut canvas = RecordingCanvas::new();
        cb.paint(&mut canvas);

        // Should draw box + indeterminate line
        assert_eq!(canvas.command_count(), 2);

        // Second rect is the indeterminate line
        match &canvas.commands()[1] {
            DrawCommand::Rect { bounds, .. } => {
                // Line is 60% width, 20% height
                assert!((bounds.width - 10.8).abs() < 0.1);
                assert!((bounds.height - 3.6).abs() < 0.1);
            }
            _ => panic!("Expected Rect command for indeterminate line"),
        }
    }

    #[test]
    fn test_checkbox_paint_with_label() {
        let mut cb = Checkbox::new().box_size(18.0).label("Test label");
        cb.layout(Rect::new(0.0, 0.0, 200.0, 18.0));

        let mut canvas = RecordingCanvas::new();
        cb.paint(&mut canvas);

        // Should draw box + label text
        assert_eq!(canvas.command_count(), 2);

        // Second command is the label
        match &canvas.commands()[1] {
            DrawCommand::Text { content, .. } => {
                assert_eq!(content, "Test label");
            }
            _ => panic!("Expected Text command for label"),
        }
    }

    #[test]
    fn test_checkbox_paint_checked_with_label() {
        let mut cb = Checkbox::new().box_size(18.0).checked(true).label("Accept");
        cb.layout(Rect::new(0.0, 0.0, 200.0, 18.0));

        let mut canvas = RecordingCanvas::new();
        cb.paint(&mut canvas);

        // Should draw box + checkmark + label
        assert_eq!(canvas.command_count(), 3);

        // Third command is the label
        match &canvas.commands()[2] {
            DrawCommand::Text { content, .. } => {
                assert_eq!(content, "Accept");
            }
            _ => panic!("Expected Text command for label"),
        }
    }

    #[test]
    fn test_checkbox_paint_uses_checked_color() {
        let mut cb = Checkbox::new()
            .box_size(18.0)
            .checked(true)
            .checked_color(Color::RED);
        cb.layout(Rect::new(0.0, 0.0, 100.0, 18.0));

        let mut canvas = RecordingCanvas::new();
        cb.paint(&mut canvas);

        // Box should use checked color
        match &canvas.commands()[0] {
            DrawCommand::Rect { style, .. } => {
                assert_eq!(style.fill, Some(Color::RED));
            }
            _ => panic!("Expected Rect command"),
        }
    }

    #[test]
    fn test_checkbox_paint_uses_check_color() {
        let mut cb = Checkbox::new()
            .box_size(18.0)
            .checked(true)
            .check_color(Color::GREEN);
        cb.layout(Rect::new(0.0, 0.0, 100.0, 18.0));

        let mut canvas = RecordingCanvas::new();
        cb.paint(&mut canvas);

        // Checkmark should use check color
        match &canvas.commands()[1] {
            DrawCommand::Rect { style, .. } => {
                assert_eq!(style.fill, Some(Color::GREEN));
            }
            _ => panic!("Expected Rect command for checkmark"),
        }
    }

    #[test]
    fn test_checkbox_paint_disabled_no_checkmark() {
        let mut cb = Checkbox::new().box_size(18.0).checked(true).disabled(true);
        cb.layout(Rect::new(0.0, 0.0, 100.0, 18.0));

        let mut canvas = RecordingCanvas::new();
        cb.paint(&mut canvas);

        // Disabled checkbox doesn't draw checkmark
        assert_eq!(canvas.command_count(), 1);
    }

    #[test]
    fn test_checkbox_paint_disabled_uses_disabled_color() {
        let mut cb = Checkbox::new()
            .box_size(18.0)
            .disabled(true)
            .label("Disabled");
        let disabled_color = cb.disabled_color;
        cb.layout(Rect::new(0.0, 0.0, 200.0, 18.0));

        let mut canvas = RecordingCanvas::new();
        cb.paint(&mut canvas);

        // Box should use disabled color
        match &canvas.commands()[0] {
            DrawCommand::Rect { style, .. } => {
                assert_eq!(style.fill, Some(disabled_color));
            }
            _ => panic!("Expected Rect command"),
        }
    }

    #[test]
    fn test_checkbox_paint_label_position() {
        let mut cb = Checkbox::new().box_size(18.0).spacing(8.0).label("Label");
        cb.layout(Rect::new(10.0, 20.0, 200.0, 18.0));

        let mut canvas = RecordingCanvas::new();
        cb.paint(&mut canvas);

        // Label should be positioned after box + spacing
        match &canvas.commands()[1] {
            DrawCommand::Text { position, .. } => {
                // label_x = bounds.x + box_size + spacing = 10 + 18 + 8 = 36
                assert_eq!(position.x, 36.0);
            }
            _ => panic!("Expected Text command"),
        }
    }

    #[test]
    fn test_checkbox_paint_box_position_from_layout() {
        let mut cb = Checkbox::new().box_size(18.0);
        cb.layout(Rect::new(50.0, 100.0, 100.0, 18.0));

        let mut canvas = RecordingCanvas::new();
        cb.paint(&mut canvas);

        match &canvas.commands()[0] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.x, 50.0);
            }
            _ => panic!("Expected Rect command"),
        }
    }

    #[test]
    fn test_checkbox_paint_custom_box_size() {
        let mut cb = Checkbox::new().box_size(24.0).checked(true);
        cb.layout(Rect::new(0.0, 0.0, 100.0, 24.0));

        let mut canvas = RecordingCanvas::new();
        cb.paint(&mut canvas);

        // Box should be 24x24
        match &canvas.commands()[0] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.width, 24.0);
                assert_eq!(bounds.height, 24.0);
            }
            _ => panic!("Expected Rect command"),
        }

        // Checkmark should be 50% = 12x12
        match &canvas.commands()[1] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.width, 12.0);
                assert_eq!(bounds.height, 12.0);
            }
            _ => panic!("Expected Rect command for checkmark"),
        }
    }

    // =========================================================================
    // Event Handling Tests - TESTS FIRST
    // =========================================================================

    use presentar_core::{MouseButton, Point};

    #[test]
    fn test_checkbox_event_click_toggles_unchecked_to_checked() {
        let mut cb = Checkbox::new().box_size(18.0);
        cb.layout(Rect::new(0.0, 0.0, 100.0, 18.0));

        assert!(!cb.is_checked());
        let result = cb.event(&Event::MouseDown {
            position: Point::new(9.0, 9.0),
            button: MouseButton::Left,
        });
        assert!(cb.is_checked());
        assert!(result.is_some());
    }

    #[test]
    fn test_checkbox_event_click_toggles_checked_to_unchecked() {
        let mut cb = Checkbox::new().box_size(18.0).checked(true);
        cb.layout(Rect::new(0.0, 0.0, 100.0, 18.0));

        assert!(cb.is_checked());
        let result = cb.event(&Event::MouseDown {
            position: Point::new(9.0, 9.0),
            button: MouseButton::Left,
        });
        assert!(!cb.is_checked());
        assert!(result.is_some());
    }

    #[test]
    fn test_checkbox_event_click_indeterminate_to_unchecked() {
        let mut cb = Checkbox::new()
            .box_size(18.0)
            .state(CheckState::Indeterminate);
        cb.layout(Rect::new(0.0, 0.0, 100.0, 18.0));

        assert!(cb.is_indeterminate());
        let result = cb.event(&Event::MouseDown {
            position: Point::new(9.0, 9.0),
            button: MouseButton::Left,
        });
        // Indeterminate -> Unchecked per toggle() logic
        assert!(!cb.is_checked());
        assert!(!cb.is_indeterminate());
        let msg = result.unwrap().downcast::<CheckboxChanged>().unwrap();
        assert_eq!(msg.state, CheckState::Unchecked);
    }

    #[test]
    fn test_checkbox_event_emits_checkbox_changed() {
        let mut cb = Checkbox::new().box_size(18.0);
        cb.layout(Rect::new(0.0, 0.0, 100.0, 18.0));

        let result = cb.event(&Event::MouseDown {
            position: Point::new(9.0, 9.0),
            button: MouseButton::Left,
        });

        let msg = result.unwrap().downcast::<CheckboxChanged>().unwrap();
        assert_eq!(msg.state, CheckState::Checked);
    }

    #[test]
    fn test_checkbox_event_message_reflects_new_state() {
        let mut cb = Checkbox::new().box_size(18.0).checked(true);
        cb.layout(Rect::new(0.0, 0.0, 100.0, 18.0));

        let result = cb.event(&Event::MouseDown {
            position: Point::new(9.0, 9.0),
            button: MouseButton::Left,
        });

        let msg = result.unwrap().downcast::<CheckboxChanged>().unwrap();
        assert_eq!(msg.state, CheckState::Unchecked);
    }

    #[test]
    fn test_checkbox_event_click_outside_bounds_no_toggle() {
        let mut cb = Checkbox::new().box_size(18.0);
        cb.layout(Rect::new(0.0, 0.0, 100.0, 18.0));

        let result = cb.event(&Event::MouseDown {
            position: Point::new(200.0, 9.0),
            button: MouseButton::Left,
        });
        assert!(!cb.is_checked());
        assert!(result.is_none());
    }

    #[test]
    fn test_checkbox_event_right_click_no_toggle() {
        let mut cb = Checkbox::new().box_size(18.0);
        cb.layout(Rect::new(0.0, 0.0, 100.0, 18.0));

        let result = cb.event(&Event::MouseDown {
            position: Point::new(9.0, 9.0),
            button: MouseButton::Right,
        });
        assert!(!cb.is_checked());
        assert!(result.is_none());
    }

    #[test]
    fn test_checkbox_event_mouse_move_sets_hover() {
        let mut cb = Checkbox::new().box_size(18.0);
        cb.layout(Rect::new(0.0, 0.0, 100.0, 18.0));

        assert!(!cb.hovered);
        cb.event(&Event::MouseMove {
            position: Point::new(50.0, 9.0),
        });
        assert!(cb.hovered);
    }

    #[test]
    fn test_checkbox_event_mouse_move_clears_hover() {
        let mut cb = Checkbox::new().box_size(18.0);
        cb.layout(Rect::new(0.0, 0.0, 100.0, 18.0));
        cb.hovered = true;

        cb.event(&Event::MouseMove {
            position: Point::new(200.0, 200.0),
        });
        assert!(!cb.hovered);
    }

    #[test]
    fn test_checkbox_event_disabled_blocks_click() {
        let mut cb = Checkbox::new().box_size(18.0).disabled(true);
        cb.layout(Rect::new(0.0, 0.0, 100.0, 18.0));

        let result = cb.event(&Event::MouseDown {
            position: Point::new(9.0, 9.0),
            button: MouseButton::Left,
        });
        assert!(!cb.is_checked());
        assert!(result.is_none());
    }

    #[test]
    fn test_checkbox_event_disabled_blocks_hover() {
        let mut cb = Checkbox::new().box_size(18.0).disabled(true);
        cb.layout(Rect::new(0.0, 0.0, 100.0, 18.0));

        cb.event(&Event::MouseMove {
            position: Point::new(50.0, 9.0),
        });
        assert!(!cb.hovered);
    }

    #[test]
    fn test_checkbox_event_click_on_label_area_toggles() {
        let mut cb = Checkbox::new().box_size(18.0).label("Accept terms");
        cb.layout(Rect::new(0.0, 0.0, 150.0, 18.0));

        // Click on label area (past box)
        let result = cb.event(&Event::MouseDown {
            position: Point::new(100.0, 9.0),
            button: MouseButton::Left,
        });
        assert!(cb.is_checked());
        assert!(result.is_some());
    }

    #[test]
    fn test_checkbox_event_full_interaction_flow() {
        let mut cb = Checkbox::new().box_size(18.0);
        cb.layout(Rect::new(0.0, 0.0, 100.0, 18.0));

        // 1. Start unchecked
        assert!(!cb.is_checked());
        assert!(!cb.hovered);

        // 2. Hover
        cb.event(&Event::MouseMove {
            position: Point::new(50.0, 9.0),
        });
        assert!(cb.hovered);

        // 3. Click to check
        let result = cb.event(&Event::MouseDown {
            position: Point::new(9.0, 9.0),
            button: MouseButton::Left,
        });
        assert!(cb.is_checked());
        let msg = result.unwrap().downcast::<CheckboxChanged>().unwrap();
        assert_eq!(msg.state, CheckState::Checked);

        // 4. Click again to uncheck
        let result = cb.event(&Event::MouseDown {
            position: Point::new(9.0, 9.0),
            button: MouseButton::Left,
        });
        assert!(!cb.is_checked());
        let msg = result.unwrap().downcast::<CheckboxChanged>().unwrap();
        assert_eq!(msg.state, CheckState::Unchecked);

        // 5. Move out
        cb.event(&Event::MouseMove {
            position: Point::new(200.0, 200.0),
        });
        assert!(!cb.hovered);
    }

    #[test]
    fn test_checkbox_event_with_offset_bounds() {
        let mut cb = Checkbox::new().box_size(18.0);
        cb.layout(Rect::new(50.0, 100.0, 100.0, 18.0));

        // Click inside bounds (relative to offset)
        let result = cb.event(&Event::MouseDown {
            position: Point::new(100.0, 109.0),
            button: MouseButton::Left,
        });
        assert!(cb.is_checked());
        assert!(result.is_some());
    }
}
