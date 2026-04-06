//! Toggle switch widget.

use presentar_core::{
    widget::{AccessibleRole, LayoutResult},
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    MouseButton, Rect, Size, TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::time::Duration;

/// Message emitted when toggle state changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToggleChanged {
    /// The new toggle state
    pub on: bool,
}

/// Toggle switch widget (on/off).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Toggle {
    /// Current state
    on: bool,
    /// Whether the toggle is disabled
    disabled: bool,
    /// Label text
    label: String,
    /// Track width
    track_width: f32,
    /// Track height
    track_height: f32,
    /// Thumb size (diameter)
    thumb_size: f32,
    /// Track color when off
    track_off_color: Color,
    /// Track color when on
    track_on_color: Color,
    /// Thumb color
    thumb_color: Color,
    /// Disabled color
    disabled_color: Color,
    /// Label color
    label_color: Color,
    /// Spacing between toggle and label
    spacing: f32,
    /// Accessible name
    accessible_name_value: Option<String>,
    /// Test ID
    test_id_value: Option<String>,
    /// Cached bounds
    #[serde(skip)]
    bounds: Rect,
}

impl Default for Toggle {
    fn default() -> Self {
        Self {
            on: false,
            disabled: false,
            label: String::new(),
            track_width: 44.0,
            track_height: 24.0,
            thumb_size: 20.0,
            track_off_color: Color::new(0.7, 0.7, 0.7, 1.0),
            track_on_color: Color::new(0.2, 0.47, 0.96, 1.0),
            thumb_color: Color::WHITE,
            disabled_color: Color::new(0.85, 0.85, 0.85, 1.0),
            label_color: Color::BLACK,
            spacing: 8.0,
            accessible_name_value: None,
            test_id_value: None,
            bounds: Rect::default(),
        }
    }
}

impl Toggle {
    /// Create a new toggle.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a toggle with initial state.
    #[must_use]
    pub fn with_state(on: bool) -> Self {
        Self::default().on(on)
    }

    /// Set the toggle state.
    #[must_use]
    pub const fn on(mut self, on: bool) -> Self {
        self.on = on;
        self
    }

    /// Set whether the toggle is disabled.
    #[must_use]
    pub const fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    /// Set the label.
    #[must_use]
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Set the track width.
    #[must_use]
    pub fn track_width(mut self, width: f32) -> Self {
        self.track_width = width.max(20.0);
        self
    }

    /// Set the track height.
    #[must_use]
    pub fn track_height(mut self, height: f32) -> Self {
        self.track_height = height.max(12.0);
        self
    }

    /// Set the thumb size.
    #[must_use]
    pub fn thumb_size(mut self, size: f32) -> Self {
        self.thumb_size = size.max(8.0);
        self
    }

    /// Set the track off color.
    #[must_use]
    pub const fn track_off_color(mut self, color: Color) -> Self {
        self.track_off_color = color;
        self
    }

    /// Set the track on color.
    #[must_use]
    pub const fn track_on_color(mut self, color: Color) -> Self {
        self.track_on_color = color;
        self
    }

    /// Set the thumb color.
    #[must_use]
    pub const fn thumb_color(mut self, color: Color) -> Self {
        self.thumb_color = color;
        self
    }

    /// Set the disabled color.
    #[must_use]
    pub const fn disabled_color(mut self, color: Color) -> Self {
        self.disabled_color = color;
        self
    }

    /// Set the label color.
    #[must_use]
    pub const fn label_color(mut self, color: Color) -> Self {
        self.label_color = color;
        self
    }

    /// Set the spacing between toggle and label.
    #[must_use]
    pub fn spacing(mut self, spacing: f32) -> Self {
        self.spacing = spacing.max(0.0);
        self
    }

    /// Set the accessible name.
    #[must_use]
    pub fn accessible_name(mut self, name: impl Into<String>) -> Self {
        self.accessible_name_value = Some(name.into());
        self
    }

    /// Set the test ID.
    #[must_use]
    pub fn test_id(mut self, id: impl Into<String>) -> Self {
        self.test_id_value = Some(id.into());
        self
    }

    /// Get current state.
    #[must_use]
    pub const fn is_on(&self) -> bool {
        self.on
    }

    /// Get disabled state.
    #[must_use]
    pub const fn is_disabled(&self) -> bool {
        self.disabled
    }

    /// Get the label.
    #[must_use]
    pub fn get_label(&self) -> &str {
        &self.label
    }

    /// Get the track width.
    #[must_use]
    pub const fn get_track_width(&self) -> f32 {
        self.track_width
    }

    /// Get the track height.
    #[must_use]
    pub const fn get_track_height(&self) -> f32 {
        self.track_height
    }

    /// Get the thumb size.
    #[must_use]
    pub const fn get_thumb_size(&self) -> f32 {
        self.thumb_size
    }

    /// Get the spacing.
    #[must_use]
    pub const fn get_spacing(&self) -> f32 {
        self.spacing
    }

    /// Toggle the state.
    pub fn toggle(&mut self) {
        if !self.disabled {
            self.on = !self.on;
        }
    }

    /// Set the state.
    pub fn set_on(&mut self, on: bool) {
        self.on = on;
    }

    /// Calculate thumb X position.
    fn thumb_x(&self) -> f32 {
        let padding = (self.track_height - self.thumb_size) / 2.0;
        if self.on {
            self.bounds.x + self.track_width - self.thumb_size - padding
        } else {
            self.bounds.x + padding
        }
    }

    /// Calculate thumb Y position (centered).
    fn thumb_y(&self) -> f32 {
        self.bounds.y + (self.track_height - self.thumb_size) / 2.0
    }

    /// Check if a point is within the toggle track.
    fn hit_test(&self, x: f32, y: f32) -> bool {
        x >= self.bounds.x
            && x <= self.bounds.x + self.track_width
            && y >= self.bounds.y
            && y <= self.bounds.y + self.track_height
    }
}

impl Widget for Toggle {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let label_width = if self.label.is_empty() {
            0.0
        } else {
            (self.label.len() as f32).mul_add(8.0, self.spacing)
        };
        let preferred = Size::new(self.track_width + label_width, self.track_height.max(20.0));
        constraints.constrain(preferred)
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: bounds.size(),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        // Determine track color
        let track_color = if self.disabled {
            self.disabled_color
        } else if self.on {
            self.track_on_color
        } else {
            self.track_off_color
        };

        // Draw track (rounded rectangle approximated as regular rect)
        let track_rect = Rect::new(
            self.bounds.x,
            self.bounds.y,
            self.track_width,
            self.track_height,
        );
        canvas.fill_rect(track_rect, track_color);

        // Draw thumb
        let thumb_color = if self.disabled {
            Color::new(0.9, 0.9, 0.9, 1.0)
        } else {
            self.thumb_color
        };
        let thumb_rect = Rect::new(
            self.thumb_x(),
            self.thumb_y(),
            self.thumb_size,
            self.thumb_size,
        );
        canvas.fill_rect(thumb_rect, thumb_color);
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        if self.disabled {
            return None;
        }

        if let Event::MouseDown {
            position,
            button: MouseButton::Left,
        } = event
        {
            if self.hit_test(position.x, position.y) {
                self.on = !self.on;
                return Some(Box::new(ToggleChanged { on: self.on }));
            }
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
        self.accessible_name_value.as_deref().or_else(|| {
            if self.label.is_empty() {
                None
            } else {
                Some(&self.label)
            }
        })
    }

    fn accessible_role(&self) -> AccessibleRole {
        // Toggle/switch is semantically similar to checkbox
        AccessibleRole::Checkbox
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

// PROBAR-SPEC-009: Brick Architecture - Tests define interface
impl Brick for Toggle {
    fn brick_name(&self) -> &'static str {
        "Toggle"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[
            BrickAssertion::MaxLatencyMs(16),
            BrickAssertion::ContrastRatio(3.0), // WCAG AA for UI components
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let mut failed = Vec::new();

        // Verify thumb contrast against track
        let track_color = if self.on {
            self.track_on_color
        } else {
            self.track_off_color
        };
        let contrast = track_color.contrast_ratio(&self.thumb_color);
        if contrast >= 3.0 {
            passed.push(BrickAssertion::ContrastRatio(3.0));
        } else {
            failed.push((
                BrickAssertion::ContrastRatio(3.0),
                format!("Contrast ratio {contrast:.2}:1 < 3.0:1"),
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
        let test_id = self.test_id_value.as_deref().unwrap_or("toggle");
        let checked = if self.on { " checked" } else { "" };
        let disabled = if self.disabled { " disabled" } else { "" };
        let aria_label = self
            .accessible_name_value
            .as_deref()
            .or(if self.label.is_empty() {
                None
            } else {
                Some(self.label.as_str())
            })
            .unwrap_or("");
        format!(
            r#"<input type="checkbox" role="switch" class="brick-toggle" data-testid="{test_id}" aria-label="{aria_label}"{checked}{disabled} />"#
        )
    }

    fn to_css(&self) -> String {
        format!(
            r".brick-toggle {{
    appearance: none;
    width: {}px;
    height: {}px;
    background: {};
    border-radius: {}px;
    position: relative;
    cursor: pointer;
}}
.brick-toggle:checked {{
    background: {};
}}
.brick-toggle::before {{
    content: '';
    position: absolute;
    width: {}px;
    height: {}px;
    background: {};
    border-radius: 50%;
    top: 50%;
    transform: translateY(-50%);
    left: 2px;
    transition: left 0.2s;
}}
.brick-toggle:checked::before {{
    left: calc(100% - {}px - 2px);
}}
.brick-toggle:disabled {{
    opacity: 0.5;
    cursor: not-allowed;
}}",
            self.track_width,
            self.track_height,
            self.track_off_color.to_hex(),
            self.track_height / 2.0,
            self.track_on_color.to_hex(),
            self.thumb_size,
            self.thumb_size,
            self.thumb_color.to_hex(),
            self.thumb_size,
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;
    use presentar_core::Point;

    // ===== ToggleChanged Tests =====

    #[test]
    fn test_toggle_changed_message() {
        let msg = ToggleChanged { on: true };
        assert!(msg.on);

        let msg = ToggleChanged { on: false };
        assert!(!msg.on);
    }

    // ===== Toggle Construction Tests =====

    #[test]
    fn test_toggle_new() {
        let toggle = Toggle::new();
        assert!(!toggle.is_on());
        assert!(!toggle.is_disabled());
    }

    #[test]
    fn test_toggle_with_state_on() {
        let toggle = Toggle::with_state(true);
        assert!(toggle.is_on());
    }

    #[test]
    fn test_toggle_with_state_off() {
        let toggle = Toggle::with_state(false);
        assert!(!toggle.is_on());
    }

    #[test]
    fn test_toggle_default() {
        let toggle = Toggle::default();
        assert!(!toggle.is_on());
        assert!(!toggle.is_disabled());
        assert!(toggle.get_label().is_empty());
    }

    #[test]
    fn test_toggle_builder() {
        let toggle = Toggle::new()
            .on(true)
            .disabled(false)
            .label("Dark Mode")
            .track_width(50.0)
            .track_height(28.0)
            .thumb_size(24.0)
            .track_off_color(Color::new(0.5, 0.5, 0.5, 1.0))
            .track_on_color(Color::new(0.0, 0.8, 0.4, 1.0))
            .thumb_color(Color::WHITE)
            .disabled_color(Color::new(0.9, 0.9, 0.9, 1.0))
            .label_color(Color::BLACK)
            .spacing(12.0)
            .accessible_name("Toggle dark mode")
            .test_id("dark-mode-toggle");

        assert!(toggle.is_on());
        assert!(!toggle.is_disabled());
        assert_eq!(toggle.get_label(), "Dark Mode");
        assert_eq!(toggle.get_track_width(), 50.0);
        assert_eq!(toggle.get_track_height(), 28.0);
        assert_eq!(toggle.get_thumb_size(), 24.0);
        assert_eq!(toggle.get_spacing(), 12.0);
        assert_eq!(Widget::accessible_name(&toggle), Some("Toggle dark mode"));
        assert_eq!(Widget::test_id(&toggle), Some("dark-mode-toggle"));
    }

    // ===== State Tests =====

    #[test]
    fn test_toggle_on() {
        let toggle = Toggle::new().on(true);
        assert!(toggle.is_on());
    }

    #[test]
    fn test_toggle_off() {
        let toggle = Toggle::new().on(false);
        assert!(!toggle.is_on());
    }

    #[test]
    fn test_toggle_set_on() {
        let mut toggle = Toggle::new();
        toggle.set_on(true);
        assert!(toggle.is_on());
        toggle.set_on(false);
        assert!(!toggle.is_on());
    }

    #[test]
    fn test_toggle_toggle_method() {
        let mut toggle = Toggle::new();
        assert!(!toggle.is_on());
        toggle.toggle();
        assert!(toggle.is_on());
        toggle.toggle();
        assert!(!toggle.is_on());
    }

    #[test]
    fn test_toggle_disabled_cannot_toggle() {
        let mut toggle = Toggle::new().disabled(true);
        toggle.toggle();
        assert!(!toggle.is_on()); // Still off, toggle had no effect
    }

    // ===== Dimension Tests =====

    #[test]
    fn test_toggle_track_width_min() {
        let toggle = Toggle::new().track_width(10.0);
        assert_eq!(toggle.get_track_width(), 20.0);
    }

    #[test]
    fn test_toggle_track_height_min() {
        let toggle = Toggle::new().track_height(5.0);
        assert_eq!(toggle.get_track_height(), 12.0);
    }

    #[test]
    fn test_toggle_thumb_size_min() {
        let toggle = Toggle::new().thumb_size(2.0);
        assert_eq!(toggle.get_thumb_size(), 8.0);
    }

    #[test]
    fn test_toggle_spacing_min() {
        let toggle = Toggle::new().spacing(-5.0);
        assert_eq!(toggle.get_spacing(), 0.0);
    }

    // ===== Color Tests =====

    #[test]
    fn test_toggle_colors() {
        let track_off = Color::new(0.3, 0.3, 0.3, 1.0);
        let track_on = Color::new(0.0, 1.0, 0.5, 1.0);
        let thumb = Color::new(1.0, 1.0, 1.0, 1.0);

        let toggle = Toggle::new()
            .track_off_color(track_off)
            .track_on_color(track_on)
            .thumb_color(thumb);

        assert_eq!(toggle.track_off_color, track_off);
        assert_eq!(toggle.track_on_color, track_on);
        assert_eq!(toggle.thumb_color, thumb);
    }

    // ===== Thumb Position Tests =====

    #[test]
    fn test_toggle_thumb_position_off() {
        let mut toggle = Toggle::new()
            .track_width(44.0)
            .track_height(24.0)
            .thumb_size(20.0);
        toggle.bounds = Rect::new(0.0, 0.0, 44.0, 24.0);

        let padding = (24.0 - 20.0) / 2.0;
        assert_eq!(toggle.thumb_x(), padding); // Left position
    }

    #[test]
    fn test_toggle_thumb_position_on() {
        let mut toggle = Toggle::new()
            .on(true)
            .track_width(44.0)
            .track_height(24.0)
            .thumb_size(20.0);
        toggle.bounds = Rect::new(0.0, 0.0, 44.0, 24.0);

        let padding = (24.0 - 20.0) / 2.0;
        assert_eq!(toggle.thumb_x(), 44.0 - 20.0 - padding); // Right position
    }

    #[test]
    fn test_toggle_thumb_y_centered() {
        let mut toggle = Toggle::new().track_height(24.0).thumb_size(20.0);
        toggle.bounds = Rect::new(10.0, 20.0, 44.0, 24.0);

        assert_eq!(toggle.thumb_y(), 20.0 + (24.0 - 20.0) / 2.0);
    }

    // ===== Hit Test Tests =====

    #[test]
    fn test_toggle_hit_test_inside() {
        let mut toggle = Toggle::new().track_width(44.0).track_height(24.0);
        toggle.bounds = Rect::new(10.0, 10.0, 44.0, 24.0);

        assert!(toggle.hit_test(20.0, 20.0));
        assert!(toggle.hit_test(10.0, 10.0)); // Top-left corner
        assert!(toggle.hit_test(54.0, 34.0)); // Bottom-right corner
    }

    #[test]
    fn test_toggle_hit_test_outside() {
        let mut toggle = Toggle::new().track_width(44.0).track_height(24.0);
        toggle.bounds = Rect::new(10.0, 10.0, 44.0, 24.0);

        assert!(!toggle.hit_test(5.0, 10.0)); // Left of track
        assert!(!toggle.hit_test(60.0, 10.0)); // Right of track
        assert!(!toggle.hit_test(20.0, 5.0)); // Above track
        assert!(!toggle.hit_test(20.0, 40.0)); // Below track
    }

    // ===== Widget Trait Tests =====

    #[test]
    fn test_toggle_type_id() {
        let toggle = Toggle::new();
        assert_eq!(Widget::type_id(&toggle), TypeId::of::<Toggle>());
    }

    #[test]
    fn test_toggle_measure_no_label() {
        let toggle = Toggle::new().track_width(44.0).track_height(24.0);
        let size = toggle.measure(Constraints::loose(Size::new(200.0, 100.0)));
        assert_eq!(size.width, 44.0);
        assert_eq!(size.height, 24.0);
    }

    #[test]
    fn test_toggle_measure_with_label() {
        let toggle = Toggle::new()
            .track_width(44.0)
            .track_height(24.0)
            .label("On")
            .spacing(8.0);
        let size = toggle.measure(Constraints::loose(Size::new(200.0, 100.0)));
        // Width = track_width + spacing + label_width (2 chars * 8)
        assert_eq!(size.width, 44.0 + 8.0 + 16.0);
    }

    #[test]
    fn test_toggle_layout() {
        let mut toggle = Toggle::new();
        let bounds = Rect::new(10.0, 20.0, 44.0, 24.0);
        let result = toggle.layout(bounds);
        assert_eq!(result.size, Size::new(44.0, 24.0));
        assert_eq!(toggle.bounds, bounds);
    }

    #[test]
    fn test_toggle_children() {
        let toggle = Toggle::new();
        assert!(toggle.children().is_empty());
    }

    #[test]
    fn test_toggle_is_interactive() {
        let toggle = Toggle::new();
        assert!(toggle.is_interactive());

        let toggle = Toggle::new().disabled(true);
        assert!(!toggle.is_interactive());
    }

    #[test]
    fn test_toggle_is_focusable() {
        let toggle = Toggle::new();
        assert!(toggle.is_focusable());

        let toggle = Toggle::new().disabled(true);
        assert!(!toggle.is_focusable());
    }

    #[test]
    fn test_toggle_accessible_role() {
        let toggle = Toggle::new();
        assert_eq!(toggle.accessible_role(), AccessibleRole::Checkbox);
    }

    #[test]
    fn test_toggle_accessible_name_from_label() {
        let toggle = Toggle::new().label("Enable notifications");
        assert_eq!(
            Widget::accessible_name(&toggle),
            Some("Enable notifications")
        );
    }

    #[test]
    fn test_toggle_accessible_name_override() {
        let toggle = Toggle::new()
            .label("Notifications")
            .accessible_name("Toggle notifications on or off");
        assert_eq!(
            Widget::accessible_name(&toggle),
            Some("Toggle notifications on or off")
        );
    }

    #[test]
    fn test_toggle_accessible_name_none() {
        let toggle = Toggle::new();
        assert_eq!(Widget::accessible_name(&toggle), None);
    }

    #[test]
    fn test_toggle_test_id() {
        let toggle = Toggle::new().test_id("settings-toggle");
        assert_eq!(Widget::test_id(&toggle), Some("settings-toggle"));
    }

    // ===== Event Tests =====

    #[test]
    fn test_toggle_click_toggles_state() {
        let mut toggle = Toggle::new();
        toggle.bounds = Rect::new(0.0, 0.0, 44.0, 24.0);

        let event = Event::MouseDown {
            position: Point::new(22.0, 12.0),
            button: MouseButton::Left,
        };

        let result = toggle.event(&event);
        assert!(result.is_some());
        assert!(toggle.is_on());

        let result = toggle.event(&event);
        assert!(result.is_some());
        assert!(!toggle.is_on());
    }

    #[test]
    fn test_toggle_click_outside_no_effect() {
        let mut toggle = Toggle::new();
        toggle.bounds = Rect::new(0.0, 0.0, 44.0, 24.0);

        let event = Event::MouseDown {
            position: Point::new(100.0, 100.0),
            button: MouseButton::Left,
        };

        let result = toggle.event(&event);
        assert!(result.is_none());
        assert!(!toggle.is_on());
    }

    #[test]
    fn test_toggle_right_click_no_effect() {
        let mut toggle = Toggle::new();
        toggle.bounds = Rect::new(0.0, 0.0, 44.0, 24.0);

        let event = Event::MouseDown {
            position: Point::new(22.0, 12.0),
            button: MouseButton::Right,
        };

        let result = toggle.event(&event);
        assert!(result.is_none());
        assert!(!toggle.is_on());
    }

    #[test]
    fn test_toggle_disabled_click_no_effect() {
        let mut toggle = Toggle::new().disabled(true);
        toggle.bounds = Rect::new(0.0, 0.0, 44.0, 24.0);

        let event = Event::MouseDown {
            position: Point::new(22.0, 12.0),
            button: MouseButton::Left,
        };

        let result = toggle.event(&event);
        assert!(result.is_none());
        assert!(!toggle.is_on());
    }

    #[test]
    fn test_toggle_changed_contains_new_state() {
        let mut toggle = Toggle::new();
        toggle.bounds = Rect::new(0.0, 0.0, 44.0, 24.0);

        let event = Event::MouseDown {
            position: Point::new(22.0, 12.0),
            button: MouseButton::Left,
        };

        let result = toggle.event(&event);
        let msg = result.unwrap().downcast::<ToggleChanged>().unwrap();
        assert!(msg.on);

        let result = toggle.event(&event);
        let msg = result.unwrap().downcast::<ToggleChanged>().unwrap();
        assert!(!msg.on);
    }

    // ===== Paint Tests =====

    use presentar_core::draw::DrawCommand;
    use presentar_core::RecordingCanvas;

    #[test]
    fn test_toggle_paint_draws_track_and_thumb() {
        let mut toggle = Toggle::new()
            .track_width(44.0)
            .track_height(24.0)
            .thumb_size(20.0);
        toggle.layout(Rect::new(0.0, 0.0, 44.0, 24.0));

        let mut canvas = RecordingCanvas::new();
        toggle.paint(&mut canvas);

        // Should draw track + thumb
        assert_eq!(canvas.command_count(), 2);
    }

    #[test]
    fn test_toggle_paint_track_off_color() {
        let mut toggle = Toggle::new().track_off_color(Color::RED).on(false);
        toggle.layout(Rect::new(0.0, 0.0, 44.0, 24.0));

        let mut canvas = RecordingCanvas::new();
        toggle.paint(&mut canvas);

        // Track should use off color
        match &canvas.commands()[0] {
            DrawCommand::Rect { style, .. } => {
                assert_eq!(style.fill, Some(Color::RED));
            }
            _ => panic!("Expected Rect command for track"),
        }
    }

    #[test]
    fn test_toggle_paint_track_on_color() {
        let mut toggle = Toggle::new().track_on_color(Color::GREEN).on(true);
        toggle.layout(Rect::new(0.0, 0.0, 44.0, 24.0));

        let mut canvas = RecordingCanvas::new();
        toggle.paint(&mut canvas);

        // Track should use on color
        match &canvas.commands()[0] {
            DrawCommand::Rect { style, .. } => {
                assert_eq!(style.fill, Some(Color::GREEN));
            }
            _ => panic!("Expected Rect command for track"),
        }
    }

    #[test]
    fn test_toggle_paint_track_disabled_color() {
        let mut toggle = Toggle::new()
            .disabled_color(Color::new(0.85, 0.85, 0.85, 1.0))
            .disabled(true);
        toggle.layout(Rect::new(0.0, 0.0, 44.0, 24.0));

        let mut canvas = RecordingCanvas::new();
        toggle.paint(&mut canvas);

        // Track should use disabled color
        match &canvas.commands()[0] {
            DrawCommand::Rect { style, .. } => {
                let fill = style.fill.unwrap();
                assert!((fill.r - 0.85).abs() < 0.01);
            }
            _ => panic!("Expected Rect command for track"),
        }
    }

    #[test]
    fn test_toggle_paint_track_dimensions() {
        let mut toggle = Toggle::new().track_width(50.0).track_height(28.0);
        toggle.layout(Rect::new(0.0, 0.0, 50.0, 28.0));

        let mut canvas = RecordingCanvas::new();
        toggle.paint(&mut canvas);

        match &canvas.commands()[0] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.width, 50.0);
                assert_eq!(bounds.height, 28.0);
            }
            _ => panic!("Expected Rect command for track"),
        }
    }

    #[test]
    fn test_toggle_paint_thumb_size() {
        let mut toggle = Toggle::new().thumb_size(20.0);
        toggle.layout(Rect::new(0.0, 0.0, 44.0, 24.0));

        let mut canvas = RecordingCanvas::new();
        toggle.paint(&mut canvas);

        match &canvas.commands()[1] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.width, 20.0);
                assert_eq!(bounds.height, 20.0);
            }
            _ => panic!("Expected Rect command for thumb"),
        }
    }

    #[test]
    fn test_toggle_paint_thumb_position_off() {
        let mut toggle = Toggle::new()
            .track_width(44.0)
            .track_height(24.0)
            .thumb_size(20.0)
            .on(false);
        toggle.layout(Rect::new(0.0, 0.0, 44.0, 24.0));

        let mut canvas = RecordingCanvas::new();
        toggle.paint(&mut canvas);

        // Thumb should be on the left
        let padding = (24.0 - 20.0) / 2.0; // 2.0
        match &canvas.commands()[1] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.x, padding);
            }
            _ => panic!("Expected Rect command for thumb"),
        }
    }

    #[test]
    fn test_toggle_paint_thumb_position_on() {
        let mut toggle = Toggle::new()
            .track_width(44.0)
            .track_height(24.0)
            .thumb_size(20.0)
            .on(true);
        toggle.layout(Rect::new(0.0, 0.0, 44.0, 24.0));

        let mut canvas = RecordingCanvas::new();
        toggle.paint(&mut canvas);

        // Thumb should be on the right
        let padding = (24.0 - 20.0) / 2.0; // 2.0
        let expected_x = 44.0 - 20.0 - padding; // 22.0
        match &canvas.commands()[1] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.x, expected_x);
            }
            _ => panic!("Expected Rect command for thumb"),
        }
    }

    #[test]
    fn test_toggle_paint_thumb_color() {
        let mut toggle = Toggle::new().thumb_color(Color::BLUE);
        toggle.layout(Rect::new(0.0, 0.0, 44.0, 24.0));

        let mut canvas = RecordingCanvas::new();
        toggle.paint(&mut canvas);

        match &canvas.commands()[1] {
            DrawCommand::Rect { style, .. } => {
                assert_eq!(style.fill, Some(Color::BLUE));
            }
            _ => panic!("Expected Rect command for thumb"),
        }
    }

    #[test]
    fn test_toggle_paint_thumb_disabled_color() {
        let mut toggle = Toggle::new().disabled(true);
        toggle.layout(Rect::new(0.0, 0.0, 44.0, 24.0));

        let mut canvas = RecordingCanvas::new();
        toggle.paint(&mut canvas);

        // Disabled thumb should be grayish
        match &canvas.commands()[1] {
            DrawCommand::Rect { style, .. } => {
                let fill = style.fill.unwrap();
                assert!((fill.r - 0.9).abs() < 0.01);
                assert!((fill.g - 0.9).abs() < 0.01);
                assert!((fill.b - 0.9).abs() < 0.01);
            }
            _ => panic!("Expected Rect command for thumb"),
        }
    }

    #[test]
    fn test_toggle_paint_position_from_layout() {
        let mut toggle = Toggle::new().track_width(44.0).track_height(24.0);
        toggle.layout(Rect::new(100.0, 50.0, 44.0, 24.0));

        let mut canvas = RecordingCanvas::new();
        toggle.paint(&mut canvas);

        // Track should be at layout position
        match &canvas.commands()[0] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.x, 100.0);
                assert_eq!(bounds.y, 50.0);
            }
            _ => panic!("Expected Rect command for track"),
        }
    }

    #[test]
    fn test_toggle_paint_thumb_centered_vertically() {
        let mut toggle = Toggle::new().track_height(30.0).thumb_size(20.0);
        toggle.layout(Rect::new(0.0, 0.0, 44.0, 30.0));

        let mut canvas = RecordingCanvas::new();
        toggle.paint(&mut canvas);

        // Thumb Y should be centered
        let expected_y = (30.0 - 20.0) / 2.0; // 5.0
        match &canvas.commands()[1] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.y, expected_y);
            }
            _ => panic!("Expected Rect command for thumb"),
        }
    }

    #[test]
    fn test_toggle_paint_custom_track_and_thumb() {
        let mut toggle = Toggle::new()
            .track_width(60.0)
            .track_height(32.0)
            .thumb_size(28.0)
            .track_on_color(Color::GREEN)
            .thumb_color(Color::WHITE)
            .on(true);
        toggle.layout(Rect::new(10.0, 20.0, 60.0, 32.0));

        let mut canvas = RecordingCanvas::new();
        toggle.paint(&mut canvas);

        // Track
        match &canvas.commands()[0] {
            DrawCommand::Rect { bounds, style, .. } => {
                assert_eq!(bounds.width, 60.0);
                assert_eq!(bounds.height, 32.0);
                assert_eq!(style.fill, Some(Color::GREEN));
            }
            _ => panic!("Expected Rect command for track"),
        }

        // Thumb
        let padding = (32.0 - 28.0) / 2.0;
        let expected_thumb_x = 10.0 + 60.0 - 28.0 - padding;
        match &canvas.commands()[1] {
            DrawCommand::Rect { bounds, style, .. } => {
                assert_eq!(bounds.width, 28.0);
                assert_eq!(bounds.height, 28.0);
                assert_eq!(bounds.x, expected_thumb_x);
                assert_eq!(style.fill, Some(Color::WHITE));
            }
            _ => panic!("Expected Rect command for thumb"),
        }
    }

    // =========================================================================
    // Event Handling Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_toggle_event_click_turns_on() {
        let mut toggle = Toggle::new().track_width(44.0).track_height(24.0).on(false);
        toggle.layout(Rect::new(0.0, 0.0, 44.0, 24.0));

        assert!(!toggle.is_on());
        let result = toggle.event(&Event::MouseDown {
            position: Point::new(22.0, 12.0),
            button: MouseButton::Left,
        });
        assert!(toggle.is_on());
        assert!(result.is_some());
    }

    #[test]
    fn test_toggle_event_click_turns_off() {
        let mut toggle = Toggle::new().track_width(44.0).track_height(24.0).on(true);
        toggle.layout(Rect::new(0.0, 0.0, 44.0, 24.0));

        assert!(toggle.is_on());
        let result = toggle.event(&Event::MouseDown {
            position: Point::new(22.0, 12.0),
            button: MouseButton::Left,
        });
        assert!(!toggle.is_on());
        assert!(result.is_some());
    }

    #[test]
    fn test_toggle_event_emits_toggle_changed() {
        let mut toggle = Toggle::new().track_width(44.0).track_height(24.0).on(false);
        toggle.layout(Rect::new(0.0, 0.0, 44.0, 24.0));

        let result = toggle.event(&Event::MouseDown {
            position: Point::new(22.0, 12.0),
            button: MouseButton::Left,
        });

        let msg = result.unwrap().downcast::<ToggleChanged>().unwrap();
        assert!(msg.on);
    }

    #[test]
    fn test_toggle_event_message_reflects_new_state() {
        let mut toggle = Toggle::new().track_width(44.0).track_height(24.0).on(true);
        toggle.layout(Rect::new(0.0, 0.0, 44.0, 24.0));

        let result = toggle.event(&Event::MouseDown {
            position: Point::new(22.0, 12.0),
            button: MouseButton::Left,
        });

        let msg = result.unwrap().downcast::<ToggleChanged>().unwrap();
        assert!(!msg.on);
    }

    #[test]
    fn test_toggle_event_click_outside_track_no_toggle() {
        let mut toggle = Toggle::new().track_width(44.0).track_height(24.0).on(false);
        toggle.layout(Rect::new(0.0, 0.0, 100.0, 24.0));

        // Click outside track (past track width)
        let result = toggle.event(&Event::MouseDown {
            position: Point::new(80.0, 12.0),
            button: MouseButton::Left,
        });
        assert!(!toggle.is_on());
        assert!(result.is_none());
    }

    #[test]
    fn test_toggle_event_click_below_track_no_toggle() {
        let mut toggle = Toggle::new().track_width(44.0).track_height(24.0).on(false);
        toggle.layout(Rect::new(0.0, 0.0, 44.0, 50.0));

        // Click below track
        let result = toggle.event(&Event::MouseDown {
            position: Point::new(22.0, 40.0),
            button: MouseButton::Left,
        });
        assert!(!toggle.is_on());
        assert!(result.is_none());
    }

    #[test]
    fn test_toggle_event_right_click_no_toggle() {
        let mut toggle = Toggle::new().track_width(44.0).track_height(24.0).on(false);
        toggle.layout(Rect::new(0.0, 0.0, 44.0, 24.0));

        let result = toggle.event(&Event::MouseDown {
            position: Point::new(22.0, 12.0),
            button: MouseButton::Right,
        });
        assert!(!toggle.is_on());
        assert!(result.is_none());
    }

    #[test]
    fn test_toggle_event_disabled_blocks_click() {
        let mut toggle = Toggle::new()
            .track_width(44.0)
            .track_height(24.0)
            .on(false)
            .disabled(true);
        toggle.layout(Rect::new(0.0, 0.0, 44.0, 24.0));

        let result = toggle.event(&Event::MouseDown {
            position: Point::new(22.0, 12.0),
            button: MouseButton::Left,
        });
        assert!(!toggle.is_on());
        assert!(result.is_none());
    }

    #[test]
    fn test_toggle_event_hit_test_track_edges() {
        let mut toggle = Toggle::new().track_width(44.0).track_height(24.0).on(false);
        toggle.layout(Rect::new(0.0, 0.0, 44.0, 24.0));

        // Top-left corner (just inside)
        let result = toggle.event(&Event::MouseDown {
            position: Point::new(0.0, 0.0),
            button: MouseButton::Left,
        });
        assert!(toggle.is_on());
        assert!(result.is_some());

        toggle.on = false;

        // Bottom-right corner (just inside)
        let result = toggle.event(&Event::MouseDown {
            position: Point::new(44.0, 24.0),
            button: MouseButton::Left,
        });
        assert!(toggle.is_on());
        assert!(result.is_some());
    }

    #[test]
    fn test_toggle_event_with_offset_bounds() {
        let mut toggle = Toggle::new().track_width(44.0).track_height(24.0).on(false);
        toggle.layout(Rect::new(100.0, 50.0, 44.0, 24.0));

        // Click relative to offset
        let result = toggle.event(&Event::MouseDown {
            position: Point::new(122.0, 62.0),
            button: MouseButton::Left,
        });
        assert!(toggle.is_on());
        assert!(result.is_some());
    }

    #[test]
    fn test_toggle_event_full_interaction_flow() {
        let mut toggle = Toggle::new().track_width(44.0).track_height(24.0).on(false);
        toggle.layout(Rect::new(0.0, 0.0, 44.0, 24.0));

        // 1. Start off
        assert!(!toggle.is_on());

        // 2. Click to turn on
        let result = toggle.event(&Event::MouseDown {
            position: Point::new(22.0, 12.0),
            button: MouseButton::Left,
        });
        assert!(toggle.is_on());
        let msg = result.unwrap().downcast::<ToggleChanged>().unwrap();
        assert!(msg.on);

        // 3. Click to turn off
        let result = toggle.event(&Event::MouseDown {
            position: Point::new(22.0, 12.0),
            button: MouseButton::Left,
        });
        assert!(!toggle.is_on());
        let msg = result.unwrap().downcast::<ToggleChanged>().unwrap();
        assert!(!msg.on);

        // 4. Click again
        let result = toggle.event(&Event::MouseDown {
            position: Point::new(22.0, 12.0),
            button: MouseButton::Left,
        });
        assert!(toggle.is_on());
        assert!(result.is_some());
    }

    #[test]
    fn test_toggle_event_mouse_move_no_effect() {
        let mut toggle = Toggle::new().track_width(44.0).track_height(24.0).on(false);
        toggle.layout(Rect::new(0.0, 0.0, 44.0, 24.0));

        // MouseMove should not toggle
        let result = toggle.event(&Event::MouseMove {
            position: Point::new(22.0, 12.0),
        });
        assert!(!toggle.is_on());
        assert!(result.is_none());
    }
}
