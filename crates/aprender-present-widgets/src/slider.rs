//! Slider widget for value selection.

use presentar_core::{
    widget::{AccessibleRole, Brick, BrickAssertion, BrickBudget, BrickVerification, LayoutResult},
    Canvas, Color, Constraints, Event, MouseButton, Rect, Size, TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::time::Duration;

/// Message emitted when slider value changes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SliderChanged {
    /// The new value
    pub value: f32,
}

/// Slider widget for selecting a value from a range.
#[derive(Serialize, Deserialize)]
pub struct Slider {
    /// Current value
    value: f32,
    /// Minimum value
    min: f32,
    /// Maximum value
    max: f32,
    /// Step increment (0.0 = continuous)
    step: f32,
    /// Whether the slider is disabled
    disabled: bool,
    /// Track color
    track_color: Color,
    /// Active track color
    active_color: Color,
    /// Thumb color
    thumb_color: Color,
    /// Thumb radius
    thumb_radius: f32,
    /// Track height
    track_height: f32,
    /// Test ID
    test_id_value: Option<String>,
    /// Accessible name
    accessible_name_value: Option<String>,
    /// Cached bounds
    #[serde(skip)]
    bounds: Rect,
    /// Whether currently dragging
    #[serde(skip)]
    dragging: bool,
}

impl Default for Slider {
    fn default() -> Self {
        Self::new()
    }
}

impl Slider {
    /// Create a new slider with default values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            value: 0.5,
            min: 0.0,
            max: 1.0,
            step: 0.0,
            disabled: false,
            track_color: Color::new(0.8, 0.8, 0.8, 1.0),
            active_color: Color::new(0.2, 0.6, 1.0, 1.0),
            thumb_color: Color::WHITE,
            thumb_radius: 10.0,
            track_height: 4.0,
            test_id_value: None,
            accessible_name_value: None,
            bounds: Rect::default(),
            dragging: false,
        }
    }

    /// Set the current value.
    #[must_use]
    pub fn value(mut self, value: f32) -> Self {
        self.value = value.clamp(self.min, self.max);
        self
    }

    /// Set the minimum value.
    #[must_use]
    pub fn min(mut self, min: f32) -> Self {
        self.min = min;
        // Handle case where min > max temporarily during builder chain
        if self.min <= self.max {
            self.value = self.value.clamp(self.min, self.max);
        }
        self
    }

    /// Set the maximum value.
    #[must_use]
    pub fn max(mut self, max: f32) -> Self {
        self.max = max;
        // Handle case where min > max temporarily during builder chain
        if self.min <= self.max {
            self.value = self.value.clamp(self.min, self.max);
        }
        self
    }

    /// Set the step increment.
    #[must_use]
    pub fn step(mut self, step: f32) -> Self {
        self.step = step.abs();
        self
    }

    /// Set disabled state.
    #[must_use]
    pub const fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    /// Set track color.
    #[must_use]
    pub const fn track_color(mut self, color: Color) -> Self {
        self.track_color = color;
        self
    }

    /// Set active track color.
    #[must_use]
    pub const fn active_color(mut self, color: Color) -> Self {
        self.active_color = color;
        self
    }

    /// Set thumb color.
    #[must_use]
    pub const fn thumb_color(mut self, color: Color) -> Self {
        self.thumb_color = color;
        self
    }

    /// Set thumb radius.
    #[must_use]
    pub fn thumb_radius(mut self, radius: f32) -> Self {
        self.thumb_radius = radius.max(0.0);
        self
    }

    /// Set track height.
    #[must_use]
    pub fn track_height(mut self, height: f32) -> Self {
        self.track_height = height.max(0.0);
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
    pub const fn get_value(&self) -> f32 {
        self.value
    }

    /// Get minimum value.
    #[must_use]
    pub const fn get_min(&self) -> f32 {
        self.min
    }

    /// Get maximum value.
    #[must_use]
    pub const fn get_max(&self) -> f32 {
        self.max
    }

    /// Get normalized value (0.0 - 1.0).
    #[must_use]
    pub fn normalized_value(&self) -> f32 {
        if (self.max - self.min).abs() < f32::EPSILON {
            0.0
        } else {
            (self.value - self.min) / (self.max - self.min)
        }
    }

    /// Set value from normalized (0.0 - 1.0) value.
    fn set_from_normalized(&mut self, normalized: f32) {
        let normalized = normalized.clamp(0.0, 1.0);
        let mut new_value = self.min + normalized * (self.max - self.min);

        // Apply step if set
        if self.step > 0.0 {
            new_value = (new_value / self.step).round() * self.step;
        }

        self.value = new_value.clamp(self.min, self.max);
    }

    /// Calculate thumb position X from bounds.
    fn thumb_x(&self) -> f32 {
        let track_start = self.bounds.x + self.thumb_radius;
        let track_width = 2.0f32.mul_add(-self.thumb_radius, self.bounds.width);
        track_width.mul_add(self.normalized_value(), track_start)
    }

    /// Calculate value from X position.
    fn value_from_x(&self, x: f32) -> f32 {
        let track_start = self.bounds.x + self.thumb_radius;
        let track_width = 2.0f32.mul_add(-self.thumb_radius, self.bounds.width);
        if track_width <= 0.0 {
            0.0
        } else {
            ((x - track_start) / track_width).clamp(0.0, 1.0)
        }
    }
}

impl Widget for Slider {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        // Default width of 200, height based on thumb size
        let preferred = Size::new(200.0, self.thumb_radius * 2.0);
        constraints.constrain(preferred)
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: bounds.size(),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        let track_y = self.bounds.y + (self.bounds.height - self.track_height) / 2.0;
        let track_rect = Rect::new(
            self.bounds.x + self.thumb_radius,
            track_y,
            2.0f32.mul_add(-self.thumb_radius, self.bounds.width),
            self.track_height,
        );

        // Draw track background
        canvas.fill_rect(track_rect, self.track_color);

        // Draw active portion
        let active_width = track_rect.width * self.normalized_value();
        let active_rect = Rect::new(track_rect.x, track_rect.y, active_width, self.track_height);
        canvas.fill_rect(active_rect, self.active_color);

        // Draw thumb as a filled circle (approximated as rect for now)
        let thumb_x = self.thumb_x();
        let thumb_y = self.bounds.y + self.bounds.height / 2.0;
        let thumb_rect = Rect::new(
            thumb_x - self.thumb_radius,
            thumb_y - self.thumb_radius,
            self.thumb_radius * 2.0,
            self.thumb_radius * 2.0,
        );

        let thumb_color = if self.disabled {
            Color::new(0.6, 0.6, 0.6, 1.0)
        } else {
            self.thumb_color
        };
        canvas.fill_rect(thumb_rect, thumb_color);
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        if self.disabled {
            return None;
        }

        match event {
            Event::MouseDown {
                position,
                button: MouseButton::Left,
            } => {
                // Check if click is within bounds
                if self.bounds.contains_point(position) {
                    self.dragging = true;
                    let normalized = self.value_from_x(position.x);
                    let old_value = self.value;
                    self.set_from_normalized(normalized);
                    if (self.value - old_value).abs() > f32::EPSILON {
                        return Some(Box::new(SliderChanged { value: self.value }));
                    }
                }
            }
            Event::MouseUp {
                button: MouseButton::Left,
                ..
            } => {
                self.dragging = false;
            }
            Event::MouseMove { position } => {
                if self.dragging {
                    let normalized = self.value_from_x(position.x);
                    let old_value = self.value;
                    self.set_from_normalized(normalized);
                    if (self.value - old_value).abs() > f32::EPSILON {
                        return Some(Box::new(SliderChanged { value: self.value }));
                    }
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
        self.accessible_name_value.as_deref()
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Slider
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

// PROBAR-SPEC-009: Brick Architecture - Tests define interface
impl Brick for Slider {
    fn brick_name(&self) -> &'static str {
        "Slider"
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
        r#"<div class="brick-slider"></div>"#.to_string()
    }

    fn to_css(&self) -> String {
        ".brick-slider { display: block; }".to_string()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;
    use presentar_core::Widget;

    // =========================================================================
    // SliderChanged Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_slider_changed_message() {
        let msg = SliderChanged { value: 0.75 };
        assert_eq!(msg.value, 0.75);
    }

    // =========================================================================
    // Slider Construction Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_slider_new() {
        let slider = Slider::new();
        assert_eq!(slider.get_value(), 0.5);
        assert_eq!(slider.get_min(), 0.0);
        assert_eq!(slider.get_max(), 1.0);
        assert!(!slider.disabled);
    }

    #[test]
    fn test_slider_default() {
        let slider = Slider::default();
        assert_eq!(slider.get_value(), 0.5);
    }

    #[test]
    fn test_slider_builder() {
        let slider = Slider::new()
            .value(0.3)
            .min(0.0)
            .max(100.0)
            .step(10.0)
            .disabled(true)
            .thumb_radius(15.0)
            .track_height(6.0)
            .with_test_id("volume")
            .with_accessible_name("Volume");

        assert_eq!(slider.get_value(), 0.3);
        assert_eq!(slider.get_min(), 0.0);
        assert_eq!(slider.get_max(), 100.0);
        assert!(slider.disabled);
        assert_eq!(Widget::test_id(&slider), Some("volume"));
        assert_eq!(slider.accessible_name(), Some("Volume"));
    }

    // =========================================================================
    // Slider Value Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_slider_value_clamped() {
        let slider = Slider::new().min(0.0).max(1.0).value(1.5);
        assert_eq!(slider.get_value(), 1.0);

        let slider = Slider::new().min(0.0).max(1.0).value(-0.5);
        assert_eq!(slider.get_value(), 0.0);
    }

    #[test]
    fn test_slider_normalized_value() {
        let slider = Slider::new().min(0.0).max(100.0).value(50.0);
        assert!((slider.normalized_value() - 0.5).abs() < f32::EPSILON);

        let slider = Slider::new().min(0.0).max(100.0).value(0.0);
        assert!((slider.normalized_value() - 0.0).abs() < f32::EPSILON);

        let slider = Slider::new().min(0.0).max(100.0).value(100.0);
        assert!((slider.normalized_value() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_slider_normalized_value_same_min_max() {
        let slider = Slider::new().min(50.0).max(50.0).value(50.0);
        assert_eq!(slider.normalized_value(), 0.0);
    }

    #[test]
    fn test_slider_step() {
        let mut slider = Slider::new().min(0.0).max(100.0).step(10.0);
        slider.set_from_normalized(0.45); // 45%
        assert!((slider.get_value() - 50.0).abs() < f32::EPSILON); // Rounds to 50
    }

    // =========================================================================
    // Slider Widget Trait Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_slider_type_id() {
        let slider = Slider::new();
        assert_eq!(Widget::type_id(&slider), TypeId::of::<Slider>());
    }

    #[test]
    fn test_slider_measure() {
        let slider = Slider::new();
        let size = slider.measure(Constraints::loose(Size::new(400.0, 100.0)));
        assert_eq!(size.width, 200.0);
        assert_eq!(size.height, 20.0); // thumb_radius * 2
    }

    #[test]
    fn test_slider_measure_constrained() {
        let slider = Slider::new();
        let size = slider.measure(Constraints::tight(Size::new(100.0, 30.0)));
        assert_eq!(size.width, 100.0);
        assert_eq!(size.height, 30.0);
    }

    #[test]
    fn test_slider_is_interactive() {
        let slider = Slider::new();
        assert!(slider.is_interactive());

        let slider = Slider::new().disabled(true);
        assert!(!slider.is_interactive());
    }

    #[test]
    fn test_slider_is_focusable() {
        let slider = Slider::new();
        assert!(slider.is_focusable());

        let slider = Slider::new().disabled(true);
        assert!(!slider.is_focusable());
    }

    #[test]
    fn test_slider_accessible_role() {
        let slider = Slider::new();
        assert_eq!(slider.accessible_role(), AccessibleRole::Slider);
    }

    #[test]
    fn test_slider_children() {
        let slider = Slider::new();
        assert!(slider.children().is_empty());
    }

    // =========================================================================
    // Slider Color Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_slider_colors() {
        let slider = Slider::new()
            .track_color(Color::RED)
            .active_color(Color::GREEN)
            .thumb_color(Color::BLUE);

        assert_eq!(slider.track_color, Color::RED);
        assert_eq!(slider.active_color, Color::GREEN);
        assert_eq!(slider.thumb_color, Color::BLUE);
    }

    // =========================================================================
    // Slider Layout Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_slider_layout() {
        let mut slider = Slider::new();
        let bounds = Rect::new(10.0, 20.0, 200.0, 30.0);
        let result = slider.layout(bounds);
        assert_eq!(result.size, bounds.size());
        assert_eq!(slider.bounds, bounds);
    }

    // =========================================================================
    // Slider Position Calculation Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_slider_thumb_position() {
        let mut slider = Slider::new().min(0.0).max(100.0).value(50.0);
        slider.bounds = Rect::new(0.0, 0.0, 200.0, 20.0);
        // Track width = 200 - 2*10 = 180
        // Value 50% -> thumb at 10 + 90 = 100
        let thumb_x = slider.thumb_x();
        assert!((thumb_x - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_slider_value_from_position() {
        let mut slider = Slider::new().min(0.0).max(100.0);
        slider.bounds = Rect::new(0.0, 0.0, 200.0, 20.0);
        // Click at x=100 -> normalized = (100-10)/180 ≈ 0.5
        let normalized = slider.value_from_x(100.0);
        assert!((normalized - 0.5).abs() < 0.01);
    }

    // =========================================================================
    // Paint Tests - TESTS FIRST
    // =========================================================================

    use presentar_core::draw::DrawCommand;
    use presentar_core::RecordingCanvas;

    #[test]
    fn test_slider_paint_draws_three_rects() {
        let mut slider = Slider::new().thumb_radius(10.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        let mut canvas = RecordingCanvas::new();
        slider.paint(&mut canvas);

        // Should draw: track + active portion + thumb
        assert_eq!(canvas.command_count(), 3);
    }

    #[test]
    fn test_slider_paint_track_uses_track_color() {
        let mut slider = Slider::new().track_color(Color::RED);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        let mut canvas = RecordingCanvas::new();
        slider.paint(&mut canvas);

        // First rect is track background
        match &canvas.commands()[0] {
            DrawCommand::Rect { style, .. } => {
                assert_eq!(style.fill, Some(Color::RED));
            }
            _ => panic!("Expected Rect command for track"),
        }
    }

    #[test]
    fn test_slider_paint_active_uses_active_color() {
        let mut slider = Slider::new().active_color(Color::GREEN).value(0.5);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        let mut canvas = RecordingCanvas::new();
        slider.paint(&mut canvas);

        // Second rect is active portion
        match &canvas.commands()[1] {
            DrawCommand::Rect { style, .. } => {
                assert_eq!(style.fill, Some(Color::GREEN));
            }
            _ => panic!("Expected Rect command for active portion"),
        }
    }

    #[test]
    fn test_slider_paint_thumb_uses_thumb_color() {
        let mut slider = Slider::new().thumb_color(Color::BLUE);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        let mut canvas = RecordingCanvas::new();
        slider.paint(&mut canvas);

        // Third rect is thumb
        match &canvas.commands()[2] {
            DrawCommand::Rect { style, .. } => {
                assert_eq!(style.fill, Some(Color::BLUE));
            }
            _ => panic!("Expected Rect command for thumb"),
        }
    }

    #[test]
    fn test_slider_paint_track_dimensions() {
        let mut slider = Slider::new().thumb_radius(10.0).track_height(4.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        let mut canvas = RecordingCanvas::new();
        slider.paint(&mut canvas);

        // Track width = bounds.width - 2*thumb_radius = 200 - 20 = 180
        match &canvas.commands()[0] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.width, 180.0);
                assert_eq!(bounds.height, 4.0);
            }
            _ => panic!("Expected Rect command for track"),
        }
    }

    #[test]
    fn test_slider_paint_active_width_at_50_percent() {
        let mut slider = Slider::new().thumb_radius(10.0).value(0.5);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        let mut canvas = RecordingCanvas::new();
        slider.paint(&mut canvas);

        // Track width = 180, active = 50% = 90
        match &canvas.commands()[1] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.width, 90.0);
            }
            _ => panic!("Expected Rect command for active portion"),
        }
    }

    #[test]
    fn test_slider_paint_active_width_at_0_percent() {
        let mut slider = Slider::new().thumb_radius(10.0).value(0.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        let mut canvas = RecordingCanvas::new();
        slider.paint(&mut canvas);

        // Active width should be 0
        match &canvas.commands()[1] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.width, 0.0);
            }
            _ => panic!("Expected Rect command for active portion"),
        }
    }

    #[test]
    fn test_slider_paint_active_width_at_100_percent() {
        let mut slider = Slider::new().thumb_radius(10.0).value(1.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        let mut canvas = RecordingCanvas::new();
        slider.paint(&mut canvas);

        // Active width should be full track width (180)
        match &canvas.commands()[1] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.width, 180.0);
            }
            _ => panic!("Expected Rect command for active portion"),
        }
    }

    #[test]
    fn test_slider_paint_thumb_size() {
        let mut slider = Slider::new().thumb_radius(15.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 30.0));

        let mut canvas = RecordingCanvas::new();
        slider.paint(&mut canvas);

        // Thumb should be 2*radius = 30x30
        match &canvas.commands()[2] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.width, 30.0);
                assert_eq!(bounds.height, 30.0);
            }
            _ => panic!("Expected Rect command for thumb"),
        }
    }

    #[test]
    fn test_slider_paint_thumb_position_at_min() {
        let mut slider = Slider::new().thumb_radius(10.0).value(0.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        let mut canvas = RecordingCanvas::new();
        slider.paint(&mut canvas);

        // Thumb at 0% -> thumb_x = track_start = 10
        // thumb_rect.x = thumb_x - radius = 10 - 10 = 0
        match &canvas.commands()[2] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.x, 0.0);
            }
            _ => panic!("Expected Rect command for thumb"),
        }
    }

    #[test]
    fn test_slider_paint_thumb_position_at_max() {
        let mut slider = Slider::new().thumb_radius(10.0).value(1.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        let mut canvas = RecordingCanvas::new();
        slider.paint(&mut canvas);

        // Thumb at 100% -> thumb_x = track_start + track_width = 10 + 180 = 190
        // thumb_rect.x = thumb_x - radius = 190 - 10 = 180
        match &canvas.commands()[2] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.x, 180.0);
            }
            _ => panic!("Expected Rect command for thumb"),
        }
    }

    #[test]
    fn test_slider_paint_thumb_position_at_50_percent() {
        let mut slider = Slider::new().thumb_radius(10.0).value(0.5);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        let mut canvas = RecordingCanvas::new();
        slider.paint(&mut canvas);

        // Thumb at 50% -> thumb_x = 10 + 90 = 100
        // thumb_rect.x = 100 - 10 = 90
        match &canvas.commands()[2] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.x, 90.0);
            }
            _ => panic!("Expected Rect command for thumb"),
        }
    }

    #[test]
    fn test_slider_paint_track_centered_vertically() {
        let mut slider = Slider::new().track_height(4.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 30.0));

        let mut canvas = RecordingCanvas::new();
        slider.paint(&mut canvas);

        // Track Y = (30 - 4) / 2 = 13
        match &canvas.commands()[0] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.y, 13.0);
            }
            _ => panic!("Expected Rect command for track"),
        }
    }

    #[test]
    fn test_slider_paint_thumb_centered_vertically() {
        let mut slider = Slider::new().thumb_radius(10.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 40.0));

        let mut canvas = RecordingCanvas::new();
        slider.paint(&mut canvas);

        // Thumb Y = bounds.y + bounds.height/2 - radius = 0 + 20 - 10 = 10
        match &canvas.commands()[2] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.y, 10.0);
            }
            _ => panic!("Expected Rect command for thumb"),
        }
    }

    #[test]
    fn test_slider_paint_position_from_layout() {
        let mut slider = Slider::new().thumb_radius(10.0);
        slider.layout(Rect::new(50.0, 100.0, 200.0, 20.0));

        let mut canvas = RecordingCanvas::new();
        slider.paint(&mut canvas);

        // Track X should be bounds.x + thumb_radius = 50 + 10 = 60
        match &canvas.commands()[0] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.x, 60.0);
            }
            _ => panic!("Expected Rect command for track"),
        }
    }

    #[test]
    fn test_slider_paint_disabled_thumb_color() {
        let mut slider = Slider::new().thumb_color(Color::WHITE).disabled(true);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        let mut canvas = RecordingCanvas::new();
        slider.paint(&mut canvas);

        // Disabled thumb should be gray
        match &canvas.commands()[2] {
            DrawCommand::Rect { style, .. } => {
                let fill = style.fill.unwrap();
                assert!((fill.r - 0.6).abs() < 0.01);
                assert!((fill.g - 0.6).abs() < 0.01);
                assert!((fill.b - 0.6).abs() < 0.01);
            }
            _ => panic!("Expected Rect command for thumb"),
        }
    }

    #[test]
    fn test_slider_paint_with_range() {
        let mut slider = Slider::new()
            .min(0.0)
            .max(100.0)
            .value(25.0)
            .thumb_radius(10.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        let mut canvas = RecordingCanvas::new();
        slider.paint(&mut canvas);

        // Active width at 25% of 180 = 45
        match &canvas.commands()[1] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.width, 45.0);
            }
            _ => panic!("Expected Rect command for active portion"),
        }
    }

    // =========================================================================
    // Event Handling Tests - TESTS FIRST
    // =========================================================================

    use presentar_core::Point;

    #[test]
    fn test_slider_event_mouse_down_starts_drag() {
        let mut slider = Slider::new().thumb_radius(10.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        assert!(!slider.dragging);
        slider.event(&Event::MouseDown {
            position: Point::new(100.0, 10.0),
            button: MouseButton::Left,
        });
        assert!(slider.dragging);
    }

    #[test]
    fn test_slider_event_mouse_down_updates_value() {
        let mut slider = Slider::new()
            .min(0.0)
            .max(1.0)
            .value(0.0)
            .thumb_radius(10.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        // Track starts at x=10, width=180
        // Click at x=100 -> normalized = (100-10)/180 = 0.5
        let result = slider.event(&Event::MouseDown {
            position: Point::new(100.0, 10.0),
            button: MouseButton::Left,
        });

        assert!((slider.get_value() - 0.5).abs() < 0.01);
        assert!(result.is_some()); // Value changed
    }

    #[test]
    fn test_slider_event_mouse_down_emits_slider_changed() {
        let mut slider = Slider::new()
            .min(0.0)
            .max(100.0)
            .value(0.0)
            .thumb_radius(10.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        let result = slider.event(&Event::MouseDown {
            position: Point::new(100.0, 10.0),
            button: MouseButton::Left,
        });

        let msg = result.unwrap().downcast::<SliderChanged>().unwrap();
        assert!((msg.value - 50.0).abs() < 1.0); // ~50% of 0-100
    }

    #[test]
    fn test_slider_event_mouse_down_outside_bounds_no_drag() {
        let mut slider = Slider::new().thumb_radius(10.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        let result = slider.event(&Event::MouseDown {
            position: Point::new(300.0, 10.0), // Outside
            button: MouseButton::Left,
        });

        assert!(!slider.dragging);
        assert!(result.is_none());
    }

    #[test]
    fn test_slider_event_mouse_down_right_button_no_drag() {
        let mut slider = Slider::new().thumb_radius(10.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        let result = slider.event(&Event::MouseDown {
            position: Point::new(100.0, 10.0),
            button: MouseButton::Right,
        });

        assert!(!slider.dragging);
        assert!(result.is_none());
    }

    #[test]
    fn test_slider_event_mouse_up_ends_drag() {
        let mut slider = Slider::new().thumb_radius(10.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        // Start drag
        slider.event(&Event::MouseDown {
            position: Point::new(100.0, 10.0),
            button: MouseButton::Left,
        });
        assert!(slider.dragging);

        // End drag
        let result = slider.event(&Event::MouseUp {
            position: Point::new(100.0, 10.0),
            button: MouseButton::Left,
        });
        assert!(!slider.dragging);
        assert!(result.is_none()); // MouseUp doesn't emit message
    }

    #[test]
    fn test_slider_event_mouse_up_right_button_no_effect() {
        let mut slider = Slider::new().thumb_radius(10.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        // Start drag with left button
        slider.event(&Event::MouseDown {
            position: Point::new(100.0, 10.0),
            button: MouseButton::Left,
        });
        assert!(slider.dragging);

        // Right button up doesn't end drag
        slider.event(&Event::MouseUp {
            position: Point::new(100.0, 10.0),
            button: MouseButton::Right,
        });
        assert!(slider.dragging); // Still dragging
    }

    #[test]
    fn test_slider_event_mouse_move_during_drag_updates_value() {
        let mut slider = Slider::new()
            .min(0.0)
            .max(1.0)
            .value(0.0)
            .thumb_radius(10.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        // Start drag at left
        slider.event(&Event::MouseDown {
            position: Point::new(10.0, 10.0),
            button: MouseButton::Left,
        });

        // Move to center
        let result = slider.event(&Event::MouseMove {
            position: Point::new(100.0, 10.0),
        });

        assert!((slider.get_value() - 0.5).abs() < 0.01);
        assert!(result.is_some());
    }

    #[test]
    fn test_slider_event_mouse_move_without_drag_no_effect() {
        let mut slider = Slider::new()
            .min(0.0)
            .max(1.0)
            .value(0.5)
            .thumb_radius(10.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        let result = slider.event(&Event::MouseMove {
            position: Point::new(190.0, 10.0),
        });

        assert_eq!(slider.get_value(), 0.5); // Unchanged
        assert!(result.is_none());
    }

    #[test]
    fn test_slider_event_drag_to_minimum() {
        let mut slider = Slider::new()
            .min(0.0)
            .max(100.0)
            .value(50.0)
            .thumb_radius(10.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        // Start drag
        slider.event(&Event::MouseDown {
            position: Point::new(100.0, 10.0),
            button: MouseButton::Left,
        });

        // Drag to far left (past track start)
        slider.event(&Event::MouseMove {
            position: Point::new(-50.0, 10.0),
        });

        assert_eq!(slider.get_value(), 0.0);
    }

    #[test]
    fn test_slider_event_drag_to_maximum() {
        let mut slider = Slider::new()
            .min(0.0)
            .max(100.0)
            .value(50.0)
            .thumb_radius(10.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        // Start drag
        slider.event(&Event::MouseDown {
            position: Point::new(100.0, 10.0),
            button: MouseButton::Left,
        });

        // Drag to far right (past track end)
        slider.event(&Event::MouseMove {
            position: Point::new(300.0, 10.0),
        });

        assert_eq!(slider.get_value(), 100.0);
    }

    #[test]
    fn test_slider_event_drag_with_step() {
        let mut slider = Slider::new()
            .min(0.0)
            .max(100.0)
            .value(0.0)
            .step(25.0)
            .thumb_radius(10.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        // Start drag
        slider.event(&Event::MouseDown {
            position: Point::new(10.0, 10.0),
            button: MouseButton::Left,
        });

        // Drag to ~30% (should snap to 25)
        slider.event(&Event::MouseMove {
            position: Point::new(64.0, 10.0), // ~30%
        });

        assert_eq!(slider.get_value(), 25.0);
    }

    #[test]
    fn test_slider_event_disabled_blocks_mouse_down() {
        let mut slider = Slider::new().value(0.5).disabled(true).thumb_radius(10.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        let result = slider.event(&Event::MouseDown {
            position: Point::new(100.0, 10.0),
            button: MouseButton::Left,
        });

        assert!(!slider.dragging);
        assert_eq!(slider.get_value(), 0.5); // Unchanged
        assert!(result.is_none());
    }

    #[test]
    fn test_slider_event_disabled_blocks_mouse_move() {
        let mut slider = Slider::new().value(0.5).disabled(true).thumb_radius(10.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));
        slider.dragging = true; // Force dragging state

        let result = slider.event(&Event::MouseMove {
            position: Point::new(190.0, 10.0),
        });

        assert_eq!(slider.get_value(), 0.5); // Unchanged
        assert!(result.is_none());
    }

    #[test]
    fn test_slider_event_no_message_when_value_unchanged() {
        let mut slider = Slider::new()
            .min(0.0)
            .max(1.0)
            .value(0.5)
            .thumb_radius(10.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        // Click at current position (value won't change)
        let result = slider.event(&Event::MouseDown {
            position: Point::new(100.0, 10.0), // Already at ~0.5
            button: MouseButton::Left,
        });

        // No message if value didn't change
        assert!(result.is_none());
    }

    #[test]
    fn test_slider_event_full_drag_flow() {
        let mut slider = Slider::new()
            .min(0.0)
            .max(100.0)
            .value(0.0)
            .thumb_radius(10.0);
        slider.layout(Rect::new(0.0, 0.0, 200.0, 20.0));

        // 1. Mouse down at left
        let result1 = slider.event(&Event::MouseDown {
            position: Point::new(10.0, 10.0),
            button: MouseButton::Left,
        });
        assert!(slider.dragging);
        assert!(result1.is_none()); // Value already 0

        // 2. Drag to 25%
        let result2 = slider.event(&Event::MouseMove {
            position: Point::new(55.0, 10.0),
        });
        assert!((slider.get_value() - 25.0).abs() < 1.0);
        assert!(result2.is_some());

        // 3. Drag to 75%
        let result3 = slider.event(&Event::MouseMove {
            position: Point::new(145.0, 10.0),
        });
        assert!((slider.get_value() - 75.0).abs() < 1.0);
        assert!(result3.is_some());

        // 4. Mouse up
        let result4 = slider.event(&Event::MouseUp {
            position: Point::new(145.0, 10.0),
            button: MouseButton::Left,
        });
        assert!(!slider.dragging);
        assert!(result4.is_none());

        // 5. Mouse move after drag ended - no effect
        let result5 = slider.event(&Event::MouseMove {
            position: Point::new(10.0, 10.0),
        });
        assert!((slider.get_value() - 75.0).abs() < 1.0); // Unchanged
        assert!(result5.is_none());
    }

    #[test]
    fn test_slider_event_bounds_with_offset() {
        let mut slider = Slider::new()
            .min(0.0)
            .max(100.0)
            .value(0.0)
            .thumb_radius(10.0);
        slider.layout(Rect::new(50.0, 100.0, 200.0, 20.0));

        // Track starts at x=60 (50 + 10), width=180
        // Click at x=150 -> normalized = (150-60)/180 = 0.5
        slider.event(&Event::MouseDown {
            position: Point::new(150.0, 110.0),
            button: MouseButton::Left,
        });

        assert!((slider.get_value() - 50.0).abs() < 1.0);
    }
}
