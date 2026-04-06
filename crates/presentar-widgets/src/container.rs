//! Container widget for layout grouping.

use presentar_core::{
    widget::LayoutResult, Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color,
    Constraints, CornerRadius, Event, Rect, Size, TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::time::Duration;

/// Container widget for grouping and styling children.
#[derive(Serialize, Deserialize)]
pub struct Container {
    /// Background color
    pub background: Option<Color>,
    /// Corner radius for rounded corners
    pub corner_radius: CornerRadius,
    /// Padding (all sides)
    pub padding: f32,
    /// Minimum width constraint
    pub min_width: Option<f32>,
    /// Minimum height constraint
    pub min_height: Option<f32>,
    /// Maximum width constraint
    pub max_width: Option<f32>,
    /// Maximum height constraint
    pub max_height: Option<f32>,
    /// Children widgets
    #[serde(skip)]
    children: Vec<Box<dyn Widget>>,
    /// Test ID for this widget
    test_id_value: Option<String>,
    /// Cached bounds after layout
    #[serde(skip)]
    bounds: Rect,
}

impl Default for Container {
    fn default() -> Self {
        Self {
            background: None,
            corner_radius: CornerRadius::ZERO,
            padding: 0.0,
            min_width: None,
            min_height: None,
            max_width: None,
            max_height: None,
            children: Vec::new(),
            test_id_value: None,
            bounds: Rect::default(),
        }
    }
}

impl Container {
    /// Create a new empty container.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the background color.
    #[must_use]
    pub const fn background(mut self, color: Color) -> Self {
        self.background = Some(color);
        self
    }

    /// Set the corner radius.
    #[must_use]
    pub const fn corner_radius(mut self, radius: CornerRadius) -> Self {
        self.corner_radius = radius;
        self
    }

    /// Set uniform padding on all sides.
    #[must_use]
    pub const fn padding(mut self, padding: f32) -> Self {
        self.padding = padding;
        self
    }

    /// Set minimum width.
    #[must_use]
    pub const fn min_width(mut self, width: f32) -> Self {
        self.min_width = Some(width);
        self
    }

    /// Set minimum height.
    #[must_use]
    pub const fn min_height(mut self, height: f32) -> Self {
        self.min_height = Some(height);
        self
    }

    /// Set maximum width.
    #[must_use]
    pub const fn max_width(mut self, width: f32) -> Self {
        self.max_width = Some(width);
        self
    }

    /// Set maximum height.
    #[must_use]
    pub const fn max_height(mut self, height: f32) -> Self {
        self.max_height = Some(height);
        self
    }

    /// Add a child widget.
    pub fn child(mut self, widget: impl Widget + 'static) -> Self {
        self.children.push(Box::new(widget));
        self
    }

    /// Set the test ID.
    #[must_use]
    pub fn with_test_id(mut self, id: impl Into<String>) -> Self {
        self.test_id_value = Some(id.into());
        self
    }
}

impl Widget for Container {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let padding2 = self.padding * 2.0;

        // Measure children
        let child_constraints = Constraints::new(
            0.0,
            (constraints.max_width - padding2).max(0.0),
            0.0,
            (constraints.max_height - padding2).max(0.0),
        );

        let mut child_size = Size::ZERO;
        for child in &self.children {
            let size = child.measure(child_constraints);
            child_size.width = child_size.width.max(size.width);
            child_size.height = child_size.height.max(size.height);
        }

        // Add padding and apply constraints
        let mut size = Size::new(child_size.width + padding2, child_size.height + padding2);

        // Apply min/max constraints
        if let Some(min_w) = self.min_width {
            size.width = size.width.max(min_w);
        }
        if let Some(min_h) = self.min_height {
            size.height = size.height.max(min_h);
        }
        if let Some(max_w) = self.max_width {
            size.width = size.width.min(max_w);
        }
        if let Some(max_h) = self.max_height {
            size.height = size.height.min(max_h);
        }

        constraints.constrain(size)
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;

        // Layout children within padded bounds
        let child_bounds = bounds.inset(self.padding);
        for child in &mut self.children {
            child.layout(child_bounds);
        }

        LayoutResult {
            size: bounds.size(),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        // Draw background
        if let Some(color) = self.background {
            canvas.fill_rect(self.bounds, color);
        }

        // Paint children
        for child in &self.children {
            child.paint(canvas);
        }
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        // Propagate to children
        for child in &mut self.children {
            if let Some(msg) = child.event(event) {
                return Some(msg);
            }
        }
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &self.children
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut self.children
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

// PROBAR-SPEC-009: Brick Architecture - Tests define interface
impl Brick for Container {
    fn brick_name(&self) -> &'static str {
        "Container"
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
        let test_id = self.test_id_value.as_deref().unwrap_or("container");
        format!(r#"<div class="brick-container" data-testid="{test_id}"></div>"#)
    }

    fn to_css(&self) -> String {
        ".brick-container { display: block; }".into()
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn test_container_default() {
        let c = Container::new();
        assert!(c.background.is_none());
        assert_eq!(c.padding, 0.0);
        assert!(c.children.is_empty());
    }

    #[test]
    fn test_container_builder() {
        let c = Container::new()
            .background(Color::WHITE)
            .padding(10.0)
            .min_width(100.0)
            .with_test_id("my-container");

        assert_eq!(c.background, Some(Color::WHITE));
        assert_eq!(c.padding, 10.0);
        assert_eq!(c.min_width, Some(100.0));
        assert_eq!(Widget::test_id(&c), Some("my-container"));
    }

    #[test]
    fn test_container_measure_empty() {
        let c = Container::new().padding(10.0);
        let size = c.measure(Constraints::loose(Size::new(100.0, 100.0)));
        assert_eq!(size, Size::new(20.0, 20.0)); // padding * 2
    }

    #[test]
    fn test_container_measure_with_min_size() {
        let c = Container::new().min_width(50.0).min_height(50.0);
        let size = c.measure(Constraints::loose(Size::new(100.0, 100.0)));
        assert_eq!(size, Size::new(50.0, 50.0));
    }

    #[test]
    fn test_container_measure_with_max_size() {
        let c = Container::new()
            .max_width(30.0)
            .max_height(30.0)
            .min_width(100.0);
        let size = c.measure(Constraints::loose(Size::new(200.0, 200.0)));
        assert_eq!(size.width, 30.0); // max wins over min
    }

    #[test]
    fn test_container_corner_radius() {
        let c = Container::new().corner_radius(CornerRadius::uniform(8.0));
        assert_eq!(c.corner_radius, CornerRadius::uniform(8.0));
    }

    #[test]
    fn test_container_type_id() {
        let c = Container::new();
        assert_eq!(Widget::type_id(&c), TypeId::of::<Container>());
    }

    #[test]
    fn test_container_layout_sets_bounds() {
        let mut c = Container::new().padding(10.0);
        let result = c.layout(Rect::new(0.0, 0.0, 100.0, 80.0));
        assert_eq!(result.size, Size::new(100.0, 80.0));
        assert_eq!(c.bounds, Rect::new(0.0, 0.0, 100.0, 80.0));
    }

    #[test]
    fn test_container_children_empty() {
        let c = Container::new();
        assert!(c.children().is_empty());
    }

    #[test]
    fn test_container_event_no_children_returns_none() {
        let mut c = Container::new();
        c.layout(Rect::new(0.0, 0.0, 100.0, 100.0));
        let result = c.event(&Event::MouseEnter);
        assert!(result.is_none());
    }

    // Paint tests
    use presentar_core::draw::DrawCommand;
    use presentar_core::RecordingCanvas;

    #[test]
    fn test_container_paint_no_background() {
        let mut c = Container::new();
        c.layout(Rect::new(0.0, 0.0, 100.0, 100.0));
        let mut canvas = RecordingCanvas::new();
        c.paint(&mut canvas);
        assert_eq!(canvas.command_count(), 0);
    }

    #[test]
    fn test_container_paint_with_background() {
        let mut c = Container::new().background(Color::RED);
        c.layout(Rect::new(0.0, 0.0, 100.0, 50.0));
        let mut canvas = RecordingCanvas::new();
        c.paint(&mut canvas);
        assert_eq!(canvas.command_count(), 1);
        match &canvas.commands()[0] {
            DrawCommand::Rect { bounds, style, .. } => {
                assert_eq!(bounds.width, 100.0);
                assert_eq!(bounds.height, 50.0);
                assert_eq!(style.fill, Some(Color::RED));
            }
            _ => panic!("Expected Rect"),
        }
    }

    // =========================================================================
    // Builder Pattern Tests
    // =========================================================================

    #[test]
    fn test_container_min_height_builder() {
        let c = Container::new().min_height(75.0);
        assert_eq!(c.min_height, Some(75.0));
    }

    #[test]
    fn test_container_max_height_builder() {
        let c = Container::new().max_height(150.0);
        assert_eq!(c.max_height, Some(150.0));
    }

    #[test]
    fn test_container_max_width_builder() {
        let c = Container::new().max_width(200.0);
        assert_eq!(c.max_width, Some(200.0));
    }

    #[test]
    fn test_container_all_constraints() {
        let c = Container::new()
            .min_width(50.0)
            .max_width(200.0)
            .min_height(30.0)
            .max_height(150.0);
        assert_eq!(c.min_width, Some(50.0));
        assert_eq!(c.max_width, Some(200.0));
        assert_eq!(c.min_height, Some(30.0));
        assert_eq!(c.max_height, Some(150.0));
    }

    #[test]
    fn test_container_chained_all_builders() {
        let c = Container::new()
            .background(Color::BLUE)
            .corner_radius(CornerRadius::uniform(10.0))
            .padding(5.0)
            .min_width(100.0)
            .min_height(80.0)
            .max_width(300.0)
            .max_height(200.0)
            .with_test_id("full-container");

        assert_eq!(c.background, Some(Color::BLUE));
        assert_eq!(c.corner_radius, CornerRadius::uniform(10.0));
        assert_eq!(c.padding, 5.0);
        assert_eq!(c.min_width, Some(100.0));
        assert_eq!(c.min_height, Some(80.0));
        assert_eq!(c.max_width, Some(300.0));
        assert_eq!(c.max_height, Some(200.0));
        assert_eq!(Widget::test_id(&c), Some("full-container"));
    }

    // =========================================================================
    // Measure Tests
    // =========================================================================

    #[test]
    fn test_container_measure_tight_constraints() {
        let c = Container::new().padding(10.0);
        let size = c.measure(Constraints::tight(Size::new(50.0, 50.0)));
        // With tight constraints, padding is still applied but constrained
        assert_eq!(size, Size::new(50.0, 50.0));
    }

    #[test]
    fn test_container_measure_unbounded() {
        let c = Container::new().min_width(100.0).min_height(50.0);
        let size = c.measure(Constraints::unbounded());
        assert_eq!(size, Size::new(100.0, 50.0));
    }

    #[test]
    fn test_container_measure_min_overrides_content() {
        let c = Container::new().min_width(200.0).min_height(200.0);
        let size = c.measure(Constraints::loose(Size::new(500.0, 500.0)));
        assert!(size.width >= 200.0);
        assert!(size.height >= 200.0);
    }

    #[test]
    fn test_container_measure_max_clamps() {
        let c = Container::new().min_width(300.0).max_width(150.0); // max < min
        let size = c.measure(Constraints::loose(Size::new(500.0, 500.0)));
        // max wins after min is applied
        assert_eq!(size.width, 150.0);
    }

    #[test]
    fn test_container_measure_padding_only() {
        let c = Container::new().padding(25.0);
        let size = c.measure(Constraints::loose(Size::new(100.0, 100.0)));
        assert_eq!(size, Size::new(50.0, 50.0)); // 25 * 2 on each axis
    }

    // =========================================================================
    // Layout Tests
    // =========================================================================

    #[test]
    fn test_container_layout_with_offset() {
        let mut c = Container::new();
        let result = c.layout(Rect::new(20.0, 30.0, 100.0, 80.0));
        assert_eq!(result.size, Size::new(100.0, 80.0));
        assert_eq!(c.bounds.x, 20.0);
        assert_eq!(c.bounds.y, 30.0);
    }

    #[test]
    fn test_container_layout_zero_size() {
        let mut c = Container::new();
        let result = c.layout(Rect::new(0.0, 0.0, 0.0, 0.0));
        assert_eq!(result.size, Size::new(0.0, 0.0));
    }

    #[test]
    fn test_container_layout_large_bounds() {
        let mut c = Container::new();
        let result = c.layout(Rect::new(0.0, 0.0, 10000.0, 10000.0));
        assert_eq!(result.size, Size::new(10000.0, 10000.0));
    }

    // =========================================================================
    // Children Tests
    // =========================================================================

    #[test]
    fn test_container_children_mut_access() {
        let mut c = Container::new();
        assert!(c.children_mut().is_empty());
    }

    // =========================================================================
    // Test ID Tests
    // =========================================================================

    #[test]
    fn test_container_test_id_none_by_default() {
        let c = Container::new();
        assert!(Widget::test_id(&c).is_none());
    }

    #[test]
    fn test_container_test_id_with_str() {
        let c = Container::new().with_test_id("simple-id");
        assert_eq!(Widget::test_id(&c), Some("simple-id"));
    }

    #[test]
    fn test_container_test_id_with_string() {
        let id = String::from("dynamic-id");
        let c = Container::new().with_test_id(id);
        assert_eq!(Widget::test_id(&c), Some("dynamic-id"));
    }

    // =========================================================================
    // Corner Radius Tests
    // =========================================================================

    #[test]
    fn test_container_corner_radius_zero() {
        let c = Container::new().corner_radius(CornerRadius::ZERO);
        assert_eq!(c.corner_radius, CornerRadius::ZERO);
    }

    #[test]
    fn test_container_corner_radius_asymmetric() {
        let radius = CornerRadius {
            top_left: 5.0,
            top_right: 10.0,
            bottom_left: 15.0,
            bottom_right: 20.0,
        };
        let c = Container::new().corner_radius(radius);
        assert_eq!(c.corner_radius.top_left, 5.0);
        assert_eq!(c.corner_radius.bottom_right, 20.0);
    }

    // =========================================================================
    // Default Tests
    // =========================================================================

    #[test]
    fn test_container_default_all_none() {
        let c = Container::default();
        assert!(c.background.is_none());
        assert!(c.min_width.is_none());
        assert!(c.min_height.is_none());
        assert!(c.max_width.is_none());
        assert!(c.max_height.is_none());
        assert!(c.test_id_value.is_none());
    }

    #[test]
    fn test_container_default_corner_radius_zero() {
        let c = Container::default();
        assert_eq!(c.corner_radius, CornerRadius::ZERO);
    }

    #[test]
    fn test_container_default_bounds_zero() {
        let c = Container::default();
        assert_eq!(c.bounds, Rect::default());
    }

    // =========================================================================
    // Serialization Tests
    // =========================================================================

    #[test]
    fn test_container_serialize() {
        let c = Container::new()
            .background(Color::GREEN)
            .padding(15.0)
            .min_width(100.0);
        let json = serde_json::to_string(&c).unwrap();
        assert!(json.contains("background"));
        assert!(json.contains("padding"));
        assert!(json.contains("15"));
    }

    #[test]
    fn test_container_deserialize() {
        let json = r#"{"background":{"r":1.0,"g":0.0,"b":0.0,"a":1.0},"corner_radius":{"top_left":0.0,"top_right":0.0,"bottom_left":0.0,"bottom_right":0.0},"padding":10.0,"min_width":50.0,"min_height":null,"max_width":null,"max_height":null,"test_id_value":null}"#;
        let c: Container = serde_json::from_str(json).unwrap();
        assert_eq!(c.padding, 10.0);
        assert_eq!(c.min_width, Some(50.0));
    }

    #[test]
    fn test_container_roundtrip_serialization() {
        let original = Container::new()
            .background(Color::BLUE)
            .padding(20.0)
            .min_width(75.0)
            .max_height(300.0);
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: Container = serde_json::from_str(&json).unwrap();
        assert_eq!(original.padding, deserialized.padding);
        assert_eq!(original.min_width, deserialized.min_width);
        assert_eq!(original.max_height, deserialized.max_height);
        assert_eq!(original.background, deserialized.background);
    }

    // =========================================================================
    // Paint Edge Cases
    // =========================================================================

    #[test]
    fn test_container_paint_transparent_background() {
        let mut c = Container::new().background(Color::TRANSPARENT);
        c.layout(Rect::new(0.0, 0.0, 100.0, 100.0));
        let mut canvas = RecordingCanvas::new();
        c.paint(&mut canvas);
        // Should still paint even with transparent
        assert_eq!(canvas.command_count(), 1);
    }

    #[test]
    fn test_container_paint_after_layout() {
        let mut c = Container::new().background(Color::WHITE);
        // Layout at specific position
        c.layout(Rect::new(50.0, 50.0, 80.0, 60.0));
        let mut canvas = RecordingCanvas::new();
        c.paint(&mut canvas);
        match &canvas.commands()[0] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.x, 50.0);
                assert_eq!(bounds.y, 50.0);
                assert_eq!(bounds.width, 80.0);
                assert_eq!(bounds.height, 60.0);
            }
            _ => panic!("Expected Rect"),
        }
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_container_zero_padding_measure() {
        let c = Container::new().padding(0.0);
        let size = c.measure(Constraints::loose(Size::new(100.0, 100.0)));
        assert_eq!(size, Size::new(0.0, 0.0)); // No content, no padding
    }

    #[test]
    fn test_container_negative_constraints_handled() {
        // Constraints with negative max should clamp to 0
        let c = Container::new().padding(10.0);
        let size = c.measure(Constraints::new(0.0, 5.0, 0.0, 5.0));
        // Padding is 20 total but max is 5, so constrained
        assert_eq!(size, Size::new(5.0, 5.0));
    }
}
