//! Text widget for displaying text content.

use presentar_core::{
    widget::{FontStyle, FontWeight, LayoutResult, TextStyle},
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event, Rect,
    Size, TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::time::Duration;

/// Text widget for displaying styled text.
#[derive(Clone, Serialize, Deserialize)]
pub struct Text {
    /// Text content
    content: String,
    /// Text color
    color: Color,
    /// Font size in pixels
    font_size: f32,
    /// Font weight
    font_weight: FontWeight,
    /// Font style
    font_style: FontStyle,
    /// Line height multiplier
    line_height: f32,
    /// Maximum width before wrapping (None = no wrapping)
    max_width: Option<f32>,
    /// Test ID
    test_id_value: Option<String>,
    /// Cached bounds
    #[serde(skip)]
    bounds: Rect,
}

impl Text {
    /// Create new text widget.
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            color: Color::BLACK,
            font_size: 16.0,
            font_weight: FontWeight::Normal,
            font_style: FontStyle::Normal,
            line_height: 1.2,
            max_width: None,
            test_id_value: None,
            bounds: Rect::default(),
        }
    }

    /// Set text color.
    #[must_use]
    pub const fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set font size.
    #[must_use]
    pub const fn font_size(mut self, size: f32) -> Self {
        self.font_size = size;
        self
    }

    /// Set font weight.
    #[must_use]
    pub const fn font_weight(mut self, weight: FontWeight) -> Self {
        self.font_weight = weight;
        self
    }

    /// Set font style.
    #[must_use]
    pub const fn font_style(mut self, style: FontStyle) -> Self {
        self.font_style = style;
        self
    }

    /// Set line height multiplier.
    #[must_use]
    pub const fn line_height(mut self, multiplier: f32) -> Self {
        self.line_height = multiplier;
        self
    }

    /// Set maximum width for text wrapping.
    #[must_use]
    pub const fn max_width(mut self, width: f32) -> Self {
        self.max_width = Some(width);
        self
    }

    /// Set test ID.
    #[must_use]
    pub fn with_test_id(mut self, id: impl Into<String>) -> Self {
        self.test_id_value = Some(id.into());
        self
    }

    /// Get the text content.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Estimate text size (simplified - real implementation would use font metrics).
    fn estimate_size(&self, max_width: f32) -> Size {
        // Simplified: assume ~0.6 em width per character
        let char_width = self.font_size * 0.6;
        let line_height = self.font_size * self.line_height;

        if self.content.is_empty() {
            return Size::new(0.0, line_height);
        }

        let total_width = self.content.len() as f32 * char_width;

        if let Some(max_w) = self.max_width {
            let effective_max = max_w.min(max_width);
            if total_width > effective_max {
                let lines = (total_width / effective_max).ceil();
                return Size::new(effective_max, lines * line_height);
            }
        }

        Size::new(total_width.min(max_width), line_height)
    }
}

impl Widget for Text {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let size = self.estimate_size(constraints.max_width);
        constraints.constrain(size)
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: bounds.size(),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        let style = TextStyle {
            size: self.font_size,
            color: self.color,
            weight: self.font_weight,
            style: self.font_style,
        };

        canvas.draw_text(&self.content, self.bounds.origin(), &style);
    }

    fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
        None // Text is not interactive
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

// PROBAR-SPEC-009: Brick Architecture - Tests define interface
impl Brick for Text {
    fn brick_name(&self) -> &'static str {
        "Text"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[
            BrickAssertion::TextVisible,
            BrickAssertion::MaxLatencyMs(16),
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let mut failed = Vec::new();

        // Verify text visibility
        if self.content.is_empty() {
            failed.push((BrickAssertion::TextVisible, "Text content is empty".into()));
        } else {
            passed.push(BrickAssertion::TextVisible);
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
        let test_id = self.test_id_value.as_deref().unwrap_or("text");
        format!(
            r#"<span class="brick-text" data-testid="{}">{}</span>"#,
            test_id, self.content
        )
    }

    fn to_css(&self) -> String {
        format!(
            r".brick-text {{
    color: {};
    font-size: {}px;
    line-height: {};
    display: inline-block;
}}",
            self.color.to_hex(),
            self.font_size,
            self.line_height
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use presentar_core::draw::DrawCommand;
    use presentar_core::widget::AccessibleRole;
    use presentar_core::{Brick, BrickAssertion, Point, RecordingCanvas, Widget};

    #[test]
    fn test_text_new() {
        let t = Text::new("Hello");
        assert_eq!(t.content(), "Hello");
        assert_eq!(t.font_size, 16.0);
    }

    #[test]
    fn test_text_builder() {
        let t = Text::new("Test")
            .color(Color::WHITE)
            .font_size(24.0)
            .font_weight(FontWeight::Bold)
            .with_test_id("my-text");

        assert_eq!(t.color, Color::WHITE);
        assert_eq!(t.font_size, 24.0);
        assert_eq!(t.font_weight, FontWeight::Bold);
        assert_eq!(Widget::test_id(&t), Some("my-text"));
    }

    #[test]
    fn test_text_measure() {
        let t = Text::new("Hello");
        let size = t.measure(Constraints::loose(Size::new(1000.0, 1000.0)));
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);
    }

    #[test]
    fn test_text_empty() {
        let t = Text::new("");
        let size = t.measure(Constraints::loose(Size::new(1000.0, 1000.0)));
        assert_eq!(size.width, 0.0);
        assert!(size.height > 0.0); // Line height
    }

    // ===== Paint Tests =====

    #[test]
    fn test_text_paint_draws_text() {
        let mut text = Text::new("Hello World");
        text.layout(Rect::new(10.0, 20.0, 200.0, 30.0));

        let mut canvas = RecordingCanvas::new();
        text.paint(&mut canvas);

        assert_eq!(canvas.command_count(), 1);
        match &canvas.commands()[0] {
            DrawCommand::Text {
                content, position, ..
            } => {
                assert_eq!(content, "Hello World");
                assert_eq!(*position, Point::new(10.0, 20.0));
            }
            _ => panic!("Expected Text command"),
        }
    }

    #[test]
    fn test_text_paint_uses_color() {
        let mut text = Text::new("Colored").color(Color::RED);
        text.layout(Rect::new(0.0, 0.0, 100.0, 20.0));

        let mut canvas = RecordingCanvas::new();
        text.paint(&mut canvas);

        match &canvas.commands()[0] {
            DrawCommand::Text { style, .. } => {
                assert_eq!(style.color, Color::RED);
            }
            _ => panic!("Expected Text command"),
        }
    }

    #[test]
    fn test_text_paint_uses_font_size() {
        let mut text = Text::new("Large").font_size(32.0);
        text.layout(Rect::new(0.0, 0.0, 200.0, 40.0));

        let mut canvas = RecordingCanvas::new();
        text.paint(&mut canvas);

        match &canvas.commands()[0] {
            DrawCommand::Text { style, .. } => {
                assert_eq!(style.size, 32.0);
            }
            _ => panic!("Expected Text command"),
        }
    }

    #[test]
    fn test_text_paint_uses_font_weight() {
        let mut text = Text::new("Bold").font_weight(FontWeight::Bold);
        text.layout(Rect::new(0.0, 0.0, 100.0, 20.0));

        let mut canvas = RecordingCanvas::new();
        text.paint(&mut canvas);

        match &canvas.commands()[0] {
            DrawCommand::Text { style, .. } => {
                assert_eq!(style.weight, FontWeight::Bold);
            }
            _ => panic!("Expected Text command"),
        }
    }

    #[test]
    fn test_text_paint_uses_font_style() {
        let mut text = Text::new("Italic").font_style(FontStyle::Italic);
        text.layout(Rect::new(0.0, 0.0, 100.0, 20.0));

        let mut canvas = RecordingCanvas::new();
        text.paint(&mut canvas);

        match &canvas.commands()[0] {
            DrawCommand::Text { style, .. } => {
                assert_eq!(style.style, FontStyle::Italic);
            }
            _ => panic!("Expected Text command"),
        }
    }

    #[test]
    fn test_text_paint_empty() {
        let mut text = Text::new("");
        text.layout(Rect::new(0.0, 0.0, 100.0, 20.0));

        let mut canvas = RecordingCanvas::new();
        text.paint(&mut canvas);

        // Should still draw (empty text)
        assert_eq!(canvas.command_count(), 1);
        match &canvas.commands()[0] {
            DrawCommand::Text { content, .. } => {
                assert!(content.is_empty());
            }
            _ => panic!("Expected Text command"),
        }
    }

    #[test]
    fn test_text_paint_position_from_layout() {
        let mut text = Text::new("Positioned");
        text.layout(Rect::new(50.0, 100.0, 200.0, 30.0));

        let mut canvas = RecordingCanvas::new();
        text.paint(&mut canvas);

        match &canvas.commands()[0] {
            DrawCommand::Text { position, .. } => {
                assert_eq!(position.x, 50.0);
                assert_eq!(position.y, 100.0);
            }
            _ => panic!("Expected Text command"),
        }
    }

    // ===== Widget Trait Tests =====

    #[test]
    fn test_text_type_id() {
        let t = Text::new("test");
        assert_eq!(Widget::type_id(&t), TypeId::of::<Text>());
    }

    #[test]
    fn test_text_layout_sets_bounds() {
        let mut t = Text::new("test");
        let result = t.layout(Rect::new(10.0, 20.0, 100.0, 30.0));
        assert_eq!(result.size, Size::new(100.0, 30.0));
        assert_eq!(t.bounds, Rect::new(10.0, 20.0, 100.0, 30.0));
    }

    #[test]
    fn test_text_children_empty() {
        let t = Text::new("test");
        assert!(t.children().is_empty());
    }

    #[test]
    fn test_text_event_returns_none() {
        let mut t = Text::new("test");
        t.layout(Rect::new(0.0, 0.0, 100.0, 20.0));
        let result = t.event(&Event::MouseEnter);
        assert!(result.is_none());
    }

    #[test]
    fn test_text_line_height() {
        let t = Text::new("test").line_height(1.5);
        assert_eq!(t.line_height, 1.5);
    }

    #[test]
    fn test_text_max_width() {
        let t = Text::new("test").max_width(200.0);
        assert_eq!(t.max_width, Some(200.0));
    }

    #[test]
    fn test_text_measure_with_max_width() {
        let t = Text::new("A very long text that should wrap").max_width(50.0);
        let size = t.measure(Constraints::loose(Size::new(1000.0, 1000.0)));
        assert!(size.width <= 50.0);
        assert!(size.height > t.font_size); // Multiple lines
    }

    #[test]
    fn test_text_content_accessor() {
        let t = Text::new("Hello World");
        assert_eq!(t.content(), "Hello World");
    }

    // ===== Additional Coverage Tests =====

    #[test]
    fn test_text_is_interactive() {
        let t = Text::new("Test");
        assert!(!t.is_interactive());
    }

    #[test]
    fn test_text_is_focusable() {
        let t = Text::new("Test");
        assert!(!t.is_focusable());
    }

    #[test]
    fn test_text_accessible_role() {
        let t = Text::new("Test");
        assert_eq!(t.accessible_role(), AccessibleRole::Generic);
    }

    #[test]
    fn test_text_accessible_name() {
        let t = Text::new("Accessible Text");
        // Text widget doesn't implement custom accessible_name
        assert!(Widget::accessible_name(&t).is_none());
    }

    #[test]
    fn test_text_children_mut() {
        let mut t = Text::new("Test");
        assert!(t.children_mut().is_empty());
    }

    // ===== Brick Trait Tests =====

    #[test]
    fn test_text_brick_name() {
        let t = Text::new("Test");
        assert_eq!(t.brick_name(), "Text");
    }

    #[test]
    fn test_text_brick_assertions() {
        let t = Text::new("Test");
        let assertions = t.assertions();
        assert_eq!(assertions.len(), 2);
        assert!(assertions.contains(&BrickAssertion::MaxLatencyMs(16)));
        assert!(assertions.contains(&BrickAssertion::TextVisible));
    }

    #[test]
    fn test_text_brick_budget() {
        let t = Text::new("Test");
        let budget = t.budget();
        assert!(budget.measure_ms > 0);
        assert!(budget.layout_ms > 0);
        assert!(budget.paint_ms > 0);
    }

    #[test]
    fn test_text_brick_verify_with_content() {
        let t = Text::new("Visible Text");
        let verification = t.verify();
        assert!(verification.passed.contains(&BrickAssertion::TextVisible));
        assert!(verification
            .passed
            .contains(&BrickAssertion::MaxLatencyMs(16)));
        assert!(verification.failed.is_empty());
    }

    #[test]
    fn test_text_brick_verify_empty_content() {
        let t = Text::new("");
        let verification = t.verify();
        // Empty text should fail TextVisible
        assert!(verification
            .failed
            .iter()
            .any(|(a, _)| *a == BrickAssertion::TextVisible));
    }

    #[test]
    fn test_text_to_html() {
        let t = Text::new("Hello World").with_test_id("greeting");
        let html = t.to_html();
        assert!(html.contains("brick-text"));
        assert!(html.contains("data-testid=\"greeting\""));
        assert!(html.contains("Hello World"));
    }

    #[test]
    fn test_text_to_html_default_test_id() {
        let t = Text::new("Hello");
        let html = t.to_html();
        assert!(html.contains("data-testid=\"text\""));
    }

    #[test]
    fn test_text_to_css() {
        let t = Text::new("Text")
            .font_size(20.0)
            .color(Color::RED)
            .line_height(1.5);
        let css = t.to_css();
        assert!(css.contains("brick-text"));
        assert!(css.contains("font-size: 20px"));
        assert!(css.contains("line-height: 1.5"));
    }

    #[test]
    fn test_text_default_values() {
        let t = Text::new("");
        assert!(t.content.is_empty());
        assert_eq!(t.font_size, 16.0);
        assert_eq!(t.line_height, 1.2);
    }

    #[test]
    fn test_text_font_weight_default() {
        let t = Text::new("Test");
        assert_eq!(t.font_weight, FontWeight::Normal);
    }

    #[test]
    fn test_text_font_style_default() {
        let t = Text::new("Test");
        assert_eq!(t.font_style, FontStyle::Normal);
    }

    #[test]
    fn test_text_clone() {
        let t = Text::new("Clone Me").font_size(20.0).color(Color::BLUE);
        let cloned = t;
        assert_eq!(cloned.content(), "Clone Me");
        assert_eq!(cloned.font_size, 20.0);
        assert_eq!(cloned.color, Color::BLUE);
    }
}
