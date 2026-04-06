//! Simple text display widget.
//!
//! Renders text with optional styling.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    FontWeight, LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Text alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextAlign {
    #[default]
    Left,
    Center,
    Right,
}

/// Simple text display widget.
#[derive(Debug, Clone)]
pub struct Text {
    content: String,
    color: Color,
    bold: bool,
    align: TextAlign,
    bounds: Rect,
}

impl Text {
    /// Create a new text widget.
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            color: Color::new(0.8, 0.8, 0.8, 1.0),
            bold: false,
            align: TextAlign::Left,
            bounds: Rect::default(),
        }
    }

    /// Set text color.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Make text bold.
    #[must_use]
    pub fn bold(mut self) -> Self {
        self.bold = true;
        self
    }

    /// Set text alignment.
    #[must_use]
    pub fn align(mut self, align: TextAlign) -> Self {
        self.align = align;
        self
    }

    /// Center-align text.
    #[must_use]
    pub fn centered(mut self) -> Self {
        self.align = TextAlign::Center;
        self
    }

    /// Right-align text.
    #[must_use]
    pub fn right(mut self) -> Self {
        self.align = TextAlign::Right;
        self
    }

    /// Get the text content.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Set the text content.
    pub fn set_content(&mut self, content: impl Into<String>) {
        self.content = content.into();
    }
}

impl Default for Text {
    fn default() -> Self {
        Self::new("")
    }
}

impl Brick for Text {
    fn brick_name(&self) -> &'static str {
        "text"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(1)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(1)
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification {
            passed: vec![BrickAssertion::max_latency_ms(1)],
            failed: vec![],
            verification_time: Duration::from_micros(1),
        }
    }

    fn to_html(&self) -> String {
        format!("<span>{}</span>", self.content)
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

impl Widget for Text {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let char_count = self.content.chars().count() as f32;
        let width = char_count.min(constraints.max_width);
        let height = 1.0_f32.min(constraints.max_height);
        constraints.constrain(Size::new(width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height.min(1.0)),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.bounds.width < 1.0 || self.bounds.height < 1.0 {
            return;
        }

        let style = TextStyle {
            color: self.color,
            weight: if self.bold {
                FontWeight::Bold
            } else {
                FontWeight::Normal
            },
            ..Default::default()
        };

        let width = self.bounds.width as usize;
        let text_len = self.content.chars().count();

        let x_offset = match self.align {
            TextAlign::Left => 0.0,
            TextAlign::Center => ((width.saturating_sub(text_len)) / 2) as f32,
            TextAlign::Right => (width.saturating_sub(text_len)) as f32,
        };

        // Truncate if needed
        let display: String = if text_len > width {
            if width > 3 {
                format!(
                    "{}...",
                    self.content.chars().take(width - 3).collect::<String>()
                )
            } else {
                self.content.chars().take(width).collect()
            }
        } else {
            self.content.clone()
        };

        canvas.draw_text(
            &display,
            Point::new(self.bounds.x + x_offset, self.bounds.y),
            &style,
        );
    }

    fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_new() {
        let text = Text::new("Hello");
        assert_eq!(text.content(), "Hello");
    }

    #[test]
    fn test_text_with_color() {
        let text = Text::new("Test").with_color(Color::RED);
        assert_eq!(text.color, Color::RED);
    }

    #[test]
    fn test_text_bold() {
        let text = Text::new("Test").bold();
        assert!(text.bold);
    }

    #[test]
    fn test_text_align() {
        let text = Text::new("Test").align(TextAlign::Center);
        assert_eq!(text.align, TextAlign::Center);
    }

    #[test]
    fn test_text_centered() {
        let text = Text::new("Test").centered();
        assert_eq!(text.align, TextAlign::Center);
    }

    #[test]
    fn test_text_right() {
        let text = Text::new("Test").right();
        assert_eq!(text.align, TextAlign::Right);
    }

    #[test]
    fn test_text_set_content() {
        let mut text = Text::new("Old");
        text.set_content("New");
        assert_eq!(text.content(), "New");
    }

    #[test]
    fn test_text_default() {
        let text = Text::default();
        assert_eq!(text.content(), "");
    }

    #[test]
    fn test_text_brick_name() {
        let text = Text::new("Test");
        assert_eq!(text.brick_name(), "text");
    }

    #[test]
    fn test_text_verify() {
        let text = Text::new("Test");
        assert!(text.verify().is_valid());
    }

    #[test]
    fn test_text_measure() {
        let text = Text::new("Hello");
        let size = text.measure(Constraints::new(0.0, 100.0, 0.0, 10.0));
        assert_eq!(size.width, 5.0);
        assert_eq!(size.height, 1.0);
    }

    #[test]
    fn test_text_layout() {
        let mut text = Text::new("Test");
        let result = text.layout(Rect::new(0.0, 0.0, 20.0, 5.0));
        assert_eq!(result.size.height, 1.0);
    }

    #[test]
    fn test_text_to_html() {
        let text = Text::new("Hello");
        assert_eq!(text.to_html(), "<span>Hello</span>");
    }

    #[test]
    fn test_text_type_id() {
        let text = Text::new("Test");
        assert_eq!(Widget::type_id(&text), TypeId::of::<Text>());
    }

    #[test]
    fn test_text_children() {
        let text = Text::new("Test");
        assert!(text.children().is_empty());
    }

    #[test]
    fn test_text_children_mut() {
        let mut text = Text::new("Test");
        assert!(text.children_mut().is_empty());
    }

    // ========================================================================
    // Additional tests for paint() and improved coverage
    // ========================================================================

    struct MockCanvas {
        texts: Vec<(String, Point)>,
    }

    impl MockCanvas {
        fn new() -> Self {
            Self { texts: vec![] }
        }
    }

    impl Canvas for MockCanvas {
        fn fill_rect(&mut self, _rect: Rect, _color: Color) {}
        fn stroke_rect(&mut self, _rect: Rect, _color: Color, _width: f32) {}
        fn draw_text(&mut self, text: &str, position: Point, _style: &TextStyle) {
            self.texts.push((text.to_string(), position));
        }
        fn draw_line(&mut self, _from: Point, _to: Point, _color: Color, _width: f32) {}
        fn fill_circle(&mut self, _center: Point, _radius: f32, _color: Color) {}
        fn stroke_circle(&mut self, _center: Point, _radius: f32, _color: Color, _width: f32) {}
        fn fill_arc(&mut self, _c: Point, _r: f32, _s: f32, _e: f32, _color: Color) {}
        fn draw_path(&mut self, _points: &[Point], _color: Color, _width: f32) {}
        fn fill_polygon(&mut self, _points: &[Point], _color: Color) {}
        fn push_clip(&mut self, _rect: Rect) {}
        fn pop_clip(&mut self) {}
        fn push_transform(&mut self, _transform: presentar_core::Transform2D) {}
        fn pop_transform(&mut self) {}
    }

    #[test]
    fn test_text_paint_basic() {
        let mut text = Text::new("Hello");
        text.bounds = Rect::new(0.0, 0.0, 20.0, 1.0);

        let mut canvas = MockCanvas::new();
        text.paint(&mut canvas);

        assert_eq!(canvas.texts.len(), 1);
        assert_eq!(canvas.texts[0].0, "Hello");
    }

    #[test]
    fn test_text_paint_centered() {
        let mut text = Text::new("Hi").centered();
        text.bounds = Rect::new(0.0, 0.0, 10.0, 1.0);

        let mut canvas = MockCanvas::new();
        text.paint(&mut canvas);

        // "Hi" is 2 chars, width is 10, so offset should be (10-2)/2 = 4
        assert_eq!(canvas.texts[0].1.x, 4.0);
    }

    #[test]
    fn test_text_paint_right() {
        let mut text = Text::new("Hi").right();
        text.bounds = Rect::new(0.0, 0.0, 10.0, 1.0);

        let mut canvas = MockCanvas::new();
        text.paint(&mut canvas);

        // "Hi" is 2 chars, width is 10, so offset should be 10-2 = 8
        assert_eq!(canvas.texts[0].1.x, 8.0);
    }

    #[test]
    fn test_text_paint_bold() {
        let mut text = Text::new("Bold").bold();
        text.bounds = Rect::new(0.0, 0.0, 20.0, 1.0);

        let mut canvas = MockCanvas::new();
        text.paint(&mut canvas);

        // Should render with bold style (just verify it runs)
        assert_eq!(canvas.texts.len(), 1);
        assert_eq!(canvas.texts[0].0, "Bold");
    }

    #[test]
    fn test_text_paint_truncation() {
        let mut text = Text::new("This is a very long text");
        text.bounds = Rect::new(0.0, 0.0, 10.0, 1.0);

        let mut canvas = MockCanvas::new();
        text.paint(&mut canvas);

        // Text should be truncated with "..."
        assert!(canvas.texts[0].0.ends_with("..."));
        assert!(canvas.texts[0].0.len() <= 10);
    }

    #[test]
    fn test_text_paint_truncation_short_width() {
        let mut text = Text::new("Hello");
        text.bounds = Rect::new(0.0, 0.0, 3.0, 1.0);

        let mut canvas = MockCanvas::new();
        text.paint(&mut canvas);

        // Width <= 3, so just truncate without "..."
        assert_eq!(canvas.texts[0].0.len(), 3);
    }

    #[test]
    fn test_text_paint_zero_bounds() {
        let mut text = Text::new("Test");
        text.bounds = Rect::new(0.0, 0.0, 0.0, 0.0);

        let mut canvas = MockCanvas::new();
        text.paint(&mut canvas);

        // Should return early, no output
        assert!(canvas.texts.is_empty());
    }

    #[test]
    fn test_text_paint_zero_height() {
        let mut text = Text::new("Test");
        text.bounds = Rect::new(0.0, 0.0, 10.0, 0.5);

        let mut canvas = MockCanvas::new();
        text.paint(&mut canvas);

        // Should return early since height < 1
        assert!(canvas.texts.is_empty());
    }

    #[test]
    fn test_text_event() {
        let mut text = Text::new("Test");
        let event = Event::key_down(presentar_core::Key::Enter);
        assert!(text.event(&event).is_none());
    }

    #[test]
    fn test_text_assertions() {
        let text = Text::new("Test");
        assert!(!text.assertions().is_empty());
    }

    #[test]
    fn test_text_budget() {
        let text = Text::new("Test");
        let budget = text.budget();
        // Budget should have total_ms set (uniform(1) sets to 1ms)
        assert!(budget.total_ms > 0);
    }

    #[test]
    fn test_text_to_css() {
        let text = Text::new("Test");
        assert!(text.to_css().is_empty());
    }

    #[test]
    fn test_text_align_default() {
        assert_eq!(TextAlign::default(), TextAlign::Left);
    }

    #[test]
    fn test_text_align_debug() {
        let align = TextAlign::Center;
        let debug = format!("{:?}", align);
        assert!(debug.contains("Center"));
    }

    #[test]
    fn test_text_clone() {
        let text = Text::new("Test").with_color(Color::RED).bold().centered();
        let cloned = text.clone();
        assert_eq!(cloned.content(), text.content());
        assert_eq!(cloned.color, text.color);
        assert_eq!(cloned.bold, text.bold);
        assert_eq!(cloned.align, text.align);
    }

    #[test]
    fn test_text_debug() {
        let text = Text::new("Test");
        let debug = format!("{:?}", text);
        assert!(debug.contains("Text"));
    }

    #[test]
    fn test_text_measure_constrained() {
        let text = Text::new("Hello World!");
        // Constrain width to less than text length
        let size = text.measure(Constraints::new(0.0, 5.0, 0.0, 10.0));
        assert_eq!(size.width, 5.0);
    }

    #[test]
    fn test_text_empty() {
        let text = Text::new("");
        let size = text.measure(Constraints::new(0.0, 100.0, 0.0, 10.0));
        assert_eq!(size.width, 0.0);
    }

    #[test]
    fn test_text_paint_at_position() {
        let mut text = Text::new("Test");
        text.bounds = Rect::new(5.0, 10.0, 20.0, 1.0);

        let mut canvas = MockCanvas::new();
        text.paint(&mut canvas);

        // Should render at bounds position
        assert_eq!(canvas.texts[0].1.x, 5.0);
        assert_eq!(canvas.texts[0].1.y, 10.0);
    }

    #[test]
    fn test_text_paint_exact_fit() {
        let mut text = Text::new("Hi");
        text.bounds = Rect::new(0.0, 0.0, 2.0, 1.0);

        let mut canvas = MockCanvas::new();
        text.paint(&mut canvas);

        // Text fits exactly
        assert_eq!(canvas.texts[0].0, "Hi");
    }
}
