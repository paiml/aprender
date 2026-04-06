//! `FlexCell` atomic widget.
//!
//! A bounded text container that enforces strict no-bleed rendering.
//! Reference: SPEC-024 Appendix I (Atomic Widget Mandate).
//!
//! # Falsification Criteria
//! - F-ATOM-FLEX-001: Text length > bounds width MUST NOT render overflow.
//! - F-ATOM-FLEX-002: Ellipsis MUST appear when truncation occurs.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::borrow::Cow;
use std::time::Duration;

/// Text overflow behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Overflow {
    /// Clip text at bounds (no indicator).
    #[default]
    Clip,
    /// Show ellipsis (…) when truncated.
    Ellipsis,
    /// Truncate from middle, preserving start/end (for paths).
    EllipsisMiddle,
}

/// Text alignment within the cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Alignment {
    #[default]
    Left,
    Center,
    Right,
}

/// `FlexCell` - bounded text atom.
///
/// Guarantees that text NEVER bleeds outside its allocated bounds.
/// This is the fundamental building block for all text rendering in the TUI.
#[derive(Debug, Clone)]
pub struct FlexCell {
    /// Text content.
    text: String,
    /// Text style (color, etc.).
    style: TextStyle,
    /// Overflow behavior.
    overflow: Overflow,
    /// Horizontal alignment.
    alignment: Alignment,
    /// Minimum width (in characters).
    min_width: Option<usize>,
    /// Cached bounds.
    bounds: Rect,
}

impl Default for FlexCell {
    fn default() -> Self {
        Self::new("")
    }
}

impl FlexCell {
    /// Create a new `FlexCell` with text.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            style: TextStyle::default(),
            overflow: Overflow::Ellipsis, // Safe default
            alignment: Alignment::Left,
            min_width: None,
            bounds: Rect::default(),
        }
    }

    /// Set text color.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.style.color = color;
        self
    }

    /// Set full text style.
    #[must_use]
    pub fn with_style(mut self, style: TextStyle) -> Self {
        self.style = style;
        self
    }

    /// Set overflow behavior.
    #[must_use]
    pub fn with_overflow(mut self, overflow: Overflow) -> Self {
        self.overflow = overflow;
        self
    }

    /// Set text alignment.
    #[must_use]
    pub fn with_alignment(mut self, alignment: Alignment) -> Self {
        self.alignment = alignment;
        self
    }

    /// Set minimum width.
    #[must_use]
    pub fn with_min_width(mut self, width: usize) -> Self {
        self.min_width = Some(width);
        self
    }

    /// Get the text content.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Set text content.
    pub fn set_text(&mut self, text: impl Into<String>) {
        self.text = text.into();
    }

    /// Truncate text to fit width based on overflow mode.
    fn truncate_to_fit(&self, max_chars: usize) -> Cow<'_, str> {
        let char_count = self.text.chars().count();
        if char_count <= max_chars {
            Cow::Borrowed(&self.text)
        } else {
            match self.overflow {
                Overflow::Clip => Cow::Owned(self.text.chars().take(max_chars).collect()),
                Overflow::Ellipsis => {
                    if max_chars == 0 {
                        Cow::Borrowed("")
                    } else if max_chars == 1 {
                        Cow::Borrowed("…")
                    } else {
                        let truncated: String = self.text.chars().take(max_chars - 1).collect();
                        Cow::Owned(format!("{truncated}…"))
                    }
                }
                Overflow::EllipsisMiddle => {
                    if max_chars <= 3 {
                        // Fall back to end ellipsis for very short
                        if max_chars == 0 {
                            Cow::Borrowed("")
                        } else if max_chars == 1 {
                            Cow::Borrowed("…")
                        } else {
                            let truncated: String = self.text.chars().take(max_chars - 1).collect();
                            Cow::Owned(format!("{truncated}…"))
                        }
                    } else {
                        let start_len = (max_chars - 1) / 3;
                        let end_len = max_chars - 1 - start_len;
                        let start: String = self.text.chars().take(start_len).collect();
                        let end: String = self.text.chars().skip(char_count - end_len).collect();
                        Cow::Owned(format!("{start}…{end}"))
                    }
                }
            }
        }
    }

    /// Calculate x offset for alignment.
    fn alignment_offset(&self, text_width: usize, cell_width: usize) -> f32 {
        if text_width >= cell_width {
            return 0.0;
        }
        let space = cell_width - text_width;
        match self.alignment {
            Alignment::Left => 0.0,
            Alignment::Center => (space / 2) as f32,
            Alignment::Right => space as f32,
        }
    }
}

impl Widget for FlexCell {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let text_width = self.text.chars().count() as f32;
        let min_w = self.min_width.map_or(0.0, |w| w as f32);
        let width = text_width.max(min_w).min(constraints.max_width);
        constraints.constrain(Size::new(width, 1.0))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.bounds.width < 1.0 || self.bounds.height < 1.0 {
            return;
        }

        let max_chars = self.bounds.width as usize;

        // F-ATOM-FLEX-001: Enforce bounds - truncate to fit
        let display_text = self.truncate_to_fit(max_chars);
        let text_width = display_text.chars().count();

        // Calculate alignment offset
        let x_offset = self.alignment_offset(text_width, max_chars);

        // Render text
        canvas.draw_text(
            &display_text,
            Point::new(self.bounds.x + x_offset, self.bounds.y),
            &self.style,
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

impl Brick for FlexCell {
    fn brick_name(&self) -> &'static str {
        "flex_cell"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[
            BrickAssertion::max_latency_ms(1), // Text rendering is fast
        ];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(1)
    }

    fn verify(&self) -> BrickVerification {
        // F-ATOM-FLEX-001: No bleed verification
        // The truncate_to_fit function guarantees output <= max_chars
        let max_chars = self.bounds.width as usize;
        let display_text = self.truncate_to_fit(max_chars);
        let no_bleed = display_text.chars().count() <= max_chars;

        if no_bleed {
            BrickVerification {
                passed: self.assertions().to_vec(),
                failed: vec![],
                verification_time: Duration::from_micros(1),
            }
        } else {
            BrickVerification {
                passed: vec![],
                failed: self
                    .assertions()
                    .iter()
                    .map(|a| (a.clone(), "Text exceeds bounds".to_string()))
                    .collect(),
                verification_time: Duration::from_micros(1),
            }
        }
    }

    fn to_html(&self) -> String {
        format!("<span class=\"flex-cell\">{}</span>", self.text)
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct::{CellBuffer, DirectTerminalCanvas};

    // =========================================================================
    // ENUM DEFAULT TESTS
    // =========================================================================

    #[test]
    fn test_overflow_default() {
        assert_eq!(Overflow::default(), Overflow::Clip);
    }

    #[test]
    fn test_alignment_default() {
        assert_eq!(Alignment::default(), Alignment::Left);
    }

    // =========================================================================
    // FLEX CELL CREATION TESTS
    // =========================================================================

    #[test]
    fn test_flex_cell_new() {
        let cell = FlexCell::new("Hello");
        assert_eq!(cell.text(), "Hello");
    }

    #[test]
    fn test_flex_cell_default() {
        let cell = FlexCell::default();
        assert_eq!(cell.text(), "");
    }

    #[test]
    fn test_flex_cell_with_color() {
        let cell = FlexCell::new("Text").with_color(Color::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(cell.style.color.r, 1.0);
    }

    #[test]
    fn test_flex_cell_with_style() {
        let style = TextStyle {
            color: Color::new(0.0, 1.0, 0.0, 1.0),
            ..Default::default()
        };
        let cell = FlexCell::new("Text").with_style(style);
        assert_eq!(cell.style.color.g, 1.0);
    }

    #[test]
    fn test_flex_cell_with_overflow() {
        let cell = FlexCell::new("Text").with_overflow(Overflow::Clip);
        assert_eq!(cell.overflow, Overflow::Clip);
    }

    #[test]
    fn test_flex_cell_with_alignment() {
        let cell = FlexCell::new("Text").with_alignment(Alignment::Center);
        assert_eq!(cell.alignment, Alignment::Center);
    }

    #[test]
    fn test_flex_cell_with_min_width() {
        let cell = FlexCell::new("Hi").with_min_width(10);
        assert_eq!(cell.min_width, Some(10));
    }

    #[test]
    fn test_flex_cell_text_getter() {
        let cell = FlexCell::new("Content");
        assert_eq!(cell.text(), "Content");
    }

    #[test]
    fn test_flex_cell_set_text() {
        let mut cell = FlexCell::new("Old");
        cell.set_text("New");
        assert_eq!(cell.text(), "New");
    }

    // =========================================================================
    // TRUNCATION TESTS
    // =========================================================================

    // F-ATOM-FLEX-001: Text never exceeds bounds
    #[test]
    fn test_no_bleed_long_text() {
        let mut cell = FlexCell::new("This is a very long text that should be truncated");
        cell.layout(Rect::new(0.0, 0.0, 10.0, 1.0));

        let truncated = cell.truncate_to_fit(10);
        assert_eq!(truncated.chars().count(), 10);
        assert!(truncated.ends_with('…'));
    }

    // F-ATOM-FLEX-001: Short text unchanged
    #[test]
    fn test_no_bleed_short_text() {
        let mut cell = FlexCell::new("Short");
        cell.layout(Rect::new(0.0, 0.0, 10.0, 1.0));

        let truncated = cell.truncate_to_fit(10);
        assert_eq!(truncated, "Short");
    }

    // F-ATOM-FLEX-002: Ellipsis appears on truncation
    #[test]
    fn test_ellipsis_on_truncation() {
        let cell = FlexCell::new("LongText").with_overflow(Overflow::Ellipsis);
        let truncated = cell.truncate_to_fit(5);
        assert_eq!(truncated, "Long…");
    }

    #[test]
    fn test_ellipsis_zero_max() {
        let cell = FlexCell::new("LongText").with_overflow(Overflow::Ellipsis);
        let truncated = cell.truncate_to_fit(0);
        assert_eq!(truncated, "");
    }

    #[test]
    fn test_ellipsis_one_max() {
        let cell = FlexCell::new("LongText").with_overflow(Overflow::Ellipsis);
        let truncated = cell.truncate_to_fit(1);
        assert_eq!(truncated, "…");
    }

    #[test]
    fn test_clip_truncation() {
        let cell = FlexCell::new("LongText").with_overflow(Overflow::Clip);
        let truncated = cell.truncate_to_fit(4);
        assert_eq!(truncated, "Long");
    }

    // Middle ellipsis for paths
    #[test]
    fn test_middle_ellipsis() {
        let cell =
            FlexCell::new("/home/user/projects/myapp").with_overflow(Overflow::EllipsisMiddle);
        let truncated = cell.truncate_to_fit(15);
        assert!(truncated.contains('…'));
        assert_eq!(truncated.chars().count(), 15);
    }

    #[test]
    fn test_middle_ellipsis_short_max() {
        let cell = FlexCell::new("LongText").with_overflow(Overflow::EllipsisMiddle);
        // For max_chars <= 3, falls back to end ellipsis
        let truncated = cell.truncate_to_fit(3);
        assert_eq!(truncated.chars().count(), 3);
    }

    #[test]
    fn test_middle_ellipsis_zero_max() {
        let cell = FlexCell::new("LongText").with_overflow(Overflow::EllipsisMiddle);
        let truncated = cell.truncate_to_fit(0);
        assert_eq!(truncated, "");
    }

    #[test]
    fn test_middle_ellipsis_one_max() {
        let cell = FlexCell::new("LongText").with_overflow(Overflow::EllipsisMiddle);
        let truncated = cell.truncate_to_fit(1);
        assert_eq!(truncated, "…");
    }

    // =========================================================================
    // ALIGNMENT TESTS
    // =========================================================================

    #[test]
    fn test_alignment_left() {
        let cell = FlexCell::new("Hi").with_alignment(Alignment::Left);
        assert_eq!(cell.alignment_offset(2, 10), 0.0);
    }

    #[test]
    fn test_alignment_right() {
        let cell = FlexCell::new("Hi").with_alignment(Alignment::Right);
        assert_eq!(cell.alignment_offset(2, 10), 8.0);
    }

    #[test]
    fn test_alignment_center() {
        let cell = FlexCell::new("Hi").with_alignment(Alignment::Center);
        assert_eq!(cell.alignment_offset(2, 10), 4.0);
    }

    #[test]
    fn test_alignment_text_wider_than_cell() {
        let cell = FlexCell::new("Very long text").with_alignment(Alignment::Right);
        // Text wider than cell should return 0 offset
        assert_eq!(cell.alignment_offset(15, 10), 0.0);
    }

    // =========================================================================
    // WIDGET TRAIT TESTS
    // =========================================================================

    #[test]
    fn test_flex_cell_type_id() {
        let cell = FlexCell::new("Test");
        let id = Widget::type_id(&cell);
        assert_eq!(id, TypeId::of::<FlexCell>());
    }

    #[test]
    fn test_flex_cell_measure() {
        let cell = FlexCell::new("Hello");
        let constraints = Constraints::loose(Size::new(100.0, 50.0));
        let size = cell.measure(constraints);
        assert_eq!(size.width, 5.0); // "Hello" is 5 chars
        assert_eq!(size.height, 1.0);
    }

    #[test]
    fn test_flex_cell_measure_with_min_width() {
        let cell = FlexCell::new("Hi").with_min_width(10);
        let constraints = Constraints::loose(Size::new(100.0, 50.0));
        let size = cell.measure(constraints);
        assert_eq!(size.width, 10.0);
    }

    #[test]
    fn test_flex_cell_layout() {
        let mut cell = FlexCell::new("Test");
        let bounds = Rect::new(0.0, 0.0, 20.0, 1.0);
        let result = cell.layout(bounds);
        assert_eq!(result.size.width, 20.0);
        assert_eq!(cell.bounds, bounds);
    }

    #[test]
    fn test_flex_cell_paint() {
        let mut buffer = CellBuffer::new(30, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let mut cell = FlexCell::new("Hello");
        cell.layout(Rect::new(0.0, 0.0, 20.0, 1.0));
        cell.paint(&mut canvas);
    }

    #[test]
    fn test_flex_cell_paint_with_alignment() {
        let mut buffer = CellBuffer::new(30, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let mut cell = FlexCell::new("Hi").with_alignment(Alignment::Right);
        cell.layout(Rect::new(0.0, 0.0, 20.0, 1.0));
        cell.paint(&mut canvas);
    }

    #[test]
    fn test_flex_cell_event() {
        let mut cell = FlexCell::new("Test");
        let event = Event::key_down(presentar_core::Key::Enter);
        let result = cell.event(&event);
        assert!(result.is_none());
    }

    #[test]
    fn test_flex_cell_children() {
        let cell = FlexCell::new("Test");
        assert!(cell.children().is_empty());
    }

    #[test]
    fn test_flex_cell_children_mut() {
        let mut cell = FlexCell::new("Test");
        assert!(cell.children_mut().is_empty());
    }

    // Render doesn't panic with zero width
    #[test]
    fn test_zero_width_no_panic() {
        let mut cell = FlexCell::new("Test");
        cell.layout(Rect::new(0.0, 0.0, 0.0, 1.0));

        let mut buffer = CellBuffer::new(10, 1);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        cell.paint(&mut canvas); // Should not panic
    }

    // Edge case: exactly fitting text
    #[test]
    fn test_exact_fit() {
        let cell = FlexCell::new("12345");
        let truncated = cell.truncate_to_fit(5);
        assert_eq!(truncated, "12345");
    }

    // =========================================================================
    // BRICK TRAIT TESTS
    // =========================================================================

    #[test]
    fn test_flex_cell_brick_name() {
        let cell = FlexCell::new("Test");
        assert_eq!(cell.brick_name(), "flex_cell");
    }

    #[test]
    fn test_flex_cell_assertions() {
        let cell = FlexCell::new("Test");
        let assertions = cell.assertions();
        assert!(!assertions.is_empty());
    }

    #[test]
    fn test_flex_cell_budget() {
        let cell = FlexCell::new("Test");
        let budget = cell.budget();
        assert!(budget.total_ms > 0);
    }

    #[test]
    fn test_brick_verification_passes() {
        let mut cell = FlexCell::new("Test");
        cell.layout(Rect::new(0.0, 0.0, 10.0, 1.0));
        let v = cell.verify();
        assert!(v.failed.is_empty());
    }

    #[test]
    fn test_flex_cell_to_html() {
        let cell = FlexCell::new("Content");
        let html = cell.to_html();
        assert!(html.contains("Content"));
        assert!(html.contains("flex-cell"));
    }

    #[test]
    fn test_flex_cell_to_css() {
        let cell = FlexCell::new("Test");
        let css = cell.to_css();
        assert!(css.is_empty());
    }
}
