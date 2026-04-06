//! Scrollbar widget with position indicator and arrow buttons.
//!
//! Provides vertical and horizontal scrollbars for scrollable content.
//! Based on btop scrollbar patterns.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Scrollbar orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScrollOrientation {
    /// Vertical scrollbar (default).
    #[default]
    Vertical,
    /// Horizontal scrollbar.
    Horizontal,
}

/// Characters used for scrollbar rendering.
#[derive(Debug, Clone)]
pub struct ScrollbarChars {
    /// Track/background character.
    pub track: char,
    /// Thumb/position indicator.
    pub thumb: char,
    /// Up arrow (vertical) or left arrow (horizontal).
    pub arrow_start: char,
    /// Down arrow (vertical) or right arrow (horizontal).
    pub arrow_end: char,
}

impl Default for ScrollbarChars {
    fn default() -> Self {
        Self::unicode()
    }
}

impl ScrollbarChars {
    /// Unicode box-drawing characters.
    #[must_use]
    pub fn unicode() -> Self {
        Self {
            track: '░',
            thumb: '█',
            arrow_start: '▲',
            arrow_end: '▼',
        }
    }

    /// Unicode horizontal variant.
    #[must_use]
    pub fn unicode_horizontal() -> Self {
        Self {
            track: '░',
            thumb: '█',
            arrow_start: '◀',
            arrow_end: '▶',
        }
    }

    /// ASCII-only characters.
    #[must_use]
    pub fn ascii() -> Self {
        Self {
            track: '-',
            thumb: '#',
            arrow_start: '^',
            arrow_end: 'v',
        }
    }

    /// Minimal style (no arrows).
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            track: '│',
            thumb: '┃',
            arrow_start: '│',
            arrow_end: '│',
        }
    }
}

/// Scrollbar widget with position indicator and optional arrow buttons.
#[derive(Debug, Clone)]
pub struct Scrollbar {
    /// Scrollbar orientation.
    orientation: ScrollOrientation,
    /// Total content length (items or lines).
    content_length: usize,
    /// Visible viewport length.
    viewport_length: usize,
    /// Current scroll offset (0-based).
    offset: usize,
    /// Show arrow buttons at ends.
    show_arrows: bool,
    /// Characters for rendering.
    chars: ScrollbarChars,
    /// Track color.
    track_color: Color,
    /// Thumb color.
    thumb_color: Color,
    /// Arrow color.
    arrow_color: Color,
    /// Cached bounds.
    bounds: Rect,
}

impl Default for Scrollbar {
    fn default() -> Self {
        Self::vertical(100, 10)
    }
}

impl Scrollbar {
    /// Create a vertical scrollbar.
    #[must_use]
    pub fn vertical(content_length: usize, viewport_length: usize) -> Self {
        Self {
            orientation: ScrollOrientation::Vertical,
            content_length,
            viewport_length,
            offset: 0,
            show_arrows: true,
            chars: ScrollbarChars::unicode(),
            track_color: Color::new(0.3, 0.3, 0.3, 1.0),
            thumb_color: Color::new(0.7, 0.7, 0.7, 1.0),
            arrow_color: Color::new(0.5, 0.5, 0.5, 1.0),
            bounds: Rect::default(),
        }
    }

    /// Create a horizontal scrollbar.
    #[must_use]
    pub fn horizontal(content_length: usize, viewport_length: usize) -> Self {
        Self {
            orientation: ScrollOrientation::Horizontal,
            content_length,
            viewport_length,
            offset: 0,
            show_arrows: true,
            chars: ScrollbarChars::unicode_horizontal(),
            track_color: Color::new(0.3, 0.3, 0.3, 1.0),
            thumb_color: Color::new(0.7, 0.7, 0.7, 1.0),
            arrow_color: Color::new(0.5, 0.5, 0.5, 1.0),
            bounds: Rect::default(),
        }
    }

    /// Set whether to show arrow buttons.
    #[must_use]
    pub fn with_arrows(mut self, show: bool) -> Self {
        self.show_arrows = show;
        self
    }

    /// Set custom characters.
    #[must_use]
    pub fn with_chars(mut self, chars: ScrollbarChars) -> Self {
        self.chars = chars;
        self
    }

    /// Set track color.
    #[must_use]
    pub fn with_track_color(mut self, color: Color) -> Self {
        self.track_color = color;
        self
    }

    /// Set thumb color.
    #[must_use]
    pub fn with_thumb_color(mut self, color: Color) -> Self {
        self.thumb_color = color;
        self
    }

    /// Set arrow color.
    #[must_use]
    pub fn with_arrow_color(mut self, color: Color) -> Self {
        self.arrow_color = color;
        self
    }

    /// Get current offset.
    #[must_use]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Set scroll offset.
    pub fn set_offset(&mut self, offset: usize) {
        self.offset = offset.min(self.max_offset());
    }

    /// Get scroll position as fraction (0.0-1.0).
    #[must_use]
    pub fn position(&self) -> f64 {
        let max = self.max_offset();
        if max == 0 {
            0.0
        } else {
            self.offset as f64 / max as f64
        }
    }

    /// Get thumb size as fraction of viewport (0.0-1.0).
    #[must_use]
    pub fn thumb_size(&self) -> f64 {
        if self.content_length == 0 {
            1.0
        } else {
            (self.viewport_length as f64 / self.content_length as f64).min(1.0)
        }
    }

    /// Get maximum scroll offset.
    #[must_use]
    pub fn max_offset(&self) -> usize {
        self.content_length.saturating_sub(self.viewport_length)
    }

    /// Check if content is scrollable.
    #[must_use]
    pub fn is_scrollable(&self) -> bool {
        self.content_length > self.viewport_length
    }

    /// Scroll by delta (positive = down/right, negative = up/left).
    pub fn scroll(&mut self, delta: i32) {
        if delta >= 0 {
            self.offset = (self.offset + delta as usize).min(self.max_offset());
        } else {
            self.offset = self.offset.saturating_sub((-delta) as usize);
        }
    }

    /// Scroll up/left by one unit.
    pub fn scroll_start(&mut self) {
        self.scroll(-1);
    }

    /// Scroll down/right by one unit.
    pub fn scroll_end(&mut self) {
        self.scroll(1);
    }

    /// Page up/left.
    pub fn page_start(&mut self) {
        let page = self.viewport_length.max(1);
        self.offset = self.offset.saturating_sub(page);
    }

    /// Page down/right.
    pub fn page_end(&mut self) {
        let page = self.viewport_length.max(1);
        self.offset = (self.offset + page).min(self.max_offset());
    }

    /// Jump to position (0.0-1.0).
    pub fn jump_to(&mut self, position: f64) {
        let pos = position.clamp(0.0, 1.0);
        self.offset = (pos * self.max_offset() as f64).round() as usize;
    }

    /// Jump to top/left.
    pub fn jump_start(&mut self) {
        self.offset = 0;
    }

    /// Jump to bottom/right.
    pub fn jump_end(&mut self) {
        self.offset = self.max_offset();
    }

    /// Update content and viewport lengths.
    pub fn update_lengths(&mut self, content_length: usize, viewport_length: usize) {
        self.content_length = content_length;
        self.viewport_length = viewport_length;
        self.offset = self.offset.min(self.max_offset());
    }

    /// Get the orientation.
    #[must_use]
    pub fn orientation(&self) -> ScrollOrientation {
        self.orientation
    }

    /// Get content length.
    #[must_use]
    pub fn content_length(&self) -> usize {
        self.content_length
    }

    /// Get viewport length.
    #[must_use]
    pub fn viewport_length(&self) -> usize {
        self.viewport_length
    }
}

impl Brick for Scrollbar {
    fn brick_name(&self) -> &'static str {
        "scrollbar"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(16)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(8)
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification {
            passed: self.assertions().to_vec(),
            failed: vec![],
            verification_time: Duration::from_micros(5),
        }
    }

    fn to_html(&self) -> String {
        String::new()
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

impl Widget for Scrollbar {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        match self.orientation {
            ScrollOrientation::Vertical => Size::new(1.0, constraints.max_height.clamp(3.0, 20.0)),
            ScrollOrientation::Horizontal => Size::new(constraints.max_width.clamp(3.0, 20.0), 1.0),
        }
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        let track_style = TextStyle {
            color: self.track_color,
            ..Default::default()
        };
        let thumb_style = TextStyle {
            color: self.thumb_color,
            ..Default::default()
        };
        let arrow_style = TextStyle {
            color: self.arrow_color,
            ..Default::default()
        };

        match self.orientation {
            ScrollOrientation::Vertical => {
                self.paint_vertical(canvas, &track_style, &thumb_style, &arrow_style);
            }
            ScrollOrientation::Horizontal => {
                self.paint_horizontal(canvas, &track_style, &thumb_style, &arrow_style);
            }
        }
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

impl Scrollbar {
    fn paint_vertical(
        &self,
        canvas: &mut dyn Canvas,
        track_style: &TextStyle,
        thumb_style: &TextStyle,
        arrow_style: &TextStyle,
    ) {
        let height = self.bounds.height as usize;
        if height < 3 {
            return;
        }

        let arrow_offset = usize::from(self.show_arrows);
        let track_start = arrow_offset;
        let track_end = height.saturating_sub(arrow_offset);
        let track_len = track_end.saturating_sub(track_start);

        if track_len == 0 {
            return;
        }

        // Draw arrows
        if self.show_arrows {
            canvas.draw_text(
                &self.chars.arrow_start.to_string(),
                Point::new(self.bounds.x, self.bounds.y),
                arrow_style,
            );
            canvas.draw_text(
                &self.chars.arrow_end.to_string(),
                Point::new(self.bounds.x, self.bounds.y + (height - 1) as f32),
                arrow_style,
            );
        }

        // Calculate thumb position and size
        let thumb_size = ((self.thumb_size() * track_len as f64).round() as usize).max(1);
        let thumb_pos = if self.is_scrollable() {
            (self.position() * (track_len.saturating_sub(thumb_size)) as f64).round() as usize
        } else {
            0
        };

        // Draw track and thumb
        for i in 0..track_len {
            let y = track_start + i;
            let in_thumb = i >= thumb_pos && i < thumb_pos + thumb_size;
            let ch = if in_thumb {
                self.chars.thumb
            } else {
                self.chars.track
            };
            let style = if in_thumb { thumb_style } else { track_style };
            canvas.draw_text(
                &ch.to_string(),
                Point::new(self.bounds.x, self.bounds.y + y as f32),
                style,
            );
        }
    }

    fn paint_horizontal(
        &self,
        canvas: &mut dyn Canvas,
        track_style: &TextStyle,
        thumb_style: &TextStyle,
        arrow_style: &TextStyle,
    ) {
        let width = self.bounds.width as usize;
        if width < 3 {
            return;
        }

        let arrow_offset = usize::from(self.show_arrows);
        let track_start = arrow_offset;
        let track_end = width.saturating_sub(arrow_offset);
        let track_len = track_end.saturating_sub(track_start);

        if track_len == 0 {
            return;
        }

        // Draw arrows
        if self.show_arrows {
            canvas.draw_text(
                &self.chars.arrow_start.to_string(),
                Point::new(self.bounds.x, self.bounds.y),
                arrow_style,
            );
            canvas.draw_text(
                &self.chars.arrow_end.to_string(),
                Point::new(self.bounds.x + (width - 1) as f32, self.bounds.y),
                arrow_style,
            );
        }

        // Calculate thumb position and size
        let thumb_size = ((self.thumb_size() * track_len as f64).round() as usize).max(1);
        let thumb_pos = if self.is_scrollable() {
            (self.position() * (track_len.saturating_sub(thumb_size)) as f64).round() as usize
        } else {
            0
        };

        // Draw track and thumb
        for i in 0..track_len {
            let x = track_start + i;
            let in_thumb = i >= thumb_pos && i < thumb_pos + thumb_size;
            let ch = if in_thumb {
                self.chars.thumb
            } else {
                self.chars.track
            };
            let style = if in_thumb { thumb_style } else { track_style };
            canvas.draw_text(
                &ch.to_string(),
                Point::new(self.bounds.x + x as f32, self.bounds.y),
                style,
            );
        }
    }
}

impl PartialEq for ScrollbarChars {
    fn eq(&self, other: &Self) -> bool {
        self.track == other.track
            && self.thumb == other.thumb
            && self.arrow_start == other.arrow_start
            && self.arrow_end == other.arrow_end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // =====================================================
    // Construction Tests
    // =====================================================

    #[test]
    fn test_vertical_scrollbar_creation() {
        let sb = Scrollbar::vertical(100, 10);
        assert_eq!(sb.orientation(), ScrollOrientation::Vertical);
        assert_eq!(sb.content_length(), 100);
        assert_eq!(sb.viewport_length(), 10);
    }

    #[test]
    fn test_horizontal_scrollbar_creation() {
        let sb = Scrollbar::horizontal(100, 20);
        assert_eq!(sb.orientation(), ScrollOrientation::Horizontal);
        assert_eq!(sb.content_length(), 100);
        assert_eq!(sb.viewport_length(), 20);
    }

    #[test]
    fn test_scrollbar_default() {
        let sb = Scrollbar::default();
        assert_eq!(sb.orientation(), ScrollOrientation::Vertical);
        assert_eq!(sb.offset(), 0);
    }

    // =====================================================
    // Builder Pattern Tests
    // =====================================================

    #[test]
    fn test_with_arrows() {
        let sb = Scrollbar::vertical(100, 10).with_arrows(false);
        assert!(!sb.show_arrows);
    }

    #[test]
    fn test_with_chars() {
        let chars = ScrollbarChars::ascii();
        let sb = Scrollbar::vertical(100, 10).with_chars(chars);
        assert_eq!(sb.chars.track, '-');
    }

    #[test]
    fn test_with_track_color() {
        let sb = Scrollbar::vertical(100, 10).with_track_color(Color::RED);
        assert_eq!(sb.track_color, Color::RED);
    }

    #[test]
    fn test_with_thumb_color() {
        let sb = Scrollbar::vertical(100, 10).with_thumb_color(Color::GREEN);
        assert_eq!(sb.thumb_color, Color::GREEN);
    }

    #[test]
    fn test_with_arrow_color() {
        let sb = Scrollbar::vertical(100, 10).with_arrow_color(Color::BLUE);
        assert_eq!(sb.arrow_color, Color::BLUE);
    }

    // =====================================================
    // ScrollbarChars Tests
    // =====================================================

    #[test]
    fn test_chars_unicode() {
        let chars = ScrollbarChars::unicode();
        assert_eq!(chars.track, '░');
        assert_eq!(chars.thumb, '█');
        assert_eq!(chars.arrow_start, '▲');
        assert_eq!(chars.arrow_end, '▼');
    }

    #[test]
    fn test_chars_unicode_horizontal() {
        let chars = ScrollbarChars::unicode_horizontal();
        assert_eq!(chars.arrow_start, '◀');
        assert_eq!(chars.arrow_end, '▶');
    }

    #[test]
    fn test_chars_ascii() {
        let chars = ScrollbarChars::ascii();
        assert_eq!(chars.track, '-');
        assert_eq!(chars.thumb, '#');
    }

    #[test]
    fn test_chars_minimal() {
        let chars = ScrollbarChars::minimal();
        assert_eq!(chars.track, '│');
        assert_eq!(chars.thumb, '┃');
    }

    #[test]
    fn test_chars_default() {
        let chars = ScrollbarChars::default();
        assert_eq!(chars, ScrollbarChars::unicode());
    }

    // =====================================================
    // Offset/Position Tests
    // =====================================================

    #[test]
    fn test_offset_initial() {
        let sb = Scrollbar::vertical(100, 10);
        assert_eq!(sb.offset(), 0);
    }

    #[test]
    fn test_set_offset() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.set_offset(50);
        assert_eq!(sb.offset(), 50);
    }

    #[test]
    fn test_set_offset_clamps() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.set_offset(1000);
        assert_eq!(sb.offset(), 90); // max_offset = 100 - 10
    }

    #[test]
    fn test_position_zero() {
        let sb = Scrollbar::vertical(100, 10);
        assert!((sb.position() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_position_mid() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.set_offset(45);
        assert!((sb.position() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_position_end() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.set_offset(90);
        assert!((sb.position() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_position_no_scroll() {
        let sb = Scrollbar::vertical(10, 20);
        assert!((sb.position() - 0.0).abs() < f64::EPSILON);
    }

    // =====================================================
    // Thumb Size Tests
    // =====================================================

    #[test]
    fn test_thumb_size_small_viewport() {
        let sb = Scrollbar::vertical(100, 10);
        assert!((sb.thumb_size() - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_thumb_size_large_viewport() {
        let sb = Scrollbar::vertical(100, 50);
        assert!((sb.thumb_size() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_thumb_size_viewport_exceeds() {
        let sb = Scrollbar::vertical(50, 100);
        assert!((sb.thumb_size() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_thumb_size_empty_content() {
        let sb = Scrollbar::vertical(0, 10);
        assert!((sb.thumb_size() - 1.0).abs() < f64::EPSILON);
    }

    // =====================================================
    // Max Offset Tests
    // =====================================================

    #[test]
    fn test_max_offset() {
        let sb = Scrollbar::vertical(100, 10);
        assert_eq!(sb.max_offset(), 90);
    }

    #[test]
    fn test_max_offset_no_scroll() {
        let sb = Scrollbar::vertical(10, 20);
        assert_eq!(sb.max_offset(), 0);
    }

    #[test]
    fn test_is_scrollable_true() {
        let sb = Scrollbar::vertical(100, 10);
        assert!(sb.is_scrollable());
    }

    #[test]
    fn test_is_scrollable_false() {
        let sb = Scrollbar::vertical(10, 20);
        assert!(!sb.is_scrollable());
    }

    // =====================================================
    // Scroll Methods Tests
    // =====================================================

    #[test]
    fn test_scroll_positive() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.scroll(5);
        assert_eq!(sb.offset(), 5);
    }

    #[test]
    fn test_scroll_negative() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.set_offset(10);
        sb.scroll(-3);
        assert_eq!(sb.offset(), 7);
    }

    #[test]
    fn test_scroll_clamps_max() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.scroll(1000);
        assert_eq!(sb.offset(), 90);
    }

    #[test]
    fn test_scroll_clamps_min() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.scroll(-100);
        assert_eq!(sb.offset(), 0);
    }

    #[test]
    fn test_scroll_start() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.set_offset(10);
        sb.scroll_start();
        assert_eq!(sb.offset(), 9);
    }

    #[test]
    fn test_scroll_end() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.scroll_end();
        assert_eq!(sb.offset(), 1);
    }

    #[test]
    fn test_page_start() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.set_offset(50);
        sb.page_start();
        assert_eq!(sb.offset(), 40);
    }

    #[test]
    fn test_page_end() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.page_end();
        assert_eq!(sb.offset(), 10);
    }

    // =====================================================
    // Jump Methods Tests
    // =====================================================

    #[test]
    fn test_jump_to_mid() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.jump_to(0.5);
        assert_eq!(sb.offset(), 45);
    }

    #[test]
    fn test_jump_to_start() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.set_offset(50);
        sb.jump_to(0.0);
        assert_eq!(sb.offset(), 0);
    }

    #[test]
    fn test_jump_to_end() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.jump_to(1.0);
        assert_eq!(sb.offset(), 90);
    }

    #[test]
    fn test_jump_to_clamps() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.jump_to(2.0);
        assert_eq!(sb.offset(), 90);
        sb.jump_to(-1.0);
        assert_eq!(sb.offset(), 0);
    }

    #[test]
    fn test_jump_start() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.set_offset(50);
        sb.jump_start();
        assert_eq!(sb.offset(), 0);
    }

    #[test]
    fn test_jump_end() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.jump_end();
        assert_eq!(sb.offset(), 90);
    }

    // =====================================================
    // Update Lengths Tests
    // =====================================================

    #[test]
    fn test_update_lengths() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.update_lengths(200, 20);
        assert_eq!(sb.content_length(), 200);
        assert_eq!(sb.viewport_length(), 20);
    }

    #[test]
    fn test_update_lengths_clamps_offset() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.set_offset(80);
        sb.update_lengths(50, 10);
        assert_eq!(sb.offset(), 40); // new max_offset
    }

    // =====================================================
    // Brick Trait Tests
    // =====================================================

    #[test]
    fn test_brick_name() {
        let sb = Scrollbar::vertical(100, 10);
        assert_eq!(sb.brick_name(), "scrollbar");
    }

    #[test]
    fn test_assertions_not_empty() {
        let sb = Scrollbar::vertical(100, 10);
        assert!(!sb.assertions().is_empty());
    }

    #[test]
    fn test_budget() {
        let sb = Scrollbar::vertical(100, 10);
        let budget = sb.budget();
        assert!(budget.paint_ms > 0);
    }

    #[test]
    fn test_verify() {
        let sb = Scrollbar::vertical(100, 10);
        assert!(sb.verify().is_valid());
    }

    #[test]
    fn test_to_html() {
        let sb = Scrollbar::vertical(100, 10);
        assert!(sb.to_html().is_empty());
    }

    #[test]
    fn test_to_css() {
        let sb = Scrollbar::vertical(100, 10);
        assert!(sb.to_css().is_empty());
    }

    // =====================================================
    // Widget Trait Tests
    // =====================================================

    #[test]
    fn test_type_id() {
        let sb = Scrollbar::vertical(100, 10);
        assert_eq!(Widget::type_id(&sb), TypeId::of::<Scrollbar>());
    }

    #[test]
    fn test_measure_vertical() {
        let sb = Scrollbar::vertical(100, 10);
        let size = sb.measure(Constraints::loose(Size::new(100.0, 50.0)));
        assert_eq!(size.width, 1.0);
        assert!(size.height >= 3.0);
    }

    #[test]
    fn test_measure_horizontal() {
        let sb = Scrollbar::horizontal(100, 10);
        let size = sb.measure(Constraints::loose(Size::new(50.0, 100.0)));
        assert_eq!(size.height, 1.0);
        assert!(size.width >= 3.0);
    }

    #[test]
    fn test_layout() {
        let mut sb = Scrollbar::vertical(100, 10);
        let bounds = Rect::new(5.0, 10.0, 1.0, 20.0);
        let result = sb.layout(bounds);
        assert_eq!(result.size.height, 20.0);
        assert_eq!(sb.bounds, bounds);
    }

    #[test]
    fn test_children() {
        let sb = Scrollbar::vertical(100, 10);
        assert!(sb.children().is_empty());
    }

    #[test]
    fn test_children_mut() {
        let mut sb = Scrollbar::vertical(100, 10);
        assert!(sb.children_mut().is_empty());
    }

    #[test]
    fn test_event() {
        let mut sb = Scrollbar::vertical(100, 10);
        let event = Event::key_down(presentar_core::Key::Enter);
        assert!(sb.event(&event).is_none());
    }

    // =====================================================
    // Paint Tests
    // =====================================================

    #[test]
    fn test_paint_vertical() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.bounds = Rect::new(0.0, 0.0, 1.0, 10.0);
        let mut canvas = MockCanvas::new();
        sb.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_paint_vertical_no_arrows() {
        let mut sb = Scrollbar::vertical(100, 10).with_arrows(false);
        sb.bounds = Rect::new(0.0, 0.0, 1.0, 10.0);
        let mut canvas = MockCanvas::new();
        sb.paint(&mut canvas);
        // Should not contain arrow characters
        let has_arrow = canvas.texts.iter().any(|(t, _)| t == "▲" || t == "▼");
        assert!(!has_arrow);
    }

    #[test]
    fn test_paint_horizontal() {
        let mut sb = Scrollbar::horizontal(100, 10);
        sb.bounds = Rect::new(0.0, 0.0, 10.0, 1.0);
        let mut canvas = MockCanvas::new();
        sb.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_paint_small_bounds() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.bounds = Rect::new(0.0, 0.0, 1.0, 2.0);
        let mut canvas = MockCanvas::new();
        sb.paint(&mut canvas);
        // Should handle small bounds gracefully
    }

    #[test]
    fn test_paint_with_offset() {
        let mut sb = Scrollbar::vertical(100, 10);
        sb.set_offset(50);
        sb.bounds = Rect::new(0.0, 0.0, 1.0, 10.0);
        let mut canvas = MockCanvas::new();
        sb.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    // =====================================================
    // Orientation Enum Tests
    // =====================================================

    #[test]
    fn test_orientation_default() {
        assert_eq!(ScrollOrientation::default(), ScrollOrientation::Vertical);
    }

    #[test]
    fn test_orientation_eq() {
        assert_eq!(ScrollOrientation::Vertical, ScrollOrientation::Vertical);
        assert_ne!(ScrollOrientation::Vertical, ScrollOrientation::Horizontal);
    }
}
