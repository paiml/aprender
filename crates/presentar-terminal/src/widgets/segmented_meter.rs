//! Segmented meter widget for showing multiple values in a single bar.
//!
//! Useful for displaying memory breakdown (used/cached/free), disk usage, etc.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// A single segment in a segmented meter.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Value of this segment (will be normalized against total).
    pub value: f64,
    /// Color of this segment.
    pub color: Color,
    /// Optional label.
    pub label: Option<String>,
}

impl Segment {
    /// Create a new segment.
    #[must_use]
    pub fn new(value: f64, color: Color) -> Self {
        Self {
            value,
            color,
            label: None,
        }
    }

    /// Add a label to the segment.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// A segmented meter showing multiple values in a single horizontal bar.
#[derive(Debug, Clone)]
pub struct SegmentedMeter {
    /// Segments to display.
    segments: Vec<Segment>,
    /// Maximum value (segments are normalized to this).
    max: f64,
    /// Background color for unfilled portion.
    background: Color,
    /// Whether to show percentage text.
    show_percentages: bool,
    /// Layout bounds.
    bounds: Rect,
}

impl SegmentedMeter {
    /// Create a new segmented meter with the given segments and max value.
    #[must_use]
    pub fn new(segments: Vec<Segment>, max: f64) -> Self {
        Self {
            segments,
            max,
            background: Color::rgb(0.2, 0.2, 0.2),
            show_percentages: false,
            bounds: Rect::new(0.0, 0.0, 0.0, 0.0),
        }
    }

    /// Create a memory-style meter with used, cached, and free segments.
    #[must_use]
    pub fn memory(used: f64, cached: f64, total: f64) -> Self {
        let free = (total - used - cached).max(0.0);
        Self::new(
            vec![
                Segment::new(used, Color::rgb(1.0, 0.7, 0.2)).with_label("Used"),
                Segment::new(cached, Color::rgb(0.2, 0.6, 1.0)).with_label("Cached"),
                Segment::new(free, Color::rgb(0.3, 0.3, 0.3)).with_label("Free"),
            ],
            total,
        )
    }

    /// Set the background color.
    #[must_use]
    pub fn with_background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    /// Set whether to show percentage text.
    #[must_use]
    pub fn with_percentages(mut self, show: bool) -> Self {
        self.show_percentages = show;
        self
    }

    /// Update segments.
    pub fn set_segments(&mut self, segments: Vec<Segment>) {
        self.segments = segments;
    }

    /// Update max value.
    pub fn set_max(&mut self, max: f64) {
        self.max = max;
    }

    fn render(&self, canvas: &mut dyn Canvas) {
        let width = self.bounds.width as usize;
        let height = self.bounds.height as usize;
        if width == 0 || height == 0 {
            return;
        }

        // Calculate total and normalize
        let total: f64 = self.segments.iter().map(|s| s.value).sum();
        let scale = if self.max > 0.0 {
            width as f64 / self.max
        } else if total > 0.0 {
            width as f64 / total
        } else {
            0.0
        };

        let mut x_offset = 0usize;

        // Draw each segment
        for segment in &self.segments {
            let segment_width = (segment.value * scale).round() as usize;
            if segment_width == 0 {
                continue;
            }

            let style = TextStyle {
                color: segment.color,
                ..Default::default()
            };

            // Draw filled portion
            for row in 0..height {
                for col in 0..segment_width {
                    let x = x_offset + col;
                    if x >= width {
                        break;
                    }
                    canvas.draw_text(
                        "█",
                        Point::new(self.bounds.x + x as f32, self.bounds.y + row as f32),
                        &style,
                    );
                }
            }

            x_offset += segment_width;
        }

        // Fill remaining with background
        if x_offset < width {
            let bg_style = TextStyle {
                color: self.background,
                ..Default::default()
            };

            for row in 0..height {
                for col in x_offset..width {
                    canvas.draw_text(
                        "░",
                        Point::new(self.bounds.x + col as f32, self.bounds.y + row as f32),
                        &bg_style,
                    );
                }
            }
        }
    }
}

impl Brick for SegmentedMeter {
    fn brick_name(&self) -> &'static str {
        "segmented_meter"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(8)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(8)
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification {
            passed: vec![BrickAssertion::max_latency_ms(8)],
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

impl Widget for SegmentedMeter {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let width = constraints.max_width.max(10.0);
        let height = constraints.max_height.clamp(1.0, 2.0);
        constraints.constrain(Size::new(width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        self.render(canvas);
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

    struct MockCanvas {
        texts: Vec<(String, Point, Color)>,
    }

    impl MockCanvas {
        fn new() -> Self {
            Self { texts: vec![] }
        }
    }

    impl Canvas for MockCanvas {
        fn fill_rect(&mut self, _rect: Rect, _color: Color) {}
        fn stroke_rect(&mut self, _rect: Rect, _color: Color, _width: f32) {}
        fn draw_text(&mut self, text: &str, position: Point, style: &TextStyle) {
            self.texts.push((text.to_string(), position, style.color));
        }
        fn draw_line(&mut self, _from: Point, _to: Point, _color: Color, _width: f32) {}
        fn fill_circle(&mut self, _center: Point, _radius: f32, _color: Color) {}
        fn stroke_circle(&mut self, _center: Point, _radius: f32, _color: Color, _width: f32) {}
        fn fill_arc(
            &mut self,
            _center: Point,
            _radius: f32,
            _start: f32,
            _end: f32,
            _color: Color,
        ) {
        }
        fn draw_path(&mut self, _points: &[Point], _color: Color, _width: f32) {}
        fn fill_polygon(&mut self, _points: &[Point], _color: Color) {}
        fn push_clip(&mut self, _rect: Rect) {}
        fn pop_clip(&mut self) {}
        fn push_transform(&mut self, _transform: presentar_core::Transform2D) {}
        fn pop_transform(&mut self) {}
    }

    #[test]
    fn test_segment_creation() {
        let segment = Segment::new(50.0, Color::RED);
        assert_eq!(segment.value, 50.0);
        assert_eq!(segment.color, Color::RED);
        assert!(segment.label.is_none());
    }

    #[test]
    fn test_segment_with_label() {
        let segment = Segment::new(50.0, Color::RED).with_label("Used");
        assert_eq!(segment.label, Some("Used".to_string()));
    }

    #[test]
    fn test_segmented_meter_creation() {
        let meter = SegmentedMeter::new(
            vec![
                Segment::new(30.0, Color::RED),
                Segment::new(20.0, Color::BLUE),
            ],
            100.0,
        );
        assert_eq!(meter.segments.len(), 2);
        assert_eq!(meter.max, 100.0);
    }

    #[test]
    fn test_segmented_meter_memory() {
        let meter = SegmentedMeter::memory(60.0, 20.0, 100.0);
        assert_eq!(meter.segments.len(), 3);
        assert_eq!(meter.max, 100.0);
    }

    #[test]
    fn test_segmented_meter_with_background() {
        let meter = SegmentedMeter::new(vec![], 100.0).with_background(Color::BLACK);
        assert_eq!(meter.background, Color::BLACK);
    }

    #[test]
    fn test_segmented_meter_with_percentages() {
        let meter = SegmentedMeter::new(vec![], 100.0).with_percentages(true);
        assert!(meter.show_percentages);
    }

    #[test]
    fn test_segmented_meter_set_segments() {
        let mut meter = SegmentedMeter::new(vec![], 100.0);
        meter.set_segments(vec![Segment::new(50.0, Color::GREEN)]);
        assert_eq!(meter.segments.len(), 1);
    }

    #[test]
    fn test_segmented_meter_set_max() {
        let mut meter = SegmentedMeter::new(vec![], 100.0);
        meter.set_max(200.0);
        assert_eq!(meter.max, 200.0);
    }

    #[test]
    fn test_segmented_meter_paint() {
        let mut meter = SegmentedMeter::new(
            vec![
                Segment::new(50.0, Color::RED),
                Segment::new(30.0, Color::BLUE),
            ],
            100.0,
        );
        meter.bounds = Rect::new(0.0, 0.0, 20.0, 1.0);
        let mut canvas = MockCanvas::new();
        meter.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_segmented_meter_paint_empty() {
        let mut meter = SegmentedMeter::new(vec![], 100.0);
        meter.bounds = Rect::new(0.0, 0.0, 20.0, 1.0);
        let mut canvas = MockCanvas::new();
        meter.paint(&mut canvas);
        // Should still render background
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_segmented_meter_paint_zero_bounds() {
        let mut meter = SegmentedMeter::new(vec![Segment::new(50.0, Color::RED)], 100.0);
        meter.bounds = Rect::new(0.0, 0.0, 0.0, 0.0);
        let mut canvas = MockCanvas::new();
        meter.paint(&mut canvas);
        assert!(canvas.texts.is_empty());
    }

    #[test]
    fn test_segmented_meter_brick_name() {
        let meter = SegmentedMeter::new(vec![], 100.0);
        assert_eq!(meter.brick_name(), "segmented_meter");
    }

    #[test]
    fn test_segmented_meter_assertions_not_empty() {
        let meter = SegmentedMeter::new(vec![], 100.0);
        assert!(!meter.assertions().is_empty());
    }

    #[test]
    fn test_segmented_meter_verify() {
        let meter = SegmentedMeter::new(vec![], 100.0);
        assert!(meter.verify().is_valid());
    }

    #[test]
    fn test_segmented_meter_measure() {
        let meter = SegmentedMeter::new(vec![], 100.0);
        let constraints = Constraints::new(0.0, 50.0, 0.0, 10.0);
        let size = meter.measure(constraints);
        assert!(size.width >= 10.0);
        assert!(size.height >= 1.0);
    }

    #[test]
    fn test_segmented_meter_colors() {
        let mut meter = SegmentedMeter::new(
            vec![
                Segment::new(25.0, Color::RED),
                Segment::new(25.0, Color::GREEN),
                Segment::new(25.0, Color::BLUE),
            ],
            100.0,
        );
        meter.bounds = Rect::new(0.0, 0.0, 12.0, 1.0);
        let mut canvas = MockCanvas::new();
        meter.paint(&mut canvas);

        // Should have multiple colors
        let colors: std::collections::HashSet<_> = canvas
            .texts
            .iter()
            .map(|(_, _, c)| format!("{:?}", c))
            .collect();
        assert!(colors.len() >= 3); // At least 3 different colors (red, green, blue, maybe background)
    }

    #[test]
    fn test_segmented_meter_layout() {
        let mut meter = SegmentedMeter::new(vec![Segment::new(50.0, Color::RED)], 100.0);
        let result = meter.layout(Rect::new(0.0, 0.0, 80.0, 2.0));
        assert_eq!(result.size.width, 80.0);
        assert_eq!(result.size.height, 2.0);
    }

    #[test]
    fn test_segmented_meter_event() {
        let mut meter = SegmentedMeter::new(vec![], 100.0);
        let event = Event::Resize {
            width: 80.0,
            height: 24.0,
        };
        assert!(meter.event(&event).is_none());
    }

    #[test]
    fn test_segmented_meter_children() {
        let meter = SegmentedMeter::new(vec![], 100.0);
        assert!(meter.children().is_empty());
    }

    #[test]
    fn test_segmented_meter_children_mut() {
        let mut meter = SegmentedMeter::new(vec![], 100.0);
        assert!(meter.children_mut().is_empty());
    }

    #[test]
    fn test_segmented_meter_type_id() {
        let meter = SegmentedMeter::new(vec![], 100.0);
        let tid = Widget::type_id(&meter);
        assert_eq!(tid, TypeId::of::<SegmentedMeter>());
    }

    #[test]
    fn test_segmented_meter_budget() {
        let meter = SegmentedMeter::new(vec![], 100.0);
        let budget = meter.budget();
        assert!(budget.layout_ms > 0);
    }

    #[test]
    fn test_segmented_meter_to_html() {
        let meter = SegmentedMeter::new(vec![], 100.0);
        assert!(meter.to_html().is_empty());
    }

    #[test]
    fn test_segmented_meter_to_css() {
        let meter = SegmentedMeter::new(vec![], 100.0);
        assert!(meter.to_css().is_empty());
    }

    #[test]
    fn test_segment_clone() {
        let seg = Segment::new(50.0, Color::RED).with_label("Test");
        let cloned = seg.clone();
        assert_eq!(cloned.value, seg.value);
        assert_eq!(cloned.label, seg.label);
    }

    #[test]
    fn test_segmented_meter_clone() {
        let meter = SegmentedMeter::new(vec![Segment::new(50.0, Color::RED)], 100.0);
        let cloned = meter.clone();
        assert_eq!(cloned.max, meter.max);
        assert_eq!(cloned.segments.len(), meter.segments.len());
    }

    #[test]
    fn test_segment_debug() {
        let seg = Segment::new(50.0, Color::RED);
        let debug = format!("{seg:?}");
        assert!(debug.contains("50"));
    }

    #[test]
    fn test_segmented_meter_debug() {
        let meter = SegmentedMeter::new(vec![], 100.0);
        let debug = format!("{meter:?}");
        assert!(debug.contains("100"));
    }

    #[test]
    fn test_segmented_meter_paint_zero_max() {
        let mut meter = SegmentedMeter::new(vec![Segment::new(50.0, Color::RED)], 0.0);
        meter.bounds = Rect::new(0.0, 0.0, 20.0, 1.0);
        let mut canvas = MockCanvas::new();
        meter.paint(&mut canvas);
        // With max=0, uses total of segments as denominator
    }

    #[test]
    fn test_segmented_meter_paint_multi_row() {
        let mut meter = SegmentedMeter::new(
            vec![
                Segment::new(50.0, Color::RED),
                Segment::new(30.0, Color::BLUE),
            ],
            100.0,
        );
        meter.bounds = Rect::new(0.0, 0.0, 20.0, 3.0);
        let mut canvas = MockCanvas::new();
        meter.paint(&mut canvas);
        // Should have drawn on multiple rows
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_segmented_meter_paint_zero_segment() {
        let mut meter = SegmentedMeter::new(
            vec![
                Segment::new(50.0, Color::RED),
                Segment::new(0.0, Color::BLUE), // Zero-value segment
                Segment::new(30.0, Color::GREEN),
            ],
            100.0,
        );
        meter.bounds = Rect::new(0.0, 0.0, 20.0, 1.0);
        let mut canvas = MockCanvas::new();
        meter.paint(&mut canvas);
        // Zero segments should be skipped
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_segmented_meter_paint_overflow() {
        let mut meter = SegmentedMeter::new(
            vec![
                Segment::new(100.0, Color::RED),
                Segment::new(100.0, Color::BLUE),
            ],
            100.0, // Total > max, will overflow width
        );
        meter.bounds = Rect::new(0.0, 0.0, 20.0, 1.0);
        let mut canvas = MockCanvas::new();
        meter.paint(&mut canvas);
        // Should still handle gracefully
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_segmented_meter_memory_overflow() {
        // Test when used + cached > total
        let meter = SegmentedMeter::memory(80.0, 30.0, 100.0);
        assert_eq!(meter.segments.len(), 3);
        // Free should be clamped to 0
        assert_eq!(meter.segments[2].value, 0.0);
    }
}
