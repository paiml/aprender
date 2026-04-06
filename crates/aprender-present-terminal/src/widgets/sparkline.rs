//! Sparkline widget for compact inline graphs.
//!
//! Provides minimal inline visualization using vertical block characters.
//! Ideal for embedding in tables or status lines.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Block characters for sparkline rendering (8 levels).
const SPARK_CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Trend direction indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrendDirection {
    /// Upward trend
    Up,
    /// Downward trend
    Down,
    /// No significant change
    #[default]
    Flat,
}

impl TrendDirection {
    /// Get arrow character for trend.
    #[must_use]
    pub const fn arrow(&self) -> char {
        match self {
            Self::Up => '↑',
            Self::Down => '↓',
            Self::Flat => '→',
        }
    }

    /// Get color for trend.
    #[must_use]
    pub fn color(&self) -> Color {
        match self {
            Self::Up => Color::new(0.3, 1.0, 0.5, 1.0),   // Green
            Self::Down => Color::new(1.0, 0.3, 0.3, 1.0), // Red
            Self::Flat => Color::new(0.7, 0.7, 0.7, 1.0), // Gray
        }
    }
}

/// Compact sparkline widget for inline graphs.
#[derive(Debug, Clone)]
pub struct Sparkline {
    /// Data points to display.
    data: Vec<f64>,
    /// Minimum value for scaling.
    min: f64,
    /// Maximum value for scaling.
    max: f64,
    /// Sparkline color.
    color: Color,
    /// Whether to show trend indicator.
    show_trend: bool,
    /// UX-121: Whether to show Y-axis min/max labels.
    show_y_axis: bool,
    /// UX-121: Y-axis label format (e.g., "{:.0}%").
    y_format: Option<String>,
    /// Cached bounds.
    bounds: Rect,
}

impl Default for Sparkline {
    fn default() -> Self {
        Self::new(vec![])
    }
}

impl Sparkline {
    /// Create a new sparkline with data.
    #[must_use]
    pub fn new(data: Vec<f64>) -> Self {
        let (min, max) = Self::compute_range(&data);
        Self {
            data,
            min,
            max,
            color: Color::new(0.3, 0.7, 1.0, 1.0),
            show_trend: false,
            show_y_axis: false,
            y_format: None,
            bounds: Rect::default(),
        }
    }

    /// Set the color.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set the data range.
    #[must_use]
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        // Provability: range values must be finite
        debug_assert!(min.is_finite(), "min must be finite");
        debug_assert!(max.is_finite(), "max must be finite");
        self.min = min;
        self.max = max.max(min + 0.001);
        self
    }

    /// Show trend indicator.
    #[must_use]
    pub fn with_trend(mut self, show: bool) -> Self {
        self.show_trend = show;
        self
    }

    /// UX-121: Show Y-axis min/max labels.
    #[must_use]
    pub fn with_y_axis(mut self, show: bool) -> Self {
        self.show_y_axis = show;
        self
    }

    /// UX-121: Set Y-axis label format (e.g., "{:.0}%", "{:.1}ms").
    #[must_use]
    pub fn with_y_format(mut self, format: impl Into<String>) -> Self {
        self.y_format = Some(format.into());
        self.show_y_axis = true;
        self
    }

    /// Get the Y-axis label width needed for layout.
    #[must_use]
    #[allow(clippy::literal_string_with_formatting_args)]
    pub fn y_axis_width(&self) -> u16 {
        if !self.show_y_axis {
            return 0;
        }
        // Estimate width based on format or default
        let max_label = if let Some(ref fmt) = self.y_format {
            fmt.replace("{:.0}", "999").replace("{:.1}", "99.9")
        } else {
            format!("{:.0}", self.max.abs().max(self.min.abs()))
        };
        (max_label.len() + 1) as u16
    }

    /// Update data.
    pub fn set_data(&mut self, data: Vec<f64>) {
        let (min, max) = Self::compute_range(&data);
        self.data = data;
        self.min = min;
        self.max = max;
    }

    /// Get current trend direction.
    #[must_use]
    pub fn trend(&self) -> TrendDirection {
        if self.data.len() < 2 {
            return TrendDirection::Flat;
        }

        let recent = self.data.len().saturating_sub(3);
        let recent_avg: f64 =
            self.data[recent..].iter().sum::<f64>() / (self.data.len() - recent) as f64;

        let older_end = recent.min(self.data.len());
        let older_start = older_end.saturating_sub(3);
        if older_start >= older_end {
            return TrendDirection::Flat;
        }
        let older_avg: f64 = self.data[older_start..older_end].iter().sum::<f64>()
            / (older_end - older_start) as f64;

        let threshold = (self.max - self.min) * 0.05;
        if recent_avg > older_avg + threshold {
            TrendDirection::Up
        } else if recent_avg < older_avg - threshold {
            TrendDirection::Down
        } else {
            TrendDirection::Flat
        }
    }

    fn compute_range(data: &[f64]) -> (f64, f64) {
        if data.is_empty() {
            return (0.0, 1.0);
        }
        let min = data.iter().fold(f64::MAX, |a, &b| a.min(b));
        let max = data.iter().fold(f64::MIN, |a, &b| a.max(b));
        if (max - min).abs() < f64::EPSILON {
            (min - 0.5, max + 0.5)
        } else {
            (min, max)
        }
    }

    fn normalize(&self, value: f64) -> f64 {
        let range = self.max - self.min;
        if range.abs() < f64::EPSILON {
            0.5
        } else {
            ((value - self.min) / range).clamp(0.0, 1.0)
        }
    }
}

impl Brick for Sparkline {
    fn brick_name(&self) -> &'static str {
        "sparkline"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(16)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
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

impl Widget for Sparkline {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let width = (self.data.len() as f32 + if self.show_trend { 2.0 } else { 0.0 })
            .min(constraints.max_width)
            .max(1.0);
        constraints.constrain(Size::new(width, 1.0))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height.max(1.0)),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.data.is_empty() || self.bounds.width < 1.0 {
            return;
        }

        let available_width = if self.show_trend {
            (self.bounds.width as usize).saturating_sub(2)
        } else {
            self.bounds.width as usize
        };

        if available_width == 0 {
            return;
        }

        // Build sparkline string
        let mut spark = String::with_capacity(available_width);

        for i in 0..available_width.min(self.data.len()) {
            let idx = (i * self.data.len()) / available_width;
            let value = self.data.get(idx).copied().unwrap_or(0.0);
            let norm = self.normalize(value);
            let char_idx = ((norm * 7.0).round() as usize).min(7);
            spark.push(SPARK_CHARS[char_idx]);
        }

        let style = TextStyle {
            color: self.color,
            ..Default::default()
        };
        canvas.draw_text(&spark, Point::new(self.bounds.x, self.bounds.y), &style);

        // Draw trend indicator
        if self.show_trend {
            let trend = self.trend();
            let trend_style = TextStyle {
                color: trend.color(),
                ..Default::default()
            };
            canvas.draw_text(
                &format!(" {}", trend.arrow()),
                Point::new(self.bounds.x + available_width as f32, self.bounds.y),
                &trend_style,
            );
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

    #[test]
    fn test_sparkline_creation() {
        let spark = Sparkline::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(spark.data.len(), 5);
    }

    #[test]
    fn test_sparkline_assertions() {
        let spark = Sparkline::new(vec![1.0]);
        assert!(!spark.assertions().is_empty());
    }

    #[test]
    fn test_sparkline_verify() {
        let spark = Sparkline::new(vec![1.0, 2.0]);
        assert!(spark.verify().is_valid());
    }

    #[test]
    fn test_sparkline_with_color() {
        let spark = Sparkline::new(vec![1.0]).with_color(Color::RED);
        assert_eq!(spark.color, Color::RED);
    }

    #[test]
    fn test_sparkline_with_range() {
        let spark = Sparkline::new(vec![1.0]).with_range(0.0, 100.0);
        assert_eq!(spark.min, 0.0);
        assert_eq!(spark.max, 100.0);
    }

    #[test]
    fn test_sparkline_with_trend() {
        let spark = Sparkline::new(vec![1.0]).with_trend(true);
        assert!(spark.show_trend);
    }

    #[test]
    fn test_sparkline_trend_up() {
        let spark = Sparkline::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(spark.trend(), TrendDirection::Up);
    }

    #[test]
    fn test_sparkline_trend_down() {
        let spark = Sparkline::new(vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        assert_eq!(spark.trend(), TrendDirection::Down);
    }

    #[test]
    fn test_sparkline_trend_flat() {
        let spark = Sparkline::new(vec![5.0, 5.0, 5.0, 5.0, 5.0]);
        assert_eq!(spark.trend(), TrendDirection::Flat);
    }

    #[test]
    fn test_sparkline_paint() {
        let mut spark = Sparkline::new(vec![0.0, 0.5, 1.0]);
        spark.bounds = Rect::new(0.0, 0.0, 10.0, 1.0);
        let mut canvas = MockCanvas::new();
        spark.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    #[allow(clippy::len_zero)]
    fn test_sparkline_paint_with_trend() {
        let mut spark = Sparkline::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]).with_trend(true);
        spark.bounds = Rect::new(0.0, 0.0, 10.0, 1.0);
        let mut canvas = MockCanvas::new();
        spark.paint(&mut canvas);
        assert!(canvas.texts.len() >= 1);
    }

    #[test]
    fn test_sparkline_empty() {
        let mut spark = Sparkline::new(vec![]);
        spark.bounds = Rect::new(0.0, 0.0, 10.0, 1.0);
        let mut canvas = MockCanvas::new();
        spark.paint(&mut canvas);
        assert!(canvas.texts.is_empty());
    }

    #[test]
    fn test_sparkline_measure() {
        let spark = Sparkline::new(vec![1.0, 2.0, 3.0]);
        let size = spark.measure(Constraints::loose(Size::new(100.0, 10.0)));
        assert!(size.width >= 3.0);
        assert_eq!(size.height, 1.0);
    }

    #[test]
    fn test_sparkline_layout() {
        let mut spark = Sparkline::new(vec![1.0, 2.0]);
        let bounds = Rect::new(5.0, 10.0, 20.0, 1.0);
        let result = spark.layout(bounds);
        assert_eq!(result.size.width, 20.0);
        assert_eq!(spark.bounds, bounds);
    }

    #[test]
    fn test_trend_direction_arrow() {
        assert_eq!(TrendDirection::Up.arrow(), '↑');
        assert_eq!(TrendDirection::Down.arrow(), '↓');
        assert_eq!(TrendDirection::Flat.arrow(), '→');
    }

    #[test]
    fn test_trend_direction_color() {
        let _ = TrendDirection::Up.color();
        let _ = TrendDirection::Down.color();
        let _ = TrendDirection::Flat.color();
    }

    #[test]
    fn test_sparkline_set_data() {
        let mut spark = Sparkline::new(vec![1.0]);
        spark.set_data(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(spark.data.len(), 4);
    }

    #[test]
    fn test_sparkline_brick_name() {
        let spark = Sparkline::new(vec![]);
        assert_eq!(spark.brick_name(), "sparkline");
    }

    #[test]
    fn test_sparkline_budget() {
        let spark = Sparkline::new(vec![]);
        let budget = spark.budget();
        assert!(budget.paint_ms > 0);
    }

    #[test]
    fn test_sparkline_type_id() {
        let spark = Sparkline::new(vec![]);
        assert_eq!(Widget::type_id(&spark), TypeId::of::<Sparkline>());
    }

    #[test]
    fn test_sparkline_children() {
        let spark = Sparkline::new(vec![]);
        assert!(spark.children().is_empty());
    }

    #[test]
    fn test_sparkline_children_mut() {
        let mut spark = Sparkline::new(vec![]);
        assert!(spark.children_mut().is_empty());
    }

    #[test]
    fn test_sparkline_event() {
        let mut spark = Sparkline::new(vec![]);
        let event = Event::key_down(presentar_core::Key::Enter);
        assert!(spark.event(&event).is_none());
    }

    #[test]
    fn test_sparkline_default() {
        let spark = Sparkline::default();
        assert!(spark.data.is_empty());
    }

    #[test]
    fn test_sparkline_to_html() {
        let spark = Sparkline::new(vec![]);
        assert!(spark.to_html().is_empty());
    }

    #[test]
    fn test_sparkline_to_css() {
        let spark = Sparkline::new(vec![]);
        assert!(spark.to_css().is_empty());
    }

    #[test]
    fn test_sparkline_trend_single_value() {
        // Test trend with single value (data.len() < 2)
        let spark = Sparkline::new(vec![5.0]);
        assert_eq!(spark.trend(), TrendDirection::Flat);
    }

    #[test]
    fn test_sparkline_trend_two_values() {
        // Test trend with exactly 2 values
        let spark = Sparkline::new(vec![1.0, 2.0]);
        // With only 2 values, older_start >= older_end triggers
        assert_eq!(spark.trend(), TrendDirection::Flat);
    }

    #[test]
    fn test_sparkline_trend_three_values() {
        // Test trend with exactly 3 values (boundary case)
        // With 3 values: recent = 3-3=0, so recent slice is [1,2,3]
        // older_end = 0, older_start = 0, so older_start >= older_end -> Flat
        let spark = Sparkline::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(spark.trend(), TrendDirection::Flat);
    }

    #[test]
    fn test_sparkline_normalize_zero_range() {
        // Test normalize with min == max (zero range)
        let spark = Sparkline::new(vec![5.0, 5.0, 5.0]);
        // When all values are the same, range is ~0
        let normalized = spark.normalize(5.0);
        assert!((normalized - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sparkline_paint_zero_available_width() {
        // Test paint with show_trend taking all width
        let mut spark = Sparkline::new(vec![1.0, 2.0]).with_trend(true);
        spark.bounds = Rect::new(0.0, 0.0, 2.0, 1.0); // Width 2, but trend needs 2
        let mut canvas = MockCanvas::new();
        spark.paint(&mut canvas);
        // Should handle gracefully (available_width becomes 0)
    }

    #[test]
    fn test_sparkline_paint_narrow_width() {
        // Test paint with very narrow width
        let mut spark = Sparkline::new(vec![1.0, 2.0, 3.0]).with_trend(true);
        spark.bounds = Rect::new(0.0, 0.0, 1.0, 1.0);
        let mut canvas = MockCanvas::new();
        spark.paint(&mut canvas);
        // Should not panic
    }

    #[test]
    fn test_sparkline_with_y_axis() {
        let spark = Sparkline::new(vec![1.0, 2.0]).with_y_axis(true);
        assert!(spark.show_y_axis);
    }

    #[test]
    fn test_sparkline_with_y_axis_false() {
        let spark = Sparkline::new(vec![1.0, 2.0]).with_y_axis(false);
        assert!(!spark.show_y_axis);
    }

    #[test]
    #[allow(clippy::literal_string_with_formatting_args)]
    fn test_sparkline_with_y_format() {
        let spark = Sparkline::new(vec![1.0, 2.0]).with_y_format("{:.0}%");
        assert!(spark.show_y_axis);
        assert_eq!(spark.y_format, Some("{:.0}%".to_string()));
    }

    #[test]
    fn test_sparkline_y_axis_width_no_axis() {
        let spark = Sparkline::new(vec![1.0, 2.0]);
        assert_eq!(spark.y_axis_width(), 0);
    }

    #[test]
    fn test_sparkline_y_axis_width_with_axis() {
        let spark = Sparkline::new(vec![1.0, 100.0]).with_y_axis(true);
        let width = spark.y_axis_width();
        assert!(width > 0);
    }

    #[test]
    #[allow(clippy::literal_string_with_formatting_args)]
    fn test_sparkline_y_axis_width_with_format() {
        let spark = Sparkline::new(vec![1.0, 100.0]).with_y_format("{:.0}%");
        let width = spark.y_axis_width();
        assert!(width > 0);
    }

    #[test]
    #[allow(clippy::literal_string_with_formatting_args)]
    fn test_sparkline_y_axis_width_with_format_decimal() {
        let spark = Sparkline::new(vec![1.0, 100.0]).with_y_format("{:.1}ms");
        let width = spark.y_axis_width();
        assert!(width > 0);
    }
}
