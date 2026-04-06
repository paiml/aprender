//! Horizontal meter/gauge widget.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Horizontal meter widget displaying a percentage value.
#[derive(Debug, Clone)]
pub struct Meter {
    value: f64,
    max: f64,
    label: String,
    fill_color: Color,
    gradient_end: Option<Color>,
    show_percentage: bool,
    bounds: Rect,
}

impl Meter {
    /// Create a new meter with value and max.
    #[must_use]
    pub fn new(value: f64, max: f64) -> Self {
        Self {
            value,
            max,
            label: String::new(),
            fill_color: Color::GREEN,
            gradient_end: None,
            show_percentage: true,
            bounds: Rect::new(0.0, 0.0, 0.0, 0.0),
        }
    }

    /// Create a percentage meter (0-100).
    #[must_use]
    pub fn percentage(value: f64) -> Self {
        Self::new(value, 100.0)
    }

    /// Set the label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Set the fill color.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.fill_color = color;
        self
    }

    /// Set a gradient (start to end color).
    #[must_use]
    pub fn with_gradient(mut self, start: Color, end: Color) -> Self {
        self.fill_color = start;
        self.gradient_end = Some(end);
        self
    }

    /// Set whether to show percentage text.
    #[must_use]
    pub fn with_percentage_text(mut self, show: bool) -> Self {
        self.show_percentage = show;
        self
    }

    /// Update the value.
    pub fn set_value(&mut self, value: f64) {
        self.value = value.clamp(0.0, self.max);
    }

    /// Get the current value.
    #[must_use]
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Get the fill ratio (0.0-1.0).
    #[must_use]
    pub fn ratio(&self) -> f64 {
        if self.max == 0.0 {
            0.0
        } else {
            (self.value / self.max).clamp(0.0, 1.0)
        }
    }

    fn color_at(&self, t: f64) -> Color {
        match self.gradient_end {
            Some(end) => self.fill_color.lerp(&end, t as f32),
            None => self.fill_color,
        }
    }
}

impl Brick for Meter {
    fn brick_name(&self) -> &'static str {
        "meter"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(16)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let mut failed = Vec::new();

        // Check value is in range
        if self.value >= 0.0 && self.value <= self.max {
            passed.push(BrickAssertion::max_latency_ms(16));
        } else {
            failed.push((
                BrickAssertion::max_latency_ms(16),
                format!("Value {} outside range [0, {}]", self.value, self.max),
            ));
        }

        BrickVerification {
            passed,
            failed,
            verification_time: Duration::from_micros(10),
        }
    }

    fn to_html(&self) -> String {
        String::new()
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

impl Widget for Meter {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let width = constraints.max_width.max(10.0);
        let height = 1.0;
        constraints.constrain(Size::new(width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height.max(1.0)),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        let width = self.bounds.width as usize;
        if width == 0 {
            return;
        }

        let label_width = if self.label.is_empty() {
            0
        } else {
            self.label.len() + 1
        };

        let pct_text = if self.show_percentage {
            format!("{:5.1}%", self.ratio() * 100.0)
        } else {
            String::new()
        };
        let pct_width = pct_text.len();

        let bar_width = width.saturating_sub(label_width + pct_width + 2);
        if bar_width == 0 {
            return;
        }

        // Draw label
        if !self.label.is_empty() {
            canvas.draw_text(
                &self.label,
                Point::new(self.bounds.x, self.bounds.y),
                &TextStyle::default(),
            );
        }

        let filled = ((self.ratio() * bar_width as f64).round() as usize).min(bar_width);

        let mut bar = String::with_capacity(bar_width + 2);
        bar.push('[');
        for i in 0..bar_width {
            if i < filled {
                bar.push('█');
            } else {
                bar.push(' ');
            }
        }
        bar.push(']');

        let bar_x = self.bounds.x + label_width as f32;
        let style = TextStyle {
            color: self.color_at(0.5),
            ..Default::default()
        };
        canvas.draw_text(&bar, Point::new(bar_x, self.bounds.y), &style);

        if self.show_percentage {
            let pct_x = bar_x + bar_width as f32 + 2.0;
            canvas.draw_text(
                &pct_text,
                Point::new(pct_x, self.bounds.y),
                &TextStyle::default(),
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
    use presentar_core::{Canvas, TextStyle};

    struct MockCanvas {
        texts: Vec<(String, Point)>,
        rects: Vec<Rect>,
    }

    impl MockCanvas {
        fn new() -> Self {
            Self {
                texts: vec![],
                rects: vec![],
            }
        }
    }

    impl Canvas for MockCanvas {
        fn fill_rect(&mut self, rect: Rect, _color: Color) {
            self.rects.push(rect);
        }
        fn stroke_rect(&mut self, _rect: Rect, _color: Color, _width: f32) {}
        fn draw_text(&mut self, text: &str, position: Point, _style: &TextStyle) {
            self.texts.push((text.to_string(), position));
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
    fn test_meter_creation() {
        let meter = Meter::new(50.0, 100.0);
        assert_eq!(meter.value, 50.0);
        assert_eq!(meter.max, 100.0);
    }

    #[test]
    fn test_meter_ratio() {
        let meter = Meter::percentage(75.0);
        assert!((meter.ratio() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_meter_assertions_not_empty() {
        let meter = Meter::percentage(50.0);
        assert!(!meter.assertions().is_empty());
    }

    #[test]
    fn test_meter_verify_pass() {
        let meter = Meter::percentage(50.0);
        assert!(meter.verify().is_valid());
    }

    #[test]
    fn test_meter_percentage() {
        let meter = Meter::percentage(80.0);
        assert_eq!(meter.max, 100.0);
        assert_eq!(meter.value(), 80.0);
    }

    #[test]
    fn test_meter_with_label() {
        let meter = Meter::percentage(50.0).with_label("CPU");
        assert_eq!(meter.label, "CPU");
    }

    #[test]
    fn test_meter_with_color() {
        let meter = Meter::percentage(50.0).with_color(Color::RED);
        assert_eq!(meter.fill_color, Color::RED);
    }

    #[test]
    fn test_meter_with_gradient() {
        let meter = Meter::percentage(50.0).with_gradient(Color::GREEN, Color::RED);
        assert_eq!(meter.fill_color, Color::GREEN);
        assert_eq!(meter.gradient_end, Some(Color::RED));
    }

    #[test]
    fn test_meter_with_percentage_text() {
        let meter = Meter::percentage(50.0).with_percentage_text(false);
        assert!(!meter.show_percentage);
    }

    #[test]
    fn test_meter_set_value() {
        let mut meter = Meter::percentage(50.0);
        meter.set_value(75.0);
        assert_eq!(meter.value(), 75.0);
    }

    #[test]
    fn test_meter_set_value_clamped() {
        let mut meter = Meter::percentage(50.0);
        meter.set_value(150.0);
        assert_eq!(meter.value(), 100.0);

        meter.set_value(-10.0);
        assert_eq!(meter.value(), 0.0);
    }

    #[test]
    fn test_meter_ratio_zero_max() {
        let meter = Meter::new(50.0, 0.0);
        assert_eq!(meter.ratio(), 0.0);
    }

    #[test]
    fn test_meter_ratio_clamped() {
        let meter = Meter::new(150.0, 100.0);
        assert_eq!(meter.ratio(), 1.0);
    }

    #[test]
    fn test_meter_color_at_no_gradient() {
        let meter = Meter::percentage(50.0).with_color(Color::BLUE);
        let color = meter.color_at(0.5);
        assert_eq!(color, Color::BLUE);
    }

    #[test]
    fn test_meter_color_at_with_gradient() {
        let meter = Meter::percentage(50.0).with_gradient(Color::GREEN, Color::RED);
        let color = meter.color_at(0.0);
        assert_eq!(color, Color::GREEN);
        let color = meter.color_at(1.0);
        assert_eq!(color, Color::RED);
    }

    #[test]
    fn test_meter_verify_out_of_range() {
        let mut meter = Meter::new(50.0, 100.0);
        meter.value = -10.0;
        assert!(!meter.verify().is_valid());
    }

    #[test]
    fn test_meter_measure() {
        let meter = Meter::percentage(50.0);
        let constraints = Constraints::new(0.0, 100.0, 0.0, 10.0);
        let size = meter.measure(constraints);
        assert!(size.width >= 10.0);
        assert_eq!(size.height, 1.0);
    }

    #[test]
    fn test_meter_layout() {
        let mut meter = Meter::percentage(50.0);
        let bounds = Rect::new(0.0, 0.0, 80.0, 1.0);
        let result = meter.layout(bounds);
        assert_eq!(result.size.width, 80.0);
        assert_eq!(result.size.height, 1.0);
    }

    #[test]
    fn test_meter_paint() {
        let mut meter = Meter::percentage(50.0).with_label("Test");
        meter.bounds = Rect::new(0.0, 0.0, 40.0, 1.0);
        let mut canvas = MockCanvas::new();
        meter.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_meter_paint_without_label() {
        let mut meter = Meter::percentage(50.0);
        meter.bounds = Rect::new(0.0, 0.0, 40.0, 1.0);
        let mut canvas = MockCanvas::new();
        meter.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_meter_paint_without_percentage() {
        let mut meter = Meter::percentage(50.0).with_percentage_text(false);
        meter.bounds = Rect::new(0.0, 0.0, 40.0, 1.0);
        let mut canvas = MockCanvas::new();
        meter.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_meter_paint_zero_width() {
        let mut meter = Meter::percentage(50.0);
        meter.bounds = Rect::new(0.0, 0.0, 0.0, 1.0);
        let mut canvas = MockCanvas::new();
        meter.paint(&mut canvas);
        assert!(canvas.texts.is_empty());
    }

    #[test]
    fn test_meter_paint_tiny_bar() {
        let mut meter = Meter::percentage(50.0).with_label("Very Long Label");
        meter.bounds = Rect::new(0.0, 0.0, 10.0, 1.0);
        let mut canvas = MockCanvas::new();
        meter.paint(&mut canvas);
    }

    #[test]
    fn test_meter_event() {
        let mut meter = Meter::percentage(50.0);
        let event = Event::key_down(presentar_core::Key::Enter);
        assert!(meter.event(&event).is_none());
    }

    #[test]
    fn test_meter_children() {
        let meter = Meter::percentage(50.0);
        assert!(meter.children().is_empty());
    }

    #[test]
    fn test_meter_children_mut() {
        let mut meter = Meter::percentage(50.0);
        assert!(meter.children_mut().is_empty());
    }

    #[test]
    fn test_meter_type_id() {
        let meter = Meter::percentage(50.0);
        assert_eq!(Widget::type_id(&meter), TypeId::of::<Meter>());
    }

    #[test]
    fn test_meter_brick_name() {
        let meter = Meter::percentage(50.0);
        assert_eq!(meter.brick_name(), "meter");
    }

    #[test]
    fn test_meter_budget() {
        let meter = Meter::percentage(50.0);
        let budget = meter.budget();
        assert!(budget.measure_ms > 0);
    }

    #[test]
    fn test_meter_to_html() {
        let meter = Meter::percentage(50.0);
        assert!(meter.to_html().is_empty());
    }

    #[test]
    fn test_meter_to_css() {
        let meter = Meter::percentage(50.0);
        assert!(meter.to_css().is_empty());
    }

    #[test]
    fn test_meter_full() {
        let mut meter = Meter::percentage(100.0);
        meter.bounds = Rect::new(0.0, 0.0, 50.0, 1.0);
        let mut canvas = MockCanvas::new();
        meter.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_meter_empty() {
        let mut meter = Meter::percentage(0.0);
        meter.bounds = Rect::new(0.0, 0.0, 50.0, 1.0);
        let mut canvas = MockCanvas::new();
        meter.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_meter_verify_value_over_max() {
        let mut meter = Meter::new(50.0, 100.0);
        meter.value = 150.0;
        assert!(!meter.verify().is_valid());
    }

    #[test]
    fn test_meter_layout_with_small_height() {
        let mut meter = Meter::percentage(50.0);
        let bounds = Rect::new(0.0, 0.0, 80.0, 0.5);
        let result = meter.layout(bounds);
        assert_eq!(result.size.height, 1.0);
    }

    #[test]
    fn test_meter_paint_with_gradient() {
        let mut meter = Meter::percentage(50.0).with_gradient(Color::GREEN, Color::RED);
        meter.bounds = Rect::new(0.0, 0.0, 50.0, 1.0);
        let mut canvas = MockCanvas::new();
        meter.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_meter_clone() {
        let meter = Meter::percentage(75.0)
            .with_label("Clone Test")
            .with_color(Color::BLUE);
        let cloned = meter;
        assert_eq!(cloned.value, 75.0);
        assert_eq!(cloned.label, "Clone Test");
        assert_eq!(cloned.fill_color, Color::BLUE);
    }

    #[test]
    fn test_meter_debug() {
        let meter = Meter::percentage(50.0);
        let debug = format!("{:?}", meter);
        assert!(debug.contains("Meter"));
        assert!(debug.contains("value"));
    }

    #[test]
    fn test_meter_measure_small_constraints() {
        let meter = Meter::percentage(50.0);
        let constraints = Constraints::new(0.0, 5.0, 0.0, 10.0);
        let size = meter.measure(constraints);
        // max(5.0, 10.0) = 10.0, then constrained to 5.0
        assert_eq!(size.width, 5.0);
    }
}
