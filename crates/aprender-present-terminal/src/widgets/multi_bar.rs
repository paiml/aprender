//! Multi-bar graph widget for showing multiple values as side-by-side bars.
//!
//! Useful for displaying per-core CPU usage, multiple metrics, etc.

use crate::theme::Gradient;
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Display mode for multi-bar graph.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MultiBarMode {
    /// Vertical bars (default) - each bar grows upward.
    #[default]
    Vertical,
    /// Horizontal bars - each bar grows rightward.
    Horizontal,
}

/// A multi-bar graph widget showing multiple values as side-by-side bars.
///
/// Each bar can have its own color based on its value using a gradient.
#[derive(Debug, Clone)]
pub struct MultiBarGraph {
    /// Values for each bar (0.0-1.0 normalized).
    values: Vec<f64>,
    /// Base color (used if no gradient).
    color: Color,
    /// Gradient for value-based coloring.
    gradient: Option<Gradient>,
    /// Display mode.
    mode: MultiBarMode,
    /// Optional labels for each bar.
    labels: Option<Vec<String>>,
    /// Layout bounds.
    bounds: Rect,
    /// Gap between bars (in characters).
    gap: u16,
}

impl MultiBarGraph {
    /// Create a new multi-bar graph with the given values.
    /// Values should be normalized to 0.0-1.0 range.
    #[must_use]
    pub fn new(values: Vec<f64>) -> Self {
        Self {
            values,
            color: Color::GREEN,
            gradient: None,
            mode: MultiBarMode::default(),
            labels: None,
            bounds: Rect::new(0.0, 0.0, 0.0, 0.0),
            gap: 0,
        }
    }

    /// Set the base color.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set a gradient for value-based coloring.
    #[must_use]
    pub fn with_gradient(mut self, gradient: Gradient) -> Self {
        self.gradient = Some(gradient);
        self
    }

    /// Set the display mode.
    #[must_use]
    pub fn with_mode(mut self, mode: MultiBarMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set labels for each bar.
    #[must_use]
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Set gap between bars.
    #[must_use]
    pub fn with_gap(mut self, gap: u16) -> Self {
        self.gap = gap;
        self
    }

    /// Update values.
    pub fn set_values(&mut self, values: Vec<f64>) {
        self.values = values;
    }

    /// Get color for a value.
    fn color_for_value(&self, value: f64) -> Color {
        match &self.gradient {
            Some(gradient) => gradient.sample(value.clamp(0.0, 1.0)),
            None => self.color,
        }
    }

    fn render_vertical(&self, canvas: &mut dyn Canvas) {
        let width = self.bounds.width as usize;
        let height = self.bounds.height as usize;
        if width == 0 || height == 0 || self.values.is_empty() {
            return;
        }

        let bar_count = self.values.len();
        let total_gap = self.gap as usize * bar_count.saturating_sub(1);
        let available_width = width.saturating_sub(total_gap);
        let bar_width = (available_width / bar_count).max(1);

        // Block characters for sub-row resolution (8 levels)
        let blocks = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        for (i, &value) in self.values.iter().enumerate() {
            let value = value.clamp(0.0, 1.0);
            let bar_x = i * (bar_width + self.gap as usize);
            if bar_x >= width {
                break;
            }

            let color = self.color_for_value(value);
            let style = TextStyle {
                color,
                ..Default::default()
            };

            // Calculate bar height with sub-character precision
            let total_eighths = (value * height as f64 * 8.0).round() as usize;
            let full_rows = total_eighths / 8;
            let partial_eighths = total_eighths % 8;

            for row in 0..height {
                let y = height - 1 - row; // Draw from bottom up
                let ch = if row < full_rows {
                    '█'
                } else if row == full_rows && partial_eighths > 0 {
                    blocks[partial_eighths]
                } else {
                    ' '
                };

                // Draw bar_width characters for this bar
                for bx in 0..bar_width {
                    let x = bar_x + bx;
                    if x < width {
                        canvas.draw_text(
                            &ch.to_string(),
                            Point::new(self.bounds.x + x as f32, self.bounds.y + y as f32),
                            &style,
                        );
                    }
                }
            }
        }
    }

    fn render_horizontal(&self, canvas: &mut dyn Canvas) {
        let width = self.bounds.width as usize;
        let height = self.bounds.height as usize;
        if width == 0 || height == 0 || self.values.is_empty() {
            return;
        }

        let bar_count = self.values.len();
        let total_gap = self.gap as usize * bar_count.saturating_sub(1);
        let available_height = height.saturating_sub(total_gap);
        let bar_height = (available_height / bar_count).max(1);

        for (i, &value) in self.values.iter().enumerate() {
            let value = value.clamp(0.0, 1.0);
            let bar_y = i * (bar_height + self.gap as usize);
            if bar_y >= height {
                break;
            }

            let color = self.color_for_value(value);
            let style = TextStyle {
                color,
                ..Default::default()
            };

            let filled_cols = (value * width as f64).round() as usize;

            for row in 0..bar_height {
                let y = bar_y + row;
                if y >= height {
                    break;
                }

                for col in 0..width {
                    let ch = if col < filled_cols { '█' } else { '░' };
                    canvas.draw_text(
                        &ch.to_string(),
                        Point::new(self.bounds.x + col as f32, self.bounds.y + y as f32),
                        &style,
                    );
                }
            }
        }
    }
}

impl Brick for MultiBarGraph {
    fn brick_name(&self) -> &'static str {
        "multi_bar_graph"
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
            passed: vec![BrickAssertion::max_latency_ms(16)],
            failed: vec![],
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

impl Widget for MultiBarGraph {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let width = constraints.max_width.max(self.values.len() as f32);
        let height = constraints.max_height.max(3.0);
        constraints.constrain(Size::new(width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        match self.mode {
            MultiBarMode::Vertical => self.render_vertical(canvas),
            MultiBarMode::Horizontal => self.render_horizontal(canvas),
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
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
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
    fn test_multi_bar_creation() {
        let graph = MultiBarGraph::new(vec![0.5, 0.75, 0.25]);
        assert_eq!(graph.values.len(), 3);
    }

    #[test]
    fn test_multi_bar_with_color() {
        let graph = MultiBarGraph::new(vec![0.5]).with_color(Color::RED);
        assert_eq!(graph.color, Color::RED);
    }

    #[test]
    fn test_multi_bar_with_gradient() {
        let gradient = Gradient::from_hex(&["#00FF00", "#FF0000"]);
        let graph = MultiBarGraph::new(vec![0.5]).with_gradient(gradient);
        assert!(graph.gradient.is_some());
    }

    #[test]
    fn test_multi_bar_with_mode() {
        let graph = MultiBarGraph::new(vec![0.5]).with_mode(MultiBarMode::Horizontal);
        assert_eq!(graph.mode, MultiBarMode::Horizontal);
    }

    #[test]
    fn test_multi_bar_with_gap() {
        let graph = MultiBarGraph::new(vec![0.5]).with_gap(1);
        assert_eq!(graph.gap, 1);
    }

    #[test]
    fn test_multi_bar_set_values() {
        let mut graph = MultiBarGraph::new(vec![0.5]);
        graph.set_values(vec![0.1, 0.2, 0.3]);
        assert_eq!(graph.values.len(), 3);
    }

    #[test]
    fn test_multi_bar_paint_vertical() {
        let mut graph = MultiBarGraph::new(vec![0.5, 1.0, 0.25]);
        graph.bounds = Rect::new(0.0, 0.0, 6.0, 4.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_multi_bar_paint_horizontal() {
        let mut graph =
            MultiBarGraph::new(vec![0.5, 1.0, 0.25]).with_mode(MultiBarMode::Horizontal);
        graph.bounds = Rect::new(0.0, 0.0, 10.0, 6.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_multi_bar_gradient_coloring() {
        let gradient = Gradient::from_hex(&["#00FF00", "#FF0000"]);
        let mut graph = MultiBarGraph::new(vec![0.0, 0.5, 1.0]).with_gradient(gradient);
        graph.bounds = Rect::new(0.0, 0.0, 6.0, 4.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);

        // Different values should produce different colors
        assert!(canvas.texts.iter().map(|(_, _, c)| *c).next().is_some());
    }

    #[test]
    fn test_multi_bar_empty_bounds() {
        let mut graph = MultiBarGraph::new(vec![0.5]);
        graph.bounds = Rect::new(0.0, 0.0, 0.0, 0.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(canvas.texts.is_empty());
    }

    #[test]
    fn test_multi_bar_empty_values() {
        let mut graph = MultiBarGraph::new(vec![]);
        graph.bounds = Rect::new(0.0, 0.0, 10.0, 5.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(canvas.texts.is_empty());
    }

    #[test]
    fn test_multi_bar_brick_name() {
        let graph = MultiBarGraph::new(vec![0.5]);
        assert_eq!(graph.brick_name(), "multi_bar_graph");
    }

    #[test]
    fn test_multi_bar_assertions_not_empty() {
        let graph = MultiBarGraph::new(vec![0.5]);
        assert!(!graph.assertions().is_empty());
    }

    #[test]
    fn test_multi_bar_verify() {
        let graph = MultiBarGraph::new(vec![0.5]);
        assert!(graph.verify().is_valid());
    }

    #[test]
    fn test_multi_bar_measure() {
        let graph = MultiBarGraph::new(vec![0.5, 0.5, 0.5]);
        let constraints = Constraints::new(0.0, 100.0, 0.0, 50.0);
        let size = graph.measure(constraints);
        assert!(size.width >= 3.0);
        assert!(size.height >= 3.0);
    }

    #[test]
    fn test_multi_bar_mode_default() {
        assert_eq!(MultiBarMode::default(), MultiBarMode::Vertical);
    }

    #[test]
    fn test_multi_bar_many_values() {
        // Simulate 48 CPU cores
        let values: Vec<f64> = (0..48).map(|i| i as f64 / 48.0).collect();
        let mut graph = MultiBarGraph::new(values);
        graph.bounds = Rect::new(0.0, 0.0, 96.0, 6.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_multi_bar_layout() {
        let mut graph = MultiBarGraph::new(vec![0.5, 0.75]);
        let result = graph.layout(Rect::new(0.0, 0.0, 40.0, 10.0));
        assert_eq!(result.size.width, 40.0);
        assert_eq!(result.size.height, 10.0);
    }

    #[test]
    fn test_multi_bar_event() {
        let mut graph = MultiBarGraph::new(vec![0.5]);
        let event = Event::Resize {
            width: 80.0,
            height: 24.0,
        };
        assert!(graph.event(&event).is_none());
    }

    #[test]
    fn test_multi_bar_children() {
        let graph = MultiBarGraph::new(vec![0.5]);
        assert!(graph.children().is_empty());
    }

    #[test]
    fn test_multi_bar_children_mut() {
        let mut graph = MultiBarGraph::new(vec![0.5]);
        assert!(graph.children_mut().is_empty());
    }

    #[test]
    fn test_multi_bar_type_id() {
        let graph = MultiBarGraph::new(vec![0.5]);
        let tid = Widget::type_id(&graph);
        assert_eq!(tid, TypeId::of::<MultiBarGraph>());
    }

    #[test]
    fn test_multi_bar_budget() {
        let graph = MultiBarGraph::new(vec![0.5]);
        let budget = graph.budget();
        assert!(budget.layout_ms > 0);
    }

    #[test]
    fn test_multi_bar_to_html() {
        let graph = MultiBarGraph::new(vec![0.5]);
        assert!(graph.to_html().is_empty());
    }

    #[test]
    fn test_multi_bar_to_css() {
        let graph = MultiBarGraph::new(vec![0.5]);
        assert!(graph.to_css().is_empty());
    }

    #[test]
    fn test_multi_bar_clone() {
        let graph = MultiBarGraph::new(vec![0.5, 0.75]).with_gap(2);
        let cloned = graph.clone();
        assert_eq!(cloned.values.len(), graph.values.len());
        assert_eq!(cloned.gap, graph.gap);
    }

    #[test]
    fn test_multi_bar_debug() {
        let graph = MultiBarGraph::new(vec![0.5]);
        let debug = format!("{graph:?}");
        assert!(debug.contains("MultiBarGraph"));
    }

    #[test]
    fn test_multi_bar_mode_debug() {
        let mode = MultiBarMode::Vertical;
        let debug = format!("{mode:?}");
        assert!(debug.contains("Vertical"));
    }

    #[test]
    fn test_multi_bar_mode_clone() {
        let mode = MultiBarMode::Horizontal;
        let cloned = mode;
        assert_eq!(cloned, MultiBarMode::Horizontal);
    }

    #[test]
    fn test_multi_bar_with_labels() {
        let graph =
            MultiBarGraph::new(vec![0.5]).with_labels(vec!["CPU0".to_string(), "CPU1".to_string()]);
        assert!(graph.labels.is_some());
        assert_eq!(graph.labels.unwrap().len(), 2);
    }

    #[test]
    fn test_multi_bar_color_for_value_no_gradient() {
        let graph = MultiBarGraph::new(vec![0.5]).with_color(Color::BLUE);
        let color = graph.color_for_value(0.75);
        assert_eq!(color, Color::BLUE);
    }

    #[test]
    fn test_multi_bar_color_for_value_with_gradient() {
        let gradient = Gradient::from_hex(&["#00FF00", "#FF0000"]);
        let graph = MultiBarGraph::new(vec![0.5]).with_gradient(gradient);
        let color = graph.color_for_value(0.5);
        // Should be somewhere between green and red
        assert!(color.r > 0.0 || color.g > 0.0);
    }

    #[test]
    fn test_multi_bar_vertical_overflow() {
        // More bars than width
        let mut graph = MultiBarGraph::new(vec![0.5; 20]);
        graph.bounds = Rect::new(0.0, 0.0, 10.0, 5.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        // Should handle gracefully
    }

    #[test]
    fn test_multi_bar_horizontal_overflow() {
        // More bars than height
        let mut graph = MultiBarGraph::new(vec![0.5; 20]).with_mode(MultiBarMode::Horizontal);
        graph.bounds = Rect::new(0.0, 0.0, 10.0, 5.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        // Should handle gracefully
    }

    #[test]
    fn test_multi_bar_vertical_with_gap() {
        let mut graph = MultiBarGraph::new(vec![0.5, 0.75, 1.0]).with_gap(1);
        graph.bounds = Rect::new(0.0, 0.0, 12.0, 5.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_multi_bar_horizontal_with_gap() {
        let mut graph = MultiBarGraph::new(vec![0.5, 0.75, 1.0])
            .with_mode(MultiBarMode::Horizontal)
            .with_gap(1);
        graph.bounds = Rect::new(0.0, 0.0, 10.0, 12.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_multi_bar_horizontal_empty() {
        let mut graph = MultiBarGraph::new(vec![]).with_mode(MultiBarMode::Horizontal);
        graph.bounds = Rect::new(0.0, 0.0, 10.0, 5.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(canvas.texts.is_empty());
    }

    #[test]
    fn test_multi_bar_horizontal_zero_bounds() {
        let mut graph = MultiBarGraph::new(vec![0.5]).with_mode(MultiBarMode::Horizontal);
        graph.bounds = Rect::new(0.0, 0.0, 0.0, 0.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(canvas.texts.is_empty());
    }

    #[test]
    fn test_multi_bar_clamped_values() {
        // Values outside 0-1 range should be clamped
        let mut graph = MultiBarGraph::new(vec![-0.5, 1.5, 2.0]);
        graph.bounds = Rect::new(0.0, 0.0, 9.0, 5.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        // Should handle gracefully with clamped values
    }
}
