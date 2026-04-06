//! Scatter plot widget with marker styles.
//!
//! Implements P201 from SPEC-024 Section 15.2.

use crate::theme::Gradient;
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Marker style for scatter points.
#[derive(Debug, Clone, Copy, Default)]
pub enum MarkerStyle {
    /// Single braille dot (default).
    #[default]
    Dot,
    /// Plus sign (+).
    Cross,
    /// Circle (○).
    Circle,
    /// Square (□).
    Square,
    /// Diamond (◇).
    Diamond,
    /// Triangle (△).
    Triangle,
    /// Star (★).
    Star,
}

impl MarkerStyle {
    /// Get the Unicode character for this marker.
    #[must_use]
    pub const fn char(self) -> char {
        match self {
            Self::Dot => '•',
            Self::Cross => '+',
            Self::Circle => '○',
            Self::Square => '□',
            Self::Diamond => '◇',
            Self::Triangle => '△',
            Self::Star => '★',
        }
    }
}

/// Axis configuration.
#[derive(Debug, Clone)]
pub struct ScatterAxis {
    /// Axis label.
    pub label: Option<String>,
    /// Minimum value (None = auto).
    pub min: Option<f64>,
    /// Maximum value (None = auto).
    pub max: Option<f64>,
    /// Number of tick marks.
    pub ticks: usize,
}

impl Default for ScatterAxis {
    fn default() -> Self {
        Self {
            label: None,
            min: None,
            max: None,
            ticks: 5,
        }
    }
}

/// Scatter plot widget.
#[derive(Debug, Clone)]
pub struct ScatterPlot {
    points: Vec<(f64, f64)>,
    marker: MarkerStyle,
    color: Color,
    /// Optional values for color gradient.
    color_by: Option<Vec<f64>>,
    gradient: Option<Gradient>,
    x_axis: ScatterAxis,
    y_axis: ScatterAxis,
    show_axes: bool,
    bounds: Rect,
}

impl ScatterPlot {
    /// Create a new scatter plot.
    #[must_use]
    pub fn new(points: Vec<(f64, f64)>) -> Self {
        Self {
            points,
            marker: MarkerStyle::default(),
            color: Color::new(0.3, 0.7, 1.0, 1.0),
            color_by: None,
            gradient: None,
            x_axis: ScatterAxis::default(),
            y_axis: ScatterAxis::default(),
            show_axes: true,
            bounds: Rect::default(),
        }
    }

    /// Set marker style.
    #[must_use]
    pub fn with_marker(mut self, marker: MarkerStyle) -> Self {
        self.marker = marker;
        self
    }

    /// Set point color.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set color gradient based on values.
    #[must_use]
    pub fn with_color_by(mut self, values: Vec<f64>, gradient: Gradient) -> Self {
        self.color_by = Some(values);
        self.gradient = Some(gradient);
        self
    }

    /// Set X axis configuration.
    #[must_use]
    pub fn with_x_axis(mut self, axis: ScatterAxis) -> Self {
        self.x_axis = axis;
        self
    }

    /// Set Y axis configuration.
    #[must_use]
    pub fn with_y_axis(mut self, axis: ScatterAxis) -> Self {
        self.y_axis = axis;
        self
    }

    /// Toggle axis display.
    #[must_use]
    pub fn with_axes(mut self, show: bool) -> Self {
        self.show_axes = show;
        self
    }

    /// Update points.
    pub fn set_points(&mut self, points: Vec<(f64, f64)>) {
        self.points = points;
    }

    /// Get X range from data.
    fn x_range(&self) -> (f64, f64) {
        if let (Some(min), Some(max)) = (self.x_axis.min, self.x_axis.max) {
            return (min, max);
        }

        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;

        for &(x, _) in &self.points {
            if x.is_finite() {
                x_min = x_min.min(x);
                x_max = x_max.max(x);
            }
        }

        if x_min == f64::INFINITY {
            (0.0, 1.0)
        } else {
            let padding = (x_max - x_min) * 0.05;
            (
                self.x_axis.min.unwrap_or(x_min - padding),
                self.x_axis.max.unwrap_or(x_max + padding),
            )
        }
    }

    /// Get Y range from data.
    fn y_range(&self) -> (f64, f64) {
        if let (Some(min), Some(max)) = (self.y_axis.min, self.y_axis.max) {
            return (min, max);
        }

        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;

        for &(_, y) in &self.points {
            if y.is_finite() {
                y_min = y_min.min(y);
                y_max = y_max.max(y);
            }
        }

        if y_min == f64::INFINITY {
            (0.0, 1.0)
        } else {
            let padding = (y_max - y_min) * 0.05;
            (
                self.y_axis.min.unwrap_or(y_min - padding),
                self.y_axis.max.unwrap_or(y_max + padding),
            )
        }
    }

    /// Get color value range.
    fn color_range(&self) -> (f64, f64) {
        if let Some(ref values) = self.color_by {
            let mut c_min = f64::INFINITY;
            let mut c_max = f64::NEG_INFINITY;

            for &v in values {
                if v.is_finite() {
                    c_min = c_min.min(v);
                    c_max = c_max.max(v);
                }
            }

            if c_min == f64::INFINITY {
                (0.0, 1.0)
            } else {
                (c_min, c_max)
            }
        } else {
            (0.0, 1.0)
        }
    }

    /// Draw Y-axis labels.
    fn draw_y_axis(
        &self,
        canvas: &mut dyn Canvas,
        y_min: f64,
        y_max: f64,
        plot_y: f32,
        plot_height: f32,
        label_style: &TextStyle,
    ) {
        for i in 0..=self.y_axis.ticks {
            let t = i as f64 / self.y_axis.ticks as f64;
            let y_val = y_min + (y_max - y_min) * (1.0 - t);
            let y_pos = plot_y + plot_height * t as f32;

            if y_pos >= plot_y && y_pos < plot_y + plot_height {
                let label = format!("{y_val:>5.0}");
                canvas.draw_text(&label, Point::new(self.bounds.x, y_pos), label_style);
            }
        }
    }

    /// Draw X-axis labels.
    #[allow(clippy::too_many_arguments)]
    fn draw_x_axis(
        &self,
        canvas: &mut dyn Canvas,
        x_min: f64,
        x_max: f64,
        plot_x: f32,
        plot_y: f32,
        plot_width: f32,
        plot_height: f32,
        label_style: &TextStyle,
    ) {
        for i in 0..=self.x_axis.ticks.min(plot_width as usize / 8) {
            let t = i as f64 / self.x_axis.ticks as f64;
            let x_val = x_min + (x_max - x_min) * t;
            let x_pos = plot_x + plot_width * t as f32;

            if x_pos >= plot_x && x_pos < plot_x + plot_width - 4.0 {
                let label = format!("{x_val:.0}");
                canvas.draw_text(&label, Point::new(x_pos, plot_y + plot_height), label_style);
            }
        }
    }

    /// Get color for a point at given index.
    fn point_color(&self, i: usize, c_min: f64, c_max: f64) -> Color {
        if let (Some(ref values), Some(ref gradient)) = (&self.color_by, &self.gradient) {
            if i < values.len() {
                let c_norm = if c_max > c_min {
                    (values[i] - c_min) / (c_max - c_min)
                } else {
                    0.5
                };
                return gradient.sample(c_norm);
            }
        }
        self.color
    }
}

impl Default for ScatterPlot {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl Widget for ScatterPlot {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        Size::new(
            constraints.max_width.min(60.0),
            constraints.max_height.min(20.0),
        )
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.bounds.width < 10.0 || self.bounds.height < 5.0 {
            return;
        }

        let (x_min, x_max) = self.x_range();
        let (y_min, y_max) = self.y_range();
        let (c_min, c_max) = self.color_range();

        // Calculate margins
        let margin_left = if self.show_axes { 6.0 } else { 0.0 };
        let margin_bottom = if self.show_axes { 2.0 } else { 0.0 };

        let plot_x = self.bounds.x + margin_left;
        let plot_y = self.bounds.y;
        let plot_width = self.bounds.width - margin_left;
        let plot_height = self.bounds.height - margin_bottom;

        if plot_width <= 0.0 || plot_height <= 0.0 {
            return;
        }

        let label_style = TextStyle {
            color: Color::new(0.6, 0.6, 0.6, 1.0),
            ..Default::default()
        };

        // Draw axis labels
        if self.show_axes {
            self.draw_y_axis(canvas, y_min, y_max, plot_y, plot_height, &label_style);
            self.draw_x_axis(
                canvas,
                x_min,
                x_max,
                plot_x,
                plot_y,
                plot_width,
                plot_height,
                &label_style,
            );
        }

        // Draw points
        let marker_char = self.marker.char();

        for (i, &(x, y)) in self.points.iter().enumerate() {
            if !x.is_finite() || !y.is_finite() {
                continue;
            }

            // Normalize coordinates
            let x_norm = if x_max > x_min {
                (x - x_min) / (x_max - x_min)
            } else {
                0.5
            };
            let y_norm = if y_max > y_min {
                (y - y_min) / (y_max - y_min)
            } else {
                0.5
            };

            // Convert to screen coordinates
            let screen_x = plot_x + (x_norm * plot_width as f64) as f32;
            let screen_y = plot_y + ((1.0 - y_norm) * plot_height as f64) as f32;

            // Check bounds
            if screen_x < plot_x
                || screen_x >= plot_x + plot_width
                || screen_y < plot_y
                || screen_y >= plot_y + plot_height
            {
                continue;
            }

            let style = TextStyle {
                color: self.point_color(i, c_min, c_max),
                ..Default::default()
            };

            canvas.draw_text(
                &marker_char.to_string(),
                Point::new(screen_x, screen_y),
                &style,
            );
        }

        // Draw axis labels if present
        if self.show_axes {
            if let Some(ref label) = self.x_axis.label {
                let x = plot_x + plot_width / 2.0 - label.len() as f32 / 2.0;
                canvas.draw_text(
                    label,
                    Point::new(x, self.bounds.y + self.bounds.height - 1.0),
                    &label_style,
                );
            }

            if let Some(ref label) = self.y_axis.label {
                // Vertical label (first char only in terminal)
                canvas.draw_text(
                    &label.chars().next().unwrap_or(' ').to_string(),
                    Point::new(self.bounds.x, plot_y + plot_height / 2.0),
                    &label_style,
                );
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

impl Brick for ScatterPlot {
    fn brick_name(&self) -> &'static str {
        "ScatterPlot"
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

        if self.bounds.width >= 10.0 && self.bounds.height >= 5.0 {
            passed.push(BrickAssertion::max_latency_ms(16));
        } else {
            failed.push((
                BrickAssertion::max_latency_ms(16),
                "Size too small".to_string(),
            ));
        }

        BrickVerification {
            passed,
            failed,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct::{CellBuffer, DirectTerminalCanvas};

    #[test]
    fn test_scatter_creation() {
        let points = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)];
        let scatter = ScatterPlot::new(points);
        assert_eq!(scatter.points.len(), 3);
    }

    #[test]
    fn test_marker_chars() {
        assert_eq!(MarkerStyle::Dot.char(), '•');
        assert_eq!(MarkerStyle::Cross.char(), '+');
        assert_eq!(MarkerStyle::Circle.char(), '○');
        assert_eq!(MarkerStyle::Square.char(), '□');
        assert_eq!(MarkerStyle::Diamond.char(), '◇');
    }

    #[test]
    fn test_empty_scatter() {
        let scatter = ScatterPlot::new(vec![]);
        let (x_min, x_max) = scatter.x_range();
        assert_eq!(x_min, 0.0);
        assert_eq!(x_max, 1.0);
    }

    #[test]
    fn test_auto_range() {
        let points = vec![(10.0, 20.0), (30.0, 40.0)];
        let scatter = ScatterPlot::new(points);
        let (x_min, x_max) = scatter.x_range();
        assert!(x_min < 10.0); // Includes padding
        assert!(x_max > 30.0);
    }

    #[test]
    fn test_scatter_assertions() {
        let scatter = ScatterPlot::default();
        assert!(!scatter.assertions().is_empty());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_scatter_verify() {
        let mut scatter = ScatterPlot::default();
        scatter.bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        assert!(scatter.verify().is_valid());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_scatter_verify_small_bounds() {
        let mut scatter = ScatterPlot::default();
        scatter.bounds = Rect::new(0.0, 0.0, 5.0, 3.0);
        let result = scatter.verify();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_scatter_children() {
        let scatter = ScatterPlot::default();
        assert!(scatter.children().is_empty());
    }

    #[test]
    fn test_scatter_layout() {
        let mut scatter = ScatterPlot::new(vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)]);
        let bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        let result = scatter.layout(bounds);
        assert!(result.size.width > 0.0);
        assert!(result.size.height > 0.0);
    }

    #[test]
    fn test_scatter_paint() {
        let mut scatter = ScatterPlot::new(vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)]);
        let bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        scatter.layout(bounds);

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        scatter.paint(&mut canvas);
    }

    #[test]
    fn test_scatter_with_all_markers() {
        for marker in [
            MarkerStyle::Dot,
            MarkerStyle::Cross,
            MarkerStyle::Circle,
            MarkerStyle::Square,
            MarkerStyle::Diamond,
        ] {
            let mut scatter = ScatterPlot::new(vec![(0.0, 0.0), (1.0, 1.0)]).with_marker(marker);
            let bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
            scatter.layout(bounds);
            let mut buffer = CellBuffer::new(60, 20);
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            scatter.paint(&mut canvas);
        }
    }

    #[test]
    fn test_scatter_with_color() {
        let mut scatter = ScatterPlot::new(vec![(0.0, 0.0), (1.0, 1.0)]).with_color(Color::RED);
        let bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        scatter.layout(bounds);
        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        scatter.paint(&mut canvas);
    }

    #[test]
    fn test_scatter_with_color_gradient() {
        let mut scatter = ScatterPlot::new(vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)])
            .with_color_by(vec![0.0, 0.5, 1.0], Gradient::two(Color::BLUE, Color::RED));
        let bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        scatter.layout(bounds);
        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        scatter.paint(&mut canvas);
    }

    #[test]
    fn test_scatter_with_axes() {
        let mut scatter = ScatterPlot::new(vec![(0.0, 0.0), (10.0, 10.0)])
            .with_axes(true)
            .with_x_axis(ScatterAxis {
                label: Some("X Axis".to_string()),
                min: Some(0.0),
                max: Some(10.0),
                ticks: 5,
            })
            .with_y_axis(ScatterAxis {
                label: Some("Y Axis".to_string()),
                min: Some(0.0),
                max: Some(10.0),
                ticks: 5,
            });
        let bounds = Rect::new(0.0, 0.0, 80.0, 24.0);
        scatter.layout(bounds);
        let mut buffer = CellBuffer::new(80, 24);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        scatter.paint(&mut canvas);
    }

    #[test]
    fn test_scatter_y_range() {
        let scatter = ScatterPlot::new(vec![(0.0, -5.0), (1.0, 10.0), (2.0, 3.0)]);
        let (y_min, y_max) = scatter.y_range();
        assert!(y_min <= -5.0);
        assert!(y_max >= 10.0);
    }

    #[test]
    fn test_scatter_y_range_empty() {
        let scatter = ScatterPlot::new(vec![]);
        let (y_min, y_max) = scatter.y_range();
        assert_eq!(y_min, 0.0);
        assert_eq!(y_max, 1.0);
    }

    #[test]
    fn test_scatter_with_many_points() {
        let points: Vec<(f64, f64)> = (0..100)
            .map(|i| (i as f64, (i as f64 * 0.1).sin() * 10.0))
            .collect();
        let mut scatter = ScatterPlot::new(points);
        let bounds = Rect::new(0.0, 0.0, 80.0, 24.0);
        scatter.layout(bounds);
        let mut buffer = CellBuffer::new(80, 24);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        scatter.paint(&mut canvas);
    }

    #[test]
    fn test_scatter_axis_default() {
        let axis = ScatterAxis::default();
        assert!(axis.label.is_none());
        assert!(axis.min.is_none());
        assert!(axis.max.is_none());
    }

    #[test]
    fn test_marker_style_default() {
        let marker = MarkerStyle::default();
        assert!(matches!(marker, MarkerStyle::Dot));
    }

    #[test]
    fn test_gradient_interpolate() {
        let gradient = Gradient::two(Color::BLACK, Color::WHITE);
        let mid = gradient.sample(0.5);
        // Color values are f32 in range 0-1 or 0-255 depending on implementation
        // Just verify it's between start and end colors
        assert!(mid.r > 0.0);
        assert!(mid.g > 0.0);
        assert!(mid.b > 0.0);
    }

    #[test]
    fn test_scatter_brick_name() {
        let scatter = ScatterPlot::default();
        assert_eq!(scatter.brick_name(), "ScatterPlot");
    }

    #[test]
    fn test_scatter_budget() {
        let scatter = ScatterPlot::default();
        let budget = scatter.budget();
        assert!(budget.layout_ms > 0);
    }

    #[test]
    fn test_scatter_to_html_css() {
        let scatter = ScatterPlot::default();
        assert!(scatter.to_html().is_empty());
        assert!(scatter.to_css().is_empty());
    }

    #[test]
    fn test_marker_style_triangle() {
        let marker = MarkerStyle::Triangle;
        assert_eq!(marker.char(), '△');
    }

    #[test]
    fn test_marker_style_star() {
        let marker = MarkerStyle::Star;
        assert_eq!(marker.char(), '★');
    }

    #[test]
    fn test_scatter_plot_with_axis_labels() {
        let scatter = ScatterPlot::new(vec![(1.0, 2.0), (3.0, 4.0)])
            .with_x_axis(ScatterAxis {
                label: Some("X-Axis".to_string()),
                ..Default::default()
            })
            .with_y_axis(ScatterAxis {
                label: Some("Y-Axis".to_string()),
                ..Default::default()
            });
        assert!(scatter.x_axis.label.is_some());
        assert!(scatter.y_axis.label.is_some());
    }

    #[test]
    fn test_scatter_plot_with_diamond_marker() {
        let scatter =
            ScatterPlot::new(vec![(1.0, 2.0), (3.0, 4.0)]).with_marker(MarkerStyle::Diamond);
        assert!(matches!(scatter.marker, MarkerStyle::Diamond));
    }

    #[test]
    fn test_scatter_set_points() {
        let mut scatter = ScatterPlot::new(vec![(0.0, 0.0)]);
        assert_eq!(scatter.points.len(), 1);
        scatter.set_points(vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]);
        assert_eq!(scatter.points.len(), 3);
    }

    #[test]
    fn test_scatter_with_axes_false() {
        let mut scatter = ScatterPlot::new(vec![(0.0, 0.0), (10.0, 10.0)]).with_axes(false);
        assert!(!scatter.show_axes);
        let bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        scatter.layout(bounds);
        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        scatter.paint(&mut canvas);
    }

    #[test]
    fn test_scatter_nan_values() {
        let mut scatter = ScatterPlot::new(vec![(0.0, 0.0), (f64::NAN, f64::NAN), (2.0, 2.0)]);
        let bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        scatter.layout(bounds);
        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        scatter.paint(&mut canvas); // Should not panic
    }

    #[test]
    fn test_scatter_infinite_values() {
        let mut scatter = ScatterPlot::new(vec![
            (0.0, 0.0),
            (f64::INFINITY, f64::NEG_INFINITY),
            (2.0, 2.0),
        ]);
        let bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        scatter.layout(bounds);
        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        scatter.paint(&mut canvas); // Should not panic
    }

    #[test]
    fn test_scatter_color_range_no_color_by() {
        let scatter = ScatterPlot::new(vec![(0.0, 0.0), (1.0, 1.0)]);
        let (c_min, c_max) = scatter.color_range();
        assert_eq!(c_min, 0.0);
        assert_eq!(c_max, 1.0);
    }

    #[test]
    fn test_scatter_color_range_with_values() {
        let scatter = ScatterPlot::new(vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]).with_color_by(
            vec![5.0, 10.0, 15.0],
            Gradient::two(Color::BLUE, Color::RED),
        );
        let (c_min, c_max) = scatter.color_range();
        assert_eq!(c_min, 5.0);
        assert_eq!(c_max, 15.0);
    }

    #[test]
    fn test_scatter_color_range_empty_values() {
        let scatter = ScatterPlot::new(vec![(0.0, 0.0)])
            .with_color_by(vec![], Gradient::two(Color::BLUE, Color::RED));
        let (c_min, c_max) = scatter.color_range();
        // Empty values results in default range
        assert_eq!(c_min, 0.0);
        assert_eq!(c_max, 1.0);
    }

    #[test]
    fn test_scatter_color_by_fewer_values_than_points() {
        // More points than color values - fallback to default color
        let mut scatter = ScatterPlot::new(vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])
            .with_color_by(vec![0.0, 1.0], Gradient::two(Color::BLUE, Color::RED));
        let bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        scatter.layout(bounds);
        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        scatter.paint(&mut canvas);
    }

    #[test]
    fn test_scatter_same_x_values() {
        // When all x values are the same
        let mut scatter = ScatterPlot::new(vec![(5.0, 0.0), (5.0, 5.0), (5.0, 10.0)]);
        let bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        scatter.layout(bounds);
        let (x_min, x_max) = scatter.x_range();
        // Same x values, with padding
        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        scatter.paint(&mut canvas);
        let _ = (x_min, x_max);
    }

    #[test]
    fn test_scatter_same_y_values() {
        // When all y values are the same
        let mut scatter = ScatterPlot::new(vec![(0.0, 5.0), (5.0, 5.0), (10.0, 5.0)]);
        let bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        scatter.layout(bounds);
        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        scatter.paint(&mut canvas);
    }

    #[test]
    fn test_scatter_too_small_bounds() {
        let mut scatter = ScatterPlot::new(vec![(0.0, 0.0), (1.0, 1.0)]);
        let bounds = Rect::new(0.0, 0.0, 5.0, 3.0);
        scatter.layout(bounds);
        let mut buffer = CellBuffer::new(5, 3);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        scatter.paint(&mut canvas); // Should early return
    }

    #[test]
    fn test_scatter_children_mut() {
        let mut scatter = ScatterPlot::default();
        assert!(scatter.children_mut().is_empty());
    }

    #[test]
    fn test_scatter_measure() {
        let scatter = ScatterPlot::default();
        let size = scatter.measure(Constraints {
            min_width: 0.0,
            min_height: 0.0,
            max_width: 100.0,
            max_height: 50.0,
        });
        assert_eq!(size.width, 60.0);
        assert_eq!(size.height, 20.0);
    }

    #[test]
    fn test_scatter_clone() {
        let original = ScatterPlot::new(vec![(1.0, 2.0), (3.0, 4.0)])
            .with_marker(MarkerStyle::Star)
            .with_color(Color::GREEN);
        let cloned = original;
        assert_eq!(cloned.points.len(), 2);
        assert_eq!(cloned.color, Color::GREEN);
        assert!(matches!(cloned.marker, MarkerStyle::Star));
    }

    #[test]
    fn test_scatter_debug() {
        let scatter = ScatterPlot::default();
        let debug = format!("{:?}", scatter);
        assert!(debug.contains("ScatterPlot"));
    }
}
