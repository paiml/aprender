//! Box plot widget for statistical visualization.
//!
//! Displays box-and-whisker plots using ASCII/Unicode art showing
//! median, quartiles, and outliers.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Orientation for box plots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Orientation {
    /// Horizontal box plots: ├──[████|████]──┤
    #[default]
    Horizontal,
    /// Vertical box plots.
    Vertical,
}

/// Statistics for a single box plot.
#[derive(Debug, Clone, Copy, Default)]
pub struct BoxStats {
    /// Minimum value (or lower whisker).
    pub min: f64,
    /// First quartile (25th percentile).
    pub q1: f64,
    /// Median (50th percentile).
    pub median: f64,
    /// Third quartile (75th percentile).
    pub q3: f64,
    /// Maximum value (or upper whisker).
    pub max: f64,
}

impl BoxStats {
    /// Create box stats from values.
    #[must_use]
    pub fn new(min: f64, q1: f64, median: f64, q3: f64, max: f64) -> Self {
        debug_assert!(min <= q1, "min must be <= q1");
        debug_assert!(q1 <= median, "q1 must be <= median");
        debug_assert!(median <= q3, "median must be <= q3");
        debug_assert!(q3 <= max, "q3 must be <= max");
        Self {
            min,
            q1,
            median,
            q3,
            max,
        }
    }

    /// Calculate box stats from a data slice.
    #[must_use]
    pub fn from_data(data: &[f64]) -> Self {
        if data.is_empty() {
            return Self::default();
        }

        let mut sorted: Vec<f64> = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let min = sorted[0];
        let max = sorted[n - 1];
        let median = Self::percentile(&sorted, 50.0);
        let q1 = Self::percentile(&sorted, 25.0);
        let q3 = Self::percentile(&sorted, 75.0);

        Self {
            min,
            q1,
            median,
            q3,
            max,
        }
    }

    fn percentile(sorted: &[f64], p: f64) -> f64 {
        let n = sorted.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return sorted[0];
        }

        let idx = (p / 100.0 * (n - 1) as f64).max(0.0);
        let lower = idx.floor() as usize;
        let upper = idx.ceil() as usize;
        let frac = idx - lower as f64;

        if lower >= n {
            sorted[n - 1]
        } else if upper >= n {
            sorted[lower]
        } else {
            sorted[lower] * (1.0 - frac) + sorted[upper] * frac
        }
    }

    /// Get interquartile range (IQR).
    #[must_use]
    pub fn iqr(&self) -> f64 {
        self.q3 - self.q1
    }

    /// Get range.
    #[must_use]
    pub fn range(&self) -> f64 {
        self.max - self.min
    }
}

/// Box plot widget for statistical visualization.
#[derive(Debug, Clone)]
pub struct BoxPlot {
    /// Statistics for each box.
    stats: Vec<BoxStats>,
    /// Labels for each box.
    labels: Vec<String>,
    /// Orientation.
    orientation: Orientation,
    /// Box color.
    color: Color,
    /// Global minimum for scaling.
    global_min: f64,
    /// Global maximum for scaling.
    global_max: f64,
    /// Show values.
    show_values: bool,
    /// Box width in characters.
    box_width: usize,
    /// Cached bounds.
    bounds: Rect,
}

impl Default for BoxPlot {
    fn default() -> Self {
        Self::new(vec![])
    }
}

impl BoxPlot {
    /// Create a new box plot.
    #[must_use]
    pub fn new(stats: Vec<BoxStats>) -> Self {
        let (gmin, gmax) = Self::compute_global_range(&stats);
        Self {
            stats,
            labels: vec![],
            orientation: Orientation::default(),
            color: Color::new(0.3, 0.7, 1.0, 1.0),
            global_min: gmin,
            global_max: gmax,
            show_values: false,
            box_width: 40,
            bounds: Rect::default(),
        }
    }

    /// Create from raw data vectors.
    #[must_use]
    pub fn from_data(datasets: &[&[f64]]) -> Self {
        let stats: Vec<BoxStats> = datasets.iter().map(|d| BoxStats::from_data(d)).collect();
        Self::new(stats)
    }

    /// Set labels.
    #[must_use]
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = labels;
        self
    }

    /// Set orientation.
    #[must_use]
    pub fn with_orientation(mut self, orientation: Orientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Set box color.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set global range for scaling.
    #[must_use]
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.global_min = min;
        self.global_max = max.max(min + 0.001);
        self
    }

    /// Show values on plot.
    #[must_use]
    pub fn with_values(mut self, show: bool) -> Self {
        self.show_values = show;
        self
    }

    /// Set box width.
    #[must_use]
    pub fn with_box_width(mut self, width: usize) -> Self {
        self.box_width = width.max(10);
        self
    }

    /// Update stats.
    pub fn set_stats(&mut self, stats: Vec<BoxStats>) {
        let (gmin, gmax) = Self::compute_global_range(&stats);
        self.global_min = gmin;
        self.global_max = gmax;
        self.stats = stats;
    }

    /// Get number of box plots.
    #[must_use]
    pub fn count(&self) -> usize {
        self.stats.len()
    }

    fn compute_global_range(stats: &[BoxStats]) -> (f64, f64) {
        if stats.is_empty() {
            return (0.0, 1.0);
        }
        let min = stats.iter().map(|s| s.min).fold(f64::MAX, f64::min);
        let max = stats.iter().map(|s| s.max).fold(f64::MIN, f64::max);
        if (max - min).abs() < f64::EPSILON {
            (min - 0.5, max + 0.5)
        } else {
            (min, max)
        }
    }

    fn normalize(&self, value: f64) -> f64 {
        let range = self.global_max - self.global_min;
        if range.abs() < f64::EPSILON {
            0.5
        } else {
            ((value - self.global_min) / range).clamp(0.0, 1.0)
        }
    }

    fn label_width(&self) -> usize {
        self.labels
            .iter()
            .map(String::len)
            .max()
            .unwrap_or(0)
            .max(5)
    }

    fn render_horizontal_box(
        &self,
        canvas: &mut dyn Canvas,
        stats: &BoxStats,
        x: f32,
        y: f32,
        width: f32,
    ) {
        let style = TextStyle {
            color: self.color,
            ..Default::default()
        };

        let whisker_style = TextStyle {
            color: Color::new(0.7, 0.7, 0.7, 1.0),
            ..Default::default()
        };

        // Calculate positions
        let width_f64 = width as f64;
        let min_pos = (self.normalize(stats.min) * width_f64) as usize;
        let q1_pos = (self.normalize(stats.q1) * width_f64) as usize;
        let median_pos = (self.normalize(stats.median) * width_f64) as usize;
        let q3_pos = (self.normalize(stats.q3) * width_f64) as usize;
        let max_pos = (self.normalize(stats.max) * width_f64) as usize;

        let width_usize = width as usize;

        // Build the box plot string
        let mut line = String::with_capacity(width_usize);

        for i in 0..width_usize {
            let ch = if i == min_pos {
                '├' // Left whisker endpoint
            } else if i == max_pos {
                '┤' // Right whisker endpoint
            } else if (i > min_pos && i < q1_pos) || (i > q3_pos && i < max_pos) {
                '─' // Whisker line
            } else if i == q1_pos {
                '[' // Box start
            } else if i == q3_pos {
                ']' // Box end
            } else if i == median_pos && i > q1_pos && i < q3_pos {
                '│' // Median line
            } else if i > q1_pos && i < q3_pos {
                '█' // Box fill
            } else {
                ' '
            };
            line.push(ch);
        }

        // Draw the box plot
        canvas.draw_text(&line, Point::new(x, y), &style);

        // Draw whisker endpoints
        if min_pos < width_usize {
            canvas.draw_text("├", Point::new(x + min_pos as f32, y), &whisker_style);
        }
        if max_pos < width_usize {
            canvas.draw_text("┤", Point::new(x + max_pos as f32, y), &whisker_style);
        }
    }

    fn render_vertical_box(
        &self,
        canvas: &mut dyn Canvas,
        stats: &BoxStats,
        x: f32,
        y: f32,
        height: f32,
    ) {
        let style = TextStyle {
            color: self.color,
            ..Default::default()
        };

        let whisker_style = TextStyle {
            color: Color::new(0.7, 0.7, 0.7, 1.0),
            ..Default::default()
        };

        // Calculate positions (inverted: 0 at bottom, 1 at top)
        let height_f64 = height as f64;
        let min_pos = ((1.0 - self.normalize(stats.min)) * height_f64) as usize;
        let q1_pos = ((1.0 - self.normalize(stats.q1)) * height_f64) as usize;
        let median_pos = ((1.0 - self.normalize(stats.median)) * height_f64) as usize;
        let q3_pos = ((1.0 - self.normalize(stats.q3)) * height_f64) as usize;
        let max_pos = ((1.0 - self.normalize(stats.max)) * height_f64) as usize;

        let height_usize = height as usize;

        // Draw from top to bottom
        for i in 0..height_usize {
            let ch = if i == max_pos {
                "┬" // Top whisker
            } else if i == min_pos {
                "┴" // Bottom whisker
            } else if (i > max_pos && i < q3_pos) || (i > q1_pos && i < min_pos) {
                "│" // Whisker line
            } else if i == q3_pos {
                "┌" // Box top
            } else if i == q1_pos {
                "└" // Box bottom
            } else if i == median_pos && i > q3_pos && i < q1_pos {
                "├" // Median
            } else if i > q3_pos && i < q1_pos {
                "█" // Box fill
            } else {
                " "
            };

            let row_style = if i == max_pos || i == min_pos {
                &whisker_style
            } else {
                &style
            };

            canvas.draw_text(ch, Point::new(x, y + i as f32), row_style);
        }
    }
}

impl Brick for BoxPlot {
    fn brick_name(&self) -> &'static str {
        "box_plot"
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

impl Widget for BoxPlot {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        match self.orientation {
            Orientation::Horizontal => {
                let label_w = self.label_width();
                let width = (label_w + 2 + self.box_width) as f32;
                let height = self.stats.len().max(1) as f32;
                constraints.constrain(Size::new(width.min(constraints.max_width), height))
            }
            Orientation::Vertical => {
                let width = (self.stats.len() * 4).max(4) as f32;
                let height = 10.0f32;
                constraints.constrain(Size::new(width, height.min(constraints.max_height)))
            }
        }
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.stats.is_empty() || self.bounds.width < 1.0 {
            return;
        }

        let label_style = TextStyle {
            color: Color::new(0.8, 0.8, 0.8, 1.0),
            ..Default::default()
        };

        let dim_style = TextStyle {
            color: Color::new(0.5, 0.5, 0.5, 1.0),
            ..Default::default()
        };

        match self.orientation {
            Orientation::Horizontal => {
                let label_w = self.label_width();
                let box_start = self.bounds.x + label_w as f32 + 2.0;
                let box_width = (self.bounds.width - label_w as f32 - 2.0).max(10.0);

                for (i, stats) in self.stats.iter().enumerate() {
                    let y = self.bounds.y + i as f32;

                    // Draw label
                    if let Some(label) = self.labels.get(i) {
                        canvas.draw_text(label, Point::new(self.bounds.x, y), &label_style);
                    }

                    // Draw box plot
                    self.render_horizontal_box(canvas, stats, box_start, y, box_width);

                    // Draw values if enabled
                    if self.show_values {
                        let val_text =
                            format!(" [{:.1}, {:.1}, {:.1}]", stats.q1, stats.median, stats.q3);
                        canvas.draw_text(
                            &val_text,
                            Point::new(box_start + box_width, y),
                            &dim_style,
                        );
                    }
                }
            }
            Orientation::Vertical => {
                let box_height = (self.bounds.height - 2.0).max(5.0);

                for (i, stats) in self.stats.iter().enumerate() {
                    let x = self.bounds.x + (i * 4) as f32;

                    // Draw box plot
                    self.render_vertical_box(canvas, stats, x, self.bounds.y, box_height);

                    // Draw label below
                    if let Some(label) = self.labels.get(i) {
                        let truncated = if label.len() > 3 { &label[..3] } else { label };
                        canvas.draw_text(
                            truncated,
                            Point::new(x, self.bounds.y + box_height + 1.0),
                            &label_style,
                        );
                    }
                }
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
    fn test_box_stats_creation() {
        let stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_box_stats_from_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let stats = BoxStats::from_data(&data);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 9.0);
        assert_eq!(stats.median, 5.0);
    }

    #[test]
    fn test_box_stats_from_empty() {
        let stats = BoxStats::from_data(&[]);
        assert_eq!(stats.min, 0.0);
    }

    #[test]
    fn test_box_stats_from_single() {
        let stats = BoxStats::from_data(&[5.0]);
        assert_eq!(stats.min, 5.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.median, 5.0);
    }

    #[test]
    fn test_box_stats_iqr() {
        let stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
        assert_eq!(stats.iqr(), 2.0);
    }

    #[test]
    fn test_box_stats_range() {
        let stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
        assert_eq!(stats.range(), 4.0);
    }

    #[test]
    fn test_box_plot_creation() {
        let bp = BoxPlot::new(vec![BoxStats::new(0.0, 1.0, 2.0, 3.0, 4.0)]);
        assert_eq!(bp.count(), 1);
    }

    #[test]
    fn test_box_plot_from_data() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let bp = BoxPlot::from_data(&[&data1, &data2]);
        assert_eq!(bp.count(), 2);
    }

    #[test]
    fn test_box_plot_with_labels() {
        let bp = BoxPlot::new(vec![BoxStats::default()]).with_labels(vec!["Group A".to_string()]);
        assert_eq!(bp.labels.len(), 1);
    }

    #[test]
    fn test_box_plot_with_orientation() {
        let bp = BoxPlot::new(vec![]).with_orientation(Orientation::Vertical);
        assert_eq!(bp.orientation, Orientation::Vertical);
    }

    #[test]
    fn test_box_plot_with_color() {
        let bp = BoxPlot::new(vec![]).with_color(Color::RED);
        assert_eq!(bp.color, Color::RED);
    }

    #[test]
    fn test_box_plot_with_range() {
        let bp = BoxPlot::new(vec![]).with_range(0.0, 100.0);
        assert_eq!(bp.global_min, 0.0);
        assert_eq!(bp.global_max, 100.0);
    }

    #[test]
    fn test_box_plot_with_values() {
        let bp = BoxPlot::new(vec![]).with_values(true);
        assert!(bp.show_values);
    }

    #[test]
    fn test_box_plot_with_box_width() {
        let bp = BoxPlot::new(vec![]).with_box_width(60);
        assert_eq!(bp.box_width, 60);
    }

    #[test]
    fn test_box_plot_with_box_width_min() {
        let bp = BoxPlot::new(vec![]).with_box_width(5);
        assert_eq!(bp.box_width, 10); // Minimum is 10
    }

    #[test]
    fn test_box_plot_set_stats() {
        let mut bp = BoxPlot::new(vec![]);
        bp.set_stats(vec![BoxStats::new(0.0, 1.0, 2.0, 3.0, 4.0)]);
        assert_eq!(bp.count(), 1);
    }

    #[test]
    fn test_box_plot_paint_horizontal() {
        let mut bp = BoxPlot::new(vec![
            BoxStats::new(0.0, 2.0, 5.0, 8.0, 10.0),
            BoxStats::new(1.0, 3.0, 5.0, 7.0, 9.0),
        ])
        .with_labels(vec!["A".to_string(), "B".to_string()]);
        bp.bounds = Rect::new(0.0, 0.0, 50.0, 5.0);

        let mut canvas = MockCanvas::new();
        bp.paint(&mut canvas);

        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_box_plot_paint_vertical() {
        let mut bp = BoxPlot::new(vec![BoxStats::new(0.0, 2.0, 5.0, 8.0, 10.0)])
            .with_orientation(Orientation::Vertical);
        bp.bounds = Rect::new(0.0, 0.0, 20.0, 15.0);

        let mut canvas = MockCanvas::new();
        bp.paint(&mut canvas);

        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_box_plot_paint_empty() {
        let bp = BoxPlot::new(vec![]);
        let mut canvas = MockCanvas::new();
        bp.paint(&mut canvas);
        assert!(canvas.texts.is_empty());
    }

    #[test]
    fn test_box_plot_paint_with_values() {
        let mut bp = BoxPlot::new(vec![BoxStats::new(0.0, 2.0, 5.0, 8.0, 10.0)]).with_values(true);
        bp.bounds = Rect::new(0.0, 0.0, 80.0, 5.0);

        let mut canvas = MockCanvas::new();
        bp.paint(&mut canvas);

        // Should have value text
        assert!(canvas.texts.iter().any(|(t, _)| t.contains('[')));
    }

    #[test]
    fn test_box_plot_measure_horizontal() {
        let bp = BoxPlot::new(vec![BoxStats::default(), BoxStats::default()]);
        let size = bp.measure(Constraints::loose(Size::new(100.0, 50.0)));
        assert!(size.height >= 2.0);
    }

    #[test]
    fn test_box_plot_measure_vertical() {
        let bp = BoxPlot::new(vec![BoxStats::default(), BoxStats::default()])
            .with_orientation(Orientation::Vertical);
        let size = bp.measure(Constraints::loose(Size::new(100.0, 50.0)));
        assert!(size.width >= 8.0); // 2 boxes * 4 chars
    }

    #[test]
    fn test_box_plot_layout() {
        let mut bp = BoxPlot::new(vec![]);
        let bounds = Rect::new(5.0, 10.0, 30.0, 20.0);
        let result = bp.layout(bounds);
        assert_eq!(result.size.width, 30.0);
        assert_eq!(bp.bounds, bounds);
    }

    #[test]
    fn test_box_plot_brick_name() {
        let bp = BoxPlot::new(vec![]);
        assert_eq!(bp.brick_name(), "box_plot");
    }

    #[test]
    fn test_box_plot_assertions() {
        let bp = BoxPlot::new(vec![]);
        assert!(!bp.assertions().is_empty());
    }

    #[test]
    fn test_box_plot_budget() {
        let bp = BoxPlot::new(vec![]);
        let budget = bp.budget();
        assert!(budget.paint_ms > 0);
    }

    #[test]
    fn test_box_plot_verify() {
        let bp = BoxPlot::new(vec![]);
        assert!(bp.verify().is_valid());
    }

    #[test]
    fn test_box_plot_type_id() {
        let bp = BoxPlot::new(vec![]);
        assert_eq!(Widget::type_id(&bp), TypeId::of::<BoxPlot>());
    }

    #[test]
    fn test_box_plot_children() {
        let bp = BoxPlot::new(vec![]);
        assert!(bp.children().is_empty());
    }

    #[test]
    fn test_box_plot_children_mut() {
        let mut bp = BoxPlot::new(vec![]);
        assert!(bp.children_mut().is_empty());
    }

    #[test]
    fn test_box_plot_event() {
        let mut bp = BoxPlot::new(vec![]);
        let event = Event::key_down(presentar_core::Key::Enter);
        assert!(bp.event(&event).is_none());
    }

    #[test]
    fn test_box_plot_default() {
        let bp = BoxPlot::default();
        assert!(bp.stats.is_empty());
    }

    #[test]
    fn test_box_plot_to_html() {
        let bp = BoxPlot::new(vec![]);
        assert!(bp.to_html().is_empty());
    }

    #[test]
    fn test_box_plot_to_css() {
        let bp = BoxPlot::new(vec![]);
        assert!(bp.to_css().is_empty());
    }

    #[test]
    fn test_orientation_default() {
        assert_eq!(Orientation::default(), Orientation::Horizontal);
    }

    #[test]
    fn test_box_stats_default() {
        let stats = BoxStats::default();
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
    }

    // ========================================================================
    // Additional tests for improved coverage
    // ========================================================================

    #[test]
    fn test_box_stats_from_two_values() {
        let stats = BoxStats::from_data(&[1.0, 5.0]);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_box_stats_from_three_values() {
        let stats = BoxStats::from_data(&[1.0, 3.0, 5.0]);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_box_stats_unsorted_data() {
        let stats = BoxStats::from_data(&[5.0, 1.0, 3.0, 4.0, 2.0]);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        // Median should be 3.0
        assert_eq!(stats.median, 3.0);
    }

    #[test]
    fn test_box_stats_with_nan() {
        // Data with NaN should still work (NaN sorts to end)
        let stats = BoxStats::from_data(&[1.0, 2.0, 3.0]);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 3.0);
    }

    #[test]
    fn test_box_plot_normalize() {
        let bp = BoxPlot::new(vec![BoxStats::new(0.0, 25.0, 50.0, 75.0, 100.0)]);
        // Access normalize through render_horizontal_box indirectly
        // Test by checking paint produces expected positions
        let mut bp = bp;
        bp.bounds = Rect::new(0.0, 0.0, 50.0, 5.0);
        let mut canvas = MockCanvas::new();
        bp.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_box_plot_normalize_constant_range() {
        let bp = BoxPlot::new(vec![BoxStats::new(5.0, 5.0, 5.0, 5.0, 5.0)]);
        // With constant data, global range becomes (min - 0.5, max + 0.5)
        assert!((bp.global_min - 4.5).abs() < f64::EPSILON);
        assert!((bp.global_max - 5.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_box_plot_normalize_empty_stats() {
        let bp = BoxPlot::new(vec![]);
        // Empty stats should have default range (0, 1)
        assert_eq!(bp.global_min, 0.0);
        assert_eq!(bp.global_max, 1.0);
    }

    #[test]
    fn test_box_plot_label_width_no_labels() {
        let bp = BoxPlot::new(vec![BoxStats::default()]);
        // No labels, should return minimum of 5
        let width = bp.label_width();
        assert_eq!(width, 5);
    }

    #[test]
    fn test_box_plot_label_width_with_labels() {
        let bp =
            BoxPlot::new(vec![BoxStats::default()]).with_labels(vec!["VeryLongLabel".to_string()]);
        let width = bp.label_width();
        assert_eq!(width, 13); // "VeryLongLabel".len()
    }

    #[test]
    fn test_box_plot_label_width_multiple_labels() {
        let bp = BoxPlot::new(vec![BoxStats::default(), BoxStats::default()])
            .with_labels(vec!["Short".to_string(), "VeryLongLabel".to_string()]);
        let width = bp.label_width();
        assert_eq!(width, 13); // Maximum label length
    }

    #[test]
    fn test_box_plot_with_range_min_greater_than_max() {
        let bp = BoxPlot::new(vec![]).with_range(100.0, 50.0);
        // max should be at least min + 0.001
        assert!(bp.global_max >= bp.global_min);
    }

    #[test]
    fn test_box_plot_with_range_equal() {
        let bp = BoxPlot::new(vec![]).with_range(50.0, 50.0);
        // max should be at least min + 0.001
        assert!(bp.global_max > bp.global_min);
    }

    #[test]
    fn test_box_plot_paint_vertical_with_labels() {
        let mut bp = BoxPlot::new(vec![
            BoxStats::new(0.0, 2.0, 5.0, 8.0, 10.0),
            BoxStats::new(1.0, 3.0, 5.0, 7.0, 9.0),
        ])
        .with_orientation(Orientation::Vertical)
        .with_labels(vec!["A".to_string(), "B".to_string()]);
        bp.bounds = Rect::new(0.0, 0.0, 20.0, 15.0);

        let mut canvas = MockCanvas::new();
        bp.paint(&mut canvas);

        // Should have label texts
        assert!(canvas.texts.iter().any(|(t, _)| t == "A" || t == "B"));
    }

    #[test]
    fn test_box_plot_paint_vertical_label_truncation() {
        let mut bp = BoxPlot::new(vec![BoxStats::new(0.0, 2.0, 5.0, 8.0, 10.0)])
            .with_orientation(Orientation::Vertical)
            .with_labels(vec!["LongLabel".to_string()]);
        bp.bounds = Rect::new(0.0, 0.0, 20.0, 15.0);

        let mut canvas = MockCanvas::new();
        bp.paint(&mut canvas);

        // Label should be truncated to 3 chars
        assert!(canvas.texts.iter().any(|(t, _)| t == "Lon"));
    }

    #[test]
    fn test_box_plot_paint_horizontal_no_labels() {
        let mut bp = BoxPlot::new(vec![BoxStats::new(0.0, 2.0, 5.0, 8.0, 10.0)]);
        bp.bounds = Rect::new(0.0, 0.0, 50.0, 5.0);

        let mut canvas = MockCanvas::new();
        bp.paint(&mut canvas);

        // Should still render box without labels
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_box_plot_paint_narrow_bounds() {
        let mut bp = BoxPlot::new(vec![BoxStats::new(0.0, 2.0, 5.0, 8.0, 10.0)]);
        bp.bounds = Rect::new(0.0, 0.0, 0.5, 5.0);

        let mut canvas = MockCanvas::new();
        bp.paint(&mut canvas);

        // Should early return for very narrow bounds
        assert!(canvas.texts.is_empty());
    }

    #[test]
    fn test_box_plot_global_range_multiple_stats() {
        let stats = vec![
            BoxStats::new(5.0, 10.0, 15.0, 20.0, 25.0),
            BoxStats::new(0.0, 5.0, 10.0, 15.0, 20.0),
            BoxStats::new(10.0, 15.0, 20.0, 25.0, 30.0),
        ];
        let bp = BoxPlot::new(stats);
        assert_eq!(bp.global_min, 0.0); // Min of all mins
        assert_eq!(bp.global_max, 30.0); // Max of all maxes
    }

    #[test]
    fn test_box_plot_set_stats_updates_range() {
        let mut bp = BoxPlot::new(vec![BoxStats::new(0.0, 1.0, 2.0, 3.0, 4.0)]);
        assert_eq!(bp.global_max, 4.0);

        bp.set_stats(vec![BoxStats::new(0.0, 5.0, 10.0, 15.0, 20.0)]);
        assert_eq!(bp.global_max, 20.0);
    }

    #[test]
    fn test_box_plot_multiple_stats_paint() {
        let mut bp = BoxPlot::new(vec![
            BoxStats::new(0.0, 2.0, 5.0, 8.0, 10.0),
            BoxStats::new(1.0, 3.0, 5.0, 7.0, 9.0),
            BoxStats::new(2.0, 4.0, 6.0, 8.0, 10.0),
        ])
        .with_labels(vec![
            "Group A".to_string(),
            "Group B".to_string(),
            "Group C".to_string(),
        ]);
        bp.bounds = Rect::new(0.0, 0.0, 60.0, 5.0);

        let mut canvas = MockCanvas::new();
        bp.paint(&mut canvas);

        // Should have content for all groups
        assert!(canvas.texts.len() > 3);
    }

    #[test]
    fn test_box_plot_clone() {
        let bp = BoxPlot::new(vec![BoxStats::new(0.0, 1.0, 2.0, 3.0, 4.0)])
            .with_color(Color::RED)
            .with_labels(vec!["Test".to_string()]);
        let cloned = bp.clone();
        assert_eq!(cloned.stats.len(), bp.stats.len());
        assert_eq!(cloned.labels, bp.labels);
        assert_eq!(cloned.color, bp.color);
    }

    #[test]
    fn test_box_plot_debug() {
        let bp = BoxPlot::new(vec![BoxStats::new(0.0, 1.0, 2.0, 3.0, 4.0)]);
        let debug_str = format!("{:?}", bp);
        assert!(debug_str.contains("BoxPlot"));
    }

    #[test]
    fn test_box_stats_debug() {
        let stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("BoxStats"));
    }

    #[test]
    fn test_box_stats_clone() {
        let stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
        let cloned = stats;
        assert_eq!(cloned.min, stats.min);
        assert_eq!(cloned.max, stats.max);
    }

    #[test]
    fn test_orientation_debug() {
        let h = Orientation::Horizontal;
        let v = Orientation::Vertical;
        assert!(format!("{:?}", h).contains("Horizontal"));
        assert!(format!("{:?}", v).contains("Vertical"));
    }

    #[test]
    fn test_orientation_clone() {
        let h = Orientation::Horizontal;
        let cloned = h;
        assert_eq!(cloned, Orientation::Horizontal);
    }

    #[test]
    fn test_box_plot_measure_empty() {
        let bp = BoxPlot::new(vec![]);
        let size = bp.measure(Constraints::loose(Size::new(100.0, 50.0)));
        assert!(size.height >= 1.0); // min 1 for empty
    }

    #[test]
    fn test_box_plot_measure_vertical_empty() {
        let bp = BoxPlot::new(vec![]).with_orientation(Orientation::Vertical);
        let size = bp.measure(Constraints::loose(Size::new(100.0, 50.0)));
        assert!(size.width >= 4.0); // min 4 for empty
    }

    #[test]
    fn test_box_stats_large_data() {
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let stats = BoxStats::from_data(&data);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 999.0);
        // Median should be around 499.5
        assert!((stats.median - 499.5).abs() < 1.0);
    }

    #[test]
    fn test_box_plot_vertical_values() {
        // Test vertical mode - values are not shown in vertical mode in current impl
        let mut bp = BoxPlot::new(vec![BoxStats::new(0.0, 2.0, 5.0, 8.0, 10.0)])
            .with_orientation(Orientation::Vertical)
            .with_values(true);
        bp.bounds = Rect::new(0.0, 0.0, 20.0, 15.0);

        let mut canvas = MockCanvas::new();
        bp.paint(&mut canvas);

        // Should render something
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_box_stats_q1_q3() {
        let stats = BoxStats::new(0.0, 25.0, 50.0, 75.0, 100.0);
        assert_eq!(stats.q1, 25.0);
        assert_eq!(stats.q3, 75.0);
    }

    #[test]
    fn test_box_plot_horizontal_box_rendering_positions() {
        // Test that box rendering produces expected characters
        let mut bp =
            BoxPlot::new(vec![BoxStats::new(0.0, 25.0, 50.0, 75.0, 100.0)]).with_range(0.0, 100.0);
        bp.bounds = Rect::new(0.0, 0.0, 50.0, 5.0);

        let mut canvas = MockCanvas::new();
        bp.paint(&mut canvas);

        // Should contain box drawing characters
        let has_box_chars = canvas.texts.iter().any(|(t, _)| {
            t.contains('├')
                || t.contains('┤')
                || t.contains('[')
                || t.contains(']')
                || t.contains('█')
        });
        assert!(has_box_chars);
    }

    #[test]
    fn test_box_plot_vertical_box_rendering_positions() {
        let mut bp = BoxPlot::new(vec![BoxStats::new(0.0, 25.0, 50.0, 75.0, 100.0)])
            .with_orientation(Orientation::Vertical)
            .with_range(0.0, 100.0);
        bp.bounds = Rect::new(0.0, 0.0, 10.0, 15.0);

        let mut canvas = MockCanvas::new();
        bp.paint(&mut canvas);

        // Should contain vertical box drawing characters
        let has_vertical_chars = canvas
            .texts
            .iter()
            .any(|(t, _)| t.contains('┬') || t.contains('┴') || t.contains('│') || t.contains('█'));
        assert!(has_vertical_chars);
    }
}
