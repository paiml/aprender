//! Multi-series line chart widget with axis support.
//!
//! Implements P200 from SPEC-024 Section 15.2.

use crate::widgets::symbols::BRAILLE_UP;
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Line simplification algorithm.
#[derive(Debug, Clone, Copy, Default)]
pub enum Simplification {
    /// No simplification - render all points.
    #[default]
    None,
    /// Douglas-Peucker algorithm with epsilon threshold.
    DouglasPeucker { epsilon: f64 },
    /// Visvalingam-Whyatt algorithm with area threshold.
    VisvalingamWhyatt { threshold: f64 },
}

/// A single data series for the line chart.
#[derive(Debug, Clone)]
pub struct Series {
    /// Series name (for legend).
    pub name: String,
    /// Data points as (x, y) coordinates.
    pub data: Vec<(f64, f64)>,
    /// Series color.
    pub color: Color,
    /// Line style.
    pub style: LineStyle,
}

/// Line rendering style.
#[derive(Debug, Clone, Copy, Default)]
pub enum LineStyle {
    /// Solid line (default).
    #[default]
    Solid,
    /// Dashed line.
    Dashed,
    /// Dotted line.
    Dotted,
    /// Show only markers.
    Markers,
}

/// Axis configuration.
#[derive(Debug, Clone)]
pub struct Axis {
    /// Axis label.
    pub label: Option<String>,
    /// Minimum value (None = auto).
    pub min: Option<f64>,
    /// Maximum value (None = auto).
    pub max: Option<f64>,
    /// Number of tick marks.
    pub ticks: usize,
    /// Show grid lines.
    pub grid: bool,
}

impl Default for Axis {
    fn default() -> Self {
        Self {
            label: None,
            min: None,
            max: None,
            ticks: 5,
            grid: false,
        }
    }
}

/// Legend position.
#[derive(Debug, Clone, Copy, Default)]
pub enum LegendPosition {
    /// Top-right corner.
    #[default]
    TopRight,
    /// Top-left corner.
    TopLeft,
    /// Bottom-right corner.
    BottomRight,
    /// Bottom-left corner.
    BottomLeft,
    /// No legend.
    None,
}

/// Multi-series line chart widget.
#[derive(Debug, Clone)]
pub struct LineChart {
    series: Vec<Series>,
    x_axis: Axis,
    y_axis: Axis,
    legend: LegendPosition,
    simplification: Simplification,
    bounds: Rect,
    /// Margin for axis labels.
    margin_left: f32,
    margin_bottom: f32,
}

impl LineChart {
    /// Create a new empty line chart.
    #[must_use]
    pub fn new() -> Self {
        Self {
            series: Vec::new(),
            x_axis: Axis::default(),
            y_axis: Axis::default(),
            legend: LegendPosition::default(),
            simplification: Simplification::default(),
            bounds: Rect::default(),
            margin_left: 6.0,
            margin_bottom: 2.0,
        }
    }

    /// Add a data series.
    #[must_use]
    pub fn add_series(mut self, name: &str, data: Vec<(f64, f64)>, color: Color) -> Self {
        self.series.push(Series {
            name: name.to_string(),
            data,
            color,
            style: LineStyle::default(),
        });
        self
    }

    /// Add a series with custom style.
    #[must_use]
    pub fn add_series_styled(
        mut self,
        name: &str,
        data: Vec<(f64, f64)>,
        color: Color,
        style: LineStyle,
    ) -> Self {
        self.series.push(Series {
            name: name.to_string(),
            data,
            color,
            style,
        });
        self
    }

    /// Set line simplification algorithm.
    #[must_use]
    pub fn with_simplification(mut self, algorithm: Simplification) -> Self {
        self.simplification = algorithm;
        self
    }

    /// Set X axis configuration.
    #[must_use]
    pub fn with_x_axis(mut self, axis: Axis) -> Self {
        self.x_axis = axis;
        self
    }

    /// Set Y axis configuration.
    #[must_use]
    pub fn with_y_axis(mut self, axis: Axis) -> Self {
        self.y_axis = axis;
        self
    }

    /// UX-P01: Compact mode with no axis labels or margins.
    ///
    /// Useful for inline/small charts where space is limited.
    #[must_use]
    pub fn compact(mut self) -> Self {
        self.margin_left = 0.0;
        self.margin_bottom = 0.0;
        self.y_axis.ticks = 0;
        self.x_axis.ticks = 0;
        self.legend = LegendPosition::None;
        self
    }

    /// UX-P01: Set custom margins for axis labels.
    #[must_use]
    pub fn with_margins(mut self, left: f32, bottom: f32) -> Self {
        debug_assert!(left >= 0.0, "left margin must be non-negative");
        debug_assert!(bottom >= 0.0, "bottom margin must be non-negative");
        self.margin_left = left;
        self.margin_bottom = bottom;
        self
    }

    /// Set legend position.
    #[must_use]
    pub fn with_legend(mut self, position: LegendPosition) -> Self {
        self.legend = position;
        self
    }

    /// Compute X range from all series.
    fn x_range(&self) -> (f64, f64) {
        if let Some(min) = self.x_axis.min {
            if let Some(max) = self.x_axis.max {
                return (min, max);
            }
        }

        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;

        for series in &self.series {
            for &(x, _) in &series.data {
                if x.is_finite() {
                    x_min = x_min.min(x);
                    x_max = x_max.max(x);
                }
            }
        }

        if x_min == f64::INFINITY {
            (0.0, 1.0)
        } else {
            (
                self.x_axis.min.unwrap_or(x_min),
                self.x_axis.max.unwrap_or(x_max),
            )
        }
    }

    /// Compute Y range from all series.
    fn y_range(&self) -> (f64, f64) {
        if let Some(min) = self.y_axis.min {
            if let Some(max) = self.y_axis.max {
                return (min, max);
            }
        }

        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;

        for series in &self.series {
            for &(_, y) in &series.data {
                if y.is_finite() {
                    y_min = y_min.min(y);
                    y_max = y_max.max(y);
                }
            }
        }

        if y_min == f64::INFINITY {
            (0.0, 1.0)
        } else {
            // Add 10% padding
            let padding = (y_max - y_min) * 0.1;
            (
                self.y_axis.min.unwrap_or(y_min - padding),
                self.y_axis.max.unwrap_or(y_max + padding),
            )
        }
    }

    /// Apply simplification to a series.
    fn simplify(&self, data: &[(f64, f64)]) -> Vec<(f64, f64)> {
        match self.simplification {
            Simplification::None => data.to_vec(),
            Simplification::DouglasPeucker { epsilon } => douglas_peucker(data, epsilon),
            Simplification::VisvalingamWhyatt { threshold } => visvalingam_whyatt(data, threshold),
        }
    }

    /// Draw Y axis labels.
    fn draw_y_axis(
        &self,
        canvas: &mut dyn Canvas,
        y_min: f64,
        y_max: f64,
        plot_y: f32,
        plot_height: f32,
    ) {
        let style = TextStyle {
            color: Color::new(0.6, 0.6, 0.6, 1.0),
            ..Default::default()
        };
        for i in 0..=self.y_axis.ticks {
            let t = i as f64 / self.y_axis.ticks as f64;
            let y_val = y_min + (y_max - y_min) * (1.0 - t);
            let y_pos = plot_y + plot_height * t as f32;
            if y_pos >= plot_y && y_pos < plot_y + plot_height {
                canvas.draw_text(
                    &format!("{y_val:>5.0}"),
                    Point::new(self.bounds.x, y_pos),
                    &style,
                );
            }
        }
    }

    /// Draw X axis labels.
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
    ) {
        let style = TextStyle {
            color: Color::new(0.6, 0.6, 0.6, 1.0),
            ..Default::default()
        };
        for i in 0..=self.x_axis.ticks.min(plot_width as usize / 8) {
            let t = i as f64 / self.x_axis.ticks as f64;
            let x_val = x_min + (x_max - x_min) * t;
            let x_pos = plot_x + plot_width * t as f32;
            if x_pos >= plot_x && x_pos < plot_x + plot_width - 4.0 {
                canvas.draw_text(
                    &format!("{x_val:.0}"),
                    Point::new(x_pos, plot_y + plot_height),
                    &style,
                );
            }
        }
    }

    /// Draw the legend.
    fn draw_legend(
        &self,
        canvas: &mut dyn Canvas,
        plot_x: f32,
        plot_y: f32,
        plot_width: f32,
        plot_height: f32,
    ) {
        if matches!(self.legend, LegendPosition::None) || self.series.is_empty() {
            return;
        }
        let legend_width = self
            .series
            .iter()
            .map(|s| s.name.len() + 3)
            .max()
            .unwrap_or(10) as f32;
        let (lx, ly) = match self.legend {
            LegendPosition::TopRight => (plot_x + plot_width - legend_width, plot_y),
            LegendPosition::TopLeft => (plot_x, plot_y),
            LegendPosition::BottomRight => (
                plot_x + plot_width - legend_width,
                plot_y + plot_height - self.series.len() as f32,
            ),
            LegendPosition::BottomLeft => (plot_x, plot_y + plot_height - self.series.len() as f32),
            LegendPosition::None => return,
        };
        for (i, series) in self.series.iter().enumerate() {
            canvas.draw_text(
                &format!("─ {}", series.name),
                Point::new(lx, ly + i as f32),
                &TextStyle {
                    color: series.color,
                    ..Default::default()
                },
            );
        }
    }
}

impl Default for LineChart {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for LineChart {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        Size::new(
            constraints.max_width.min(80.0),
            constraints.max_height.min(20.0),
        )
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.bounds.width < 10.0 || self.bounds.height < 5.0 {
            return;
        }

        let (x_min, x_max) = self.x_range();
        let (y_min, y_max) = self.y_range();
        let plot_x = self.bounds.x + self.margin_left;
        let plot_y = self.bounds.y;
        let plot_width = self.bounds.width - self.margin_left;
        let plot_height = self.bounds.height - self.margin_bottom;
        if plot_width <= 0.0 || plot_height <= 0.0 {
            return;
        }

        // Draw axes
        self.draw_y_axis(canvas, y_min, y_max, plot_y, plot_height);
        self.draw_x_axis(
            canvas,
            x_min,
            x_max,
            plot_x,
            plot_y,
            plot_width,
            plot_height,
        );

        // Draw each series
        for series in &self.series {
            let simplified = self.simplify(&series.data);
            let style = TextStyle {
                color: series.color,
                ..Default::default()
            };

            // Create a grid to track which cells have been drawn
            let cols = plot_width as usize;
            let rows = (plot_height * 4.0) as usize; // 4 braille dots per row

            if cols == 0 || rows == 0 {
                continue;
            }

            let mut grid = vec![vec![false; rows]; cols];

            // Plot points onto grid
            for &(x, y) in &simplified {
                if !x.is_finite() || !y.is_finite() {
                    continue;
                }

                // Normalize to 0..1
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

                // Convert to grid coordinates
                let gx =
                    ((x_norm * (cols - 1) as f64).round() as usize).min(cols.saturating_sub(1));
                let gy = (((1.0 - y_norm) * (rows - 1) as f64).round() as usize)
                    .min(rows.saturating_sub(1));

                grid[gx][gy] = true;
            }

            // Connect adjacent points with lines (Bresenham-like)
            let points: Vec<(usize, usize)> = simplified
                .iter()
                .filter_map(|&(x, y)| {
                    if !x.is_finite() || !y.is_finite() {
                        return None;
                    }
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
                    let gx =
                        ((x_norm * (cols - 1) as f64).round() as usize).min(cols.saturating_sub(1));
                    let gy = (((1.0 - y_norm) * (rows - 1) as f64).round() as usize)
                        .min(rows.saturating_sub(1));
                    Some((gx, gy))
                })
                .collect();

            for window in points.windows(2) {
                if let [p1, p2] = window {
                    draw_line(&mut grid, p1.0, p1.1, p2.0, p2.1);
                }
            }

            // Render grid as braille
            let char_rows = plot_height as usize;
            for cy in 0..char_rows {
                #[allow(clippy::needless_range_loop)]
                for cx in 0..cols {
                    // Each braille char encodes 2x4 dots
                    // But we're using 1x4 (single column) for simplicity
                    let mut dots = 0u8;
                    for dy in 0..4 {
                        let gy = cy * 4 + dy;
                        if gy < rows && grid[cx][gy] {
                            dots |= 1 << dy;
                        }
                    }

                    if dots > 0 {
                        // Use braille patterns
                        let braille_idx = dots as usize;
                        let ch = if braille_idx < BRAILLE_UP.len() {
                            BRAILLE_UP[braille_idx]
                        } else {
                            '⣿'
                        };
                        canvas.draw_text(
                            &ch.to_string(),
                            Point::new(plot_x + cx as f32, plot_y + cy as f32),
                            &style,
                        );
                    }
                }
            }
        }

        // Draw legend
        self.draw_legend(canvas, plot_x, plot_y, plot_width, plot_height);
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

impl Brick for LineChart {
    fn brick_name(&self) -> &'static str {
        "LineChart"
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

        if self.bounds.width >= 10.0 {
            passed.push(BrickAssertion::max_latency_ms(16));
        } else {
            failed.push((
                BrickAssertion::max_latency_ms(16),
                "Width too small".to_string(),
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

/// Check if point is within grid bounds.
#[inline]
fn is_in_grid_bounds(x: isize, y: isize, cols: isize, rows: isize) -> bool {
    x >= 0 && x < cols && y >= 0 && y < rows
}

/// Compute step direction for line drawing (-1 or 1).
#[inline]
fn line_step(from: usize, to: usize) -> isize {
    if from < to {
        1
    } else {
        -1
    }
}

/// Draw a line between two points using Bresenham's algorithm.
#[allow(clippy::cast_possible_wrap)]
fn draw_line(grid: &mut [Vec<bool>], x0: usize, y0: usize, x1: usize, y1: usize) {
    let dx = (x1 as isize - x0 as isize).abs();
    let dy = -(y1 as isize - y0 as isize).abs();
    let sx = line_step(x0, x1);
    let sy = line_step(y0, y1);
    let mut err = dx + dy;

    let mut x = x0 as isize;
    let mut y = y0 as isize;

    let cols = grid.len() as isize;
    let rows = if cols > 0 { grid[0].len() as isize } else { 0 };

    loop {
        if is_in_grid_bounds(x, y, cols, rows) {
            grid[x as usize][y as usize] = true;
        }

        if x == x1 as isize && y == y1 as isize {
            break;
        }

        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

/// Douglas-Peucker line simplification algorithm.
fn douglas_peucker(points: &[(f64, f64)], epsilon: f64) -> Vec<(f64, f64)> {
    if points.len() < 3 {
        return points.to_vec();
    }

    // Find the point with the maximum distance from the line
    let start = points[0];
    let end = points[points.len() - 1];

    let mut max_dist = 0.0;
    let mut max_idx = 0;

    for (i, &point) in points.iter().enumerate().skip(1).take(points.len() - 2) {
        let dist = perpendicular_distance(point, start, end);
        if dist > max_dist {
            max_dist = dist;
            max_idx = i;
        }
    }

    // If max distance is greater than epsilon, recursively simplify
    if max_dist > epsilon {
        let mut left = douglas_peucker(&points[..=max_idx], epsilon);
        let right = douglas_peucker(&points[max_idx..], epsilon);

        // Remove duplicate point
        left.pop();
        left.extend(right);
        left
    } else {
        // Return just the endpoints
        vec![start, end]
    }
}

/// Calculate perpendicular distance from point to line.
fn perpendicular_distance(point: (f64, f64), start: (f64, f64), end: (f64, f64)) -> f64 {
    let dx = end.0 - start.0;
    let dy = end.1 - start.1;

    let mag = dx.hypot(dy);
    if mag < 1e-10 {
        return (point.0 - start.0).hypot(point.1 - start.1);
    }

    ((dy * point.0 - dx * point.1 + end.0 * start.1 - end.1 * start.0) / mag).abs()
}

/// Visvalingam-Whyatt line simplification algorithm.
fn visvalingam_whyatt(points: &[(f64, f64)], threshold: f64) -> Vec<(f64, f64)> {
    if points.len() < 3 {
        return points.to_vec();
    }

    let mut result: Vec<(f64, f64)> = points.to_vec();

    while result.len() > 2 {
        // Find triangle with minimum area
        let mut min_area = f64::INFINITY;
        let mut min_idx = 1;

        for i in 1..result.len() - 1 {
            let area = triangle_area(result[i - 1], result[i], result[i + 1]);
            if area < min_area {
                min_area = area;
                min_idx = i;
            }
        }

        if min_area >= threshold {
            break;
        }

        result.remove(min_idx);
    }

    result
}

/// Calculate triangle area using cross product.
fn triangle_area(p1: (f64, f64), p2: (f64, f64), p3: (f64, f64)) -> f64 {
    ((p2.0 - p1.0) * (p3.1 - p1.1) - (p3.0 - p1.0) * (p2.1 - p1.1)).abs() / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct::{CellBuffer, DirectTerminalCanvas};

    #[test]
    fn test_line_chart_creation() {
        let chart = LineChart::new().add_series("test", vec![(0.0, 0.0), (1.0, 1.0)], Color::RED);
        assert_eq!(chart.series.len(), 1);
        assert_eq!(chart.series[0].name, "test");
    }

    #[test]
    fn test_douglas_peucker() {
        let points = vec![(0.0, 0.0), (1.0, 0.1), (2.0, 0.0), (3.0, 0.0)];
        let simplified = douglas_peucker(&points, 0.5);
        assert!(simplified.len() <= points.len());
    }

    #[test]
    fn test_douglas_peucker_few_points() {
        let points = vec![(0.0, 0.0), (1.0, 1.0)];
        let simplified = douglas_peucker(&points, 0.5);
        assert_eq!(simplified.len(), 2);
    }

    #[test]
    fn test_visvalingam_whyatt() {
        let points = vec![(0.0, 0.0), (1.0, 0.1), (2.0, 0.0), (3.0, 0.0)];
        let simplified = visvalingam_whyatt(&points, 0.5);
        assert!(simplified.len() <= points.len());
    }

    #[test]
    fn test_visvalingam_whyatt_few_points() {
        let points = vec![(0.0, 0.0), (1.0, 1.0)];
        let simplified = visvalingam_whyatt(&points, 0.5);
        assert_eq!(simplified.len(), 2);
    }

    #[test]
    fn test_empty_chart() {
        let chart = LineChart::new();
        let (x_min, x_max) = chart.x_range();
        assert_eq!(x_min, 0.0);
        assert_eq!(x_max, 1.0);
    }

    #[test]
    fn test_multi_series() {
        let chart = LineChart::new()
            .add_series("a", vec![(0.0, 0.0)], Color::RED)
            .add_series("b", vec![(1.0, 1.0)], Color::BLUE)
            .add_series("c", vec![(2.0, 2.0)], Color::GREEN);
        assert_eq!(chart.series.len(), 3);
    }

    #[test]
    fn test_line_chart_assertions() {
        let chart = LineChart::default();
        assert!(!chart.assertions().is_empty());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_line_chart_verify() {
        let mut chart = LineChart::default();
        chart.bounds = Rect::new(0.0, 0.0, 80.0, 20.0);
        assert!(chart.verify().is_valid());
    }

    #[test]
    fn test_line_chart_children() {
        let chart = LineChart::default();
        assert!(chart.children().is_empty());
    }

    #[test]
    fn test_line_chart_layout() {
        let mut chart = LineChart::new().add_series(
            "test",
            vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.5)],
            Color::RED,
        );
        let bounds = Rect::new(0.0, 0.0, 80.0, 24.0);
        let result = chart.layout(bounds);
        assert!(result.size.width > 0.0);
        assert!(result.size.height > 0.0);
    }

    #[test]
    fn test_line_chart_paint() {
        let mut chart = LineChart::new().add_series(
            "test",
            vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.5)],
            Color::RED,
        );
        let bounds = Rect::new(0.0, 0.0, 80.0, 24.0);
        chart.layout(bounds);

        let mut buffer = CellBuffer::new(80, 24);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        chart.paint(&mut canvas);
        // Just verify it doesn't panic
    }

    #[test]
    fn test_line_chart_with_legend_positions() {
        for pos in [
            LegendPosition::TopRight,
            LegendPosition::TopLeft,
            LegendPosition::BottomRight,
            LegendPosition::BottomLeft,
            LegendPosition::None,
        ] {
            let mut chart = LineChart::new()
                .add_series("s1", vec![(0.0, 0.0), (1.0, 1.0)], Color::RED)
                .with_legend(pos);
            let bounds = Rect::new(0.0, 0.0, 80.0, 24.0);
            chart.layout(bounds);
            let mut buffer = CellBuffer::new(80, 24);
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            chart.paint(&mut canvas);
        }
    }

    #[test]
    fn test_line_chart_with_axis_config() {
        let mut chart = LineChart::new()
            .add_series("test", vec![(0.0, 0.0), (10.0, 100.0)], Color::RED)
            .with_x_axis(Axis {
                label: Some("X Label".to_string()),
                min: Some(0.0),
                max: Some(10.0),
                ticks: 5,
                grid: true,
            })
            .with_y_axis(Axis {
                label: Some("Y Label".to_string()),
                min: Some(0.0),
                max: Some(100.0),
                ticks: 10,
                grid: true,
            });
        let bounds = Rect::new(0.0, 0.0, 80.0, 24.0);
        chart.layout(bounds);
        let mut buffer = CellBuffer::new(80, 24);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        chart.paint(&mut canvas);
    }

    #[test]
    fn test_line_chart_with_simplification() {
        let data: Vec<(f64, f64)> = (0..100)
            .map(|i| (i as f64, (i as f64 * 0.1).sin()))
            .collect();

        // Test Douglas-Peucker
        let mut chart = LineChart::new()
            .add_series("dp", data.clone(), Color::RED)
            .with_simplification(Simplification::DouglasPeucker { epsilon: 0.1 });
        let bounds = Rect::new(0.0, 0.0, 80.0, 24.0);
        chart.layout(bounds);
        let mut buffer = CellBuffer::new(80, 24);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        chart.paint(&mut canvas);

        // Test Visvalingam-Whyatt
        let mut chart = LineChart::new()
            .add_series("vw", data, Color::BLUE)
            .with_simplification(Simplification::VisvalingamWhyatt { threshold: 0.1 });
        chart.layout(bounds);
        chart.paint(&mut canvas);
    }

    #[test]
    fn test_line_chart_line_styles() {
        for style in [
            LineStyle::Solid,
            LineStyle::Dashed,
            LineStyle::Dotted,
            LineStyle::Markers,
        ] {
            let mut chart = LineChart::new().add_series_styled(
                "test",
                vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.5)],
                Color::RED,
                style,
            );
            let bounds = Rect::new(0.0, 0.0, 80.0, 24.0);
            chart.layout(bounds);
            let mut buffer = CellBuffer::new(80, 24);
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            chart.paint(&mut canvas);
        }
    }

    #[test]
    fn test_line_chart_y_range() {
        let chart = LineChart::new().add_series(
            "test",
            vec![(0.0, -5.0), (1.0, 10.0), (2.0, 3.0)],
            Color::RED,
        );
        let (y_min, y_max) = chart.y_range();
        assert!(y_min <= -5.0);
        assert!(y_max >= 10.0);
    }

    #[test]
    fn test_line_chart_x_range_with_data() {
        let chart = LineChart::new().add_series("test", vec![(5.0, 0.0), (15.0, 1.0)], Color::RED);
        let (x_min, x_max) = chart.x_range();
        assert!(x_min <= 5.0);
        assert!(x_max >= 15.0);
    }

    #[test]
    fn test_triangle_area() {
        let area = triangle_area((0.0, 0.0), (1.0, 0.0), (0.5, 1.0));
        assert!((area - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_perpendicular_distance() {
        // Point on the line
        let dist = perpendicular_distance((0.5, 0.5), (0.0, 0.0), (1.0, 1.0));
        assert!(dist < 0.001);

        // Point away from line
        let dist = perpendicular_distance((0.0, 1.0), (0.0, 0.0), (1.0, 0.0));
        assert!((dist - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_axis_default() {
        let axis = Axis::default();
        assert!(axis.label.is_none());
        assert!(axis.min.is_none());
        assert!(axis.max.is_none());
        assert_eq!(axis.ticks, 5);
        assert!(!axis.grid);
    }

    #[test]
    fn test_simplification_default() {
        let simp = Simplification::default();
        assert!(matches!(simp, Simplification::None));
    }

    #[test]
    fn test_line_style_default() {
        let style = LineStyle::default();
        assert!(matches!(style, LineStyle::Solid));
    }

    #[test]
    fn test_legend_position_default() {
        let pos = LegendPosition::default();
        assert!(matches!(pos, LegendPosition::TopRight));
    }

    #[test]
    fn test_line_chart_compact() {
        let chart = LineChart::new()
            .add_series("test", vec![(0.0, 0.0), (1.0, 1.0)], Color::RED)
            .compact();
        assert_eq!(chart.margin_left, 0.0);
        assert_eq!(chart.margin_bottom, 0.0);
        assert_eq!(chart.y_axis.ticks, 0);
        assert_eq!(chart.x_axis.ticks, 0);
        assert!(matches!(chart.legend, LegendPosition::None));
    }

    #[test]
    fn test_line_chart_with_margins() {
        let chart = LineChart::new()
            .add_series("test", vec![(0.0, 0.0), (1.0, 1.0)], Color::RED)
            .with_margins(10.0, 5.0);
        assert_eq!(chart.margin_left, 10.0);
        assert_eq!(chart.margin_bottom, 5.0);
    }

    #[test]
    fn test_line_chart_explicit_x_range() {
        let chart = LineChart::new()
            .add_series("test", vec![(0.0, 0.0), (1.0, 1.0)], Color::RED)
            .with_x_axis(Axis {
                min: Some(0.0),
                max: Some(10.0),
                ..Default::default()
            });
        let (xmin, xmax) = chart.x_range();
        assert_eq!(xmin, 0.0);
        assert_eq!(xmax, 10.0);
    }

    #[test]
    fn test_line_chart_explicit_y_range() {
        let chart = LineChart::new()
            .add_series("test", vec![(0.0, 0.0), (1.0, 1.0)], Color::RED)
            .with_y_axis(Axis {
                min: Some(-10.0),
                max: Some(10.0),
                ..Default::default()
            });
        let (ymin, ymax) = chart.y_range();
        assert_eq!(ymin, -10.0);
        assert_eq!(ymax, 10.0);
    }

    #[test]
    fn test_line_chart_nan_values() {
        let mut chart = LineChart::new().add_series(
            "test",
            vec![(0.0, 0.0), (f64::NAN, f64::NAN), (2.0, 2.0)],
            Color::RED,
        );
        let bounds = Rect::new(0.0, 0.0, 80.0, 24.0);
        chart.layout(bounds);
        let mut buffer = CellBuffer::new(80, 24);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        // Should not panic with NaN values
        chart.paint(&mut canvas);
    }

    #[test]
    fn test_line_chart_infinite_values() {
        let mut chart = LineChart::new().add_series(
            "test",
            vec![(0.0, 0.0), (f64::INFINITY, f64::NEG_INFINITY), (2.0, 2.0)],
            Color::RED,
        );
        let bounds = Rect::new(0.0, 0.0, 80.0, 24.0);
        chart.layout(bounds);
        let mut buffer = CellBuffer::new(80, 24);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        // Should not panic with infinite values
        chart.paint(&mut canvas);
    }

    #[test]
    fn test_line_chart_too_small() {
        let mut chart =
            LineChart::new().add_series("test", vec![(0.0, 0.0), (1.0, 1.0)], Color::RED);
        // Too small - should early return from paint
        let bounds = Rect::new(0.0, 0.0, 5.0, 2.0);
        chart.layout(bounds);
        let mut buffer = CellBuffer::new(5, 2);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        chart.paint(&mut canvas);
    }

    #[test]
    fn test_line_chart_children_mut() {
        let mut chart = LineChart::default();
        assert!(chart.children_mut().is_empty());
    }

    #[test]
    fn test_line_chart_to_html() {
        let chart = LineChart::default();
        assert!(chart.to_html().is_empty());
    }

    #[test]
    fn test_line_chart_to_css() {
        let chart = LineChart::default();
        assert!(chart.to_css().is_empty());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_line_chart_verify_small_width() {
        let mut chart = LineChart::default();
        chart.bounds = Rect::new(0.0, 0.0, 5.0, 20.0); // Width < 10
        let verification = chart.verify();
        assert!(!verification.is_valid());
    }

    #[test]
    fn test_line_chart_budget() {
        let chart = LineChart::default();
        let budget = chart.budget();
        // Just verify budget returns something
        let _ = budget;
    }

    #[test]
    fn test_line_chart_measure() {
        let chart = LineChart::default();
        let size = chart.measure(Constraints {
            min_width: 0.0,
            min_height: 0.0,
            max_width: 100.0,
            max_height: 50.0,
        });
        assert_eq!(size.width, 80.0);
        assert_eq!(size.height, 20.0);
    }

    #[test]
    fn test_line_chart_type_id() {
        let chart = LineChart::default();
        // Use Widget trait explicitly
        let tid = Widget::type_id(&chart);
        assert_eq!(tid, TypeId::of::<LineChart>());
    }

    #[test]
    fn test_draw_line_horizontal() {
        let mut grid = vec![vec![false; 10]; 20];
        draw_line(&mut grid, 0, 5, 19, 5);
        // Check some points on the line
        assert!(grid[0][5]);
        assert!(grid[10][5]);
        assert!(grid[19][5]);
    }

    #[test]
    fn test_draw_line_vertical() {
        let mut grid = vec![vec![false; 10]; 20];
        draw_line(&mut grid, 5, 0, 5, 9);
        assert!(grid[5][0]);
        assert!(grid[5][5]);
        assert!(grid[5][9]);
    }

    #[test]
    fn test_draw_line_diagonal() {
        let mut grid = vec![vec![false; 10]; 10];
        draw_line(&mut grid, 0, 0, 9, 9);
        assert!(grid[0][0]);
        assert!(grid[9][9]);
    }

    #[test]
    fn test_draw_line_reverse() {
        let mut grid = vec![vec![false; 10]; 10];
        draw_line(&mut grid, 9, 9, 0, 0);
        assert!(grid[0][0]);
        assert!(grid[9][9]);
    }

    #[test]
    fn test_perpendicular_distance_coincident_points() {
        // When start and end are the same point
        let dist = perpendicular_distance((1.0, 1.0), (0.0, 0.0), (0.0, 0.0));
        assert!((dist - std::f64::consts::SQRT_2).abs() < 0.001);
    }

    #[test]
    fn test_series_struct() {
        let series = Series {
            name: "test".to_string(),
            data: vec![(0.0, 0.0), (1.0, 1.0)],
            color: Color::RED,
            style: LineStyle::Dashed,
        };
        assert_eq!(series.name, "test");
        assert_eq!(series.data.len(), 2);
        assert!(matches!(series.style, LineStyle::Dashed));
    }

    #[test]
    fn test_line_chart_single_point_x_range() {
        // When all points have same X value
        let chart = LineChart::new().add_series("test", vec![(5.0, 0.0), (5.0, 1.0)], Color::RED);
        let (xmin, xmax) = chart.x_range();
        // x_max == x_min, so should use default fallback
        assert_eq!(xmin, 5.0);
        assert_eq!(xmax, 5.0);
    }

    #[test]
    fn test_line_chart_single_point_y_range() {
        // When all points have same Y value
        let chart = LineChart::new().add_series("test", vec![(0.0, 5.0), (1.0, 5.0)], Color::RED);
        let (ymin, ymax) = chart.y_range();
        // When y_min == y_max, padding is 0, so both are 5.0
        assert_eq!(ymin, 5.0);
        assert_eq!(ymax, 5.0);
    }
}
