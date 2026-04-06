//! Line chart implementation with Douglas-Peucker simplification.
//!
//! Performance-optimized line rendering for time series and continuous data.
//!
//! # Algorithms
//!
//! - **Douglas-Peucker**: Line simplification for large datasets
//! - **Wu's Line Algorithm**: Anti-aliased rendering
//!
//! # References
//!
//! - Douglas, D. H., & Peucker, T. K. (1973). "Algorithms for the reduction of
//!   the number of points required to represent a digitized line or its caricature."
//!   Cartographica, 10(2), 112-122.
//! - Wu, X. (1991). "An Efficient Antialiasing Technique." SIGGRAPH '91.

use crate::color::Rgba;
use crate::error::{Error, Result};
use crate::framebuffer::Framebuffer;
use crate::geometry::Point;
use crate::render::{draw_line, draw_line_aa};
use crate::scale::{LinearScale, Scale};

// ============================================================================
// Douglas-Peucker Line Simplification
// ============================================================================

/// Simplify a polyline using the Douglas-Peucker algorithm.
///
/// This algorithm recursively decimates a curve composed of line segments to a
/// similar curve with fewer points. The simplification threshold (epsilon) determines
/// the maximum allowed perpendicular distance from the simplified line to points
/// in the original curve.
///
/// # Arguments
///
/// * `points` - The original points
/// * `epsilon` - Maximum perpendicular distance threshold (in pixels)
///
/// # Returns
///
/// A simplified list of points
///
/// # References
///
/// Douglas, D. H., & Peucker, T. K. (1973).
pub fn douglas_peucker(points: &[Point], epsilon: f32) -> Vec<Point> {
    if points.len() < 3 {
        return points.to_vec();
    }

    // Find the point with the maximum distance from the line segment
    let (max_distance, max_index) = find_max_distance(points);

    // If max distance is greater than epsilon, recursively simplify
    if max_distance > epsilon {
        // Recursive call on both halves
        let left = douglas_peucker(&points[..=max_index], epsilon);
        let right = douglas_peucker(&points[max_index..], epsilon);

        // Combine results, avoiding duplicate of the split point
        let mut result = left;
        result.extend_from_slice(&right[1..]);
        result
    } else {
        // Return just the endpoints (safe: we checked len >= 3 above)
        vec![points[0], points[points.len() - 1]]
    }
}

/// Find the point with maximum perpendicular distance from the line between first and last points.
fn find_max_distance(points: &[Point]) -> (f32, usize) {
    let first = points[0];
    let last = points[points.len() - 1];

    let mut max_distance = 0.0;
    let mut max_index = 0;

    for (i, point) in points.iter().enumerate().skip(1).take(points.len() - 2) {
        let distance = perpendicular_distance(*point, first, last);
        if distance > max_distance {
            max_distance = distance;
            max_index = i;
        }
    }

    (max_distance, max_index)
}

/// Calculate perpendicular distance from a point to a line segment.
fn perpendicular_distance(point: Point, line_start: Point, line_end: Point) -> f32 {
    let dx = line_end.x - line_start.x;
    let dy = line_end.y - line_start.y;

    // Handle degenerate case (line_start == line_end)
    let line_length_sq = dx * dx + dy * dy;
    if line_length_sq < f32::EPSILON {
        return point.distance(line_start);
    }

    // Calculate perpendicular distance using cross product formula
    let numerator = ((dy * point.x) - (dx * point.y) + (line_end.x * line_start.y)
        - (line_end.y * line_start.x))
        .abs();
    let denominator = line_length_sq.sqrt();

    numerator / denominator
}

// ============================================================================
// Line Series
// ============================================================================

/// A data series for line charts.
#[derive(Debug, Clone)]
pub struct LineSeries {
    /// Series name/label.
    pub name: String,
    /// X-axis data.
    pub x_data: Vec<f32>,
    /// Y-axis data.
    pub y_data: Vec<f32>,
    /// Line color.
    pub color: Rgba,
    /// Line thickness.
    pub thickness: f32,
    /// Use anti-aliasing.
    pub antialiased: bool,
}

impl LineSeries {
    /// Create a new line series.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            x_data: Vec::new(),
            y_data: Vec::new(),
            color: Rgba::BLUE,
            thickness: 1.0,
            antialiased: true,
        }
    }

    /// Set the x and y data.
    #[must_use]
    pub fn data(mut self, x: &[f32], y: &[f32]) -> Self {
        self.x_data = x.to_vec();
        self.y_data = y.to_vec();
        self
    }

    /// Set the line color.
    #[must_use]
    pub fn color(mut self, color: Rgba) -> Self {
        self.color = color;
        self
    }

    /// Set the line thickness.
    #[must_use]
    pub fn thickness(mut self, thickness: f32) -> Self {
        self.thickness = thickness.max(0.5);
        self
    }

    /// Enable or disable anti-aliasing.
    #[must_use]
    pub fn antialiased(mut self, enabled: bool) -> Self {
        self.antialiased = enabled;
        self
    }

    /// Get the number of points.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.x_data.len().min(self.y_data.len())
    }
}

// ============================================================================
// Line Chart
// ============================================================================

/// Builder for creating line charts.
#[derive(Debug, Clone)]
pub struct LineChart {
    /// Data series.
    series: Vec<LineSeries>,
    /// Output width in pixels.
    width: u32,
    /// Output height in pixels.
    height: u32,
    /// Margin around the plot.
    margin: u32,
    /// Douglas-Peucker simplification epsilon (0 = disabled).
    simplify_epsilon: f32,
    /// Show data points as markers.
    show_markers: bool,
    /// Marker size.
    marker_size: f32,
}

impl Default for LineChart {
    fn default() -> Self {
        Self::new()
    }
}

impl LineChart {
    /// Create a new line chart builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            series: Vec::new(),
            width: 800,
            height: 600,
            margin: 40,
            simplify_epsilon: 0.0,
            show_markers: false,
            marker_size: 4.0,
        }
    }

    /// Add a data series.
    #[must_use]
    pub fn add_series(mut self, series: LineSeries) -> Self {
        self.series.push(series);
        self
    }

    /// Add data as a single series (convenience method).
    #[must_use]
    pub fn data(self, x: &[f32], y: &[f32]) -> Self {
        let series = LineSeries::new("default").data(x, y);
        self.add_series(series)
    }

    /// Set the line color for the first/default series.
    #[must_use]
    pub fn color(mut self, color: Rgba) -> Self {
        if let Some(series) = self.series.last_mut() {
            series.color = color;
        }
        self
    }

    /// Set the margin around the plot.
    #[must_use]
    pub fn margin(mut self, margin: u32) -> Self {
        self.margin = margin;
        self
    }

    /// Enable Douglas-Peucker simplification.
    ///
    /// The epsilon value determines the maximum allowed perpendicular distance
    /// from the simplified line to points in the original curve.
    /// Larger values produce more aggressive simplification.
    ///
    /// Set to 0 to disable simplification.
    #[must_use]
    pub fn simplify(mut self, epsilon: f32) -> Self {
        self.simplify_epsilon = epsilon.max(0.0);
        self
    }

    /// Enable or disable data point markers.
    #[must_use]
    pub fn markers(mut self, show: bool) -> Self {
        self.show_markers = show;
        self
    }

    /// Set the marker size.
    #[must_use]
    pub fn marker_size(mut self, size: f32) -> Self {
        self.marker_size = size.max(1.0);
        self
    }

    /// Build and validate the line chart.
    ///
    /// # Errors
    ///
    /// Returns an error if no data series or data is empty.
    pub fn build(self) -> Result<Self> {
        if self.series.is_empty() {
            return Err(Error::EmptyData);
        }

        for series in &self.series {
            if series.x_data.is_empty() || series.y_data.is_empty() {
                return Err(Error::EmptyData);
            }

            if series.x_data.len() != series.y_data.len() {
                return Err(Error::DataLengthMismatch {
                    x_len: series.x_data.len(),
                    y_len: series.y_data.len(),
                });
            }
        }

        Ok(self)
    }

    /// Get the data extent across all series.
    fn data_extent(&self) -> ((f32, f32), (f32, f32)) {
        let mut x_min = f32::INFINITY;
        let mut x_max = f32::NEG_INFINITY;
        let mut y_min = f32::INFINITY;
        let mut y_max = f32::NEG_INFINITY;

        for series in &self.series {
            for &x in &series.x_data {
                x_min = x_min.min(x);
                x_max = x_max.max(x);
            }
            for &y in &series.y_data {
                y_min = y_min.min(y);
                y_max = y_max.max(y);
            }
        }

        ((x_min, x_max), (y_min, y_max))
    }

    /// Render the line chart to a framebuffer.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering fails.
    pub fn render(&self, fb: &mut Framebuffer) -> Result<()> {
        let ((x_min, x_max), (y_min, y_max)) = self.data_extent();

        // Calculate plot area
        let plot_width = self.width - 2 * self.margin;
        let plot_height = self.height - 2 * self.margin;

        // Create scales
        let x_scale = LinearScale::new(
            (x_min, x_max),
            (self.margin as f32, (self.margin + plot_width) as f32),
        )?;
        let y_scale = LinearScale::new(
            (y_min, y_max),
            ((self.margin + plot_height) as f32, self.margin as f32),
        )?;

        // Render each series
        for series in &self.series {
            self.render_series(fb, series, &x_scale, &y_scale);
        }

        Ok(())
    }

    /// Render a single series.
    fn render_series(
        &self,
        fb: &mut Framebuffer,
        series: &LineSeries,
        x_scale: &LinearScale,
        y_scale: &LinearScale,
    ) {
        let point_count = series.point_count();
        if point_count < 2 {
            return;
        }

        // Convert data to screen coordinates
        let mut points: Vec<Point> = (0..point_count)
            .map(|i| Point::new(x_scale.scale(series.x_data[i]), y_scale.scale(series.y_data[i])))
            .collect();

        // Apply Douglas-Peucker simplification if enabled
        if self.simplify_epsilon > 0.0 {
            points = douglas_peucker(&points, self.simplify_epsilon);
        }

        // Draw lines between consecutive points
        for i in 0..points.len() - 1 {
            let p1 = points[i];
            let p2 = points[i + 1];

            if series.antialiased {
                draw_line_aa(fb, p1.x, p1.y, p2.x, p2.y, series.color);
            } else {
                draw_line(fb, p1.x as i32, p1.y as i32, p2.x as i32, p2.y as i32, series.color);
            }
        }

        // Draw markers if enabled
        if self.show_markers {
            for point in &points {
                self.draw_marker(fb, point.x, point.y, series.color);
            }
        }
    }

    /// Draw a circular marker at the given position.
    fn draw_marker(&self, fb: &mut Framebuffer, x: f32, y: f32, color: Rgba) {
        let radius = (self.marker_size / 2.0) as i32;
        let cx = x as i32;
        let cy = y as i32;

        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    let px = cx + dx;
                    let py = cy + dy;
                    if px >= 0 && py >= 0 {
                        fb.set_pixel(px as u32, py as u32, color);
                    }
                }
            }
        }
    }

    /// Render to a new framebuffer.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering fails.
    pub fn to_framebuffer(&self) -> Result<Framebuffer> {
        let mut fb = Framebuffer::new(self.width, self.height)?;
        fb.clear(Rgba::WHITE);
        self.render(&mut fb)?;
        Ok(fb)
    }

    /// Get the total number of points across all series.
    #[must_use]
    pub fn total_points(&self) -> usize {
        self.series.iter().map(LineSeries::point_count).sum()
    }

    /// Get the number of series.
    #[must_use]
    pub fn series_count(&self) -> usize {
        self.series.len()
    }
}

impl batuta_common::display::WithDimensions for LineChart {
    fn set_dimensions(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use batuta_common::display::WithDimensions;

    #[test]
    fn test_douglas_peucker_simple() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.1),
            Point::new(2.0, -0.1),
            Point::new(3.0, 5.0),
            Point::new(4.0, 6.0),
            Point::new(5.0, 7.0),
            Point::new(6.0, 8.0),
            Point::new(7.0, 9.0),
        ];

        let simplified = douglas_peucker(&points, 1.0);

        // Should reduce the number of points
        assert!(simplified.len() < points.len());
        // First and last points should be preserved
        assert!(simplified.first().expect("collection should not be empty").x.abs() < f32::EPSILON);
        assert!(
            (simplified.last().expect("collection should not be empty").x - 7.0).abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn test_douglas_peucker_straight_line() {
        let points: Vec<Point> = (0..10).map(|i| Point::new(i as f32, i as f32)).collect();

        let simplified = douglas_peucker(&points, 0.1);

        // Straight line should simplify to just 2 points
        assert_eq!(simplified.len(), 2);
    }

    #[test]
    fn test_douglas_peucker_too_few_points() {
        let points = vec![Point::new(0.0, 0.0), Point::new(1.0, 1.0)];

        let simplified = douglas_peucker(&points, 1.0);

        // Should return original points
        assert_eq!(simplified.len(), 2);
    }

    #[test]
    fn test_line_series_builder() {
        let series = LineSeries::new("test")
            .data(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0])
            .color(Rgba::RED)
            .thickness(2.0)
            .antialiased(true);

        assert_eq!(series.name, "test");
        assert_eq!(series.point_count(), 3);
        assert_eq!(series.color, Rgba::RED);
    }

    #[test]
    fn test_line_chart_builder() {
        let chart = LineChart::new()
            .data(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0])
            .dimensions(100, 100)
            .build()
            .expect("operation should succeed");

        assert_eq!(chart.series_count(), 1);
        assert_eq!(chart.total_points(), 3);
    }

    #[test]
    fn test_line_chart_empty_data() {
        let result = LineChart::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_line_chart_data_mismatch() {
        let result = LineChart::new().data(&[1.0, 2.0, 3.0], &[4.0, 5.0]).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_line_chart_render() {
        let chart = LineChart::new()
            .data(&[0.0, 1.0, 2.0, 3.0], &[0.0, 1.0, 0.5, 2.0])
            .dimensions(100, 100)
            .build()
            .expect("operation should succeed");

        let fb = chart.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_line_chart_with_simplification() {
        let x: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let y: Vec<f32> = x.iter().map(|&x| (x * 0.1).sin()).collect();

        let chart = LineChart::new()
            .data(&x, &y)
            .simplify(1.0)
            .dimensions(200, 100)
            .build()
            .expect("builder should produce valid result");

        let fb = chart.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_line_chart_multi_series() {
        let series1 =
            LineSeries::new("series1").data(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0]).color(Rgba::RED);

        let series2 =
            LineSeries::new("series2").data(&[0.0, 1.0, 2.0], &[2.0, 1.0, 0.0]).color(Rgba::BLUE);

        let chart = LineChart::new()
            .add_series(series1)
            .add_series(series2)
            .dimensions(100, 100)
            .build()
            .expect("operation should succeed");

        assert_eq!(chart.series_count(), 2);

        let fb = chart.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_line_chart_with_markers() {
        let chart = LineChart::new()
            .data(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0])
            .markers(true)
            .marker_size(6.0)
            .dimensions(100, 100)
            .build()
            .expect("operation should succeed");

        let fb = chart.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_perpendicular_distance() {
        // Point directly on the line should have distance 0
        let distance = perpendicular_distance(
            Point::new(1.0, 1.0),
            Point::new(0.0, 0.0),
            Point::new(2.0, 2.0),
        );
        assert!(distance.abs() < 0.001);

        // Point perpendicular to a horizontal line
        let distance = perpendicular_distance(
            Point::new(1.0, 1.0),
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
        );
        assert!((distance - 1.0).abs() < 0.001);
    }
}
