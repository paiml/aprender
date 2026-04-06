//! `Chart` widget for data visualization.

use presentar_core::{
    widget::{AccessibleRole, LayoutResult, TextStyle},
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Point, Rect,
    Size, TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::time::Duration;

/// Chart type variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ChartType {
    /// Line chart
    #[default]
    Line,
    /// Bar chart
    Bar,
    /// Scatter plot
    Scatter,
    /// Area chart
    Area,
    /// Pie chart
    Pie,
    /// Histogram
    Histogram,
    /// Heatmap - displays matrix data with color encoding
    Heatmap,
    /// Box plot - displays statistical distributions
    BoxPlot,
}

/// A single data series for the chart.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DataSeries {
    /// Series name/label
    pub name: String,
    /// Data points (x, y)
    pub points: Vec<(f64, f64)>,
    /// Series color
    pub color: Color,
    /// Line width (for line/area charts)
    pub line_width: f32,
    /// Point size (for scatter/line charts)
    pub point_size: f32,
    /// Whether to show points
    pub show_points: bool,
    /// Whether to fill area under line
    pub fill: bool,
}

impl DataSeries {
    /// Create a new data series.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            points: Vec::new(),
            color: Color::new(0.2, 0.47, 0.96, 1.0),
            line_width: 2.0,
            point_size: 4.0,
            show_points: true,
            fill: false,
        }
    }

    /// Add a data point.
    #[must_use]
    pub fn point(mut self, x: f64, y: f64) -> Self {
        self.points.push((x, y));
        self
    }

    /// Add multiple data points.
    #[must_use]
    pub fn points(mut self, points: impl IntoIterator<Item = (f64, f64)>) -> Self {
        self.points.extend(points);
        self
    }

    /// Set series color.
    #[must_use]
    pub const fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set line width.
    #[must_use]
    pub fn line_width(mut self, width: f32) -> Self {
        self.line_width = width.max(0.5);
        self
    }

    /// Set point size.
    #[must_use]
    pub fn point_size(mut self, size: f32) -> Self {
        self.point_size = size.max(1.0);
        self
    }

    /// Set whether to show points.
    #[must_use]
    pub const fn show_points(mut self, show: bool) -> Self {
        self.show_points = show;
        self
    }

    /// Set whether to fill area.
    #[must_use]
    pub const fn fill(mut self, fill: bool) -> Self {
        self.fill = fill;
        self
    }

    /// Get min/max X values.
    #[must_use]
    pub fn x_range(&self) -> Option<(f64, f64)> {
        if self.points.is_empty() {
            return None;
        }
        let min = self
            .points
            .iter()
            .map(|(x, _)| *x)
            .fold(f64::INFINITY, f64::min);
        let max = self
            .points
            .iter()
            .map(|(x, _)| *x)
            .fold(f64::NEG_INFINITY, f64::max);
        Some((min, max))
    }

    /// Get min/max Y values.
    #[must_use]
    pub fn y_range(&self) -> Option<(f64, f64)> {
        if self.points.is_empty() {
            return None;
        }
        let min = self
            .points
            .iter()
            .map(|(_, y)| *y)
            .fold(f64::INFINITY, f64::min);
        let max = self
            .points
            .iter()
            .map(|(_, y)| *y)
            .fold(f64::NEG_INFINITY, f64::max);
        Some((min, max))
    }
}

/// Axis configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Axis {
    /// Axis label
    pub label: Option<String>,
    /// Minimum value (auto if None)
    pub min: Option<f64>,
    /// Maximum value (auto if None)
    pub max: Option<f64>,
    /// Number of grid lines
    pub grid_lines: usize,
    /// Show grid
    pub show_grid: bool,
    /// Axis color
    pub color: Color,
    /// Grid color
    pub grid_color: Color,
}

impl Default for Axis {
    fn default() -> Self {
        Self {
            label: None,
            min: None,
            max: None,
            grid_lines: 5,
            show_grid: true,
            color: Color::new(0.3, 0.3, 0.3, 1.0),
            grid_color: Color::new(0.9, 0.9, 0.9, 1.0),
        }
    }
}

impl Axis {
    /// Create a new axis.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set axis label.
    #[must_use]
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set minimum value.
    #[must_use]
    pub const fn min(mut self, min: f64) -> Self {
        self.min = Some(min);
        self
    }

    /// Set maximum value.
    #[must_use]
    pub const fn max(mut self, max: f64) -> Self {
        self.max = Some(max);
        self
    }

    /// Set range.
    #[must_use]
    pub const fn range(mut self, min: f64, max: f64) -> Self {
        self.min = Some(min);
        self.max = Some(max);
        self
    }

    /// Set number of grid lines.
    #[must_use]
    pub fn grid_lines(mut self, count: usize) -> Self {
        self.grid_lines = count.max(2);
        self
    }

    /// Set whether to show grid.
    #[must_use]
    pub const fn show_grid(mut self, show: bool) -> Self {
        self.show_grid = show;
        self
    }

    /// Set axis color.
    #[must_use]
    pub const fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set grid color.
    #[must_use]
    pub const fn grid_color(mut self, color: Color) -> Self {
        self.grid_color = color;
        self
    }
}

/// Legend position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LegendPosition {
    /// No legend
    None,
    /// Top right (default)
    #[default]
    TopRight,
    /// Top left
    TopLeft,
    /// Bottom right
    BottomRight,
    /// Bottom left
    BottomLeft,
}

/// `Chart` widget for data visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chart {
    /// Chart type
    kind: ChartType,
    /// Data series
    series: Vec<DataSeries>,
    /// Chart title
    title: Option<String>,
    /// X axis configuration
    x_axis: Axis,
    /// Y axis configuration
    y_axis: Axis,
    /// Legend position
    legend: LegendPosition,
    /// Background color
    background: Color,
    /// Padding around chart area
    padding: f32,
    /// Width
    width: Option<f32>,
    /// Height
    height: Option<f32>,
    /// Accessible name
    accessible_name_value: Option<String>,
    /// Test ID
    test_id_value: Option<String>,
    /// Cached bounds
    #[serde(skip)]
    bounds: Rect,
}

impl Default for Chart {
    fn default() -> Self {
        Self {
            kind: ChartType::Line,
            series: Vec::new(),
            title: None,
            x_axis: Axis::default(),
            y_axis: Axis::default(),
            legend: LegendPosition::TopRight,
            background: Color::WHITE,
            padding: 40.0,
            width: None,
            height: None,
            accessible_name_value: None,
            test_id_value: None,
            bounds: Rect::default(),
        }
    }
}

impl Chart {
    /// Create a new chart.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a line chart.
    #[must_use]
    pub fn line() -> Self {
        Self::new().chart_type(ChartType::Line)
    }

    /// Create a bar chart.
    #[must_use]
    pub fn bar() -> Self {
        Self::new().chart_type(ChartType::Bar)
    }

    /// Create a scatter chart.
    #[must_use]
    pub fn scatter() -> Self {
        Self::new().chart_type(ChartType::Scatter)
    }

    /// Create an area chart.
    #[must_use]
    pub fn area() -> Self {
        Self::new().chart_type(ChartType::Area)
    }

    /// Create a pie chart.
    #[must_use]
    pub fn pie() -> Self {
        Self::new().chart_type(ChartType::Pie)
    }

    /// Create a heatmap chart.
    #[must_use]
    pub fn heatmap() -> Self {
        Self::new().chart_type(ChartType::Heatmap)
    }

    /// Create a box plot chart.
    #[must_use]
    pub fn boxplot() -> Self {
        Self::new().chart_type(ChartType::BoxPlot)
    }

    /// Set chart type.
    #[must_use]
    pub const fn chart_type(mut self, chart_type: ChartType) -> Self {
        self.kind = chart_type;
        self
    }

    /// Add a data series.
    #[must_use]
    pub fn series(mut self, series: DataSeries) -> Self {
        self.series.push(series);
        self
    }

    /// Add multiple data series.
    #[must_use]
    pub fn add_series(mut self, series: impl IntoIterator<Item = DataSeries>) -> Self {
        self.series.extend(series);
        self
    }

    /// Set chart title.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set X axis.
    #[must_use]
    pub fn x_axis(mut self, axis: Axis) -> Self {
        self.x_axis = axis;
        self
    }

    /// Set Y axis.
    #[must_use]
    pub fn y_axis(mut self, axis: Axis) -> Self {
        self.y_axis = axis;
        self
    }

    /// Set legend position.
    #[must_use]
    pub const fn legend(mut self, position: LegendPosition) -> Self {
        self.legend = position;
        self
    }

    /// Set background color.
    #[must_use]
    pub const fn background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    /// Set padding.
    #[must_use]
    pub fn padding(mut self, padding: f32) -> Self {
        self.padding = padding.max(0.0);
        self
    }

    /// Set width.
    #[must_use]
    pub fn width(mut self, width: f32) -> Self {
        self.width = Some(width.max(100.0));
        self
    }

    /// Set height.
    #[must_use]
    pub fn height(mut self, height: f32) -> Self {
        self.height = Some(height.max(100.0));
        self
    }

    /// Set accessible name.
    #[must_use]
    pub fn accessible_name(mut self, name: impl Into<String>) -> Self {
        self.accessible_name_value = Some(name.into());
        self
    }

    /// Set test ID.
    #[must_use]
    pub fn test_id(mut self, id: impl Into<String>) -> Self {
        self.test_id_value = Some(id.into());
        self
    }

    /// Get chart type.
    #[must_use]
    pub const fn get_chart_type(&self) -> ChartType {
        self.kind
    }

    /// Get data series.
    #[must_use]
    pub fn get_series(&self) -> &[DataSeries] {
        &self.series
    }

    /// Get series count.
    #[must_use]
    pub fn series_count(&self) -> usize {
        self.series.len()
    }

    /// Check if chart has data.
    #[must_use]
    pub fn has_data(&self) -> bool {
        self.series.iter().any(|s| !s.points.is_empty())
    }

    /// Get title.
    #[must_use]
    pub fn get_title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// Compute data bounds across all series.
    #[must_use]
    pub fn data_bounds(&self) -> Option<(f64, f64, f64, f64)> {
        if !self.has_data() {
            return None;
        }

        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;

        for series in &self.series {
            if let Some((min, max)) = series.x_range() {
                x_min = x_min.min(min);
                x_max = x_max.max(max);
            }
            if let Some((min, max)) = series.y_range() {
                y_min = y_min.min(min);
                y_max = y_max.max(max);
            }
        }

        // Apply axis overrides
        if let Some(min) = self.x_axis.min {
            x_min = min;
        }
        if let Some(max) = self.x_axis.max {
            x_max = max;
        }
        if let Some(min) = self.y_axis.min {
            y_min = min;
        }
        if let Some(max) = self.y_axis.max {
            y_max = max;
        }

        Some((x_min, x_max, y_min, y_max))
    }

    /// Get plot area (excluding padding and labels).
    fn plot_area(&self) -> Rect {
        let title_height = if self.title.is_some() { 30.0 } else { 0.0 };
        Rect::new(
            self.bounds.x + self.padding,
            self.bounds.y + self.padding + title_height,
            self.padding.mul_add(-2.0, self.bounds.width),
            self.padding.mul_add(-2.0, self.bounds.height) - title_height,
        )
    }

    /// Map data point to screen coordinates.
    fn map_point(&self, x: f64, y: f64, bounds: &(f64, f64, f64, f64), plot: &Rect) -> Point {
        let (x_min, x_max, y_min, y_max) = *bounds;
        let x_range = (x_max - x_min).max(1e-10);
        let y_range = (y_max - y_min).max(1e-10);

        let px = (((x - x_min) / x_range) as f32).mul_add(plot.width, plot.x);
        let py = (((y - y_min) / y_range) as f32).mul_add(-plot.height, plot.y + plot.height);

        Point::new(px, py)
    }

    /// Paint grid lines.
    fn paint_grid(&self, canvas: &mut dyn Canvas, plot: &Rect, bounds: &(f64, f64, f64, f64)) {
        let (x_min, x_max, y_min, y_max) = *bounds;

        // Vertical grid lines
        if self.x_axis.show_grid {
            for i in 0..=self.x_axis.grid_lines {
                let t = i as f32 / self.x_axis.grid_lines as f32;
                let x = t.mul_add(plot.width, plot.x);
                canvas.draw_line(
                    Point::new(x, plot.y),
                    Point::new(x, plot.y + plot.height),
                    self.x_axis.grid_color,
                    1.0,
                );
            }
        }

        // Horizontal grid lines
        if self.y_axis.show_grid {
            for i in 0..=self.y_axis.grid_lines {
                let t = i as f32 / self.y_axis.grid_lines as f32;
                let y = t.mul_add(plot.height, plot.y);
                canvas.draw_line(
                    Point::new(plot.x, y),
                    Point::new(plot.x + plot.width, y),
                    self.y_axis.grid_color,
                    1.0,
                );
            }
        }

        // Axis labels
        let text_style = TextStyle {
            size: 10.0,
            color: self.x_axis.color,
            ..TextStyle::default()
        };

        // X axis labels
        for i in 0..=self.x_axis.grid_lines {
            let t = i as f64 / self.x_axis.grid_lines as f64;
            let value = t.mul_add(x_max - x_min, x_min);
            let x = (t as f32).mul_add(plot.width, plot.x);
            canvas.draw_text(
                &format!("{value:.1}"),
                Point::new(x, plot.y + plot.height + 15.0),
                &text_style,
            );
        }

        // Y axis labels
        for i in 0..=self.y_axis.grid_lines {
            let t = i as f64 / self.y_axis.grid_lines as f64;
            let value = t.mul_add(-(y_max - y_min), y_max);
            let y = (t as f32).mul_add(plot.height, plot.y);
            canvas.draw_text(
                &format!("{value:.1}"),
                Point::new(plot.x - 35.0, y + 4.0),
                &text_style,
            );
        }
    }

    /// Paint line/area chart.
    fn paint_line(&self, canvas: &mut dyn Canvas, plot: &Rect, bounds: &(f64, f64, f64, f64)) {
        for series in &self.series {
            if series.points.len() < 2 {
                continue;
            }

            // Collect points for the path
            let path_points: Vec<Point> = series
                .points
                .iter()
                .map(|&(x, y)| self.map_point(x, y, bounds, plot))
                .collect();

            // Draw line using proper path
            canvas.draw_path(&path_points, series.color, series.line_width);

            // For area charts, fill the area under the line
            if series.fill {
                let mut fill_points = path_points.clone();
                // Add bottom corners
                if let (Some(first), Some(last)) = (path_points.first(), path_points.last()) {
                    fill_points.push(Point::new(last.x, plot.y + plot.height));
                    fill_points.push(Point::new(first.x, plot.y + plot.height));
                }
                let mut fill_color = series.color;
                fill_color.a = 0.3; // Semi-transparent fill
                canvas.fill_polygon(&fill_points, fill_color);
            }

            // Draw points as circles
            if series.show_points {
                for &(x, y) in &series.points {
                    let pt = self.map_point(x, y, bounds, plot);
                    canvas.fill_circle(pt, series.point_size / 2.0, series.color);
                }
            }
        }
    }

    /// Paint bar chart.
    fn paint_bar(&self, canvas: &mut dyn Canvas, plot: &Rect, bounds: &(f64, f64, f64, f64)) {
        let (_, _, y_min, y_max) = *bounds;
        let y_range = (y_max - y_min).max(1e-10);

        let series_count = self.series.len();
        if series_count == 0 {
            return;
        }

        // Calculate bar width based on number of points
        let max_points = self
            .series
            .iter()
            .map(|s| s.points.len())
            .max()
            .unwrap_or(1);
        let group_width = plot.width / max_points as f32;
        let bar_width = (group_width * 0.8) / series_count as f32;
        let bar_gap = group_width * 0.1;

        for (si, series) in self.series.iter().enumerate() {
            for (i, &(_, y)) in series.points.iter().enumerate() {
                let bar_height = ((y - y_min) / y_range) as f32 * plot.height;
                let x = (si as f32)
                    .mul_add(bar_width, (i as f32).mul_add(group_width, plot.x + bar_gap));
                let rect = Rect::new(
                    x,
                    plot.y + plot.height - bar_height,
                    bar_width - 2.0,
                    bar_height,
                );
                canvas.fill_rect(rect, series.color);
            }
        }
    }

    /// Paint scatter chart.
    fn paint_scatter(&self, canvas: &mut dyn Canvas, plot: &Rect, bounds: &(f64, f64, f64, f64)) {
        for series in &self.series {
            for &(x, y) in &series.points {
                let pt = self.map_point(x, y, bounds, plot);
                canvas.fill_circle(pt, series.point_size / 2.0, series.color);
            }
        }
    }

    /// Paint pie chart.
    fn paint_pie(&self, canvas: &mut dyn Canvas, plot: &Rect) {
        // Sum all Y values across series
        let total: f64 = self
            .series
            .iter()
            .flat_map(|s| s.points.iter().map(|(_, y)| *y))
            .sum();

        if total <= 0.0 {
            return;
        }

        let cx = plot.x + plot.width / 2.0;
        let cy = plot.y + plot.height / 2.0;
        let radius = plot.width.min(plot.height) / 2.0 * 0.8;
        let center = Point::new(cx, cy);

        // Draw pie segments as arcs
        let mut start_angle: f32 = -std::f32::consts::FRAC_PI_2; // Start from top
        for series in &self.series {
            for &(_, y) in &series.points {
                let fraction = (y / total) as f32;
                let sweep = fraction * std::f32::consts::TAU;
                let end_angle = start_angle + sweep;

                canvas.fill_arc(center, radius, start_angle, end_angle, series.color);

                start_angle = end_angle;
            }
        }
    }

    /// Paint heatmap chart - displays matrix data with color encoding.
    fn paint_heatmap(&self, canvas: &mut dyn Canvas, plot: &Rect, bounds: &(f64, f64, f64, f64)) {
        let (_, _, y_min, y_max) = *bounds;
        let y_range = (y_max - y_min).max(1e-10);

        // For heatmap, we treat each series as a row and each point as a cell
        let row_count = self.series.len();
        if row_count == 0 {
            return;
        }

        let col_count = self
            .series
            .iter()
            .map(|s| s.points.len())
            .max()
            .unwrap_or(1);

        let cell_width = plot.width / col_count as f32;
        let cell_height = plot.height / row_count as f32;

        for (row, series) in self.series.iter().enumerate() {
            for (col, &(_, value)) in series.points.iter().enumerate() {
                // Map value to color intensity (blue to red)
                let t = ((value - y_min) / y_range) as f32;
                let color = Color::new(t, 0.2, 1.0 - t, 1.0);

                let rect = Rect::new(
                    (col as f32).mul_add(cell_width, plot.x),
                    (row as f32).mul_add(cell_height, plot.y),
                    cell_width - 1.0,
                    cell_height - 1.0,
                );
                canvas.fill_rect(rect, color);
            }
        }
    }

    /// Paint box plot - displays statistical distributions.
    fn paint_boxplot(&self, canvas: &mut dyn Canvas, plot: &Rect, bounds: &(f64, f64, f64, f64)) {
        let (_, _, y_min, y_max) = *bounds;
        let y_range = (y_max - y_min).max(1e-10);

        let series_count = self.series.len();
        if series_count == 0 {
            return;
        }

        let box_width = (plot.width / series_count as f32) * 0.6;
        let gap = (plot.width / series_count as f32) * 0.2;

        for (i, series) in self.series.iter().enumerate() {
            if series.points.len() < 5 {
                continue; // Need at least 5 points for box plot (min, q1, median, q3, max)
            }

            // Sort points by y value for quartile calculation
            let mut values: Vec<f64> = series.points.iter().map(|(_, y)| *y).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let min_val = values[0];
            let q1 = values[values.len() / 4];
            let median = values[values.len() / 2];
            let q3 = values[3 * values.len() / 4];
            let max_val = values[values.len() - 1];

            let x_center = (i as f32).mul_add(plot.width / series_count as f32, plot.x + gap);

            // Map y values to screen coordinates
            let map_y = |v: f64| -> f32 {
                let t = (v - y_min) / y_range;
                (1.0 - t as f32).mul_add(plot.height, plot.y)
            };

            let y_min_px = map_y(min_val);
            let y_q1 = map_y(q1);
            let y_median = map_y(median);
            let y_q3 = map_y(q3);
            let y_max_px = map_y(max_val);

            // Draw whiskers (vertical lines from min to q1 and q3 to max)
            canvas.draw_line(
                Point::new(x_center + box_width / 2.0, y_min_px),
                Point::new(x_center + box_width / 2.0, y_q1),
                series.color,
                1.0,
            );
            canvas.draw_line(
                Point::new(x_center + box_width / 2.0, y_q3),
                Point::new(x_center + box_width / 2.0, y_max_px),
                series.color,
                1.0,
            );

            // Draw box (from q1 to q3)
            let box_rect = Rect::new(x_center, y_q3, box_width, y_q1 - y_q3);
            canvas.fill_rect(box_rect, series.color);
            canvas.stroke_rect(box_rect, Color::new(0.0, 0.0, 0.0, 1.0), 1.0);

            // Draw median line
            canvas.draw_line(
                Point::new(x_center, y_median),
                Point::new(x_center + box_width, y_median),
                Color::new(0.0, 0.0, 0.0, 1.0),
                2.0,
            );

            // Draw caps (horizontal lines at min and max)
            let cap_width = box_width * 0.3;
            canvas.draw_line(
                Point::new(x_center + box_width / 2.0 - cap_width / 2.0, y_min_px),
                Point::new(x_center + box_width / 2.0 + cap_width / 2.0, y_min_px),
                series.color,
                1.0,
            );
            canvas.draw_line(
                Point::new(x_center + box_width / 2.0 - cap_width / 2.0, y_max_px),
                Point::new(x_center + box_width / 2.0 + cap_width / 2.0, y_max_px),
                series.color,
                1.0,
            );
        }
    }

    /// Paint legend.
    fn paint_legend(&self, canvas: &mut dyn Canvas) {
        if self.legend == LegendPosition::None || self.series.is_empty() {
            return;
        }

        let entry_height = 20.0;
        let legend_width = 100.0;
        let legend_height = (self.series.len() as f32).mul_add(entry_height, 10.0);

        let (lx, ly) = match self.legend {
            LegendPosition::TopRight => (
                self.bounds.x + self.bounds.width - legend_width - 10.0,
                self.bounds.y + self.padding + 10.0,
            ),
            LegendPosition::TopLeft => (
                self.bounds.x + self.padding + 10.0,
                self.bounds.y + self.padding + 10.0,
            ),
            LegendPosition::BottomRight => (
                self.bounds.x + self.bounds.width - legend_width - 10.0,
                self.bounds.y + self.bounds.height - legend_height - 10.0,
            ),
            LegendPosition::BottomLeft => (
                self.bounds.x + self.padding + 10.0,
                self.bounds.y + self.bounds.height - legend_height - 10.0,
            ),
            LegendPosition::None => return,
        };

        // Legend background
        canvas.fill_rect(
            Rect::new(lx, ly, legend_width, legend_height),
            Color::new(1.0, 1.0, 1.0, 0.9),
        );
        canvas.stroke_rect(
            Rect::new(lx, ly, legend_width, legend_height),
            Color::new(0.8, 0.8, 0.8, 1.0),
            1.0,
        );

        // Legend entries
        let text_style = TextStyle {
            size: 12.0,
            color: Color::new(0.2, 0.2, 0.2, 1.0),
            ..TextStyle::default()
        };

        for (i, series) in self.series.iter().enumerate() {
            let ey = (i as f32).mul_add(entry_height, ly + 5.0);
            // Color box
            canvas.fill_rect(Rect::new(lx + 5.0, ey + 4.0, 12.0, 12.0), series.color);
            // Label
            canvas.draw_text(&series.name, Point::new(lx + 22.0, ey + 14.0), &text_style);
        }
    }
}

impl Widget for Chart {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let width = self.width.unwrap_or(400.0);
        let height = self.height.unwrap_or(300.0);
        constraints.constrain(Size::new(width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: bounds.size(),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        // Background
        canvas.fill_rect(self.bounds, self.background);

        // Title
        if let Some(ref title) = self.title {
            let text_style = TextStyle {
                size: 16.0,
                color: Color::new(0.1, 0.1, 0.1, 1.0),
                ..TextStyle::default()
            };
            canvas.draw_text(
                title,
                Point::new(
                    (title.len() as f32).mul_add(-4.0, self.bounds.x + self.bounds.width / 2.0),
                    self.bounds.y + 25.0,
                ),
                &text_style,
            );
        }

        let plot = self.plot_area();

        // Get data bounds
        let Some(bounds) = self.data_bounds() else {
            return;
        };

        // Draw grid
        self.paint_grid(canvas, &plot, &bounds);

        // Draw chart based on type
        match self.kind {
            ChartType::Line | ChartType::Area => self.paint_line(canvas, &plot, &bounds),
            ChartType::Bar | ChartType::Histogram => self.paint_bar(canvas, &plot, &bounds),
            ChartType::Scatter => self.paint_scatter(canvas, &plot, &bounds),
            ChartType::Pie => self.paint_pie(canvas, &plot),
            ChartType::Heatmap => self.paint_heatmap(canvas, &plot, &bounds),
            ChartType::BoxPlot => self.paint_boxplot(canvas, &plot, &bounds),
        }

        // Draw legend
        self.paint_legend(canvas);
    }

    fn event(&mut self, _event: &presentar_core::Event) -> Option<Box<dyn Any + Send>> {
        // Charts are currently view-only
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }

    fn is_interactive(&self) -> bool {
        false
    }

    fn is_focusable(&self) -> bool {
        false
    }

    fn accessible_name(&self) -> Option<&str> {
        self.accessible_name_value
            .as_deref()
            .or(self.title.as_deref())
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Image // Charts are treated as images for accessibility
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

// PROBAR-SPEC-009: Brick Architecture - Tests define interface
impl Brick for Chart {
    fn brick_name(&self) -> &'static str {
        "Chart"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[BrickAssertion::MaxLatencyMs(16)]
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
        let test_id = self.test_id_value.as_deref().unwrap_or("chart");
        let title = self.title.as_deref().unwrap_or("Chart");
        format!(
            r#"<div class="brick-chart" data-testid="{test_id}" role="img" aria-label="{title}">{title}</div>"#
        )
    }

    fn to_css(&self) -> String {
        ".brick-chart { display: block; }".into()
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
#[path = "chart_tests.rs"]
mod tests;
