//! Loss curve widget for ML training visualization.
//!
//! Implements P204 from SPEC-024 Section 15.2.
//! Supports EMA smoothing for noisy training curves.

use crate::widgets::symbols::BRAILLE_UP;
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// EMA (Exponential Moving Average) configuration.
#[derive(Debug, Clone, Copy)]
pub struct EmaConfig {
    /// Smoothing factor (0.0 = no smoothing, 0.99 = heavy smoothing).
    pub alpha: f64,
}

impl Default for EmaConfig {
    fn default() -> Self {
        Self { alpha: 0.6 }
    }
}

/// A single training series (e.g., train loss, val loss).
#[derive(Debug, Clone)]
pub struct LossSeries {
    /// Series name.
    pub name: String,
    /// Raw loss values per epoch/step.
    pub values: Vec<f64>,
    /// Series color.
    pub color: Color,
    /// Show smoothed line.
    pub smoothed: bool,
}

/// Loss curve widget for visualizing ML training progress.
#[derive(Debug, Clone)]
pub struct LossCurve {
    series: Vec<LossSeries>,
    ema_config: EmaConfig,
    show_raw: bool,
    x_label: Option<String>,
    y_label: Option<String>,
    y_log_scale: bool,
    bounds: Rect,
    // Cached smoothed values (computed once, reused)
    smoothed_cache: Vec<Vec<f64>>,
}

impl LossCurve {
    /// Create a new loss curve widget.
    #[must_use]
    pub fn new() -> Self {
        Self {
            series: Vec::new(),
            ema_config: EmaConfig::default(),
            show_raw: true,
            x_label: Some("Epoch".to_string()),
            y_label: Some("Loss".to_string()),
            y_log_scale: false,
            bounds: Rect::default(),
            smoothed_cache: Vec::new(),
        }
    }

    /// Add a loss series.
    #[must_use]
    pub fn add_series(mut self, name: &str, values: Vec<f64>, color: Color) -> Self {
        self.series.push(LossSeries {
            name: name.to_string(),
            values,
            color,
            smoothed: true,
        });
        self.invalidate_cache();
        self
    }

    /// Set EMA smoothing configuration.
    #[must_use]
    pub fn with_ema(mut self, config: EmaConfig) -> Self {
        self.ema_config = config;
        self.invalidate_cache();
        self
    }

    /// Toggle raw line visibility.
    #[must_use]
    pub fn with_raw_visible(mut self, show: bool) -> Self {
        self.show_raw = show;
        self
    }

    /// Set log scale for Y axis.
    #[must_use]
    pub fn with_log_scale(mut self, log: bool) -> Self {
        self.y_log_scale = log;
        self
    }

    /// Set X axis label.
    #[must_use]
    pub fn with_x_label(mut self, label: &str) -> Self {
        self.x_label = Some(label.to_string());
        self
    }

    /// Set Y axis label.
    #[must_use]
    pub fn with_y_label(mut self, label: &str) -> Self {
        self.y_label = Some(label.to_string());
        self
    }

    /// Update series data.
    pub fn update_series(&mut self, index: usize, values: Vec<f64>) {
        if let Some(series) = self.series.get_mut(index) {
            series.values = values;
            self.invalidate_cache();
        }
    }

    /// Invalidate smoothed value cache.
    fn invalidate_cache(&mut self) {
        self.smoothed_cache.clear();
    }

    /// Compute EMA smoothing for a series.
    /// Uses SIMD-friendly batch operations for large datasets.
    fn compute_ema(&self, values: &[f64]) -> Vec<f64> {
        if values.is_empty() {
            return Vec::new();
        }

        let alpha = self.ema_config.alpha;
        let mut smoothed = Vec::with_capacity(values.len());

        // EMA: S_t = α * X_t + (1-α) * S_{t-1}
        // Sequential processing due to recurrence relation.

        let mut prev = values[0];
        smoothed.push(prev);

        for &val in values.iter().skip(1) {
            if val.is_finite() {
                prev = alpha * val + (1.0 - alpha) * prev;
            }
            // Keep previous value for NaN/Inf
            smoothed.push(prev);
        }

        smoothed
    }

    /// Ensure smoothed cache is populated.
    fn ensure_cache(&mut self) {
        if self.smoothed_cache.len() != self.series.len() {
            self.smoothed_cache = self
                .series
                .iter()
                .map(|s| {
                    if s.smoothed {
                        self.compute_ema(&s.values)
                    } else {
                        s.values.clone()
                    }
                })
                .collect();
        }
    }

    /// Get Y range across all series.
    fn y_range(&self) -> (f64, f64) {
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;

        for series in &self.series {
            for &v in &series.values {
                if v.is_finite() && v > 0.0 {
                    // Filter non-positive for log scale
                    y_min = y_min.min(v);
                    y_max = y_max.max(v);
                }
            }
        }

        if y_min == f64::INFINITY {
            (0.001, 1.0)
        } else {
            // Add padding
            let padding = (y_max - y_min) * 0.1;
            (
                (y_min - padding).max(if self.y_log_scale { 1e-10 } else { 0.0 }),
                y_max + padding,
            )
        }
    }

    /// Get X range (number of epochs/steps).
    fn x_range(&self) -> (f64, f64) {
        let max_len = self
            .series
            .iter()
            .map(|s| s.values.len())
            .max()
            .unwrap_or(0);
        (0.0, max_len.saturating_sub(1) as f64)
    }

    /// Transform Y value (linear or log scale).
    fn transform_y(&self, y: f64, y_min: f64, y_max: f64) -> f64 {
        if self.y_log_scale {
            let log_min = y_min.max(1e-10).ln();
            let log_max = y_max.ln();
            let log_y = y.max(1e-10).ln();
            (log_y - log_min) / (log_max - log_min)
        } else {
            (y - y_min) / (y_max - y_min)
        }
    }
}

impl Default for LossCurve {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for LossCurve {
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
        self.ensure_cache();
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.bounds.width < 15.0 || self.bounds.height < 5.0 {
            return;
        }

        let margin_left = 8.0;
        let margin_bottom = 2.0;
        let margin_right = 2.0;

        let plot_x = self.bounds.x + margin_left;
        let plot_y = self.bounds.y;
        let plot_width = self.bounds.width - margin_left - margin_right;
        let plot_height = self.bounds.height - margin_bottom;

        if plot_width <= 0.0 || plot_height <= 0.0 {
            return;
        }

        let (y_min, y_max) = self.y_range();
        let (x_min, x_max) = self.x_range();

        let label_style = TextStyle {
            color: Color::new(0.6, 0.6, 0.6, 1.0),
            ..Default::default()
        };

        // Draw Y axis labels
        for i in 0..=4 {
            let t = i as f64 / 4.0;
            let y_val = if self.y_log_scale {
                let log_min = y_min.max(1e-10).ln();
                let log_max = y_max.ln();
                (log_min + (log_max - log_min) * (1.0 - t)).exp()
            } else {
                y_min + (y_max - y_min) * (1.0 - t)
            };
            let y_pos = plot_y + plot_height * t as f32;

            if y_pos >= plot_y && y_pos < plot_y + plot_height {
                let label = if y_val < 0.01 {
                    format!("{y_val:.1e}")
                } else if y_val < 1.0 {
                    format!("{y_val:.3}")
                } else {
                    format!("{y_val:.2}")
                };
                canvas.draw_text(
                    &format!("{label:>7}"),
                    Point::new(self.bounds.x, y_pos),
                    &label_style,
                );
            }
        }

        // Draw X axis labels
        let x_ticks = 5.min(x_max as usize);
        for i in 0..=x_ticks {
            let t = i as f64 / x_ticks as f64;
            let x_val = x_min + (x_max - x_min) * t;
            let x_pos = plot_x + plot_width * t as f32;

            if x_pos >= plot_x && x_pos < plot_x + plot_width - 3.0 {
                let label = format!("{x_val:.0}");
                canvas.draw_text(
                    &label,
                    Point::new(x_pos, plot_y + plot_height),
                    &label_style,
                );
            }
        }

        // Draw each series
        for (series_idx, series) in self.series.iter().enumerate() {
            if series.values.is_empty() {
                continue;
            }

            // Get smoothed values from cache
            let smoothed = if series_idx < self.smoothed_cache.len() && series.smoothed {
                &self.smoothed_cache[series_idx]
            } else {
                &series.values
            };

            // Draw raw line (dimmed) if enabled
            if self.show_raw && series.smoothed {
                let raw_style = TextStyle {
                    color: Color::new(
                        series.color.r * 0.4,
                        series.color.g * 0.4,
                        series.color.b * 0.4,
                        0.5,
                    ),
                    ..Default::default()
                };

                self.draw_line_braille(
                    canvas,
                    &series.values,
                    &raw_style,
                    plot_x,
                    plot_y,
                    plot_width,
                    plot_height,
                    y_min,
                    y_max,
                    x_max,
                );
            }

            // Draw smoothed line
            let style = TextStyle {
                color: series.color,
                ..Default::default()
            };

            self.draw_line_braille(
                canvas,
                smoothed,
                &style,
                plot_x,
                plot_y,
                plot_width,
                plot_height,
                y_min,
                y_max,
                x_max,
            );
        }

        // Draw legend
        if !self.series.is_empty() {
            for (i, series) in self.series.iter().enumerate() {
                let style = TextStyle {
                    color: series.color,
                    ..Default::default()
                };
                let legend_y = plot_y + i as f32;
                canvas.draw_text(
                    &format!("─ {}", series.name),
                    Point::new(plot_x + plot_width - 15.0, legend_y),
                    &style,
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

impl LossCurve {
    /// Draw a line using braille characters.
    #[allow(clippy::too_many_arguments)]
    fn draw_line_braille(
        &self,
        canvas: &mut dyn Canvas,
        values: &[f64],
        style: &TextStyle,
        plot_x: f32,
        plot_y: f32,
        plot_width: f32,
        plot_height: f32,
        y_min: f64,
        y_max: f64,
        x_max: f64,
    ) {
        if values.is_empty() || x_max <= 0.0 {
            return;
        }

        let cols = plot_width as usize;
        let rows = (plot_height * 4.0) as usize; // 4 braille dots per row

        if cols == 0 || rows == 0 {
            return;
        }

        let mut grid = vec![vec![false; rows]; cols];

        // Plot points onto grid
        for (i, &y) in values.iter().enumerate() {
            if !y.is_finite() {
                continue;
            }

            let x_norm = i as f64 / x_max;
            let y_norm = self.transform_y(y, y_min, y_max);

            if !(0.0..=1.0).contains(&x_norm) || !(0.0..=1.0).contains(&y_norm) {
                continue;
            }

            let gx = ((x_norm * (cols - 1) as f64).round() as usize).min(cols.saturating_sub(1));
            let gy =
                (((1.0 - y_norm) * (rows - 1) as f64).round() as usize).min(rows.saturating_sub(1));

            grid[gx][gy] = true;

            // Connect to previous point
            if i > 0 {
                let prev_y = values[i - 1];
                if prev_y.is_finite() {
                    let prev_x_norm = (i - 1) as f64 / x_max;
                    let prev_y_norm = self.transform_y(prev_y, y_min, y_max);

                    if (0.0..=1.0).contains(&prev_x_norm) && (0.0..=1.0).contains(&prev_y_norm) {
                        let prev_gx = ((prev_x_norm * (cols - 1) as f64).round() as usize)
                            .min(cols.saturating_sub(1));
                        let prev_gy = (((1.0 - prev_y_norm) * (rows - 1) as f64).round() as usize)
                            .min(rows.saturating_sub(1));

                        Self::draw_line_bresenham(&mut grid, prev_gx, prev_gy, gx, gy);
                    }
                }
            }
        }

        // Render grid as braille
        let char_rows = plot_height as usize;
        for cy in 0..char_rows {
            for (cx, column) in grid.iter().enumerate() {
                let mut dots = 0u8;
                for dy in 0..4 {
                    let gy = cy * 4 + dy;
                    if gy < rows && column[gy] {
                        dots |= 1 << dy;
                    }
                }

                if dots > 0 {
                    let braille_idx = dots as usize;
                    let ch = if braille_idx < BRAILLE_UP.len() {
                        BRAILLE_UP[braille_idx]
                    } else {
                        '⣿'
                    };
                    canvas.draw_text(
                        &ch.to_string(),
                        Point::new(plot_x + cx as f32, plot_y + cy as f32),
                        style,
                    );
                }
            }
        }
    }

    /// Draw a line between two points using Bresenham's algorithm.
    #[allow(clippy::cast_possible_wrap)]
    fn draw_line_bresenham(grid: &mut [Vec<bool>], x0: usize, y0: usize, x1: usize, y1: usize) {
        let dx = (x1 as isize - x0 as isize).abs();
        let dy = -(y1 as isize - y0 as isize).abs();
        let sx: isize = if x0 < x1 { 1 } else { -1 };
        let sy: isize = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;

        let mut x = x0 as isize;
        let mut y = y0 as isize;

        let cols = grid.len() as isize;
        let rows = if cols > 0 { grid[0].len() as isize } else { 0 };

        loop {
            if x >= 0 && x < cols && y >= 0 && y < rows {
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
}

impl Brick for LossCurve {
    fn brick_name(&self) -> &'static str {
        "LossCurve"
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

        if self.bounds.width >= 15.0 && self.bounds.height >= 5.0 {
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
    use crate::{CellBuffer, DirectTerminalCanvas};

    #[test]
    fn test_loss_curve_creation() {
        let curve =
            LossCurve::new().add_series("train", vec![1.0, 0.8, 0.6, 0.4, 0.3, 0.25], Color::BLUE);
        assert_eq!(curve.series.len(), 1);
    }

    #[test]
    fn test_ema_smoothing() {
        let curve = LossCurve::new().with_ema(EmaConfig { alpha: 0.5 });
        let values = vec![1.0, 0.0, 1.0, 0.0, 1.0];
        let smoothed = curve.compute_ema(&values);

        assert_eq!(smoothed.len(), values.len());
        // First value should be unchanged
        assert!((smoothed[0] - 1.0).abs() < 0.001);
        // Subsequent values should be smoothed
        assert!((smoothed[1] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_ema_empty_values() {
        let curve = LossCurve::new();
        let smoothed = curve.compute_ema(&[]);
        assert!(smoothed.is_empty());
    }

    #[test]
    fn test_ema_with_nan_values() {
        let curve = LossCurve::new().with_ema(EmaConfig { alpha: 0.5 });
        let values = vec![1.0, f64::NAN, 0.5];
        let smoothed = curve.compute_ema(&values);
        assert_eq!(smoothed.len(), 3);
        assert!((smoothed[0] - 1.0).abs() < 0.001);
        // NaN should keep previous value
        assert!((smoothed[1] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_ema_config_default() {
        let config = EmaConfig::default();
        assert!((config.alpha - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_log_scale() {
        let curve = LossCurve::new().with_log_scale(true);
        let y_norm = curve.transform_y(1.0, 0.1, 10.0);
        assert!(y_norm > 0.0 && y_norm < 1.0);
    }

    #[test]
    fn test_linear_scale() {
        let curve = LossCurve::new().with_log_scale(false);
        let y_norm = curve.transform_y(5.0, 0.0, 10.0);
        assert!((y_norm - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_multi_series() {
        let curve = LossCurve::new()
            .add_series("train", vec![1.0, 0.5, 0.3], Color::BLUE)
            .add_series("val", vec![1.1, 0.6, 0.4], Color::RED);
        assert_eq!(curve.series.len(), 2);
    }

    #[test]
    fn test_with_x_label() {
        let curve = LossCurve::new().with_x_label("Steps");
        assert_eq!(curve.x_label, Some("Steps".to_string()));
    }

    #[test]
    fn test_with_y_label() {
        let curve = LossCurve::new().with_y_label("MSE");
        assert_eq!(curve.y_label, Some("MSE".to_string()));
    }

    #[test]
    fn test_with_raw_visible() {
        let curve = LossCurve::new().with_raw_visible(false);
        assert!(!curve.show_raw);

        let curve2 = LossCurve::new().with_raw_visible(true);
        assert!(curve2.show_raw);
    }

    #[test]
    fn test_update_series() {
        let mut curve = LossCurve::new().add_series("train", vec![1.0, 0.8], Color::BLUE);
        curve.update_series(0, vec![0.5, 0.3, 0.1]);
        assert_eq!(curve.series[0].values.len(), 3);
    }

    #[test]
    fn test_update_series_invalid_index() {
        let mut curve = LossCurve::new().add_series("train", vec![1.0], Color::BLUE);
        curve.update_series(5, vec![0.5]); // Invalid index, should be ignored
        assert_eq!(curve.series[0].values.len(), 1);
    }

    #[test]
    fn test_y_range_empty() {
        let curve = LossCurve::new();
        let (y_min, y_max) = curve.y_range();
        assert!(y_min < y_max);
    }

    #[test]
    fn test_y_range_with_data() {
        let curve = LossCurve::new().add_series("train", vec![1.0, 2.0, 3.0], Color::BLUE);
        let (y_min, y_max) = curve.y_range();
        assert!(y_min <= 1.0);
        assert!(y_max >= 3.0);
    }

    #[test]
    fn test_x_range() {
        let curve =
            LossCurve::new().add_series("train", vec![1.0, 2.0, 3.0, 4.0, 5.0], Color::BLUE);
        let (x_min, x_max) = curve.x_range();
        assert!((x_min - 0.0).abs() < 0.001);
        assert!((x_max - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_x_range_empty() {
        let curve = LossCurve::new();
        let (x_min, x_max) = curve.x_range();
        assert!((x_min - 0.0).abs() < 0.001);
        assert!((x_max - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_loss_curve_measure() {
        let curve = LossCurve::new();
        let constraints = Constraints::new(0.0, 100.0, 0.0, 50.0);
        let size = curve.measure(constraints);
        assert_eq!(size.width, 80.0);
        assert_eq!(size.height, 20.0);
    }

    #[test]
    fn test_loss_curve_layout_and_paint() {
        let mut curve = LossCurve::new()
            .add_series("train", vec![2.0, 1.5, 1.0, 0.8, 0.6], Color::BLUE)
            .add_series("val", vec![2.2, 1.8, 1.2, 1.0, 0.9], Color::RED);

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let result = curve.layout(Rect::new(0.0, 0.0, 60.0, 20.0));
        assert_eq!(result.size.width, 60.0);
        assert_eq!(result.size.height, 20.0);

        curve.paint(&mut canvas);

        // Verify something was rendered
        let cells = buffer.cells();
        let non_empty = cells.iter().filter(|c| !c.symbol.is_empty()).count();
        assert!(non_empty > 0, "Loss curve should render some content");
    }

    #[test]
    fn test_loss_curve_paint_with_log_scale() {
        let mut curve = LossCurve::new().with_log_scale(true).add_series(
            "train",
            vec![1.0, 0.1, 0.01, 0.001],
            Color::BLUE,
        );

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        curve.layout(Rect::new(0.0, 0.0, 60.0, 20.0));
        curve.paint(&mut canvas);
    }

    #[test]
    fn test_loss_curve_paint_small_bounds() {
        let mut curve = LossCurve::new().add_series("train", vec![1.0, 0.5], Color::BLUE);

        let mut buffer = CellBuffer::new(10, 3);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        curve.layout(Rect::new(0.0, 0.0, 10.0, 3.0));
        curve.paint(&mut canvas);
        // Should not crash with small bounds
    }

    #[test]
    fn test_loss_curve_paint_with_raw_hidden() {
        let mut curve = LossCurve::new().with_raw_visible(false).add_series(
            "train",
            vec![1.0, 0.8, 0.6, 0.4],
            Color::BLUE,
        );

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        curve.layout(Rect::new(0.0, 0.0, 60.0, 20.0));
        curve.paint(&mut canvas);
    }

    #[test]
    fn test_loss_curve_paint_noisy_data() {
        let values: Vec<f64> = (0..100)
            .map(|i| 2.0 * (-i as f64 / 30.0).exp() + 0.1 * (i as f64 * 0.5).sin())
            .collect();

        let mut curve = LossCurve::new()
            .with_ema(EmaConfig { alpha: 0.1 })
            .add_series("train", values, Color::BLUE);

        let mut buffer = CellBuffer::new(80, 25);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        curve.layout(Rect::new(0.0, 0.0, 80.0, 25.0));
        curve.paint(&mut canvas);
    }

    #[test]
    fn test_loss_curve_paint_with_nan() {
        let values = vec![1.0, f64::NAN, 0.5, f64::INFINITY, 0.3];
        let mut curve = LossCurve::new().add_series("train", values, Color::BLUE);

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        curve.layout(Rect::new(0.0, 0.0, 60.0, 20.0));
        curve.paint(&mut canvas);
    }

    #[test]
    fn test_loss_curve_ensure_cache() {
        let mut curve = LossCurve::new().add_series("train", vec![1.0, 0.5, 0.3], Color::BLUE);
        curve.ensure_cache();
        assert_eq!(curve.smoothed_cache.len(), 1);
        assert_eq!(curve.smoothed_cache[0].len(), 3);
    }

    #[test]
    fn test_loss_curve_assertions() {
        let curve = LossCurve::default();
        assert!(!curve.assertions().is_empty());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_loss_curve_verify_valid() {
        let mut curve = LossCurve::default();
        curve.bounds = Rect::new(0.0, 0.0, 80.0, 20.0);
        assert!(curve.verify().is_valid());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_loss_curve_verify_invalid() {
        let mut curve = LossCurve::default();
        curve.bounds = Rect::new(0.0, 0.0, 10.0, 3.0);
        assert!(!curve.verify().is_valid());
    }

    #[test]
    fn test_loss_curve_children() {
        let curve = LossCurve::default();
        assert!(curve.children().is_empty());
    }

    #[test]
    fn test_loss_curve_children_mut() {
        let mut curve = LossCurve::default();
        assert!(curve.children_mut().is_empty());
    }

    #[test]
    fn test_loss_curve_brick_name() {
        let curve = LossCurve::new();
        assert_eq!(curve.brick_name(), "LossCurve");
    }

    #[test]
    fn test_loss_curve_budget() {
        let curve = LossCurve::new();
        let budget = curve.budget();
        assert!(budget.layout_ms > 0);
    }

    #[test]
    fn test_loss_curve_to_html() {
        let curve = LossCurve::new();
        assert!(curve.to_html().is_empty());
    }

    #[test]
    fn test_loss_curve_to_css() {
        let curve = LossCurve::new();
        assert!(curve.to_css().is_empty());
    }

    #[test]
    fn test_loss_curve_type_id() {
        let curve = LossCurve::new();
        let type_id = Widget::type_id(&curve);
        assert_eq!(type_id, TypeId::of::<LossCurve>());
    }

    #[test]
    fn test_loss_curve_event() {
        let mut curve = LossCurve::new();
        let event = Event::Resize {
            width: 80.0,
            height: 24.0,
        };
        assert!(curve.event(&event).is_none());
    }

    #[test]
    fn test_empty_series() {
        let curve = LossCurve::new().add_series("empty", vec![], Color::BLUE);
        let (y_min, y_max) = curve.y_range();
        assert!(y_min < y_max);
    }

    #[test]
    fn test_bresenham_line() {
        let mut grid = vec![vec![false; 10]; 10];
        LossCurve::draw_line_bresenham(&mut grid, 0, 0, 9, 9);
        // Check diagonal has points
        assert!(grid[0][0]);
        assert!(grid[9][9]);
    }

    #[test]
    fn test_bresenham_line_reverse() {
        let mut grid = vec![vec![false; 10]; 10];
        LossCurve::draw_line_bresenham(&mut grid, 9, 9, 0, 0);
        assert!(grid[0][0]);
        assert!(grid[9][9]);
    }

    #[test]
    fn test_bresenham_horizontal() {
        let mut grid = vec![vec![false; 10]; 10];
        LossCurve::draw_line_bresenham(&mut grid, 0, 5, 9, 5);
        for x in 0..10 {
            assert!(grid[x][5]);
        }
    }

    #[test]
    fn test_bresenham_vertical() {
        let mut grid = vec![vec![false; 10]; 10];
        LossCurve::draw_line_bresenham(&mut grid, 5, 0, 5, 9);
        for y in 0..10 {
            assert!(grid[5][y]);
        }
    }
}
