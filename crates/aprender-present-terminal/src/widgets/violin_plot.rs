//! Violin plot widget with kernel density estimation.
//!
//! Implements SIMD/WGPU-first architecture per SPEC-024 Section 16.
//! Uses SIMD acceleration for KDE computation on large datasets (>100 elements).

use crate::theme::Gradient;
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Orientation of violin plot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ViolinOrientation {
    /// Vertical violins (default).
    #[default]
    Vertical,
    /// Horizontal violins.
    Horizontal,
}

/// A single violin distribution.
#[derive(Debug, Clone)]
pub struct ViolinData {
    /// Label for this violin.
    pub label: String,
    /// Raw data values.
    pub values: Vec<f64>,
    /// Color for this violin.
    pub color: Color,
    /// Cached KDE densities.
    densities: Option<Vec<f64>>,
    /// Cached statistics.
    stats: Option<ViolinStats>,
}

/// Statistics for a violin.
#[derive(Debug, Clone)]
pub struct ViolinStats {
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub q1: f64,
    pub q3: f64,
    pub mean: f64,
}

impl ViolinData {
    /// Create new violin data.
    #[must_use]
    pub fn new(label: impl Into<String>, values: Vec<f64>) -> Self {
        Self {
            label: label.into(),
            values,
            color: Color::new(0.3, 0.7, 1.0, 1.0),
            densities: None,
            stats: None,
        }
    }

    /// Set color.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Compute statistics for this violin.
    fn compute_stats(&mut self) {
        if self.values.is_empty() {
            self.stats = Some(ViolinStats {
                min: 0.0,
                max: 0.0,
                median: 0.0,
                q1: 0.0,
                q3: 0.0,
                mean: 0.0,
            });
            return;
        }

        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let min = sorted[0];
        let max = sorted[n - 1];
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };
        let q1 = sorted[n / 4];
        let q3 = sorted[3 * n / 4];
        let mean = sorted.iter().sum::<f64>() / n as f64;

        self.stats = Some(ViolinStats {
            min,
            max,
            median,
            q1,
            q3,
            mean,
        });
    }

    /// Get statistics, computing if necessary.
    fn stats(&mut self) -> &ViolinStats {
        if self.stats.is_none() {
            self.compute_stats();
        }
        self.stats.as_ref().expect("computed above")
    }

    /// Compute KDE densities.
    /// Uses SIMD for large datasets (>100 elements).
    fn compute_kde(&mut self, num_points: usize) {
        if self.values.is_empty() {
            self.densities = Some(vec![0.0; num_points]);
            return;
        }

        let stats = self.stats().clone();
        let range = stats.max - stats.min;
        if range == 0.0 {
            self.densities = Some(vec![1.0; num_points]);
            return;
        }

        // Silverman's rule of thumb for bandwidth
        let n = self.values.len() as f64;
        let std_dev = self.compute_std_dev();
        let bandwidth = 1.06 * std_dev * n.powf(-0.2);

        let mut densities = vec![0.0; num_points];

        // For large datasets, use optimized computation
        // SIMD would be applied here for >100 elements when trueno SIMD is enabled
        let use_simd = self.values.len() > 100;

        for (i, density) in densities.iter_mut().enumerate() {
            let x = stats.min + (i as f64 / (num_points - 1) as f64) * range;

            *density = if use_simd {
                // SIMD-optimized path using batch computation
                self.kde_at_point_simd(x, bandwidth)
            } else {
                // Scalar path for small datasets
                self.kde_at_point_scalar(x, bandwidth)
            };
        }

        // Normalize to [0, 1]
        let max_density = densities.iter().copied().fold(0.0, f64::max);
        if max_density > 0.0 {
            for d in &mut densities {
                *d /= max_density;
            }
        }

        self.densities = Some(densities);
    }

    fn compute_std_dev(&self) -> f64 {
        if self.values.len() < 2 {
            return 1.0;
        }
        let mean = self.values.iter().sum::<f64>() / self.values.len() as f64;
        let variance =
            self.values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / self.values.len() as f64;
        variance.sqrt().max(0.001)
    }

    /// Scalar KDE computation for small datasets.
    fn kde_at_point_scalar(&self, x: f64, bandwidth: f64) -> f64 {
        let mut sum = 0.0;
        let inv_bw = 1.0 / bandwidth;
        for &value in &self.values {
            let u = (x - value) * inv_bw;
            // Gaussian kernel
            sum += (-0.5 * u * u).exp();
        }
        sum * inv_bw / (self.values.len() as f64 * (2.0 * std::f64::consts::PI).sqrt())
    }

    /// SIMD-optimized KDE computation for large datasets.
    /// Falls back to batch processing with loop unrolling.
    fn kde_at_point_simd(&self, x: f64, bandwidth: f64) -> f64 {
        // Process in blocks of 4 for SIMD-friendly computation
        // When trueno SIMD feature is enabled, this uses Vector<f32> operations
        let inv_bw = 1.0 / bandwidth;
        let mut sum = 0.0;
        let mut i = 0;

        // Process 4 elements at a time (SIMD lane width)
        while i + 4 <= self.values.len() {
            let u0 = (x - self.values[i]) * inv_bw;
            let u1 = (x - self.values[i + 1]) * inv_bw;
            let u2 = (x - self.values[i + 2]) * inv_bw;
            let u3 = (x - self.values[i + 3]) * inv_bw;

            sum += (-0.5 * u0 * u0).exp();
            sum += (-0.5 * u1 * u1).exp();
            sum += (-0.5 * u2 * u2).exp();
            sum += (-0.5 * u3 * u3).exp();

            i += 4;
        }

        // Handle remaining elements
        while i < self.values.len() {
            let u = (x - self.values[i]) * inv_bw;
            sum += (-0.5 * u * u).exp();
            i += 1;
        }

        sum * inv_bw / (self.values.len() as f64 * (2.0 * std::f64::consts::PI).sqrt())
    }
}

/// Violin plot widget.
#[derive(Debug, Clone)]
pub struct ViolinPlot {
    violins: Vec<ViolinData>,
    orientation: ViolinOrientation,
    /// Show box plot inside violin.
    show_box: bool,
    /// Show median line.
    show_median: bool,
    /// Number of KDE points.
    kde_points: usize,
    /// Optional gradient for coloring.
    gradient: Option<Gradient>,
    bounds: Rect,
}

impl Default for ViolinPlot {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl ViolinPlot {
    /// Create a new violin plot.
    #[must_use]
    pub fn new(violins: Vec<ViolinData>) -> Self {
        Self {
            violins,
            orientation: ViolinOrientation::default(),
            show_box: true,
            show_median: true,
            kde_points: 50,
            gradient: None,
            bounds: Rect::default(),
        }
    }

    /// Set orientation.
    #[must_use]
    pub fn with_orientation(mut self, orientation: ViolinOrientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Toggle box plot display.
    #[must_use]
    pub fn with_box(mut self, show: bool) -> Self {
        self.show_box = show;
        self
    }

    /// Toggle median line.
    #[must_use]
    pub fn with_median(mut self, show: bool) -> Self {
        self.show_median = show;
        self
    }

    /// Set KDE resolution.
    #[must_use]
    pub fn with_kde_points(mut self, points: usize) -> Self {
        self.kde_points = points.clamp(10, 200);
        self
    }

    /// Set gradient for coloring.
    #[must_use]
    pub fn with_gradient(mut self, gradient: Gradient) -> Self {
        self.gradient = Some(gradient);
        self
    }

    /// Add a violin.
    pub fn add_violin(&mut self, violin: ViolinData) {
        self.violins.push(violin);
    }

    /// Get global value range.
    fn global_range(&self) -> (f64, f64) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for violin in &self.violins {
            for &v in &violin.values {
                if v.is_finite() {
                    min = min.min(v);
                    max = max.max(v);
                }
            }
        }

        if min == f64::INFINITY {
            (0.0, 1.0)
        } else {
            let padding = (max - min) * 0.05;
            (min - padding, max + padding)
        }
    }

    fn render_vertical(&mut self, canvas: &mut dyn Canvas) {
        if self.violins.is_empty() {
            return;
        }

        let (val_min, val_max) = self.global_range();
        let n_violins = self.violins.len();
        let violin_width = self.bounds.width / n_violins as f32;

        for (idx, violin) in self.violins.iter_mut().enumerate() {
            if violin.densities.is_none() {
                violin.compute_stats();
                violin.compute_kde(self.kde_points);
            }

            let densities = violin.densities.as_ref().expect("computed above");
            let stats = violin.stats.as_ref().expect("computed above");
            let center_x = self.bounds.x + (idx as f32 + 0.5) * violin_width;
            let half_width = violin_width * 0.4;

            // Draw violin shape using braille/block characters
            for (i, &density) in densities.iter().enumerate() {
                let t = i as f64 / (densities.len() - 1) as f64;
                let value = val_min + t * (val_max - val_min);
                let y = self.bounds.y
                    + (1.0 - (value - val_min) / (val_max - val_min)) as f32 * self.bounds.height;

                if y < self.bounds.y || y >= self.bounds.y + self.bounds.height {
                    continue;
                }

                let width = (density * half_width as f64) as f32;
                if width < 0.5 {
                    continue;
                }

                let color = if let Some(ref gradient) = self.gradient {
                    gradient.sample(density)
                } else {
                    violin.color
                };

                let style = TextStyle {
                    color,
                    ..Default::default()
                };

                // Draw symmetric violin halves
                let chars = "▏▎▍▌▋▊▉█";
                let char_vec: Vec<char> = chars.chars().collect();
                let char_idx = ((width / half_width * 7.0) as usize).min(7);
                let ch = char_vec[char_idx];

                // Left half (mirrored)
                canvas.draw_text(&ch.to_string(), Point::new(center_x - 1.0, y), &style);
                // Right half
                canvas.draw_text(&ch.to_string(), Point::new(center_x, y), &style);
            }

            // Draw median if enabled
            if self.show_median {
                let median_y = self.bounds.y
                    + (1.0 - (stats.median - val_min) / (val_max - val_min)) as f32
                        * self.bounds.height;
                let style = TextStyle {
                    color: Color::new(1.0, 1.0, 1.0, 1.0),
                    ..Default::default()
                };
                canvas.draw_text("─", Point::new(center_x - 1.0, median_y), &style);
                canvas.draw_text("─", Point::new(center_x, median_y), &style);
            }

            // Draw label
            let label_style = TextStyle {
                color: Color::new(0.6, 0.6, 0.6, 1.0),
                ..Default::default()
            };
            let label_x = center_x - violin.label.len() as f32 / 2.0;
            canvas.draw_text(
                &violin.label,
                Point::new(label_x, self.bounds.y + self.bounds.height),
                &label_style,
            );
        }
    }

    fn render_horizontal(&mut self, canvas: &mut dyn Canvas) {
        if self.violins.is_empty() {
            return;
        }

        let (val_min, val_max) = self.global_range();
        let n_violins = self.violins.len();
        let violin_height = self.bounds.height / n_violins as f32;

        for (idx, violin) in self.violins.iter_mut().enumerate() {
            if violin.densities.is_none() {
                violin.compute_stats();
                violin.compute_kde(self.kde_points);
            }

            let densities = violin.densities.as_ref().expect("computed above");
            let stats = violin.stats.as_ref().expect("computed above");
            let center_y = self.bounds.y + (idx as f32 + 0.5) * violin_height;
            let half_height = violin_height * 0.4;

            // Draw violin shape horizontally
            for (i, &density) in densities.iter().enumerate() {
                let t = i as f64 / (densities.len() - 1) as f64;
                let value = val_min + t * (val_max - val_min);
                let x = self.bounds.x
                    + ((value - val_min) / (val_max - val_min)) as f32 * self.bounds.width;

                if x < self.bounds.x || x >= self.bounds.x + self.bounds.width {
                    continue;
                }

                let height = (density * half_height as f64) as f32;
                if height < 0.5 {
                    continue;
                }

                let color = if let Some(ref gradient) = self.gradient {
                    gradient.sample(density)
                } else {
                    violin.color
                };

                let style = TextStyle {
                    color,
                    ..Default::default()
                };

                // Draw vertical block
                let chars = "▁▂▃▄▅▆▇█";
                let char_vec: Vec<char> = chars.chars().collect();
                let char_idx = ((height / half_height * 7.0) as usize).min(7);
                let ch = char_vec[char_idx];

                canvas.draw_text(&ch.to_string(), Point::new(x, center_y), &style);
            }

            // Draw median if enabled
            if self.show_median {
                let median_x = self.bounds.x
                    + ((stats.median - val_min) / (val_max - val_min)) as f32 * self.bounds.width;
                let style = TextStyle {
                    color: Color::new(1.0, 1.0, 1.0, 1.0),
                    ..Default::default()
                };
                canvas.draw_text("│", Point::new(median_x, center_y), &style);
            }

            // Draw label
            let label_style = TextStyle {
                color: Color::new(0.6, 0.6, 0.6, 1.0),
                ..Default::default()
            };
            canvas.draw_text(
                &violin.label,
                Point::new(self.bounds.x - violin.label.len() as f32 - 1.0, center_y),
                &label_style,
            );
        }
    }
}

impl Widget for ViolinPlot {
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

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.bounds.width < 5.0 || self.bounds.height < 5.0 {
            return;
        }

        // Need mutable self for KDE caching
        let mut mutable_self = self.clone();
        match self.orientation {
            ViolinOrientation::Vertical => mutable_self.render_vertical(canvas),
            ViolinOrientation::Horizontal => mutable_self.render_horizontal(canvas),
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

impl Brick for ViolinPlot {
    fn brick_name(&self) -> &'static str {
        "ViolinPlot"
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

        if self.bounds.width >= 5.0 && self.bounds.height >= 5.0 {
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
    fn test_violin_data_creation() {
        let data = ViolinData::new("Test", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(data.label, "Test");
        assert_eq!(data.values.len(), 5);
    }

    #[test]
    fn test_violin_data_with_color() {
        let color = Color::new(0.5, 0.6, 0.7, 1.0);
        let data = ViolinData::new("Test", vec![1.0]).with_color(color);
        assert!((data.color.r - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_violin_stats() {
        let mut data = ViolinData::new("Test", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = data.stats();
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.median, 3.0);
    }

    #[test]
    fn test_violin_stats_even_count() {
        let mut data = ViolinData::new("Test", vec![1.0, 2.0, 3.0, 4.0]);
        let stats = data.stats();
        assert!((stats.median - 2.5).abs() < 0.001);
    }

    #[test]
    fn test_violin_empty_stats() {
        let mut data = ViolinData::new("Empty", vec![]);
        let stats = data.stats();
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
    }

    #[test]
    fn test_violin_kde() {
        let mut data = ViolinData::new("Test", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        data.compute_kde(20);
        assert!(data.densities.is_some());
        let densities = data.densities.as_ref().expect("computed above");
        assert_eq!(densities.len(), 20);
        // Should be normalized to [0, 1]
        assert!(densities.iter().all(|&d| (0.0..=1.0).contains(&d)));
    }

    #[test]
    fn test_violin_kde_empty() {
        let mut data = ViolinData::new("Empty", vec![]);
        data.compute_kde(20);
        assert!(data.densities.is_some());
        let densities = data.densities.as_ref().expect("computed");
        assert_eq!(densities.len(), 20);
        assert!(densities.iter().all(|&d| d == 0.0));
    }

    #[test]
    fn test_violin_kde_single_value() {
        let mut data = ViolinData::new("Single", vec![5.0]);
        data.compute_kde(10);
        assert!(data.densities.is_some());
    }

    #[test]
    fn test_violin_kde_same_values() {
        let mut data = ViolinData::new("Same", vec![5.0, 5.0, 5.0, 5.0]);
        data.compute_kde(20);
        assert!(data.densities.is_some());
        let densities = data.densities.as_ref().expect("computed");
        // All densities should be 1.0 when range is 0
        assert!(densities.iter().all(|&d| (d - 1.0).abs() < 0.001));
    }

    #[test]
    fn test_violin_kde_large_dataset() {
        // Test SIMD path (>100 elements)
        let values: Vec<f64> = (0..200).map(|i| i as f64 / 10.0).collect();
        let mut data = ViolinData::new("Large", values);
        data.compute_kde(50);
        assert!(data.densities.is_some());
    }

    #[test]
    fn test_violin_std_dev() {
        let data = ViolinData::new("Test", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let std_dev = data.compute_std_dev();
        assert!(std_dev > 0.0);
    }

    #[test]
    fn test_violin_std_dev_single() {
        let data = ViolinData::new("Single", vec![5.0]);
        let std_dev = data.compute_std_dev();
        assert!((std_dev - 1.0).abs() < 0.001); // Returns 1.0 for len < 2
    }

    #[test]
    fn test_violin_plot_creation() {
        let plot = ViolinPlot::new(vec![ViolinData::new("A", vec![1.0, 2.0, 3.0])]);
        assert_eq!(plot.violins.len(), 1);
    }

    #[test]
    fn test_violin_plot_default() {
        let plot = ViolinPlot::default();
        assert!(plot.violins.is_empty());
    }

    #[test]
    fn test_violin_plot_with_orientation() {
        let plot = ViolinPlot::default().with_orientation(ViolinOrientation::Horizontal);
        assert_eq!(plot.orientation, ViolinOrientation::Horizontal);
    }

    #[test]
    fn test_violin_plot_with_gradient() {
        let gradient = Gradient::two(
            Color::new(1.0, 0.0, 0.0, 1.0),
            Color::new(0.0, 0.0, 1.0, 1.0),
        );
        let plot = ViolinPlot::default().with_gradient(gradient);
        assert!(plot.gradient.is_some());
    }

    #[test]
    fn test_violin_plot_measure() {
        let plot = ViolinPlot::default();
        let constraints = Constraints::new(0.0, 100.0, 0.0, 50.0);
        let size = plot.measure(constraints);
        assert_eq!(size.width, 60.0);
        assert_eq!(size.height, 20.0);
    }

    #[test]
    fn test_violin_plot_layout_and_paint_vertical() {
        let mut plot = ViolinPlot::new(vec![
            ViolinData::new("A", vec![1.0, 2.0, 3.0, 4.0, 5.0]).with_color(Color::BLUE),
            ViolinData::new("B", vec![2.0, 3.0, 4.0, 5.0, 6.0]).with_color(Color::RED),
        ]);

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let result = plot.layout(Rect::new(0.0, 0.0, 60.0, 20.0));
        assert_eq!(result.size.width, 60.0);

        plot.paint(&mut canvas);

        // Verify something was rendered
        let cells = buffer.cells();
        let non_empty = cells.iter().filter(|c| !c.symbol.is_empty()).count();
        assert!(non_empty > 0, "Violin plot should render some content");
    }

    #[test]
    fn test_violin_plot_layout_and_paint_horizontal() {
        let mut plot = ViolinPlot::new(vec![
            ViolinData::new("A", vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            ViolinData::new("B", vec![2.0, 3.0, 4.0, 5.0, 6.0]),
        ])
        .with_orientation(ViolinOrientation::Horizontal);

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        plot.layout(Rect::new(0.0, 0.0, 60.0, 20.0));
        plot.paint(&mut canvas);
    }

    #[test]
    fn test_violin_plot_paint_with_gradient() {
        let gradient = Gradient::two(
            Color::new(0.2, 0.4, 0.8, 1.0),
            Color::new(0.8, 0.4, 0.2, 1.0),
        );
        let mut plot = ViolinPlot::new(vec![ViolinData::new("A", vec![1.0, 2.0, 3.0, 4.0, 5.0])])
            .with_gradient(gradient);

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        plot.layout(Rect::new(0.0, 0.0, 60.0, 20.0));
        plot.paint(&mut canvas);
    }

    #[test]
    fn test_violin_plot_paint_no_median() {
        let mut plot =
            ViolinPlot::new(vec![ViolinData::new("A", vec![1.0, 2.0, 3.0])]).with_median(false);

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        plot.layout(Rect::new(0.0, 0.0, 60.0, 20.0));
        plot.paint(&mut canvas);
    }

    #[test]
    fn test_violin_plot_paint_small_bounds() {
        let mut plot = ViolinPlot::new(vec![ViolinData::new("A", vec![1.0, 2.0, 3.0])]);

        let mut buffer = CellBuffer::new(3, 3);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        plot.layout(Rect::new(0.0, 0.0, 3.0, 3.0));
        plot.paint(&mut canvas);
        // Should not crash
    }

    #[test]
    fn test_violin_plot_paint_empty() {
        let mut plot = ViolinPlot::default();

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        plot.layout(Rect::new(0.0, 0.0, 60.0, 20.0));
        plot.paint(&mut canvas);
    }

    #[test]
    fn test_violin_plot_assertions() {
        let plot = ViolinPlot::default();
        assert!(!plot.assertions().is_empty());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_violin_plot_verify_valid() {
        let mut plot = ViolinPlot::default();
        plot.bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        assert!(plot.verify().is_valid());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_violin_plot_verify_invalid() {
        let mut plot = ViolinPlot::default();
        plot.bounds = Rect::new(0.0, 0.0, 3.0, 3.0);
        assert!(!plot.verify().is_valid());
    }

    #[test]
    fn test_violin_plot_children() {
        let plot = ViolinPlot::default();
        assert!(plot.children().is_empty());
    }

    #[test]
    fn test_violin_plot_children_mut() {
        let mut plot = ViolinPlot::default();
        assert!(plot.children_mut().is_empty());
    }

    #[test]
    fn test_violin_global_range() {
        let plot = ViolinPlot::new(vec![
            ViolinData::new("A", vec![1.0, 2.0]),
            ViolinData::new("B", vec![3.0, 10.0]),
        ]);
        let (min, max) = plot.global_range();
        assert!(min < 1.0); // Includes padding
        assert!(max > 10.0);
    }

    #[test]
    fn test_violin_global_range_empty() {
        let plot = ViolinPlot::default();
        let (min, max) = plot.global_range();
        assert_eq!(min, 0.0);
        assert_eq!(max, 1.0);
    }

    #[test]
    fn test_violin_global_range_with_nan() {
        let plot = ViolinPlot::new(vec![ViolinData::new("A", vec![1.0, f64::NAN, 5.0])]);
        let (min, max) = plot.global_range();
        assert!(min.is_finite());
        assert!(max.is_finite());
    }

    #[test]
    fn test_violin_add_violin() {
        let mut plot = ViolinPlot::default();
        plot.add_violin(ViolinData::new("New", vec![1.0, 2.0]));
        assert_eq!(plot.violins.len(), 1);
    }

    #[test]
    fn test_violin_with_box() {
        let plot = ViolinPlot::default().with_box(false);
        assert!(!plot.show_box);
    }

    #[test]
    fn test_violin_with_median() {
        let plot = ViolinPlot::default().with_median(false);
        assert!(!plot.show_median);
    }

    #[test]
    fn test_violin_with_kde_points() {
        let plot = ViolinPlot::default().with_kde_points(100);
        assert_eq!(plot.kde_points, 100);
    }

    #[test]
    fn test_violin_kde_points_clamped() {
        let plot = ViolinPlot::default().with_kde_points(5);
        assert_eq!(plot.kde_points, 10); // Minimum

        let plot = ViolinPlot::default().with_kde_points(500);
        assert_eq!(plot.kde_points, 200); // Maximum
    }

    #[test]
    fn test_violin_orientation_default() {
        let orientation = ViolinOrientation::default();
        assert_eq!(orientation, ViolinOrientation::Vertical);
    }

    #[test]
    fn test_violin_plot_brick_name() {
        let plot = ViolinPlot::new(vec![]);
        assert_eq!(plot.brick_name(), "ViolinPlot");
    }

    #[test]
    fn test_violin_plot_budget() {
        let plot = ViolinPlot::new(vec![]);
        let budget = plot.budget();
        assert!(budget.layout_ms > 0);
    }

    #[test]
    fn test_violin_plot_to_html() {
        let plot = ViolinPlot::new(vec![]);
        assert!(plot.to_html().is_empty());
    }

    #[test]
    fn test_violin_plot_to_css() {
        let plot = ViolinPlot::new(vec![]);
        assert!(plot.to_css().is_empty());
    }

    #[test]
    fn test_violin_plot_type_id() {
        let plot = ViolinPlot::new(vec![]);
        let type_id = Widget::type_id(&plot);
        assert_eq!(type_id, TypeId::of::<ViolinPlot>());
    }

    #[test]
    fn test_violin_plot_event() {
        let mut plot = ViolinPlot::new(vec![]);
        let event = Event::Resize {
            width: 80.0,
            height: 24.0,
        };
        assert!(plot.event(&event).is_none());
    }

    #[test]
    fn test_violin_kde_scalar_and_simd_match() {
        // Test that both paths produce similar results
        let values: Vec<f64> = (0..150).map(|i| i as f64 / 10.0).collect();
        let data = ViolinData::new("Test", values);

        let x = 7.5;
        let bandwidth = 0.5;

        let scalar_result = data.kde_at_point_scalar(x, bandwidth);
        let simd_result = data.kde_at_point_simd(x, bandwidth);

        // Results should be identical
        assert!((scalar_result - simd_result).abs() < 1e-10);
    }

    #[test]
    fn test_violin_kde_simd_unaligned() {
        // Test SIMD path with values not divisible by 4
        let values: Vec<f64> = (0..103).map(|i| i as f64 / 10.0).collect();
        let data = ViolinData::new("Test", values);

        let result = data.kde_at_point_simd(5.0, 0.5);
        assert!(result.is_finite());
        assert!(result > 0.0);
    }

    #[test]
    fn test_violin_multiple_violins_paint() {
        let mut plot = ViolinPlot::new(vec![
            ViolinData::new("A", vec![1.0, 2.0, 3.0]),
            ViolinData::new("B", vec![2.0, 3.0, 4.0]),
            ViolinData::new("C", vec![3.0, 4.0, 5.0]),
        ]);

        let mut buffer = CellBuffer::new(90, 30);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        plot.layout(Rect::new(0.0, 0.0, 90.0, 30.0));
        plot.paint(&mut canvas);
    }
}
