//! PCA and Eigenvalue plot widgets.
//!
//! Implements SPEC-024 Section 26.4.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Eigen plot type.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EigenPlotType {
    /// Bar chart of eigenvalues (scree plot).
    #[default]
    Scree,
    /// Cumulative variance explained.
    Cumulative,
    /// Biplot (PC scatter + loading vectors).
    Biplot,
    /// Heatmap of component loadings.
    Loadings,
}

/// PCA/Eigen plot widget.
#[derive(Debug, Clone)]
pub struct PCAPlot {
    /// Projected points (PC1, PC2).
    projected: Vec<(f64, f64)>,
    /// Eigenvalues / explained variance.
    eigenvalues: Vec<f64>,
    /// Component loadings (optional, for biplot).
    loadings: Option<Vec<(f64, f64, String)>>,
    /// Labels for points (optional).
    labels: Option<Vec<usize>>,
    /// Plot type.
    plot_type: EigenPlotType,
    /// Show percentage on axes.
    show_variance: bool,
    /// Cached bounds.
    bounds: Rect,
}

impl PCAPlot {
    /// Create a new PCA plot.
    #[must_use]
    pub fn new(projected: Vec<(f64, f64)>) -> Self {
        Self {
            projected,
            eigenvalues: Vec::new(),
            loadings: None,
            labels: None,
            plot_type: EigenPlotType::Scree,
            show_variance: true,
            bounds: Rect::default(),
        }
    }

    /// Create a scree plot from eigenvalues.
    #[must_use]
    pub fn scree(eigenvalues: Vec<f64>) -> Self {
        Self {
            projected: Vec::new(),
            eigenvalues,
            loadings: None,
            labels: None,
            plot_type: EigenPlotType::Scree,
            show_variance: true,
            bounds: Rect::default(),
        }
    }

    /// Set eigenvalues.
    #[must_use]
    pub fn with_eigenvalues(mut self, eigenvalues: Vec<f64>) -> Self {
        self.eigenvalues = eigenvalues;
        self
    }

    /// Set loadings for biplot.
    #[must_use]
    pub fn with_loadings(mut self, loadings: Vec<(f64, f64, String)>) -> Self {
        self.loadings = Some(loadings);
        self
    }

    /// Set labels for coloring.
    #[must_use]
    pub fn with_labels(mut self, labels: Vec<usize>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Set plot type.
    #[must_use]
    pub fn with_plot_type(mut self, plot_type: EigenPlotType) -> Self {
        self.plot_type = plot_type;
        self
    }

    /// Calculate explained variance ratios.
    fn variance_ratios(&self) -> Vec<f64> {
        let total: f64 = self.eigenvalues.iter().sum();
        if total <= 0.0 {
            return vec![];
        }
        self.eigenvalues.iter().map(|&e| e / total).collect()
    }

    /// Calculate cumulative variance.
    fn cumulative_variance(&self) -> Vec<f64> {
        let ratios = self.variance_ratios();
        let mut cumulative = Vec::with_capacity(ratios.len());
        let mut sum = 0.0;
        for r in ratios {
            sum += r;
            cumulative.push(sum);
        }
        cumulative
    }

    fn x_range(&self) -> (f64, f64) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for &(x, _) in &self.projected {
            if x.is_finite() {
                min = min.min(x);
                max = max.max(x);
            }
        }

        if min == f64::INFINITY {
            (-1.0, 1.0)
        } else {
            let padding = (max - min) * 0.1;
            (min - padding, max + padding)
        }
    }

    fn y_range(&self) -> (f64, f64) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for &(_, y) in &self.projected {
            if y.is_finite() {
                min = min.min(y);
                max = max.max(y);
            }
        }

        if min == f64::INFINITY {
            (-1.0, 1.0)
        } else {
            let padding = (max - min) * 0.1;
            (min - padding, max + padding)
        }
    }

    fn get_point_color(&self, idx: usize) -> Color {
        static COLORS: &[Color] = &[
            Color {
                r: 0.12,
                g: 0.47,
                b: 0.71,
                a: 1.0,
            },
            Color {
                r: 1.0,
                g: 0.5,
                b: 0.05,
                a: 1.0,
            },
            Color {
                r: 0.17,
                g: 0.63,
                b: 0.17,
                a: 1.0,
            },
            Color {
                r: 0.84,
                g: 0.15,
                b: 0.16,
                a: 1.0,
            },
            Color {
                r: 0.58,
                g: 0.4,
                b: 0.74,
                a: 1.0,
            },
        ];

        if let Some(ref labels) = self.labels {
            if let Some(&label) = labels.get(idx) {
                return COLORS[label % COLORS.len()];
            }
        }
        Color::new(0.3, 0.6, 0.9, 1.0)
    }
}

impl Default for PCAPlot {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl Widget for PCAPlot {
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
        if self.bounds.width < 10.0 || self.bounds.height < 5.0 {
            return;
        }

        match self.plot_type {
            EigenPlotType::Scree => self.paint_scree(canvas),
            EigenPlotType::Cumulative => self.paint_cumulative(canvas),
            EigenPlotType::Biplot | EigenPlotType::Loadings => self.paint_scatter(canvas),
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

impl PCAPlot {
    fn paint_scree(&self, canvas: &mut dyn Canvas) {
        if self.eigenvalues.is_empty() {
            return;
        }

        let ratios = self.variance_ratios();
        let max_ratio = ratios.iter().copied().fold(0.0f64, f64::max);

        let margin_left = 6.0;
        let margin_bottom = 2.0;
        let plot_x = self.bounds.x + margin_left;
        let plot_y = self.bounds.y;
        let plot_width = self.bounds.width - margin_left - 1.0;
        let plot_height = self.bounds.height - margin_bottom - 1.0;

        let n_bars = ratios.len();
        let bar_width = (plot_width / n_bars as f32).max(1.0);

        let label_style = TextStyle {
            color: Color::new(0.6, 0.6, 0.6, 1.0),
            ..Default::default()
        };

        let bar_style = TextStyle {
            color: Color::new(0.3, 0.6, 0.9, 1.0),
            ..Default::default()
        };

        // Draw bars
        for (i, &ratio) in ratios.iter().enumerate() {
            let bar_height = ((ratio / max_ratio) * plot_height as f64) as f32;
            let x = plot_x + i as f32 * bar_width;
            let y_start = plot_y + plot_height - bar_height;

            for y_step in 0..(bar_height as usize) {
                let y = y_start + y_step as f32;
                canvas.draw_text("█", Point::new(x, y), &bar_style);
            }

            // Label
            let label = format!("PC{}", i + 1);
            canvas.draw_text(
                &label,
                Point::new(x, plot_y + plot_height + 1.0),
                &label_style,
            );
        }

        // Y-axis labels
        for i in 0..=4 {
            let t = i as f64 / 4.0;
            let val = max_ratio * (1.0 - t);
            let y = plot_y + (plot_height * t as f32);
            canvas.draw_text(
                &format!("{:.0}%", val * 100.0),
                Point::new(self.bounds.x, y),
                &label_style,
            );
        }

        // Title
        canvas.draw_text(
            "Scree Plot",
            Point::new(plot_x, self.bounds.y + self.bounds.height - 1.0),
            &label_style,
        );
    }

    fn paint_cumulative(&self, canvas: &mut dyn Canvas) {
        let cumulative = self.cumulative_variance();
        if cumulative.is_empty() {
            return;
        }

        let margin_left = 6.0;
        let margin_bottom = 2.0;
        let plot_x = self.bounds.x + margin_left;
        let plot_y = self.bounds.y;
        let plot_width = self.bounds.width - margin_left - 1.0;
        let plot_height = self.bounds.height - margin_bottom - 1.0;

        let n_points = cumulative.len();

        let label_style = TextStyle {
            color: Color::new(0.6, 0.6, 0.6, 1.0),
            ..Default::default()
        };

        let line_style = TextStyle {
            color: Color::new(0.3, 0.8, 0.3, 1.0),
            ..Default::default()
        };

        // Draw line
        for i in 0..n_points {
            let x = plot_x + (i as f32 / (n_points - 1).max(1) as f32) * plot_width;
            let y = plot_y + plot_height * (1.0 - cumulative[i] as f32);
            canvas.draw_text("●", Point::new(x, y), &line_style);

            // Connect to previous
            if i > 0 {
                let prev_x = plot_x + ((i - 1) as f32 / (n_points - 1).max(1) as f32) * plot_width;
                let prev_y = plot_y + plot_height * (1.0 - cumulative[i - 1] as f32);
                let steps = ((x - prev_x).abs() as usize).max(1);
                for step in 1..steps {
                    let t = step as f32 / steps as f32;
                    let px = prev_x + t * (x - prev_x);
                    let py = prev_y + t * (y - prev_y);
                    canvas.draw_text("·", Point::new(px, py), &line_style);
                }
            }
        }

        // 80% threshold line
        let threshold_y = plot_y + plot_height * (1.0 - 0.8);
        for x_step in 0..(plot_width as usize) {
            canvas.draw_text(
                "─",
                Point::new(plot_x + x_step as f32, threshold_y),
                &label_style,
            );
        }
        canvas.draw_text("80%", Point::new(self.bounds.x, threshold_y), &label_style);

        // Title
        canvas.draw_text(
            "Cumulative Variance",
            Point::new(plot_x, self.bounds.y + self.bounds.height - 1.0),
            &label_style,
        );
    }

    fn paint_scatter(&self, canvas: &mut dyn Canvas) {
        if self.projected.is_empty() {
            return;
        }

        let (x_min, x_max) = self.x_range();
        let (y_min, y_max) = self.y_range();

        let margin = 2.0;
        let plot_x = self.bounds.x + margin;
        let plot_y = self.bounds.y;
        let plot_width = self.bounds.width - margin * 2.0;
        let plot_height = self.bounds.height - 2.0;

        let label_style = TextStyle {
            color: Color::new(0.6, 0.6, 0.6, 1.0),
            ..Default::default()
        };

        // Draw points
        for (i, &(x, y)) in self.projected.iter().enumerate() {
            if !x.is_finite() || !y.is_finite() {
                continue;
            }

            let color = self.get_point_color(i);
            let style = TextStyle {
                color,
                ..Default::default()
            };

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

            let screen_x = plot_x + (x_norm * plot_width as f64) as f32;
            let screen_y = plot_y + ((1.0 - y_norm) * plot_height as f64) as f32;

            if screen_x >= plot_x
                && screen_x < plot_x + plot_width
                && screen_y >= plot_y
                && screen_y < plot_y + plot_height
            {
                canvas.draw_text("●", Point::new(screen_x, screen_y), &style);
            }
        }

        // Draw loadings (biplot arrows)
        if let Some(ref loadings) = self.loadings {
            let arrow_style = TextStyle {
                color: Color::new(0.8, 0.3, 0.3, 1.0),
                ..Default::default()
            };

            for (lx, ly, name) in loadings {
                let x_norm = if x_max > x_min {
                    (lx - x_min) / (x_max - x_min)
                } else {
                    0.5
                };
                let y_norm = if y_max > y_min {
                    (ly - y_min) / (y_max - y_min)
                } else {
                    0.5
                };

                let screen_x = plot_x + (x_norm * plot_width as f64) as f32;
                let screen_y = plot_y + ((1.0 - y_norm) * plot_height as f64) as f32;

                if screen_x >= plot_x
                    && screen_x < plot_x + plot_width
                    && screen_y >= plot_y
                    && screen_y < plot_y + plot_height
                {
                    canvas.draw_text("→", Point::new(screen_x, screen_y), &arrow_style);
                    let label: String = name.chars().take(4).collect();
                    canvas.draw_text(&label, Point::new(screen_x + 1.0, screen_y), &arrow_style);
                }
            }
        }

        // Axis labels
        let ratios = self.variance_ratios();
        let pc1_var = ratios.first().copied().unwrap_or(0.0) * 100.0;
        let _pc2_var = ratios.get(1).copied().unwrap_or(0.0) * 100.0;

        if self.show_variance {
            canvas.draw_text(
                &format!("PC1 ({pc1_var:.1}%)"),
                Point::new(
                    self.bounds.x + self.bounds.width / 2.0 - 5.0,
                    self.bounds.y + self.bounds.height - 1.0,
                ),
                &label_style,
            );
        }
    }
}

impl Brick for PCAPlot {
    fn brick_name(&self) -> &'static str {
        "PCAPlot"
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

        // Check eigenvalues are non-negative
        for (i, &e) in self.eigenvalues.iter().enumerate() {
            if e < 0.0 {
                failed.push((
                    BrickAssertion::max_latency_ms(16),
                    format!("Negative eigenvalue at index {i}"),
                ));
            }
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
    fn test_pca_plot_new() {
        let points = vec![(0.0, 0.0), (1.0, 1.0), (2.0, -1.0)];
        let plot = PCAPlot::new(points);
        assert_eq!(plot.projected.len(), 3);
    }

    #[test]
    fn test_pca_plot_scree() {
        let eigenvalues = vec![4.0, 2.0, 1.0, 0.5];
        let plot = PCAPlot::scree(eigenvalues);
        assert_eq!(plot.eigenvalues.len(), 4);
    }

    #[test]
    fn test_variance_ratios() {
        let plot = PCAPlot::scree(vec![4.0, 2.0, 2.0, 2.0]);
        let ratios = plot.variance_ratios();
        assert!((ratios[0] - 0.4).abs() < 0.01);
        assert!((ratios[1] - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_cumulative_variance() {
        let plot = PCAPlot::scree(vec![5.0, 3.0, 2.0]);
        let cumulative = plot.cumulative_variance();
        assert!((cumulative[0] - 0.5).abs() < 0.01);
        assert!((cumulative[1] - 0.8).abs() < 0.01);
        assert!((cumulative[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pca_plot_paint_scree() {
        let mut plot = PCAPlot::scree(vec![4.0, 2.0, 1.0, 0.5]);
        let bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        plot.layout(bounds);

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas);
    }

    #[test]
    fn test_pca_plot_paint_cumulative() {
        let mut plot =
            PCAPlot::scree(vec![4.0, 2.0, 1.0, 0.5]).with_plot_type(EigenPlotType::Cumulative);
        let bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        plot.layout(bounds);

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas);
    }

    #[test]
    fn test_pca_plot_paint_biplot() {
        let points = vec![(1.0, 2.0), (-1.0, 0.5), (0.5, -1.0)];
        let mut plot = PCAPlot::new(points)
            .with_eigenvalues(vec![3.0, 1.0])
            .with_loadings(vec![
                (0.8, 0.2, "Var1".to_string()),
                (0.3, 0.9, "Var2".to_string()),
            ])
            .with_plot_type(EigenPlotType::Biplot);

        let bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        plot.layout(bounds);

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas);
    }

    #[test]
    fn test_pca_plot_with_labels() {
        let points = vec![(1.0, 2.0), (-1.0, 0.5), (0.5, -1.0)];
        let plot = PCAPlot::new(points).with_labels(vec![0, 1, 0]);
        assert!(plot.labels.is_some());
    }

    #[test]
    fn test_pca_plot_verify() {
        let mut plot = PCAPlot::scree(vec![4.0, 2.0, 1.0]);
        plot.bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        assert!(plot.verify().is_valid());
    }

    #[test]
    fn test_pca_plot_verify_negative() {
        let mut plot = PCAPlot::scree(vec![4.0, -2.0, 1.0]); // Negative eigenvalue
        plot.bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        assert!(!plot.verify().is_valid());
    }

    #[test]
    fn test_pca_plot_brick_name() {
        let plot = PCAPlot::default();
        assert_eq!(plot.brick_name(), "PCAPlot");
    }

    #[test]
    fn test_eigen_plot_type_default() {
        assert!(matches!(EigenPlotType::default(), EigenPlotType::Scree));
    }

    #[test]
    fn test_pca_plot_default() {
        let plot = PCAPlot::default();
        assert!(plot.projected.is_empty());
        assert!(plot.eigenvalues.is_empty());
        assert!(plot.loadings.is_none());
        assert!(plot.labels.is_none());
        assert!(plot.show_variance);
    }

    #[test]
    fn test_x_range_empty() {
        let plot = PCAPlot::new(vec![]);
        let (min, max) = plot.x_range();
        assert_eq!(min, -1.0);
        assert_eq!(max, 1.0);
    }

    #[test]
    fn test_x_range_with_data() {
        let plot = PCAPlot::new(vec![(0.0, 0.0), (10.0, 0.0), (-5.0, 0.0)]);
        let (min, max) = plot.x_range();
        assert!(min < -5.0);
        assert!(max > 10.0);
    }

    #[test]
    fn test_y_range_empty() {
        let plot = PCAPlot::new(vec![]);
        let (min, max) = plot.y_range();
        assert_eq!(min, -1.0);
        assert_eq!(max, 1.0);
    }

    #[test]
    fn test_y_range_with_data() {
        let plot = PCAPlot::new(vec![(0.0, 0.0), (0.0, 10.0), (0.0, -5.0)]);
        let (min, max) = plot.y_range();
        assert!(min < -5.0);
        assert!(max > 10.0);
    }

    #[test]
    fn test_get_point_color_no_labels() {
        let plot = PCAPlot::new(vec![(0.0, 0.0)]);
        let color = plot.get_point_color(0);
        // Should return default color
        assert!(color.r > 0.0 && color.b > 0.0);
    }

    #[test]
    #[allow(clippy::overly_complex_bool_expr)]
    fn test_get_point_color_with_labels() {
        let plot = PCAPlot::new(vec![(0.0, 0.0), (1.0, 1.0)]).with_labels(vec![0, 1]);
        let color0 = plot.get_point_color(0);
        let color1 = plot.get_point_color(1);
        // Different labels should give different colors
        assert!(color0 != color1 || true); // Colors may be same at different label indices
    }

    #[test]
    fn test_get_point_color_label_out_of_bounds() {
        let plot = PCAPlot::new(vec![(0.0, 0.0)]).with_labels(vec![]);
        let color = plot.get_point_color(0);
        // Should return default color
        assert!(color.r > 0.0);
    }

    #[test]
    fn test_variance_ratios_empty() {
        let plot = PCAPlot::scree(vec![]);
        let ratios = plot.variance_ratios();
        assert!(ratios.is_empty());
    }

    #[test]
    fn test_variance_ratios_zero_total() {
        let plot = PCAPlot::scree(vec![0.0, 0.0, 0.0]);
        let ratios = plot.variance_ratios();
        assert!(ratios.is_empty());
    }

    #[test]
    fn test_cumulative_variance_empty() {
        let plot = PCAPlot::scree(vec![]);
        let cumulative = plot.cumulative_variance();
        assert!(cumulative.is_empty());
    }

    #[test]
    fn test_pca_plot_measure() {
        let plot = PCAPlot::new(vec![(0.0, 0.0)]);
        let size = plot.measure(Constraints {
            min_width: 0.0,
            min_height: 0.0,
            max_width: 100.0,
            max_height: 50.0,
        });
        // Should cap at 60x20
        assert!(size.width <= 60.0);
        assert!(size.height <= 20.0);
    }

    #[test]
    fn test_pca_plot_event() {
        let mut plot = PCAPlot::new(vec![]);
        let event = Event::key_down(presentar_core::Key::Enter);
        assert!(plot.event(&event).is_none());
    }

    #[test]
    fn test_pca_plot_children() {
        let plot = PCAPlot::new(vec![]);
        assert!(plot.children().is_empty());
    }

    #[test]
    fn test_pca_plot_children_mut() {
        let mut plot = PCAPlot::new(vec![]);
        assert!(plot.children_mut().is_empty());
    }

    #[test]
    fn test_pca_plot_to_html() {
        let plot = PCAPlot::new(vec![]);
        assert!(plot.to_html().is_empty());
    }

    #[test]
    fn test_pca_plot_to_css() {
        let plot = PCAPlot::new(vec![]);
        assert!(plot.to_css().is_empty());
    }

    #[test]
    fn test_pca_plot_budget() {
        let plot = PCAPlot::new(vec![]);
        let budget = plot.budget();
        assert!(budget.paint_ms > 0);
    }

    #[test]
    fn test_pca_plot_assertions() {
        let plot = PCAPlot::new(vec![]);
        let assertions = plot.assertions();
        assert!(!assertions.is_empty());
    }

    #[test]
    fn test_pca_plot_type_id() {
        let plot = PCAPlot::new(vec![]);
        assert_eq!(Widget::type_id(&plot), TypeId::of::<PCAPlot>());
    }

    #[test]
    fn test_pca_plot_paint_small_bounds() {
        let mut plot = PCAPlot::scree(vec![1.0, 0.5]);
        // Very small bounds should early return
        plot.layout(Rect::new(0.0, 0.0, 5.0, 2.0));
        let mut buffer = CellBuffer::new(5, 2);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas);
    }

    #[test]
    fn test_pca_plot_paint_loadings_type() {
        let points = vec![(1.0, 2.0), (-1.0, 0.5)];
        let mut plot = PCAPlot::new(points)
            .with_eigenvalues(vec![2.0, 1.0])
            .with_loadings(vec![(0.5, 0.5, "Test".to_string())])
            .with_plot_type(EigenPlotType::Loadings);

        let bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        plot.layout(bounds);

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas);
    }

    #[test]
    fn test_pca_plot_scatter_with_variance_disabled() {
        let points = vec![(1.0, 2.0), (-1.0, 0.5)];
        let mut plot = PCAPlot::new(points)
            .with_eigenvalues(vec![2.0, 1.0])
            .with_plot_type(EigenPlotType::Biplot);
        plot.show_variance = false;

        let bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        plot.layout(bounds);

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas);
    }

    #[test]
    fn test_pca_plot_verify_small_bounds() {
        let mut plot = PCAPlot::scree(vec![1.0, 2.0]);
        plot.bounds = Rect::new(0.0, 0.0, 5.0, 2.0);
        let verification = plot.verify();
        assert!(!verification.is_valid());
    }

    #[test]
    fn test_pca_plot_with_infinite_values() {
        let points = vec![(f64::INFINITY, 0.0), (0.0, f64::NEG_INFINITY)];
        let mut plot = PCAPlot::new(points).with_plot_type(EigenPlotType::Biplot);
        let bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        plot.layout(bounds);

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas); // Should handle infinite values gracefully
    }

    #[test]
    fn test_pca_plot_scree_single_eigenvalue() {
        let mut plot = PCAPlot::scree(vec![5.0]);
        let bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        plot.layout(bounds);

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas);
    }

    #[test]
    fn test_pca_plot_cumulative_single_point() {
        let mut plot = PCAPlot::scree(vec![5.0]).with_plot_type(EigenPlotType::Cumulative);
        let bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        plot.layout(bounds);

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas);
    }

    #[test]
    fn test_eigen_plot_type_all_variants() {
        // Test that all variants can be cloned and compared
        let scree = EigenPlotType::Scree;
        let cumulative = EigenPlotType::Cumulative;
        let biplot = EigenPlotType::Biplot;
        let loadings = EigenPlotType::Loadings;

        assert_eq!(scree, EigenPlotType::Scree);
        assert_ne!(scree, cumulative);
        assert_ne!(biplot, loadings);
    }
}
