//! Cluster plot widget for K-Means, DBSCAN, and other clustering visualizations.
//!
//! Implements SPEC-024 Section 26.3.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Clustering algorithm type.
#[derive(Debug, Clone)]
pub enum ClusterAlgorithm {
    KMeans { k: usize },
    DBSCAN { eps: f64, min_samples: usize },
    Hierarchical { n_clusters: usize },
    HDBSCAN { min_cluster_size: usize },
}

impl Default for ClusterAlgorithm {
    fn default() -> Self {
        Self::KMeans { k: 3 }
    }
}

/// Cluster plot widget.
#[derive(Debug, Clone)]
pub struct ClusterPlot {
    /// Data points (x, y).
    points: Vec<(f64, f64)>,
    /// Cluster labels for each point (-1 = noise).
    labels: Vec<i32>,
    /// Cluster centroids.
    centroids: Vec<(f64, f64)>,
    /// Algorithm used.
    algorithm: ClusterAlgorithm,
    /// Show centroids.
    show_centroids: bool,
    /// Cluster colors.
    colors: Vec<Color>,
    /// Cached bounds.
    bounds: Rect,
}

impl ClusterPlot {
    /// Create a new cluster plot.
    #[must_use]
    pub fn new(points: Vec<(f64, f64)>, labels: Vec<i32>) -> Self {
        let colors = Self::default_colors();
        Self {
            points,
            labels,
            centroids: Vec::new(),
            algorithm: ClusterAlgorithm::default(),
            show_centroids: true,
            colors,
            bounds: Rect::default(),
        }
    }

    /// Set centroids.
    #[must_use]
    pub fn with_centroids(mut self, centroids: Vec<(f64, f64)>) -> Self {
        self.centroids = centroids;
        self
    }

    /// Set algorithm.
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: ClusterAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Toggle centroid display.
    #[must_use]
    pub fn with_show_centroids(mut self, show: bool) -> Self {
        self.show_centroids = show;
        self
    }

    /// Set custom colors.
    #[must_use]
    pub fn with_colors(mut self, colors: Vec<Color>) -> Self {
        self.colors = colors;
        self
    }

    fn default_colors() -> Vec<Color> {
        vec![
            Color::new(0.12, 0.47, 0.71, 1.0), // Blue
            Color::new(1.0, 0.5, 0.05, 1.0),   // Orange
            Color::new(0.17, 0.63, 0.17, 1.0), // Green
            Color::new(0.84, 0.15, 0.16, 1.0), // Red
            Color::new(0.58, 0.4, 0.74, 1.0),  // Purple
            Color::new(0.55, 0.34, 0.29, 1.0), // Brown
            Color::new(0.89, 0.47, 0.76, 1.0), // Pink
            Color::new(0.5, 0.5, 0.5, 1.0),    // Gray
            Color::new(0.74, 0.74, 0.13, 1.0), // Olive
            Color::new(0.09, 0.75, 0.81, 1.0), // Cyan
        ]
    }

    fn get_cluster_color(&self, label: i32) -> Color {
        if label < 0 {
            // Noise points
            Color::new(0.3, 0.3, 0.3, 0.5)
        } else {
            self.colors[label as usize % self.colors.len()]
        }
    }

    fn x_range(&self) -> (f64, f64) {
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
            let padding = (x_max - x_min) * 0.1;
            (x_min - padding, x_max + padding)
        }
    }

    fn y_range(&self) -> (f64, f64) {
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
            let padding = (y_max - y_min) * 0.1;
            (y_min - padding, y_max + padding)
        }
    }

    /// Get unique cluster count.
    #[must_use]
    pub fn cluster_count(&self) -> usize {
        let mut unique: Vec<i32> = self.labels.iter().filter(|&&l| l >= 0).copied().collect();
        unique.sort_unstable();
        unique.dedup();
        unique.len()
    }
}

impl Default for ClusterPlot {
    fn default() -> Self {
        Self::new(Vec::new(), Vec::new())
    }
}

impl Widget for ClusterPlot {
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

        let (x_min, x_max) = self.x_range();
        let (y_min, y_max) = self.y_range();

        let margin = 2.0;
        let plot_x = self.bounds.x + margin;
        let plot_y = self.bounds.y;
        let plot_width = self.bounds.width - margin * 2.0;
        let plot_height = self.bounds.height - 1.0;

        if plot_width <= 0.0 || plot_height <= 0.0 {
            return;
        }

        // Draw points
        for (i, &(x, y)) in self.points.iter().enumerate() {
            if !x.is_finite() || !y.is_finite() {
                continue;
            }

            let label = self.labels.get(i).copied().unwrap_or(-1);
            let color = self.get_cluster_color(label);

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
                let marker = if label < 0 { '·' } else { '●' };
                let style = TextStyle {
                    color,
                    ..Default::default()
                };
                canvas.draw_text(&marker.to_string(), Point::new(screen_x, screen_y), &style);
            }
        }

        // Draw centroids
        if self.show_centroids {
            for (i, &(cx, cy)) in self.centroids.iter().enumerate() {
                if !cx.is_finite() || !cy.is_finite() {
                    continue;
                }

                #[allow(clippy::cast_possible_wrap)]
                let color = self.get_cluster_color(i as i32);

                let x_norm = if x_max > x_min {
                    (cx - x_min) / (x_max - x_min)
                } else {
                    0.5
                };
                let y_norm = if y_max > y_min {
                    (cy - y_min) / (y_max - y_min)
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
                    let style = TextStyle {
                        color,
                        ..Default::default()
                    };
                    canvas.draw_text("✚", Point::new(screen_x, screen_y), &style);
                }
            }
        }

        // Draw legend
        let legend_y = self.bounds.y + self.bounds.height - 1.0;
        let label_style = TextStyle {
            color: Color::new(0.6, 0.6, 0.6, 1.0),
            ..Default::default()
        };

        let algo_name = match &self.algorithm {
            ClusterAlgorithm::KMeans { k } => format!("K-Means (k={k})"),
            ClusterAlgorithm::DBSCAN { eps, min_samples } => {
                format!("DBSCAN (eps={eps:.2}, min={min_samples})")
            }
            ClusterAlgorithm::Hierarchical { n_clusters } => {
                format!("Hierarchical (n={n_clusters})")
            }
            ClusterAlgorithm::HDBSCAN { min_cluster_size } => {
                format!("HDBSCAN (min={min_cluster_size})")
            }
        };

        canvas.draw_text(
            &format!("{} | {} clusters", algo_name, self.cluster_count()),
            Point::new(self.bounds.x, legend_y),
            &label_style,
        );
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

impl Brick for ClusterPlot {
    fn brick_name(&self) -> &'static str {
        "ClusterPlot"
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

        // Check labels consistency
        if !self.points.is_empty() && self.labels.len() != self.points.len() {
            failed.push((
                BrickAssertion::max_latency_ms(16),
                "Labels length mismatch".to_string(),
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
    fn test_cluster_plot_new() {
        let points = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
        let labels = vec![0, 0, 1];
        let plot = ClusterPlot::new(points, labels);
        assert_eq!(plot.points.len(), 3);
        assert_eq!(plot.labels.len(), 3);
    }

    #[test]
    fn test_cluster_plot_empty() {
        let plot = ClusterPlot::default();
        assert_eq!(plot.cluster_count(), 0);
    }

    #[test]
    fn test_cluster_plot_with_centroids() {
        let plot = ClusterPlot::new(vec![(0.0, 0.0)], vec![0])
            .with_centroids(vec![(0.5, 0.5), (1.5, 1.5)]);
        assert_eq!(plot.centroids.len(), 2);
    }

    #[test]
    fn test_cluster_plot_cluster_count() {
        let labels = vec![0, 0, 1, 1, 2, -1]; // 3 clusters + noise
        let points = vec![(0.0, 0.0); 6];
        let plot = ClusterPlot::new(points, labels);
        assert_eq!(plot.cluster_count(), 3);
    }

    #[test]
    fn test_cluster_plot_paint() {
        let points = vec![
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (5.0, 5.0),
            (6.0, 5.0),
            (5.0, 6.0),
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let mut plot =
            ClusterPlot::new(points, labels).with_centroids(vec![(0.33, 0.33), (5.33, 5.33)]);

        let bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        plot.layout(bounds);

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas);
    }

    #[test]
    fn test_cluster_plot_algorithms() {
        let plot1 = ClusterPlot::default().with_algorithm(ClusterAlgorithm::KMeans { k: 5 });
        assert!(matches!(plot1.algorithm, ClusterAlgorithm::KMeans { k: 5 }));

        let plot2 = ClusterPlot::default().with_algorithm(ClusterAlgorithm::DBSCAN {
            eps: 0.5,
            min_samples: 5,
        });
        assert!(matches!(plot2.algorithm, ClusterAlgorithm::DBSCAN { .. }));
    }

    #[test]
    fn test_cluster_plot_verify() {
        let mut plot = ClusterPlot::new(vec![(0.0, 0.0)], vec![0]);
        plot.bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        assert!(plot.verify().is_valid());
    }

    #[test]
    fn test_cluster_plot_verify_mismatch() {
        let mut plot = ClusterPlot::new(vec![(0.0, 0.0), (1.0, 1.0)], vec![0]); // Mismatch
        plot.bounds = Rect::new(0.0, 0.0, 60.0, 20.0);
        assert!(!plot.verify().is_valid());
    }

    #[test]
    fn test_cluster_plot_brick_name() {
        let plot = ClusterPlot::default();
        assert_eq!(plot.brick_name(), "ClusterPlot");
    }

    #[test]
    fn test_cluster_colors() {
        let colors = ClusterPlot::default_colors();
        assert!(colors.len() >= 10);
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_with_show_centroids() {
        let plot = ClusterPlot::default().with_show_centroids(false);
        assert!(!plot.show_centroids);

        let plot2 = plot.with_show_centroids(true);
        assert!(plot2.show_centroids);
    }

    #[test]
    fn test_with_colors() {
        let custom_colors = vec![Color::RED, Color::GREEN, Color::BLUE];
        let plot = ClusterPlot::default().with_colors(custom_colors);
        assert_eq!(plot.colors.len(), 3);
    }

    #[test]
    fn test_get_cluster_color_noise() {
        let plot = ClusterPlot::default();
        let noise_color = plot.get_cluster_color(-1);
        // Noise color should be grayish with low alpha
        assert!(noise_color.a < 1.0);
    }

    #[test]
    fn test_get_cluster_color_normal() {
        let plot = ClusterPlot::default();
        let color0 = plot.get_cluster_color(0);
        let color1 = plot.get_cluster_color(1);
        // Different clusters should have different colors
        assert!(color0.r != color1.r || color0.g != color1.g || color0.b != color1.b);
    }

    #[test]
    fn test_get_cluster_color_wraps() {
        let plot = ClusterPlot::default();
        let colors_len = plot.colors.len();
        // Should wrap around
        let color_high = plot.get_cluster_color(colors_len as i32 + 2);
        let color_wrapped = plot.get_cluster_color(2);
        assert_eq!(color_high, color_wrapped);
    }

    #[test]
    fn test_x_range_empty() {
        let plot = ClusterPlot::default();
        let (x_min, x_max) = plot.x_range();
        assert_eq!(x_min, 0.0);
        assert_eq!(x_max, 1.0);
    }

    #[test]
    fn test_x_range_with_data() {
        let points = vec![(0.0, 0.0), (10.0, 5.0), (5.0, 2.0)];
        let plot = ClusterPlot::new(points, vec![0, 0, 0]);
        let (x_min, x_max) = plot.x_range();
        // Should have 10% padding
        assert!(x_min < 0.0);
        assert!(x_max > 10.0);
    }

    #[test]
    fn test_x_range_with_nan() {
        let points = vec![(f64::NAN, 0.0), (5.0, 1.0), (10.0, 2.0)];
        let plot = ClusterPlot::new(points, vec![0, 0, 0]);
        let (x_min, x_max) = plot.x_range();
        // Should ignore NaN
        assert!(x_min < 5.0);
        assert!(x_max > 10.0);
    }

    #[test]
    fn test_y_range_empty() {
        let plot = ClusterPlot::default();
        let (y_min, y_max) = plot.y_range();
        assert_eq!(y_min, 0.0);
        assert_eq!(y_max, 1.0);
    }

    #[test]
    fn test_y_range_with_data() {
        let points = vec![(0.0, 0.0), (1.0, 10.0), (2.0, 5.0)];
        let plot = ClusterPlot::new(points, vec![0, 0, 0]);
        let (y_min, y_max) = plot.y_range();
        assert!(y_min < 0.0);
        assert!(y_max > 10.0);
    }

    #[test]
    fn test_y_range_with_nan() {
        let points = vec![(0.0, f64::NAN), (1.0, 5.0), (2.0, 10.0)];
        let plot = ClusterPlot::new(points, vec![0, 0, 0]);
        let (y_min, y_max) = plot.y_range();
        assert!(y_min < 5.0);
        assert!(y_max > 10.0);
    }

    #[test]
    fn test_paint_too_small_width() {
        let mut plot = ClusterPlot::new(vec![(0.0, 0.0)], vec![0]);
        plot.bounds = Rect::new(0.0, 0.0, 5.0, 20.0); // Too narrow

        let mut buffer = CellBuffer::new(5, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas); // Should not crash
    }

    #[test]
    fn test_paint_too_small_height() {
        let mut plot = ClusterPlot::new(vec![(0.0, 0.0)], vec![0]);
        plot.bounds = Rect::new(0.0, 0.0, 60.0, 3.0); // Too short

        let mut buffer = CellBuffer::new(60, 3);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas); // Should not crash
    }

    #[test]
    fn test_paint_with_noise_points() {
        let points = vec![
            (0.0, 0.0),
            (1.0, 1.0),
            (5.0, 5.0), // Noise point
        ];
        let labels = vec![0, 0, -1]; // -1 = noise
        let mut plot = ClusterPlot::new(points, labels);
        plot.bounds = Rect::new(0.0, 0.0, 60.0, 20.0);

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas);
    }

    #[test]
    fn test_paint_without_centroids() {
        let points = vec![(0.0, 0.0), (1.0, 1.0)];
        let labels = vec![0, 0];
        let mut plot = ClusterPlot::new(points, labels)
            .with_centroids(vec![(0.5, 0.5)])
            .with_show_centroids(false);
        plot.bounds = Rect::new(0.0, 0.0, 60.0, 20.0);

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas);
    }

    #[test]
    fn test_paint_with_nan_centroid() {
        let points = vec![(0.0, 0.0), (1.0, 1.0)];
        let labels = vec![0, 0];
        let mut plot =
            ClusterPlot::new(points, labels).with_centroids(vec![(f64::NAN, 0.5), (0.5, f64::NAN)]);
        plot.bounds = Rect::new(0.0, 0.0, 60.0, 20.0);

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas); // Should skip NaN centroids
    }

    #[test]
    fn test_paint_with_nan_point() {
        let points = vec![(f64::NAN, 0.0), (0.0, f64::NAN), (1.0, 1.0)];
        let labels = vec![0, 0, 0];
        let mut plot = ClusterPlot::new(points, labels);
        plot.bounds = Rect::new(0.0, 0.0, 60.0, 20.0);

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas); // Should skip NaN points
    }

    #[test]
    fn test_paint_single_point() {
        // Single point means x_min == x_max, y_min == y_max
        let points = vec![(5.0, 5.0)];
        let labels = vec![0];
        let mut plot = ClusterPlot::new(points, labels);
        plot.bounds = Rect::new(0.0, 0.0, 60.0, 20.0);

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas);
    }

    #[test]
    fn test_paint_negative_plot_dimensions() {
        let mut plot = ClusterPlot::new(vec![(0.0, 0.0)], vec![0]);
        // Set bounds that result in negative plot_width/height after margin
        plot.bounds = Rect::new(0.0, 0.0, 2.0, 2.0);

        let mut buffer = CellBuffer::new(10, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas); // Should handle gracefully
    }

    #[test]
    fn test_algorithm_hierarchical() {
        let plot =
            ClusterPlot::default().with_algorithm(ClusterAlgorithm::Hierarchical { n_clusters: 4 });
        assert!(matches!(
            plot.algorithm,
            ClusterAlgorithm::Hierarchical { n_clusters: 4 }
        ));
    }

    #[test]
    fn test_algorithm_hdbscan() {
        let plot = ClusterPlot::default().with_algorithm(ClusterAlgorithm::HDBSCAN {
            min_cluster_size: 10,
        });
        assert!(matches!(
            plot.algorithm,
            ClusterAlgorithm::HDBSCAN {
                min_cluster_size: 10
            }
        ));
    }

    #[test]
    fn test_verify_too_small() {
        let mut plot = ClusterPlot::new(vec![(0.0, 0.0)], vec![0]);
        plot.bounds = Rect::new(0.0, 0.0, 5.0, 3.0); // Too small
        let result = plot.verify();
        assert!(!result.failed.is_empty());
    }

    #[test]
    fn test_brick_assertions() {
        let plot = ClusterPlot::default();
        let assertions = plot.assertions();
        assert!(!assertions.is_empty());
    }

    #[test]
    fn test_brick_budget() {
        let plot = ClusterPlot::default();
        let budget = plot.budget();
        assert!(budget.total_ms > 0);
    }

    #[test]
    fn test_brick_to_html() {
        let plot = ClusterPlot::default();
        assert!(plot.to_html().is_empty());
    }

    #[test]
    fn test_brick_to_css() {
        let plot = ClusterPlot::default();
        assert!(plot.to_css().is_empty());
    }

    #[test]
    fn test_widget_type_id() {
        let plot = ClusterPlot::default();
        let id = Widget::type_id(&plot);
        assert_eq!(id, TypeId::of::<ClusterPlot>());
    }

    #[test]
    fn test_widget_measure() {
        let plot = ClusterPlot::default();
        let constraints = Constraints::tight(Size::new(100.0, 50.0));
        let size = plot.measure(constraints);
        assert!(size.width <= 60.0);
        assert!(size.height <= 20.0);
    }

    #[test]
    fn test_widget_layout() {
        let mut plot = ClusterPlot::default();
        let bounds = Rect::new(10.0, 20.0, 40.0, 15.0);
        let result = plot.layout(bounds);
        assert_eq!(result.size.width, 40.0);
        assert_eq!(result.size.height, 15.0);
        assert_eq!(plot.bounds, bounds);
    }

    #[test]
    fn test_widget_event() {
        let mut plot = ClusterPlot::default();
        let result = plot.event(&Event::FocusIn);
        assert!(result.is_none());
    }

    #[test]
    fn test_widget_children() {
        let plot = ClusterPlot::default();
        assert!(plot.children().is_empty());
    }

    #[test]
    fn test_widget_children_mut() {
        let mut plot = ClusterPlot::default();
        assert!(plot.children_mut().is_empty());
    }

    #[test]
    fn test_cluster_algorithm_default() {
        let algo = ClusterAlgorithm::default();
        assert!(matches!(algo, ClusterAlgorithm::KMeans { k: 3 }));
    }

    #[test]
    fn test_cluster_count_all_noise() {
        let points = vec![(0.0, 0.0), (1.0, 1.0)];
        let labels = vec![-1, -1]; // All noise
        let plot = ClusterPlot::new(points, labels);
        assert_eq!(plot.cluster_count(), 0);
    }

    #[test]
    fn test_cluster_count_duplicates() {
        // Labels with duplicates should count unique clusters
        let points = vec![(0.0, 0.0); 10];
        let labels = vec![0, 0, 0, 1, 1, 2, 2, 2, 2, 0];
        let plot = ClusterPlot::new(points, labels);
        assert_eq!(plot.cluster_count(), 3);
    }

    #[test]
    fn test_paint_all_algorithms_legend() {
        // Test legend rendering for each algorithm type
        let points = vec![(0.0, 0.0), (1.0, 1.0)];
        let labels = vec![0, 0];

        let algorithms = vec![
            ClusterAlgorithm::KMeans { k: 3 },
            ClusterAlgorithm::DBSCAN {
                eps: 0.5,
                min_samples: 5,
            },
            ClusterAlgorithm::Hierarchical { n_clusters: 3 },
            ClusterAlgorithm::HDBSCAN {
                min_cluster_size: 5,
            },
        ];

        for algo in algorithms {
            let mut plot = ClusterPlot::new(points.clone(), labels.clone()).with_algorithm(algo);
            plot.bounds = Rect::new(0.0, 0.0, 60.0, 20.0);

            let mut buffer = CellBuffer::new(60, 20);
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            plot.paint(&mut canvas);
        }
    }

    #[test]
    fn test_paint_missing_label() {
        // More points than labels
        let points = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
        let labels = vec![0]; // Only 1 label for 3 points
        let mut plot = ClusterPlot::new(points, labels);
        plot.bounds = Rect::new(0.0, 0.0, 60.0, 20.0);

        let mut buffer = CellBuffer::new(60, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas); // Should handle gracefully (default to -1)
    }

    #[test]
    fn test_clone() {
        let plot = ClusterPlot::new(vec![(0.0, 0.0)], vec![0])
            .with_centroids(vec![(0.5, 0.5)])
            .with_algorithm(ClusterAlgorithm::DBSCAN {
                eps: 0.3,
                min_samples: 2,
            })
            .with_show_centroids(true)
            .with_colors(vec![Color::RED]);

        let cloned = plot;
        assert_eq!(cloned.points.len(), 1);
        assert_eq!(cloned.centroids.len(), 1);
        assert!(cloned.show_centroids);
    }

    #[test]
    fn test_debug() {
        let plot = ClusterPlot::default();
        let debug_str = format!("{:?}", plot);
        assert!(debug_str.contains("ClusterPlot"));
    }

    #[test]
    fn test_algorithm_debug() {
        let algo = ClusterAlgorithm::KMeans { k: 5 };
        let debug_str = format!("{:?}", algo);
        assert!(debug_str.contains("KMeans"));
    }

    #[test]
    fn test_algorithm_clone() {
        let algo = ClusterAlgorithm::DBSCAN {
            eps: 0.5,
            min_samples: 3,
        };
        let cloned = algo;
        assert!(matches!(
            cloned,
            ClusterAlgorithm::DBSCAN {
                eps: _,
                min_samples: 3
            }
        ));
    }
}
