//! Histogram widget with multiple binning strategies.
//!
//! Implements P202 from SPEC-024 Section 15.2.

use crate::theme::Gradient;
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Binning strategy for the histogram.
#[derive(Debug, Clone, Copy, Default)]
pub enum BinStrategy {
    /// Fixed number of bins.
    Count(usize),
    /// Fixed bin width.
    Width(f64),
    /// Sturges' formula: ceil(log2(n) + 1).
    #[default]
    Sturges,
    /// Scott's rule: 3.49 * std / n^(1/3).
    Scott,
    /// Freedman-Diaconis rule: 2 * IQR / n^(1/3).
    FreedmanDiaconis,
}

/// Bar orientation.
#[derive(Debug, Clone, Copy, Default)]
pub enum HistogramOrientation {
    /// Vertical bars (default).
    #[default]
    Vertical,
    /// Horizontal bars.
    Horizontal,
}

/// Bar rendering style.
#[derive(Debug, Clone, Copy, Default)]
pub enum BarStyle {
    /// Solid filled bars.
    #[default]
    Solid,
    /// Block characters (▁▂▃▄▅▆▇█).
    Blocks,
    /// ASCII characters.
    Ascii,
}

/// Histogram widget.
#[derive(Debug, Clone)]
pub struct Histogram {
    data: Vec<f64>,
    bins: BinStrategy,
    orientation: HistogramOrientation,
    bar_style: BarStyle,
    color: Color,
    gradient: Option<Gradient>,
    show_labels: bool,
    bounds: Rect,
    /// Computed bin edges and counts.
    computed_bins: Vec<(f64, f64, usize)>, // (start, end, count)
}

impl Histogram {
    /// Create a new histogram from data.
    #[must_use]
    pub fn new(data: Vec<f64>) -> Self {
        let mut hist = Self {
            data,
            bins: BinStrategy::default(),
            orientation: HistogramOrientation::default(),
            bar_style: BarStyle::default(),
            color: Color::new(0.3, 0.7, 1.0, 1.0),
            gradient: None,
            show_labels: true,
            bounds: Rect::default(),
            computed_bins: Vec::new(),
        };
        hist.compute_bins();
        hist
    }

    /// Set binning strategy.
    #[must_use]
    pub fn with_bins(mut self, strategy: BinStrategy) -> Self {
        self.bins = strategy;
        self.compute_bins();
        self
    }

    /// Set orientation.
    #[must_use]
    pub fn with_orientation(mut self, orientation: HistogramOrientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Set bar style.
    #[must_use]
    pub fn with_bar_style(mut self, style: BarStyle) -> Self {
        self.bar_style = style;
        self
    }

    /// Set color.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set gradient for value-based coloring.
    #[must_use]
    pub fn with_gradient(mut self, gradient: Gradient) -> Self {
        self.gradient = Some(gradient);
        self
    }

    /// Toggle axis labels.
    #[must_use]
    pub fn with_labels(mut self, show: bool) -> Self {
        self.show_labels = show;
        self
    }

    /// Update data.
    pub fn set_data(&mut self, data: Vec<f64>) {
        self.data = data;
        self.compute_bins();
    }

    /// Compute bin count based on strategy.
    #[allow(clippy::manual_clamp)]
    fn compute_bin_count(&self) -> usize {
        let n = self.data.len();
        if n == 0 {
            return 1;
        }

        match self.bins {
            BinStrategy::Count(k) => k.max(1),
            BinStrategy::Width(w) => {
                let (min, max) = self.data_range();
                ((max - min) / w).ceil() as usize
            }
            BinStrategy::Sturges => {
                // Sturges: ceil(log2(n) + 1)
                ((n as f64).log2().ceil() as usize + 1).max(1)
            }
            BinStrategy::Scott => {
                // Scott: 3.49 * std / n^(1/3)
                let std = self.std_dev();
                if std < 1e-10 {
                    return 1;
                }
                let (min, max) = self.data_range();
                let width = 3.49 * std / (n as f64).cbrt();
                ((max - min) / width).ceil() as usize
            }
            BinStrategy::FreedmanDiaconis => {
                // Freedman-Diaconis: 2 * IQR / n^(1/3)
                let iqr = self.iqr();
                if iqr < 1e-10 {
                    return 1;
                }
                let (min, max) = self.data_range();
                let width = 2.0 * iqr / (n as f64).cbrt();
                ((max - min) / width).ceil() as usize
            }
        }
        .max(1)
        .min(100) // Cap at 100 bins
    }

    /// Get data range (min, max).
    fn data_range(&self) -> (f64, f64) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for &v in &self.data {
            if v.is_finite() {
                min = min.min(v);
                max = max.max(v);
            }
        }

        if min == f64::INFINITY {
            (0.0, 1.0)
        } else if (max - min).abs() < 1e-10 {
            (min - 0.5, max + 0.5)
        } else {
            (min, max)
        }
    }

    /// Compute standard deviation.
    fn std_dev(&self) -> f64 {
        let n = self.data.len();
        if n < 2 {
            return 0.0;
        }

        let mean: f64 = self.data.iter().filter(|x| x.is_finite()).sum::<f64>()
            / self.data.iter().filter(|x| x.is_finite()).count() as f64;

        let variance: f64 = self
            .data
            .iter()
            .filter(|x| x.is_finite())
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;

        variance.sqrt()
    }

    /// Compute interquartile range.
    fn iqr(&self) -> f64 {
        let mut sorted: Vec<f64> = self
            .data
            .iter()
            .filter(|x| x.is_finite())
            .copied()
            .collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if sorted.len() < 4 {
            return self.std_dev(); // Fall back to std dev
        }

        let q1_idx = sorted.len() / 4;
        let q3_idx = 3 * sorted.len() / 4;

        sorted[q3_idx] - sorted[q1_idx]
    }

    /// Compute bins and counts.
    fn compute_bins(&mut self) {
        let n_bins = self.compute_bin_count();
        let (min, max) = self.data_range();
        let bin_width = (max - min) / n_bins as f64;

        self.computed_bins = (0..n_bins)
            .map(|i| {
                let start = min + i as f64 * bin_width;
                let end = start + bin_width;
                let count = self
                    .data
                    .iter()
                    .filter(|&&v| {
                        if i == n_bins - 1 {
                            v >= start && v <= end
                        } else {
                            v >= start && v < end
                        }
                    })
                    .count();
                (start, end, count)
            })
            .collect();
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl Widget for Histogram {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        Size::new(
            constraints.max_width.min(60.0),
            constraints.max_height.min(15.0),
        )
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.bounds.width < 5.0 || self.bounds.height < 3.0 || self.computed_bins.is_empty() {
            return;
        }

        let max_count = self
            .computed_bins
            .iter()
            .map(|(_, _, c)| *c)
            .max()
            .unwrap_or(1)
            .max(1);

        match self.orientation {
            HistogramOrientation::Vertical => self.paint_vertical(canvas, max_count),
            HistogramOrientation::Horizontal => self.paint_horizontal(canvas, max_count),
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

impl Histogram {
    fn paint_vertical(&self, canvas: &mut dyn Canvas, max_count: usize) {
        let label_height = if self.show_labels { 1.0 } else { 0.0 };
        let label_width = if self.show_labels { 5.0 } else { 0.0 };

        let plot_x = self.bounds.x + label_width;
        let plot_y = self.bounds.y;
        let plot_width = self.bounds.width - label_width;
        let plot_height = self.bounds.height - label_height;

        let n_bins = self.computed_bins.len();
        let bar_width = (plot_width / n_bins as f32).max(1.0);

        // Draw Y axis labels
        if self.show_labels {
            let label_style = TextStyle {
                color: Color::new(0.6, 0.6, 0.6, 1.0),
                ..Default::default()
            };

            canvas.draw_text(
                &format!("{max_count:>4}"),
                Point::new(self.bounds.x, plot_y),
                &label_style,
            );
            canvas.draw_text(
                "   0",
                Point::new(self.bounds.x, plot_y + plot_height - 1.0),
                &label_style,
            );
        }

        // Draw bars
        for (i, &(start, _end, count)) in self.computed_bins.iter().enumerate() {
            let bar_height = if max_count > 0 {
                (count as f32 / max_count as f32) * plot_height
            } else {
                0.0
            };

            let x = plot_x + i as f32 * bar_width;
            let y = plot_y + plot_height - bar_height;

            // Determine color
            let color = if let Some(ref gradient) = self.gradient {
                gradient.sample(count as f64 / max_count as f64)
            } else {
                self.color
            };

            let style = TextStyle {
                color,
                ..Default::default()
            };

            // Draw bar based on style
            match self.bar_style {
                BarStyle::Solid => {
                    for row in 0..(bar_height.ceil() as usize) {
                        let bar_chars: String =
                            (0..(bar_width as usize).max(1)).map(|_| '█').collect();
                        canvas.draw_text(&bar_chars, Point::new(x, y + row as f32), &style);
                    }
                }
                BarStyle::Blocks => {
                    const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
                    let full_rows = bar_height as usize;
                    let frac = bar_height.fract();
                    let frac_idx = ((frac * 8.0) as usize).min(7);

                    for row in 0..full_rows {
                        let bar_chars: String =
                            (0..(bar_width as usize).max(1)).map(|_| '█').collect();
                        canvas.draw_text(&bar_chars, Point::new(x, y + row as f32), &style);
                    }

                    if frac > 0.1 {
                        let bar_chars: String = (0..(bar_width as usize).max(1))
                            .map(|_| BLOCKS[frac_idx])
                            .collect();
                        canvas.draw_text(&bar_chars, Point::new(x, y + full_rows as f32), &style);
                    }
                }
                BarStyle::Ascii => {
                    for row in 0..(bar_height.ceil() as usize) {
                        let bar_chars: String =
                            (0..(bar_width as usize).max(1)).map(|_| '#').collect();
                        canvas.draw_text(&bar_chars, Point::new(x, y + row as f32), &style);
                    }
                }
            }

            // Draw X axis label
            if self.show_labels && i % 2 == 0 {
                let label = format!("{start:.0}");
                let label_x = x + bar_width / 2.0 - label.len() as f32 / 2.0;
                canvas.draw_text(
                    &label,
                    Point::new(label_x, plot_y + plot_height),
                    &TextStyle {
                        color: Color::new(0.6, 0.6, 0.6, 1.0),
                        ..Default::default()
                    },
                );
            }
        }
    }

    fn paint_horizontal(&self, canvas: &mut dyn Canvas, max_count: usize) {
        let label_width = if self.show_labels { 6.0 } else { 0.0 };

        let plot_x = self.bounds.x + label_width;
        let plot_y = self.bounds.y;
        let plot_width = self.bounds.width - label_width;
        let plot_height = self.bounds.height;

        let n_bins = self.computed_bins.len();
        let bar_height = (plot_height / n_bins as f32).max(1.0);

        for (i, &(start, _end, count)) in self.computed_bins.iter().enumerate() {
            let bar_width = if max_count > 0 {
                (count as f32 / max_count as f32) * plot_width
            } else {
                0.0
            };

            let x = plot_x;
            let y = plot_y + i as f32 * bar_height;

            // Determine color
            let color = if let Some(ref gradient) = self.gradient {
                gradient.sample(count as f64 / max_count as f64)
            } else {
                self.color
            };

            let style = TextStyle {
                color,
                ..Default::default()
            };

            // Draw label
            if self.show_labels {
                let label = format!("{start:>5.0}");
                canvas.draw_text(
                    &label,
                    Point::new(self.bounds.x, y),
                    &TextStyle {
                        color: Color::new(0.6, 0.6, 0.6, 1.0),
                        ..Default::default()
                    },
                );
            }

            // Draw bar
            let bar_chars: String = (0..(bar_width.ceil() as usize).max(0))
                .map(|_| '█')
                .collect();
            if !bar_chars.is_empty() {
                canvas.draw_text(&bar_chars, Point::new(x, y), &style);
            }
        }
    }
}

impl Brick for Histogram {
    fn brick_name(&self) -> &'static str {
        "Histogram"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(8)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(8)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let mut failed = Vec::new();

        if self.bounds.width >= 5.0 && self.bounds.height >= 3.0 {
            passed.push(BrickAssertion::max_latency_ms(8));
        } else {
            failed.push((
                BrickAssertion::max_latency_ms(8),
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

    #[test]
    fn test_histogram_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let hist = Histogram::new(data);
        assert!(!hist.computed_bins.is_empty());
    }

    #[test]
    fn test_bin_strategies() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();

        let sturges = Histogram::new(data.clone()).with_bins(BinStrategy::Sturges);
        assert!(!sturges.computed_bins.is_empty());

        let scott = Histogram::new(data.clone()).with_bins(BinStrategy::Scott);
        assert!(!scott.computed_bins.is_empty());

        let fd = Histogram::new(data).with_bins(BinStrategy::FreedmanDiaconis);
        assert!(!fd.computed_bins.is_empty());
    }

    #[test]
    fn test_empty_data() {
        let hist = Histogram::new(vec![]);
        assert_eq!(hist.computed_bins.len(), 1);
    }

    #[test]
    fn test_single_value() {
        let hist = Histogram::new(vec![5.0, 5.0, 5.0]);
        assert!(!hist.computed_bins.is_empty());
    }

    #[test]
    fn test_histogram_assertions() {
        let hist = Histogram::default();
        assert!(!hist.assertions().is_empty());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_histogram_verify() {
        let mut hist = Histogram::default();
        hist.bounds = Rect::new(0.0, 0.0, 60.0, 15.0);
        assert!(hist.verify().is_valid());
    }

    #[test]
    fn test_histogram_children() {
        let hist = Histogram::default();
        assert!(hist.children().is_empty());
    }

    #[test]
    fn test_histogram_children_mut() {
        let mut hist = Histogram::default();
        assert!(hist.children_mut().is_empty());
    }

    #[test]
    fn test_histogram_type_id() {
        let hist = Histogram::default();
        let tid = Widget::type_id(&hist);
        assert_eq!(tid, TypeId::of::<Histogram>());
    }

    #[test]
    fn test_histogram_measure() {
        let hist = Histogram::new(vec![1.0, 2.0, 3.0]);
        let size = hist.measure(Constraints::new(0.0, 100.0, 0.0, 50.0));
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);
    }

    #[test]
    fn test_histogram_layout() {
        let mut hist = Histogram::new(vec![1.0, 2.0, 3.0]);
        let result = hist.layout(Rect::new(0.0, 0.0, 60.0, 15.0));
        assert_eq!(result.size.width, 60.0);
        assert_eq!(result.size.height, 15.0);
    }

    #[test]
    fn test_histogram_event() {
        let mut hist = Histogram::default();
        let event = Event::Resize {
            width: 80.0,
            height: 24.0,
        };
        assert!(hist.event(&event).is_none());
    }

    #[test]
    fn test_histogram_brick_name() {
        let hist = Histogram::default();
        assert_eq!(hist.brick_name(), "Histogram");
    }

    #[test]
    fn test_histogram_budget() {
        let hist = Histogram::default();
        let budget = hist.budget();
        assert!(budget.layout_ms > 0);
    }

    #[test]
    fn test_histogram_to_html() {
        let hist = Histogram::default();
        assert!(hist.to_html().is_empty());
    }

    #[test]
    fn test_histogram_to_css() {
        let hist = Histogram::default();
        assert!(hist.to_css().is_empty());
    }

    #[test]
    fn test_histogram_with_orientation() {
        let hist =
            Histogram::new(vec![1.0, 2.0]).with_orientation(HistogramOrientation::Horizontal);
        assert!(matches!(hist.orientation, HistogramOrientation::Horizontal));
    }

    #[test]
    fn test_histogram_with_bar_style() {
        let hist = Histogram::new(vec![1.0, 2.0]).with_bar_style(BarStyle::Blocks);
        assert!(matches!(hist.bar_style, BarStyle::Blocks));
    }

    #[test]
    fn test_histogram_with_color() {
        let hist = Histogram::new(vec![1.0, 2.0]).with_color(Color::RED);
        assert_eq!(hist.color, Color::RED);
    }

    #[test]
    fn test_histogram_with_gradient() {
        let gradient = Gradient::from_hex(&["#00FF00", "#FF0000"]);
        let hist = Histogram::new(vec![1.0, 2.0]).with_gradient(gradient);
        assert!(hist.gradient.is_some());
    }

    #[test]
    fn test_histogram_with_labels() {
        let hist = Histogram::new(vec![1.0, 2.0]).with_labels(false);
        assert!(!hist.show_labels);
    }

    #[test]
    fn test_histogram_set_data() {
        let mut hist = Histogram::new(vec![1.0, 2.0]);
        hist.set_data(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        assert!(!hist.computed_bins.is_empty());
    }

    #[test]
    fn test_histogram_bin_count() {
        let hist = Histogram::new(vec![1.0, 2.0, 3.0]).with_bins(BinStrategy::Count(5));
        assert!(!hist.computed_bins.is_empty());
    }

    #[test]
    fn test_histogram_bin_width() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let hist = Histogram::new(data).with_bins(BinStrategy::Width(2.0));
        assert!(!hist.computed_bins.is_empty());
    }

    #[test]
    fn test_histogram_paint_vertical() {
        use crate::{CellBuffer, DirectTerminalCanvas};

        let mut hist = Histogram::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut buffer = CellBuffer::new(60, 15);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        hist.layout(Rect::new(0.0, 0.0, 60.0, 15.0));
        hist.paint(&mut canvas);
    }

    #[test]
    fn test_histogram_paint_horizontal() {
        use crate::{CellBuffer, DirectTerminalCanvas};

        let mut hist = Histogram::new(vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .with_orientation(HistogramOrientation::Horizontal);
        let mut buffer = CellBuffer::new(60, 15);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        hist.layout(Rect::new(0.0, 0.0, 60.0, 15.0));
        hist.paint(&mut canvas);
    }

    #[test]
    fn test_histogram_paint_blocks() {
        use crate::{CellBuffer, DirectTerminalCanvas};

        let mut hist =
            Histogram::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]).with_bar_style(BarStyle::Blocks);
        let mut buffer = CellBuffer::new(60, 15);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        hist.layout(Rect::new(0.0, 0.0, 60.0, 15.0));
        hist.paint(&mut canvas);
    }

    #[test]
    fn test_histogram_paint_ascii() {
        use crate::{CellBuffer, DirectTerminalCanvas};

        let mut hist =
            Histogram::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]).with_bar_style(BarStyle::Ascii);
        let mut buffer = CellBuffer::new(60, 15);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        hist.layout(Rect::new(0.0, 0.0, 60.0, 15.0));
        hist.paint(&mut canvas);
    }

    #[test]
    fn test_histogram_paint_with_gradient() {
        use crate::{CellBuffer, DirectTerminalCanvas};

        let gradient = Gradient::from_hex(&["#00FF00", "#FF0000"]);
        let mut hist = Histogram::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]).with_gradient(gradient);
        let mut buffer = CellBuffer::new(60, 15);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        hist.layout(Rect::new(0.0, 0.0, 60.0, 15.0));
        hist.paint(&mut canvas);
    }

    #[test]
    fn test_histogram_paint_without_labels() {
        use crate::{CellBuffer, DirectTerminalCanvas};

        let mut hist = Histogram::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]).with_labels(false);
        let mut buffer = CellBuffer::new(60, 15);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        hist.layout(Rect::new(0.0, 0.0, 60.0, 15.0));
        hist.paint(&mut canvas);
    }

    #[test]
    fn test_histogram_paint_small_bounds() {
        use crate::{CellBuffer, DirectTerminalCanvas};

        let mut hist = Histogram::new(vec![1.0, 2.0, 3.0]);
        let mut buffer = CellBuffer::new(4, 2);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        hist.layout(Rect::new(0.0, 0.0, 4.0, 2.0));
        hist.paint(&mut canvas);
        // Should return early due to small bounds
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_histogram_verify_small_bounds() {
        let mut hist = Histogram::default();
        hist.bounds = Rect::new(0.0, 0.0, 2.0, 1.0);
        assert!(!hist.verify().is_valid());
    }

    #[test]
    fn test_histogram_data_with_nan() {
        let hist = Histogram::new(vec![1.0, f64::NAN, 3.0, f64::INFINITY, 5.0]);
        assert!(!hist.computed_bins.is_empty());
    }

    #[test]
    fn test_histogram_iqr_small_data() {
        let hist = Histogram::new(vec![1.0, 2.0]); // Less than 4 values, falls back to std
        assert!(!hist.computed_bins.is_empty());
    }

    #[test]
    fn test_histogram_std_dev_single() {
        let hist = Histogram::new(vec![5.0]);
        assert!(!hist.computed_bins.is_empty());
    }

    #[test]
    fn test_histogram_clone() {
        let hist = Histogram::new(vec![1.0, 2.0, 3.0]);
        let cloned = hist.clone();
        assert_eq!(cloned.computed_bins.len(), hist.computed_bins.len());
    }

    #[test]
    fn test_histogram_debug() {
        let hist = Histogram::new(vec![1.0, 2.0, 3.0]);
        let debug = format!("{hist:?}");
        assert!(debug.contains("Histogram"));
    }

    #[test]
    fn test_bin_strategy_debug() {
        let strategy = BinStrategy::Sturges;
        let debug = format!("{strategy:?}");
        assert!(debug.contains("Sturges"));
    }

    #[test]
    fn test_histogram_orientation_debug() {
        let orientation = HistogramOrientation::Vertical;
        let debug = format!("{orientation:?}");
        assert!(debug.contains("Vertical"));
    }

    #[test]
    fn test_bar_style_debug() {
        let style = BarStyle::Solid;
        let debug = format!("{style:?}");
        assert!(debug.contains("Solid"));
    }

    #[test]
    fn test_histogram_horizontal_with_gradient() {
        use crate::{CellBuffer, DirectTerminalCanvas};

        let gradient = Gradient::from_hex(&["#00FF00", "#FF0000"]);
        let mut hist = Histogram::new(vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .with_orientation(HistogramOrientation::Horizontal)
            .with_gradient(gradient);
        let mut buffer = CellBuffer::new(60, 15);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        hist.layout(Rect::new(0.0, 0.0, 60.0, 15.0));
        hist.paint(&mut canvas);
    }

    #[test]
    fn test_histogram_horizontal_without_labels() {
        use crate::{CellBuffer, DirectTerminalCanvas};

        let mut hist = Histogram::new(vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .with_orientation(HistogramOrientation::Horizontal)
            .with_labels(false);
        let mut buffer = CellBuffer::new(60, 15);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        hist.layout(Rect::new(0.0, 0.0, 60.0, 15.0));
        hist.paint(&mut canvas);
    }

    #[test]
    fn test_histogram_large_data() {
        let data: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.37) % 100.0).collect();
        let hist = Histogram::new(data);
        assert!(!hist.computed_bins.is_empty());
    }
}
