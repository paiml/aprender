//! Histogram widget with multiple binning strategies.
//!
//! ## Binning Strategies
//! - **Sturges (1926)**: k = ceil(log2(n) + 1) - good for small datasets
//! - **Scott (1979)**: h = 3.49 * std / n^(1/3) - optimal for normal distributions
//! - **Freedman-Diaconis (1981)**: h = 2 * IQR / n^(1/3) - robust to outliers

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::Widget,
};

/// Binning strategy for the histogram.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum HistogramOrientation {
    /// Vertical bars (default).
    #[default]
    Vertical,
    /// Horizontal bars.
    Horizontal,
}

/// Bar rendering style.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BarStyle {
    /// Solid filled bars (full blocks).
    #[default]
    Solid,
    /// Block characters for sub-cell resolution (▁▂▃▄▅▆▇█).
    Blocks,
    /// ASCII characters (#).
    Ascii,
}

/// A computed histogram bin.
#[derive(Debug, Clone, PartialEq)]
pub struct Bin {
    /// Start of bin range (inclusive).
    pub start: f64,
    /// End of bin range (exclusive, except last bin).
    pub end: f64,
    /// Count of values in this bin.
    pub count: usize,
}

/// Histogram widget.
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Raw data values.
    data: Vec<f64>,
    /// Binning strategy.
    bin_strategy: BinStrategy,
    /// Bar orientation.
    orientation: HistogramOrientation,
    /// Bar rendering style.
    bar_style: BarStyle,
    /// Bar color.
    color: Color,
    /// Show axis labels.
    show_labels: bool,
    /// Optional title.
    title: Option<String>,
    /// Computed bins (cached).
    computed_bins: Vec<Bin>,
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl Histogram {
    /// Create a new histogram from data.
    #[must_use]
    pub fn new(data: Vec<f64>) -> Self {
        let mut hist = Self {
            data,
            bin_strategy: BinStrategy::default(),
            orientation: HistogramOrientation::default(),
            bar_style: BarStyle::default(),
            color: Color::Cyan,
            show_labels: true,
            title: None,
            computed_bins: Vec::new(),
        };
        hist.compute_bins();
        hist
    }

    /// Set binning strategy.
    #[must_use]
    pub fn bin_strategy(mut self, strategy: BinStrategy) -> Self {
        self.bin_strategy = strategy;
        self.compute_bins();
        self
    }

    /// Set orientation.
    #[must_use]
    pub fn orientation(mut self, orientation: HistogramOrientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Set bar style.
    #[must_use]
    pub fn bar_style(mut self, style: BarStyle) -> Self {
        self.bar_style = style;
        self
    }

    /// Set bar color.
    #[must_use]
    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Toggle axis labels.
    #[must_use]
    pub fn show_labels(mut self, show: bool) -> Self {
        self.show_labels = show;
        self
    }

    /// Set title.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Update data.
    pub fn set_data(&mut self, data: Vec<f64>) {
        self.data = data;
        self.compute_bins();
    }

    /// Get computed bins.
    #[must_use]
    pub fn bins(&self) -> &[Bin] {
        &self.computed_bins
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
        let finite_data: Vec<f64> = self.data.iter().filter(|x| x.is_finite()).copied().collect();
        let n = finite_data.len();
        if n < 2 {
            return 0.0;
        }

        let mean: f64 = finite_data.iter().sum::<f64>() / n as f64;
        let variance: f64 =
            finite_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        variance.sqrt()
    }

    /// Compute interquartile range.
    fn iqr(&self) -> f64 {
        let mut sorted: Vec<f64> = self.data.iter().filter(|x| x.is_finite()).copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if sorted.len() < 4 {
            return self.std_dev(); // Fall back to std dev
        }

        let q1_idx = sorted.len() / 4;
        let q3_idx = 3 * sorted.len() / 4;
        sorted[q3_idx] - sorted[q1_idx]
    }

    /// Compute bin count based on strategy.
    fn compute_bin_count(&self) -> usize {
        let n = self.data.iter().filter(|x| x.is_finite()).count();
        if n == 0 {
            return 1;
        }

        let count = match self.bin_strategy {
            BinStrategy::Count(k) => k.max(1),
            BinStrategy::Width(w) => {
                if w <= 0.0 {
                    return 1;
                }
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
                if width < 1e-10 {
                    return 1;
                }
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
                if width < 1e-10 {
                    return 1;
                }
                ((max - min) / width).ceil() as usize
            }
        };

        count.clamp(1, 100) // Cap at 100 bins
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
                    .filter(|&&v| v.is_finite())
                    .filter(|&&v| {
                        if i == n_bins - 1 {
                            v >= start && v <= end
                        } else {
                            v >= start && v < end
                        }
                    })
                    .count();
                Bin { start, end, count }
            })
            .collect();
    }

    /// Render vertical histogram.
    fn render_vertical(&self, area: Rect, buf: &mut Buffer) {
        if self.computed_bins.is_empty() || area.width < 3 || area.height < 2 {
            return;
        }

        let max_count = self.computed_bins.iter().map(|b| b.count).max().unwrap_or(1).max(1);
        let n_bins = self.computed_bins.len();

        // Calculate layout
        let label_width = if self.show_labels { 5 } else { 0 };
        let label_height = u16::from(self.show_labels);

        let plot_x = area.x + label_width;
        let plot_y = area.y;
        let plot_width = area.width.saturating_sub(label_width);
        let plot_height = area.height.saturating_sub(label_height);

        if plot_width == 0 || plot_height == 0 {
            return;
        }

        let bar_width = (plot_width as usize / n_bins).max(1);
        let style = Style::default().fg(self.color);

        // Draw Y axis labels
        if self.show_labels {
            let max_label = format!("{max_count:>4}");
            for (i, ch) in max_label.chars().enumerate() {
                if let Some(cell) = buf.cell_mut((area.x + i as u16, plot_y)) {
                    cell.set_char(ch).set_style(Style::default().fg(Color::DarkGray));
                }
            }
        }

        // Draw bars
        for (i, bin) in self.computed_bins.iter().enumerate() {
            let bar_height_f = if max_count > 0 {
                (bin.count as f64 / max_count as f64) * f64::from(plot_height)
            } else {
                0.0
            };

            let x = plot_x + (i * bar_width) as u16;
            let bar_height = bar_height_f.ceil() as u16;

            // Draw bar cells
            for row in 0..bar_height {
                let y = plot_y + plot_height - 1 - row;
                if y < plot_y + plot_height {
                    let ch = match self.bar_style {
                        BarStyle::Solid => '█',
                        BarStyle::Blocks => {
                            if row == bar_height - 1 {
                                // Partial fill for top of bar
                                const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
                                let frac = bar_height_f.fract();
                                let idx = ((frac * 8.0) as usize).min(7);
                                BLOCKS[idx]
                            } else {
                                '█'
                            }
                        }
                        BarStyle::Ascii => '#',
                    };

                    for col in 0..bar_width {
                        let cell_x = x + col as u16;
                        if cell_x < area.x + area.width {
                            if let Some(cell) = buf.cell_mut((cell_x, y)) {
                                cell.set_char(ch).set_style(style);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Render horizontal histogram.
    fn render_horizontal(&self, area: Rect, buf: &mut Buffer) {
        if self.computed_bins.is_empty() || area.width < 3 || area.height < 2 {
            return;
        }

        let max_count = self.computed_bins.iter().map(|b| b.count).max().unwrap_or(1).max(1);
        let n_bins = self.computed_bins.len();

        // Calculate layout
        let label_width = if self.show_labels { 6 } else { 0 };

        let plot_x = area.x + label_width;
        let plot_width = area.width.saturating_sub(label_width);
        let bar_height = (area.height as usize / n_bins).max(1);

        if plot_width == 0 {
            return;
        }

        let style = Style::default().fg(self.color);

        // Draw bars
        for (i, bin) in self.computed_bins.iter().enumerate() {
            let bar_width_f = if max_count > 0 {
                (bin.count as f64 / max_count as f64) * f64::from(plot_width)
            } else {
                0.0
            };

            let y = area.y + (i * bar_height) as u16;
            let bar_width = bar_width_f.ceil() as u16;

            // Draw label
            if self.show_labels && y < area.y + area.height {
                let label = format!("{:>5.0}", bin.start);
                for (j, ch) in label.chars().enumerate() {
                    let lx = area.x + j as u16;
                    if lx < area.x + area.width {
                        if let Some(cell) = buf.cell_mut((lx, y)) {
                            cell.set_char(ch).set_style(Style::default().fg(Color::DarkGray));
                        }
                    }
                }
            }

            // Draw bar
            let ch = match self.bar_style {
                BarStyle::Solid | BarStyle::Blocks => '█',
                BarStyle::Ascii => '#',
            };

            for col in 0..bar_width {
                let x = plot_x + col;
                if x < area.x + area.width && y < area.y + area.height {
                    if let Some(cell) = buf.cell_mut((x, y)) {
                        cell.set_char(ch).set_style(style);
                    }
                }
            }
        }
    }
}

impl Widget for Histogram {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width == 0 || area.height == 0 {
            return;
        }

        // Draw title if set
        let plot_area = if let Some(ref title) = self.title {
            let style = Style::default().fg(Color::White);
            for (i, ch) in title.chars().enumerate() {
                let x = area.x + i as u16;
                if x < area.x + area.width {
                    if let Some(cell) = buf.cell_mut((x, area.y)) {
                        cell.set_char(ch).set_style(style);
                    }
                }
            }
            Rect {
                x: area.x,
                y: area.y + 1,
                width: area.width,
                height: area.height.saturating_sub(1),
            }
        } else {
            area
        };

        match self.orientation {
            HistogramOrientation::Vertical => self.render_vertical(plot_area, buf),
            HistogramOrientation::Horizontal => self.render_horizontal(plot_area, buf),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod bin_strategy_tests {
        use super::*;

        #[test]
        fn test_default() {
            assert_eq!(BinStrategy::default(), BinStrategy::Sturges);
        }

        #[test]
        fn test_equality() {
            assert_eq!(BinStrategy::Count(5), BinStrategy::Count(5));
            assert_ne!(BinStrategy::Count(5), BinStrategy::Count(10));
            assert_eq!(BinStrategy::Sturges, BinStrategy::Sturges);
        }

        #[test]
        fn test_debug() {
            let s = format!("{:?}", BinStrategy::Scott);
            assert!(s.contains("Scott"));
        }
    }

    mod orientation_tests {
        use super::*;

        #[test]
        fn test_default() {
            assert_eq!(HistogramOrientation::default(), HistogramOrientation::Vertical);
        }

        #[test]
        fn test_equality() {
            assert_eq!(HistogramOrientation::Vertical, HistogramOrientation::Vertical);
            assert_ne!(HistogramOrientation::Vertical, HistogramOrientation::Horizontal);
        }
    }

    mod bar_style_tests {
        use super::*;

        #[test]
        fn test_default() {
            assert_eq!(BarStyle::default(), BarStyle::Solid);
        }

        #[test]
        fn test_equality() {
            assert_eq!(BarStyle::Blocks, BarStyle::Blocks);
            assert_ne!(BarStyle::Solid, BarStyle::Ascii);
        }
    }

    mod histogram_construction_tests {
        use super::*;

        #[test]
        fn test_new() {
            let hist = Histogram::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            assert!(!hist.computed_bins.is_empty());
        }

        #[test]
        fn test_default() {
            let hist = Histogram::default();
            assert_eq!(hist.computed_bins.len(), 1);
        }

        #[test]
        fn test_bin_strategy() {
            let hist = Histogram::new(vec![1.0, 2.0, 3.0]).bin_strategy(BinStrategy::Count(5));
            assert_eq!(hist.bin_strategy, BinStrategy::Count(5));
        }

        #[test]
        fn test_orientation() {
            let hist = Histogram::new(vec![1.0]).orientation(HistogramOrientation::Horizontal);
            assert_eq!(hist.orientation, HistogramOrientation::Horizontal);
        }

        #[test]
        fn test_bar_style() {
            let hist = Histogram::new(vec![1.0]).bar_style(BarStyle::Blocks);
            assert_eq!(hist.bar_style, BarStyle::Blocks);
        }

        #[test]
        fn test_color() {
            let hist = Histogram::new(vec![1.0]).color(Color::Red);
            assert_eq!(hist.color, Color::Red);
        }

        #[test]
        fn test_show_labels() {
            let hist = Histogram::new(vec![1.0]).show_labels(false);
            assert!(!hist.show_labels);
        }

        #[test]
        fn test_title() {
            let hist = Histogram::new(vec![1.0]).title("My Histogram");
            assert_eq!(hist.title.as_deref(), Some("My Histogram"));
        }

        #[test]
        fn test_set_data() {
            let mut hist = Histogram::new(vec![1.0]);
            hist.set_data(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            assert!(!hist.computed_bins.is_empty());
        }

        #[test]
        fn test_bins() {
            let hist = Histogram::new(vec![1.0, 2.0, 3.0]);
            assert!(!hist.bins().is_empty());
        }

        #[test]
        fn test_builder_chaining() {
            let hist = Histogram::new(vec![1.0, 2.0, 3.0])
                .bin_strategy(BinStrategy::Scott)
                .orientation(HistogramOrientation::Horizontal)
                .bar_style(BarStyle::Ascii)
                .color(Color::Green)
                .show_labels(false)
                .title("Test");

            assert_eq!(hist.bin_strategy, BinStrategy::Scott);
            assert_eq!(hist.orientation, HistogramOrientation::Horizontal);
            assert_eq!(hist.bar_style, BarStyle::Ascii);
        }
    }

    mod binning_tests {
        use super::*;

        #[test]
        fn test_sturges() {
            let data: Vec<f64> = (0..100).map(f64::from).collect();
            let hist = Histogram::new(data).bin_strategy(BinStrategy::Sturges);
            // Sturges for 100 elements: ceil(log2(100) + 1) ≈ 8
            assert!(!hist.computed_bins.is_empty());
            assert!(hist.computed_bins.len() <= 20);
        }

        #[test]
        fn test_scott() {
            let data: Vec<f64> = (0..100).map(f64::from).collect();
            let hist = Histogram::new(data).bin_strategy(BinStrategy::Scott);
            assert!(!hist.computed_bins.is_empty());
        }

        #[test]
        fn test_freedman_diaconis() {
            let data: Vec<f64> = (0..100).map(f64::from).collect();
            let hist = Histogram::new(data).bin_strategy(BinStrategy::FreedmanDiaconis);
            assert!(!hist.computed_bins.is_empty());
        }

        #[test]
        fn test_count() {
            let hist =
                Histogram::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]).bin_strategy(BinStrategy::Count(3));
            // With count strategy, we might get close to 3 bins
            assert!(!hist.computed_bins.is_empty());
        }

        #[test]
        fn test_width() {
            let data: Vec<f64> = (0..10).map(f64::from).collect();
            let hist = Histogram::new(data).bin_strategy(BinStrategy::Width(2.0));
            assert!(!hist.computed_bins.is_empty());
        }

        #[test]
        fn test_width_zero() {
            let hist = Histogram::new(vec![1.0, 2.0]).bin_strategy(BinStrategy::Width(0.0));
            assert_eq!(hist.computed_bins.len(), 1);
        }

        #[test]
        fn test_empty_data() {
            let hist = Histogram::new(vec![]);
            assert_eq!(hist.computed_bins.len(), 1);
        }

        #[test]
        fn test_single_value() {
            let hist = Histogram::new(vec![5.0]);
            assert!(!hist.computed_bins.is_empty());
        }

        #[test]
        fn test_same_values() {
            let hist = Histogram::new(vec![5.0, 5.0, 5.0, 5.0]);
            assert!(!hist.computed_bins.is_empty());
        }

        #[test]
        fn test_with_nan() {
            let hist = Histogram::new(vec![1.0, f64::NAN, 3.0, f64::INFINITY, 5.0]);
            assert!(!hist.computed_bins.is_empty());
        }

        #[test]
        fn test_large_data() {
            let data: Vec<f64> = (0..1000).map(|i| (f64::from(i) * 0.37) % 100.0).collect();
            let hist = Histogram::new(data);
            // Should cap at 100 bins
            assert!(hist.computed_bins.len() <= 100);
        }
    }

    mod statistics_tests {
        use super::*;

        #[test]
        fn test_data_range_normal() {
            let hist = Histogram::new(vec![1.0, 5.0, 10.0]);
            let (min, max) = hist.data_range();
            assert_eq!(min, 1.0);
            assert_eq!(max, 10.0);
        }

        #[test]
        fn test_data_range_empty() {
            let hist = Histogram::new(vec![]);
            let (min, max) = hist.data_range();
            assert_eq!(min, 0.0);
            assert_eq!(max, 1.0);
        }

        #[test]
        fn test_data_range_same_values() {
            let hist = Histogram::new(vec![5.0, 5.0, 5.0]);
            let (min, max) = hist.data_range();
            assert!(min < 5.0);
            assert!(max > 5.0);
        }

        #[test]
        fn test_std_dev_normal() {
            let hist = Histogram::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            let std = hist.std_dev();
            // std dev of 1,2,3,4,5 is sqrt(2.5) ≈ 1.58
            assert!(std > 1.0 && std < 2.0);
        }

        #[test]
        fn test_std_dev_single() {
            let hist = Histogram::new(vec![5.0]);
            assert_eq!(hist.std_dev(), 0.0);
        }

        #[test]
        fn test_iqr_normal() {
            let data: Vec<f64> = (0..100).map(f64::from).collect();
            let hist = Histogram::new(data);
            let iqr = hist.iqr();
            // IQR of 0-99 is approximately 50
            assert!(iqr > 40.0 && iqr < 60.0);
        }

        #[test]
        fn test_iqr_small() {
            let hist = Histogram::new(vec![1.0, 2.0]);
            // Falls back to std dev
            let iqr = hist.iqr();
            assert!(iqr >= 0.0);
        }
    }

    mod bin_tests {
        use super::*;

        #[test]
        fn test_bin_struct() {
            let bin = Bin { start: 0.0, end: 10.0, count: 5 };
            assert_eq!(bin.start, 0.0);
            assert_eq!(bin.end, 10.0);
            assert_eq!(bin.count, 5);
        }

        #[test]
        fn test_bin_equality() {
            let a = Bin { start: 0.0, end: 1.0, count: 5 };
            let b = Bin { start: 0.0, end: 1.0, count: 5 };
            assert_eq!(a, b);
        }

        #[test]
        fn test_bin_debug() {
            let bin = Bin { start: 0.0, end: 1.0, count: 5 };
            let debug = format!("{bin:?}");
            assert!(debug.contains("Bin"));
        }
    }

    mod rendering_tests {
        use super::*;

        fn create_test_buffer(width: u16, height: u16) -> (Rect, Buffer) {
            let area = Rect::new(0, 0, width, height);
            let buf = Buffer::empty(area);
            (area, buf)
        }

        #[test]
        fn test_render_empty_area() {
            let area = Rect::new(0, 0, 0, 0);
            let mut buf = Buffer::empty(Rect::new(0, 0, 1, 1));
            let hist = Histogram::new(vec![1.0, 2.0, 3.0]);
            hist.render(area, &mut buf);
        }

        #[test]
        fn test_render_vertical() {
            let (area, mut buf) = create_test_buffer(60, 20);
            let hist = Histogram::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            hist.render(area, &mut buf);
        }

        #[test]
        fn test_render_horizontal() {
            let (area, mut buf) = create_test_buffer(60, 20);
            let hist = Histogram::new(vec![1.0, 2.0, 3.0, 4.0, 5.0])
                .orientation(HistogramOrientation::Horizontal);
            hist.render(area, &mut buf);
        }

        #[test]
        fn test_render_with_title() {
            let (area, mut buf) = create_test_buffer(60, 20);
            let hist = Histogram::new(vec![1.0, 2.0, 3.0]).title("Test Histogram");
            hist.render(area, &mut buf);
        }

        #[test]
        fn test_render_no_labels() {
            let (area, mut buf) = create_test_buffer(60, 20);
            let hist = Histogram::new(vec![1.0, 2.0, 3.0]).show_labels(false);
            hist.render(area, &mut buf);
        }

        #[test]
        fn test_render_blocks_style() {
            let (area, mut buf) = create_test_buffer(60, 20);
            let hist = Histogram::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]).bar_style(BarStyle::Blocks);
            hist.render(area, &mut buf);
        }

        #[test]
        fn test_render_ascii_style() {
            let (area, mut buf) = create_test_buffer(60, 20);
            let hist = Histogram::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]).bar_style(BarStyle::Ascii);
            hist.render(area, &mut buf);
        }

        #[test]
        fn test_render_small_area() {
            let (area, mut buf) = create_test_buffer(5, 3);
            let hist = Histogram::new(vec![1.0, 2.0, 3.0]);
            hist.render(area, &mut buf);
        }

        #[test]
        fn test_render_large_data() {
            let (area, mut buf) = create_test_buffer(80, 30);
            let data: Vec<f64> = (0..500).map(|i| f64::from(i).sin() * 100.0).collect();
            let hist = Histogram::new(data);
            hist.render(area, &mut buf);
        }

        #[test]
        fn test_render_horizontal_no_labels() {
            let (area, mut buf) = create_test_buffer(60, 20);
            let hist = Histogram::new(vec![1.0, 2.0, 3.0])
                .orientation(HistogramOrientation::Horizontal)
                .show_labels(false);
            hist.render(area, &mut buf);
        }

        #[test]
        fn test_render_all_strategies() {
            let (area, mut buf) = create_test_buffer(60, 20);
            let data: Vec<f64> = (0..50).map(f64::from).collect();

            for strategy in [
                BinStrategy::Sturges,
                BinStrategy::Scott,
                BinStrategy::FreedmanDiaconis,
                BinStrategy::Count(10),
                BinStrategy::Width(5.0),
            ] {
                let hist = Histogram::new(data.clone()).bin_strategy(strategy);
                hist.render(area, &mut buf);
            }
        }

        #[test]
        fn test_render_vertical_narrow() {
            // Area too narrow for vertical (width < 3)
            let area = Rect::new(0, 0, 2, 10);
            let mut buf = Buffer::empty(Rect::new(0, 0, 10, 10));
            let hist = Histogram::new(vec![1.0, 2.0, 3.0]);
            hist.render(area, &mut buf);
        }

        #[test]
        fn test_render_horizontal_short() {
            // Area too short for horizontal (height < 3)
            let area = Rect::new(0, 0, 20, 2);
            let mut buf = Buffer::empty(Rect::new(0, 0, 30, 10));
            let hist =
                Histogram::new(vec![1.0, 2.0, 3.0]).orientation(HistogramOrientation::Horizontal);
            hist.render(area, &mut buf);
        }

        #[test]
        fn test_render_horizontal_blocks_style() {
            let (area, mut buf) = create_test_buffer(60, 20);
            let hist = Histogram::new(vec![1.0, 2.0, 3.0, 4.0, 5.0])
                .orientation(HistogramOrientation::Horizontal)
                .bar_style(BarStyle::Blocks);
            hist.render(area, &mut buf);
        }

        #[test]
        fn test_render_horizontal_ascii_style() {
            let (area, mut buf) = create_test_buffer(60, 20);
            let hist = Histogram::new(vec![1.0, 2.0, 3.0, 4.0, 5.0])
                .orientation(HistogramOrientation::Horizontal)
                .bar_style(BarStyle::Ascii);
            hist.render(area, &mut buf);
        }

        #[test]
        fn test_bin_count_scott_constant_data() {
            // Constant data has std dev = 0, should return 1 bin
            let hist =
                Histogram::new(vec![5.0, 5.0, 5.0, 5.0, 5.0]).bin_strategy(BinStrategy::Scott);
            assert!(!hist.computed_bins.is_empty());
        }

        #[test]
        fn test_bin_count_fd_constant_data() {
            // Constant data has IQR = 0, should return 1 bin
            let hist = Histogram::new(vec![5.0, 5.0, 5.0, 5.0, 5.0])
                .bin_strategy(BinStrategy::FreedmanDiaconis);
            assert!(!hist.computed_bins.is_empty());
        }

        #[test]
        fn test_render_zero_count_bin() {
            // Data that creates bins with zero count
            let data = vec![0.0, 0.0, 100.0, 100.0];
            let (area, mut buf) = create_test_buffer(40, 15);
            let hist = Histogram::new(data).bin_strategy(BinStrategy::Count(10));
            hist.render(area, &mut buf);
        }

        #[test]
        fn test_render_vertical_with_labels() {
            let (area, mut buf) = create_test_buffer(50, 20);
            let hist = Histogram::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]).show_labels(true);
            hist.render(area, &mut buf);
        }

        #[test]
        fn test_render_horizontal_with_labels() {
            let (area, mut buf) = create_test_buffer(50, 20);
            let hist = Histogram::new(vec![1.0, 2.0, 3.0, 4.0, 5.0])
                .orientation(HistogramOrientation::Horizontal)
                .show_labels(true);
            hist.render(area, &mut buf);
        }
    }
}
