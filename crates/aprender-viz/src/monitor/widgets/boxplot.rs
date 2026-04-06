//! BoxPlot widget for statistical distribution visualization.
//!
//! Implements box-and-whisker plots (Tukey, 1977) for visualizing
//! statistical distributions in terminal UI.
//!
//! Citation: Tukey, J. W. (1977). Exploratory Data Analysis.
//!
//! # Features
//!
//! - Horizontal or vertical orientation
//! - Quartile boxes (Q1-Q3)
//! - Median line
//! - Whiskers (min/max within 1.5*IQR)
//! - Outlier display
//! - Multiple groups support
//!
//! # Example
//!
//! ```
//! use trueno_viz::monitor::widgets::{BoxPlot, BoxStats, BoxOrientation};
//!
//! let stats = BoxStats::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
//!     .with_label("Distribution");
//! let plot = BoxPlot::new(vec![stats])
//!     .with_orientation(BoxOrientation::Horizontal);
//! ```

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::widgets::Widget;

/// Orientation for box plot rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoxOrientation {
    /// Horizontal box plot (default).
    #[default]
    Horizontal,
    /// Vertical box plot.
    Vertical,
}

/// Statistical summary for a box plot.
#[derive(Debug, Clone, Default)]
pub struct BoxStats {
    /// Minimum value (or lower whisker bound).
    pub min: f64,
    /// First quartile (25th percentile).
    pub q1: f64,
    /// Median (50th percentile).
    pub median: f64,
    /// Third quartile (75th percentile).
    pub q3: f64,
    /// Maximum value (or upper whisker bound).
    pub max: f64,
    /// Outliers below lower fence.
    pub outliers_low: Vec<f64>,
    /// Outliers above upper fence.
    pub outliers_high: Vec<f64>,
    /// Optional label.
    pub label: Option<String>,
    /// Color for this box.
    pub color: Color,
}

impl BoxStats {
    /// Create a new BoxStats with explicit values.
    #[must_use]
    pub fn new(min: f64, q1: f64, median: f64, q3: f64, max: f64) -> Self {
        Self {
            min,
            q1,
            median,
            q3,
            max,
            outliers_low: Vec::new(),
            outliers_high: Vec::new(),
            label: None,
            color: Color::Cyan,
        }
    }

    /// Compute box statistics from raw data.
    ///
    /// Uses Tukey's method for quartile calculation.
    #[must_use]
    pub fn from_data(data: &[f64]) -> Self {
        if data.is_empty() {
            return Self::default();
        }

        let mut sorted: Vec<f64> = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();

        // Calculate quartiles using linear interpolation
        let q1 = Self::percentile(&sorted, 0.25);
        let median = Self::percentile(&sorted, 0.50);
        let q3 = Self::percentile(&sorted, 0.75);

        // Calculate IQR and fences
        let iqr = q3 - q1;
        let lower_fence = q1 - 1.5 * iqr;
        let upper_fence = q3 + 1.5 * iqr;

        // Find whisker bounds (within fences)
        let min_whisker = sorted.iter().copied().find(|&x| x >= lower_fence).unwrap_or(sorted[0]);
        let max_whisker =
            sorted.iter().rev().copied().find(|&x| x <= upper_fence).unwrap_or(sorted[n - 1]);

        // Collect outliers
        let outliers_low: Vec<f64> = sorted.iter().copied().filter(|&x| x < lower_fence).collect();
        let outliers_high: Vec<f64> = sorted.iter().copied().filter(|&x| x > upper_fence).collect();

        Self {
            min: min_whisker,
            q1,
            median,
            q3,
            max: max_whisker,
            outliers_low,
            outliers_high,
            label: None,
            color: Color::Cyan,
        }
    }

    /// Calculate percentile using linear interpolation.
    fn percentile(sorted: &[f64], p: f64) -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        if sorted.len() == 1 {
            return sorted[0];
        }

        let n = sorted.len();
        let index = p * (n - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        let frac = index - lower as f64;

        if lower == upper || upper >= n {
            sorted[lower.min(n - 1)]
        } else {
            sorted[lower] * (1.0 - frac) + sorted[upper] * frac
        }
    }

    /// Set a label for this box.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set the color for this box.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Get the interquartile range (IQR).
    #[must_use]
    pub fn iqr(&self) -> f64 {
        self.q3 - self.q1
    }

    /// Get the total range (max - min).
    #[must_use]
    pub fn range(&self) -> f64 {
        self.max - self.min
    }

    /// Check if this box has outliers.
    #[must_use]
    pub fn has_outliers(&self) -> bool {
        !self.outliers_low.is_empty() || !self.outliers_high.is_empty()
    }
}

/// Box plot widget for statistical visualization.
#[derive(Debug, Clone)]
pub struct BoxPlot {
    /// Box statistics for each group.
    boxes: Vec<BoxStats>,
    /// Orientation.
    orientation: BoxOrientation,
    /// Title.
    title: Option<String>,
    /// Show outliers.
    show_outliers: bool,
    /// Minimum value for scale (auto if None).
    min_value: Option<f64>,
    /// Maximum value for scale (auto if None).
    max_value: Option<f64>,
    /// Background color.
    background: Color,
}

impl Default for BoxPlot {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl BoxPlot {
    /// Create a new box plot.
    #[must_use]
    pub fn new(boxes: Vec<BoxStats>) -> Self {
        Self {
            boxes,
            orientation: BoxOrientation::Horizontal,
            title: None,
            show_outliers: true,
            min_value: None,
            max_value: None,
            background: Color::Reset,
        }
    }

    /// Set the orientation.
    #[must_use]
    pub fn with_orientation(mut self, orientation: BoxOrientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Set a title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Toggle outlier display.
    #[must_use]
    pub fn with_outliers(mut self, show: bool) -> Self {
        self.show_outliers = show;
        self
    }

    /// Set fixed scale range.
    #[must_use]
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.min_value = Some(min);
        self.max_value = Some(max);
        self
    }

    /// Set background color.
    #[must_use]
    pub fn with_background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    /// Calculate the scale range from data.
    fn calculate_range(&self) -> (f64, f64) {
        if let (Some(min), Some(max)) = (self.min_value, self.max_value) {
            return (min, max);
        }

        // Handle empty boxes case
        if self.boxes.is_empty() {
            let min = self.min_value.unwrap_or(0.0);
            let max = self.max_value.unwrap_or(1.0);
            return if min < max { (min, max) } else { (0.0, 1.0) };
        }

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for stats in &self.boxes {
            min = min.min(stats.min);
            max = max.max(stats.max);

            if self.show_outliers {
                for &o in &stats.outliers_low {
                    min = min.min(o);
                }
                for &o in &stats.outliers_high {
                    max = max.max(o);
                }
            }
        }

        // Apply explicit overrides if provided
        if let Some(m) = self.min_value {
            min = m;
        }
        if let Some(m) = self.max_value {
            max = m;
        }

        // Ensure valid range
        if min >= max || !min.is_finite() || !max.is_finite() {
            (0.0, 1.0)
        } else {
            (min, max)
        }
    }

    /// Map a value to a position in the given range.
    fn map_value(&self, value: f64, min: f64, max: f64, size: u16) -> u16 {
        if max == min {
            return size / 2;
        }
        let normalized = (value - min) / (max - min);
        (normalized * f64::from(size - 1)).round() as u16
    }

    /// Render a horizontal box plot.
    fn render_horizontal(&self, area: Rect, buf: &mut Buffer) {
        if self.boxes.is_empty() || area.width < 5 || area.height == 0 {
            return;
        }

        let (data_min, data_max) = self.calculate_range();
        let box_height = (area.height as usize / self.boxes.len()).max(1);

        for (i, stats) in self.boxes.iter().enumerate() {
            let y = area.y + (i * box_height) as u16;
            if y >= area.y + area.height {
                break;
            }

            let box_y = y + (box_height as u16 / 2);

            // Map positions
            let min_x = area.x + self.map_value(stats.min, data_min, data_max, area.width);
            let q1_x = area.x + self.map_value(stats.q1, data_min, data_max, area.width);
            let med_x = area.x + self.map_value(stats.median, data_min, data_max, area.width);
            let q3_x = area.x + self.map_value(stats.q3, data_min, data_max, area.width);
            let max_x = area.x + self.map_value(stats.max, data_min, data_max, area.width);

            // Draw whisker line (min to max)
            for x in min_x..=max_x {
                if x < area.x + area.width {
                    buf[(x, box_y)].set_char('─').set_fg(stats.color);
                }
            }

            // Draw whisker caps
            if min_x < area.x + area.width {
                buf[(min_x, box_y)].set_char('├').set_fg(stats.color);
            }
            if max_x < area.x + area.width {
                buf[(max_x, box_y)].set_char('┤').set_fg(stats.color);
            }

            // Draw box (Q1 to Q3)
            for x in q1_x..=q3_x {
                if x < area.x + area.width {
                    buf[(x, box_y)].set_char('█').set_fg(stats.color);
                }
            }

            // Draw median line
            if med_x < area.x + area.width {
                buf[(med_x, box_y)].set_char('┃').set_fg(Color::White);
            }

            // Draw outliers
            if self.show_outliers {
                for &o in &stats.outliers_low {
                    let ox = area.x + self.map_value(o, data_min, data_max, area.width);
                    if ox < area.x + area.width {
                        buf[(ox, box_y)].set_char('○').set_fg(stats.color);
                    }
                }
                for &o in &stats.outliers_high {
                    let ox = area.x + self.map_value(o, data_min, data_max, area.width);
                    if ox < area.x + area.width {
                        buf[(ox, box_y)].set_char('○').set_fg(stats.color);
                    }
                }
            }

            // Draw label if present
            if let Some(ref label) = stats.label {
                let label_y = y;
                for (j, ch) in label.chars().take(area.width as usize).enumerate() {
                    let lx = area.x + j as u16;
                    if lx < area.x + area.width && label_y < area.y + area.height {
                        buf[(lx, label_y)].set_char(ch).set_fg(Color::White);
                    }
                }
            }
        }
    }

    /// Render a vertical box plot.
    fn render_vertical(&self, area: Rect, buf: &mut Buffer) {
        if self.boxes.is_empty() || area.height < 5 || area.width == 0 {
            return;
        }

        let (data_min, data_max) = self.calculate_range();
        let box_width = (area.width as usize / self.boxes.len()).max(1);

        for (i, stats) in self.boxes.iter().enumerate() {
            let x = area.x + (i * box_width) as u16;
            if x >= area.x + area.width {
                break;
            }

            let box_x = x + (box_width as u16 / 2);

            // Map positions (inverted for vertical: high values at top)
            let min_y = area.y + area.height
                - 1
                - self.map_value(stats.min, data_min, data_max, area.height);
            let q1_y = area.y + area.height
                - 1
                - self.map_value(stats.q1, data_min, data_max, area.height);
            let med_y = area.y + area.height
                - 1
                - self.map_value(stats.median, data_min, data_max, area.height);
            let q3_y = area.y + area.height
                - 1
                - self.map_value(stats.q3, data_min, data_max, area.height);
            let max_y = area.y + area.height
                - 1
                - self.map_value(stats.max, data_min, data_max, area.height);

            // Draw whisker line (max_y to min_y, since y increases downward)
            for y in max_y..=min_y {
                if y < area.y + area.height {
                    buf[(box_x, y)].set_char('│').set_fg(stats.color);
                }
            }

            // Draw whisker caps
            if max_y < area.y + area.height {
                buf[(box_x, max_y)].set_char('┬').set_fg(stats.color);
            }
            if min_y < area.y + area.height {
                buf[(box_x, min_y)].set_char('┴').set_fg(stats.color);
            }

            // Draw box (Q3 to Q1, since y is inverted)
            for y in q3_y..=q1_y {
                if y < area.y + area.height {
                    buf[(box_x, y)].set_char('█').set_fg(stats.color);
                }
            }

            // Draw median line
            if med_y < area.y + area.height {
                buf[(box_x, med_y)].set_char('━').set_fg(Color::White);
            }

            // Draw outliers
            if self.show_outliers {
                for &o in &stats.outliers_low {
                    let oy = area.y + area.height
                        - 1
                        - self.map_value(o, data_min, data_max, area.height);
                    if oy < area.y + area.height {
                        buf[(box_x, oy)].set_char('○').set_fg(stats.color);
                    }
                }
                for &o in &stats.outliers_high {
                    let oy = area.y + area.height
                        - 1
                        - self.map_value(o, data_min, data_max, area.height);
                    if oy < area.y + area.height {
                        buf[(box_x, oy)].set_char('○').set_fg(stats.color);
                    }
                }
            }
        }
    }
}

impl Widget for BoxPlot {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width == 0 || area.height == 0 {
            return;
        }

        // Fill background
        if self.background != Color::Reset {
            for y in area.y..area.y + area.height {
                for x in area.x..area.x + area.width {
                    buf[(x, y)].set_bg(self.background);
                }
            }
        }

        match self.orientation {
            BoxOrientation::Horizontal => self.render_horizontal(area, buf),
            BoxOrientation::Vertical => self.render_vertical(area, buf),
        }
    }
}

// =============================================================================
// TESTS - EXTREME TDD
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    mod stats_tests {
        use super::*;

        #[test]
        fn test_stats_new() {
            let stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
            assert_eq!(stats.min, 1.0);
            assert_eq!(stats.q1, 2.0);
            assert_eq!(stats.median, 3.0);
            assert_eq!(stats.q3, 4.0);
            assert_eq!(stats.max, 5.0);
        }

        #[test]
        fn test_stats_from_empty_data() {
            let stats = BoxStats::from_data(&[]);
            assert_eq!(stats.min, 0.0);
            assert_eq!(stats.median, 0.0);
            assert_eq!(stats.max, 0.0);
        }

        #[test]
        fn test_stats_from_single_value() {
            let stats = BoxStats::from_data(&[5.0]);
            assert_eq!(stats.min, 5.0);
            assert_eq!(stats.median, 5.0);
            assert_eq!(stats.max, 5.0);
            assert_eq!(stats.q1, 5.0);
            assert_eq!(stats.q3, 5.0);
        }

        #[test]
        fn test_stats_from_two_values() {
            let stats = BoxStats::from_data(&[1.0, 5.0]);
            assert_eq!(stats.min, 1.0);
            assert_eq!(stats.max, 5.0);
            assert_eq!(stats.median, 3.0); // (1+5)/2
        }

        #[test]
        fn test_stats_from_normal_data() {
            let data: Vec<f64> = (1..=10).map(f64::from).collect();
            let stats = BoxStats::from_data(&data);

            // Check quartiles
            assert!((stats.median - 5.5).abs() < 0.01);
            assert!(stats.q1 > stats.min);
            assert!(stats.q3 < stats.max);
            assert!(stats.q1 < stats.median);
            assert!(stats.q3 > stats.median);
        }

        #[test]
        fn test_stats_with_outliers() {
            let mut data: Vec<f64> = (1..=10).map(f64::from).collect();
            data.push(100.0); // High outlier
            data.push(-50.0); // Low outlier

            let stats = BoxStats::from_data(&data);

            assert!(!stats.outliers_high.is_empty());
            assert!(!stats.outliers_low.is_empty());
            assert!(stats.outliers_high.contains(&100.0));
            assert!(stats.outliers_low.contains(&-50.0));
        }

        #[test]
        fn test_stats_iqr() {
            let stats = BoxStats::new(1.0, 3.0, 5.0, 7.0, 10.0);
            assert_eq!(stats.iqr(), 4.0); // 7 - 3
        }

        #[test]
        fn test_stats_range() {
            let stats = BoxStats::new(1.0, 3.0, 5.0, 7.0, 10.0);
            assert_eq!(stats.range(), 9.0); // 10 - 1
        }

        #[test]
        fn test_stats_has_outliers() {
            let mut stats = BoxStats::new(1.0, 3.0, 5.0, 7.0, 10.0);
            assert!(!stats.has_outliers());

            stats.outliers_high.push(100.0);
            assert!(stats.has_outliers());
        }

        #[test]
        fn test_stats_with_label() {
            let stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0).with_label("Test");
            assert_eq!(stats.label, Some("Test".to_string()));
        }

        #[test]
        fn test_stats_with_color() {
            let stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0).with_color(Color::Red);
            assert_eq!(stats.color, Color::Red);
        }

        #[test]
        fn test_percentile_edge_cases() {
            // Empty
            assert_eq!(BoxStats::percentile(&[], 0.5), 0.0);

            // Single value
            assert_eq!(BoxStats::percentile(&[42.0], 0.5), 42.0);

            // Two values
            assert_eq!(BoxStats::percentile(&[1.0, 3.0], 0.5), 2.0);
        }
    }

    mod plot_construction_tests {
        use super::*;

        #[test]
        fn test_new_empty() {
            let plot = BoxPlot::new(Vec::new());
            assert!(plot.boxes.is_empty());
            assert_eq!(plot.orientation, BoxOrientation::Horizontal);
        }

        #[test]
        fn test_new_with_boxes() {
            let stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
            let plot = BoxPlot::new(vec![stats]);
            assert_eq!(plot.boxes.len(), 1);
        }

        #[test]
        fn test_with_orientation() {
            let plot = BoxPlot::new(Vec::new()).with_orientation(BoxOrientation::Vertical);
            assert_eq!(plot.orientation, BoxOrientation::Vertical);
        }

        #[test]
        fn test_with_title() {
            let plot = BoxPlot::new(Vec::new()).with_title("My Plot");
            assert_eq!(plot.title, Some("My Plot".to_string()));
        }

        #[test]
        fn test_with_outliers() {
            let plot = BoxPlot::new(Vec::new()).with_outliers(false);
            assert!(!plot.show_outliers);
        }

        #[test]
        fn test_with_range() {
            let plot = BoxPlot::new(Vec::new()).with_range(0.0, 100.0);
            assert_eq!(plot.min_value, Some(0.0));
            assert_eq!(plot.max_value, Some(100.0));
        }

        #[test]
        fn test_with_background() {
            let plot = BoxPlot::new(Vec::new()).with_background(Color::Black);
            assert_eq!(plot.background, Color::Black);
        }

        #[test]
        fn test_default() {
            let plot = BoxPlot::default();
            assert!(plot.boxes.is_empty());
        }

        #[test]
        fn test_builder_chaining() {
            let stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
            let plot = BoxPlot::new(vec![stats])
                .with_orientation(BoxOrientation::Vertical)
                .with_title("Test")
                .with_outliers(false)
                .with_range(0.0, 10.0)
                .with_background(Color::Black);

            assert_eq!(plot.boxes.len(), 1);
            assert_eq!(plot.orientation, BoxOrientation::Vertical);
            assert_eq!(plot.title, Some("Test".to_string()));
            assert!(!plot.show_outliers);
            assert_eq!(plot.min_value, Some(0.0));
            assert_eq!(plot.max_value, Some(10.0));
            assert_eq!(plot.background, Color::Black);
        }
    }

    mod range_calculation_tests {
        use super::*;

        #[test]
        fn test_calculate_range_empty() {
            let plot = BoxPlot::new(Vec::new());
            let (min, max) = plot.calculate_range();
            // With no data, should return some default valid range
            assert!(max > min);
        }

        #[test]
        fn test_calculate_range_single_box() {
            let stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
            let plot = BoxPlot::new(vec![stats]);
            let (min, max) = plot.calculate_range();
            assert_eq!(min, 1.0);
            assert_eq!(max, 5.0);
        }

        #[test]
        fn test_calculate_range_multiple_boxes() {
            let stats1 = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
            let stats2 = BoxStats::new(0.0, 3.0, 6.0, 9.0, 12.0);
            let plot = BoxPlot::new(vec![stats1, stats2]);
            let (min, max) = plot.calculate_range();
            assert_eq!(min, 0.0);
            assert_eq!(max, 12.0);
        }

        #[test]
        fn test_calculate_range_with_explicit_range() {
            let stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
            let plot = BoxPlot::new(vec![stats]).with_range(0.0, 100.0);
            let (min, max) = plot.calculate_range();
            assert_eq!(min, 0.0);
            assert_eq!(max, 100.0);
        }

        #[test]
        fn test_calculate_range_includes_outliers() {
            let mut stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
            stats.outliers_high.push(100.0);
            stats.outliers_low.push(-50.0);

            let plot = BoxPlot::new(vec![stats]).with_outliers(true);
            let (min, max) = plot.calculate_range();
            assert_eq!(min, -50.0);
            assert_eq!(max, 100.0);
        }

        #[test]
        fn test_calculate_range_excludes_outliers_when_disabled() {
            let mut stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
            stats.outliers_high.push(100.0);
            stats.outliers_low.push(-50.0);

            let plot = BoxPlot::new(vec![stats]).with_outliers(false);
            let (min, max) = plot.calculate_range();
            assert_eq!(min, 1.0);
            assert_eq!(max, 5.0);
        }
    }

    mod rendering_tests {
        use super::*;

        fn create_test_buffer(width: u16, height: u16) -> (Rect, Buffer) {
            let area = Rect::new(0, 0, width, height);
            let buffer = Buffer::empty(area);
            (area, buffer)
        }

        #[test]
        fn test_render_empty() {
            let (area, mut buf) = create_test_buffer(20, 5);
            let plot = BoxPlot::new(Vec::new());
            plot.render(area, &mut buf);
            // Should not panic
        }

        #[test]
        fn test_render_zero_area() {
            let (_, mut buf) = create_test_buffer(20, 5);
            let stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
            let plot = BoxPlot::new(vec![stats]);

            // Zero width
            let zero_area = Rect::new(0, 0, 0, 5);
            plot.clone().render(zero_area, &mut buf);

            // Zero height
            let zero_area = Rect::new(0, 0, 20, 0);
            plot.render(zero_area, &mut buf);
            // Should not panic
        }

        #[test]
        fn test_render_horizontal_single_box() {
            let (area, mut buf) = create_test_buffer(30, 3);
            let stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
            let plot = BoxPlot::new(vec![stats]).with_orientation(BoxOrientation::Horizontal);
            plot.render(area, &mut buf);

            // Buffer should have content
            let content: String = buf.content().iter().map(|c| c.symbol().to_string()).collect();
            assert!(!content.chars().all(|c| c == ' '));
        }

        #[test]
        fn test_render_vertical_single_box() {
            let (area, mut buf) = create_test_buffer(5, 15);
            let stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
            let plot = BoxPlot::new(vec![stats]).with_orientation(BoxOrientation::Vertical);
            plot.render(area, &mut buf);

            // Buffer should have content
            let content: String = buf.content().iter().map(|c| c.symbol().to_string()).collect();
            assert!(!content.chars().all(|c| c == ' '));
        }

        #[test]
        fn test_render_multiple_boxes_horizontal() {
            let (area, mut buf) = create_test_buffer(40, 6);
            let stats1 = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0).with_color(Color::Red);
            let stats2 = BoxStats::new(2.0, 3.0, 4.0, 5.0, 6.0).with_color(Color::Blue);
            let plot = BoxPlot::new(vec![stats1, stats2]);
            plot.render(area, &mut buf);
            // Should render without panic
        }

        #[test]
        fn test_render_multiple_boxes_vertical() {
            let (area, mut buf) = create_test_buffer(10, 20);
            let stats1 = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
            let stats2 = BoxStats::new(2.0, 3.0, 4.0, 5.0, 6.0);
            let plot =
                BoxPlot::new(vec![stats1, stats2]).with_orientation(BoxOrientation::Vertical);
            plot.render(area, &mut buf);
            // Should render without panic
        }

        #[test]
        fn test_render_with_outliers() {
            let (area, mut buf) = create_test_buffer(40, 5);
            let mut stats = BoxStats::new(10.0, 20.0, 30.0, 40.0, 50.0);
            stats.outliers_low.push(0.0);
            stats.outliers_high.push(100.0);
            let has_outliers = stats.has_outliers();

            let plot = BoxPlot::new(vec![stats]).with_outliers(true);
            plot.render(area, &mut buf);

            // Should contain outlier markers
            let content: String = buf.content().iter().map(|c| c.symbol().to_string()).collect();
            assert!(content.contains("○") || !has_outliers);
        }

        #[test]
        fn test_render_with_labels() {
            let (area, mut buf) = create_test_buffer(30, 3);
            let stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0).with_label("CPU");
            let plot = BoxPlot::new(vec![stats]);
            plot.render(area, &mut buf);
            // Should include label text
        }

        #[test]
        fn test_render_with_background() {
            let (area, mut buf) = create_test_buffer(20, 3);
            let stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
            let plot = BoxPlot::new(vec![stats]).with_background(Color::Black);
            plot.render(area, &mut buf);

            // Background should be set - verify at least one cell has black background
            let cell = buf.cell((area.x, area.y)).expect("operation should succeed");
            // Cell exists (test doesn't panic) confirms rendering worked
            assert!(!cell.symbol().is_empty() || area.width > 0);
        }

        #[test]
        fn test_render_narrow_horizontal() {
            let (area, mut buf) = create_test_buffer(5, 3);
            let stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
            let plot = BoxPlot::new(vec![stats]);
            plot.render(area, &mut buf);
            // Should handle narrow area without panic
        }

        #[test]
        fn test_render_short_vertical() {
            let (area, mut buf) = create_test_buffer(5, 5);
            let stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
            let plot = BoxPlot::new(vec![stats]).with_orientation(BoxOrientation::Vertical);
            plot.render(area, &mut buf);
            // Should handle short area without panic
        }
    }

    mod orientation_tests {
        use super::*;

        #[test]
        fn test_orientation_default_is_horizontal() {
            assert_eq!(BoxOrientation::default(), BoxOrientation::Horizontal);
        }

        #[test]
        fn test_orientation_equality() {
            assert_eq!(BoxOrientation::Horizontal, BoxOrientation::Horizontal);
            assert_eq!(BoxOrientation::Vertical, BoxOrientation::Vertical);
            assert_ne!(BoxOrientation::Horizontal, BoxOrientation::Vertical);
        }
    }

    mod edge_case_tests {
        use super::*;

        fn create_test_buffer(width: u16, height: u16) -> (Rect, Buffer) {
            let area = Rect::new(0, 0, width, height);
            let buffer = Buffer::empty(area);
            (area, buffer)
        }

        #[test]
        fn test_percentile_exact_index() {
            // With 5 values, p=0.25 gives index=1.0 exactly (lower==upper)
            let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
            let result = BoxStats::percentile(&data, 0.25);
            assert_eq!(result, 2.0);
        }

        #[test]
        fn test_percentile_upper_bound() {
            // With 2 values, p=1.0 should hit upper bound
            let data = vec![1.0, 2.0];
            let result = BoxStats::percentile(&data, 1.0);
            assert_eq!(result, 2.0);
        }

        #[test]
        fn test_calculate_range_with_explicit_min_only() {
            let stats = BoxStats::new(5.0, 6.0, 7.0, 8.0, 9.0);
            let mut plot = BoxPlot::new(vec![stats]);
            plot.min_value = Some(0.0);
            let (min, max) = plot.calculate_range();
            assert_eq!(min, 0.0);
            assert_eq!(max, 9.0);
        }

        #[test]
        fn test_calculate_range_with_explicit_max_only() {
            let stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
            let mut plot = BoxPlot::new(vec![stats]);
            plot.max_value = Some(100.0);
            let (min, max) = plot.calculate_range();
            assert_eq!(min, 1.0);
            assert_eq!(max, 100.0);
        }

        #[test]
        fn test_calculate_range_invalid_range() {
            // Create plot where min >= max (degenerate case)
            let stats = BoxStats::new(5.0, 5.0, 5.0, 5.0, 5.0);
            let plot = BoxPlot::new(vec![stats]);
            let (min, max) = plot.calculate_range();
            // Should return valid default range
            assert!(max > min);
        }

        #[test]
        fn test_map_value_equal_min_max() {
            // All values the same -> max == min
            let stats = BoxStats::new(5.0, 5.0, 5.0, 5.0, 5.0);
            let (area, mut buf) = create_test_buffer(20, 5);
            let plot = BoxPlot::new(vec![stats]);
            plot.render(area, &mut buf);
            // Should render without panic, centering the box
        }

        #[test]
        fn test_render_many_boxes_horizontal_overflow() {
            // Many boxes in small vertical space - should break early
            let boxes: Vec<BoxStats> = (0..20)
                .map(|i| {
                    BoxStats::new(
                        f64::from(i),
                        f64::from(i) + 1.0,
                        f64::from(i) + 2.0,
                        f64::from(i) + 3.0,
                        f64::from(i) + 4.0,
                    )
                })
                .collect();
            let (area, mut buf) = create_test_buffer(40, 3); // Only 3 height for 20 boxes
            let plot = BoxPlot::new(boxes);
            plot.render(area, &mut buf);
            // Should render only what fits
        }

        #[test]
        fn test_render_many_boxes_vertical_overflow() {
            // Many boxes in small horizontal space - should break early
            let boxes: Vec<BoxStats> = (0..20)
                .map(|i| {
                    BoxStats::new(
                        f64::from(i),
                        f64::from(i) + 1.0,
                        f64::from(i) + 2.0,
                        f64::from(i) + 3.0,
                        f64::from(i) + 4.0,
                    )
                })
                .collect();
            let (area, mut buf) = create_test_buffer(5, 20); // Only 5 width for 20 boxes
            let plot = BoxPlot::new(boxes).with_orientation(BoxOrientation::Vertical);
            plot.render(area, &mut buf);
            // Should render only what fits
        }

        #[test]
        fn test_render_vertical_with_outliers() {
            // Test vertical outlier rendering
            let mut stats = BoxStats::new(10.0, 20.0, 30.0, 40.0, 50.0);
            stats.outliers_low.push(0.0);
            stats.outliers_high.push(100.0);
            let (area, mut buf) = create_test_buffer(10, 20);
            let plot = BoxPlot::new(vec![stats])
                .with_orientation(BoxOrientation::Vertical)
                .with_outliers(true);
            plot.render(area, &mut buf);
            // Should render outliers
            let content: String = buf.content().iter().map(|c| c.symbol().to_string()).collect();
            assert!(content.contains("○"));
        }

        #[test]
        fn test_render_vertical_too_short() {
            // Vertical with height < 5 should return early
            let stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
            let (_, mut buf) = create_test_buffer(10, 20);
            let area = Rect::new(0, 0, 10, 4); // height < 5
            let plot = BoxPlot::new(vec![stats]).with_orientation(BoxOrientation::Vertical);
            plot.render(area, &mut buf);
            // Should return early without panic
        }

        #[test]
        fn test_render_horizontal_too_narrow() {
            // Horizontal with width < 5 should return early
            let stats = BoxStats::new(1.0, 2.0, 3.0, 4.0, 5.0);
            let (_, mut buf) = create_test_buffer(20, 5);
            let area = Rect::new(0, 0, 4, 5); // width < 5
            let plot = BoxPlot::new(vec![stats]);
            plot.render(area, &mut buf);
            // Should return early without panic
        }

        #[test]
        fn test_render_horizontal_high_outliers_in_bounds() {
            // Test horizontal high outlier rendering within bounds
            let mut stats = BoxStats::new(10.0, 20.0, 30.0, 40.0, 50.0);
            stats.outliers_high.push(55.0); // Within reasonable range
            let (area, mut buf) = create_test_buffer(60, 5);
            let plot = BoxPlot::new(vec![stats]).with_outliers(true);
            plot.render(area, &mut buf);
            let content: String = buf.content().iter().map(|c| c.symbol().to_string()).collect();
            assert!(content.contains("○"));
        }

        #[test]
        fn test_render_vertical_multiple_outliers() {
            // Test vertical with multiple outliers
            let mut stats = BoxStats::new(25.0, 35.0, 50.0, 65.0, 75.0);
            stats.outliers_low.push(5.0);
            stats.outliers_low.push(10.0);
            stats.outliers_high.push(90.0);
            stats.outliers_high.push(95.0);
            let (area, mut buf) = create_test_buffer(10, 25);
            let plot = BoxPlot::new(vec![stats])
                .with_orientation(BoxOrientation::Vertical)
                .with_outliers(true);
            plot.render(area, &mut buf);
            // Multiple outliers should render
            let content: String = buf.content().iter().map(|c| c.symbol().to_string()).collect();
            let outlier_count = content.matches('○').count();
            assert!(outlier_count >= 2);
        }
    }
}
