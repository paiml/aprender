//! Violin plot widget with Kernel Density Estimation (KDE).
//!
//! Implements Silverman's rule (1986) for bandwidth selection and
//! Gaussian kernel for density estimation. Supports both scalar and
//! SIMD-optimized computation paths for large datasets.
//!
//! ## References
//! - Hintze & Nelson (1998): Violin Plots: A Box Plot-Density Trace Synergism
//! - Silverman (1986): Density Estimation for Statistics and Data Analysis

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::Widget,
};

/// Orientation of violin plot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ViolinOrientation {
    /// Vertical violins (default) - data range on Y axis.
    #[default]
    Vertical,
    /// Horizontal violins - data range on X axis.
    Horizontal,
}

/// Statistics for a violin distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct ViolinStats {
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Median (50th percentile).
    pub median: f64,
    /// First quartile (25th percentile).
    pub q1: f64,
    /// Third quartile (75th percentile).
    pub q3: f64,
    /// Mean value.
    pub mean: f64,
}

impl Default for ViolinStats {
    fn default() -> Self {
        Self { min: 0.0, max: 0.0, median: 0.0, q1: 0.0, q3: 0.0, mean: 0.0 }
    }
}

/// A single violin distribution with KDE.
#[derive(Debug, Clone)]
pub struct ViolinData {
    /// Label for this violin.
    pub label: String,
    /// Raw data values.
    values: Vec<f64>,
    /// Color for this violin.
    pub color: Color,
    /// Cached KDE densities.
    densities: Option<Vec<f64>>,
    /// Cached statistics.
    stats: Option<ViolinStats>,
}

impl ViolinData {
    /// Create new violin data from a slice of values.
    #[must_use]
    pub fn new(label: impl Into<String>, values: &[f64]) -> Self {
        Self {
            label: label.into(),
            values: values.to_vec(),
            color: Color::Cyan,
            densities: None,
            stats: None,
        }
    }

    /// Create violin data from a Vec (avoids clone).
    #[must_use]
    pub fn from_vec(label: impl Into<String>, values: Vec<f64>) -> Self {
        Self { label: label.into(), values, color: Color::Cyan, densities: None, stats: None }
    }

    /// Set color for this violin.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Get the raw data values.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Get count of values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Compute statistics for this violin.
    fn compute_stats(&mut self) {
        if self.values.is_empty() {
            self.stats = Some(ViolinStats::default());
            return;
        }

        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let min = sorted[0];
        let max = sorted[n - 1];

        let median = Self::percentile(&sorted, 0.5);
        let q1 = Self::percentile(&sorted, 0.25);
        let q3 = Self::percentile(&sorted, 0.75);
        let mean = sorted.iter().sum::<f64>() / n as f64;

        self.stats = Some(ViolinStats { min, max, median, q1, q3, mean });
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
        let pos = p * (n - 1) as f64;
        let floor = pos.floor() as usize;
        let ceil = pos.ceil() as usize;

        if floor == ceil {
            sorted[floor]
        } else {
            let frac = pos - floor as f64;
            sorted[floor] * (1.0 - frac) + sorted[ceil] * frac
        }
    }

    /// Get statistics, computing if necessary.
    pub fn stats(&mut self) -> &ViolinStats {
        if self.stats.is_none() {
            self.compute_stats();
        }
        self.stats.as_ref().expect("computed above")
    }

    /// Get cached stats without computation.
    #[must_use]
    pub fn cached_stats(&self) -> Option<&ViolinStats> {
        self.stats.as_ref()
    }

    /// Compute standard deviation.
    fn compute_std_dev(&self) -> f64 {
        if self.values.len() < 2 {
            return 1.0;
        }
        let mean = self.values.iter().sum::<f64>() / self.values.len() as f64;
        let variance =
            self.values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / self.values.len() as f64;
        variance.sqrt().max(0.001)
    }

    /// Compute KDE densities.
    /// Uses SIMD-friendly batch processing for large datasets (>100 elements).
    pub fn compute_kde(&mut self, num_points: usize) {
        if self.values.is_empty() {
            self.densities = Some(vec![0.0; num_points]);
            return;
        }

        // Ensure stats are computed
        if self.stats.is_none() {
            self.compute_stats();
        }
        let stats = self.stats.as_ref().expect("computed above").clone();

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

        // Use SIMD-friendly path for large datasets
        let use_simd = self.values.len() > 100;

        for (i, density) in densities.iter_mut().enumerate() {
            let t = if num_points > 1 { i as f64 / (num_points - 1) as f64 } else { 0.5 };
            let x = stats.min + t * range;

            *density = if use_simd {
                self.kde_at_point_simd(x, bandwidth)
            } else {
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

    /// Scalar KDE computation using Gaussian kernel.
    fn kde_at_point_scalar(&self, x: f64, bandwidth: f64) -> f64 {
        let mut sum = 0.0;
        let inv_bw = 1.0 / bandwidth;
        for &value in &self.values {
            let u = (x - value) * inv_bw;
            // Gaussian kernel: K(u) = (1/sqrt(2π)) * exp(-u²/2)
            sum += (-0.5 * u * u).exp();
        }
        sum * inv_bw / (self.values.len() as f64 * std::f64::consts::TAU.sqrt())
    }

    /// SIMD-optimized KDE computation for large datasets.
    /// Uses 4-wide batch processing for SIMD-friendly operation.
    fn kde_at_point_simd(&self, x: f64, bandwidth: f64) -> f64 {
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

        sum * inv_bw / (self.values.len() as f64 * std::f64::consts::TAU.sqrt())
    }

    /// Get cached KDE densities.
    #[must_use]
    pub fn densities(&self) -> Option<&[f64]> {
        self.densities.as_deref()
    }
}

/// Violin plot widget for ratatui.
#[derive(Debug, Clone)]
pub struct ViolinPlot {
    /// Violins to display.
    violins: Vec<ViolinData>,
    /// Orientation of the plot.
    orientation: ViolinOrientation,
    /// Show box plot inside violin.
    show_box: bool,
    /// Show median line.
    show_median: bool,
    /// Number of KDE points.
    kde_points: usize,
    /// Optional title.
    title: Option<String>,
    /// Background color.
    background: Option<Color>,
    /// Explicit value range.
    min_value: Option<f64>,
    max_value: Option<f64>,
}

impl Default for ViolinPlot {
    fn default() -> Self {
        Self::new()
    }
}

impl ViolinPlot {
    /// Create a new empty violin plot.
    #[must_use]
    pub fn new() -> Self {
        Self {
            violins: Vec::new(),
            orientation: ViolinOrientation::default(),
            show_box: true,
            show_median: true,
            kde_points: 50,
            title: None,
            background: None,
            min_value: None,
            max_value: None,
        }
    }

    /// Create violin plot with data.
    #[must_use]
    pub fn with_data(data: Vec<ViolinData>) -> Self {
        Self {
            violins: data,
            orientation: ViolinOrientation::default(),
            show_box: true,
            show_median: true,
            kde_points: 50,
            title: None,
            background: None,
            min_value: None,
            max_value: None,
        }
    }

    /// Set orientation.
    #[must_use]
    pub fn orientation(mut self, orientation: ViolinOrientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Toggle box plot display.
    #[must_use]
    pub fn show_box(mut self, show: bool) -> Self {
        self.show_box = show;
        self
    }

    /// Toggle median line.
    #[must_use]
    pub fn show_median(mut self, show: bool) -> Self {
        self.show_median = show;
        self
    }

    /// Set KDE resolution (clamped to 10-200).
    #[must_use]
    pub fn kde_points(mut self, points: usize) -> Self {
        self.kde_points = points.clamp(10, 200);
        self
    }

    /// Set title.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set background color.
    #[must_use]
    pub fn background(mut self, color: Color) -> Self {
        self.background = Some(color);
        self
    }

    /// Set explicit value range.
    #[must_use]
    pub fn range(mut self, min: f64, max: f64) -> Self {
        self.min_value = Some(min);
        self.max_value = Some(max);
        self
    }

    /// Calculate global value range across all violins.
    fn global_range(&self) -> (f64, f64) {
        if let (Some(min), Some(max)) = (self.min_value, self.max_value) {
            if min < max {
                return (min, max);
            }
        }

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for violin in &self.violins {
            for &v in violin.values() {
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

    /// Render vertical violin plot.
    fn render_vertical(&mut self, area: Rect, buf: &mut Buffer) {
        if self.violins.is_empty() || area.width < 3 || area.height < 3 {
            return;
        }

        let (val_min, val_max) = self.global_range();
        let val_range = val_max - val_min;
        if val_range <= 0.0 {
            return;
        }

        let n_violins = self.violins.len();
        let violin_width = area.width as usize / n_violins;

        for (idx, violin) in self.violins.iter_mut().enumerate() {
            // Compute KDE if not cached
            if violin.densities.is_none() {
                violin.compute_kde(self.kde_points);
            }

            let densities = match violin.densities() {
                Some(d) => d,
                None => continue,
            };

            let center_x = area.x + (idx * violin_width + violin_width / 2) as u16;
            let half_width = (violin_width as f64 * 0.4).max(1.0);

            // Draw violin shape
            for (i, &density) in densities.iter().enumerate() {
                let t =
                    if densities.len() > 1 { i as f64 / (densities.len() - 1) as f64 } else { 0.5 };
                let value = val_min + t * val_range;
                let y_norm = 1.0 - (value - val_min) / val_range;
                let y = area.y + (y_norm * f64::from(area.height - 1)) as u16;

                if y < area.y || y >= area.y + area.height {
                    continue;
                }

                let width = (density * half_width) as u16;
                if width == 0 {
                    continue;
                }

                let style = Style::default().fg(violin.color);

                // Draw symmetric violin halves using block characters
                let block_chars = ['░', '▒', '▓', '█'];
                let char_idx = ((density * 3.0).round() as usize).min(3);
                let ch = block_chars[char_idx];

                // Left half
                if center_x >= width && center_x - width >= area.x {
                    let cell = buf.cell_mut((center_x - width, y));
                    if let Some(cell) = cell {
                        cell.set_char(ch).set_style(style);
                    }
                }
                // Right half
                if center_x + width < area.x + area.width {
                    let cell = buf.cell_mut((center_x + width, y));
                    if let Some(cell) = cell {
                        cell.set_char(ch).set_style(style);
                    }
                }
            }

            // Draw median if enabled
            if self.show_median {
                let stats = violin.stats();
                let median_y_norm = 1.0 - (stats.median - val_min) / val_range;
                let median_y = area.y + (median_y_norm * f64::from(area.height - 1)) as u16;

                if median_y >= area.y && median_y < area.y + area.height {
                    let style = Style::default().fg(Color::White);
                    if center_x > area.x {
                        let cell = buf.cell_mut((center_x - 1, median_y));
                        if let Some(cell) = cell {
                            cell.set_char('─').set_style(style);
                        }
                    }
                    let cell = buf.cell_mut((center_x, median_y));
                    if let Some(cell) = cell {
                        cell.set_char('─').set_style(style);
                    }
                }
            }

            // Draw label at bottom
            let label = &violin.label;
            let label_x = center_x.saturating_sub(label.len() as u16 / 2);
            let label_y = area.y + area.height - 1;
            let style = Style::default().fg(Color::DarkGray);

            for (i, ch) in label.chars().enumerate() {
                let x = label_x + i as u16;
                if x < area.x + area.width {
                    let cell = buf.cell_mut((x, label_y));
                    if let Some(cell) = cell {
                        cell.set_char(ch).set_style(style);
                    }
                }
            }
        }
    }

    /// Render horizontal violin plot.
    fn render_horizontal(&mut self, area: Rect, buf: &mut Buffer) {
        if self.violins.is_empty() || area.width < 3 || area.height < 3 {
            return;
        }

        let (val_min, val_max) = self.global_range();
        let val_range = val_max - val_min;
        if val_range <= 0.0 {
            return;
        }

        let n_violins = self.violins.len();
        let violin_height = area.height as usize / n_violins;

        for (idx, violin) in self.violins.iter_mut().enumerate() {
            if violin.densities.is_none() {
                violin.compute_kde(self.kde_points);
            }

            let densities = match violin.densities() {
                Some(d) => d,
                None => continue,
            };

            let center_y = area.y + (idx * violin_height + violin_height / 2) as u16;
            let half_height = (violin_height as f64 * 0.4).max(1.0);

            // Draw violin shape horizontally
            for (i, &density) in densities.iter().enumerate() {
                let t =
                    if densities.len() > 1 { i as f64 / (densities.len() - 1) as f64 } else { 0.5 };
                let value = val_min + t * val_range;
                let x_norm = (value - val_min) / val_range;
                let x = area.x + (x_norm * f64::from(area.width - 1)) as u16;

                if x < area.x || x >= area.x + area.width {
                    continue;
                }

                let height = (density * half_height) as u16;
                if height == 0 {
                    continue;
                }

                let style = Style::default().fg(violin.color);
                let block_chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
                let char_idx = ((density * 7.0).round() as usize).min(7);
                let ch = block_chars[char_idx];

                let cell = buf.cell_mut((x, center_y));
                if let Some(cell) = cell {
                    cell.set_char(ch).set_style(style);
                }
            }

            // Draw median if enabled
            if self.show_median {
                let stats = violin.stats();
                let median_x_norm = (stats.median - val_min) / val_range;
                let median_x = area.x + (median_x_norm * f64::from(area.width - 1)) as u16;

                if median_x >= area.x && median_x < area.x + area.width {
                    let style = Style::default().fg(Color::White);
                    let cell = buf.cell_mut((median_x, center_y));
                    if let Some(cell) = cell {
                        cell.set_char('│').set_style(style);
                    }
                }
            }
        }
    }
}

impl Widget for ViolinPlot {
    fn render(mut self, area: Rect, buf: &mut Buffer) {
        if area.width == 0 || area.height == 0 {
            return;
        }

        // Fill background if set
        if let Some(bg) = self.background {
            let style = Style::default().bg(bg);
            for y in area.y..area.y + area.height {
                for x in area.x..area.x + area.width {
                    if let Some(cell) = buf.cell_mut((x, y)) {
                        cell.set_style(style);
                    }
                }
            }
        }

        // Render title if set
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
            ViolinOrientation::Vertical => self.render_vertical(plot_area, buf),
            ViolinOrientation::Horizontal => self.render_horizontal(plot_area, buf),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod violin_stats_tests {
        use super::*;

        #[test]
        fn test_stats_default() {
            let stats = ViolinStats::default();
            assert_eq!(stats.min, 0.0);
            assert_eq!(stats.max, 0.0);
            assert_eq!(stats.median, 0.0);
            assert_eq!(stats.q1, 0.0);
            assert_eq!(stats.q3, 0.0);
            assert_eq!(stats.mean, 0.0);
        }

        #[test]
        fn test_stats_partial_eq() {
            let a = ViolinStats::default();
            let b = ViolinStats::default();
            assert_eq!(a, b);
        }
    }

    mod violin_data_tests {
        use super::*;

        #[test]
        fn test_new() {
            let data = ViolinData::new("Test", &[1.0, 2.0, 3.0]);
            assert_eq!(data.label, "Test");
            assert_eq!(data.values.len(), 3);
            assert_eq!(data.color, Color::Cyan);
        }

        #[test]
        fn test_from_vec() {
            let data = ViolinData::from_vec("Test", vec![1.0, 2.0, 3.0]);
            assert_eq!(data.label, "Test");
            assert_eq!(data.values.len(), 3);
        }

        #[test]
        fn test_with_color() {
            let data = ViolinData::new("Test", &[1.0]).with_color(Color::Red);
            assert_eq!(data.color, Color::Red);
        }

        #[test]
        fn test_values() {
            let data = ViolinData::new("Test", &[1.0, 2.0, 3.0]);
            assert_eq!(data.values(), &[1.0, 2.0, 3.0]);
        }

        #[test]
        fn test_len() {
            let data = ViolinData::new("Test", &[1.0, 2.0, 3.0]);
            assert_eq!(data.len(), 3);
        }

        #[test]
        fn test_is_empty() {
            let empty = ViolinData::new("Empty", &[]);
            let non_empty = ViolinData::new("Test", &[1.0]);
            assert!(empty.is_empty());
            assert!(!non_empty.is_empty());
        }

        #[test]
        fn test_stats_normal_data() {
            let mut data = ViolinData::new("Test", &[1.0, 2.0, 3.0, 4.0, 5.0]);
            let stats = data.stats();
            assert_eq!(stats.min, 1.0);
            assert_eq!(stats.max, 5.0);
            assert_eq!(stats.median, 3.0);
            assert_eq!(stats.mean, 3.0);
        }

        #[test]
        fn test_stats_empty_data() {
            let mut data = ViolinData::new("Empty", &[]);
            let stats = data.stats();
            assert_eq!(stats.min, 0.0);
            assert_eq!(stats.max, 0.0);
        }

        #[test]
        fn test_stats_single_value() {
            let mut data = ViolinData::new("Single", &[5.0]);
            let stats = data.stats();
            assert_eq!(stats.min, 5.0);
            assert_eq!(stats.max, 5.0);
            assert_eq!(stats.median, 5.0);
        }

        #[test]
        fn test_stats_two_values() {
            let mut data = ViolinData::new("Two", &[1.0, 5.0]);
            let stats = data.stats();
            assert_eq!(stats.min, 1.0);
            assert_eq!(stats.max, 5.0);
            assert!((stats.median - 3.0).abs() < 0.001);
        }

        #[test]
        fn test_cached_stats() {
            let mut data = ViolinData::new("Test", &[1.0, 2.0, 3.0]);
            assert!(data.cached_stats().is_none());
            let _ = data.stats();
            assert!(data.cached_stats().is_some());
        }

        #[test]
        fn test_percentile_empty() {
            let sorted: Vec<f64> = vec![];
            assert_eq!(ViolinData::percentile(&sorted, 0.5), 0.0);
        }

        #[test]
        fn test_percentile_single() {
            let sorted = vec![5.0];
            assert_eq!(ViolinData::percentile(&sorted, 0.5), 5.0);
        }

        #[test]
        fn test_percentile_interpolation() {
            let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
            assert_eq!(ViolinData::percentile(&sorted, 0.0), 1.0);
            assert_eq!(ViolinData::percentile(&sorted, 1.0), 5.0);
            assert_eq!(ViolinData::percentile(&sorted, 0.5), 3.0);
        }
    }

    mod kde_tests {
        use super::*;

        #[test]
        fn test_kde_empty() {
            let mut data = ViolinData::new("Empty", &[]);
            data.compute_kde(20);
            let densities = data.densities().expect("computed");
            assert_eq!(densities.len(), 20);
            assert!(densities.iter().all(|&d| d == 0.0));
        }

        #[test]
        fn test_kde_single_value() {
            let mut data = ViolinData::new("Single", &[5.0]);
            data.compute_kde(10);
            assert!(data.densities().is_some());
        }

        #[test]
        fn test_kde_same_values() {
            let mut data = ViolinData::new("Same", &[5.0, 5.0, 5.0]);
            data.compute_kde(20);
            let densities = data.densities().expect("computed");
            // All densities should be 1.0 when range is 0
            assert!(densities.iter().all(|&d| (d - 1.0).abs() < 0.001));
        }

        #[test]
        fn test_kde_normal() {
            let mut data = ViolinData::new("Normal", &[1.0, 2.0, 3.0, 4.0, 5.0]);
            data.compute_kde(20);
            let densities = data.densities().expect("computed");
            assert_eq!(densities.len(), 20);
            // Densities should be in [0, 1] after normalization
            assert!(densities.iter().all(|&d| (0.0..=1.0).contains(&d)));
            // At least one density should be 1.0 (the max)
            assert!(densities.iter().any(|&d| d > 0.9));
        }

        #[test]
        fn test_kde_large_dataset_simd_path() {
            // Test SIMD path (>100 elements)
            let values: Vec<f64> = (0..200).map(|i| f64::from(i) / 10.0).collect();
            let mut data = ViolinData::from_vec("Large", values);
            data.compute_kde(50);
            assert!(data.densities().is_some());
        }

        #[test]
        fn test_std_dev_single() {
            let data = ViolinData::new("Single", &[5.0]);
            let std_dev = data.compute_std_dev();
            assert!((std_dev - 1.0).abs() < 0.001); // Returns 1.0 for len < 2
        }

        #[test]
        fn test_std_dev_normal() {
            let data = ViolinData::new("Normal", &[1.0, 2.0, 3.0, 4.0, 5.0]);
            let std_dev = data.compute_std_dev();
            assert!(std_dev > 0.0);
            // Known std dev for 1,2,3,4,5 is sqrt(2) ≈ 1.414
            assert!((std_dev - 1.414).abs() < 0.01);
        }

        #[test]
        fn test_scalar_and_simd_match() {
            let values: Vec<f64> = (0..150).map(|i| f64::from(i) / 10.0).collect();
            let data = ViolinData::from_vec("Test", values);

            let x = 7.5;
            let bandwidth = 0.5;

            let scalar = data.kde_at_point_scalar(x, bandwidth);
            let simd = data.kde_at_point_simd(x, bandwidth);

            assert!((scalar - simd).abs() < 1e-10);
        }

        #[test]
        fn test_simd_unaligned() {
            // Test SIMD path with values not divisible by 4
            let values: Vec<f64> = (0..103).map(|i| f64::from(i) / 10.0).collect();
            let data = ViolinData::from_vec("Test", values);

            let result = data.kde_at_point_simd(5.0, 0.5);
            assert!(result.is_finite());
            assert!(result > 0.0);
        }
    }

    mod orientation_tests {
        use super::*;

        #[test]
        fn test_default_is_vertical() {
            assert_eq!(ViolinOrientation::default(), ViolinOrientation::Vertical);
        }

        #[test]
        fn test_equality() {
            assert_eq!(ViolinOrientation::Vertical, ViolinOrientation::Vertical);
            assert_eq!(ViolinOrientation::Horizontal, ViolinOrientation::Horizontal);
            assert_ne!(ViolinOrientation::Vertical, ViolinOrientation::Horizontal);
        }
    }

    mod plot_construction_tests {
        use super::*;

        #[test]
        fn test_new() {
            let plot = ViolinPlot::new();
            assert!(plot.violins.is_empty());
        }

        #[test]
        fn test_default() {
            let plot = ViolinPlot::default();
            assert!(plot.violins.is_empty());
        }

        #[test]
        fn test_with_data() {
            let data = vec![ViolinData::new("A", &[1.0, 2.0, 3.0])];
            let plot = ViolinPlot::with_data(data);
            assert_eq!(plot.violins.len(), 1);
        }

        #[test]
        fn test_orientation() {
            let plot = ViolinPlot::new().orientation(ViolinOrientation::Horizontal);
            assert_eq!(plot.orientation, ViolinOrientation::Horizontal);
        }

        #[test]
        fn test_show_box() {
            let plot = ViolinPlot::new().show_box(false);
            assert!(!plot.show_box);
        }

        #[test]
        fn test_show_median() {
            let plot = ViolinPlot::new().show_median(false);
            assert!(!plot.show_median);
        }

        #[test]
        fn test_kde_points() {
            let plot = ViolinPlot::new().kde_points(100);
            assert_eq!(plot.kde_points, 100);
        }

        #[test]
        fn test_kde_points_clamped_min() {
            let plot = ViolinPlot::new().kde_points(5);
            assert_eq!(plot.kde_points, 10);
        }

        #[test]
        fn test_kde_points_clamped_max() {
            let plot = ViolinPlot::new().kde_points(500);
            assert_eq!(plot.kde_points, 200);
        }

        #[test]
        fn test_title() {
            let plot = ViolinPlot::new().title("My Plot");
            assert_eq!(plot.title.as_deref(), Some("My Plot"));
        }

        #[test]
        fn test_background() {
            let plot = ViolinPlot::new().background(Color::Black);
            assert_eq!(plot.background, Some(Color::Black));
        }

        #[test]
        fn test_range() {
            let plot = ViolinPlot::new().range(0.0, 100.0);
            assert_eq!(plot.min_value, Some(0.0));
            assert_eq!(plot.max_value, Some(100.0));
        }

        #[test]
        fn test_builder_chaining() {
            let data = vec![ViolinData::new("A", &[1.0, 2.0, 3.0])];
            let plot = ViolinPlot::with_data(data)
                .orientation(ViolinOrientation::Horizontal)
                .show_box(false)
                .show_median(true)
                .kde_points(75)
                .title("Test")
                .background(Color::Black)
                .range(0.0, 10.0);

            assert_eq!(plot.orientation, ViolinOrientation::Horizontal);
            assert!(!plot.show_box);
            assert!(plot.show_median);
            assert_eq!(plot.kde_points, 75);
            assert_eq!(plot.title.as_deref(), Some("Test"));
            assert_eq!(plot.background, Some(Color::Black));
        }
    }

    mod range_calculation_tests {
        use super::*;

        #[test]
        fn test_global_range_empty() {
            let plot = ViolinPlot::new();
            let (min, max) = plot.global_range();
            assert_eq!(min, 0.0);
            assert_eq!(max, 1.0);
        }

        #[test]
        fn test_global_range_single_violin() {
            let data = vec![ViolinData::new("A", &[1.0, 2.0, 3.0, 4.0, 5.0])];
            let plot = ViolinPlot::with_data(data);
            let (min, max) = plot.global_range();
            assert!(min < 1.0); // Includes padding
            assert!(max > 5.0);
        }

        #[test]
        fn test_global_range_multiple_violins() {
            let data = vec![ViolinData::new("A", &[1.0, 2.0]), ViolinData::new("B", &[3.0, 10.0])];
            let plot = ViolinPlot::with_data(data);
            let (min, max) = plot.global_range();
            assert!(min < 1.0);
            assert!(max > 10.0);
        }

        #[test]
        fn test_global_range_explicit() {
            let data = vec![ViolinData::new("A", &[1.0, 5.0])];
            let plot = ViolinPlot::with_data(data).range(0.0, 100.0);
            let (min, max) = plot.global_range();
            assert_eq!(min, 0.0);
            assert_eq!(max, 100.0);
        }

        #[test]
        fn test_global_range_invalid_explicit() {
            // Invalid range (min >= max) should fall back to auto
            let data = vec![ViolinData::new("A", &[1.0, 5.0])];
            let plot = ViolinPlot::with_data(data).range(100.0, 0.0);
            let (min, max) = plot.global_range();
            // Should compute from data, not use invalid range
            assert!(min < 1.0);
            assert!(max > 5.0);
        }

        #[test]
        fn test_global_range_with_nan() {
            let data = vec![ViolinData::new("A", &[1.0, f64::NAN, 5.0])];
            let plot = ViolinPlot::with_data(data);
            let (min, max) = plot.global_range();
            assert!(min.is_finite());
            assert!(max.is_finite());
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
        fn test_render_empty() {
            let (area, mut buf) = create_test_buffer(60, 20);
            let plot = ViolinPlot::new();
            plot.render(area, &mut buf);
            // Should not panic
        }

        #[test]
        fn test_render_zero_area() {
            let area = Rect::new(0, 0, 0, 0);
            let mut buf = Buffer::empty(Rect::new(0, 0, 1, 1));
            let plot = ViolinPlot::new();
            plot.render(area, &mut buf);
            // Should not panic
        }

        #[test]
        fn test_render_vertical_single() {
            let (area, mut buf) = create_test_buffer(30, 15);
            let data = vec![ViolinData::new("A", &[1.0, 2.0, 3.0, 4.0, 5.0])];
            let plot = ViolinPlot::with_data(data);
            plot.render(area, &mut buf);
        }

        #[test]
        fn test_render_vertical_multiple() {
            let (area, mut buf) = create_test_buffer(60, 20);
            let data = vec![
                ViolinData::new("A", &[1.0, 2.0, 3.0]).with_color(Color::Blue),
                ViolinData::new("B", &[2.0, 3.0, 4.0]).with_color(Color::Red),
                ViolinData::new("C", &[3.0, 4.0, 5.0]).with_color(Color::Green),
            ];
            let plot = ViolinPlot::with_data(data);
            plot.render(area, &mut buf);
        }

        #[test]
        fn test_render_horizontal_single() {
            let (area, mut buf) = create_test_buffer(60, 10);
            let data = vec![ViolinData::new("A", &[1.0, 2.0, 3.0, 4.0, 5.0])];
            let plot = ViolinPlot::with_data(data).orientation(ViolinOrientation::Horizontal);
            plot.render(area, &mut buf);
        }

        #[test]
        fn test_render_horizontal_multiple() {
            let (area, mut buf) = create_test_buffer(60, 20);
            let data = vec![
                ViolinData::new("A", &[1.0, 2.0, 3.0]),
                ViolinData::new("B", &[2.0, 3.0, 4.0]),
            ];
            let plot = ViolinPlot::with_data(data).orientation(ViolinOrientation::Horizontal);
            plot.render(area, &mut buf);
        }

        #[test]
        fn test_render_with_title() {
            let (area, mut buf) = create_test_buffer(60, 20);
            let data = vec![ViolinData::new("A", &[1.0, 2.0, 3.0])];
            let plot = ViolinPlot::with_data(data).title("My Violin Plot");
            plot.render(area, &mut buf);
        }

        #[test]
        fn test_render_with_background() {
            let (area, mut buf) = create_test_buffer(30, 15);
            let data = vec![ViolinData::new("A", &[1.0, 2.0, 3.0])];
            let plot = ViolinPlot::with_data(data).background(Color::Black);
            plot.render(area, &mut buf);
        }

        #[test]
        fn test_render_no_median() {
            let (area, mut buf) = create_test_buffer(30, 15);
            let data = vec![ViolinData::new("A", &[1.0, 2.0, 3.0])];
            let plot = ViolinPlot::with_data(data).show_median(false);
            plot.render(area, &mut buf);
        }

        #[test]
        fn test_render_small_area() {
            let (area, mut buf) = create_test_buffer(5, 5);
            let data = vec![ViolinData::new("A", &[1.0, 2.0, 3.0])];
            let plot = ViolinPlot::with_data(data);
            plot.render(area, &mut buf);
            // Should not panic with small area
        }

        #[test]
        fn test_render_narrow_vertical() {
            let (area, mut buf) = create_test_buffer(10, 20);
            let data = vec![ViolinData::new("A", &[1.0, 2.0, 3.0])];
            let plot = ViolinPlot::with_data(data);
            plot.render(area, &mut buf);
        }

        #[test]
        fn test_render_short_horizontal() {
            let (area, mut buf) = create_test_buffer(60, 5);
            let data = vec![ViolinData::new("A", &[1.0, 2.0, 3.0])];
            let plot = ViolinPlot::with_data(data).orientation(ViolinOrientation::Horizontal);
            plot.render(area, &mut buf);
        }

        #[test]
        fn test_render_large_dataset() {
            let (area, mut buf) = create_test_buffer(80, 30);
            let values: Vec<f64> = (0..200).map(|i| f64::from(i) / 20.0).collect();
            let data = vec![ViolinData::from_vec("Large", values)];
            let plot = ViolinPlot::with_data(data);
            plot.render(area, &mut buf);
        }

        #[test]
        fn test_render_with_single_kde_point() {
            let (area, mut buf) = create_test_buffer(30, 15);
            let data = vec![ViolinData::new("A", &[1.0, 2.0, 3.0])];
            let plot = ViolinPlot::with_data(data).kde_points(10);
            plot.render(area, &mut buf);
        }

        #[test]
        fn test_render_horizontal_show_box() {
            let (area, mut buf) = create_test_buffer(60, 15);
            let data = vec![ViolinData::new("A", &[1.0, 2.0, 3.0, 4.0, 5.0])];
            let plot = ViolinPlot::with_data(data)
                .orientation(ViolinOrientation::Horizontal)
                .show_box(true);
            plot.render(area, &mut buf);
        }

        #[test]
        fn test_render_horizontal_show_median() {
            let (area, mut buf) = create_test_buffer(60, 15);
            let data = vec![ViolinData::new("A", &[1.0, 2.0, 3.0, 4.0, 5.0])];
            let plot = ViolinPlot::with_data(data)
                .orientation(ViolinOrientation::Horizontal)
                .show_median(true);
            plot.render(area, &mut buf);
        }

        #[test]
        fn test_render_very_narrow() {
            let (area, mut buf) = create_test_buffer(3, 10);
            let data = vec![ViolinData::new("A", &[1.0, 2.0, 3.0])];
            let plot = ViolinPlot::with_data(data);
            plot.render(area, &mut buf);
        }

        #[test]
        fn test_render_vertical_many_violins() {
            let (area, mut buf) = create_test_buffer(15, 30);
            let data: Vec<ViolinData> = (0..5)
                .map(|i| {
                    ViolinData::new(
                        format!("V{i}"),
                        &[f64::from(i), f64::from(i) + 1.0, f64::from(i) + 2.0],
                    )
                })
                .collect();
            let plot = ViolinPlot::with_data(data);
            plot.render(area, &mut buf);
        }

        #[test]
        fn test_render_horizontal_many_violins() {
            let (area, mut buf) = create_test_buffer(60, 10);
            let data: Vec<ViolinData> = (0..5)
                .map(|i| {
                    ViolinData::new(
                        format!("V{i}"),
                        &[f64::from(i), f64::from(i) + 1.0, f64::from(i) + 2.0],
                    )
                })
                .collect();
            let plot = ViolinPlot::with_data(data).orientation(ViolinOrientation::Horizontal);
            plot.render(area, &mut buf);
        }

        #[test]
        fn test_render_vertical_with_box_and_median() {
            let (area, mut buf) = create_test_buffer(40, 20);
            let data = vec![ViolinData::new("Test", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])];
            let plot = ViolinPlot::with_data(data).show_box(true).show_median(true);
            plot.render(area, &mut buf);
        }

        #[test]
        fn test_render_with_explicit_range() {
            let (area, mut buf) = create_test_buffer(40, 15);
            let data = vec![ViolinData::new("A", &[3.0, 4.0, 5.0])];
            let plot = ViolinPlot::with_data(data).range(0.0, 10.0);
            plot.render(area, &mut buf);
        }

        #[test]
        fn test_compute_kde_single_point_data() {
            let mut data = ViolinData::new("Single", &[5.0]);
            data.compute_kde(20);
            assert!(data.densities.is_some());
        }

        #[test]
        fn test_render_title_clipping() {
            let (area, mut buf) = create_test_buffer(15, 10);
            let data = vec![ViolinData::new("A", &[1.0, 2.0, 3.0])];
            let plot = ViolinPlot::with_data(data).title("Very Long Title That Will Be Clipped");
            plot.render(area, &mut buf);
        }
    }
}
