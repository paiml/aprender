//! Box and Violin plot implementations.
//!
//! Box plots display the distribution of data through quartiles.
//! Violin plots extend this with kernel density estimation.

use crate::color::Rgba;
use crate::error::{Error, Result};
use crate::framebuffer::Framebuffer;
use crate::render::{draw_line, draw_rect, i32_px};

/// Statistics computed for a box plot.
#[derive(Debug, Clone)]
pub struct BoxStats {
    /// Minimum value (excluding outliers)
    pub min: f32,
    /// First quartile (25th percentile)
    pub q1: f32,
    /// Median (50th percentile)
    pub median: f32,
    /// Third quartile (75th percentile)
    pub q3: f32,
    /// Maximum value (excluding outliers)
    pub max: f32,
    /// Interquartile range (Q3 - Q1)
    pub iqr: f32,
    /// Outlier values
    pub outliers: Vec<f32>,
}

impl BoxStats {
    /// Compute box plot statistics from data.
    ///
    /// Uses the 1.5 * IQR rule for outlier detection.
    pub fn from_data(data: &[f32]) -> Option<Self> {
        if data.is_empty() {
            return None;
        }

        let mut sorted: Vec<f32> = data.iter().copied().filter(|x| x.is_finite()).collect();
        if sorted.is_empty() {
            return None;
        }
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let q1 = percentile(&sorted, 25.0);
        let median = percentile(&sorted, 50.0);
        let q3 = percentile(&sorted, 75.0);
        let iqr = q3 - q1;

        // Whisker bounds: 1.5 * IQR from Q1 and Q3
        let lower_fence = q1 - 1.5 * iqr;
        let upper_fence = q3 + 1.5 * iqr;

        // Find actual min/max within fences
        let min = sorted.iter().copied().find(|&x| x >= lower_fence).unwrap_or(sorted[0]);
        let max = sorted.iter().rev().copied().find(|&x| x <= upper_fence).unwrap_or(sorted[n - 1]);

        // Collect outliers
        let outliers: Vec<f32> =
            sorted.iter().copied().filter(|&x| x < lower_fence || x > upper_fence).collect();

        Some(Self { min, q1, median, q3, max, iqr, outliers })
    }
}

/// Calculate percentile using linear interpolation.
fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let k = (p / 100.0) * (sorted.len() - 1) as f32;
    let f = k.floor() as usize;
    let c = k.ceil() as usize;

    if f == c || c >= sorted.len() {
        sorted[f.min(sorted.len() - 1)]
    } else {
        let d = k - f as f32;
        sorted[f] * (1.0 - d) + sorted[c] * d
    }
}

/// Box plot visualization.
#[derive(Debug, Clone)]
pub struct BoxPlot {
    /// Data groups (each group is a separate box)
    groups: Vec<Vec<f32>>,
    /// Group labels
    labels: Vec<String>,
    /// Box fill color
    fill_color: Rgba,
    /// Outline color
    outline_color: Rgba,
    /// Median line color
    median_color: Rgba,
    /// Outlier color
    outlier_color: Rgba,
    /// Image width
    width: u32,
    /// Image height
    height: u32,
    /// Margin around plot
    margin: u32,
    /// Box width as fraction of available space
    box_width: f32,
    /// Show outliers
    show_outliers: bool,
    /// Show notches (confidence interval for median) - reserved for future use
    #[allow(dead_code)]
    show_notches: bool,
}

impl Default for BoxPlot {
    fn default() -> Self {
        Self::new()
    }
}

impl BoxPlot {
    /// Create a new box plot builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            groups: Vec::new(),
            labels: Vec::new(),
            fill_color: Rgba::new(100, 149, 237, 200), // Cornflower blue
            outline_color: Rgba::BLACK,
            median_color: Rgba::new(255, 140, 0, 255), // Dark orange
            outlier_color: Rgba::new(200, 50, 50, 255),
            width: 600,
            height: 400,
            margin: 50,
            box_width: 0.6,
            show_outliers: true,
            show_notches: false,
        }
    }

    /// Add a data group.
    #[must_use]
    pub fn add_group(mut self, data: &[f32], label: &str) -> Self {
        self.groups.push(data.to_vec());
        self.labels.push(label.to_string());
        self
    }

    /// Set multiple data groups at once.
    #[must_use]
    pub fn data(mut self, groups: Vec<Vec<f32>>) -> Self {
        self.groups = groups;
        self
    }

    /// Set group labels.
    #[must_use]
    pub fn labels(mut self, labels: &[&str]) -> Self {
        self.labels = labels.iter().map(|s| (*s).to_string()).collect();
        self
    }

    /// Set box fill color.
    #[must_use]
    pub fn fill_color(mut self, color: Rgba) -> Self {
        self.fill_color = color;
        self
    }

    /// Set outline color.
    #[must_use]
    pub fn outline_color(mut self, color: Rgba) -> Self {
        self.outline_color = color;
        self
    }

    /// Set median line color.
    #[must_use]
    pub fn median_color(mut self, color: Rgba) -> Self {
        self.median_color = color;
        self
    }

    /// Set margin.
    #[must_use]
    pub fn margin(mut self, margin: u32) -> Self {
        self.margin = margin;
        self
    }

    /// Set box width as fraction (0.0 to 1.0).
    #[must_use]
    pub fn box_width(mut self, width: f32) -> Self {
        self.box_width = width.clamp(0.1, 1.0);
        self
    }

    /// Show or hide outliers.
    #[must_use]
    pub fn show_outliers(mut self, show: bool) -> Self {
        self.show_outliers = show;
        self
    }

    /// Build the box plot.
    ///
    /// # Errors
    ///
    /// Returns an error if no data groups are provided.
    pub fn build(self) -> Result<BuiltBoxPlot> {
        if self.groups.is_empty() {
            return Err(Error::EmptyData);
        }

        // Compute statistics for each group
        let stats: Vec<BoxStats> =
            self.groups.iter().filter_map(|g| BoxStats::from_data(g)).collect();

        if stats.is_empty() {
            return Err(Error::EmptyData);
        }

        Ok(BuiltBoxPlot {
            stats,
            labels: self.labels,
            fill_color: self.fill_color,
            outline_color: self.outline_color,
            median_color: self.median_color,
            outlier_color: self.outlier_color,
            width: self.width,
            height: self.height,
            margin: self.margin,
            box_width: self.box_width,
            show_outliers: self.show_outliers,
        })
    }
}

/// A built box plot ready for rendering.
#[derive(Debug)]
pub struct BuiltBoxPlot {
    stats: Vec<BoxStats>,
    labels: Vec<String>,
    fill_color: Rgba,
    outline_color: Rgba,
    median_color: Rgba,
    outlier_color: Rgba,
    width: u32,
    height: u32,
    margin: u32,
    box_width: f32,
    show_outliers: bool,
}

impl BuiltBoxPlot {
    /// Get number of groups.
    #[must_use]
    pub fn num_groups(&self) -> usize {
        self.stats.len()
    }

    /// Get statistics for a group.
    #[must_use]
    pub fn stats(&self, index: usize) -> Option<&BoxStats> {
        self.stats.get(index)
    }

    /// Get group labels.
    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Render to a new framebuffer.
    ///
    /// # Errors
    ///
    /// Returns an error if framebuffer creation fails.
    pub fn to_framebuffer(&self) -> Result<Framebuffer> {
        let mut fb = Framebuffer::new(self.width, self.height)?;
        fb.clear(Rgba::WHITE);
        self.render(&mut fb)?;
        Ok(fb)
    }

    /// Render onto an existing framebuffer.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering fails.
    pub fn render(&self, fb: &mut Framebuffer) -> Result<()> {
        let plot_width = self.width.saturating_sub(2 * self.margin);
        let plot_height = self.height.saturating_sub(2 * self.margin);

        if plot_width == 0 || plot_height == 0 {
            return Err(Error::Rendering("Plot area too small".into()));
        }

        let n_groups = self.stats.len();
        if n_groups == 0 {
            return Ok(());
        }

        // Find global min/max for y-axis scaling
        let (global_min, global_max) = self.stats.iter().fold((f32::MAX, f32::MIN), |acc, s| {
            let min = if self.show_outliers && !s.outliers.is_empty() {
                s.outliers.iter().copied().fold(s.min, f32::min)
            } else {
                s.min
            };
            let max = if self.show_outliers && !s.outliers.is_empty() {
                s.outliers.iter().copied().fold(s.max, f32::max)
            } else {
                s.max
            };
            (acc.0.min(min), acc.1.max(max))
        });

        // Add padding to y range
        let y_range = global_max - global_min;
        let y_padding = y_range * 0.1;
        let y_min = global_min - y_padding;
        let y_max = global_max + y_padding;

        // Calculate box positions
        let group_width = plot_width as f32 / n_groups as f32;
        let actual_box_width = (group_width * self.box_width) as u32;

        for (i, stats) in self.stats.iter().enumerate() {
            let center_x = self.margin + (i as f32 * group_width + group_width / 2.0) as u32;
            let half_box = actual_box_width / 2;

            // Map y values to pixel coordinates
            let map_y = |val: f32| -> u32 {
                let normalized = (val - y_min) / (y_max - y_min);
                (self.margin + plot_height) - (normalized * plot_height as f32) as u32
            };

            let y_min_px = map_y(stats.min);
            let y_q1 = map_y(stats.q1);
            let y_median = map_y(stats.median);
            let y_q3 = map_y(stats.q3);
            let y_max_px = map_y(stats.max);

            Self::draw_whiskers(
                fb,
                center_x,
                (y_min_px, y_q1, y_q3, y_max_px),
                half_box,
                self.outline_color,
            );
            Self::draw_box_and_median(
                fb,
                center_x,
                half_box,
                actual_box_width,
                y_q1,
                y_q3,
                y_median,
                self.fill_color,
                self.outline_color,
                self.median_color,
            );
            Self::draw_outlier_markers(
                fb,
                stats,
                center_x,
                self.show_outliers,
                self.outlier_color,
                &map_y,
            );
        }

        Ok(())
    }

    /// Draw whisker lines and caps. ys = (y_min, y_q1, y_q3, y_max).
    fn draw_whiskers(
        fb: &mut Framebuffer,
        cx: u32,
        ys: (u32, u32, u32, u32),
        half_box: u32,
        color: Rgba,
    ) {
        let (y_min, y_q1, y_q3, y_max) = ys;
        draw_line(fb, i32_px(cx), i32_px(y_min), i32_px(cx), i32_px(y_q1), color);
        draw_line(fb, i32_px(cx), i32_px(y_q3), i32_px(cx), i32_px(y_max), color);
        let cap = half_box / 2;
        draw_line(fb, i32_px(cx - cap), i32_px(y_min), i32_px(cx + cap), i32_px(y_min), color);
        draw_line(fb, i32_px(cx - cap), i32_px(y_max), i32_px(cx + cap), i32_px(y_max), color);
    }

    /// Draw filled box, outline, and median line.
    #[allow(clippy::too_many_arguments)]
    fn draw_box_and_median(
        fb: &mut Framebuffer,
        cx: u32,
        half_box: u32,
        box_w: u32,
        y_q1: u32,
        y_q3: u32,
        y_med: u32,
        fill: Rgba,
        outline: Rgba,
        median_color: Rgba,
    ) {
        let left = cx.saturating_sub(half_box);
        let top = y_q3.min(y_q1);
        let bottom = y_q3.max(y_q1);
        let height = bottom.saturating_sub(top);
        draw_rect(fb, i32_px(left), i32_px(top), box_w, height, fill);
        Self::draw_box_outline(fb, left, top, bottom, box_w, outline);
        draw_line(
            fb,
            i32_px(left),
            i32_px(y_med),
            i32_px(left + box_w),
            i32_px(y_med),
            median_color,
        );
    }

    /// Draw box outline (4 sides).
    pub(crate) fn draw_box_outline(
        fb: &mut Framebuffer,
        left: u32,
        top: u32,
        bottom: u32,
        w: u32,
        color: Rgba,
    ) {
        draw_line(fb, i32_px(left), i32_px(top), i32_px(left + w), i32_px(top), color);
        draw_line(fb, i32_px(left), i32_px(bottom), i32_px(left + w), i32_px(bottom), color);
        draw_line(fb, i32_px(left), i32_px(top), i32_px(left), i32_px(bottom), color);
        draw_line(fb, i32_px(left + w), i32_px(top), i32_px(left + w), i32_px(bottom), color);
    }

    /// Draw outlier markers as crosses.
    fn draw_outlier_markers(
        fb: &mut Framebuffer,
        stats: &BoxStats,
        cx: u32,
        show: bool,
        color: Rgba,
        map_y: &dyn Fn(f32) -> u32,
    ) {
        if !show {
            return;
        }
        for &outlier in &stats.outliers {
            let y = map_y(outlier);
            draw_line(fb, i32_px(cx - 2), i32_px(y), i32_px(cx + 2), i32_px(y), color);
            draw_line(fb, i32_px(cx), i32_px(y - 2), i32_px(cx), i32_px(y + 2), color);
        }
    }
}

/// Violin plot visualization combining box plot with kernel density estimation.
#[derive(Debug, Clone)]
pub struct ViolinPlot {
    /// Data groups
    groups: Vec<Vec<f32>>,
    /// Group labels
    labels: Vec<String>,
    /// Fill color
    fill_color: Rgba,
    /// Outline color
    outline_color: Rgba,
    /// Show inner box plot
    show_box: bool,
    /// Bandwidth for KDE (None = automatic)
    bandwidth: Option<f32>,
    /// Image width
    width: u32,
    /// Image height
    height: u32,
    /// Margin
    margin: u32,
    /// Violin width as fraction
    violin_width: f32,
}

impl Default for ViolinPlot {
    fn default() -> Self {
        Self::new()
    }
}

impl ViolinPlot {
    /// Create a new violin plot builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            groups: Vec::new(),
            labels: Vec::new(),
            fill_color: Rgba::new(147, 112, 219, 180), // Medium purple
            outline_color: Rgba::BLACK,
            show_box: true,
            bandwidth: None,
            width: 600,
            height: 400,
            margin: 50,
            violin_width: 0.8,
        }
    }

    /// Add a data group.
    #[must_use]
    pub fn add_group(mut self, data: &[f32], label: &str) -> Self {
        self.groups.push(data.to_vec());
        self.labels.push(label.to_string());
        self
    }

    /// Set multiple data groups.
    #[must_use]
    pub fn data(mut self, groups: Vec<Vec<f32>>) -> Self {
        self.groups = groups;
        self
    }

    /// Set fill color.
    #[must_use]
    pub fn fill_color(mut self, color: Rgba) -> Self {
        self.fill_color = color;
        self
    }

    /// Show or hide inner box plot.
    #[must_use]
    pub fn show_box(mut self, show: bool) -> Self {
        self.show_box = show;
        self
    }

    /// Set KDE bandwidth (None for automatic).
    #[must_use]
    pub fn bandwidth(mut self, bw: Option<f32>) -> Self {
        self.bandwidth = bw;
        self
    }

    /// Set margin.
    #[must_use]
    pub fn margin(mut self, margin: u32) -> Self {
        self.margin = margin;
        self
    }

    /// Build the violin plot.
    ///
    /// # Errors
    ///
    /// Returns an error if no data is provided.
    pub fn build(self) -> Result<BuiltViolinPlot> {
        if self.groups.is_empty() {
            return Err(Error::EmptyData);
        }

        // Compute KDE for each group
        let kdes: Vec<Vec<(f32, f32)>> =
            self.groups.iter().map(|g| compute_kde(g, self.bandwidth, 50)).collect();

        let stats: Vec<Option<BoxStats>> =
            self.groups.iter().map(|g| BoxStats::from_data(g)).collect();

        Ok(BuiltViolinPlot {
            kdes,
            stats,
            labels: self.labels,
            fill_color: self.fill_color,
            outline_color: self.outline_color,
            show_box: self.show_box,
            width: self.width,
            height: self.height,
            margin: self.margin,
            violin_width: self.violin_width,
        })
    }
}

/// Compute kernel density estimation using Gaussian kernel.
fn compute_kde(data: &[f32], bandwidth: Option<f32>, n_points: usize) -> Vec<(f32, f32)> {
    if data.is_empty() {
        return Vec::new();
    }

    let clean: Vec<f32> = data.iter().copied().filter(|x| x.is_finite()).collect();
    if clean.is_empty() {
        return Vec::new();
    }

    let min_val = clean.iter().copied().fold(f32::MAX, f32::min);
    let max_val = clean.iter().copied().fold(f32::MIN, f32::max);
    let range = max_val - min_val;

    if range == 0.0 {
        return vec![(min_val, 1.0)];
    }

    // Silverman's rule of thumb for bandwidth
    let std_dev = {
        let mean = clean.iter().sum::<f32>() / clean.len() as f32;
        let variance = clean.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / clean.len() as f32;
        variance.sqrt()
    };
    let h = bandwidth.unwrap_or_else(|| 1.06 * std_dev * (clean.len() as f32).powf(-0.2));
    let h = h.max(range * 0.01); // Minimum bandwidth

    // Extend range slightly for smoother edges
    let padding = range * 0.1;
    let x_min = min_val - padding;
    let x_max = max_val + padding;

    let mut kde_points = Vec::with_capacity(n_points);
    let step = (x_max - x_min) / (n_points - 1) as f32;

    for i in 0..n_points {
        let x = x_min + i as f32 * step;
        let density: f32 = clean
            .iter()
            .map(|&xi| {
                let u = (x - xi) / h;
                (-0.5 * u * u).exp() / (2.506_628 * h) // Gaussian kernel
            })
            .sum();
        let density = density / clean.len() as f32;
        kde_points.push((x, density));
    }

    // Normalize to max density = 1
    let max_density = kde_points.iter().map(|&(_, d)| d).fold(0.0f32, f32::max);
    if max_density > 0.0 {
        for point in &mut kde_points {
            point.1 /= max_density;
        }
    }

    kde_points
}

/// A built violin plot ready for rendering.
#[derive(Debug)]
pub struct BuiltViolinPlot {
    kdes: Vec<Vec<(f32, f32)>>,
    stats: Vec<Option<BoxStats>>,
    labels: Vec<String>,
    fill_color: Rgba,
    outline_color: Rgba,
    show_box: bool,
    width: u32,
    height: u32,
    margin: u32,
    violin_width: f32,
}

impl BuiltViolinPlot {
    /// Get number of groups.
    #[must_use]
    pub fn num_groups(&self) -> usize {
        self.kdes.len()
    }

    /// Get group labels.
    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Render to a new framebuffer.
    ///
    /// # Errors
    ///
    /// Returns an error if framebuffer creation fails.
    pub fn to_framebuffer(&self) -> Result<Framebuffer> {
        let mut fb = Framebuffer::new(self.width, self.height)?;
        fb.clear(Rgba::WHITE);
        self.render(&mut fb)?;
        Ok(fb)
    }

    /// Render onto an existing framebuffer.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering fails.
    pub fn render(&self, fb: &mut Framebuffer) -> Result<()> {
        let plot_width = self.width.saturating_sub(2 * self.margin);
        let plot_height = self.height.saturating_sub(2 * self.margin);

        if plot_width == 0 || plot_height == 0 {
            return Err(Error::Rendering("Plot area too small".into()));
        }

        let n_groups = self.kdes.len();
        if n_groups == 0 {
            return Ok(());
        }

        let (y_min, y_max) = Self::compute_y_range(&self.kdes);

        let group_width = plot_width as f32 / n_groups as f32;
        let max_violin_half_width = (group_width * self.violin_width / 2.0) as u32;

        for (i, kde) in self.kdes.iter().enumerate() {
            if kde.is_empty() {
                continue;
            }

            let center_x = self.margin + (i as f32 * group_width + group_width / 2.0) as u32;

            // Map y value to pixel
            let map_y = |val: f32| -> u32 {
                let normalized = (val - y_min) / (y_max - y_min);
                (self.margin + plot_height) - (normalized * plot_height as f32) as u32
            };

            Self::draw_violin_fill(
                fb,
                kde,
                center_x,
                max_violin_half_width,
                self.fill_color,
                &map_y,
            );
            Self::draw_violin_outline(
                fb,
                kde,
                center_x,
                max_violin_half_width,
                self.outline_color,
                &map_y,
            );
            if self.show_box {
                if let Some(ref stats) = self.stats[i] {
                    Self::draw_inner_box(fb, stats, center_x, max_violin_half_width, &map_y);
                }
            }
        }

        Ok(())
    }

    /// Compute padded y-axis range from KDE data.
    fn compute_y_range(kdes: &[Vec<(f32, f32)>]) -> (f32, f32) {
        let (gmin, gmax) = kdes.iter().fold((f32::MAX, f32::MIN), |acc, kde| {
            if kde.is_empty() {
                return acc;
            }
            let min = kde.iter().map(|&(y, _)| y).fold(f32::MAX, f32::min);
            let max = kde.iter().map(|&(y, _)| y).fold(f32::MIN, f32::max);
            (acc.0.min(min), acc.1.max(max))
        });
        let pad = (gmax - gmin) * 0.05;
        (gmin - pad, gmax + pad)
    }

    /// Draw the filled violin shape.
    fn draw_violin_fill(
        fb: &mut Framebuffer,
        kde: &[(f32, f32)],
        cx: u32,
        max_hw: u32,
        color: Rgba,
        map_y: &dyn Fn(f32) -> u32,
    ) {
        for j in 0..kde.len().saturating_sub(1) {
            let (y1, d1) = kde[j];
            let (y2, d2) = kde[j + 1];
            let py1 = map_y(y1);
            let py2 = map_y(y2);
            let w1 = (d1 * max_hw as f32) as u32;
            let w2 = (d2 * max_hw as f32) as u32;
            for py in py2.min(py1)..=py1.max(py2) {
                let t = if py1 == py2 {
                    0.5
                } else {
                    (py as f32 - py1 as f32) / (py2 as f32 - py1 as f32)
                };
                let w = (w1 as f32 * (1.0 - t) + w2 as f32 * t) as u32;
                draw_line(
                    fb,
                    i32_px(cx.saturating_sub(w)),
                    i32_px(py),
                    i32_px(cx + w),
                    i32_px(py),
                    color,
                );
            }
        }
    }

    /// Draw the violin outline edges.
    fn draw_violin_outline(
        fb: &mut Framebuffer,
        kde: &[(f32, f32)],
        cx: u32,
        max_hw: u32,
        color: Rgba,
        map_y: &dyn Fn(f32) -> u32,
    ) {
        for j in 0..kde.len().saturating_sub(1) {
            let (y1, d1) = kde[j];
            let (y2, d2) = kde[j + 1];
            let py1 = map_y(y1);
            let py2 = map_y(y2);
            let w1 = (d1 * max_hw as f32) as i32;
            let w2 = (d2 * max_hw as f32) as i32;
            draw_line(fb, i32_px(cx) - w1, i32_px(py1), i32_px(cx) - w2, i32_px(py2), color);
            draw_line(fb, i32_px(cx) + w1, i32_px(py1), i32_px(cx) + w2, i32_px(py2), color);
        }
    }

    /// Draw inner box plot within a violin.
    fn draw_inner_box(
        fb: &mut Framebuffer,
        stats: &BoxStats,
        cx: u32,
        max_hw: u32,
        map_y: &dyn Fn(f32) -> u32,
    ) {
        let y_q1 = map_y(stats.q1);
        let y_med = map_y(stats.median);
        let y_q3 = map_y(stats.q3);
        let box_half = max_hw / 4;
        let left = cx.saturating_sub(box_half);
        let w = box_half * 2;
        let top = y_q3.min(y_q1);
        let bottom = y_q3.max(y_q1);
        draw_rect(fb, i32_px(left), i32_px(top), w, bottom.saturating_sub(top), Rgba::WHITE);
        BuiltBoxPlot::draw_box_outline(fb, left, top, bottom, w, Rgba::BLACK);
        draw_line(fb, i32_px(left), i32_px(y_med), i32_px(left + w), i32_px(y_med), Rgba::BLACK);
    }
}

impl batuta_common::display::WithDimensions for BoxPlot {
    fn set_dimensions(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
}

impl batuta_common::display::WithDimensions for ViolinPlot {
    fn set_dimensions(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use batuta_common::display::WithDimensions;

    #[test]
    fn test_box_stats_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let stats = BoxStats::from_data(&data).expect("operation should succeed");

        assert!((stats.median - 5.0).abs() < 0.01);
        // Q1 at 25th percentile: index 2 = value 3.0
        assert!((stats.q1 - 3.0).abs() < 0.5);
        // Q3 at 75th percentile: index 6 = value 7.0
        assert!((stats.q3 - 7.0).abs() < 0.5);
        assert!(stats.outliers.is_empty());
    }

    #[test]
    fn test_box_stats_with_outliers() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100 is an outlier
        let stats = BoxStats::from_data(&data).expect("operation should succeed");

        assert!(!stats.outliers.is_empty());
        assert!(stats.outliers.contains(&100.0));
    }

    #[test]
    fn test_box_stats_empty() {
        let data: Vec<f32> = vec![];
        assert!(BoxStats::from_data(&data).is_none());
    }

    #[test]
    fn test_box_stats_single() {
        let data = vec![42.0];
        let stats = BoxStats::from_data(&data).expect("operation should succeed");
        assert!((stats.median - 42.0).abs() < 0.01);
    }

    #[test]
    fn test_percentile() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&sorted, 0.0) - 1.0).abs() < 0.01);
        assert!((percentile(&sorted, 50.0) - 3.0).abs() < 0.01);
        assert!((percentile(&sorted, 100.0) - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_boxplot_build() {
        let plot = BoxPlot::new()
            .add_group(&[1.0, 2.0, 3.0, 4.0, 5.0], "Group A")
            .add_group(&[2.0, 4.0, 6.0, 8.0, 10.0], "Group B")
            .build()
            .expect("operation should succeed");

        assert_eq!(plot.num_groups(), 2);
    }

    #[test]
    fn test_boxplot_empty_error() {
        let result = BoxPlot::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_boxplot_render() {
        let plot = BoxPlot::new()
            .add_group(&[1.0, 2.0, 3.0, 4.0, 5.0], "A")
            .dimensions(200, 150)
            .build()
            .expect("operation should succeed");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert_eq!(fb.width(), 200);
        assert_eq!(fb.height(), 150);
    }

    #[test]
    fn test_violin_build() {
        let plot = ViolinPlot::new()
            .add_group(&[1.0, 2.0, 3.0, 4.0, 5.0], "Group A")
            .build()
            .expect("builder should produce valid result");

        assert_eq!(plot.num_groups(), 1);
    }

    #[test]
    fn test_violin_render() {
        let plot = ViolinPlot::new()
            .add_group(&[1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0], "A")
            .dimensions(200, 150)
            .build()
            .expect("operation should succeed");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert_eq!(fb.width(), 200);
        assert_eq!(fb.height(), 150);
    }

    #[test]
    fn test_kde_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kde = compute_kde(&data, None, 20);

        assert!(!kde.is_empty());
        // KDE should be normalized
        let max_d = kde.iter().map(|&(_, d)| d).fold(0.0f32, f32::max);
        assert!((max_d - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_kde_empty() {
        let data: Vec<f32> = vec![];
        let kde = compute_kde(&data, None, 20);
        assert!(kde.is_empty());
    }

    #[test]
    fn test_boxplot_data_method() {
        let groups = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let plot = BoxPlot::new()
            .data(groups)
            .labels(&["A", "B"])
            .build()
            .expect("builder should produce valid result");
        assert_eq!(plot.num_groups(), 2);
    }

    #[test]
    fn test_boxplot_colors() {
        let plot = BoxPlot::new()
            .add_group(&[1.0, 2.0, 3.0, 4.0, 5.0], "A")
            .fill_color(Rgba::RED)
            .outline_color(Rgba::BLACK)
            .median_color(Rgba::BLUE)
            .build()
            .expect("operation should succeed");
        let _ = plot.to_framebuffer().expect("operation should succeed");
    }

    #[test]
    fn test_boxplot_margin() {
        let plot = BoxPlot::new()
            .add_group(&[1.0, 2.0, 3.0, 4.0, 5.0], "A")
            .margin(10)
            .build()
            .expect("builder should produce valid result");
        let _ = plot.to_framebuffer().expect("operation should succeed");
    }

    #[test]
    fn test_boxplot_box_width() {
        let plot = BoxPlot::new()
            .add_group(&[1.0, 2.0, 3.0, 4.0, 5.0], "A")
            .box_width(0.8)
            .build()
            .expect("operation should succeed");
        let _ = plot.to_framebuffer().expect("operation should succeed");
    }

    #[test]
    fn test_boxplot_box_width_clamp() {
        let plot1 = BoxPlot::new()
            .add_group(&[1.0, 2.0, 3.0, 4.0, 5.0], "A")
            .box_width(2.0) // Should clamp to 1.0
            .build()
            .expect("operation should succeed");
        let _ = plot1.to_framebuffer().expect("operation should succeed");

        let plot2 = BoxPlot::new()
            .add_group(&[1.0, 2.0, 3.0, 4.0, 5.0], "A")
            .box_width(0.05) // Should clamp to 0.1
            .build()
            .expect("operation should succeed");
        let _ = plot2.to_framebuffer().expect("operation should succeed");
    }

    #[test]
    fn test_boxplot_show_outliers_false() {
        let plot = BoxPlot::new()
            .add_group(&[1.0, 2.0, 3.0, 4.0, 5.0, 100.0], "A") // Has outlier
            .show_outliers(false)
            .build()
            .expect("operation should succeed");
        let _ = plot.to_framebuffer().expect("operation should succeed");
    }

    #[test]
    fn test_boxplot_show_outliers_true() {
        let plot = BoxPlot::new()
            .add_group(&[1.0, 2.0, 3.0, 4.0, 5.0, 100.0], "A") // Has outlier
            .show_outliers(true)
            .build()
            .expect("operation should succeed");
        let _ = plot.to_framebuffer().expect("operation should succeed");
    }

    #[test]
    fn test_built_boxplot_stats_labels() {
        let plot = BoxPlot::new()
            .add_group(&[1.0, 2.0, 3.0, 4.0, 5.0], "Group A")
            .build()
            .expect("builder should produce valid result");

        assert!(plot.stats(0).is_some());
        assert!(plot.stats(99).is_none());
        assert_eq!(plot.labels(), &["Group A".to_string()]);
    }

    #[test]
    fn test_violin_data_method() {
        let groups = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let plot =
            ViolinPlot::new().data(groups).build().expect("builder should produce valid result");
        assert_eq!(plot.num_groups(), 1);
    }

    #[test]
    fn test_violin_fill_color() {
        let plot = ViolinPlot::new()
            .add_group(&[1.0, 2.0, 3.0, 4.0, 5.0], "A")
            .fill_color(Rgba::GREEN)
            .build()
            .expect("operation should succeed");
        let _ = plot.to_framebuffer().expect("operation should succeed");
    }

    #[test]
    fn test_violin_show_box_false() {
        let plot = ViolinPlot::new()
            .add_group(&[1.0, 2.0, 3.0, 4.0, 5.0], "A")
            .show_box(false)
            .build()
            .expect("operation should succeed");
        let _ = plot.to_framebuffer().expect("operation should succeed");
    }

    #[test]
    fn test_violin_bandwidth() {
        let plot = ViolinPlot::new()
            .add_group(&[1.0, 2.0, 3.0, 4.0, 5.0], "A")
            .bandwidth(Some(0.5))
            .build()
            .expect("operation should succeed");
        let _ = plot.to_framebuffer().expect("operation should succeed");
    }

    #[test]
    fn test_violin_margin() {
        let plot = ViolinPlot::new()
            .add_group(&[1.0, 2.0, 3.0, 4.0, 5.0], "A")
            .margin(20)
            .build()
            .expect("operation should succeed");
        let _ = plot.to_framebuffer().expect("operation should succeed");
    }

    #[test]
    fn test_violin_labels() {
        let plot = ViolinPlot::new()
            .add_group(&[1.0, 2.0, 3.0], "Test Label")
            .build()
            .expect("builder should produce valid result");
        assert_eq!(plot.labels(), &["Test Label".to_string()]);
    }

    #[test]
    fn test_violin_empty_error() {
        let result = ViolinPlot::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_box_stats_nan_filtered() {
        let data = vec![1.0, f32::NAN, 3.0, 4.0, 5.0];
        let stats = BoxStats::from_data(&data).expect("operation should succeed");
        assert!((stats.median - 3.5).abs() < 0.5); // NaN filtered out
    }

    #[test]
    fn test_box_stats_all_nan() {
        let data = vec![f32::NAN, f32::NAN];
        assert!(BoxStats::from_data(&data).is_none());
    }

    #[test]
    fn test_kde_constant_data() {
        let data = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let kde = compute_kde(&data, None, 10);
        assert_eq!(kde.len(), 1);
        assert!((kde[0].0 - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_kde_custom_bandwidth() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kde = compute_kde(&data, Some(0.3), 20);
        assert!(!kde.is_empty());
    }

    #[test]
    fn test_kde_nan_filtered() {
        let data = vec![1.0, f32::NAN, 3.0, 4.0, 5.0];
        let kde = compute_kde(&data, None, 20);
        assert!(!kde.is_empty());
    }

    #[test]
    fn test_kde_all_nan() {
        let data = vec![f32::NAN, f32::NAN];
        let kde = compute_kde(&data, None, 20);
        assert!(kde.is_empty());
    }

    #[test]
    fn test_percentile_empty() {
        let sorted: Vec<f32> = vec![];
        assert!((percentile(&sorted, 50.0) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_percentile_single() {
        let sorted = vec![42.0];
        assert!((percentile(&sorted, 50.0) - 42.0).abs() < 0.01);
    }

    #[test]
    fn test_boxplot_default() {
        let plot = BoxPlot::default();
        let result = plot.build();
        assert!(result.is_err()); // No data
    }

    #[test]
    fn test_violin_default() {
        let plot = ViolinPlot::default();
        let result = plot.build();
        assert!(result.is_err()); // No data
    }

    #[test]
    fn test_boxplot_render_tiny_margin() {
        let plot = BoxPlot::new()
            .add_group(&[1.0, 2.0, 3.0, 4.0, 5.0], "A")
            .dimensions(50, 50)
            .margin(30) // Margin larger than half dimensions
            .build()
            .expect("operation should succeed");
        // This should still render (plot area will be small but not zero)
        let _ = plot.to_framebuffer();
    }

    #[test]
    fn test_violin_render_tiny_margin() {
        let plot = ViolinPlot::new()
            .add_group(&[1.0, 2.0, 3.0, 4.0, 5.0], "A")
            .dimensions(50, 50)
            .margin(30)
            .build()
            .expect("operation should succeed");
        let _ = plot.to_framebuffer();
    }

    #[test]
    fn test_boxplot_debug_clone() {
        let plot = BoxPlot::new().add_group(&[1.0, 2.0, 3.0], "A");
        let plot2 = plot.clone();
        let _ = format!("{plot2:?}");
    }

    #[test]
    fn test_violin_debug_clone() {
        let plot = ViolinPlot::new().add_group(&[1.0, 2.0, 3.0], "A");
        let plot2 = plot.clone();
        let _ = format!("{plot2:?}");
    }

    #[test]
    fn test_box_stats_debug_clone() {
        let stats =
            BoxStats::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0]).expect("operation should succeed");
        let stats2 = stats.clone();
        let _ = format!("{stats2:?}");
    }

    #[test]
    fn test_built_boxplot_debug() {
        let built = BoxPlot::new()
            .add_group(&[1.0, 2.0, 3.0, 4.0, 5.0], "A")
            .build()
            .expect("builder should produce valid result");
        let _ = format!("{built:?}");
    }

    #[test]
    fn test_built_violin_debug() {
        let built = ViolinPlot::new()
            .add_group(&[1.0, 2.0, 3.0, 4.0, 5.0], "A")
            .build()
            .expect("builder should produce valid result");
        let _ = format!("{built:?}");
    }

    #[test]
    fn test_boxplot_all_empty_groups() {
        // Groups that result in no valid stats
        let result = BoxPlot::new().data(vec![vec![f32::NAN, f32::NAN]]).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_boxplot_multiple_groups_outliers() {
        let plot = BoxPlot::new()
            .add_group(&[1.0, 2.0, 3.0, 4.0, 5.0, 100.0, -100.0], "A")
            .add_group(&[10.0, 20.0, 30.0, 40.0, 50.0], "B")
            .show_outliers(true)
            .dimensions(300, 200)
            .build()
            .expect("operation should succeed");
        let _ = plot.to_framebuffer().expect("operation should succeed");
    }

    #[test]
    fn test_violin_multiple_groups() {
        let plot = ViolinPlot::new()
            .add_group(&[1.0, 2.0, 3.0, 4.0, 5.0], "A")
            .add_group(&[10.0, 20.0, 30.0, 40.0, 50.0], "B")
            .dimensions(300, 200)
            .build()
            .expect("operation should succeed");
        let _ = plot.to_framebuffer().expect("operation should succeed");
    }
}
