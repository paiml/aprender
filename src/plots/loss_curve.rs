//! Streaming loss curve visualization for ML training.
//!
//! Provides real-time visualization of training metrics like loss, accuracy,
//! and learning rate across epochs. Supports streaming updates and smoothing.
//!
//! # Features
//!
//! - Multiple metrics (train loss, validation loss, etc.)
//! - Exponential moving average smoothing
//! - Best value markers
//! - Streaming data updates

use crate::color::Rgba;
use crate::error::{Error, Result};
use crate::framebuffer::Framebuffer;
use crate::render::{draw_circle, draw_line_aa};
use crate::scale::{LinearScale, Scale};

/// A single metric series for loss curves.
#[derive(Debug, Clone)]
pub struct MetricSeries {
    /// Series name/label.
    pub name: String,
    /// Raw metric values per epoch.
    values: Vec<f32>,
    /// Smoothed values (if smoothing enabled).
    smoothed: Vec<f32>,
    /// Line color.
    pub color: Rgba,
    /// Whether to show raw values.
    pub show_raw: bool,
    /// Whether to show smoothed values.
    pub show_smoothed: bool,
    /// Smoothing factor (0.0 = no smoothing, 0.99 = heavy smoothing).
    smoothing_factor: f32,
}

impl MetricSeries {
    /// Create a new metric series.
    #[must_use]
    pub fn new(name: impl Into<String>, color: Rgba) -> Self {
        Self {
            name: name.into(),
            values: Vec::new(),
            smoothed: Vec::new(),
            color,
            show_raw: true,
            show_smoothed: true,
            smoothing_factor: 0.6,
        }
    }

    /// Set the smoothing factor (0.0 to 0.99).
    ///
    /// Higher values produce smoother curves.
    #[must_use]
    pub fn smoothing(mut self, factor: f32) -> Self {
        self.smoothing_factor = factor.clamp(0.0, 0.99);
        self
    }

    /// Set whether to show raw values.
    #[must_use]
    pub fn raw(mut self, show: bool) -> Self {
        self.show_raw = show;
        self
    }

    /// Set whether to show smoothed values.
    #[must_use]
    pub fn smooth(mut self, show: bool) -> Self {
        self.show_smoothed = show;
        self
    }

    /// Add a new value to the series.
    pub fn push(&mut self, value: f32) {
        self.values.push(value);

        // Calculate smoothed value using exponential moving average
        let smoothed_value = if self.smoothed.is_empty() {
            value
        } else {
            let prev = self.smoothed[self.smoothed.len() - 1];
            self.smoothing_factor * prev + (1.0 - self.smoothing_factor) * value
        };
        self.smoothed.push(smoothed_value);
    }

    /// Get the raw values.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Get the smoothed values.
    #[must_use]
    pub fn smoothed_values(&self) -> &[f32] {
        &self.smoothed
    }

    /// Get the number of values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the series is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get the minimum value.
    #[must_use]
    pub fn min(&self) -> Option<f32> {
        self.values.iter().copied().reduce(f32::min)
    }

    /// Get the maximum value.
    #[must_use]
    pub fn max(&self) -> Option<f32> {
        self.values.iter().copied().reduce(f32::max)
    }

    /// Get the index of the minimum value.
    #[must_use]
    pub fn argmin(&self) -> Option<usize> {
        self.values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    }

    /// Get the index of the maximum value.
    #[must_use]
    pub fn argmax(&self) -> Option<usize> {
        self.values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    }

    /// Get the last value.
    #[must_use]
    pub fn last(&self) -> Option<f32> {
        self.values.last().copied()
    }

    /// Get the last smoothed value.
    #[must_use]
    pub fn last_smoothed(&self) -> Option<f32> {
        self.smoothed.last().copied()
    }

    /// Clear all values.
    pub fn clear(&mut self) {
        self.values.clear();
        self.smoothed.clear();
    }
}

/// Builder for streaming loss curve visualization.
#[derive(Debug, Clone)]
pub struct LossCurve {
    /// Metric series to display.
    series: Vec<MetricSeries>,
    /// Output width.
    width: u32,
    /// Output height.
    height: u32,
    /// Margin around the plot.
    margin: u32,
    /// Show best value markers.
    show_best_markers: bool,
    /// Marker size.
    marker_size: f32,
    /// Whether lower is better (for loss) or higher is better (for accuracy).
    lower_is_better: bool,
    /// Y-axis minimum (None for auto).
    y_min: Option<f32>,
    /// Y-axis maximum (None for auto).
    y_max: Option<f32>,
}

impl Default for LossCurve {
    fn default() -> Self {
        Self::new()
    }
}

impl LossCurve {
    /// Create a new loss curve builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            series: Vec::new(),
            width: 800,
            height: 400,
            margin: 40,
            show_best_markers: true,
            marker_size: 6.0,
            lower_is_better: true,
            y_min: None,
            y_max: None,
        }
    }

    /// Add a metric series.
    #[must_use]
    pub fn add_series(mut self, series: MetricSeries) -> Self {
        self.series.push(series);
        self
    }

    /// Add training loss series (convenience method).
    #[must_use]
    pub fn train_loss(self) -> Self {
        self.add_series(MetricSeries::new("Train Loss", Rgba::BLUE))
    }

    /// Add validation loss series (convenience method).
    #[must_use]
    pub fn val_loss(self) -> Self {
        self.add_series(MetricSeries::new("Val Loss", Rgba::rgb(255, 128, 0)))
    }

    /// Set the margin.
    #[must_use]
    pub fn margin(mut self, margin: u32) -> Self {
        self.margin = margin;
        self
    }

    /// Enable or disable best value markers.
    #[must_use]
    pub fn best_markers(mut self, show: bool) -> Self {
        self.show_best_markers = show;
        self
    }

    /// Set whether lower values are better (loss) or higher (accuracy).
    #[must_use]
    pub fn lower_is_better(mut self, lower: bool) -> Self {
        self.lower_is_better = lower;
        self
    }

    /// Set fixed Y-axis range.
    #[must_use]
    pub fn y_range(mut self, min: f32, max: f32) -> Self {
        self.y_min = Some(min);
        self.y_max = Some(max);
        self
    }

    /// Get a mutable reference to a series by index.
    pub fn series_mut(&mut self, index: usize) -> Option<&mut MetricSeries> {
        self.series.get_mut(index)
    }

    /// Get a mutable reference to a series by name.
    pub fn series_by_name_mut(&mut self, name: &str) -> Option<&mut MetricSeries> {
        self.series.iter_mut().find(|s| s.name == name)
    }

    /// Push a value to a series by index.
    pub fn push(&mut self, series_index: usize, value: f32) {
        if let Some(series) = self.series.get_mut(series_index) {
            series.push(value);
        }
    }

    /// Push values to all series at once (one value per series).
    pub fn push_all(&mut self, values: &[f32]) {
        for (series, &value) in self.series.iter_mut().zip(values.iter()) {
            series.push(value);
        }
    }

    /// Get the total number of epochs across all series.
    #[must_use]
    pub fn max_epochs(&self) -> usize {
        self.series.iter().map(MetricSeries::len).max().unwrap_or(0)
    }

    /// Get the number of series.
    #[must_use]
    pub fn series_count(&self) -> usize {
        self.series.len()
    }

    /// Calculate Y-axis extent.
    fn y_extent(&self) -> (f32, f32) {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;

        for series in &self.series {
            if let Some(s_min) = series.min() {
                min = min.min(s_min);
            }
            if let Some(s_max) = series.max() {
                max = max.max(s_max);
            }
        }

        // Use fixed range if specified
        let min = self.y_min.unwrap_or(min);
        let max = self.y_max.unwrap_or(max);

        // Add some padding
        let padding = (max - min) * 0.05;
        (min - padding, max + padding)
    }

    /// Build and validate.
    pub fn build(self) -> Result<Self> {
        if self.series.is_empty() {
            return Err(Error::EmptyData);
        }
        Ok(self)
    }

    /// Render the loss curves to a framebuffer.
    pub fn render(&self, fb: &mut Framebuffer) -> Result<()> {
        let max_epochs = self.max_epochs();
        if max_epochs == 0 {
            return Ok(()); // Nothing to render yet
        }

        let plot_width = self.width.saturating_sub(2 * self.margin);
        let plot_height = self.height.saturating_sub(2 * self.margin);

        if plot_width == 0 || plot_height == 0 {
            return Ok(());
        }

        let (y_min, y_max) = self.y_extent();

        // Create scales
        let x_scale = LinearScale::new(
            (0.0, (max_epochs - 1).max(1) as f32),
            (self.margin as f32, (self.margin + plot_width) as f32),
        )?;
        let y_scale = LinearScale::new(
            (y_min, y_max),
            ((self.margin + plot_height) as f32, self.margin as f32),
        )?;

        // Draw each series
        for series in &self.series {
            Self::render_series(fb, series, &x_scale, &y_scale);
        }

        // Draw best markers
        if self.show_best_markers {
            self.render_best_markers(fb, &x_scale, &y_scale);
        }

        Ok(())
    }

    /// Render a single series.
    fn render_series(
        fb: &mut Framebuffer,
        series: &MetricSeries,
        x_scale: &LinearScale,
        y_scale: &LinearScale,
    ) {
        let values = series.values();
        let smoothed = series.smoothed_values();

        if values.len() < 2 {
            return;
        }

        // Draw raw values (faded)
        if series.show_raw && values.len() >= 2 {
            let raw_color = series.color.with_alpha(100);
            for i in 1..values.len() {
                let x0 = x_scale.scale((i - 1) as f32);
                let y0 = y_scale.scale(values[i - 1]);
                let x1 = x_scale.scale(i as f32);
                let y1 = y_scale.scale(values[i]);
                draw_line_aa(fb, x0, y0, x1, y1, raw_color);
            }
        }

        // Draw smoothed values (solid)
        if series.show_smoothed && smoothed.len() >= 2 {
            for i in 1..smoothed.len() {
                let x0 = x_scale.scale((i - 1) as f32);
                let y0 = y_scale.scale(smoothed[i - 1]);
                let x1 = x_scale.scale(i as f32);
                let y1 = y_scale.scale(smoothed[i]);
                draw_line_aa(fb, x0, y0, x1, y1, series.color);
            }
        }
    }

    /// Render best value markers.
    fn render_best_markers(
        &self,
        fb: &mut Framebuffer,
        x_scale: &LinearScale,
        y_scale: &LinearScale,
    ) {
        let marker_radius = (self.marker_size / 2.0) as i32;

        for series in &self.series {
            let best_idx = if self.lower_is_better { series.argmin() } else { series.argmax() };

            if let Some(idx) = best_idx {
                if let Some(&value) = series.values().get(idx) {
                    let x = x_scale.scale(idx as f32) as i32;
                    let y = y_scale.scale(value) as i32;

                    // Draw a filled circle marker
                    draw_circle(fb, x, y, marker_radius, series.color);

                    // Draw a white border
                    let border_color = Rgba::WHITE;
                    for dy in -marker_radius - 1..=marker_radius + 1 {
                        for dx in -marker_radius - 1..=marker_radius + 1 {
                            let dist_sq = dx * dx + dy * dy;
                            let outer_r = marker_radius + 1;
                            if dist_sq > marker_radius * marker_radius
                                && dist_sq <= outer_r * outer_r
                            {
                                let px = x + dx;
                                let py = y + dy;
                                if px >= 0 && py >= 0 {
                                    fb.set_pixel(px as u32, py as u32, border_color);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Render to a new framebuffer.
    pub fn to_framebuffer(&self) -> Result<Framebuffer> {
        let mut fb = Framebuffer::new(self.width, self.height)?;
        fb.clear(Rgba::WHITE);
        self.render(&mut fb)?;
        Ok(fb)
    }

    /// Get a summary of all series.
    #[must_use]
    pub fn summary(&self) -> Vec<SeriesSummary> {
        self.series
            .iter()
            .map(|s| SeriesSummary {
                name: s.name.clone(),
                epochs: s.len(),
                min: s.min(),
                max: s.max(),
                last: s.last(),
                last_smoothed: s.last_smoothed(),
                best_epoch: if self.lower_is_better { s.argmin() } else { s.argmax() },
            })
            .collect()
    }
}

impl batuta_common::display::WithDimensions for LossCurve {
    fn set_dimensions(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
}

/// Summary statistics for a metric series.
#[derive(Debug, Clone)]
pub struct SeriesSummary {
    /// Series name.
    pub name: String,
    /// Number of epochs.
    pub epochs: usize,
    /// Minimum value.
    pub min: Option<f32>,
    /// Maximum value.
    pub max: Option<f32>,
    /// Last value.
    pub last: Option<f32>,
    /// Last smoothed value.
    pub last_smoothed: Option<f32>,
    /// Best epoch index.
    pub best_epoch: Option<usize>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use batuta_common::display::WithDimensions;

    #[test]
    fn test_metric_series_basic() {
        let mut series = MetricSeries::new("test", Rgba::BLUE);

        series.push(1.0);
        series.push(0.8);
        series.push(0.6);

        assert_eq!(series.len(), 3);
        assert!(!series.is_empty());
        assert_eq!(series.values(), &[1.0, 0.8, 0.6]);
    }

    #[test]
    fn test_metric_series_smoothing() {
        let mut series = MetricSeries::new("test", Rgba::BLUE).smoothing(0.5);

        series.push(1.0);
        series.push(0.0);
        series.push(1.0);
        series.push(0.0);

        let smoothed = series.smoothed_values();
        // Smoothed values should be less extreme than raw values
        assert!(smoothed[1] > 0.0); // Not exactly 0
        assert!(smoothed[2] < 1.0); // Not exactly 1
    }

    #[test]
    fn test_metric_series_min_max() {
        let mut series = MetricSeries::new("test", Rgba::BLUE);

        series.push(0.5);
        series.push(0.2);
        series.push(0.8);
        series.push(0.3);

        assert_eq!(series.min(), Some(0.2));
        assert_eq!(series.max(), Some(0.8));
        assert_eq!(series.argmin(), Some(1));
        assert_eq!(series.argmax(), Some(2));
    }

    #[test]
    fn test_loss_curve_builder() {
        let loss_curve = LossCurve::new()
            .train_loss()
            .val_loss()
            .dimensions(400, 200)
            .build()
            .expect("builder should produce valid result");

        assert_eq!(loss_curve.series_count(), 2);
    }

    #[test]
    fn test_loss_curve_push() {
        let mut loss_curve = LossCurve::new()
            .train_loss()
            .val_loss()
            .build()
            .expect("builder should produce valid result");

        // Push values
        loss_curve.push(0, 1.0);
        loss_curve.push(1, 1.2);
        loss_curve.push_all(&[0.8, 1.0]);
        loss_curve.push_all(&[0.6, 0.8]);

        assert_eq!(loss_curve.max_epochs(), 3);
    }

    #[test]
    fn test_loss_curve_empty() {
        let result = LossCurve::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_loss_curve_render() {
        let mut loss_curve = LossCurve::new()
            .train_loss()
            .val_loss()
            .dimensions(200, 100)
            .build()
            .expect("builder should produce valid result");

        // Add some data
        for i in 0..10 {
            let t = i as f32 / 10.0;
            loss_curve.push_all(&[1.0 - t * 0.5, 1.2 - t * 0.4]);
        }

        let fb = loss_curve.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_loss_curve_render_empty_series() {
        let loss_curve = LossCurve::new()
            .train_loss()
            .dimensions(200, 100)
            .build()
            .expect("builder should produce valid result");

        // Render with empty series should not panic
        let fb = loss_curve.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_loss_curve_summary() {
        let mut loss_curve = LossCurve::new()
            .train_loss()
            .lower_is_better(true)
            .build()
            .expect("builder should produce valid result");

        loss_curve.push(0, 1.0);
        loss_curve.push(0, 0.5);
        loss_curve.push(0, 0.3);
        loss_curve.push(0, 0.4);

        let summary = loss_curve.summary();
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0].epochs, 4);
        assert_eq!(summary[0].min, Some(0.3));
        assert_eq!(summary[0].best_epoch, Some(2)); // Index of minimum
    }

    #[test]
    fn test_loss_curve_higher_is_better() {
        let mut loss_curve = LossCurve::new()
            .add_series(MetricSeries::new("Accuracy", Rgba::GREEN))
            .lower_is_better(false)
            .build()
            .expect("operation should succeed");

        loss_curve.push(0, 0.5);
        loss_curve.push(0, 0.7);
        loss_curve.push(0, 0.9);
        loss_curve.push(0, 0.85);

        let summary = loss_curve.summary();
        assert_eq!(summary[0].best_epoch, Some(2)); // Index of maximum
    }

    #[test]
    fn test_loss_curve_fixed_y_range() {
        let mut loss_curve = LossCurve::new()
            .train_loss()
            .y_range(0.0, 2.0)
            .dimensions(200, 100)
            .build()
            .expect("builder should produce valid result");

        loss_curve.push(0, 1.0);
        loss_curve.push(0, 0.5);

        let fb = loss_curve.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_loss_curve_series_by_name() {
        let mut loss_curve = LossCurve::new()
            .train_loss()
            .val_loss()
            .build()
            .expect("builder should produce valid result");

        if let Some(train) = loss_curve.series_by_name_mut("Train Loss") {
            train.push(1.0);
        }

        assert_eq!(loss_curve.series_mut(0).expect("value should be present").len(), 1);
    }

    #[test]
    fn test_metric_series_clear() {
        let mut series = MetricSeries::new("test", Rgba::BLUE);

        series.push(1.0);
        series.push(0.5);
        assert_eq!(series.len(), 2);

        series.clear();
        assert!(series.is_empty());
    }

    #[test]
    fn test_loss_curve_best_markers() {
        let mut loss_curve = LossCurve::new()
            .train_loss()
            .best_markers(true)
            .dimensions(200, 100)
            .build()
            .expect("builder should produce valid result");

        for i in 0..5 {
            loss_curve.push(0, 1.0 - (i as f32) * 0.1);
        }

        let fb = loss_curve.to_framebuffer();
        assert!(fb.is_ok());
    }
}
