//! Sparkline widget for compact trend visualization.
//!
//! Renders mini line charts suitable for embedding in tables and dashboards.
//! Commonly used to show loss/accuracy trends over training epochs.

use crate::color::Rgba;
use crate::error::Result;
use crate::framebuffer::Framebuffer;
use crate::render::draw_line_aa;
use crate::scale::{LinearScale, Scale};

/// Direction of the trend indicated by the sparkline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    /// Values are increasing (arrow up).
    Rising,
    /// Values are decreasing (arrow down).
    Falling,
    /// Values are relatively stable (horizontal line).
    Stable,
}

impl TrendDirection {
    /// Get a display character for the trend direction.
    #[must_use]
    pub fn indicator(&self) -> &'static str {
        match self {
            Self::Rising => "\u{2191}",  // ↑
            Self::Falling => "\u{2193}", // ↓
            Self::Stable => "\u{2192}",  // →
        }
    }
}

/// A compact sparkline chart for displaying trends.
///
/// Sparklines are minimalist line charts designed to be embedded
/// inline with text or in table cells.
#[derive(Debug, Clone)]
pub struct Sparkline {
    /// Data points to display.
    data: Vec<f64>,
    /// Width in pixels.
    width: u32,
    /// Height in pixels.
    height: u32,
    /// Line color.
    color: Rgba,
    /// Whether to show a trend indicator arrow.
    show_trend: bool,
    /// Threshold for considering a trend "stable" (as percentage of range).
    stability_threshold: f64,
}

impl Default for Sparkline {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            width: 100,
            height: 20,
            color: Rgba::rgb(66, 133, 244), // Material Blue
            show_trend: false,
            stability_threshold: 0.05,
        }
    }
}

impl Sparkline {
    /// Create a new sparkline with the given data.
    #[must_use]
    pub fn new(data: &[f64]) -> Self {
        Self { data: data.to_vec(), ..Self::default() }
    }

    /// Set the sparkline dimensions.
    #[must_use]
    pub fn dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width.max(10);
        self.height = height.max(5);
        self
    }

    /// Set the line color.
    #[must_use]
    pub fn color(mut self, color: Rgba) -> Self {
        self.color = color;
        self
    }

    /// Enable the trend indicator arrow.
    #[must_use]
    pub fn with_trend_indicator(mut self) -> Self {
        self.show_trend = true;
        self
    }

    /// Set the stability threshold (as a fraction of data range).
    ///
    /// If the change from first to last value is less than this fraction
    /// of the total range, the trend is considered "stable".
    #[must_use]
    pub fn stability_threshold(mut self, threshold: f64) -> Self {
        self.stability_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Get the trend direction based on the data.
    #[must_use]
    pub fn trend(&self) -> TrendDirection {
        if self.data.len() < 2 {
            return TrendDirection::Stable;
        }

        let first = self.data[0];
        let last = self.data[self.data.len() - 1];
        let change = last - first;

        // Calculate the data range for threshold comparison
        let (min, max) = self.data_extent();
        let range = max - min;

        // Handle case where all values are the same
        if range < f64::EPSILON {
            return TrendDirection::Stable;
        }

        let change_ratio = change.abs() / range;

        if change_ratio < self.stability_threshold {
            TrendDirection::Stable
        } else if change > 0.0 {
            TrendDirection::Rising
        } else {
            TrendDirection::Falling
        }
    }

    /// Check if trend indicator is enabled.
    #[must_use]
    pub fn has_trend_indicator(&self) -> bool {
        self.show_trend
    }

    /// Get the data extent (min, max).
    fn data_extent(&self) -> (f64, f64) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for &value in &self.data {
            min = min.min(value);
            max = max.max(value);
        }

        // Handle edge cases
        if min.is_infinite() || max.is_infinite() {
            return (0.0, 1.0);
        }

        // Add small padding if min == max
        if (max - min).abs() < f64::EPSILON {
            return (min - 0.5, max + 0.5);
        }

        (min, max)
    }

    /// Render the sparkline to a framebuffer.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering fails.
    pub fn render(&self, fb: &mut Framebuffer) -> Result<()> {
        if self.data.len() < 2 {
            return Ok(());
        }

        let (min, max) = self.data_extent();

        // Create scales for x and y
        let x_scale =
            LinearScale::new((0.0, (self.data.len() - 1) as f32), (1.0, (self.width - 2) as f32))?;
        let y_scale = LinearScale::new((min as f32, max as f32), ((self.height - 2) as f32, 1.0))?;

        // Draw the line segments
        for i in 0..self.data.len() - 1 {
            let x1 = x_scale.scale(i as f32);
            let y1 = y_scale.scale(self.data[i] as f32);
            let x2 = x_scale.scale((i + 1) as f32);
            let y2 = y_scale.scale(self.data[i + 1] as f32);

            draw_line_aa(fb, x1, y1, x2, y2, self.color);
        }

        Ok(())
    }

    /// Render to a new framebuffer.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering fails.
    pub fn to_framebuffer(&self) -> Result<Framebuffer> {
        let mut fb = Framebuffer::new(self.width, self.height)?;
        fb.clear(Rgba::TRANSPARENT);
        self.render(&mut fb)?;
        Ok(fb)
    }

    /// Get the data points.
    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Get the width.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the height.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparkline_render() {
        let sparkline =
            Sparkline::new(&[0.9, 0.7, 0.5, 0.3, 0.2]).dimensions(100, 20).color(Rgba::BLUE);

        let fb = sparkline.to_framebuffer();
        assert!(fb.is_ok());

        let fb = fb.expect("operation should succeed");
        assert_eq!(fb.width(), 100);
        assert_eq!(fb.height(), 20);
    }

    #[test]
    fn test_sparkline_trend_rising() {
        let sparkline = Sparkline::new(&[0.1, 0.3, 0.5, 0.7, 0.9]);
        assert_eq!(sparkline.trend(), TrendDirection::Rising);
    }

    #[test]
    fn test_sparkline_trend_falling() {
        let sparkline = Sparkline::new(&[0.9, 0.7, 0.5, 0.3, 0.1]);
        assert_eq!(sparkline.trend(), TrendDirection::Falling);
    }

    #[test]
    fn test_sparkline_trend_stable() {
        // All same values
        let sparkline = Sparkline::new(&[0.5, 0.5, 0.5, 0.5, 0.5]);
        assert_eq!(sparkline.trend(), TrendDirection::Stable);

        // Very small change relative to range
        let sparkline = Sparkline::new(&[0.5, 0.6, 0.4, 0.55, 0.51]).stability_threshold(0.1);
        assert_eq!(sparkline.trend(), TrendDirection::Stable);
    }

    #[test]
    fn test_sparkline_trend_indicator() {
        assert_eq!(TrendDirection::Rising.indicator(), "\u{2191}");
        assert_eq!(TrendDirection::Falling.indicator(), "\u{2193}");
        assert_eq!(TrendDirection::Stable.indicator(), "\u{2192}");
    }

    #[test]
    fn test_sparkline_empty_data() {
        let sparkline = Sparkline::new(&[]);
        assert_eq!(sparkline.trend(), TrendDirection::Stable);

        // Rendering should succeed (but draw nothing)
        let fb = sparkline.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_sparkline_single_point() {
        let sparkline = Sparkline::new(&[0.5]);
        assert_eq!(sparkline.trend(), TrendDirection::Stable);
    }

    #[test]
    fn test_sparkline_with_trend() {
        let sparkline = Sparkline::new(&[0.9, 0.7, 0.5, 0.3, 0.1]).with_trend_indicator();

        assert!(sparkline.has_trend_indicator());
        assert_eq!(sparkline.trend(), TrendDirection::Falling);
    }

    #[test]
    fn test_sparkline_default() {
        let sparkline = Sparkline::default();
        assert_eq!(sparkline.trend(), TrendDirection::Stable);
    }

    #[test]
    fn test_sparkline_clone_debug() {
        let sparkline = Sparkline::new(&[1.0, 2.0, 3.0]);
        let cloned = sparkline.clone();
        let debug = format!("{cloned:?}");
        assert!(debug.contains("Sparkline"));
    }

    #[test]
    fn test_trend_direction_debug() {
        let dirs = [TrendDirection::Rising, TrendDirection::Falling, TrendDirection::Stable];
        for dir in dirs {
            let debug = format!("{dir:?}");
            assert!(!debug.is_empty());
            let cloned = dir;
            assert_eq!(dir, cloned);
        }
    }

    #[test]
    fn test_stability_threshold_clamp() {
        // Test clamping above 1.0
        let sparkline = Sparkline::new(&[0.0, 1.0]).stability_threshold(2.0);
        let fb = sparkline.to_framebuffer();
        assert!(fb.is_ok());

        // Test clamping below 0.0
        let sparkline = Sparkline::new(&[0.0, 1.0]).stability_threshold(-0.5);
        let fb = sparkline.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_dimensions_minimum() {
        // Test that dimensions are clamped to minimums
        let sparkline = Sparkline::new(&[0.5, 0.6]).dimensions(1, 1);
        let fb = sparkline.to_framebuffer();
        assert!(fb.is_ok());
    }
}
