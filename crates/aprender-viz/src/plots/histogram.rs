//! Histogram implementation.
//!
//! Supports automatic binning with Sturges, Scott, and Freedman-Diaconis rules.

use crate::color::Rgba;
use crate::error::{Error, Result};
use crate::framebuffer::Framebuffer;

/// Binning strategy for histogram.
#[derive(Debug, Clone, Copy, Default)]
pub enum BinStrategy {
    /// Sturges' rule: ceil(log2(n) + 1)
    #[default]
    Sturges,
    /// Scott's rule: 3.5 * std / n^(1/3)
    Scott,
    /// Freedman-Diaconis rule: 2 * IQR / n^(1/3)
    FreedmanDiaconis,
    /// Fixed number of bins
    Fixed(usize),
}

/// Builder for creating histograms.
#[derive(Debug, Clone)]
pub struct Histogram {
    data: Vec<f32>,
    bin_strategy: BinStrategy,
    color: Rgba,
    width: u32,
    height: u32,
    margin: u32,
    normalize: bool,
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

impl Histogram {
    /// Create a new histogram builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            bin_strategy: BinStrategy::default(),
            color: Rgba::rgb(70, 130, 180), // Steel blue
            width: 800,
            height: 600,
            margin: 40,
            normalize: false,
        }
    }

    /// Set the data.
    #[must_use]
    pub fn data(mut self, data: &[f32]) -> Self {
        self.data = data.to_vec();
        self
    }

    /// Set the binning strategy.
    #[must_use]
    pub fn bins(mut self, strategy: BinStrategy) -> Self {
        self.bin_strategy = strategy;
        self
    }

    /// Set the bar color.
    #[must_use]
    pub fn color(mut self, color: Rgba) -> Self {
        self.color = color;
        self
    }

    /// Enable density normalization.
    #[must_use]
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Calculate the optimal number of bins.
    #[must_use]
    pub fn bin_count(&self) -> usize {
        let n = self.data.len();
        if n == 0 {
            return 1;
        }

        match self.bin_strategy {
            BinStrategy::Sturges => ((n as f32).log2().ceil() + 1.0) as usize,
            BinStrategy::Scott => {
                let std = self.std_dev();
                let width = 3.5 * std / (n as f32).powf(1.0 / 3.0);
                let range = self.data_range();
                (range / width).ceil() as usize
            }
            BinStrategy::FreedmanDiaconis => {
                let iqr = self.iqr();
                let width = 2.0 * iqr / (n as f32).powf(1.0 / 3.0);
                let range = self.data_range();
                if width > 0.0 {
                    (range / width).ceil() as usize
                } else {
                    ((n as f32).log2().ceil() + 1.0) as usize
                }
            }
            BinStrategy::Fixed(bins) => bins.max(1),
        }
        .max(1)
    }

    fn data_range(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let min = self.data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        max - min
    }

    fn std_dev(&self) -> f32 {
        if self.data.len() < 2 {
            return 0.0;
        }
        let mean = self.data.iter().sum::<f32>() / self.data.len() as f32;
        let variance = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
            / (self.data.len() - 1) as f32;
        variance.sqrt()
    }

    fn iqr(&self) -> f32 {
        if self.data.len() < 4 {
            return self.data_range();
        }
        let mut sorted = self.data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q1_idx = sorted.len() / 4;
        let q3_idx = 3 * sorted.len() / 4;
        sorted[q3_idx] - sorted[q1_idx]
    }

    /// Build and validate the histogram.
    ///
    /// # Errors
    ///
    /// Returns an error if data is empty.
    pub fn build(self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(Error::EmptyData);
        }
        Ok(self)
    }

    /// Render to a new framebuffer.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering fails.
    pub fn to_framebuffer(&self) -> Result<Framebuffer> {
        let mut fb = Framebuffer::new(self.width, self.height)?;
        fb.clear(Rgba::WHITE);

        // Calculate bins
        let bin_count = self.bin_count();
        let min = self.data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        let bin_width = if range == 0.0 { 1.0 } else { range / bin_count as f32 };

        // Count values in each bin
        let mut counts = vec![0usize; bin_count];
        for &value in &self.data {
            let bin = ((value - min) / bin_width).floor() as usize;
            let bin = bin.min(bin_count - 1);
            counts[bin] += 1;
        }

        // Find max count for scaling
        let max_count = *counts.iter().max().unwrap_or(&1);

        // Calculate plot area
        let plot_width = self.width - 2 * self.margin;
        let plot_height = self.height - 2 * self.margin;
        let bar_width = plot_width / bin_count as u32;

        // Draw bars
        for (i, &count) in counts.iter().enumerate() {
            let bar_height = if max_count > 0 {
                (count as f32 / max_count as f32 * plot_height as f32) as u32
            } else {
                0
            };

            let x_start = self.margin + i as u32 * bar_width;
            let y_start = self.margin + plot_height - bar_height;

            // Draw filled rectangle
            for y in y_start..(y_start + bar_height) {
                for x in x_start..(x_start + bar_width.saturating_sub(1)) {
                    fb.set_pixel(x, y, self.color);
                }
            }
        }

        Ok(fb)
    }
}

impl batuta_common::display::WithDimensions for Histogram {
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
    fn test_histogram_builder() {
        let hist = Histogram::new()
            .data(&[1.0, 2.0, 3.0, 4.0, 5.0])
            .bins(BinStrategy::Fixed(5))
            .build()
            .expect("operation should succeed");

        assert_eq!(hist.bin_count(), 5);
    }

    #[test]
    fn test_histogram_sturges() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let hist = Histogram::new()
            .data(&data)
            .bins(BinStrategy::Sturges)
            .build()
            .expect("builder should produce valid result");

        // log2(100) + 1 ≈ 8
        assert!(hist.bin_count() >= 7 && hist.bin_count() <= 9);
    }

    #[test]
    fn test_histogram_empty_data() {
        let result = Histogram::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_histogram_render() {
        let hist = Histogram::new()
            .data(&[1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0])
            .dimensions(100, 100)
            .build()
            .expect("operation should succeed");

        let fb = hist.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_histogram_scott() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let hist = Histogram::new()
            .data(&data)
            .bins(BinStrategy::Scott)
            .build()
            .expect("builder should produce valid result");

        assert!(hist.bin_count() >= 1);
    }

    #[test]
    fn test_histogram_freedman_diaconis() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let hist = Histogram::new()
            .data(&data)
            .bins(BinStrategy::FreedmanDiaconis)
            .build()
            .expect("builder should produce valid result");

        assert!(hist.bin_count() >= 1);
    }

    #[test]
    fn test_histogram_freedman_diaconis_zero_iqr() {
        // All same values = zero IQR, should fall back to Sturges
        let data: Vec<f32> = vec![5.0; 100];
        let hist = Histogram::new()
            .data(&data)
            .bins(BinStrategy::FreedmanDiaconis)
            .build()
            .expect("builder should produce valid result");

        assert!(hist.bin_count() >= 1);
    }

    #[test]
    fn test_histogram_fixed_zero() {
        let hist = Histogram::new()
            .data(&[1.0, 2.0, 3.0])
            .bins(BinStrategy::Fixed(0))
            .build()
            .expect("builder should produce valid result");

        // Should be at least 1
        assert_eq!(hist.bin_count(), 1);
    }

    #[test]
    fn test_histogram_color() {
        let hist = Histogram::new()
            .data(&[1.0, 2.0, 3.0])
            .color(Rgba::RED)
            .build()
            .expect("builder should produce valid result");

        assert_eq!(hist.color, Rgba::RED);
    }

    #[test]
    fn test_histogram_normalize() {
        let hist = Histogram::new()
            .data(&[1.0, 2.0, 3.0])
            .normalize(true)
            .build()
            .expect("builder should produce valid result");

        assert!(hist.normalize);
    }

    #[test]
    fn test_histogram_default() {
        let hist = Histogram::default();
        assert!(hist.data.is_empty());
    }

    #[test]
    fn test_histogram_small_data() {
        // Test with 1 element
        let hist1 =
            Histogram::new().data(&[5.0]).build().expect("builder should produce valid result");
        assert!(hist1.bin_count() >= 1);
        let _ = hist1.to_framebuffer().expect("operation should succeed");

        // Test with 2 elements
        let hist2 = Histogram::new()
            .data(&[1.0, 2.0])
            .build()
            .expect("builder should produce valid result");
        assert!(hist2.bin_count() >= 1);
    }

    #[test]
    fn test_histogram_iqr_small() {
        // Test IQR with small data (< 4 elements)
        let hist = Histogram::new()
            .data(&[1.0, 2.0, 3.0])
            .bins(BinStrategy::FreedmanDiaconis)
            .build()
            .expect("operation should succeed");

        assert!(hist.bin_count() >= 1);
    }

    #[test]
    fn test_histogram_std_small() {
        // Test std_dev with 1 element
        let hist = Histogram::new()
            .data(&[5.0])
            .bins(BinStrategy::Scott)
            .build()
            .expect("builder should produce valid result");

        assert!(hist.bin_count() >= 1);
    }

    #[test]
    fn test_bin_strategy_default() {
        assert!(matches!(BinStrategy::default(), BinStrategy::Sturges));
    }

    #[test]
    fn test_histogram_debug_clone() {
        let hist = Histogram::new().data(&[1.0, 2.0, 3.0]);
        let hist2 = hist.clone();
        let _ = format!("{hist2:?}");
    }

    #[test]
    fn test_histogram_bin_count_empty() {
        let hist = Histogram::new();
        // Empty data should return 1
        assert_eq!(hist.bin_count(), 1);
    }
}
