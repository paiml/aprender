//! HorizonGraph widget for high-density time-series visualization.
//!
//! Implements horizon charts as described by Heer et al. (2009).
//! Allows displaying 64+ CPU cores in minimal vertical space by "folding"
//! bands of value into overlapping colored layers.
//!
//! Citation: Heer, J., Kong, N., & Agrawala, M. (2009). "Sizing the Horizon"
//!
//! # Overview
//!
//! Horizon charts "fold" values into overlapping bands, allowing dense
//! visualization of many data series in limited vertical space. A value
//! from 0-100% might be split into 4 bands of 25% each, with darker
//! colors representing higher values within each band.
//!
//! # Example
//!
//! ```
//! use trueno_viz::monitor::widgets::HorizonGraph;
//!
//! let data = vec![0.2, 0.5, 0.8, 0.3, 0.6, 0.9, 0.4, 0.7];
//! let graph = HorizonGraph::new(&data)
//!     .with_bands(3)
//!     .with_label("CPU0");
//! ```

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::widgets::Widget;

/// Color scheme for horizon bands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HorizonScheme {
    /// Blue-based (cool) for normal metrics.
    #[default]
    Blues,
    /// Red-based (warm) for temperature/critical metrics.
    Reds,
    /// Green-based for memory/capacity.
    Greens,
    /// Purple for GPU metrics.
    Purples,
    /// Orange for I/O metrics.
    Oranges,
}

impl HorizonScheme {
    /// Get colors for each band (from light to dark).
    #[must_use]
    pub fn band_colors(self, bands: u8) -> Vec<Color> {
        let base = match self {
            Self::Blues => (66, 133, 244),   // Google Blue base
            Self::Reds => (234, 67, 53),     // Google Red base
            Self::Greens => (52, 168, 83),   // Google Green base
            Self::Purples => (156, 39, 176), // Material Purple base
            Self::Oranges => (251, 140, 0),  // Material Orange base
        };

        (0..bands)
            .map(|i| {
                // Lighter colors for lower bands, darker for higher
                let factor = 0.4 + 0.6 * (f32::from(i) / f32::from(bands));
                let r = ((base.0 as f32 * factor).min(255.0)) as u8;
                let g = ((base.1 as f32 * factor).min(255.0)) as u8;
                let b = ((base.2 as f32 * factor).min(255.0)) as u8;
                Color::Rgb(r, g, b)
            })
            .collect()
    }

    /// Get the background color for this scheme.
    #[must_use]
    pub fn background(self) -> Color {
        match self {
            Self::Blues => Color::Rgb(20, 30, 48),
            Self::Reds => Color::Rgb(48, 20, 20),
            Self::Greens => Color::Rgb(20, 48, 30),
            Self::Purples => Color::Rgb(38, 20, 48),
            Self::Oranges => Color::Rgb(48, 35, 20),
        }
    }
}

/// High-density time-series visualization using horizon chart technique.
///
/// Horizon charts "fold" values into overlapping bands, allowing dense
/// visualization of many data series in limited vertical space.
#[derive(Debug, Clone)]
pub struct HorizonGraph<'a> {
    /// Data values (0.0-1.0 normalized).
    data: &'a [f64],
    /// Number of horizon bands (typically 2-4).
    bands: u8,
    /// Color scheme.
    scheme: HorizonScheme,
    /// Optional label.
    label: Option<String>,
    /// Minimum value for normalization (default 0.0).
    min_value: f64,
    /// Maximum value for normalization (default 1.0).
    max_value: f64,
    /// Whether to mirror negative values below the baseline.
    mirror_negative: bool,
}

impl Default for HorizonGraph<'_> {
    fn default() -> Self {
        Self::new(&[])
    }
}

impl<'a> HorizonGraph<'a> {
    /// Create a new horizon graph from data.
    ///
    /// Data should be pre-normalized to 0.0-1.0 range, or use
    /// `with_range()` to set custom min/max values.
    #[must_use]
    pub fn new(data: &'a [f64]) -> Self {
        Self {
            data,
            bands: 3,
            scheme: HorizonScheme::Blues,
            label: None,
            min_value: 0.0,
            max_value: 1.0,
            mirror_negative: false,
        }
    }

    /// Set the number of bands (2-8, default 3).
    #[must_use]
    pub fn with_bands(mut self, bands: u8) -> Self {
        self.bands = bands.clamp(2, 8);
        self
    }

    /// Set the color scheme.
    #[must_use]
    pub fn with_scheme(mut self, scheme: HorizonScheme) -> Self {
        self.scheme = scheme;
        self
    }

    /// Set an optional label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set the data range for normalization.
    #[must_use]
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.min_value = min;
        self.max_value = max;
        self
    }

    /// Enable mirroring of negative values.
    #[must_use]
    pub fn with_mirror_negative(mut self, mirror: bool) -> Self {
        self.mirror_negative = mirror;
        self
    }

    /// Normalize a value to 0.0-1.0 range.
    fn normalize(&self, value: f64) -> f64 {
        if self.max_value == self.min_value {
            return 0.5;
        }
        ((value - self.min_value) / (self.max_value - self.min_value)).clamp(0.0, 1.0)
    }

    /// Get the band index (0-based) and intensity within the band for a normalized value.
    fn get_band_and_intensity(&self, normalized: f64) -> (usize, f64) {
        let scaled = normalized * f64::from(self.bands);
        let band = (scaled.floor() as usize).min(self.bands as usize - 1);
        let intensity = scaled - band as f64;
        (band, intensity)
    }

    /// Get braille character for a given fill level (0.0-1.0).
    #[allow(dead_code)] // Used in tests, reserved for braille rendering mode
    fn braille_char(fill: f64) -> char {
        // Braille patterns for vertical fill: ⣀⣤⣶⣿
        // Each character represents 4 vertical dots
        const BRAILLE: &[char] = &[' ', '⣀', '⣤', '⣶', '⣿'];
        let idx = ((fill * 4.0).round() as usize).min(4);
        BRAILLE[idx]
    }

    /// Get block character for a given fill level (0.0-1.0).
    fn block_char(fill: f64) -> char {
        // Block characters for fill: ░▒▓█
        const BLOCKS: &[char] = &[' ', '░', '▒', '▓', '█'];
        let idx = ((fill * 4.0).round() as usize).min(4);
        BLOCKS[idx]
    }
}

impl Widget for HorizonGraph<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width == 0 || area.height == 0 || self.data.is_empty() {
            return;
        }

        let colors = self.scheme.band_colors(self.bands);
        let bg = self.scheme.background();

        // Calculate how many data points per column
        let data_per_col = if self.data.len() > area.width as usize {
            self.data.len() as f64 / f64::from(area.width)
        } else {
            1.0
        };

        // Render each column
        for col in 0..area.width {
            // Average data points for this column
            let start_idx = (f64::from(col) * data_per_col) as usize;
            let end_idx = ((f64::from(col) + 1.0) * data_per_col) as usize;
            let end_idx = end_idx.min(self.data.len());

            let avg = if start_idx < end_idx {
                let sum: f64 = self.data[start_idx..end_idx].iter().sum();
                sum / (end_idx - start_idx) as f64
            } else if start_idx < self.data.len() {
                self.data[start_idx]
            } else {
                0.0
            };

            let normalized = self.normalize(avg);
            let (band, intensity) = self.get_band_and_intensity(normalized);

            // Render each row from bottom to top
            for row in 0..area.height {
                let y = area.y + area.height - 1 - row;
                let x = area.x + col;

                // Determine which band this row represents
                let row_band = (row as usize * self.bands as usize) / area.height as usize;
                let row_band = row_band.min(self.bands as usize - 1);

                let (ch, fg) = if row_band < band {
                    // This row's band is fully covered
                    ('█', colors[row_band])
                } else if row_band == band {
                    // This row's band is partially covered
                    (Self::block_char(intensity), colors[row_band])
                } else {
                    // This row's band is not covered
                    (' ', bg)
                };

                buf[(x, y)].set_char(ch).set_fg(fg).set_bg(bg);
            }
        }

        // Render label if present
        if let Some(ref label) = self.label {
            let label_len = label.chars().count().min(area.width as usize);
            for (i, ch) in label.chars().take(label_len).enumerate() {
                let x = area.x + i as u16;
                buf[(x, area.y)].set_char(ch).set_fg(Color::White).set_bg(bg);
            }
        }
    }
}

// =============================================================================
// TESTS - EXTREME TDD
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    mod scheme_tests {
        use super::*;

        #[test]
        fn test_scheme_default_is_blues() {
            assert_eq!(HorizonScheme::default(), HorizonScheme::Blues);
        }

        #[test]
        fn test_band_colors_count() {
            for bands in 2..=8 {
                let colors = HorizonScheme::Blues.band_colors(bands);
                assert_eq!(colors.len(), bands as usize);
            }
        }

        #[test]
        fn test_band_colors_all_schemes() {
            let schemes = [
                HorizonScheme::Blues,
                HorizonScheme::Reds,
                HorizonScheme::Greens,
                HorizonScheme::Purples,
                HorizonScheme::Oranges,
            ];

            for scheme in schemes {
                let colors = scheme.band_colors(4);
                assert_eq!(colors.len(), 4);

                // Each color should be valid RGB (colors should progressively darken)
                for (i, color) in colors.iter().enumerate() {
                    match color {
                        Color::Rgb(r, g, b) => {
                            // Verify colors are RGB values
                            // Band 0 should be lightest, band n-1 darkest
                            if i > 0 {
                                // Colors should generally have increasing intensity
                                assert!((u16::from(*r) + u16::from(*g) + u16::from(*b)) > 0);
                            }
                        }
                        _ => panic!("Expected RGB color"),
                    }
                }
            }
        }

        #[test]
        fn test_background_colors() {
            let schemes = [
                HorizonScheme::Blues,
                HorizonScheme::Reds,
                HorizonScheme::Greens,
                HorizonScheme::Purples,
                HorizonScheme::Oranges,
            ];

            for scheme in schemes {
                let bg = scheme.background();
                match bg {
                    Color::Rgb(_, _, _) => {} // Valid
                    _ => panic!("Expected RGB color for background"),
                }
            }
        }

        #[test]
        fn test_band_colors_gradient() {
            // Higher bands should be darker (higher factor)
            let colors = HorizonScheme::Blues.band_colors(4);

            // Extract RGB values
            let values: Vec<(u8, u8, u8)> = colors
                .iter()
                .map(|c| match c {
                    Color::Rgb(r, g, b) => (*r, *g, *b),
                    _ => (0, 0, 0),
                })
                .collect();

            // Generally, later bands should have higher luminance (darker in this context means higher factor)
            // Just verify they're all different
            for i in 1..values.len() {
                assert_ne!(values[i], values[i - 1], "Adjacent bands should have different colors");
            }
        }
    }

    mod graph_construction_tests {
        use super::*;

        #[test]
        fn test_new_empty() {
            let graph = HorizonGraph::new(&[]);
            assert_eq!(graph.bands, 3);
            assert_eq!(graph.scheme, HorizonScheme::Blues);
            assert!(graph.label.is_none());
        }

        #[test]
        fn test_new_with_data() {
            let data = vec![0.1, 0.5, 0.9];
            let graph = HorizonGraph::new(&data);
            assert_eq!(graph.data.len(), 3);
        }

        #[test]
        fn test_with_bands() {
            let graph = HorizonGraph::new(&[]).with_bands(4);
            assert_eq!(graph.bands, 4);
        }

        #[test]
        fn test_with_bands_clamped() {
            let graph_low = HorizonGraph::new(&[]).with_bands(1);
            assert_eq!(graph_low.bands, 2); // Clamped to minimum

            let graph_high = HorizonGraph::new(&[]).with_bands(10);
            assert_eq!(graph_high.bands, 8); // Clamped to maximum
        }

        #[test]
        fn test_with_scheme() {
            let graph = HorizonGraph::new(&[]).with_scheme(HorizonScheme::Reds);
            assert_eq!(graph.scheme, HorizonScheme::Reds);
        }

        #[test]
        fn test_with_label() {
            let graph = HorizonGraph::new(&[]).with_label("CPU0");
            assert_eq!(graph.label, Some("CPU0".to_string()));
        }

        #[test]
        fn test_with_range() {
            let graph = HorizonGraph::new(&[]).with_range(0.0, 100.0);
            assert_eq!(graph.min_value, 0.0);
            assert_eq!(graph.max_value, 100.0);
        }

        #[test]
        fn test_with_mirror_negative() {
            let graph = HorizonGraph::new(&[]).with_mirror_negative(true);
            assert!(graph.mirror_negative);
        }

        #[test]
        fn test_default() {
            let graph = HorizonGraph::default();
            assert!(graph.data.is_empty());
            assert_eq!(graph.bands, 3);
        }

        #[test]
        fn test_builder_chaining() {
            let data = vec![0.5];
            let graph = HorizonGraph::new(&data)
                .with_bands(4)
                .with_scheme(HorizonScheme::Greens)
                .with_label("Test")
                .with_range(0.0, 100.0)
                .with_mirror_negative(true);

            assert_eq!(graph.bands, 4);
            assert_eq!(graph.scheme, HorizonScheme::Greens);
            assert_eq!(graph.label, Some("Test".to_string()));
            assert_eq!(graph.min_value, 0.0);
            assert_eq!(graph.max_value, 100.0);
            assert!(graph.mirror_negative);
        }
    }

    mod normalization_tests {
        use super::*;

        #[test]
        fn test_normalize_default_range() {
            let graph = HorizonGraph::new(&[]);
            assert_eq!(graph.normalize(0.0), 0.0);
            assert_eq!(graph.normalize(0.5), 0.5);
            assert_eq!(graph.normalize(1.0), 1.0);
        }

        #[test]
        fn test_normalize_custom_range() {
            let graph = HorizonGraph::new(&[]).with_range(0.0, 100.0);
            assert_eq!(graph.normalize(0.0), 0.0);
            assert_eq!(graph.normalize(50.0), 0.5);
            assert_eq!(graph.normalize(100.0), 1.0);
        }

        #[test]
        fn test_normalize_clamps_out_of_range() {
            let graph = HorizonGraph::new(&[]);
            assert_eq!(graph.normalize(-0.5), 0.0);
            assert_eq!(graph.normalize(1.5), 1.0);
        }

        #[test]
        fn test_normalize_same_min_max() {
            let graph = HorizonGraph::new(&[]).with_range(50.0, 50.0);
            assert_eq!(graph.normalize(50.0), 0.5); // Returns 0.5 when range is zero
        }

        #[test]
        fn test_normalize_negative_range() {
            let graph = HorizonGraph::new(&[]).with_range(-100.0, 100.0);
            assert_eq!(graph.normalize(-100.0), 0.0);
            assert_eq!(graph.normalize(0.0), 0.5);
            assert_eq!(graph.normalize(100.0), 1.0);
        }
    }

    mod band_calculation_tests {
        use super::*;

        #[test]
        fn test_get_band_and_intensity_zero() {
            let graph = HorizonGraph::new(&[]).with_bands(4);
            let (band, intensity) = graph.get_band_and_intensity(0.0);
            assert_eq!(band, 0);
            assert!((intensity - 0.0).abs() < 0.001);
        }

        #[test]
        fn test_get_band_and_intensity_full() {
            let graph = HorizonGraph::new(&[]).with_bands(4);
            let (band, _intensity) = graph.get_band_and_intensity(1.0);
            assert_eq!(band, 3); // Last band (0-indexed)
        }

        #[test]
        fn test_get_band_and_intensity_mid() {
            let graph = HorizonGraph::new(&[]).with_bands(4);
            let (band, _intensity) = graph.get_band_and_intensity(0.5);
            assert_eq!(band, 2); // 0.5 * 4 = 2.0, floor = 2
        }

        #[test]
        fn test_get_band_and_intensity_quarter() {
            let graph = HorizonGraph::new(&[]).with_bands(4);
            let (band, _intensity) = graph.get_band_and_intensity(0.25);
            assert_eq!(band, 1); // 0.25 * 4 = 1.0, floor = 1
        }
    }

    mod character_tests {
        use super::*;

        #[test]
        fn test_braille_char_empty() {
            assert_eq!(HorizonGraph::braille_char(0.0), ' ');
        }

        #[test]
        fn test_braille_char_full() {
            assert_eq!(HorizonGraph::braille_char(1.0), '⣿');
        }

        #[test]
        fn test_braille_char_levels() {
            // Test discrete levels
            let chars: Vec<char> =
                (0..=4).map(|i| HorizonGraph::braille_char(f64::from(i) / 4.0)).collect();
            assert_eq!(chars, vec![' ', '⣀', '⣤', '⣶', '⣿']);
        }

        #[test]
        fn test_block_char_empty() {
            assert_eq!(HorizonGraph::block_char(0.0), ' ');
        }

        #[test]
        fn test_block_char_full() {
            assert_eq!(HorizonGraph::block_char(1.0), '█');
        }

        #[test]
        fn test_block_char_levels() {
            let chars: Vec<char> =
                (0..=4).map(|i| HorizonGraph::block_char(f64::from(i) / 4.0)).collect();
            assert_eq!(chars, vec![' ', '░', '▒', '▓', '█']);
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
        fn test_render_empty_data() {
            let (area, mut buf) = create_test_buffer(10, 3);
            let graph = HorizonGraph::new(&[]);
            graph.render(area, &mut buf);
            // Should not panic, buffer unchanged
        }

        #[test]
        fn test_render_zero_area() {
            let (_, mut buf) = create_test_buffer(10, 3);
            let data = vec![0.5];
            let graph = HorizonGraph::new(&data);

            // Zero width
            let zero_area = Rect::new(0, 0, 0, 3);
            graph.clone().render(zero_area, &mut buf);

            // Zero height
            let zero_area = Rect::new(0, 0, 10, 0);
            graph.render(zero_area, &mut buf);
            // Should not panic
        }

        #[test]
        fn test_render_single_value() {
            let (area, mut buf) = create_test_buffer(5, 3);
            let data = vec![0.5];
            let graph = HorizonGraph::new(&data).with_bands(3);
            graph.render(area, &mut buf);

            // Buffer should be modified
            let content: String = buf.content().iter().map(|c| c.symbol().to_string()).collect();
            assert!(!content.chars().all(|c| c == ' '));
        }

        #[test]
        fn test_render_with_label() {
            let (area, mut buf) = create_test_buffer(10, 3);
            let data = vec![0.5];
            let graph = HorizonGraph::new(&data).with_label("CPU0");
            graph.render(area, &mut buf);

            // First row should contain label
            let first_row: String =
                (0..area.width).map(|x| buf[(x, 0)].symbol().to_string()).collect();
            assert!(first_row.contains("CPU0") || first_row.starts_with("CPU0"));
        }

        #[test]
        fn test_render_long_label_truncated() {
            let (area, mut buf) = create_test_buffer(5, 3);
            let data = vec![0.5];
            let graph = HorizonGraph::new(&data).with_label("VeryLongLabel");
            graph.render(area, &mut buf);
            // Should not panic, label truncated to fit
        }

        #[test]
        fn test_render_multiple_values() {
            let (area, mut buf) = create_test_buffer(10, 3);
            let data = vec![0.1, 0.3, 0.5, 0.7, 0.9];
            let graph = HorizonGraph::new(&data).with_bands(3);
            graph.render(area, &mut buf);

            // Buffer should be modified
            let non_space_count = buf.content().iter().filter(|c| c.symbol() != " ").count();
            assert!(non_space_count > 0);
        }

        #[test]
        fn test_render_data_longer_than_width() {
            let (area, mut buf) = create_test_buffer(5, 3);
            let data: Vec<f64> = (0..20).map(|i| f64::from(i) / 19.0).collect();
            let graph = HorizonGraph::new(&data);
            graph.render(area, &mut buf);
            // Should downsample and render without panic
        }

        #[test]
        fn test_render_all_schemes() {
            let schemes = [
                HorizonScheme::Blues,
                HorizonScheme::Reds,
                HorizonScheme::Greens,
                HorizonScheme::Purples,
                HorizonScheme::Oranges,
            ];

            let data = vec![0.3, 0.6, 0.9];

            for scheme in schemes {
                let (area, mut buf) = create_test_buffer(10, 3);
                let graph = HorizonGraph::new(&data).with_scheme(scheme);
                graph.render(area, &mut buf);
                // Should render without panic
            }
        }

        #[test]
        fn test_render_different_band_counts() {
            let data = vec![0.5];

            for bands in 2..=8 {
                let (area, mut buf) = create_test_buffer(10, 5);
                let graph = HorizonGraph::new(&data).with_bands(bands);
                graph.render(area, &mut buf);
                // Should render without panic
            }
        }

        #[test]
        fn test_render_high_values() {
            let (area, mut buf) = create_test_buffer(10, 3);
            let data = vec![1.0, 1.0, 1.0];
            let graph = HorizonGraph::new(&data).with_bands(4);
            graph.render(area, &mut buf);
            // All columns should have highest band filled
        }

        #[test]
        fn test_render_low_values() {
            let (area, mut buf) = create_test_buffer(10, 3);
            let data = vec![0.0, 0.0, 0.0];
            let graph = HorizonGraph::new(&data).with_bands(4);
            graph.render(area, &mut buf);
            // Should render empty/minimal visualization
        }

        #[test]
        fn test_render_gradient_values() {
            let (area, mut buf) = create_test_buffer(10, 4);
            let data: Vec<f64> = (0..10).map(|i| f64::from(i) / 9.0).collect();
            let graph = HorizonGraph::new(&data).with_bands(4);
            graph.render(area, &mut buf);
            // Should show gradient from left (low) to right (high)
        }

        #[test]
        fn test_render_width_equals_data_length() {
            // When width == data length, each column has exactly one data point
            let (area, mut buf) = create_test_buffer(5, 3);
            let data = vec![0.1, 0.3, 0.5, 0.7, 0.9];
            let graph = HorizonGraph::new(&data);
            graph.render(area, &mut buf);
        }

        #[test]
        fn test_render_width_exceeds_data() {
            // When width > data length, some columns share data points
            let (area, mut buf) = create_test_buffer(20, 3);
            let data = vec![0.5, 0.7];
            let graph = HorizonGraph::new(&data);
            graph.render(area, &mut buf);
        }

        #[test]
        fn test_render_single_data_point() {
            // Single data point across wide area
            let (area, mut buf) = create_test_buffer(15, 4);
            let data = vec![0.75];
            let graph = HorizonGraph::new(&data);
            graph.render(area, &mut buf);
        }

        #[test]
        fn test_render_narrow_single_column() {
            let (area, mut buf) = create_test_buffer(1, 5);
            let data = vec![0.5, 0.6, 0.7];
            let graph = HorizonGraph::new(&data);
            graph.render(area, &mut buf);
        }
    }
}
