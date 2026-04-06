//! Heatmap visualization for 2D data matrices.
//!
//! Renders a 2D grid of values as colored cells using a color scale.
//!
//! # References
//!
//! - Wilkinson, L. (2005). *The Grammar of Graphics*. Springer.
//! - Borland, D., & Taylor, R. M. (2007). "Rainbow Color Map (Still) Considered Harmful."
//!   IEEE Computer Graphics and Applications.

use crate::color::Rgba;
use crate::error::{Error, Result};
use crate::framebuffer::Framebuffer;
use crate::scale::{ColorScale, Scale};

/// Color palette type for heatmaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HeatmapPalette {
    /// Viridis (perceptually uniform, colorblind-safe).
    #[default]
    Viridis,
    /// Sequential blues.
    Blues,
    /// Diverging red-blue.
    RedBlue,
    /// Magma (perceptually uniform).
    Magma,
    /// Heat (black-red-yellow-white).
    Heat,
    /// Greyscale.
    Greyscale,
}

/// Builder for creating heatmaps.
#[derive(Debug, Clone)]
pub struct Heatmap {
    /// 2D data matrix in row-major order.
    data: Vec<f32>,
    /// Number of rows in the matrix.
    rows: usize,
    /// Number of columns in the matrix.
    cols: usize,
    /// Color palette to use.
    palette: HeatmapPalette,
    /// Custom color scale (overrides palette if set).
    custom_scale: Option<ColorScale>,
    /// Output width in pixels.
    width: u32,
    /// Output height in pixels.
    height: u32,
    /// Margin around the heatmap.
    margin: u32,
    /// Show cell borders.
    show_borders: bool,
    /// Border color.
    border_color: Rgba,
    /// Border width in pixels.
    border_width: u32,
}

impl Default for Heatmap {
    fn default() -> Self {
        Self::new()
    }
}

impl Heatmap {
    /// Create a new heatmap builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            rows: 0,
            cols: 0,
            palette: HeatmapPalette::default(),
            custom_scale: None,
            width: 800,
            height: 600,
            margin: 40,
            show_borders: true,
            border_color: Rgba::rgb(200, 200, 200),
            border_width: 1,
        }
    }

    /// Set the 2D data matrix.
    ///
    /// Data should be provided in row-major order.
    #[must_use]
    pub fn data(mut self, data: &[f32], rows: usize, cols: usize) -> Self {
        self.data = data.to_vec();
        self.rows = rows;
        self.cols = cols;
        self
    }

    /// Set the data from a 2D vector (row-major).
    #[must_use]
    pub fn data_2d(mut self, data: &[Vec<f32>]) -> Self {
        if data.is_empty() {
            return self;
        }

        self.rows = data.len();
        self.cols = data[0].len();
        self.data = data.iter().flatten().copied().collect();
        self
    }

    /// Set the color palette.
    #[must_use]
    pub fn palette(mut self, palette: HeatmapPalette) -> Self {
        self.palette = palette;
        self
    }

    /// Set a custom color scale.
    #[must_use]
    pub fn color_scale(mut self, scale: ColorScale) -> Self {
        self.custom_scale = Some(scale);
        self
    }

    /// Set the margin around the heatmap.
    #[must_use]
    pub fn margin(mut self, margin: u32) -> Self {
        self.margin = margin;
        self
    }

    /// Enable or disable cell borders.
    #[must_use]
    pub fn borders(mut self, show: bool) -> Self {
        self.show_borders = show;
        self
    }

    /// Set the border color.
    #[must_use]
    pub fn border_color(mut self, color: Rgba) -> Self {
        self.border_color = color;
        self
    }

    /// Set the border width.
    #[must_use]
    pub fn border_width(mut self, width: u32) -> Self {
        self.border_width = width;
        self
    }

    /// Build and validate the heatmap.
    ///
    /// # Errors
    ///
    /// Returns an error if data is empty or dimensions don't match.
    pub fn build(self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(Error::EmptyData);
        }

        if self.rows == 0 || self.cols == 0 {
            return Err(Error::InvalidDimensions {
                width: self.cols as u32,
                height: self.rows as u32,
            });
        }

        let expected_len = self.rows * self.cols;
        if self.data.len() != expected_len {
            return Err(Error::DataLengthMismatch { x_len: expected_len, y_len: self.data.len() });
        }

        Ok(self)
    }

    /// Get the data extent (min, max).
    fn data_extent(&self) -> (f32, f32) {
        let min = self.data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (min, max)
    }

    /// Create a color scale based on the palette.
    fn create_color_scale(&self) -> Option<ColorScale> {
        let (min, max) = self.data_extent();

        // Handle case where all values are the same
        let (min, max) =
            if (max - min).abs() < f32::EPSILON { (min - 0.5, max + 0.5) } else { (min, max) };

        if let Some(ref custom) = self.custom_scale {
            return Some(custom.clone());
        }

        match self.palette {
            HeatmapPalette::Viridis => ColorScale::viridis((min, max)),
            HeatmapPalette::Blues => ColorScale::blues((min, max)),
            HeatmapPalette::RedBlue => ColorScale::red_blue((min, max)),
            HeatmapPalette::Magma => ColorScale::magma((min, max)),
            HeatmapPalette::Heat => ColorScale::heat((min, max)),
            HeatmapPalette::Greyscale => ColorScale::greyscale((min, max)),
        }
    }

    /// Render the heatmap to a framebuffer.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering fails.
    pub fn render(&self, fb: &mut Framebuffer) -> Result<()> {
        let color_scale = self.create_color_scale().ok_or(Error::EmptyData)?;

        // Calculate plot area
        let plot_width = self.width - 2 * self.margin;
        let plot_height = self.height - 2 * self.margin;

        // Calculate cell dimensions
        if self.cols == 0 || self.rows == 0 {
            return Ok(());
        }
        let cell_width = plot_width / self.cols as u32;
        let cell_height = plot_height / self.rows as u32;

        // Render cells
        for row in 0..self.rows {
            for col in 0..self.cols {
                let idx = row * self.cols + col;
                let value = self.data[idx];
                let color = color_scale.scale(value);

                let x = self.margin + (col as u32) * cell_width;
                let y = self.margin + (row as u32) * cell_height;

                // Draw filled cell
                fb.fill_rect(x, y, cell_width, cell_height, color);

                // Draw border if enabled
                if self.show_borders && self.border_width > 0 {
                    self.draw_cell_border(fb, x, y, cell_width, cell_height);
                }
            }
        }

        Ok(())
    }

    /// Draw a cell border.
    fn draw_cell_border(&self, fb: &mut Framebuffer, x: u32, y: u32, width: u32, height: u32) {
        let bw = self.border_width;

        // Right border
        if x + width <= fb.width() {
            fb.fill_rect(x + width - bw, y, bw, height, self.border_color);
        }

        // Bottom border
        if y + height <= fb.height() {
            fb.fill_rect(x, y + height - bw, width, bw, self.border_color);
        }
    }

    /// Render to a new framebuffer.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering fails.
    pub fn to_framebuffer(&self) -> Result<Framebuffer> {
        let mut fb = Framebuffer::new(self.width, self.height)?;
        fb.clear(Rgba::WHITE);
        self.render(&mut fb)?;
        Ok(fb)
    }

    /// Get the number of rows.
    #[must_use]
    pub const fn row_count(&self) -> usize {
        self.rows
    }

    /// Get the number of columns.
    #[must_use]
    pub const fn col_count(&self) -> usize {
        self.cols
    }

    /// Get the total cell count.
    #[must_use]
    pub const fn cell_count(&self) -> usize {
        self.rows * self.cols
    }
}

impl batuta_common::display::WithDimensions for Heatmap {
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
    fn test_heatmap_builder() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let heatmap = Heatmap::new()
            .data(&data, 2, 3)
            .palette(HeatmapPalette::Viridis)
            .build()
            .expect("builder should produce valid result");

        assert_eq!(heatmap.row_count(), 2);
        assert_eq!(heatmap.col_count(), 3);
        assert_eq!(heatmap.cell_count(), 6);
    }

    #[test]
    fn test_heatmap_empty_data() {
        let result = Heatmap::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_heatmap_dimension_mismatch() {
        let data = vec![1.0, 2.0, 3.0];
        let result = Heatmap::new().data(&data, 2, 3).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_heatmap_render() {
        let data = vec![0.0, 0.5, 1.0, 0.25, 0.75, 0.5];
        let heatmap = Heatmap::new()
            .data(&data, 2, 3)
            .dimensions(100, 100)
            .build()
            .expect("builder should produce valid result");

        let fb = heatmap.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_heatmap_data_2d() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let heatmap =
            Heatmap::new().data_2d(&data).build().expect("builder should produce valid result");

        assert_eq!(heatmap.row_count(), 3);
        assert_eq!(heatmap.col_count(), 2);
    }

    #[test]
    fn test_heatmap_palettes() {
        let data = vec![0.0, 0.5, 1.0, 1.5];

        for palette in [
            HeatmapPalette::Viridis,
            HeatmapPalette::Blues,
            HeatmapPalette::RedBlue,
            HeatmapPalette::Magma,
            HeatmapPalette::Heat,
            HeatmapPalette::Greyscale,
        ] {
            let heatmap = Heatmap::new()
                .data(&data, 2, 2)
                .palette(palette)
                .dimensions(100, 100)
                .build()
                .expect("operation should succeed");

            let result = heatmap.to_framebuffer();
            assert!(result.is_ok(), "Failed for palette {palette:?}");
        }
    }

    #[test]
    fn test_heatmap_no_borders() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let heatmap = Heatmap::new()
            .data(&data, 2, 2)
            .borders(false)
            .dimensions(100, 100)
            .build()
            .expect("builder should produce valid result");

        let fb = heatmap.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_heatmap_custom_color_scale() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let scale = ColorScale::new(vec![Rgba::RED, Rgba::GREEN, Rgba::BLUE], (0.0, 3.0))
            .expect("color scale creation should succeed");

        let heatmap = Heatmap::new()
            .data(&data, 2, 2)
            .color_scale(scale)
            .dimensions(100, 100)
            .build()
            .expect("operation should succeed");

        let fb = heatmap.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_heatmap_constant_values() {
        // All same values should not cause division by zero
        let data = vec![5.0, 5.0, 5.0, 5.0];
        let heatmap = Heatmap::new()
            .data(&data, 2, 2)
            .dimensions(100, 100)
            .build()
            .expect("builder should produce valid result");

        let fb = heatmap.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_heatmap_default() {
        let heatmap = Heatmap::default();
        let result = heatmap.build();
        assert!(result.is_err()); // No data
    }

    #[test]
    fn test_heatmap_clone_debug() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let heatmap = Heatmap::new().data(&data, 2, 2);
        let cloned = heatmap.clone();
        let debug = format!("{cloned:?}");
        assert!(debug.contains("Heatmap"));
    }

    #[test]
    fn test_heatmap_palette_default() {
        assert_eq!(HeatmapPalette::default(), HeatmapPalette::Viridis);
    }

    #[test]
    fn test_heatmap_palette_debug() {
        let palettes = [
            HeatmapPalette::Viridis,
            HeatmapPalette::Blues,
            HeatmapPalette::RedBlue,
            HeatmapPalette::Magma,
            HeatmapPalette::Heat,
            HeatmapPalette::Greyscale,
        ];
        for p in palettes {
            let debug = format!("{p:?}");
            assert!(!debug.is_empty());
            let cloned = p;
            assert_eq!(p, cloned);
        }
    }

    #[test]
    fn test_heatmap_margin() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let heatmap = Heatmap::new()
            .data(&data, 2, 2)
            .margin(20)
            .dimensions(100, 100)
            .build()
            .expect("builder should produce valid result");

        let fb = heatmap.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_heatmap_border_customization() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let heatmap = Heatmap::new()
            .data(&data, 2, 2)
            .border_color(Rgba::BLACK)
            .border_width(2)
            .dimensions(100, 100)
            .build()
            .expect("operation should succeed");

        let fb = heatmap.to_framebuffer();
        assert!(fb.is_ok());
    }
}
