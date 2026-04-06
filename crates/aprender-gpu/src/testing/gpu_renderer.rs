//! GPU Output Pixel Renderer
//!
//! Converts f32 GPU output buffers to PNG images for visual regression testing.
//! Uses trueno-viz from sovereign stack - NO external crates.

use trueno_viz::color::Rgba;
use trueno_viz::framebuffer::Framebuffer;
use trueno_viz::output::PngEncoder;

/// RGB color representation (without alpha for palette)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb {
    /// Red component
    pub r: u8,
    /// Green component
    pub g: u8,
    /// Blue component
    pub b: u8,
}

impl Rgb {
    /// Create new RGB color
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Magenta for NaN values
    pub const NAN_COLOR: Self = Self::new(255, 0, 255);
    /// White for +Infinity
    pub const INF_COLOR: Self = Self::new(255, 255, 255);
    /// Black for -Infinity
    pub const NEG_INF_COLOR: Self = Self::new(0, 0, 0);

    /// Convert to trueno-viz Rgba
    #[must_use]
    pub const fn to_rgba(self) -> Rgba {
        Rgba::rgb(self.r, self.g, self.b)
    }
}

/// Color palette for heatmap rendering
#[derive(Debug, Clone)]
pub struct ColorPalette {
    colors: Vec<Rgb>,
}

impl Default for ColorPalette {
    fn default() -> Self {
        Self::viridis()
    }
}

impl ColorPalette {
    /// Viridis colorblind-friendly palette
    #[must_use]
    pub fn viridis() -> Self {
        Self {
            colors: vec![
                Rgb::new(68, 1, 84),
                Rgb::new(59, 82, 139),
                Rgb::new(33, 145, 140),
                Rgb::new(94, 201, 98),
                Rgb::new(253, 231, 37),
            ],
        }
    }

    /// Grayscale palette
    #[must_use]
    pub fn grayscale() -> Self {
        Self {
            colors: vec![
                Rgb::new(0, 0, 0),
                Rgb::new(128, 128, 128),
                Rgb::new(255, 255, 255),
            ],
        }
    }

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    fn interpolate(&self, t: f32) -> Rgb {
        let t = t.clamp(0.0, 1.0);
        let n = self.colors.len() - 1;
        let idx = (t * n as f32).floor() as usize;
        let idx = idx.min(n - 1);
        let local_t = t * n as f32 - idx as f32;

        let c1 = &self.colors[idx];
        let c2 = &self.colors[idx + 1];

        Rgb {
            r: (c1.r as f32 + (c2.r as f32 - c1.r as f32) * local_t) as u8,
            g: (c1.g as f32 + (c2.g as f32 - c1.g as f32) * local_t) as u8,
            b: (c1.b as f32 + (c2.b as f32 - c1.b as f32) * local_t) as u8,
        }
    }
}

/// Renders f32 GPU output to PNG for visual regression testing
/// Uses trueno-viz (sovereign stack) - no external image crates
#[derive(Debug, Clone)]
pub struct GpuPixelRenderer {
    palette: ColorPalette,
    range: Option<(f32, f32)>,
}

impl Default for GpuPixelRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuPixelRenderer {
    /// Create renderer with auto-normalization
    #[must_use]
    pub fn new() -> Self {
        Self {
            palette: ColorPalette::default(),
            range: None,
        }
    }

    /// Set fixed range for normalization
    #[must_use]
    pub const fn with_range(mut self, min: f32, max: f32) -> Self {
        self.range = Some((min, max));
        self
    }

    /// Set color palette
    #[must_use]
    pub fn with_palette(mut self, palette: ColorPalette) -> Self {
        self.palette = palette;
        self
    }

    /// Render f32 buffer to PNG bytes using trueno-viz
    ///
    /// # Panics
    /// Panics if buffer length doesn't match width * height
    #[must_use]
    #[allow(clippy::expect_used)]
    pub fn render_to_png(&self, buffer: &[f32], width: u32, height: u32) -> Vec<u8> {
        assert_eq!(buffer.len(), (width * height) as usize);

        let (min_val, max_val) = self.range.unwrap_or_else(|| {
            let valid: Vec<f32> = buffer.iter().copied().filter(|v| v.is_finite()).collect();
            if valid.is_empty() {
                (0.0, 1.0)
            } else {
                let min = valid.iter().copied().fold(f32::INFINITY, f32::min);
                let max = valid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                (min, max.max(min + f32::EPSILON))
            }
        });

        // Use trueno-viz Framebuffer
        let mut fb = Framebuffer::new(width, height).expect("Failed to create framebuffer");

        for (i, &value) in buffer.iter().enumerate() {
            let x = (i as u32) % width;
            let y = (i as u32) / width;

            let color = if value.is_nan() {
                Rgb::NAN_COLOR
            } else if value.is_infinite() {
                if value > 0.0 {
                    Rgb::INF_COLOR
                } else {
                    Rgb::NEG_INF_COLOR
                }
            } else {
                let t = (value - min_val) / (max_val - min_val);
                self.palette.interpolate(t)
            };

            fb.set_pixel(x, y, color.to_rgba());
        }

        // Encode to PNG using trueno-viz
        PngEncoder::to_bytes(&fb).expect("PNG encoding failed")
    }
}

/// Pixel diff result for visual regression testing
#[derive(Debug, Clone)]
pub struct PixelDiffResult {
    /// Number of pixels that differ
    pub different_pixels: usize,
    /// Total number of pixels
    pub total_pixels: usize,
    /// Maximum color difference found
    pub max_diff: u32,
}

impl PixelDiffResult {
    /// Calculate percentage of different pixels
    #[must_use]
    pub fn diff_percentage(&self) -> f64 {
        if self.total_pixels == 0 {
            0.0
        } else {
            (self.different_pixels as f64 / self.total_pixels as f64) * 100.0
        }
    }

    /// Check if images match within threshold
    #[must_use]
    pub fn matches(&self, threshold: f64) -> bool {
        self.diff_percentage() <= threshold
    }
}

/// Compare two PNG images and return diff result
/// Uses sovereign stack only - no external image crates
#[must_use]
pub fn compare_png_bytes(a: &[u8], b: &[u8], tolerance: u8) -> PixelDiffResult {
    // PNG header check
    let png_header = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    if a.len() < 8 || b.len() < 8 || a[0..8] != png_header || b[0..8] != png_header {
        return PixelDiffResult {
            different_pixels: 1,
            total_pixels: 1,
            max_diff: 255,
        };
    }

    // For exact comparison, just compare bytes
    if a == b {
        return PixelDiffResult {
            different_pixels: 0,
            total_pixels: a.len(),
            max_diff: 0,
        };
    }

    // Byte-level comparison with tolerance
    let min_len = a.len().min(b.len());
    let mut different = 0;
    let mut max_diff: u32 = 0;

    for i in 0..min_len {
        let diff = (a[i] as i32 - b[i] as i32).unsigned_abs();
        if diff > tolerance as u32 {
            different += 1;
            max_diff = max_diff.max(diff);
        }
    }

    // Count length difference as differences
    different += a.len().abs_diff(b.len());

    PixelDiffResult {
        different_pixels: different,
        total_pixels: min_len.max(a.len()).max(b.len()),
        max_diff,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    /// Shared GPU pixel renderer for fast test execution (initialized once)
    static SHARED_RENDERER: OnceLock<GpuPixelRenderer> = OnceLock::new();

    /// Get shared renderer (fast) - always succeeds since GpuPixelRenderer is CPU-based
    fn get_shared_renderer() -> &'static GpuPixelRenderer {
        SHARED_RENDERER.get_or_init(GpuPixelRenderer::new)
    }

    #[test]
    fn test_render_produces_valid_png() {
        let renderer = get_shared_renderer();
        let buffer: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let png = renderer.render_to_png(&buffer, 8, 8);

        // PNG magic bytes
        assert_eq!(
            &png[0..8],
            &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
        );
    }

    #[test]
    fn test_special_values() {
        let renderer = get_shared_renderer();
        let buffer = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5];
        let png = renderer.render_to_png(&buffer, 2, 2);
        assert!(!png.is_empty());
        // Verify PNG header
        assert_eq!(
            &png[0..8],
            &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
        );
    }

    #[test]
    fn test_compare_identical() {
        let renderer = get_shared_renderer();
        let buffer: Vec<f32> = (0..16).map(|i| i as f32 / 16.0).collect();
        let png = renderer.render_to_png(&buffer, 4, 4);

        let result = compare_png_bytes(&png, &png, 0);
        assert_eq!(result.different_pixels, 0);
        assert!(result.matches(0.0));
    }

    #[test]
    fn test_compare_different() {
        let renderer = get_shared_renderer();
        let buffer_a: Vec<f32> = (0..16).map(|i| i as f32 / 16.0).collect();
        let buffer_b: Vec<f32> = (0..16).map(|i| 1.0 - i as f32 / 16.0).collect();

        let png_a = renderer.render_to_png(&buffer_a, 4, 4);
        let png_b = renderer.render_to_png(&buffer_b, 4, 4);

        let result = compare_png_bytes(&png_a, &png_b, 0);
        assert!(result.different_pixels > 0);
    }

    #[test]
    fn test_grayscale_palette() {
        let renderer = GpuPixelRenderer::new().with_palette(ColorPalette::grayscale());
        let buffer: Vec<f32> = (0..9).map(|i| i as f32 / 8.0).collect();
        let png = renderer.render_to_png(&buffer, 3, 3);
        assert!(!png.is_empty());
    }

    #[test]
    fn test_fixed_range() {
        let renderer = GpuPixelRenderer::new().with_range(0.0, 10.0);
        let buffer = vec![0.0, 5.0, 10.0, 15.0]; // 15.0 will be clamped
        let png = renderer.render_to_png(&buffer, 2, 2);
        assert!(!png.is_empty());
    }
}
