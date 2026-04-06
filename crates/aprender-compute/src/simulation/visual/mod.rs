//! Visual Regression Testing (Genchi Genbutsu: Go and See)
//!
//! Provides pixel-perfect validation of compute outputs through
//! heatmap rendering and golden baseline comparison.

use std::path::PathBuf;

/// RGB color for visualization
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
}

/// Color palette for heatmap rendering
#[derive(Debug, Clone)]
pub struct ColorPalette {
    pub(crate) colors: Vec<Rgb>,
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
        Self { colors: vec![Rgb::new(0, 0, 0), Rgb::new(128, 128, 128), Rgb::new(255, 255, 255)] }
    }

    /// Interpolate color at position t (0.0 to 1.0)
    #[must_use]
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    pub fn interpolate(&self, t: f32) -> Rgb {
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

/// Visual regression test configuration (Genchi Genbutsu)
#[derive(Debug, Clone)]
pub struct VisualRegressionConfig {
    /// Golden baseline directory
    pub golden_dir: PathBuf,
    /// Output directory for test results
    pub output_dir: PathBuf,
    /// Maximum allowed different pixels (percentage)
    pub max_diff_pct: f64,
    /// Color palette for visualization
    pub palette: ColorPalette,
}

impl Default for VisualRegressionConfig {
    fn default() -> Self {
        Self {
            golden_dir: PathBuf::from("golden"),
            output_dir: PathBuf::from("test_output"),
            max_diff_pct: 0.0, // Exact match by default
            palette: ColorPalette::default(),
        }
    }
}

impl VisualRegressionConfig {
    /// Create new config with custom golden directory
    #[must_use]
    pub fn new(golden_dir: impl Into<PathBuf>) -> Self {
        Self { golden_dir: golden_dir.into(), ..Default::default() }
    }

    /// Set output directory
    #[must_use]
    pub fn with_output_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.output_dir = dir.into();
        self
    }

    /// Set maximum diff percentage
    #[must_use]
    pub const fn with_max_diff_pct(mut self, pct: f64) -> Self {
        self.max_diff_pct = pct;
        self
    }

    /// Set color palette
    #[must_use]
    pub fn with_palette(mut self, palette: ColorPalette) -> Self {
        self.palette = palette;
        self
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
    pub fn matches(&self, threshold_pct: f64) -> bool {
        self.diff_percentage() <= threshold_pct
    }

    /// Create a passing result (no differences)
    #[must_use]
    pub const fn pass(total_pixels: usize) -> Self {
        Self { different_pixels: 0, total_pixels, max_diff: 0 }
    }
}

/// Simple buffer renderer for SIMD output visualization
///
/// Converts f32 buffers to raw RGBA bytes for testing
#[derive(Debug, Clone)]
pub struct BufferRenderer {
    palette: ColorPalette,
    pub(crate) range: Option<(f32, f32)>,
}

impl Default for BufferRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferRenderer {
    /// Create renderer with auto-normalization
    #[must_use]
    pub fn new() -> Self {
        Self { palette: ColorPalette::default(), range: None }
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

    /// Render f32 buffer to raw RGBA bytes
    ///
    /// Returns Vec<u8> with RGBA pixels (4 bytes per pixel)
    #[must_use]
    pub fn render_to_rgba(&self, buffer: &[f32], width: u32, height: u32) -> Vec<u8> {
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

        let mut rgba = Vec::with_capacity(buffer.len() * 4);

        for &value in buffer {
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

            rgba.push(color.r);
            rgba.push(color.g);
            rgba.push(color.b);
            rgba.push(255); // Alpha
        }

        rgba
    }

    /// Compare two RGBA buffers and return diff result
    #[must_use]
    pub fn compare_rgba(&self, a: &[u8], b: &[u8], tolerance: u8) -> PixelDiffResult {
        if a == b {
            return PixelDiffResult::pass(a.len() / 4);
        }

        let min_len = a.len().min(b.len());
        let mut different = 0;
        let mut max_diff: u32 = 0;

        // Compare pixels (4 bytes each: RGBA)
        for i in (0..min_len).step_by(4) {
            let mut pixel_diff = false;
            for j in 0..4 {
                if i + j < min_len {
                    let diff = (a[i + j] as i32 - b[i + j] as i32).unsigned_abs();
                    if diff > tolerance as u32 {
                        pixel_diff = true;
                        max_diff = max_diff.max(diff);
                    }
                }
            }
            if pixel_diff {
                different += 1;
            }
        }

        // Count size difference as pixel differences
        if a.len() != b.len() {
            different += a.len().abs_diff(b.len()) / 4;
        }

        PixelDiffResult {
            different_pixels: different,
            total_pixels: min_len.max(a.len()).max(b.len()) / 4,
            max_diff,
        }
    }
}

/// Golden baseline manager for visual regression testing
#[derive(Debug, Clone)]
pub struct GoldenBaseline {
    config: VisualRegressionConfig,
}

impl GoldenBaseline {
    /// Create new golden baseline manager
    #[must_use]
    pub fn new(config: VisualRegressionConfig) -> Self {
        Self { config }
    }

    /// Get path for a golden baseline file
    #[must_use]
    pub fn golden_path(&self, name: &str) -> PathBuf {
        self.config.golden_dir.join(format!("{name}.golden"))
    }

    /// Get path for an output file
    #[must_use]
    pub fn output_path(&self, name: &str) -> PathBuf {
        self.config.output_dir.join(format!("{name}.output"))
    }

    /// Get the config
    #[must_use]
    pub const fn config(&self) -> &VisualRegressionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests;
