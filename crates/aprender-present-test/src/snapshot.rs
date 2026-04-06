//! Visual regression testing via snapshot comparison.
//!
//! Pure Rust implementation - no external dependencies.

use presentar_core::draw::DrawCommand;
use presentar_core::Color;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

/// Image data for snapshot comparison.
#[derive(Debug, Clone)]
pub struct Image {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// RGBA pixel data (4 bytes per pixel)
    pub data: Vec<u8>,
}

impl Image {
    /// Create a new image with the given dimensions.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width as usize) * (height as usize) * 4;
        Self {
            width,
            height,
            data: vec![0; size],
        }
    }

    /// Create an image filled with a single color.
    #[must_use]
    pub fn filled(width: u32, height: u32, r: u8, g: u8, b: u8, a: u8) -> Self {
        let size = (width as usize) * (height as usize) * 4;
        let mut data = Vec::with_capacity(size);
        for _ in 0..(width * height) {
            data.extend_from_slice(&[r, g, b, a]);
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Get raw bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Get pixel at position.
    #[must_use]
    pub fn get_pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = ((y * self.width + x) * 4) as usize;
        Some([
            self.data[idx],
            self.data[idx + 1],
            self.data[idx + 2],
            self.data[idx + 3],
        ])
    }

    /// Set pixel at position.
    pub fn set_pixel(&mut self, x: u32, y: u32, rgba: [u8; 4]) {
        if x < self.width && y < self.height {
            let idx = ((y * self.width + x) * 4) as usize;
            self.data[idx..idx + 4].copy_from_slice(&rgba);
        }
    }

    /// Fill a rectangle with a color (software rendering).
    #[allow(clippy::cast_possible_wrap)]
    pub fn fill_rect(&mut self, x: i32, y: i32, width: u32, height: u32, color: &Color) {
        let rgba = [
            (color.r * 255.0) as u8,
            (color.g * 255.0) as u8,
            (color.b * 255.0) as u8,
            (color.a * 255.0) as u8,
        ];

        let x_start = x.max(0) as u32;
        let y_start = y.max(0) as u32;
        let x_end = ((x + width as i32) as u32).min(self.width);
        let y_end = ((y + height as i32) as u32).min(self.height);

        for py in y_start..y_end {
            for px in x_start..x_end {
                self.blend_pixel(px, py, rgba);
            }
        }
    }

    /// Blend a pixel with alpha compositing.
    #[allow(clippy::cast_lossless, clippy::needless_range_loop)]
    fn blend_pixel(&mut self, x: u32, y: u32, src: [u8; 4]) {
        if x >= self.width || y >= self.height {
            return;
        }
        let idx = ((y * self.width + x) * 4) as usize;

        let src_a = src[3] as f32 / 255.0;
        if src_a >= 0.999 {
            self.data[idx..idx + 4].copy_from_slice(&src);
            return;
        }

        let dst_a = self.data[idx + 3] as f32 / 255.0;
        let out_a = src_a + dst_a * (1.0 - src_a);

        if out_a > 0.0 {
            for i in 0..3 {
                let src_c = src[i] as f32 / 255.0;
                let dst_c = self.data[idx + i] as f32 / 255.0;
                let out_c = (src_c * src_a + dst_c * dst_a * (1.0 - src_a)) / out_a;
                self.data[idx + i] = (out_c * 255.0) as u8;
            }
            self.data[idx + 3] = (out_a * 255.0) as u8;
        }
    }

    /// Fill a circle with a color (software rendering).
    #[allow(clippy::cast_possible_wrap)]
    pub fn fill_circle(&mut self, cx: i32, cy: i32, radius: u32, color: &Color) {
        let rgba = [
            (color.r * 255.0) as u8,
            (color.g * 255.0) as u8,
            (color.b * 255.0) as u8,
            (color.a * 255.0) as u8,
        ];

        let r = radius as i32;
        let r_sq = (r * r) as f32;

        for dy in -r..=r {
            for dx in -r..=r {
                let dist_sq = (dx * dx + dy * dy) as f32;
                if dist_sq <= r_sq {
                    let px = cx + dx;
                    let py = cy + dy;
                    if px >= 0 && py >= 0 {
                        self.blend_pixel(px as u32, py as u32, rgba);
                    }
                }
            }
        }
    }

    /// Render draw commands to this image (software renderer).
    pub fn render(&mut self, commands: &[DrawCommand]) {
        for cmd in commands {
            self.render_command(cmd);
        }
    }

    fn render_command(&mut self, cmd: &DrawCommand) {
        match cmd {
            DrawCommand::Rect { bounds, style, .. } => {
                if let Some(fill) = style.fill {
                    self.fill_rect(
                        bounds.x as i32,
                        bounds.y as i32,
                        bounds.width as u32,
                        bounds.height as u32,
                        &fill,
                    );
                }
            }
            DrawCommand::Circle {
                center,
                radius,
                style,
            } => {
                if let Some(fill) = style.fill {
                    self.fill_circle(center.x as i32, center.y as i32, *radius as u32, &fill);
                }
            }
            DrawCommand::Group { children, .. } => {
                self.render(children);
            }
            _ => {}
        }
    }

    /// Compute a hash of the image data.
    #[must_use]
    pub fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.width.hash(&mut hasher);
        self.height.hash(&mut hasher);
        self.data.hash(&mut hasher);
        hasher.finish()
    }

    /// Extract a sub-region of the image.
    #[must_use]
    pub fn region(&self, x: u32, y: u32, width: u32, height: u32) -> Image {
        let mut result = Image::new(width, height);

        for dy in 0..height {
            for dx in 0..width {
                if let Some(pixel) = self.get_pixel(x + dx, y + dy) {
                    result.set_pixel(dx, dy, pixel);
                }
            }
        }

        result
    }

    /// Scale image to new dimensions (nearest neighbor).
    #[must_use]
    pub fn scale(&self, new_width: u32, new_height: u32) -> Image {
        let mut result = Image::new(new_width, new_height);

        if self.width == 0 || self.height == 0 {
            return result;
        }

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = (x as f32 * self.width as f32 / new_width as f32) as u32;
                let src_y = (y as f32 * self.height as f32 / new_height as f32) as u32;
                if let Some(pixel) = self.get_pixel(src_x, src_y) {
                    result.set_pixel(x, y, pixel);
                }
            }
        }

        result
    }

    /// Count pixels matching a specific color (with tolerance).
    #[must_use]
    pub fn count_color(&self, target: [u8; 4], tolerance: u8) -> usize {
        let mut count = 0;

        for y in 0..self.height {
            for x in 0..self.width {
                if let Some(pixel) = self.get_pixel(x, y) {
                    let matches = (0..4).all(|i| {
                        let diff = (pixel[i] as i32 - target[i] as i32).unsigned_abs() as u8;
                        diff <= tolerance
                    });
                    if matches {
                        count += 1;
                    }
                }
            }
        }

        count
    }

    /// Calculate histogram for each channel.
    #[must_use]
    pub fn histogram(&self) -> [[u32; 256]; 4] {
        let mut hist = [[0u32; 256]; 4];

        for chunk in self.data.chunks_exact(4) {
            for (i, &val) in chunk.iter().enumerate() {
                hist[i][val as usize] += 1;
            }
        }

        hist
    }

    /// Calculate mean color.
    #[must_use]
    pub fn mean_color(&self) -> [f32; 4] {
        let pixel_count = (self.width * self.height) as f64;
        if pixel_count == 0.0 {
            return [0.0; 4];
        }

        let mut sums = [0.0f64; 4];

        for chunk in self.data.chunks_exact(4) {
            for (i, &val) in chunk.iter().enumerate() {
                sums[i] += f64::from(val);
            }
        }

        [
            (sums[0] / pixel_count) as f32,
            (sums[1] / pixel_count) as f32,
            (sums[2] / pixel_count) as f32,
            (sums[3] / pixel_count) as f32,
        ]
    }

    /// Draw a line using Bresenham's algorithm.
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    pub fn draw_line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: &Color) {
        let rgba = [
            (color.r * 255.0) as u8,
            (color.g * 255.0) as u8,
            (color.b * 255.0) as u8,
            (color.a * 255.0) as u8,
        ];

        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;

        let mut x = x0;
        let mut y = y0;

        loop {
            if x >= 0 && y >= 0 {
                self.blend_pixel(x as u32, y as u32, rgba);
            }

            if x == x1 && y == y1 {
                break;
            }

            let e2 = 2 * err;
            if e2 >= dy {
                if x == x1 {
                    break;
                }
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                if y == y1 {
                    break;
                }
                err += dx;
                y += sy;
            }
        }
    }

    /// Draw a stroked rectangle.
    #[allow(clippy::cast_possible_wrap)]
    pub fn stroke_rect(&mut self, x: i32, y: i32, width: u32, height: u32, color: &Color) {
        let x2 = x + width as i32 - 1;
        let y2 = y + height as i32 - 1;

        self.draw_line(x, y, x2, y, color);
        self.draw_line(x2, y, x2, y2, color);
        self.draw_line(x2, y2, x, y2, color);
        self.draw_line(x, y2, x, y, color);
    }
}

/// Result of a snapshot comparison.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Byte-level difference ratio (0.0 to 1.0)
    pub byte_diff: f64,
    /// Perceptual difference (0.0 to 1.0)
    pub perceptual_diff: f64,
    /// Structural similarity index (0.0 to 1.0, 1.0 = identical)
    pub ssim: f64,
    /// Whether dimensions match
    pub same_dimensions: bool,
    /// Number of changed pixels
    pub changed_pixels: u64,
    /// Total pixels
    pub total_pixels: u64,
}

impl ComparisonResult {
    /// Check if images are considered equivalent given a threshold.
    #[must_use]
    pub fn is_match(&self, threshold: f64) -> bool {
        self.same_dimensions && self.byte_diff <= threshold
    }

    /// Get the percentage of changed pixels.
    #[must_use]
    pub fn changed_percentage(&self) -> f64 {
        if self.total_pixels == 0 {
            return 0.0;
        }
        self.changed_pixels as f64 / self.total_pixels as f64 * 100.0
    }
}

/// Snapshot comparison utilities.
pub struct Snapshot;

impl Snapshot {
    /// Compare an actual image against a baseline snapshot.
    ///
    /// # Panics
    ///
    /// Panics if the difference exceeds the threshold.
    pub fn assert_match(name: &str, actual: &Image, threshold: f64) {
        let baseline_path = Self::baseline_path(name);

        if let Some(baseline) = Self::load_baseline(&baseline_path) {
            let diff_ratio = Self::diff(&baseline, actual);

            if diff_ratio > threshold {
                // Save actual and diff for debugging
                let actual_path = Self::actual_path(name);
                let diff_path = Self::diff_path(name);

                Self::save_image(&actual_path, actual);
                Self::save_diff(&diff_path, &baseline, actual);

                panic!(
                    "Visual regression '{}': {:.2}% diff (threshold: {:.2}%)\n\
                     Baseline: {}\n\
                     Actual: {}\n\
                     Diff: {}",
                    name,
                    diff_ratio * 100.0,
                    threshold * 100.0,
                    baseline_path.display(),
                    actual_path.display(),
                    diff_path.display()
                );
            }
        } else if std::env::var("SNAPSHOT_UPDATE").is_ok() {
            // Create new baseline
            Self::save_image(&baseline_path, actual);
            println!("Created new baseline: {}", baseline_path.display());
        } else {
            panic!(
                "No baseline found for '{}'. Run with SNAPSHOT_UPDATE=1 to create.\n\
                 Expected path: {}",
                name,
                baseline_path.display()
            );
        }
    }

    /// Calculate difference ratio between two images (byte-level).
    #[must_use]
    pub fn diff(a: &Image, b: &Image) -> f64 {
        if a.width != b.width || a.height != b.height {
            return 1.0; // Completely different if sizes don't match
        }

        let mut diff_count = 0u64;
        let total = a.data.len() as u64;

        for (a_byte, b_byte) in a.data.iter().zip(b.data.iter()) {
            if a_byte != b_byte {
                diff_count += 1;
            }
        }

        diff_count as f64 / total as f64
    }

    /// Calculate perceptual difference (color distance per pixel).
    /// Returns mean squared error normalized to 0.0-1.0.
    #[must_use]
    pub fn perceptual_diff(a: &Image, b: &Image) -> f64 {
        if a.width != b.width || a.height != b.height {
            return 1.0;
        }

        let pixel_count = f64::from(a.width * a.height);
        if pixel_count == 0.0 {
            return 0.0;
        }

        let mut total_error = 0.0;

        for i in 0..(a.data.len() / 4) {
            let idx = i * 4;
            // Calculate color distance (squared Euclidean in RGB space)
            for j in 0..3 {
                let diff = f64::from(a.data[idx + j]) - f64::from(b.data[idx + j]);
                total_error += diff * diff;
            }
        }

        // Normalize: max possible error is 255^2 * 3 * pixel_count
        let max_error = 255.0 * 255.0 * 3.0 * pixel_count;
        total_error / max_error
    }

    /// Generate a difference image highlighting changed pixels.
    #[must_use]
    pub fn generate_diff_image(a: &Image, b: &Image) -> Image {
        let width = a.width.max(b.width);
        let height = a.height.max(b.height);
        let mut diff = Image::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let pixel_a = a.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                let pixel_b = b.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);

                let color_diff: i32 = (0..3)
                    .map(|i| (i32::from(pixel_a[i]) - i32::from(pixel_b[i])).abs())
                    .sum();

                if color_diff == 0 {
                    // Same pixel - show dimmed
                    diff.set_pixel(x, y, [pixel_a[0] / 4, pixel_a[1] / 4, pixel_a[2] / 4, 255]);
                } else {
                    // Different - highlight in red
                    let intensity = (color_diff.min(255 * 3) / 3) as u8;
                    diff.set_pixel(x, y, [255, intensity, intensity, 255]);
                }
            }
        }

        diff
    }

    /// Perform a comprehensive comparison between two images.
    #[must_use]
    pub fn compare(a: &Image, b: &Image) -> ComparisonResult {
        let same_dimensions = a.width == b.width && a.height == b.height;
        let total_pixels = u64::from(a.width) * u64::from(a.height);

        if !same_dimensions {
            return ComparisonResult {
                byte_diff: 1.0,
                perceptual_diff: 1.0,
                ssim: 0.0,
                same_dimensions: false,
                changed_pixels: total_pixels,
                total_pixels,
            };
        }

        let byte_diff = Self::diff(a, b);
        let perceptual_diff = Self::perceptual_diff(a, b);
        let ssim = Self::ssim(a, b);
        let changed_pixels = Self::count_changed_pixels(a, b);

        ComparisonResult {
            byte_diff,
            perceptual_diff,
            ssim,
            same_dimensions: true,
            changed_pixels,
            total_pixels,
        }
    }

    /// Count the number of pixels that differ between two images.
    #[must_use]
    pub fn count_changed_pixels(a: &Image, b: &Image) -> u64 {
        if a.width != b.width || a.height != b.height {
            return u64::from(a.width.max(b.width)) * u64::from(a.height.max(b.height));
        }

        let mut count = 0u64;
        for i in 0..(a.data.len() / 4) {
            let idx = i * 4;
            let diff = (0..4).any(|j| a.data[idx + j] != b.data[idx + j]);
            if diff {
                count += 1;
            }
        }
        count
    }

    /// Calculate Structural Similarity Index (SSIM).
    ///
    /// Simplified implementation of SSIM that considers luminance similarity.
    /// Returns 1.0 for identical images, 0.0 for completely different.
    #[must_use]
    pub fn ssim(a: &Image, b: &Image) -> f64 {
        if a.width != b.width || a.height != b.height {
            return 0.0;
        }

        let pixel_count = (a.width * a.height) as usize;
        if pixel_count == 0 {
            return 1.0;
        }

        // Calculate luminance for each image
        let mut lum_a = Vec::with_capacity(pixel_count);
        let mut lum_b = Vec::with_capacity(pixel_count);

        for i in 0..pixel_count {
            let idx = i * 4;
            // Standard luminance calculation
            let la = 0.299 * f64::from(a.data[idx])
                + 0.587 * f64::from(a.data[idx + 1])
                + 0.114 * f64::from(a.data[idx + 2]);
            let lb = 0.299 * f64::from(b.data[idx])
                + 0.587 * f64::from(b.data[idx + 1])
                + 0.114 * f64::from(b.data[idx + 2]);
            lum_a.push(la);
            lum_b.push(lb);
        }

        // Calculate means
        let mean_a: f64 = lum_a.iter().sum::<f64>() / pixel_count as f64;
        let mean_b: f64 = lum_b.iter().sum::<f64>() / pixel_count as f64;

        // Calculate variances and covariance
        let mut var_a = 0.0;
        let mut var_b = 0.0;
        let mut covar = 0.0;

        for i in 0..pixel_count {
            let da = lum_a[i] - mean_a;
            let db = lum_b[i] - mean_b;
            var_a += da * da;
            var_b += db * db;
            covar += da * db;
        }

        var_a /= pixel_count as f64;
        var_b /= pixel_count as f64;
        covar /= pixel_count as f64;

        // SSIM constants
        const C1: f64 = 6.5025; // (0.01 * 255)^2
        const C2: f64 = 58.5225; // (0.03 * 255)^2

        // Simplified SSIM formula
        let numerator = (2.0 * mean_a * mean_b + C1) * (2.0 * covar + C2);
        let denominator = (mean_a * mean_a + mean_b * mean_b + C1) * (var_a + var_b + C2);

        numerator / denominator
    }

    /// Compare a region of two images.
    #[must_use]
    pub fn compare_region(
        a: &Image,
        b: &Image,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> ComparisonResult {
        let region_a = a.region(x, y, width, height);
        let region_b = b.region(x, y, width, height);
        Self::compare(&region_a, &region_b)
    }

    /// Assert that a region matches within threshold.
    ///
    /// # Panics
    ///
    /// Panics if the region difference exceeds the threshold.
    pub fn assert_region_match(
        name: &str,
        actual: &Image,
        baseline: &Image,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        threshold: f64,
    ) {
        let result = Self::compare_region(actual, baseline, x, y, width, height);
        if !result.is_match(threshold) {
            panic!(
                "Region mismatch in '{}' at ({}, {}) {}x{}: {:.2}% diff (threshold: {:.2}%)",
                name,
                x,
                y,
                width,
                height,
                result.byte_diff * 100.0,
                threshold * 100.0
            );
        }
    }

    fn baseline_path(name: &str) -> PathBuf {
        PathBuf::from(format!("tests/snapshots/{name}.png"))
    }

    fn actual_path(name: &str) -> PathBuf {
        PathBuf::from(format!("tests/snapshots/{name}.actual.png"))
    }

    fn diff_path(name: &str) -> PathBuf {
        PathBuf::from(format!("tests/snapshots/{name}.diff.png"))
    }

    fn load_baseline(path: &Path) -> Option<Image> {
        // Simplified - would use a pure Rust PNG decoder
        if path.exists() {
            // Return placeholder
            Some(Image::new(100, 100))
        } else {
            None
        }
    }

    const fn save_image(_path: &Path, _image: &Image) {
        // Would use a pure Rust PNG encoder
        // Placeholder implementation
    }

    const fn save_diff(_path: &Path, _baseline: &Image, _actual: &Image) {
        // Would generate a visual diff image
        // Placeholder implementation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_new() {
        let img = Image::new(100, 100);
        assert_eq!(img.width, 100);
        assert_eq!(img.height, 100);
        assert_eq!(img.data.len(), 100 * 100 * 4);
    }

    #[test]
    fn test_image_filled() {
        let img = Image::filled(10, 10, 255, 0, 0, 255);
        assert_eq!(img.get_pixel(0, 0), Some([255, 0, 0, 255]));
        assert_eq!(img.get_pixel(5, 5), Some([255, 0, 0, 255]));
    }

    #[test]
    fn test_image_get_set_pixel() {
        let mut img = Image::new(10, 10);
        img.set_pixel(5, 5, [255, 128, 64, 255]);
        assert_eq!(img.get_pixel(5, 5), Some([255, 128, 64, 255]));
    }

    #[test]
    fn test_image_get_pixel_out_of_bounds() {
        let img = Image::new(10, 10);
        assert_eq!(img.get_pixel(100, 100), None);
    }

    #[test]
    fn test_diff_identical() {
        let a = Image::filled(10, 10, 255, 0, 0, 255);
        let b = Image::filled(10, 10, 255, 0, 0, 255);
        assert_eq!(Snapshot::diff(&a, &b), 0.0);
    }

    #[test]
    fn test_diff_completely_different() {
        let a = Image::filled(10, 10, 255, 0, 0, 255);
        let b = Image::filled(10, 10, 0, 255, 0, 255);
        let diff = Snapshot::diff(&a, &b);
        assert!(diff > 0.0);
    }

    #[test]
    fn test_diff_different_sizes() {
        let a = Image::new(10, 10);
        let b = Image::new(20, 20);
        assert_eq!(Snapshot::diff(&a, &b), 1.0);
    }

    #[test]
    fn test_diff_partial() {
        let a = Image::filled(10, 10, 255, 0, 0, 255);
        let mut b = Image::filled(10, 10, 255, 0, 0, 255);

        // Change one pixel from [255,0,0,255] to [0,0,0,255]
        // Only R channel changes (255 -> 0)
        b.set_pixel(0, 0, [0, 0, 0, 255]);

        let diff = Snapshot::diff(&a, &b);
        // 1 byte changed out of 400 total bytes = 0.0025
        assert!((diff - 0.0025).abs() < 0.001);
    }

    #[test]
    fn test_fill_rect() {
        let mut img = Image::new(100, 100);
        img.fill_rect(10, 10, 20, 20, &Color::RED);

        // Inside rect should be red
        assert_eq!(img.get_pixel(15, 15), Some([255, 0, 0, 255]));
        // Outside rect should be black/transparent
        assert_eq!(img.get_pixel(0, 0), Some([0, 0, 0, 0]));
    }

    #[test]
    fn test_fill_rect_clipping() {
        let mut img = Image::new(50, 50);
        // Rect extends beyond bounds
        img.fill_rect(40, 40, 20, 20, &Color::BLUE);

        // Inside visible portion should be blue
        assert_eq!(img.get_pixel(45, 45), Some([0, 0, 255, 255]));
        // Edge should also be blue
        assert_eq!(img.get_pixel(49, 49), Some([0, 0, 255, 255]));
    }

    #[test]
    fn test_fill_circle() {
        let mut img = Image::new(100, 100);
        img.fill_circle(50, 50, 10, &Color::GREEN);

        // Center should be green
        assert_eq!(img.get_pixel(50, 50), Some([0, 255, 0, 255]));
        // Just outside radius should be transparent
        assert_eq!(img.get_pixel(50, 65), Some([0, 0, 0, 0]));
    }

    #[test]
    fn test_render_rect_command() {
        use presentar_core::draw::DrawCommand;
        use presentar_core::Rect;

        let mut img = Image::new(100, 100);
        let commands = vec![DrawCommand::filled_rect(
            Rect::new(10.0, 10.0, 30.0, 30.0),
            Color::RED,
        )];
        img.render(&commands);

        assert_eq!(img.get_pixel(20, 20), Some([255, 0, 0, 255]));
    }

    #[test]
    fn test_render_circle_command() {
        use presentar_core::draw::DrawCommand;
        use presentar_core::Point;

        let mut img = Image::new(100, 100);
        let commands = vec![DrawCommand::filled_circle(
            Point::new(50.0, 50.0),
            15.0,
            Color::BLUE,
        )];
        img.render(&commands);

        assert_eq!(img.get_pixel(50, 50), Some([0, 0, 255, 255]));
    }

    #[test]
    fn test_perceptual_diff_identical() {
        let a = Image::filled(10, 10, 128, 64, 32, 255);
        let b = Image::filled(10, 10, 128, 64, 32, 255);
        assert_eq!(Snapshot::perceptual_diff(&a, &b), 0.0);
    }

    #[test]
    fn test_perceptual_diff_different() {
        let a = Image::filled(10, 10, 255, 255, 255, 255);
        let b = Image::filled(10, 10, 0, 0, 0, 255);
        let diff = Snapshot::perceptual_diff(&a, &b);
        // Maximum difference white vs black = 1.0
        assert!((diff - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_perceptual_diff_partial() {
        let a = Image::filled(10, 10, 100, 100, 100, 255);
        let b = Image::filled(10, 10, 110, 100, 100, 255);
        let diff = Snapshot::perceptual_diff(&a, &b);
        // Small difference should be small value
        assert!(diff > 0.0);
        assert!(diff < 0.01);
    }

    #[test]
    fn test_generate_diff_image() {
        let a = Image::filled(10, 10, 255, 0, 0, 255);
        let mut b = Image::filled(10, 10, 255, 0, 0, 255);
        b.set_pixel(5, 5, [0, 255, 0, 255]);

        let diff = Snapshot::generate_diff_image(&a, &b);

        // Changed pixel should be highlighted (red channel = 255)
        let pixel = diff.get_pixel(5, 5).expect("pixel exists");
        assert_eq!(pixel[0], 255); // Red highlight

        // Unchanged pixels should be dimmed
        let unchanged = diff.get_pixel(0, 0).expect("pixel exists");
        assert!(unchanged[0] < 100); // Dimmed red
    }

    #[test]
    fn test_alpha_blending() {
        let mut img = Image::filled(10, 10, 255, 0, 0, 255); // Red background
        img.fill_rect(0, 0, 10, 10, &Color::new(0.0, 0.0, 1.0, 0.5)); // 50% blue overlay

        let pixel = img.get_pixel(5, 5).expect("pixel exists");
        // Should be a blend of red and blue
        assert!(pixel[0] > 100); // Still has red
        assert!(pixel[2] > 100); // Has blue
    }

    // ===== New functionality tests =====

    #[test]
    fn test_image_hash() {
        let a = Image::filled(10, 10, 255, 0, 0, 255);
        let b = Image::filled(10, 10, 255, 0, 0, 255);
        let c = Image::filled(10, 10, 0, 255, 0, 255);

        assert_eq!(a.hash(), b.hash());
        assert_ne!(a.hash(), c.hash());
    }

    #[test]
    fn test_image_region() {
        let mut img = Image::new(100, 100);
        img.fill_rect(10, 10, 20, 20, &Color::RED);

        let region = img.region(10, 10, 20, 20);
        assert_eq!(region.width, 20);
        assert_eq!(region.height, 20);
        assert_eq!(region.get_pixel(5, 5), Some([255, 0, 0, 255]));
    }

    #[test]
    fn test_image_region_out_of_bounds() {
        let img = Image::filled(10, 10, 255, 0, 0, 255);
        let region = img.region(8, 8, 5, 5);

        // Only 2x2 pixels should be valid (from 8-9 in each dimension)
        assert_eq!(region.width, 5);
        assert_eq!(region.height, 5);
        // Inside original image bounds
        assert_eq!(region.get_pixel(0, 0), Some([255, 0, 0, 255]));
        // Outside original image bounds
        assert_eq!(region.get_pixel(3, 3), Some([0, 0, 0, 0]));
    }

    #[test]
    fn test_image_scale() {
        let img = Image::filled(10, 10, 255, 0, 0, 255);
        let scaled = img.scale(20, 20);

        assert_eq!(scaled.width, 20);
        assert_eq!(scaled.height, 20);
        assert_eq!(scaled.get_pixel(10, 10), Some([255, 0, 0, 255]));
    }

    #[test]
    fn test_image_scale_down() {
        let img = Image::filled(20, 20, 255, 0, 0, 255);
        let scaled = img.scale(10, 10);

        assert_eq!(scaled.width, 10);
        assert_eq!(scaled.height, 10);
        assert_eq!(scaled.get_pixel(5, 5), Some([255, 0, 0, 255]));
    }

    #[test]
    fn test_image_count_color() {
        let img = Image::filled(10, 10, 255, 0, 0, 255);
        let count = img.count_color([255, 0, 0, 255], 0);
        assert_eq!(count, 100);

        let count = img.count_color([255, 5, 0, 255], 10);
        assert_eq!(count, 100);

        let count = img.count_color([0, 255, 0, 255], 0);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_image_histogram() {
        let img = Image::filled(10, 10, 255, 128, 0, 255);
        let hist = img.histogram();

        assert_eq!(hist[0][255], 100); // R channel
        assert_eq!(hist[1][128], 100); // G channel
        assert_eq!(hist[2][0], 100); // B channel
        assert_eq!(hist[3][255], 100); // A channel
    }

    #[test]
    fn test_image_mean_color() {
        let img = Image::filled(10, 10, 100, 100, 100, 255);
        let mean = img.mean_color();

        assert!((mean[0] - 100.0).abs() < 0.01);
        assert!((mean[1] - 100.0).abs() < 0.01);
        assert!((mean[2] - 100.0).abs() < 0.01);
        assert!((mean[3] - 255.0).abs() < 0.01);
    }

    #[test]
    fn test_image_draw_line() {
        let mut img = Image::new(100, 100);
        img.draw_line(0, 0, 99, 99, &Color::WHITE);

        // Check start and end
        assert_eq!(img.get_pixel(0, 0), Some([255, 255, 255, 255]));
        assert_eq!(img.get_pixel(99, 99), Some([255, 255, 255, 255]));
        // Check diagonal
        assert_eq!(img.get_pixel(50, 50), Some([255, 255, 255, 255]));
    }

    #[test]
    fn test_image_stroke_rect() {
        let mut img = Image::new(100, 100);
        img.stroke_rect(10, 10, 20, 20, &Color::WHITE);

        // Check corners
        assert_eq!(img.get_pixel(10, 10), Some([255, 255, 255, 255]));
        assert_eq!(img.get_pixel(29, 29), Some([255, 255, 255, 255]));
        // Inside should be transparent
        assert_eq!(img.get_pixel(15, 15), Some([0, 0, 0, 0]));
    }

    #[test]
    fn test_comparison_result_is_match() {
        let result = ComparisonResult {
            byte_diff: 0.01,
            perceptual_diff: 0.01,
            ssim: 0.99,
            same_dimensions: true,
            changed_pixels: 1,
            total_pixels: 100,
        };

        assert!(result.is_match(0.05));
        assert!(!result.is_match(0.005));
    }

    #[test]
    fn test_comparison_result_changed_percentage() {
        let result = ComparisonResult {
            byte_diff: 0.0,
            perceptual_diff: 0.0,
            ssim: 1.0,
            same_dimensions: true,
            changed_pixels: 10,
            total_pixels: 100,
        };

        assert!((result.changed_percentage() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_snapshot_compare_identical() {
        let a = Image::filled(10, 10, 255, 0, 0, 255);
        let b = Image::filled(10, 10, 255, 0, 0, 255);

        let result = Snapshot::compare(&a, &b);

        assert!(result.is_match(0.0));
        assert_eq!(result.byte_diff, 0.0);
        assert_eq!(result.changed_pixels, 0);
        assert!((result.ssim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_snapshot_compare_different_dimensions() {
        let a = Image::new(10, 10);
        let b = Image::new(20, 20);

        let result = Snapshot::compare(&a, &b);

        assert!(!result.same_dimensions);
        assert_eq!(result.byte_diff, 1.0);
        assert_eq!(result.ssim, 0.0);
    }

    #[test]
    fn test_snapshot_count_changed_pixels() {
        let a = Image::filled(10, 10, 255, 0, 0, 255);
        let mut b = Image::filled(10, 10, 255, 0, 0, 255);
        b.set_pixel(0, 0, [0, 255, 0, 255]);
        b.set_pixel(1, 0, [0, 255, 0, 255]);

        let count = Snapshot::count_changed_pixels(&a, &b);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_snapshot_ssim_identical() {
        let a = Image::filled(10, 10, 128, 128, 128, 255);
        let b = Image::filled(10, 10, 128, 128, 128, 255);

        let ssim = Snapshot::ssim(&a, &b);
        assert!((ssim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_snapshot_ssim_different() {
        let a = Image::filled(10, 10, 255, 255, 255, 255);
        let b = Image::filled(10, 10, 0, 0, 0, 255);

        let ssim = Snapshot::ssim(&a, &b);
        assert!(ssim < 0.5); // Should be low for black vs white
    }

    #[test]
    fn test_snapshot_ssim_different_dimensions() {
        let a = Image::new(10, 10);
        let b = Image::new(20, 20);

        let ssim = Snapshot::ssim(&a, &b);
        assert_eq!(ssim, 0.0);
    }

    #[test]
    fn test_snapshot_compare_region() {
        let mut a = Image::new(100, 100);
        a.fill_rect(10, 10, 20, 20, &Color::RED);
        let mut b = Image::new(100, 100);
        b.fill_rect(10, 10, 20, 20, &Color::RED);

        let result = Snapshot::compare_region(&a, &b, 10, 10, 20, 20);
        assert!(result.is_match(0.0));
    }

    #[test]
    fn test_snapshot_compare_region_different() {
        let mut a = Image::new(100, 100);
        a.fill_rect(10, 10, 20, 20, &Color::RED);
        let mut b = Image::new(100, 100);
        b.fill_rect(10, 10, 20, 20, &Color::BLUE);

        let result = Snapshot::compare_region(&a, &b, 10, 10, 20, 20);
        assert!(!result.is_match(0.0));
        assert!(result.byte_diff > 0.0);
    }
}
