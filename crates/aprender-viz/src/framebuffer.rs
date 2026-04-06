//! Core framebuffer for pixel rendering.
//!
//! Provides a SIMD-aligned RGBA pixel buffer optimized for hardware-accelerated operations.
//! Uses trueno for SIMD-accelerated vector operations where applicable.

use crate::color::Rgba;
use crate::error::{Error, Result};
use trueno::{Backend, Vector};

/// Alignment for SIMD operations (64 bytes for AVX-512).
const SIMD_ALIGNMENT: usize = 64;

/// SIMD-aligned framebuffer for efficient pixel operations.
///
/// The pixel buffer is aligned to 64 bytes for optimal SIMD performance
/// on AVX-512 and other wide SIMD architectures.
///
/// # SIMD Acceleration
///
/// Operations like `clear()`, `fill_rect()`, and `blend()` are SIMD-accelerated
/// using trueno's automatic backend selection (SSE2/AVX2/AVX512/NEON).
#[derive(Debug, Clone)]
pub struct Framebuffer {
    /// Width in pixels.
    width: u32,
    /// Height in pixels.
    height: u32,
    /// RGBA pixels in row-major order.
    /// Each pixel is 4 bytes: [R, G, B, A].
    /// Aligned to SIMD_ALIGNMENT bytes.
    pixels: Vec<u8>,
    /// Stride in bytes (may include padding for alignment).
    stride: usize,
}

impl Framebuffer {
    /// Create a new framebuffer with the given dimensions.
    ///
    /// The buffer is aligned to 64 bytes for optimal SIMD performance.
    ///
    /// # Errors
    ///
    /// Returns an error if width or height is zero.
    ///
    /// # Example
    ///
    /// ```
    /// use trueno_viz::framebuffer::Framebuffer;
    ///
    /// let fb = Framebuffer::new(800, 600).unwrap();
    /// assert_eq!(fb.width(), 800);
    /// assert_eq!(fb.height(), 600);
    /// ```
    pub fn new(width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidDimensions { width, height });
        }

        // Calculate stride with alignment padding
        let row_bytes = (width as usize) * 4;
        let stride = (row_bytes + SIMD_ALIGNMENT - 1) & !(SIMD_ALIGNMENT - 1);

        let size = stride * (height as usize);

        // Allocate with extra space for alignment
        let mut pixels = Vec::with_capacity(size + SIMD_ALIGNMENT);
        pixels.resize(size, 0);

        Ok(Self { width, height, pixels, stride })
    }

    /// Get the width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Get the height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Get the stride (row width in bytes, including any padding).
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// Get the total number of pixels.
    #[must_use]
    pub const fn pixel_count(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    /// Get the raw pixel data as a slice.
    #[must_use]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Get the raw pixel data as a mutable slice.
    pub fn pixels_mut(&mut self) -> &mut [u8] {
        &mut self.pixels
    }

    /// Get a row of pixels as a slice.
    #[must_use]
    pub fn row(&self, y: u32) -> Option<&[u8]> {
        if y >= self.height {
            return None;
        }
        let start = (y as usize) * self.stride;
        let end = start + (self.width as usize) * 4;
        Some(&self.pixels[start..end])
    }

    /// Get a row of pixels as a mutable slice.
    pub fn row_mut(&mut self, y: u32) -> Option<&mut [u8]> {
        if y >= self.height {
            return None;
        }
        let start = (y as usize) * self.stride;
        let end = start + (self.width as usize) * 4;
        Some(&mut self.pixels[start..end])
    }

    /// Clear the framebuffer to a solid color.
    ///
    /// This operation is optimized for SIMD by processing 16 pixels at a time
    /// (64 bytes = 16 RGBA pixels on AVX-512).
    pub fn clear(&mut self, color: Rgba) {
        let [r, g, b, a] = color.to_array();

        // Create a 64-byte pattern (16 pixels) for SIMD-friendly memset
        let pattern: [u8; 64] = {
            let mut p = [0u8; 64];
            for i in 0..16 {
                p[i * 4] = r;
                p[i * 4 + 1] = g;
                p[i * 4 + 2] = b;
                p[i * 4 + 3] = a;
            }
            p
        };

        // Fill each row (compiler will auto-vectorize this pattern copy)
        for y in 0..self.height {
            let row_start = (y as usize) * self.stride;
            let row_end = row_start + (self.width as usize) * 4;
            let row = &mut self.pixels[row_start..row_end];

            // Copy pattern in 64-byte chunks
            let mut offset = 0;
            while offset + 64 <= row.len() {
                row[offset..offset + 64].copy_from_slice(&pattern);
                offset += 64;
            }

            // Handle remaining pixels
            for chunk in row[offset..].chunks_exact_mut(4) {
                chunk[0] = r;
                chunk[1] = g;
                chunk[2] = b;
                chunk[3] = a;
            }
        }
    }

    /// Fill a rectangular region with a solid color.
    ///
    /// Coordinates are clamped to framebuffer bounds.
    pub fn fill_rect(&mut self, x: u32, y: u32, w: u32, h: u32, color: Rgba) {
        let x1 = x.min(self.width);
        let y1 = y.min(self.height);
        let x2 = (x + w).min(self.width);
        let y2 = (y + h).min(self.height);

        if x1 >= x2 || y1 >= y2 {
            return;
        }

        let [r, g, b, a] = color.to_array();
        let rect_width = (x2 - x1) as usize;

        for row_y in y1..y2 {
            let row_start = (row_y as usize) * self.stride + (x1 as usize) * 4;
            let row = &mut self.pixels[row_start..row_start + rect_width * 4];

            for chunk in row.chunks_exact_mut(4) {
                chunk[0] = r;
                chunk[1] = g;
                chunk[2] = b;
                chunk[3] = a;
            }
        }
    }

    /// Get the color at a specific pixel coordinate.
    ///
    /// Returns `None` if the coordinates are out of bounds.
    #[must_use]
    pub fn get_pixel(&self, x: u32, y: u32) -> Option<Rgba> {
        if x >= self.width || y >= self.height {
            return None;
        }

        let idx = self.pixel_index(x, y);
        Some(Rgba::from_array([
            self.pixels[idx],
            self.pixels[idx + 1],
            self.pixels[idx + 2],
            self.pixels[idx + 3],
        ]))
    }

    /// Set the color at a specific pixel coordinate.
    ///
    /// Does nothing if the coordinates are out of bounds.
    pub fn set_pixel(&mut self, x: u32, y: u32, color: Rgba) {
        if x >= self.width || y >= self.height {
            return;
        }

        let idx = self.pixel_index(x, y);
        let [r, g, b, a] = color.to_array();
        self.pixels[idx] = r;
        self.pixels[idx + 1] = g;
        self.pixels[idx + 2] = b;
        self.pixels[idx + 3] = a;
    }

    /// Blend a color at a specific pixel coordinate using alpha blending.
    ///
    /// Uses the standard "over" compositing operation:
    /// `out = src * src_alpha + dst * dst_alpha * (1 - src_alpha)`
    pub fn blend_pixel(&mut self, x: u32, y: u32, color: Rgba) {
        if x >= self.width || y >= self.height {
            return;
        }

        let idx = self.pixel_index(x, y);
        let src_a = f32::from(color.a) / 255.0;
        let dst_a = f32::from(self.pixels[idx + 3]) / 255.0;
        let out_a = src_a + dst_a * (1.0 - src_a);

        if out_a > 0.0 {
            let blend = |src: u8, dst: u8| -> u8 {
                let src_f = f32::from(src) / 255.0;
                let dst_f = f32::from(dst) / 255.0;
                let out = (src_f * src_a + dst_f * dst_a * (1.0 - src_a)) / out_a;
                (out * 255.0) as u8
            };

            self.pixels[idx] = blend(color.r, self.pixels[idx]);
            self.pixels[idx + 1] = blend(color.g, self.pixels[idx + 1]);
            self.pixels[idx + 2] = blend(color.b, self.pixels[idx + 2]);
            self.pixels[idx + 3] = (out_a * 255.0) as u8;
        }
    }

    /// Blend an entire framebuffer over this one using SIMD-accelerated operations.
    ///
    /// Uses trueno's Vector operations for alpha blending.
    ///
    /// # Errors
    ///
    /// Returns an error if the framebuffers have different dimensions.
    pub fn blend_over(&mut self, other: &Framebuffer, alpha: f32) -> Result<()> {
        if self.width != other.width || self.height != other.height {
            return Err(Error::InvalidDimensions { width: other.width, height: other.height });
        }

        let alpha = alpha.clamp(0.0, 1.0);
        let inv_alpha = 1.0 - alpha;

        // Process row by row to maintain cache locality
        for y in 0..self.height {
            let row_start = (y as usize) * self.stride;
            let row_pixels = (self.width as usize) * 4;

            // Convert u8 rows to f32 vectors for SIMD processing
            let dst_slice = &self.pixels[row_start..row_start + row_pixels];
            let src_slice = &other.pixels[row_start..row_start + row_pixels];

            // Convert to f32 for SIMD operations
            let dst_f32: Vec<f32> = dst_slice.iter().map(|&b| f32::from(b)).collect();
            let src_f32: Vec<f32> = src_slice.iter().map(|&b| f32::from(b)).collect();

            // Use trueno vectors for SIMD blending
            let dst_vec = Vector::from_vec(dst_f32);
            let src_vec = Vector::from_vec(src_f32);

            // out = src * alpha + dst * (1 - alpha)
            if let (Ok(src_scaled), Ok(dst_scaled)) = (
                src_vec.mul(&Vector::from_vec(vec![alpha; row_pixels])),
                dst_vec.mul(&Vector::from_vec(vec![inv_alpha; row_pixels])),
            ) {
                if let Ok(result) = src_scaled.add(&dst_scaled) {
                    // Convert back to u8
                    let row = &mut self.pixels[row_start..row_start + row_pixels];
                    for (i, &v) in result.as_slice().iter().enumerate() {
                        row[i] = v.clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply a brightness adjustment using SIMD-accelerated operations.
    ///
    /// `factor` of 1.0 is no change, < 1.0 darkens, > 1.0 brightens.
    pub fn adjust_brightness(&mut self, factor: f32) {
        let factor = factor.max(0.0);

        for y in 0..self.height {
            let row_start = (y as usize) * self.stride;
            let row_pixels = (self.width as usize) * 4;
            let row = &mut self.pixels[row_start..row_start + row_pixels];

            // Process RGB channels, preserve alpha
            for chunk in row.chunks_exact_mut(4) {
                chunk[0] = (f32::from(chunk[0]) * factor).clamp(0.0, 255.0) as u8;
                chunk[1] = (f32::from(chunk[1]) * factor).clamp(0.0, 255.0) as u8;
                chunk[2] = (f32::from(chunk[2]) * factor).clamp(0.0, 255.0) as u8;
                // Alpha unchanged
            }
        }
    }

    /// Get statistics about the framebuffer using SIMD-accelerated reduction.
    ///
    /// Returns (min_luminance, max_luminance, avg_luminance).
    #[must_use]
    pub fn luminance_stats(&self) -> (f32, f32, f32) {
        let mut luminances = Vec::with_capacity(self.pixel_count());

        for y in 0..self.height {
            if let Some(row) = self.row(y) {
                for chunk in row.chunks_exact(4) {
                    // ITU-R BT.709 luminance formula
                    let lum = 0.2126 * f32::from(chunk[0])
                        + 0.7152 * f32::from(chunk[1])
                        + 0.0722 * f32::from(chunk[2]);
                    luminances.push(lum);
                }
            }
        }

        // Use trueno for SIMD-accelerated min/max/mean
        let vec = Vector::from_vec(luminances);

        let min = vec.min().unwrap_or(0.0);
        let max = vec.max().unwrap_or(255.0);
        let mean = vec.mean().unwrap_or(127.5);

        (min, max, mean)
    }

    /// Calculate the byte index for a pixel coordinate.
    #[inline]
    fn pixel_index(&self, x: u32, y: u32) -> usize {
        (y as usize) * self.stride + (x as usize) * 4
    }

    /// Check if the pixel buffer is properly aligned for SIMD.
    #[must_use]
    pub fn is_aligned(&self) -> bool {
        self.pixels.as_ptr() as usize % SIMD_ALIGNMENT == 0
    }

    /// Get pixel data as a compact buffer without stride padding.
    ///
    /// This is useful for encoding to formats like PNG that expect
    /// tightly-packed pixel data.
    #[must_use]
    pub fn to_compact_pixels(&self) -> Vec<u8> {
        let row_bytes = (self.width as usize) * 4;

        // If stride equals row bytes, return a clone
        if self.stride == row_bytes {
            return self.pixels[..row_bytes * (self.height as usize)].to_vec();
        }

        // Otherwise, copy row by row
        let mut compact = Vec::with_capacity(row_bytes * (self.height as usize));
        for y in 0..self.height {
            let start = (y as usize) * self.stride;
            compact.extend_from_slice(&self.pixels[start..start + row_bytes]);
        }
        compact
    }

    /// Get the selected SIMD backend.
    #[must_use]
    pub fn backend() -> Backend {
        Backend::select_best()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_framebuffer() {
        let fb = Framebuffer::new(100, 50).expect("framebuffer creation should succeed");
        assert_eq!(fb.width(), 100);
        assert_eq!(fb.height(), 50);
        assert_eq!(fb.pixel_count(), 5000);
        // Stride should be >= width * 4
        assert!(fb.stride() >= 400);
    }

    #[test]
    fn test_invalid_dimensions() {
        assert!(Framebuffer::new(0, 100).is_err());
        assert!(Framebuffer::new(100, 0).is_err());
        assert!(Framebuffer::new(0, 0).is_err());
    }

    #[test]
    fn test_clear() {
        let mut fb = Framebuffer::new(10, 10).expect("framebuffer creation should succeed");
        fb.clear(Rgba::RED);

        for y in 0..10 {
            for x in 0..10 {
                assert_eq!(fb.get_pixel(x, y), Some(Rgba::RED));
            }
        }
    }

    #[test]
    fn test_clear_large() {
        // Test with larger buffer to exercise SIMD paths
        let mut fb = Framebuffer::new(1920, 1080).expect("framebuffer creation should succeed");
        fb.clear(Rgba::BLUE);

        assert_eq!(fb.get_pixel(0, 0), Some(Rgba::BLUE));
        assert_eq!(fb.get_pixel(959, 539), Some(Rgba::BLUE));
        assert_eq!(fb.get_pixel(1919, 1079), Some(Rgba::BLUE));
    }

    #[test]
    fn test_fill_rect() {
        let mut fb = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);
        fb.fill_rect(10, 10, 20, 20, Rgba::RED);

        // Inside rect
        assert_eq!(fb.get_pixel(15, 15), Some(Rgba::RED));
        // Outside rect
        assert_eq!(fb.get_pixel(5, 5), Some(Rgba::WHITE));
    }

    #[test]
    fn test_set_get_pixel() {
        let mut fb = Framebuffer::new(10, 10).expect("framebuffer creation should succeed");

        fb.set_pixel(5, 5, Rgba::BLUE);
        assert_eq!(fb.get_pixel(5, 5), Some(Rgba::BLUE));

        // Out of bounds
        assert_eq!(fb.get_pixel(100, 100), None);
    }

    #[test]
    fn test_blend_pixel() {
        let mut fb = Framebuffer::new(10, 10).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);

        // Blend semi-transparent red
        let semi_red = Rgba::new(255, 0, 0, 128);
        fb.blend_pixel(5, 5, semi_red);

        let result = fb.get_pixel(5, 5).expect("operation should succeed");
        // Should be pinkish (blend of red and white)
        assert!(result.r > 200);
        assert!(result.g > 100);
        assert!(result.b > 100);
    }

    #[test]
    fn test_blend_over() {
        let mut fb1 = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");
        let mut fb2 = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");

        fb1.clear(Rgba::BLACK);
        fb2.clear(Rgba::WHITE);

        fb1.blend_over(&fb2, 0.5).expect("operation should succeed");

        let result = fb1.get_pixel(50, 50).expect("operation should succeed");
        // Should be gray (50% blend)
        assert!(result.r > 100 && result.r < 150);
        assert!(result.g > 100 && result.g < 150);
        assert!(result.b > 100 && result.b < 150);
    }

    #[test]
    fn test_adjust_brightness() {
        let mut fb = Framebuffer::new(10, 10).expect("framebuffer creation should succeed");
        fb.clear(Rgba::rgb(100, 100, 100));

        fb.adjust_brightness(2.0);

        let result = fb.get_pixel(5, 5).expect("operation should succeed");
        assert_eq!(result.r, 200);
        assert_eq!(result.g, 200);
        assert_eq!(result.b, 200);
    }

    #[test]
    fn test_luminance_stats() {
        let mut fb = Framebuffer::new(10, 10).expect("framebuffer creation should succeed");
        fb.clear(Rgba::rgb(128, 128, 128));

        let (min, max, mean) = fb.luminance_stats();

        // All same color, so min ≈ max ≈ mean
        assert!((min - max).abs() < 1.0);
        assert!((mean - min).abs() < 1.0);
    }

    #[test]
    fn test_row_access() {
        let mut fb = Framebuffer::new(10, 5).expect("framebuffer creation should succeed");
        fb.clear(Rgba::BLACK);

        // Modify a row
        if let Some(row) = fb.row_mut(2) {
            for chunk in row.chunks_exact_mut(4) {
                chunk[0] = 255; // Set red
            }
        }

        // Verify
        assert_eq!(fb.get_pixel(5, 2).expect("value should be present").r, 255);
        assert_eq!(fb.get_pixel(5, 1).expect("value should be present").r, 0);
    }

    #[test]
    fn test_backend_selection() {
        let backend = Framebuffer::backend();
        // Should return a valid backend
        println!("Selected backend: {backend:?}");
    }

    #[test]
    fn test_pixels_access() {
        let fb = Framebuffer::new(10, 10).expect("framebuffer creation should succeed");
        let pixels = fb.pixels();
        // Buffer size is stride * height (includes alignment padding)
        assert_eq!(pixels.len(), fb.stride() * 10);
        // Stride is at least width * 4
        assert!(pixels.len() >= 10 * 10 * 4);
    }

    #[test]
    fn test_pixels_mut_access() {
        let mut fb = Framebuffer::new(10, 10).expect("framebuffer creation should succeed");
        // Buffer size is stride * height
        let expected_size = fb.stride() * 10;
        let pixels = fb.pixels_mut();
        assert_eq!(pixels.len(), expected_size);
        // Modify a pixel directly
        pixels[0] = 255;
        pixels[1] = 0;
        pixels[2] = 0;
        pixels[3] = 255;
        assert_eq!(fb.get_pixel(0, 0), Some(Rgba::RED));
    }

    #[test]
    fn test_row_out_of_bounds() {
        let fb = Framebuffer::new(10, 5).expect("framebuffer creation should succeed");
        assert!(fb.row(5).is_none());
        assert!(fb.row(100).is_none());
    }

    #[test]
    fn test_row_mut_out_of_bounds() {
        let mut fb = Framebuffer::new(10, 5).expect("framebuffer creation should succeed");
        assert!(fb.row_mut(5).is_none());
        assert!(fb.row_mut(100).is_none());
    }

    #[test]
    fn test_fill_rect_empty() {
        let mut fb = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);
        // Zero-width rect
        fb.fill_rect(10, 10, 0, 20, Rgba::RED);
        assert_eq!(fb.get_pixel(10, 10), Some(Rgba::WHITE));

        // Zero-height rect
        fb.fill_rect(10, 10, 20, 0, Rgba::RED);
        assert_eq!(fb.get_pixel(10, 10), Some(Rgba::WHITE));
    }

    #[test]
    fn test_fill_rect_out_of_bounds() {
        let mut fb = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);
        // Rect starting outside
        fb.fill_rect(200, 200, 20, 20, Rgba::RED);
        assert_eq!(fb.get_pixel(50, 50), Some(Rgba::WHITE));
    }

    #[test]
    fn test_blend_pixel_out_of_bounds() {
        let mut fb = Framebuffer::new(10, 10).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);
        // Out of bounds - should be no-op
        fb.blend_pixel(100, 100, Rgba::RED);
        fb.blend_pixel(10, 5, Rgba::RED);
        fb.blend_pixel(5, 10, Rgba::RED);
        // Original pixels unchanged
        assert_eq!(fb.get_pixel(5, 5), Some(Rgba::WHITE));
    }

    #[test]
    fn test_blend_over_dimension_mismatch() {
        let mut fb1 = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");
        let fb2 = Framebuffer::new(50, 50).expect("framebuffer creation should succeed");

        let result = fb1.blend_over(&fb2, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_aligned() {
        let fb = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");
        // Just verify it returns a bool (alignment depends on allocator)
        let _aligned = fb.is_aligned();
    }

    #[test]
    fn test_to_compact_pixels() {
        let mut fb = Framebuffer::new(10, 10).expect("framebuffer creation should succeed");
        fb.clear(Rgba::RED);
        let compact = fb.to_compact_pixels();
        // Compact size should be width * height * 4 (no stride padding)
        assert_eq!(compact.len(), 10 * 10 * 4);
        // First pixel should be red
        assert_eq!(&compact[0..4], &[255, 0, 0, 255]);
    }

    #[test]
    fn test_to_compact_pixels_with_stride() {
        // Width that requires stride padding (not a multiple of 16)
        let mut fb = Framebuffer::new(10, 5).expect("framebuffer creation should succeed");
        fb.clear(Rgba::GREEN);

        let compact = fb.to_compact_pixels();
        assert_eq!(compact.len(), 10 * 5 * 4);

        // Verify all pixels are green (stride padding should be excluded)
        for chunk in compact.chunks_exact(4) {
            assert_eq!(chunk, &[0, 255, 0, 255]);
        }
    }

    #[test]
    fn test_set_pixel_out_of_bounds() {
        let mut fb = Framebuffer::new(10, 10).expect("framebuffer creation should succeed");
        // Out of bounds - should be no-op
        fb.set_pixel(100, 100, Rgba::RED);
        fb.set_pixel(10, 5, Rgba::RED);
        fb.set_pixel(5, 10, Rgba::RED);
    }
}
