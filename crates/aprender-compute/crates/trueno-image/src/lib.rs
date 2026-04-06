#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::unwrap_used,
        clippy::disallowed_methods,
        clippy::float_cmp,
        clippy::panic
    )
)]
//! GPU image processing primitives.
//!
//! # Contract: image-conv2d-v1.yaml
//!
//! Provides convolution, Gaussian blur, Sobel edge detection, and
//! Canny edge detection with provable properties.
//!
//! # Example
//!
//! ```
//! use trueno_image::{conv2d, BorderMode};
//!
//! // Identity convolution (delta kernel)
//! let image = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0_f32];
//! let delta = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0_f32];
//! let out = conv2d(&image, 3, 3, &delta, 3, 3, BorderMode::Zero).unwrap();
//! assert!((out[4] - 5.0).abs() < 1e-6); // Center pixel preserved
//! ```

mod buf;
mod color;
mod conv;
mod error;
mod histogram;
mod morphology;
mod resize;

#[cfg(test)]
mod tests;

pub use buf::{DType, ImageBuf};
pub use color::{connected_components, hsv_to_rgb, rgb_to_gray, rgb_to_hsv};
pub use conv::{
    canny, canny_rgb, conv2d, gaussian_blur, gradient_magnitude, separable_conv2d, sobel,
    BorderMode,
};
pub use error::ImageError;
pub use histogram::{cumulative_histogram, equalize, histogram};
pub use morphology::{closing, dilate, erode, opening};
pub use resize::{resize, Interpolation};

/// Image operations trait for `ImageBuf` method dispatch.
///
/// Provides a unified interface for applying image processing operations
/// to structured `ImageBuf` instances with automatic dimension handling.
pub trait ImageOps {
    /// Apply 2D convolution with given kernel and border mode.
    fn apply_conv2d(
        &self,
        kernel: &[f32],
        kw: usize,
        kh: usize,
        border: BorderMode,
    ) -> Result<ImageBuf, ImageError>;

    /// Apply Gaussian blur with given sigma.
    fn blur(&self, sigma: f32) -> Result<ImageBuf, ImageError>;

    /// Compute Sobel gradients (gx, gy).
    fn sobel_gradients(&self) -> Result<(ImageBuf, ImageBuf), ImageError>;

    /// Apply Canny edge detection (converts multi-channel to grayscale).
    fn canny_edges(&self, sigma: f32, low: f32, high: f32) -> Result<ImageBuf, ImageError>;

    /// Morphological dilation with structuring element.
    fn apply_dilate(&self, se: &[f32], sw: usize, sh: usize) -> Result<ImageBuf, ImageError>;

    /// Morphological erosion with structuring element.
    fn apply_erode(&self, se: &[f32], sw: usize, sh: usize) -> Result<ImageBuf, ImageError>;

    /// Resize image to new dimensions.
    fn apply_resize(
        &self,
        new_w: usize,
        new_h: usize,
        interp: Interpolation,
    ) -> Result<ImageBuf, ImageError>;

    /// Convert to grayscale.
    fn to_gray(&self) -> Result<ImageBuf, ImageError>;

    /// Convert RGB to HSV color space.
    fn to_hsv(&self) -> Result<ImageBuf, ImageError>;

    /// Compute histogram with given number of bins.
    fn compute_histogram(&self, bins: usize) -> Result<Vec<u32>, ImageError>;

    /// Label connected components in binary image.
    fn label_components(&self) -> Result<(Vec<u32>, u32), ImageError>;
}

impl ImageOps for ImageBuf {
    fn apply_conv2d(
        &self,
        kernel: &[f32],
        kw: usize,
        kh: usize,
        border: BorderMode,
    ) -> Result<ImageBuf, ImageError> {
        if self.channels() == 1 {
            let data = conv2d(self.data(), self.width(), self.height(), kernel, kw, kh, border)?;
            ImageBuf::new(data, self.width(), self.height(), 1)
        } else {
            let npix = self.width() * self.height();
            let mut out = vec![0.0_f32; npix * self.channels()];
            for c in 0..self.channels() {
                let ch = self.channel(c)?;
                let filtered =
                    conv2d(ch.data(), self.width(), self.height(), kernel, kw, kh, border)?;
                for i in 0..npix {
                    out[i * self.channels() + c] = filtered[i];
                }
            }
            ImageBuf::new(out, self.width(), self.height(), self.channels())
        }
    }

    fn blur(&self, sigma: f32) -> Result<ImageBuf, ImageError> {
        if self.channels() == 1 {
            let data = gaussian_blur(self.data(), self.width(), self.height(), sigma)?;
            ImageBuf::new(data, self.width(), self.height(), 1)
        } else {
            let npix = self.width() * self.height();
            let mut out = vec![0.0_f32; npix * self.channels()];
            for c in 0..self.channels() {
                let ch = self.channel(c)?;
                let blurred = gaussian_blur(ch.data(), self.width(), self.height(), sigma)?;
                for i in 0..npix {
                    out[i * self.channels() + c] = blurred[i];
                }
            }
            ImageBuf::new(out, self.width(), self.height(), self.channels())
        }
    }

    fn sobel_gradients(&self) -> Result<(ImageBuf, ImageBuf), ImageError> {
        let gray = if self.channels() == 1 {
            self.data().to_vec()
        } else {
            rgb_to_gray(self.data(), self.width(), self.height())?
        };
        let (gx, gy) = sobel(&gray, self.width(), self.height())?;
        Ok((
            ImageBuf::new(gx, self.width(), self.height(), 1)?,
            ImageBuf::new(gy, self.width(), self.height(), 1)?,
        ))
    }

    fn canny_edges(&self, sigma: f32, low: f32, high: f32) -> Result<ImageBuf, ImageError> {
        let edges =
            canny_rgb(self.data(), self.width(), self.height(), self.channels(), sigma, low, high)?;
        ImageBuf::new(edges, self.width(), self.height(), 1)
    }

    fn apply_dilate(&self, se: &[f32], sw: usize, sh: usize) -> Result<ImageBuf, ImageError> {
        if self.channels() == 1 {
            let data = dilate(self.data(), self.width(), self.height(), se, sw, sh)?;
            ImageBuf::new(data, self.width(), self.height(), 1)
        } else {
            let npix = self.width() * self.height();
            let mut out = vec![0.0_f32; npix * self.channels()];
            for c in 0..self.channels() {
                let ch = self.channel(c)?;
                let d = dilate(ch.data(), self.width(), self.height(), se, sw, sh)?;
                for i in 0..npix {
                    out[i * self.channels() + c] = d[i];
                }
            }
            ImageBuf::new(out, self.width(), self.height(), self.channels())
        }
    }

    fn apply_erode(&self, se: &[f32], sw: usize, sh: usize) -> Result<ImageBuf, ImageError> {
        if self.channels() == 1 {
            let data = erode(self.data(), self.width(), self.height(), se, sw, sh)?;
            ImageBuf::new(data, self.width(), self.height(), 1)
        } else {
            let npix = self.width() * self.height();
            let mut out = vec![0.0_f32; npix * self.channels()];
            for c in 0..self.channels() {
                let ch = self.channel(c)?;
                let e = erode(ch.data(), self.width(), self.height(), se, sw, sh)?;
                for i in 0..npix {
                    out[i * self.channels() + c] = e[i];
                }
            }
            ImageBuf::new(out, self.width(), self.height(), self.channels())
        }
    }

    fn apply_resize(
        &self,
        new_w: usize,
        new_h: usize,
        interp: Interpolation,
    ) -> Result<ImageBuf, ImageError> {
        if self.channels() == 1 {
            let data = resize(self.data(), self.width(), self.height(), new_w, new_h, interp)?;
            ImageBuf::new(data, new_w, new_h, 1)
        } else {
            let new_npix = new_w * new_h;
            let mut out = vec![0.0_f32; new_npix * self.channels()];
            for c in 0..self.channels() {
                let ch = self.channel(c)?;
                let resized = resize(ch.data(), self.width(), self.height(), new_w, new_h, interp)?;
                for i in 0..new_npix {
                    out[i * self.channels() + c] = resized[i];
                }
            }
            ImageBuf::new(out, new_w, new_h, self.channels())
        }
    }

    fn to_gray(&self) -> Result<ImageBuf, ImageError> {
        if self.channels() == 1 {
            return Ok(self.clone());
        }
        let gray = rgb_to_gray(self.data(), self.width(), self.height())?;
        ImageBuf::new(gray, self.width(), self.height(), 1)
    }

    fn to_hsv(&self) -> Result<ImageBuf, ImageError> {
        let hsv = rgb_to_hsv(self.data(), self.width(), self.height())?;
        ImageBuf::new(hsv, self.width(), self.height(), self.channels())
    }

    fn compute_histogram(&self, bins: usize) -> Result<Vec<u32>, ImageError> {
        let gray = if self.channels() == 1 {
            self.data().to_vec()
        } else {
            rgb_to_gray(self.data(), self.width(), self.height())?
        };
        histogram(&gray, self.width(), self.height(), bins)
    }

    fn label_components(&self) -> Result<(Vec<u32>, u32), ImageError> {
        let gray = if self.channels() == 1 {
            self.data().to_vec()
        } else {
            rgb_to_gray(self.data(), self.width(), self.height())?
        };
        let labels = connected_components(&gray, self.width(), self.height())?;
        let max_label = labels.iter().copied().max().unwrap_or(0);
        Ok((labels, max_label))
    }
}
