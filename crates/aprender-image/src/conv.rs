//! 2D convolution with border handling.
//!
//! # Contract: image-conv2d-v1.yaml
//!
//! O(i,j) = Σ_p Σ_q I(i+p, j+q) · K(p, q)
//!
//! ## Proof obligations
//! - Output dimensions match input (same-padding)
//! - Separable kernel produces same result as 2D convolution
//! - Identity kernel preserves input

use crate::error::ImageError;

/// Border handling mode for convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BorderMode {
    /// Zero-pad outside image bounds.
    #[default]
    Zero,
    /// Clamp to edge pixel values.
    Clamp,
    /// Reflect at image boundary.
    Reflect,
    /// Wrap (periodic boundary).
    Wrap,
}

/// 2D convolution with same-padding.
///
/// # Contract: image-conv2d-v1.yaml / conv2d
///
/// Kernel is row-major, dimensions kw×kh (both must be odd).
/// Output has same dimensions as input.
///
/// # Errors
///
/// Returns error on invalid dimensions or buffer mismatch.
pub fn conv2d(
    image: &[f32],
    width: usize,
    height: usize,
    kernel: &[f32],
    kw: usize,
    kh: usize,
    border: BorderMode,
) -> Result<Vec<f32>, ImageError> {
    validate_conv_args(image, width, height, kernel, kw, kh)?;

    let half_kw = kw / 2;
    let half_kh = kh / 2;
    let mut output = vec![0.0f32; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f64;
            for ky in 0..kh {
                for kx in 0..kw {
                    let iy = y as isize + ky as isize - half_kh as isize;
                    let ix = x as isize + kx as isize - half_kw as isize;

                    let pixel = sample_border(image, width, height, ix, iy, border);
                    sum += f64::from(pixel) * f64::from(kernel[ky * kw + kx]);
                }
            }
            output[y * width + x] = sum as f32;
        }
    }

    Ok(output)
}

/// Separable 2D convolution: apply horizontal then vertical 1D kernels.
///
/// Equivalent to conv2d(image, outer_product(v, h)) but O(kw + kh) per pixel
/// instead of O(kw * kh).
///
/// # Errors
///
/// Returns error on invalid dimensions.
pub fn separable_conv2d(
    image: &[f32],
    width: usize,
    height: usize,
    h_kernel: &[f32],
    v_kernel: &[f32],
    border: BorderMode,
) -> Result<Vec<f32>, ImageError> {
    if width == 0 || height == 0 {
        return Err(ImageError::ZeroDimension { width, height });
    }
    if image.len() != width * height {
        return Err(ImageError::BufferLengthMismatch {
            expected: width * height,
            got: image.len(),
            width,
            height,
        });
    }

    // Horizontal pass
    let hk = h_kernel.len();
    let half_hk = hk / 2;
    let mut temp = vec![0.0f32; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f64;
            for k in 0..hk {
                let ix = x as isize + k as isize - half_hk as isize;
                let pixel = sample_border(image, width, height, ix, y as isize, border);
                sum += f64::from(pixel) * f64::from(h_kernel[k]);
            }
            temp[y * width + x] = sum as f32;
        }
    }

    // Vertical pass
    let vk = v_kernel.len();
    let half_vk = vk / 2;
    let mut output = vec![0.0f32; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f64;
            for k in 0..vk {
                let iy = y as isize + k as isize - half_vk as isize;
                let pixel = sample_border(&temp, width, height, x as isize, iy, border);
                sum += f64::from(pixel) * f64::from(v_kernel[k]);
            }
            output[y * width + x] = sum as f32;
        }
    }

    Ok(output)
}

/// Gaussian blur using separable convolution.
///
/// # Errors
///
/// Returns error on invalid dimensions.
pub fn gaussian_blur(
    image: &[f32],
    width: usize,
    height: usize,
    sigma: f32,
) -> Result<Vec<f32>, ImageError> {
    let kernel = gaussian_kernel_1d(sigma);
    separable_conv2d(image, width, height, &kernel, &kernel, BorderMode::Clamp)
}

/// Generate a 1D Gaussian kernel with the given sigma.
///
/// Kernel size = ceil(6*sigma) | 1 (minimum 3, always odd).
#[allow(clippy::cast_precision_loss)]
fn gaussian_kernel_1d(sigma: f32) -> Vec<f32> {
    let radius = ((3.0 * sigma).ceil() as usize).max(1);
    let size = 2 * radius + 1;
    let sigma_sq = f64::from(sigma) * f64::from(sigma);

    let mut kernel = vec![0.0f32; size];
    let mut sum = 0.0f64;

    for i in 0..size {
        let x = i as f64 - radius as f64;
        let v = (-x * x / (2.0 * sigma_sq)).exp();
        kernel[i] = v as f32;
        sum += v;
    }

    // Normalize
    for k in &mut kernel {
        *k = (*k as f64 / sum) as f32;
    }

    kernel
}

/// Sobel edge detection: returns (gradient_x, gradient_y).
///
/// # Errors
///
/// Returns error on invalid dimensions.
pub fn sobel(
    image: &[f32],
    width: usize,
    height: usize,
) -> Result<(Vec<f32>, Vec<f32>), ImageError> {
    if width == 0 || height == 0 {
        return Err(ImageError::ZeroDimension { width, height });
    }
    if image.len() != width * height {
        return Err(ImageError::BufferLengthMismatch {
            expected: width * height,
            got: image.len(),
            width,
            height,
        });
    }

    // Sobel kernels
    let sx = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0_f32];
    let sy = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0_f32];

    let gx = conv2d(image, width, height, &sx, 3, 3, BorderMode::Zero)?;
    let gy = conv2d(image, width, height, &sy, 3, 3, BorderMode::Zero)?;

    Ok((gx, gy))
}

/// Gradient magnitude from Sobel output.
pub fn gradient_magnitude(gx: &[f32], gy: &[f32]) -> Vec<f32> {
    gx.iter().zip(gy.iter()).map(|(&x, &y)| (x * x + y * y).sqrt()).collect()
}

/// Canny edge detection.
///
/// # Contract: image-conv2d-v1.yaml / canny
///
/// Steps: Gaussian blur → Sobel gradients → NMS → hysteresis thresholding.
///
/// # Errors
///
/// Returns error on invalid dimensions or thresholds.
pub fn canny(
    image: &[f32],
    width: usize,
    height: usize,
    sigma: f32,
    low_threshold: f32,
    high_threshold: f32,
) -> Result<Vec<f32>, ImageError> {
    if low_threshold < 0.0 || high_threshold < low_threshold || high_threshold > 1.0 {
        return Err(ImageError::InvalidThresholds { low: low_threshold, high: high_threshold });
    }

    // Step 1: Gaussian blur
    let blurred = gaussian_blur(image, width, height, sigma)?;

    // Step 2: Sobel gradients
    let (gx, gy) = sobel(&blurred, width, height)?;
    let mag = gradient_magnitude(&gx, &gy);

    // Normalize magnitude to [0, 1]
    let max_mag = mag.iter().copied().fold(0.0f32, f32::max);
    let mag_norm: Vec<f32> =
        if max_mag > 0.0 { mag.iter().map(|&m| m / max_mag).collect() } else { mag };

    // Step 3: Non-maximum suppression
    let nms = non_maximum_suppression(&mag_norm, &gx, &gy, width, height);

    // Step 4: Hysteresis thresholding
    Ok(hysteresis_threshold(&nms, width, height, low_threshold, high_threshold))
}

/// Non-maximum suppression: thin edges to 1-pixel width.
fn non_maximum_suppression(
    mag: &[f32],
    gx: &[f32],
    gy: &[f32],
    width: usize,
    height: usize,
) -> Vec<f32> {
    let mut nms = vec![0.0f32; width * height];
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let idx = y * width + x;
            let angle = gy[idx].atan2(gx[idx]);
            let m = mag[idx];

            let dir =
                ((angle + std::f32::consts::PI) / std::f32::consts::FRAC_PI_4).round() as usize % 4;
            let (n1, n2) = match dir {
                0 => (mag[idx - 1], mag[idx + 1]),
                1 => (mag[(y - 1) * width + x + 1], mag[(y + 1) * width + x - 1]),
                2 => (mag[(y - 1) * width + x], mag[(y + 1) * width + x]),
                _ => (mag[(y - 1) * width + x - 1], mag[(y + 1) * width + x + 1]),
            };

            if m >= n1 && m >= n2 {
                nms[idx] = m;
            }
        }
    }
    nms
}

/// Hysteresis thresholding: connect weak edges to strong edges.
fn hysteresis_threshold(nms: &[f32], width: usize, height: usize, low: f32, high: f32) -> Vec<f32> {
    let mut edges = vec![0.0f32; width * height];
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let idx = y * width + x;
            if nms[idx] >= high {
                edges[idx] = 1.0;
            } else if nms[idx] >= low {
                let has_strong = [
                    (y - 1, x - 1),
                    (y - 1, x),
                    (y - 1, x + 1),
                    (y, x - 1),
                    (y, x + 1),
                    (y + 1, x - 1),
                    (y + 1, x),
                    (y + 1, x + 1),
                ]
                .iter()
                .any(|&(ny, nx)| nms[ny * width + nx] >= high);

                if has_strong {
                    edges[idx] = 1.0;
                }
            }
        }
    }
    edges
}

fn sample_border(
    image: &[f32],
    width: usize,
    height: usize,
    x: isize,
    y: isize,
    border: BorderMode,
) -> f32 {
    match border {
        BorderMode::Zero => {
            if x < 0 || y < 0 || x >= width as isize || y >= height as isize {
                0.0
            } else {
                image[y as usize * width + x as usize]
            }
        }
        BorderMode::Clamp => {
            let cx = x.clamp(0, width as isize - 1) as usize;
            let cy = y.clamp(0, height as isize - 1) as usize;
            image[cy * width + cx]
        }
        BorderMode::Reflect => {
            let rx = reflect(x, width);
            let ry = reflect(y, height);
            image[ry * width + rx]
        }
        BorderMode::Wrap => {
            let wx = wrap(x, width);
            let wy = wrap(y, height);
            image[wy * width + wx]
        }
    }
}

fn wrap(i: isize, size: usize) -> usize {
    let s = size as isize;
    ((i % s + s) % s) as usize
}

fn reflect(i: isize, size: usize) -> usize {
    if i < 0 {
        (-i - 1).min(size as isize - 1) as usize
    } else if i >= size as isize {
        (2 * size as isize - i - 1).max(0) as usize
    } else {
        i as usize
    }
}

fn validate_conv_args(
    image: &[f32],
    width: usize,
    height: usize,
    kernel: &[f32],
    kw: usize,
    kh: usize,
) -> Result<(), ImageError> {
    if width == 0 || height == 0 {
        return Err(ImageError::ZeroDimension { width, height });
    }
    if kw == 0 || kh == 0 || kw.is_multiple_of(2) || kh.is_multiple_of(2) {
        return Err(ImageError::InvalidKernelSize { kw, kh });
    }
    if image.len() != width * height {
        return Err(ImageError::BufferLengthMismatch {
            expected: width * height,
            got: image.len(),
            width,
            height,
        });
    }
    if kernel.len() != kw * kh {
        return Err(ImageError::InvalidKernelSize { kw, kh });
    }
    Ok(())
}

/// Multi-channel Canny edge detection (NPP 3-channel parity).
///
/// Converts RGB input to grayscale using BT.601 weights, then applies
/// the standard Canny pipeline (Gaussian blur → Sobel → NMS → hysteresis).
///
/// Input is `width * height * channels` row-major interleaved pixels.
/// Output is `width * height` binary edge map (single channel).
///
/// # Errors
///
/// Returns error on dimension mismatch or invalid thresholds.
pub fn canny_rgb(
    image: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    sigma: f32,
    low_threshold: f32,
    high_threshold: f32,
) -> Result<Vec<f32>, ImageError> {
    let expected = width * height * channels;
    if image.len() != expected {
        return Err(ImageError::BufferLengthMismatch { expected, got: image.len(), width, height });
    }

    // Convert to grayscale (BT.601 weights: 0.299R + 0.587G + 0.114B)
    let gray = if channels == 1 {
        image.to_vec()
    } else {
        let mut g = Vec::with_capacity(width * height);
        for i in 0..width * height {
            let base = i * channels;
            let r = image[base];
            let green = if channels > 1 { image[base + 1] } else { 0.0 };
            let b = if channels > 2 { image[base + 2] } else { 0.0 };
            g.push(0.299 * r + 0.587 * green + 0.114 * b);
        }
        g
    };

    canny(&gray, width, height, sigma, low_threshold, high_threshold)
}
