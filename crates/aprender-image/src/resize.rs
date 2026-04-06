//! Image resize: bilinear and nearest-neighbor interpolation (NPP parity).

use crate::error::ImageError;

/// Interpolation method for resize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    /// Nearest-neighbor (fastest, blocky).
    Nearest,
    /// Bilinear interpolation (smooth).
    Bilinear,
    /// Bicubic interpolation (sharper than bilinear, slower).
    Bicubic,
    /// Lanczos interpolation with a=3 (highest quality, slowest).
    Lanczos,
}

/// Resize a grayscale image.
///
/// # Errors
///
/// Returns error if dimensions don't match or output size is zero.
pub fn resize(
    image: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    method: Interpolation,
) -> Result<Vec<f32>, ImageError> {
    if image.len() != src_w * src_h {
        return Err(ImageError::BufferLengthMismatch {
            expected: src_w * src_h,
            got: image.len(),
            width: src_w,
            height: src_h,
        });
    }
    if dst_w == 0 || dst_h == 0 {
        return Err(ImageError::ZeroDimension { width: dst_w, height: dst_h });
    }

    let mut output = vec![0.0f32; dst_w * dst_h];

    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = src_h as f32 / dst_h as f32;

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = (dx as f32 + 0.5) * scale_x - 0.5;
            let sy = (dy as f32 + 0.5) * scale_y - 0.5;

            output[dy * dst_w + dx] = match method {
                Interpolation::Nearest => {
                    let ix = (sx + 0.5) as usize;
                    let iy = (sy + 0.5) as usize;
                    let ix = ix.min(src_w - 1);
                    let iy = iy.min(src_h - 1);
                    image[iy * src_w + ix]
                }
                Interpolation::Bilinear => bilinear_sample(image, src_w, src_h, sx, sy),
                Interpolation::Bicubic => bicubic_sample(image, src_w, src_h, sx, sy),
                Interpolation::Lanczos => lanczos_sample(image, src_w, src_h, sx, sy),
            };
        }
    }

    Ok(output)
}

/// Clamp index to [0, size-1].
#[inline]
fn clamp_idx(i: isize, size: usize) -> usize {
    i.clamp(0, size as isize - 1) as usize
}

/// Bilinear interpolation at fractional coordinates.
fn bilinear_sample(image: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x0 = (x.floor() as isize).max(0) as usize;
    let y0 = (y.floor() as isize).max(0) as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);

    let fx = (x - x0 as f32).clamp(0.0, 1.0);
    let fy = (y - y0 as f32).clamp(0.0, 1.0);

    let p00 = image[y0 * w + x0];
    let p10 = image[y0 * w + x1];
    let p01 = image[y1 * w + x0];
    let p11 = image[y1 * w + x1];

    p00 * (1.0 - fx) * (1.0 - fy) + p10 * fx * (1.0 - fy) + p01 * (1.0 - fx) * fy + p11 * fx * fy
}

/// Cubic interpolation weight (Keys' convolution, a = -0.5).
#[inline]
fn cubic_weight(t: f32) -> f32 {
    let t = t.abs();
    if t <= 1.0 {
        (1.5 * t - 2.5) * t * t + 1.0
    } else if t < 2.0 {
        ((-0.5 * t + 2.5) * t - 4.0) * t + 2.0
    } else {
        0.0
    }
}

/// Bicubic interpolation at fractional coordinates (4×4 neighborhood).
fn bicubic_sample(image: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let ix = x.floor() as isize;
    let iy = y.floor() as isize;
    let fx = x - ix as f32;
    let fy = y - iy as f32;

    let mut sum = 0.0f64;
    for j in -1..=2_isize {
        let wy = cubic_weight(fy - j as f32) as f64;
        let cy = clamp_idx(iy + j, h);
        for i in -1..=2_isize {
            let wx = cubic_weight(fx - i as f32) as f64;
            let cx = clamp_idx(ix + i, w);
            sum += wy * wx * f64::from(image[cy * w + cx]);
        }
    }
    sum as f32
}

/// Lanczos kernel with a=3.
#[inline]
fn lanczos_weight(t: f32) -> f32 {
    let t = t.abs();
    if t < 1e-7 {
        1.0
    } else if t < 3.0 {
        let pi_t = std::f32::consts::PI * t;
        let pi_t_over_a = pi_t / 3.0;
        (pi_t.sin() * pi_t_over_a.sin()) / (pi_t * pi_t_over_a)
    } else {
        0.0
    }
}

/// Lanczos interpolation at fractional coordinates (6×6 neighborhood, a=3).
fn lanczos_sample(image: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let ix = x.floor() as isize;
    let iy = y.floor() as isize;
    let fx = x - ix as f32;
    let fy = y - iy as f32;

    let mut sum = 0.0f64;
    let mut weight_sum = 0.0f64;
    for j in -2..=3_isize {
        let wy = lanczos_weight(fy - j as f32) as f64;
        let cy = clamp_idx(iy + j, h);
        for i in -2..=3_isize {
            let wx = lanczos_weight(fx - i as f32) as f64;
            let cx = clamp_idx(ix + i, w);
            let w_total = wy * wx;
            sum += w_total * f64::from(image[cy * w + cx]);
            weight_sum += w_total;
        }
    }
    if weight_sum.abs() > 1e-12 {
        (sum / weight_sum) as f32
    } else {
        0.0
    }
}
