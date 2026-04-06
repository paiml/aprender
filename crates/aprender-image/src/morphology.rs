//! Morphological operations: dilate and erode (NPP parity).
//!
//! Operates on grayscale images with flat structuring elements.

use crate::error::ImageError;

/// Dilate: output pixel = max of neighborhood defined by structuring element.
///
/// # Errors
///
/// Returns error if dimensions don't match or SE is invalid.
pub fn dilate(
    image: &[f32],
    width: usize,
    height: usize,
    se: &[f32],
    se_w: usize,
    se_h: usize,
) -> Result<Vec<f32>, ImageError> {
    validate_inputs(image, width, height, se, se_w, se_h)?;
    let mut output = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            output[y * width + x] = neighborhood_max(image, width, height, se, se_w, se_h, x, y);
        }
    }
    Ok(output)
}

/// Erode: output pixel = min of neighborhood defined by structuring element.
///
/// # Errors
///
/// Returns error if dimensions don't match or SE is invalid.
pub fn erode(
    image: &[f32],
    width: usize,
    height: usize,
    se: &[f32],
    se_w: usize,
    se_h: usize,
) -> Result<Vec<f32>, ImageError> {
    validate_inputs(image, width, height, se, se_w, se_h)?;
    let mut output = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            output[y * width + x] = neighborhood_min(image, width, height, se, se_w, se_h, x, y);
        }
    }
    Ok(output)
}

/// Compute max over SE neighborhood at pixel (px, py).
#[allow(clippy::too_many_arguments)]
fn neighborhood_max(
    image: &[f32],
    width: usize,
    height: usize,
    se: &[f32],
    se_w: usize,
    se_h: usize,
    px: usize,
    py: usize,
) -> f32 {
    let half_w = se_w / 2;
    let half_h = se_h / 2;
    let mut max_val = f32::NEG_INFINITY;
    for sy in 0..se_h {
        for sx in 0..se_w {
            if se[sy * se_w + sx] <= 0.0 {
                continue;
            }
            let iy = py as isize + sy as isize - half_h as isize;
            let ix = px as isize + sx as isize - half_w as isize;
            if iy >= 0 && iy < height as isize && ix >= 0 && ix < width as isize {
                let val = image[iy as usize * width + ix as usize];
                if val > max_val {
                    max_val = val;
                }
            }
        }
    }
    if max_val == f32::NEG_INFINITY {
        0.0
    } else {
        max_val
    }
}

/// Compute min over SE neighborhood at pixel (px, py).
#[allow(clippy::too_many_arguments)]
fn neighborhood_min(
    image: &[f32],
    width: usize,
    height: usize,
    se: &[f32],
    se_w: usize,
    se_h: usize,
    px: usize,
    py: usize,
) -> f32 {
    let half_w = se_w / 2;
    let half_h = se_h / 2;
    let mut min_val = f32::INFINITY;
    for sy in 0..se_h {
        for sx in 0..se_w {
            if se[sy * se_w + sx] <= 0.0 {
                continue;
            }
            let iy = py as isize + sy as isize - half_h as isize;
            let ix = px as isize + sx as isize - half_w as isize;
            if iy >= 0 && iy < height as isize && ix >= 0 && ix < width as isize {
                let val = image[iy as usize * width + ix as usize];
                if val < min_val {
                    min_val = val;
                }
            }
        }
    }
    if min_val == f32::INFINITY {
        0.0
    } else {
        min_val
    }
}

/// Morphological opening: erode then dilate.
///
/// # Errors
///
/// Returns error if dimensions don't match.
pub fn opening(
    image: &[f32],
    width: usize,
    height: usize,
    se: &[f32],
    se_w: usize,
    se_h: usize,
) -> Result<Vec<f32>, ImageError> {
    let eroded = erode(image, width, height, se, se_w, se_h)?;
    dilate(&eroded, width, height, se, se_w, se_h)
}

/// Morphological closing: dilate then erode.
///
/// # Errors
///
/// Returns error if dimensions don't match.
pub fn closing(
    image: &[f32],
    width: usize,
    height: usize,
    se: &[f32],
    se_w: usize,
    se_h: usize,
) -> Result<Vec<f32>, ImageError> {
    let dilated = dilate(image, width, height, se, se_w, se_h)?;
    erode(&dilated, width, height, se, se_w, se_h)
}

fn validate_inputs(
    image: &[f32],
    width: usize,
    height: usize,
    se: &[f32],
    se_w: usize,
    se_h: usize,
) -> Result<(), ImageError> {
    if image.len() != width * height {
        return Err(ImageError::BufferLengthMismatch {
            expected: width * height,
            got: image.len(),
            width,
            height,
        });
    }
    if se.len() != se_w * se_h || se_w == 0 || se_h == 0 {
        return Err(ImageError::InvalidKernelSize { kw: se_w, kh: se_h });
    }
    Ok(())
}
