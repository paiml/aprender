//! Histogram computation for grayscale images (NPP parity).

use crate::error::ImageError;

/// Compute histogram of a grayscale image with values in [0, 1].
///
/// Returns a histogram with `bins` buckets. Values outside [0, 1] are clamped.
///
/// # Errors
///
/// Returns error if dimensions don't match buffer or bins is zero.
pub fn histogram(
    image: &[f32],
    width: usize,
    height: usize,
    bins: usize,
) -> Result<Vec<u32>, ImageError> {
    if image.len() != width * height {
        return Err(ImageError::BufferLengthMismatch {
            expected: width * height,
            got: image.len(),
            width,
            height,
        });
    }
    if bins == 0 {
        return Err(ImageError::InvalidKernelSize { kw: bins, kh: 0 });
    }

    let mut hist = vec![0u32; bins];
    let scale = bins as f32;
    for &pixel in image {
        let clamped = pixel.clamp(0.0, 1.0 - f32::EPSILON);
        let bucket = (clamped * scale) as usize;
        hist[bucket] += 1;
    }

    Ok(hist)
}

/// Compute cumulative histogram (CDF).
pub fn cumulative_histogram(hist: &[u32]) -> Vec<u32> {
    let mut cdf = vec![0u32; hist.len()];
    if hist.is_empty() {
        return cdf;
    }
    cdf[0] = hist[0];
    for i in 1..hist.len() {
        cdf[i] = cdf[i - 1] + hist[i];
    }
    cdf
}

/// Histogram equalization: remap image to uniform histogram.
///
/// # Errors
///
/// Returns error if dimensions don't match.
pub fn equalize(
    image: &[f32],
    width: usize,
    height: usize,
    bins: usize,
) -> Result<Vec<f32>, ImageError> {
    let hist = histogram(image, width, height, bins)?;
    let cdf = cumulative_histogram(&hist);
    let total = (width * height) as f32;

    let scale = bins as f32;
    let mut output = vec![0.0f32; image.len()];
    for (i, &pixel) in image.iter().enumerate() {
        let clamped = pixel.clamp(0.0, 1.0 - f32::EPSILON);
        let bucket = (clamped * scale) as usize;
        output[i] = cdf[bucket] as f32 / total;
    }

    Ok(output)
}
