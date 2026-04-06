//! Color space conversions and connected component labeling.
//!
//! # Contract: image-color-v1.yaml
//!
//! Provides RGB→Grayscale, RGB→HSV, HSV→RGB, and connected component labeling.
//!
//! ## Proof obligations
//! - HSV roundtrip: RGB→HSV→RGB = identity within tolerance
//! - Grayscale luminance: uses ITU-R BT.601 weights
//! - Connected components: each label is 4-connected

use crate::error::ImageError;

/// Convert RGB image to grayscale using ITU-R BT.601 weights.
///
/// Input: interleaved RGB (3 channels), length = width * height * 3.
/// Output: grayscale, length = width * height.
///
/// # Errors
///
/// Returns error if buffer lengths don't match dimensions.
pub fn rgb_to_gray(rgb: &[f32], width: usize, height: usize) -> Result<Vec<f32>, ImageError> {
    let pixels = width * height;
    if rgb.len() != pixels * 3 {
        return Err(ImageError::BufferLengthMismatch {
            expected: pixels * 3,
            got: rgb.len(),
            width,
            height,
        });
    }

    let mut gray = Vec::with_capacity(pixels);
    for i in 0..pixels {
        let r = rgb[i * 3];
        let g = rgb[i * 3 + 1];
        let b = rgb[i * 3 + 2];
        // ITU-R BT.601 luma
        gray.push(0.299 * r + 0.587 * g + 0.114 * b);
    }
    Ok(gray)
}

/// Convert RGB to HSV color space.
///
/// Input/output: interleaved 3-channel, length = width * height * 3.
/// H in [0, 360), S in [0, 1], V in [0, 1]. RGB in [0, 1].
///
/// # Errors
///
/// Returns error if buffer length doesn't match dimensions.
pub fn rgb_to_hsv(rgb: &[f32], width: usize, height: usize) -> Result<Vec<f32>, ImageError> {
    let pixels = width * height;
    if rgb.len() != pixels * 3 {
        return Err(ImageError::BufferLengthMismatch {
            expected: pixels * 3,
            got: rgb.len(),
            width,
            height,
        });
    }

    let mut hsv = vec![0.0_f32; pixels * 3];
    for i in 0..pixels {
        let (h, s, v) = rgb_pixel_to_hsv(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
        hsv[i * 3] = h;
        hsv[i * 3 + 1] = s;
        hsv[i * 3 + 2] = v;
    }
    Ok(hsv)
}

/// Convert a single RGB pixel to HSV.
fn rgb_pixel_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = max;
    let s = if max > f32::EPSILON { delta / max } else { 0.0 };

    let h = if delta < f32::EPSILON {
        0.0
    } else if (max - r).abs() < f32::EPSILON {
        60.0 * (((g - b) / delta) % 6.0)
    } else if (max - g).abs() < f32::EPSILON {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };

    let h = if h < 0.0 { h + 360.0 } else { h };
    (h, s, v)
}

/// Convert HSV to RGB color space.
///
/// H in [0, 360), S in [0, 1], V in [0, 1]. Output RGB in [0, 1].
///
/// # Errors
///
/// Returns error if buffer length doesn't match dimensions.
pub fn hsv_to_rgb(hsv: &[f32], width: usize, height: usize) -> Result<Vec<f32>, ImageError> {
    let pixels = width * height;
    if hsv.len() != pixels * 3 {
        return Err(ImageError::BufferLengthMismatch {
            expected: pixels * 3,
            got: hsv.len(),
            width,
            height,
        });
    }

    let mut rgb = vec![0.0_f32; pixels * 3];
    for i in 0..pixels {
        let (r, g, b) = hsv_pixel_to_rgb(hsv[i * 3], hsv[i * 3 + 1], hsv[i * 3 + 2]);
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }
    Ok(rgb)
}

/// Convert a single HSV pixel to RGB.
fn hsv_pixel_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    if s < f32::EPSILON {
        return (v, v, v);
    }

    let h = h % 360.0;
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r1, g1, b1) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    (r1 + m, g1 + m, b1 + m)
}

/// Connected component labeling using union-find (4-connectivity).
///
/// Input: binary image (0.0 = background, nonzero = foreground).
/// Output: label array where each connected region has a unique ID (0 = background).
///
/// # Errors
///
/// Returns error if buffer length doesn't match dimensions.
pub fn connected_components(
    image: &[f32],
    width: usize,
    height: usize,
) -> Result<Vec<u32>, ImageError> {
    if image.len() != width * height {
        return Err(ImageError::BufferLengthMismatch {
            expected: width * height,
            got: image.len(),
            width,
            height,
        });
    }

    let pixels = width * height;
    let mut labels = vec![0u32; pixels];
    let mut parent = Vec::with_capacity(pixels / 4);
    parent.push(0); // label 0 = background

    // First pass: assign provisional labels
    first_pass(image, &mut labels, &mut parent, width, height);

    // Flatten all paths
    for i in 0..parent.len() {
        parent[i] = find(&parent, i as u32);
    }

    // Second pass: relabel with canonical labels
    let mut remap = vec![0u32; parent.len()];
    let mut next_label = 1u32;
    for i in 1..parent.len() {
        let root = parent[i] as usize;
        if remap[root] == 0 {
            remap[root] = next_label;
            next_label += 1;
        }
        remap[i] = remap[root];
    }

    for label in &mut labels {
        if *label > 0 {
            *label = remap[*label as usize];
        }
    }

    Ok(labels)
}

/// First pass: scan pixels, assign labels, merge with union-find.
fn first_pass(
    image: &[f32],
    labels: &mut [u32],
    parent: &mut Vec<u32>,
    width: usize,
    height: usize,
) {
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if image[idx].abs() < f32::EPSILON {
                continue; // background
            }

            let left = if x > 0 { labels[idx - 1] } else { 0 };
            let above = if y > 0 { labels[idx - width] } else { 0 };

            match (left > 0, above > 0) {
                (false, false) => {
                    // New label
                    let new_label = parent.len() as u32;
                    parent.push(new_label);
                    labels[idx] = new_label;
                }
                (true, false) => labels[idx] = left,
                (false, true) => labels[idx] = above,
                (true, true) => {
                    labels[idx] = left.min(above);
                    union(parent, left, above);
                }
            }
        }
    }
}

/// Find root with path compression (iterative).
fn find(parent: &[u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        x = parent[x as usize];
    }
    x
}

/// Union two labels.
fn union(parent: &mut [u32], a: u32, b: u32) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        let min = ra.min(rb);
        let max = ra.max(rb);
        parent[max as usize] = min;
    }
}
