//! Image processing tests — provable contracts and falsification.

use crate::conv::*;
use crate::histogram::{cumulative_histogram, equalize, histogram};
use crate::morphology::{closing, dilate, erode, opening};
use crate::resize::{resize, Interpolation};

// ============================================================================
// FALSIFY-IMG-002: Identity kernel preserves input
// ============================================================================

#[test]
fn test_identity_kernel() {
    let image = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0_f32];
    let delta = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0_f32];
    let out = conv2d(&image, 3, 3, &delta, 3, 3, BorderMode::Zero).expect("ok");

    // Interior pixel should be exactly preserved
    assert!((out[4] - 5.0).abs() < 1e-6, "Identity kernel failed at center: {}", out[4]);
}

#[test]
fn test_identity_kernel_larger() {
    let w = 5;
    let h = 5;
    let image: Vec<f32> = (0..w * h).map(|i| i as f32).collect();
    let delta = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0_f32];
    let out = conv2d(&image, w, h, &delta, 3, 3, BorderMode::Clamp).expect("ok");

    // Interior pixels should be preserved
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let idx = y * w + x;
            assert!(
                (out[idx] - image[idx]).abs() < 1e-6,
                "Identity failed at ({x},{y}): out={}, expected={}",
                out[idx],
                image[idx]
            );
        }
    }
}

// ============================================================================
// FALSIFY-IMG-001: Separable equivalence
// ============================================================================

#[test]
fn test_separable_matches_2d() {
    let w = 8;
    let h = 8;
    let image: Vec<f32> = (0..w * h).map(|i| (i as f32).sin()).collect();

    // Gaussian kernel σ=1.0
    let h_kernel = [0.2742, 0.4514, 0.2742_f32]; // Approximate
    let v_kernel = h_kernel;

    // Full 2D kernel = outer product
    let mut kernel_2d = [0.0f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            kernel_2d[i * 3 + j] = v_kernel[i] * h_kernel[j];
        }
    }

    let out_2d = conv2d(&image, w, h, &kernel_2d, 3, 3, BorderMode::Zero).expect("ok");
    let out_sep =
        separable_conv2d(&image, w, h, &h_kernel, &v_kernel, BorderMode::Zero).expect("ok");

    for i in 0..w * h {
        assert!(
            (out_2d[i] - out_sep[i]).abs() < 1e-4,
            "Separable mismatch at {i}: 2d={}, sep={}",
            out_2d[i],
            out_sep[i]
        );
    }
}

// ============================================================================
// Gaussian blur tests
// ============================================================================

#[test]
fn test_gaussian_blur_constant_image() {
    let w = 10;
    let h = 10;
    let image = vec![5.0f32; w * h];
    let blurred = gaussian_blur(&image, w, h, 1.0).expect("ok");

    // Blurring a constant image should give the same constant
    for (i, &v) in blurred.iter().enumerate() {
        assert!((v - 5.0).abs() < 1e-3, "Gaussian blur changed constant at {i}: {v}");
    }
}

#[test]
fn test_gaussian_blur_reduces_range() {
    let w = 10;
    let h = 10;
    let mut image = vec![0.0f32; w * h];
    image[5 * w + 5] = 100.0; // Single bright pixel

    let blurred = gaussian_blur(&image, w, h, 1.0).expect("ok");
    let max_blurred = blurred.iter().copied().fold(0.0f32, f32::max);

    // Blurring should reduce the peak
    assert!(max_blurred < 100.0, "Gaussian blur didn't reduce peak: {max_blurred}");
    // Total energy should be approximately conserved
    let sum_orig: f32 = image.iter().sum();
    let sum_blurred: f32 = blurred.iter().sum();
    assert!(
        (sum_orig - sum_blurred).abs() / sum_orig < 0.05,
        "Energy not conserved: orig={sum_orig}, blurred={sum_blurred}"
    );
}

// ============================================================================
// Sobel tests
// ============================================================================

#[test]
fn test_sobel_uniform_image() {
    let w = 5;
    let h = 5;
    let image = vec![3.0f32; w * h]; // Uniform → zero gradient

    let (gx, gy) = sobel(&image, w, h).expect("ok");

    // Interior gradients should be ~0
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let idx = y * w + x;
            assert!(
                gx[idx].abs() < 1e-5 && gy[idx].abs() < 1e-5,
                "Non-zero gradient on uniform at ({x},{y}): gx={}, gy={}",
                gx[idx],
                gy[idx]
            );
        }
    }
}

#[test]
fn test_sobel_horizontal_edge() {
    let w = 5;
    let h = 5;
    let mut image = vec![0.0f32; w * h];
    // Top half = 0, bottom half = 1 → vertical gradient
    for y in h / 2 + 1..h {
        for x in 0..w {
            image[y * w + x] = 1.0;
        }
    }

    let (_gx, gy) = sobel(&image, w, h).expect("ok");

    // Should have non-zero vertical gradient near the edge
    let edge_y = h / 2;
    let center_x = w / 2;
    let idx = edge_y * w + center_x;
    assert!(gy[idx].abs() > 0.1, "Expected vertical gradient at edge: gy={}", gy[idx]);
}

// ============================================================================
// Canny tests
// ============================================================================

#[test]
fn test_canny_uniform_no_edges() {
    let w = 20;
    let h = 20;
    let image = vec![0.5f32; w * h];

    let edges = canny(&image, w, h, 1.0, 0.1, 0.3).expect("ok");
    let edge_count: usize = edges.iter().filter(|&&v| v > 0.5).count();
    assert_eq!(edge_count, 0, "Uniform image should have no edges");
}

#[test]
fn test_canny_strong_edge() {
    let w = 20;
    let h = 20;
    let mut image = vec![0.0f32; w * h];
    // Left half = 0, right half = 1 → strong vertical edge
    for y in 0..h {
        for x in w / 2..w {
            image[y * w + x] = 1.0;
        }
    }

    let edges = canny(&image, w, h, 1.0, 0.05, 0.15).expect("ok");
    let edge_count: usize = edges.iter().filter(|&&v| v > 0.5).count();
    assert!(edge_count > 0, "Strong edge should be detected");
}

#[test]
fn test_canny_invalid_thresholds() {
    let image = vec![0.0f32; 100];
    assert!(canny(&image, 10, 10, 1.0, 0.5, 0.3).is_err()); // low > high
    assert!(canny(&image, 10, 10, 1.0, -0.1, 0.3).is_err()); // negative
}

// ============================================================================
// Error handling tests
// ============================================================================

#[test]
fn test_conv2d_zero_dimensions() {
    assert!(conv2d(&[], 0, 0, &[1.0], 1, 1, BorderMode::Zero).is_err());
}

#[test]
fn test_conv2d_even_kernel() {
    let image = vec![1.0f32; 4];
    assert!(conv2d(&image, 2, 2, &[1.0; 4], 2, 2, BorderMode::Zero).is_err());
}

#[test]
fn test_conv2d_buffer_mismatch() {
    let image = vec![1.0f32; 3]; // 3 pixels but claiming 2x2
    assert!(conv2d(&image, 2, 2, &[1.0], 1, 1, BorderMode::Zero).is_err());
}

// ============================================================================
// Border mode tests
// ============================================================================

#[test]
fn test_border_clamp() {
    let image = vec![1.0, 2.0, 3.0, 4.0_f32]; // 2×2
    let k = [0.0, 1.0, 0.0_f32]; // 3×1 horizontal kernel (just right neighbor)
                                 // Actually let's use a 3x1 → need it as 1x3 for conv2d (kw=3, kh=1)
    let out = conv2d(&image, 2, 2, &k, 3, 1, BorderMode::Clamp).expect("ok");

    // At (0,0): neighbors are clamp(-1,0)=image[0]=1, image[0]=1, image[1]=2
    // kernel [0, 1, 0] → picks center = image[0] = 1.0
    assert!((out[0] - 1.0).abs() < 1e-5, "Clamp border: {}", out[0]);
}

// ============================================================================
// Property-based tests
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_identity_kernel_preserves(
            w in 3usize..20,
            h in 3usize..20,
        ) {
            let image: Vec<f32> = (0..w * h).map(|i| i as f32).collect();
            let delta = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0_f32];
            let out = conv2d(&image, w, h, &delta, 3, 3, BorderMode::Clamp).expect("ok");

            // Interior pixels preserved
            for y in 1..h-1 {
                for x in 1..w-1 {
                    let idx = y * w + x;
                    prop_assert!((out[idx] - image[idx]).abs() < 1e-5);
                }
            }
        }

        #[test]
        fn prop_gaussian_constant_preserves(
            w in 5usize..15,
            h in 5usize..15,
            val in -100.0f32..100.0,
        ) {
            let image = vec![val; w * h];
            let blurred = gaussian_blur(&image, w, h, 1.0).expect("ok");

            for (i, &v) in blurred.iter().enumerate() {
                prop_assert!(
                    (v - val).abs() < 0.1,
                    "Gaussian changed constant at {i}: {v} vs {val}"
                );
            }
        }
    }
}

// ============================================================================
// Histogram
// ============================================================================

#[test]
fn test_histogram_uniform() -> Result<(), Box<dyn std::error::Error>> {
    let image: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
    let hist = histogram(&image, 16, 16, 256)?;
    let total: u32 = hist.iter().sum();
    assert_eq!(total, 256);
    Ok(())
}

#[test]
fn test_histogram_constant() -> Result<(), Box<dyn std::error::Error>> {
    let image = vec![0.5f32; 100];
    let hist = histogram(&image, 10, 10, 10)?;
    let total: u32 = hist.iter().sum();
    assert_eq!(total, 100);
    assert_eq!(hist[5], 100);
    Ok(())
}

#[test]
fn test_cumulative_histogram() {
    let hist = vec![1, 2, 3, 4u32];
    let cdf = cumulative_histogram(&hist);
    assert_eq!(cdf, vec![1, 3, 6, 10]);
}

#[test]
fn test_equalize_constant() -> Result<(), Box<dyn std::error::Error>> {
    let image = vec![0.5f32; 25];
    let eq = equalize(&image, 5, 5, 10)?;
    let first = eq[0];
    for &v in &eq {
        assert!((v - first).abs() < 1e-6);
    }
    Ok(())
}

// ============================================================================
// Morphology
// ============================================================================

#[test]
fn test_dilate_expands() -> Result<(), Box<dyn std::error::Error>> {
    let mut image = vec![0.0f32; 25];
    image[12] = 1.0;
    let se = vec![1.0f32; 9];
    let result = dilate(&image, 5, 5, &se, 3, 3)?;
    assert!((result[12] - 1.0).abs() < 1e-6);
    assert!((result[7] - 1.0).abs() < 1e-6);
    assert!((result[17] - 1.0).abs() < 1e-6);
    assert!((result[11] - 1.0).abs() < 1e-6);
    assert!((result[13] - 1.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_erode_shrinks() -> Result<(), Box<dyn std::error::Error>> {
    let image = vec![1.0f32; 25];
    let se = vec![1.0f32; 9];
    let result = erode(&image, 5, 5, &se, 3, 3)?;
    assert!((result[12] - 1.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_opening_removes_small_bright() -> Result<(), Box<dyn std::error::Error>> {
    let mut image = vec![0.0f32; 25];
    image[12] = 1.0;
    let se = vec![1.0f32; 9];
    let result = opening(&image, 5, 5, &se, 3, 3)?;
    assert!(result[12] < 0.5);
    Ok(())
}

#[test]
fn test_closing_fills_small_dark() -> Result<(), Box<dyn std::error::Error>> {
    let mut image = vec![1.0f32; 25];
    image[12] = 0.0;
    let se = vec![1.0f32; 9];
    let result = closing(&image, 5, 5, &se, 3, 3)?;
    assert!(result[12] > 0.5);
    Ok(())
}

#[test]
fn test_dilate_erode_duality() -> Result<(), Box<dyn std::error::Error>> {
    let image = vec![0.5f32; 25];
    let se = vec![1.0f32; 9];
    let d = dilate(&image, 5, 5, &se, 3, 3)?;
    let e = erode(&image, 5, 5, &se, 3, 3)?;
    for i in 0..25 {
        assert!((d[i] - 0.5).abs() < 1e-6);
        assert!((e[i] - 0.5).abs() < 1e-6);
    }
    Ok(())
}

// ============================================================================
// Resize
// ============================================================================

#[test]
fn test_resize_nearest_identity() -> Result<(), Box<dyn std::error::Error>> {
    let image = vec![1.0, 2.0, 3.0, 4.0f32];
    let result = resize(&image, 2, 2, 2, 2, Interpolation::Nearest)?;
    for i in 0..4 {
        assert!((result[i] - image[i]).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn test_resize_bilinear_identity() -> Result<(), Box<dyn std::error::Error>> {
    let image = vec![1.0, 2.0, 3.0, 4.0f32];
    let result = resize(&image, 2, 2, 2, 2, Interpolation::Bilinear)?;
    for i in 0..4 {
        assert!((result[i] - image[i]).abs() < 1e-5);
    }
    Ok(())
}

#[test]
fn test_resize_upscale_nearest() -> Result<(), Box<dyn std::error::Error>> {
    let image = vec![0.5f32];
    let result = resize(&image, 1, 1, 4, 4, Interpolation::Nearest)?;
    assert_eq!(result.len(), 16);
    for &v in &result {
        assert!((v - 0.5).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn test_resize_downscale_bilinear() -> Result<(), Box<dyn std::error::Error>> {
    let image = vec![0.7f32; 16];
    let result = resize(&image, 4, 4, 2, 2, Interpolation::Bilinear)?;
    assert_eq!(result.len(), 4);
    for &v in &result {
        assert!((v - 0.7).abs() < 1e-5);
    }
    Ok(())
}

#[test]
fn test_resize_zero_output() {
    let image = vec![1.0f32; 4];
    assert!(resize(&image, 2, 2, 0, 2, Interpolation::Nearest).is_err());
}

// ============================================================================
// Color conversion tests (Contract: image-color-v1.yaml)
// ============================================================================

use crate::color::{connected_components, hsv_to_rgb, rgb_to_gray, rgb_to_hsv};

#[test]
fn test_rgb_to_gray_bt601() -> Result<(), Box<dyn std::error::Error>> {
    // Pure white → 1.0
    let rgb = vec![1.0, 1.0, 1.0_f32];
    let gray = rgb_to_gray(&rgb, 1, 1)?;
    assert!((gray[0] - 1.0).abs() < 1e-5, "White → {}", gray[0]);

    // Pure red → 0.299
    let rgb = vec![1.0, 0.0, 0.0_f32];
    let gray = rgb_to_gray(&rgb, 1, 1)?;
    assert!((gray[0] - 0.299).abs() < 1e-3, "Red → {}", gray[0]);

    // Pure green → 0.587
    let rgb = vec![0.0, 1.0, 0.0_f32];
    let gray = rgb_to_gray(&rgb, 1, 1)?;
    assert!((gray[0] - 0.587).abs() < 1e-3, "Green → {}", gray[0]);
    Ok(())
}

#[test]
fn test_rgb_to_gray_buffer_mismatch() {
    // 2 pixels but only 5 values (need 6)
    let rgb = vec![1.0; 5];
    assert!(rgb_to_gray(&rgb, 2, 1).is_err());
}

#[test]
fn test_hsv_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    // Various RGB colors → HSV → RGB should roundtrip
    let colors = [
        1.0, 0.0, 0.0, // red
        0.0, 1.0, 0.0, // green
        0.0, 0.0, 1.0, // blue
        0.5, 0.5, 0.5, // gray
        1.0, 1.0, 0.0, // yellow
        0.0, 1.0, 1.0, // cyan
    ];
    let w = 6;
    let h = 1;
    let hsv = rgb_to_hsv(&colors, w, h)?;
    let recovered = hsv_to_rgb(&hsv, w, h)?;

    for i in 0..colors.len() {
        let err = (colors[i] - recovered[i]).abs();
        assert!(
            err < 1e-4,
            "HSV roundtrip at {i}: orig={}, rec={}, err={err}",
            colors[i],
            recovered[i]
        );
    }
    Ok(())
}

#[test]
fn test_hsv_black() -> Result<(), Box<dyn std::error::Error>> {
    let rgb = vec![0.0, 0.0, 0.0_f32];
    let hsv = rgb_to_hsv(&rgb, 1, 1)?;
    // Black: V=0, S=0
    assert!(hsv[2].abs() < 1e-6);
    assert!(hsv[1].abs() < 1e-6);
    Ok(())
}

#[test]
fn test_hsv_white() -> Result<(), Box<dyn std::error::Error>> {
    let rgb = vec![1.0, 1.0, 1.0_f32];
    let hsv = rgb_to_hsv(&rgb, 1, 1)?;
    // White: V=1, S=0
    assert!((hsv[2] - 1.0).abs() < 1e-6);
    assert!(hsv[1].abs() < 1e-6);
    Ok(())
}

// ============================================================================
// Connected components tests
// ============================================================================

#[test]
fn test_connected_components_single_blob() -> Result<(), Box<dyn std::error::Error>> {
    #[rustfmt::skip]
    let image = vec![
        0.0, 1.0, 1.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0_f32,
    ];
    let labels = connected_components(&image, 3, 3)?;
    // The three foreground pixels should share one label
    assert_eq!(labels[0], 0); // background
    assert!(labels[1] > 0);
    assert_eq!(labels[1], labels[2]); // connected
    assert_eq!(labels[1], labels[4]); // connected to below
    Ok(())
}

#[test]
fn test_connected_components_two_blobs() -> Result<(), Box<dyn std::error::Error>> {
    #[rustfmt::skip]
    let image = vec![
        1.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 1.0_f32,
    ];
    let labels = connected_components(&image, 3, 3)?;
    // Three separate components (4-connectivity)
    let l0 = labels[0]; // top-left
    let l2 = labels[2]; // top-right
    let l8 = labels[8]; // bottom-right
    assert!(l0 > 0);
    assert!(l2 > 0);
    assert!(l8 > 0);
    assert_ne!(l0, l2);
    assert_ne!(l0, l8);
    assert_ne!(l2, l8);
    Ok(())
}

#[test]
fn test_connected_components_all_background() -> Result<(), Box<dyn std::error::Error>> {
    let image = vec![0.0_f32; 9];
    let labels = connected_components(&image, 3, 3)?;
    assert!(labels.iter().all(|&l| l == 0));
    Ok(())
}

#[test]
fn test_connected_components_all_foreground() -> Result<(), Box<dyn std::error::Error>> {
    let image = vec![1.0_f32; 9];
    let labels = connected_components(&image, 3, 3)?;
    // All one component
    let first = labels[0];
    assert!(first > 0);
    assert!(labels.iter().all(|&l| l == first));
    Ok(())
}

#[test]
fn test_connected_components_buffer_mismatch() {
    let image = vec![1.0_f32; 5];
    assert!(connected_components(&image, 3, 3).is_err());
}

// ============================================================================
// Bicubic and Lanczos interpolation tests
// ============================================================================

#[test]
fn test_resize_bicubic_identity() -> Result<(), Box<dyn std::error::Error>> {
    let image = vec![1.0, 2.0, 3.0, 4.0f32];
    let result = resize(&image, 2, 2, 2, 2, Interpolation::Bicubic)?;
    for i in 0..4 {
        assert!((result[i] - image[i]).abs() < 0.2, "Bicubic identity at {i}: {}", result[i]);
    }
    Ok(())
}

#[test]
fn test_resize_lanczos_identity() -> Result<(), Box<dyn std::error::Error>> {
    let image = vec![1.0, 2.0, 3.0, 4.0f32];
    let result = resize(&image, 2, 2, 2, 2, Interpolation::Lanczos)?;
    for i in 0..4 {
        assert!((result[i] - image[i]).abs() < 0.2, "Lanczos identity at {i}: {}", result[i]);
    }
    Ok(())
}

#[test]
fn test_resize_bicubic_constant() -> Result<(), Box<dyn std::error::Error>> {
    let image = vec![0.7f32; 16];
    let result = resize(&image, 4, 4, 8, 8, Interpolation::Bicubic)?;
    assert_eq!(result.len(), 64);
    for &v in &result {
        assert!((v - 0.7).abs() < 0.01, "Bicubic constant: {v}");
    }
    Ok(())
}

#[test]
fn test_resize_lanczos_constant() -> Result<(), Box<dyn std::error::Error>> {
    let image = vec![0.7f32; 16];
    let result = resize(&image, 4, 4, 8, 8, Interpolation::Lanczos)?;
    assert_eq!(result.len(), 64);
    for &v in &result {
        assert!((v - 0.7).abs() < 0.01, "Lanczos constant: {v}");
    }
    Ok(())
}

#[test]
fn test_resize_bicubic_downscale() -> Result<(), Box<dyn std::error::Error>> {
    let image = vec![0.5f32; 64];
    let result = resize(&image, 8, 8, 4, 4, Interpolation::Bicubic)?;
    assert_eq!(result.len(), 16);
    for &v in &result {
        assert!((v - 0.5).abs() < 0.01);
    }
    Ok(())
}

#[test]
fn test_resize_lanczos_downscale() -> Result<(), Box<dyn std::error::Error>> {
    let image = vec![0.5f32; 64];
    let result = resize(&image, 8, 8, 4, 4, Interpolation::Lanczos)?;
    assert_eq!(result.len(), 16);
    for &v in &result {
        assert!((v - 0.5).abs() < 0.01);
    }
    Ok(())
}

// ============================================================================
// BorderMode::Wrap tests
// ============================================================================

#[test]
fn test_conv_wrap_border() -> Result<(), Box<dyn std::error::Error>> {
    // 3×3 image with wrap boundary: average kernel should see wrapped pixels
    let image = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0_f32];
    let avg = [1.0 / 9.0; 9];
    let out = conv2d(&image, 3, 3, &avg, 3, 3, BorderMode::Wrap)?;
    // Center pixel sees 1/9 contribution from wrapped (0,0)
    assert!(out[4] > 0.0, "Wrap should contribute to center");
    Ok(())
}

#[test]
fn test_conv_wrap_periodic() -> Result<(), Box<dyn std::error::Error>> {
    // Constant image under wrap should be identity with identity kernel
    let image = vec![3.0f32; 9];
    let delta = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0_f32];
    let out = conv2d(&image, 3, 3, &delta, 3, 3, BorderMode::Wrap)?;
    for (i, &v) in out.iter().enumerate() {
        assert!((v - 3.0).abs() < 1e-5, "Wrap identity at {i}: {v}");
    }
    Ok(())
}

// ── ImageBuf tests ──────────────────────────────────────────────

use crate::buf::{DType, ImageBuf};

#[test]
fn test_imagebuf_new() -> Result<(), Box<dyn std::error::Error>> {
    let buf = ImageBuf::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2, 1)?;
    assert_eq!(buf.width(), 3);
    assert_eq!(buf.height(), 2);
    assert_eq!(buf.channels(), 1);
    assert_eq!(buf.dtype(), DType::F32);
    assert_eq!(buf.len(), 6);
    assert!(!buf.is_empty());
    Ok(())
}

#[test]
fn test_imagebuf_zeros() {
    let buf = ImageBuf::zeros(4, 4, 3);
    assert_eq!(buf.len(), 48);
    assert!(buf.data().iter().all(|&v| v == 0.0));
}

#[test]
fn test_imagebuf_channel_extract() -> Result<(), Box<dyn std::error::Error>> {
    // RGB image: 2×1, 3 channels
    let buf = ImageBuf::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 1, 3)?;
    let r = buf.channel(0)?;
    assert_eq!(r.channels(), 1);
    assert_eq!(r.data(), &[1.0, 4.0]);
    let g = buf.channel(1)?;
    assert_eq!(g.data(), &[2.0, 5.0]);
    let b = buf.channel(2)?;
    assert_eq!(b.data(), &[3.0, 6.0]);
    Ok(())
}

#[test]
fn test_imagebuf_invalid_channel() -> Result<(), Box<dyn std::error::Error>> {
    let buf = ImageBuf::new(vec![1.0, 2.0, 3.0], 1, 1, 3)?;
    assert!(buf.channel(3).is_err());
    Ok(())
}

#[test]
fn test_imagebuf_dimension_mismatch() {
    let result = ImageBuf::new(vec![1.0, 2.0, 3.0], 2, 2, 1);
    assert!(result.is_err());
}

// ── canny_rgb tests (NPP 3-channel parity) ─────────────────────

#[test]
fn test_canny_rgb_uniform_no_edges() -> Result<(), Box<dyn std::error::Error>> {
    let w = 20;
    let h = 20;
    let image = vec![0.5f32; w * h * 3];
    let edges = canny_rgb(&image, w, h, 3, 1.0, 0.1, 0.3)?;
    assert_eq!(edges.len(), w * h);
    let edge_count: usize = edges.iter().filter(|&&v| v > 0.5).count();
    assert_eq!(edge_count, 0, "Uniform RGB should have no edges");
    Ok(())
}

#[test]
fn test_canny_rgb_strong_edge() -> Result<(), Box<dyn std::error::Error>> {
    let w = 20;
    let h = 20;
    let mut image = vec![0.0f32; w * h * 3];
    // Right half = bright white
    for y in 0..h {
        for x in w / 2..w {
            let base = (y * w + x) * 3;
            image[base] = 1.0;
            image[base + 1] = 1.0;
            image[base + 2] = 1.0;
        }
    }
    let edges = canny_rgb(&image, w, h, 3, 1.0, 0.05, 0.15)?;
    let edge_count: usize = edges.iter().filter(|&&v| v > 0.5).count();
    assert!(edge_count > 0, "Strong RGB edge should be detected");
    Ok(())
}

#[test]
fn test_canny_rgb_single_channel() -> Result<(), Box<dyn std::error::Error>> {
    let w = 10;
    let h = 10;
    let image = vec![0.5f32; w * h];
    let edges = canny_rgb(&image, w, h, 1, 1.0, 0.1, 0.3)?;
    assert_eq!(edges.len(), w * h);
    Ok(())
}

#[test]
fn test_canny_rgb_buffer_mismatch() {
    let image = vec![0.5f32; 10]; // 10 pixels but claiming 10x10x3
    assert!(canny_rgb(&image, 10, 10, 3, 1.0, 0.1, 0.3).is_err());
}

// ── ImageOps trait tests (Contract: §5C) ────────────────────────

use crate::ImageOps;

#[test]
fn test_imageops_blur_grayscale() -> Result<(), Box<dyn std::error::Error>> {
    let buf = ImageBuf::new(vec![5.0f32; 25], 5, 5, 1)?;
    let blurred = buf.blur(1.0)?;
    assert_eq!(blurred.width(), 5);
    assert_eq!(blurred.height(), 5);
    assert_eq!(blurred.channels(), 1);
    for &v in blurred.data() {
        assert!((v - 5.0).abs() < 0.1, "Blur changed constant: {v}");
    }
    Ok(())
}

#[test]
fn test_imageops_blur_rgb() -> Result<(), Box<dyn std::error::Error>> {
    let buf = ImageBuf::new(vec![0.5f32; 75], 5, 5, 3)?;
    let blurred = buf.blur(1.0)?;
    assert_eq!(blurred.channels(), 3);
    assert_eq!(blurred.len(), 75);
    for &v in blurred.data() {
        assert!((v - 0.5).abs() < 0.1, "RGB blur changed constant: {v}");
    }
    Ok(())
}

#[test]
fn test_imageops_to_gray() -> Result<(), Box<dyn std::error::Error>> {
    // White pixel → gray = 1.0
    let buf = ImageBuf::new(vec![1.0, 1.0, 1.0], 1, 1, 3)?;
    let gray = buf.to_gray()?;
    assert_eq!(gray.channels(), 1);
    assert!((gray.data()[0] - 1.0).abs() < 1e-3);
    Ok(())
}

#[test]
fn test_imageops_to_gray_already_gray() -> Result<(), Box<dyn std::error::Error>> {
    let buf = ImageBuf::new(vec![0.7, 0.3, 0.5, 0.9], 2, 2, 1)?;
    let gray = buf.to_gray()?;
    assert_eq!(gray.data(), buf.data());
    Ok(())
}

#[test]
fn test_imageops_canny_edges() -> Result<(), Box<dyn std::error::Error>> {
    let w = 20;
    let h = 20;
    let buf = ImageBuf::new(vec![0.5f32; w * h * 3], w, h, 3)?;
    let edges = buf.canny_edges(1.0, 0.1, 0.3)?;
    assert_eq!(edges.channels(), 1);
    assert_eq!(edges.len(), w * h);
    let edge_count: usize = edges.data().iter().filter(|&&v| v > 0.5).count();
    assert_eq!(edge_count, 0, "Uniform image should have no edges");
    Ok(())
}

// ── Expanded ImageOps trait tests (§5C parity) ──────────────────

#[test]
fn test_imageops_conv2d() -> Result<(), Box<dyn std::error::Error>> {
    let buf = ImageBuf::new(vec![5.0f32; 25], 5, 5, 1)?;
    let delta = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0_f32];
    let result = buf.apply_conv2d(&delta, 3, 3, BorderMode::Zero)?;
    assert_eq!(result.width(), 5);
    // Interior pixel should be preserved
    assert!((result.data()[12] - 5.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_imageops_sobel_gradients() -> Result<(), Box<dyn std::error::Error>> {
    let buf = ImageBuf::new(vec![3.0f32; 25], 5, 5, 1)?;
    let (gx, gy) = buf.sobel_gradients()?;
    assert_eq!(gx.channels(), 1);
    assert_eq!(gy.channels(), 1);
    // Uniform image → zero gradients at interior
    assert!(gx.data()[12].abs() < 1e-5);
    assert!(gy.data()[12].abs() < 1e-5);
    Ok(())
}

#[test]
fn test_imageops_dilate_erode() -> Result<(), Box<dyn std::error::Error>> {
    let mut data = vec![0.0f32; 25];
    data[12] = 1.0;
    let buf = ImageBuf::new(data, 5, 5, 1)?;
    let se = [1.0f32; 9];
    let dilated = buf.apply_dilate(&se, 3, 3)?;
    assert!((dilated.data()[12] - 1.0).abs() < 1e-6);
    assert!((dilated.data()[7] - 1.0).abs() < 1e-6); // neighbor expanded

    let full = ImageBuf::new(vec![1.0f32; 25], 5, 5, 1)?;
    let eroded = full.apply_erode(&se, 3, 3)?;
    assert!((eroded.data()[12] - 1.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_imageops_resize() -> Result<(), Box<dyn std::error::Error>> {
    let buf = ImageBuf::new(vec![0.5f32; 16], 4, 4, 1)?;
    let resized = buf.apply_resize(2, 2, Interpolation::Bilinear)?;
    assert_eq!(resized.width(), 2);
    assert_eq!(resized.height(), 2);
    assert_eq!(resized.len(), 4);
    for &v in resized.data() {
        assert!((v - 0.5).abs() < 0.1);
    }
    Ok(())
}

#[test]
fn test_imageops_to_hsv() -> Result<(), Box<dyn std::error::Error>> {
    // Pure red → H≈0, S=1, V=1
    let buf = ImageBuf::new(vec![1.0, 0.0, 0.0], 1, 1, 3)?;
    let hsv = buf.to_hsv()?;
    assert_eq!(hsv.channels(), 3);
    assert!((hsv.data()[1] - 1.0).abs() < 1e-4); // S = 1
    assert!((hsv.data()[2] - 1.0).abs() < 1e-4); // V = 1
    Ok(())
}

#[test]
fn test_imageops_histogram() -> Result<(), Box<dyn std::error::Error>> {
    let buf = ImageBuf::new(vec![0.5f32; 25], 5, 5, 1)?;
    let hist = buf.compute_histogram(10)?;
    let total: u32 = hist.iter().sum();
    assert_eq!(total, 25);
    Ok(())
}

#[test]
fn test_imageops_connected_components() -> Result<(), Box<dyn std::error::Error>> {
    #[rustfmt::skip]
    let data = vec![
        1.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 1.0_f32,
    ];
    let buf = ImageBuf::new(data, 3, 3, 1)?;
    let (labels, num) = buf.label_components()?;
    assert!(num >= 3); // 3 separate components
    assert!(labels[0] > 0);
    assert!(labels[2] > 0);
    assert_ne!(labels[0], labels[2]);
    Ok(())
}

// ============================================================================
// POPPERIAN FALSIFICATION: Adversarial edge cases (GH #152)
// ============================================================================

#[test]
fn test_falsify_conv2d_1x1_image() {
    let image = vec![42.0_f32];
    let kernel = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0_f32];
    let out = conv2d(&image, 1, 1, &kernel, 3, 3, BorderMode::Zero).expect("ok");
    assert!((out[0] - 42.0).abs() < 1e-5, "1×1 identity conv: {}", out[0]);
}

#[test]
fn test_falsify_conv2d_large_kernel() {
    let image = vec![1.0_f32; 25]; // 5×5 constant
    let kernel = vec![1.0 / 25.0; 25]; // 5×5 averaging
    let out = conv2d(&image, 5, 5, &kernel, 5, 5, BorderMode::Clamp).expect("ok");
    // Interior pixel: constant image → same value regardless of kernel
    assert!((out[12] - 1.0).abs() < 1e-3, "Large kernel on constant: {}", out[12]);
}

#[test]
fn test_falsify_gaussian_blur_1x1() {
    let image = vec![7.0_f32];
    let blurred = gaussian_blur(&image, 1, 1, 1.0).expect("ok");
    assert!((blurred[0] - 7.0).abs() < 1e-3, "1×1 blur: {}", blurred[0]);
}

#[test]
fn test_falsify_canny_1x1_no_edges() {
    let image = vec![0.5_f32];
    let edges = canny(&image, 1, 1, 1.0, 0.05, 0.15).expect("ok");
    assert!(edges[0] < 0.5, "1×1 image should have no edges");
}

#[test]
fn test_falsify_canny_binary_output() {
    // Verify Canny output is truly binary (0 or 1)
    let w = 20;
    let h = 20;
    let mut image = vec![0.0_f32; w * h];
    for y in 0..h {
        for x in w / 2..w {
            image[y * w + x] = 1.0;
        }
    }
    let edges = canny(&image, w, h, 1.0, 0.05, 0.15).expect("ok");
    for (i, &v) in edges.iter().enumerate() {
        assert!(v.abs() < 1e-5 || (v - 1.0).abs() < 1e-5, "Canny output not binary at {i}: {v}");
    }
}

#[test]
fn test_falsify_resize_same_size() -> Result<(), Box<dyn std::error::Error>> {
    let image: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
    for interp in [Interpolation::Nearest, Interpolation::Bilinear] {
        let result = resize(&image, 4, 4, 4, 4, interp)?;
        for i in 0..16 {
            assert!(
                (result[i] - image[i]).abs() < 0.01,
                "Same-size resize changed pixel {i}: {} → {} ({interp:?})",
                image[i],
                result[i]
            );
        }
    }
    Ok(())
}

#[test]
fn test_falsify_resize_1x1() -> Result<(), Box<dyn std::error::Error>> {
    let image = vec![0.42_f32];
    for interp in [
        Interpolation::Nearest,
        Interpolation::Bilinear,
        Interpolation::Bicubic,
        Interpolation::Lanczos,
    ] {
        let result = resize(&image, 1, 1, 1, 1, interp)?;
        assert!((result[0] - 0.42).abs() < 0.01, "1×1 resize ({interp:?}): {}", result[0]);
    }
    Ok(())
}

#[test]
fn test_falsify_morphology_constant_unchanged() -> Result<(), Box<dyn std::error::Error>> {
    let se = vec![1.0_f32; 9];
    let image = vec![0.5_f32; 25];
    let d = dilate(&image, 5, 5, &se, 3, 3)?;
    let e = erode(&image, 5, 5, &se, 3, 3)?;
    for i in 0..25 {
        assert!((d[i] - 0.5).abs() < 1e-6, "Dilate changed constant at {i}");
        assert!((e[i] - 0.5).abs() < 1e-6, "Erode changed constant at {i}");
    }
    Ok(())
}

#[test]
fn test_falsify_histogram_single_value() -> Result<(), Box<dyn std::error::Error>> {
    let image = vec![0.0_f32; 100]; // All zeros
    let hist = histogram(&image, 10, 10, 10)?;
    assert_eq!(hist[0], 100, "All-zero image: all in bin 0");
    for i in 1..10 {
        assert_eq!(hist[i], 0, "Bin {i} should be empty");
    }
    Ok(())
}

#[test]
fn test_falsify_connected_components_single_pixel() -> Result<(), Box<dyn std::error::Error>> {
    let image = vec![1.0_f32];
    let labels = connected_components(&image, 1, 1)?;
    assert_eq!(labels.len(), 1);
    assert!(labels[0] > 0);
    Ok(())
}

#[test]
fn test_falsify_rgb_to_gray_preserves_zero() -> Result<(), Box<dyn std::error::Error>> {
    let rgb = vec![0.0_f32; 12]; // 4 black pixels
    let gray = rgb_to_gray(&rgb, 4, 1)?;
    for &v in &gray {
        assert!(v.abs() < 1e-7, "Black should map to zero: {v}");
    }
    Ok(())
}

#[test]
fn test_falsify_hsv_extreme_saturation() -> Result<(), Box<dyn std::error::Error>> {
    // Pure saturated colors with very low value
    let rgb = vec![0.01, 0.0, 0.0_f32]; // very dark red
    let hsv = rgb_to_hsv(&rgb, 1, 1)?;
    let recovered = hsv_to_rgb(&hsv, 1, 1)?;
    for i in 0..3 {
        assert!(
            (rgb[i] - recovered[i]).abs() < 1e-3,
            "Dark red roundtrip at ch{i}: {} → {}",
            rgb[i],
            recovered[i]
        );
    }
    Ok(())
}
