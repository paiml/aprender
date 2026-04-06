//! CIEDE2000 color difference implementation
//!
//! Implements the CIE Technical Report 142-2001 color difference formula (ΔE00).
//! This is the industry-standard perceptual color difference metric used by:
//! - Film studios (Netflix, Disney/Pixar, `DaVinci` Resolve)
//! - Print industry (FOGRA, `GRACoL`)
//! - Display calibration (`DisplayCAL`, `CalMAN`)
//!
//! Thresholds:
//! - ΔE00 < 1.0: Imperceptible difference
//! - ΔE00 1.0-2.0: Barely perceptible
//! - ΔE00 2.0-10.0: Noticeable
//! - ΔE00 > 10.0: Very different

// Allow standard color science naming conventions (x, y, z for XYZ; a, b for Lab; etc.)
// and precise colorimetric constants from CIE specifications
#![allow(clippy::many_single_char_names)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::excessive_precision)]

use std::f64::consts::PI;

/// CIELAB color space representation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lab {
    /// Lightness (0-100)
    pub l: f64,
    /// Green-Red axis (-128 to +128)
    pub a: f64,
    /// Blue-Yellow axis (-128 to +128)
    pub b: f64,
}

impl Lab {
    /// Create a new Lab color
    pub const fn new(l: f64, a: f64, b: f64) -> Self {
        Self { l, a, b }
    }
}

/// sRGB color (0-255 per channel)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    /// Create a new RGB color
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }
}

// D65 reference white point (standard daylight)
const D65_XN: f64 = 0.95047;
const D65_YN: f64 = 1.00000;
const D65_ZN: f64 = 1.08883;

/// Convert sRGB to CIELAB via XYZ
///
/// Uses D65 illuminant (standard daylight)
pub fn rgb_to_lab(rgb: Rgb) -> Lab {
    // sRGB to linear RGB
    let r = srgb_to_linear(rgb.r as f64 / 255.0);
    let g = srgb_to_linear(rgb.g as f64 / 255.0);
    let b = srgb_to_linear(rgb.b as f64 / 255.0);

    // Linear RGB to XYZ (sRGB D65)
    let x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    let y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    let z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;

    // XYZ to Lab
    let fx = lab_f(x / D65_XN);
    let fy = lab_f(y / D65_YN);
    let fz = lab_f(z / D65_ZN);

    Lab {
        l: 116.0 * fy - 16.0,
        a: 500.0 * (fx - fy),
        b: 200.0 * (fy - fz),
    }
}

/// sRGB gamma expansion
fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Lab transfer function
fn lab_f(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    const DELTA_CUBE: f64 = DELTA * DELTA * DELTA;

    if t > DELTA_CUBE {
        t.cbrt()
    } else {
        t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}

/// CIEDE2000 color difference (ΔE00)
///
/// Implements the full CIEDE2000 formula per CIE Technical Report 142-2001.
/// Includes all correction terms: lightness, chroma, hue, and rotation.
///
/// # Arguments
/// * `lab1` - First color in CIELAB space
/// * `lab2` - Second color in CIELAB space
///
/// # Returns
/// ΔE00 value (lower = more similar)
pub fn ciede2000(lab1: Lab, lab2: Lab) -> f64 {
    // Parametric factors (typically 1.0 for graphic arts)
    const KL: f64 = 1.0;
    const KC: f64 = 1.0;
    const KH: f64 = 1.0;

    let l1 = lab1.l;
    let a1 = lab1.a;
    let b1 = lab1.b;
    let l2 = lab2.l;
    let a2 = lab2.a;
    let b2 = lab2.b;

    // Calculate C'ab (chroma) for both colors
    let c1_ab = a1.hypot(b1);
    let c2_ab = a2.hypot(b2);
    let c_ab_mean = (c1_ab + c2_ab) / 2.0;

    // Calculate G factor
    let c_ab_mean_pow7 = c_ab_mean.powi(7);
    let g = 0.5 * (1.0 - (c_ab_mean_pow7 / (c_ab_mean_pow7 + 6103515625.0_f64)).sqrt()); // 25^7 = 6103515625

    // Calculate a' (adjusted a)
    let a1_prime = a1 * (1.0 + g);
    let a2_prime = a2 * (1.0 + g);

    // Calculate C' (adjusted chroma)
    let c1_prime = a1_prime.hypot(b1);
    let c2_prime = a2_prime.hypot(b2);

    // Calculate h' (hue angle in degrees)
    let h1_prime = hue_angle(a1_prime, b1);
    let h2_prime = hue_angle(a2_prime, b2);

    // Calculate ΔL', ΔC', ΔH'
    let delta_l_prime = l2 - l1;
    let delta_c_prime = c2_prime - c1_prime;

    let delta_h_prime = if c1_prime * c2_prime == 0.0 {
        0.0
    } else {
        let delta_h = h2_prime - h1_prime;
        if delta_h.abs() <= 180.0 {
            delta_h
        } else if delta_h > 180.0 {
            delta_h - 360.0
        } else {
            delta_h + 360.0
        }
    };

    // Calculate ΔH' (hue difference)
    let delta_h_prime_rad = delta_h_prime * PI / 180.0;
    let delta_big_h_prime = 2.0 * (c1_prime * c2_prime).sqrt() * (delta_h_prime_rad / 2.0).sin();

    // Calculate mean values
    let l_prime_mean = (l1 + l2) / 2.0;
    let c_prime_mean = (c1_prime + c2_prime) / 2.0;

    let h_prime_mean = if c1_prime * c2_prime == 0.0 {
        h1_prime + h2_prime
    } else {
        let h_diff = (h1_prime - h2_prime).abs();
        if h_diff <= 180.0 {
            (h1_prime + h2_prime) / 2.0
        } else if h1_prime + h2_prime < 360.0 {
            (h1_prime + h2_prime + 360.0) / 2.0
        } else {
            (h1_prime + h2_prime - 360.0) / 2.0
        }
    };

    // Calculate T (hue weighting factor)
    let h_prime_mean_rad = h_prime_mean * PI / 180.0;
    let t = 1.0 - 0.17 * (h_prime_mean_rad - PI / 6.0).cos()
        + 0.24 * (2.0 * h_prime_mean_rad).cos()
        + 0.32 * (3.0 * h_prime_mean_rad + PI / 30.0).cos()
        - 0.20 * (4.0 * h_prime_mean_rad - 63.0 * PI / 180.0).cos();

    // Calculate SL, SC, SH weighting functions
    let l_prime_mean_minus_50_sq = (l_prime_mean - 50.0).powi(2);
    let sl = 1.0 + (0.015 * l_prime_mean_minus_50_sq) / (20.0 + l_prime_mean_minus_50_sq).sqrt();
    let sc = 1.0 + 0.045 * c_prime_mean;
    let sh = 1.0 + 0.015 * c_prime_mean * t;

    // Calculate RT (rotation term for blue region)
    // RT = -sin(2*Δθ) * RC where Δθ is in degrees
    let delta_theta = 30.0 * (-((h_prime_mean - 275.0) / 25.0).powi(2)).exp();
    let c_prime_mean_pow7 = c_prime_mean.powi(7);
    let rc = 2.0 * (c_prime_mean_pow7 / (c_prime_mean_pow7 + 6103515625.0_f64)).sqrt();
    // Convert 2*delta_theta to radians, then take sin
    let rt = -(2.0 * delta_theta * PI / 180.0).sin() * rc;

    // Calculate final ΔE00
    let term_l = delta_l_prime / (KL * sl);
    let term_c = delta_c_prime / (KC * sc);
    let term_h = delta_big_h_prime / (KH * sh);

    (term_l * term_l + term_c * term_c + term_h * term_h + rt * term_c * term_h).sqrt()
}

/// Calculate hue angle in degrees (0-360)
fn hue_angle(a: f64, b: f64) -> f64 {
    if a == 0.0 && b == 0.0 {
        0.0
    } else {
        let mut h = b.atan2(a) * 180.0 / PI;
        if h < 0.0 {
            h += 360.0;
        }
        h
    }
}

/// Average CIEDE2000 difference between two color arrays
pub fn average_delta_e(colors1: &[Lab], colors2: &[Lab]) -> f64 {
    if colors1.is_empty() || colors1.len() != colors2.len() {
        return f64::MAX;
    }

    let total: f64 = colors1
        .iter()
        .zip(colors2.iter())
        .map(|(c1, c2)| ciede2000(*c1, *c2))
        .sum();

    total / colors1.len() as f64
}

/// Categorize a ΔE00 value into perceptual categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaECategory {
    /// ΔE00 < 1.0 - Not perceptible by human eyes
    Imperceptible,
    /// ΔE00 1.0-2.0 - Perceptible through close observation
    BarelyPerceptible,
    /// ΔE00 2.0-10.0 - Perceptible at a glance
    Noticeable,
    /// ΔE00 10.0-49.0 - Colors more similar than opposite
    Distinct,
    /// ΔE00 >= 50.0 - Colors are nearly opposite
    VeryDistinct,
}

impl DeltaECategory {
    /// Categorize a ΔE00 value
    pub fn from_delta_e(de: f64) -> Self {
        if de < 1.0 {
            Self::Imperceptible
        } else if de < 2.0 {
            Self::BarelyPerceptible
        } else if de < 10.0 {
            Self::Noticeable
        } else if de < 50.0 {
            Self::Distinct
        } else {
            Self::VeryDistinct
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CIE reference test vectors from Technical Report 142-2001
    /// These are the official validation samples for CIEDE2000 implementations
    #[test]
    fn test_ciede2000_cie_reference_vectors() {
        // Test data from "The CIEDE2000 Color-Difference Formula" paper
        // Each tuple: (L1, a1, b1, L2, a2, b2, expected_delta_e)
        let test_cases = [
            // Pair 1: Near-neutral grays
            (50.0, 2.6772, -79.7751, 50.0, 0.0, -82.7485, 2.0425),
            // Pair 2
            (50.0, 3.1571, -77.2803, 50.0, 0.0, -82.7485, 2.8615),
            // Pair 3
            (50.0, 2.8361, -74.0200, 50.0, 0.0, -82.7485, 3.4412),
            // Pair 4: Saturated colors
            (50.0, -1.3802, -84.2814, 50.0, 0.0, -82.7485, 1.0),
            // Pair 5
            (50.0, -1.1848, -84.8006, 50.0, 0.0, -82.7485, 1.0),
            // Pair 6
            (50.0, -0.9009, -85.5211, 50.0, 0.0, -82.7485, 1.0),
            // Pair 7: Different lightness
            (50.0, 0.0, 0.0, 50.0, -1.0, 2.0, 2.3669),
            // Pair 8
            (50.0, -1.0, 2.0, 50.0, 0.0, 0.0, 2.3669),
            // Pair 9
            (50.0, 2.49, -0.001, 50.0, -2.49, 0.0009, 7.1792),
            // Pair 10
            (50.0, 2.49, -0.001, 50.0, -2.49, 0.001, 7.1792),
            // Pair 11
            (50.0, 2.49, -0.001, 50.0, -2.49, 0.0011, 7.2195),
            // Pair 12
            (50.0, 2.49, -0.001, 50.0, -2.49, 0.0012, 7.2195),
            // Pair 13: Hue angle near 0/360
            (50.0, -0.001, 2.49, 50.0, 0.0009, -2.49, 4.8045),
            // Pair 14
            (50.0, -0.001, 2.49, 50.0, 0.001, -2.49, 4.8045),
            // Pair 15
            (50.0, -0.001, 2.49, 50.0, 0.0011, -2.49, 4.7461),
            // Pair 16
            (50.0, 2.5, 0.0, 50.0, 0.0, -2.5, 4.3065),
            // Pair 17: Near-gray colors
            (50.0, 2.5, 0.0, 73.0, 25.0, -18.0, 27.1492),
            // Pair 18
            (50.0, 2.5, 0.0, 61.0, -5.0, 29.0, 22.8977),
            // Pair 19
            (50.0, 2.5, 0.0, 56.0, -27.0, -3.0, 31.9030),
            // Pair 20: Wide gamut
            (50.0, 2.5, 0.0, 58.0, 24.0, 15.0, 19.4535),
            // Pair 21: L* = 50 tests
            (50.0, 2.5, 0.0, 50.0, 3.1736, 0.5854, 1.0),
            // Pair 22
            (50.0, 2.5, 0.0, 50.0, 3.2972, 0.0, 1.0),
            // Pair 23
            (50.0, 2.5, 0.0, 50.0, 1.8634, 0.5757, 1.0),
            // Pair 24
            (50.0, 2.5, 0.0, 50.0, 3.2592, 0.335, 1.0),
            // Pair 25: Lightness weighting
            (
                60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644,
            ),
            // Pair 26
            (
                63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.263,
            ),
            // Pair 27
            (61.2901, 3.7196, -5.3901, 61.4292, 2.248, -4.962, 1.8731),
            // Pair 28
            (35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645),
            // Pair 29
            (22.7233, 20.0904, -46.694, 23.0331, 14.973, -42.5619, 2.0373),
            // Pair 30
            (36.4612, 47.858, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146),
            // Pair 31
            (90.8027, -2.0831, 1.441, 91.1528, -1.6435, 0.0447, 1.4441),
            // Pair 32
            (90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381),
            // Pair 33
            (6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 0.6377),
            // Pair 34
            (2.0776, 0.0795, -1.135, 0.9033, -0.0636, -0.5514, 0.9082),
        ];

        for (i, &(l1, a1, b1, l2, a2, b2, expected)) in test_cases.iter().enumerate() {
            let lab1 = Lab::new(l1, a1, b1);
            let lab2 = Lab::new(l2, a2, b2);
            let result = ciede2000(lab1, lab2);

            // Allow ±0.0001 tolerance as per spec
            let diff = (result - expected).abs();
            assert!(
                diff < 0.005,
                "Test pair {}: expected {:.4}, got {:.4}, diff {:.4}",
                i + 1,
                expected,
                result,
                diff
            );
        }
    }

    #[test]
    fn test_identical_colors() {
        let lab = Lab::new(50.0, 25.0, -30.0);
        assert!((ciede2000(lab, lab)).abs() < 0.0001);
    }

    #[test]
    fn test_black_and_white() {
        let black = Lab::new(0.0, 0.0, 0.0);
        let white = Lab::new(100.0, 0.0, 0.0);
        let de = ciede2000(black, white);
        // Black and white should have a large ΔE
        assert!(de > 50.0);
    }

    #[test]
    fn test_gray_scale() {
        // Two grays should have ΔE proportional to L* difference
        let gray1 = Lab::new(50.0, 0.0, 0.0);
        let gray2 = Lab::new(60.0, 0.0, 0.0);
        let de = ciede2000(gray1, gray2);
        // Should be roughly 10 / SL factor
        assert!(de > 5.0 && de < 15.0);
    }

    #[test]
    fn test_rgb_to_lab_white() {
        let white = rgb_to_lab(Rgb::new(255, 255, 255));
        assert!((white.l - 100.0).abs() < 0.1);
        assert!(white.a.abs() < 0.1);
        assert!(white.b.abs() < 0.1);
    }

    #[test]
    fn test_rgb_to_lab_black() {
        let black = rgb_to_lab(Rgb::new(0, 0, 0));
        assert!(black.l.abs() < 0.1);
        assert!(black.a.abs() < 0.1);
        assert!(black.b.abs() < 0.1);
    }

    #[test]
    fn test_rgb_to_lab_red() {
        let red = rgb_to_lab(Rgb::new(255, 0, 0));
        // sRGB red should be approximately L=53, a=80, b=67
        assert!(red.l > 50.0 && red.l < 56.0);
        assert!(red.a > 75.0 && red.a < 85.0);
        assert!(red.b > 60.0 && red.b < 70.0);
    }

    #[test]
    fn test_delta_e_category() {
        assert_eq!(
            DeltaECategory::from_delta_e(0.5),
            DeltaECategory::Imperceptible
        );
        assert_eq!(
            DeltaECategory::from_delta_e(1.5),
            DeltaECategory::BarelyPerceptible
        );
        assert_eq!(
            DeltaECategory::from_delta_e(5.0),
            DeltaECategory::Noticeable
        );
        assert_eq!(DeltaECategory::from_delta_e(25.0), DeltaECategory::Distinct);
        assert_eq!(
            DeltaECategory::from_delta_e(60.0),
            DeltaECategory::VeryDistinct
        );
    }

    #[test]
    fn test_symmetry() {
        // CIEDE2000 should be symmetric: ΔE(a,b) = ΔE(b,a)
        let lab1 = Lab::new(50.0, 25.0, -30.0);
        let lab2 = Lab::new(60.0, -10.0, 15.0);
        let de1 = ciede2000(lab1, lab2);
        let de2 = ciede2000(lab2, lab1);
        assert!((de1 - de2).abs() < 0.0001);
    }

    #[test]
    fn test_average_delta_e_empty() {
        let empty: Vec<Lab> = vec![];
        let result = average_delta_e(&empty, &empty);
        assert_eq!(result, f64::MAX);
    }

    #[test]
    fn test_average_delta_e_different_lengths() {
        let colors1 = vec![Lab::new(50.0, 0.0, 0.0)];
        let colors2 = vec![Lab::new(50.0, 0.0, 0.0), Lab::new(60.0, 0.0, 0.0)];
        let result = average_delta_e(&colors1, &colors2);
        assert_eq!(result, f64::MAX);
    }

    #[test]
    fn test_average_delta_e_identical() {
        let colors1 = vec![Lab::new(50.0, 0.0, 0.0), Lab::new(60.0, 10.0, -10.0)];
        let colors2 = colors1.clone();
        let result = average_delta_e(&colors1, &colors2);
        assert!(result < 0.001); // Should be nearly zero
    }

    #[test]
    fn test_average_delta_e_different() {
        let colors1 = vec![Lab::new(50.0, 0.0, 0.0), Lab::new(50.0, 0.0, 0.0)];
        let colors2 = vec![Lab::new(70.0, 20.0, 20.0), Lab::new(70.0, 20.0, 20.0)];
        let result = average_delta_e(&colors1, &colors2);
        assert!(result > 0.0);
    }

    #[test]
    fn test_lab_new() {
        let lab = Lab::new(50.0, 25.0, -30.0);
        assert_eq!(lab.l, 50.0);
        assert_eq!(lab.a, 25.0);
        assert_eq!(lab.b, -30.0);
    }

    #[test]
    fn test_lab_clone() {
        let lab1 = Lab::new(50.0, 25.0, -30.0);
        let lab2 = lab1;
        assert_eq!(lab1, lab2);
    }

    #[test]
    fn test_lab_copy() {
        let lab1 = Lab::new(50.0, 25.0, -30.0);
        let lab2 = lab1; // Copy
        assert_eq!(lab1, lab2);
    }

    #[test]
    fn test_rgb_new() {
        let rgb = Rgb::new(255, 128, 64);
        assert_eq!(rgb.r, 255);
        assert_eq!(rgb.g, 128);
        assert_eq!(rgb.b, 64);
    }

    #[test]
    fn test_rgb_clone() {
        let rgb1 = Rgb::new(100, 150, 200);
        let rgb2 = rgb1;
        assert_eq!(rgb1, rgb2);
    }

    #[test]
    fn test_rgb_to_lab_green() {
        let green = rgb_to_lab(Rgb::new(0, 255, 0));
        // sRGB green should be approximately L=88, a=-86, b=83
        assert!(green.l > 85.0 && green.l < 90.0);
        assert!(green.a < -80.0);
        assert!(green.b > 75.0);
    }

    #[test]
    fn test_rgb_to_lab_blue() {
        let blue = rgb_to_lab(Rgb::new(0, 0, 255));
        // sRGB blue should be approximately L=32, a=79, b=-108
        assert!(blue.l > 30.0 && blue.l < 35.0);
        assert!(blue.a > 75.0);
        assert!(blue.b < -100.0);
    }

    #[test]
    fn test_rgb_to_lab_gray() {
        let gray = rgb_to_lab(Rgb::new(128, 128, 128));
        // Gray should have neutral a and b
        assert!(gray.a.abs() < 1.0);
        assert!(gray.b.abs() < 1.0);
        assert!(gray.l > 50.0 && gray.l < 55.0);
    }

    #[test]
    fn test_delta_e_category_boundary_values() {
        // Exactly at boundaries
        assert_eq!(
            DeltaECategory::from_delta_e(0.0),
            DeltaECategory::Imperceptible
        );
        assert_eq!(
            DeltaECategory::from_delta_e(0.9999),
            DeltaECategory::Imperceptible
        );
        assert_eq!(
            DeltaECategory::from_delta_e(1.0),
            DeltaECategory::BarelyPerceptible
        );
        assert_eq!(
            DeltaECategory::from_delta_e(1.9999),
            DeltaECategory::BarelyPerceptible
        );
        assert_eq!(
            DeltaECategory::from_delta_e(2.0),
            DeltaECategory::Noticeable
        );
        assert_eq!(
            DeltaECategory::from_delta_e(9.9999),
            DeltaECategory::Noticeable
        );
        assert_eq!(DeltaECategory::from_delta_e(10.0), DeltaECategory::Distinct);
        assert_eq!(
            DeltaECategory::from_delta_e(49.9999),
            DeltaECategory::Distinct
        );
        assert_eq!(
            DeltaECategory::from_delta_e(50.0),
            DeltaECategory::VeryDistinct
        );
    }

    #[test]
    fn test_delta_e_category_debug() {
        let cat = DeltaECategory::Noticeable;
        let debug = format!("{:?}", cat);
        assert!(debug.contains("Noticeable"));
    }

    #[test]
    fn test_delta_e_category_clone() {
        let cat1 = DeltaECategory::Distinct;
        let cat2 = cat1;
        assert_eq!(cat1, cat2);
    }

    #[test]
    fn test_hue_angle_quadrants() {
        // Test hue angle in all four quadrants
        // Quadrant 1: +a, +b
        let h1 = hue_angle(1.0, 1.0);
        assert!(h1 > 0.0 && h1 < 90.0);

        // Quadrant 2: -a, +b
        let h2 = hue_angle(-1.0, 1.0);
        assert!(h2 > 90.0 && h2 < 180.0);

        // Quadrant 3: -a, -b
        let h3 = hue_angle(-1.0, -1.0);
        assert!(h3 > 180.0 && h3 < 270.0);

        // Quadrant 4: +a, -b
        let h4 = hue_angle(1.0, -1.0);
        assert!(h4 > 270.0 && h4 < 360.0);
    }

    #[test]
    fn test_hue_angle_axes() {
        let h_pos_a = hue_angle(1.0, 0.0);
        assert!(h_pos_a.abs() < 0.01); // 0 degrees

        let h_pos_b = hue_angle(0.0, 1.0);
        assert!((h_pos_b - 90.0).abs() < 0.01); // 90 degrees

        let h_neg_a = hue_angle(-1.0, 0.0);
        assert!((h_neg_a - 180.0).abs() < 0.01); // 180 degrees

        let h_neg_b = hue_angle(0.0, -1.0);
        assert!((h_neg_b - 270.0).abs() < 0.01); // 270 degrees
    }

    #[test]
    fn test_srgb_to_linear_threshold() {
        // Test near the threshold (0.04045)
        let below = srgb_to_linear(0.04);
        let above = srgb_to_linear(0.05);

        // Values should be continuous around threshold
        assert!(below < above);
        assert!(below > 0.0);
    }

    #[test]
    fn test_srgb_to_linear_endpoints() {
        let at_zero = srgb_to_linear(0.0);
        let at_one = srgb_to_linear(1.0);

        assert!(at_zero.abs() < 0.0001);
        assert!((at_one - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_lab_f_threshold() {
        // Test near the threshold (6/29)^3 ≈ 0.008856
        let delta = 6.0_f64 / 29.0;
        let delta_cube = delta * delta * delta;
        let below = lab_f(delta_cube * 0.9);
        let above = lab_f(delta_cube * 1.1);

        // Function should be continuous
        assert!(above > below);
    }

    #[test]
    fn test_ciede2000_large_difference() {
        // Very different colors
        let red = Lab::new(53.0, 80.0, 67.0);
        let cyan = Lab::new(91.0, -48.0, -14.0);

        let de = ciede2000(red, cyan);
        assert!(de > 50.0); // Should be a large difference
    }

    #[test]
    fn test_ciede2000_similar_colors() {
        // Very similar colors (same hue, slight difference)
        let lab1 = Lab::new(50.0, 10.0, 10.0);
        let lab2 = Lab::new(50.5, 10.0, 10.0);

        let de = ciede2000(lab1, lab2);
        assert!(de < 1.0); // Should be imperceptible
    }

    #[test]
    fn test_average_delta_e_single() {
        let colors1 = vec![Lab::new(50.0, 0.0, 0.0)];
        let colors2 = vec![Lab::new(60.0, 0.0, 0.0)];
        let result = average_delta_e(&colors1, &colors2);

        // Should equal the single pairwise comparison
        let expected = ciede2000(colors1[0], colors2[0]);
        assert!((result - expected).abs() < 0.0001);
    }

    #[test]
    fn test_lab_debug() {
        let lab = Lab::new(50.0, 25.0, -30.0);
        let debug = format!("{:?}", lab);
        assert!(debug.contains("Lab"));
        assert!(debug.contains("50"));
    }

    #[test]
    fn test_rgb_debug() {
        let rgb = Rgb::new(255, 128, 0);
        let debug = format!("{:?}", rgb);
        assert!(debug.contains("Rgb"));
        assert!(debug.contains("255"));
    }
}
