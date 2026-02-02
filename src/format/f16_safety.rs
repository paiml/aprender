//! F16 Safety Constants and Helpers
//!
//! Centralized location for f16 (half-precision float) safety handling.
//! Prevents GH-186 class bugs where NaN/Inf/subnormal f16 values propagate
//! through dequantization and corrupt model weights.
//!
//! # IEEE 754 Half-Precision (f16) Overview
//!
//! ```text
//! Normal range:      [6.1e-5, 65504]  (positive) or [-65504, -6.1e-5] (negative)
//! Subnormal range:   (0, 6.1e-5)      (positive) or (-6.1e-5, 0) (negative)
//! Special values:    ±0, ±Inf, NaN
//! ```
//!
//! The smallest positive NORMAL f16 is 2^-14 ≈ 6.103515625e-5.
//! Subnormals (denormalized numbers) are smaller but lose precision.
//!
//! # Usage
//!
//! ```rust,ignore
//! use aprender::format::f16_safety::{F16_MIN_NORMAL, safe_f16_scale};
//!
//! // Clamp a scale factor read from GGUF/APR quantized weights
//! let raw_scale = f16_to_f32(scale_bits);
//! let safe_scale = safe_f16_scale(raw_scale);
//! ```

/// Minimum positive normal f16 value: 2^(-14) ≈ 6.1e-5
///
/// Values below this threshold are either:
/// - Subnormal (lose precision, may underflow)
/// - Zero (safe, preserved)
///
/// Used to clamp scale factors in GGUF/APR quantization to prevent NaN propagation.
///
/// # IEEE 754 Reference
///
/// The exact value is 2^(-14) = 0.00006103515625, but we use 6.1e-5 as a
/// conservative threshold that accounts for floating-point representation errors.
///
/// # GH-186 Context
///
/// This constant was introduced to fix GH-186 where malformed f16 scale factors
/// in GGUF files caused NaN propagation through dequantization, corrupting
/// model weights on round-trip conversion.
pub const F16_MIN_NORMAL: f32 = 6.1e-5;

/// Smallest positive normal f16 bit pattern: 0x0400
///
/// This is the IEEE 754 half-precision representation of 2^(-14).
/// Bit pattern: sign=0, exponent=00001 (biased 1), mantissa=0000000000
///
/// Used for boundary testing of f16 handling.
pub const F16_SMALLEST_NORMAL_BITS: u16 = 0x0400;

/// Clamp an f16-derived scale factor to a safe range.
///
/// Returns 0.0 for NaN, Inf, or subnormal values to prevent propagation
/// of invalid data through dequantization.
///
/// # Arguments
///
/// * `val` - The f32 value converted from f16 scale bits
///
/// # Returns
///
/// * `0.0` if val is NaN, infinite, or below [`F16_MIN_NORMAL`]
/// * `val` otherwise (preserves the original value)
///
/// # Example
///
/// ```rust,ignore
/// let scale_bits: u16 = read_u16_le(data);
/// let scale_raw = f16_to_f32(scale_bits);
/// let scale = safe_f16_scale(scale_raw);  // Safe to use in multiplication
/// ```
#[inline]
#[must_use]
pub fn safe_f16_scale(val: f32) -> f32 {
    if val.is_nan() || val.is_infinite() || val.abs() < F16_MIN_NORMAL {
        0.0
    } else {
        val
    }
}

/// Check if an f16-derived value is safe for use as a scale factor.
///
/// # Returns
///
/// * `true` if the value is finite and >= [`F16_MIN_NORMAL`]
/// * `false` for NaN, Inf, subnormal, or zero values
#[inline]
#[must_use]
pub fn is_safe_f16_scale(val: f32) -> bool {
    val.is_finite() && val.abs() >= F16_MIN_NORMAL
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // F16_MIN_NORMAL Tests
    // ========================================================================

    #[test]
    fn test_f16_min_normal_value() {
        // 2^(-14) = 0.00006103515625
        // Our constant 6.1e-5 = 0.000061 is slightly below for safety margin
        assert!(F16_MIN_NORMAL > 0.0);
        assert!(F16_MIN_NORMAL < 1e-4);
        assert!((F16_MIN_NORMAL - 6.1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_f16_smallest_normal_bits() {
        // 0x0400 = 0b0000_0100_0000_0000
        // sign=0, exponent=00001, mantissa=0
        assert_eq!(F16_SMALLEST_NORMAL_BITS, 0x0400);
    }

    // ========================================================================
    // safe_f16_scale Tests
    // ========================================================================

    #[test]
    fn test_safe_f16_scale_normal_preserved() {
        assert_eq!(safe_f16_scale(1.0), 1.0);
        assert_eq!(safe_f16_scale(0.5), 0.5);
        assert_eq!(safe_f16_scale(0.001), 0.001);
        assert_eq!(safe_f16_scale(F16_MIN_NORMAL), F16_MIN_NORMAL);
    }

    #[test]
    fn test_safe_f16_scale_nan_clamped() {
        assert_eq!(safe_f16_scale(f32::NAN), 0.0);
    }

    #[test]
    fn test_safe_f16_scale_inf_clamped() {
        assert_eq!(safe_f16_scale(f32::INFINITY), 0.0);
        assert_eq!(safe_f16_scale(f32::NEG_INFINITY), 0.0);
    }

    #[test]
    fn test_safe_f16_scale_subnormal_clamped() {
        // Values below F16_MIN_NORMAL are clamped
        assert_eq!(safe_f16_scale(1e-6), 0.0);
        assert_eq!(safe_f16_scale(1e-8), 0.0);
        assert_eq!(safe_f16_scale(F16_MIN_NORMAL * 0.5), 0.0);
    }

    #[test]
    fn test_safe_f16_scale_zero_preserved() {
        // Zero is clamped (abs(0) < F16_MIN_NORMAL)
        assert_eq!(safe_f16_scale(0.0), 0.0);
        assert_eq!(safe_f16_scale(-0.0), 0.0);
    }

    #[test]
    fn test_safe_f16_scale_negative_normal_preserved() {
        assert_eq!(safe_f16_scale(-1.0), -1.0);
        assert_eq!(safe_f16_scale(-0.5), -0.5);
    }

    #[test]
    fn test_safe_f16_scale_boundary() {
        // Just above threshold - preserved
        let just_above = F16_MIN_NORMAL * 1.01;
        assert_eq!(safe_f16_scale(just_above), just_above);

        // Just below threshold - clamped
        let just_below = F16_MIN_NORMAL * 0.99;
        assert_eq!(safe_f16_scale(just_below), 0.0);
    }

    // ========================================================================
    // is_safe_f16_scale Tests
    // ========================================================================

    #[test]
    fn test_is_safe_f16_scale_normal() {
        assert!(is_safe_f16_scale(1.0));
        assert!(is_safe_f16_scale(0.5));
        assert!(is_safe_f16_scale(F16_MIN_NORMAL));
        assert!(is_safe_f16_scale(-1.0));
    }

    #[test]
    fn test_is_safe_f16_scale_unsafe_values() {
        assert!(!is_safe_f16_scale(f32::NAN));
        assert!(!is_safe_f16_scale(f32::INFINITY));
        assert!(!is_safe_f16_scale(f32::NEG_INFINITY));
        assert!(!is_safe_f16_scale(0.0));
        assert!(!is_safe_f16_scale(1e-6));
    }

    // ========================================================================
    // P3: Boundary test for 0x0400 (smallest normal f16)
    // ========================================================================

    #[test]
    fn test_gh186_boundary_0x0400_smallest_normal() {
        // 0x0400 is the smallest positive normal f16
        // When converted to f32, it should be >= F16_MIN_NORMAL and thus preserved

        // Simulate f16_to_f32 for 0x0400
        // 0x0400 = sign=0, exp=1 (biased), mantissa=0
        // value = 2^(1-15) * 1.0 = 2^(-14) ≈ 6.1e-5
        let smallest_normal_f32 = 2.0_f32.powi(-14);

        // This value should be at or above our threshold
        assert!(
            smallest_normal_f32 >= F16_MIN_NORMAL,
            "0x0400 ({smallest_normal_f32}) should be >= F16_MIN_NORMAL ({F16_MIN_NORMAL})"
        );

        // And should be preserved by safe_f16_scale
        assert_eq!(
            safe_f16_scale(smallest_normal_f32),
            smallest_normal_f32,
            "Smallest normal f16 should be preserved"
        );
    }

    #[test]
    fn test_gh186_boundary_0x03ff_largest_subnormal() {
        // 0x03FF is the largest positive subnormal f16
        // When converted to f32, it should be < F16_MIN_NORMAL and thus clamped

        // 0x03FF = sign=0, exp=0 (subnormal), mantissa=0x3FF (all 1s)
        // value = 2^(-14) * (0.1111111111)_2 = 2^(-14) * (1 - 2^(-10))
        //       ≈ 6.1e-5 * 0.999 ≈ 6.09e-5
        let largest_subnormal_f32 = 2.0_f32.powi(-14) * (1.0 - 2.0_f32.powi(-10));

        // This value should be below our threshold
        assert!(
            largest_subnormal_f32 < F16_MIN_NORMAL,
            "0x03FF ({largest_subnormal_f32}) should be < F16_MIN_NORMAL ({F16_MIN_NORMAL})"
        );

        // And should be clamped by safe_f16_scale
        assert_eq!(
            safe_f16_scale(largest_subnormal_f32),
            0.0,
            "Largest subnormal f16 should be clamped to 0.0"
        );
    }
}
