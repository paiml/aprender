//! ULP (Unit in the Last Place) distance utilities for floating-point comparison.
//!
//! Used to verify SIMD and PTX kernels produce results within acceptable
//! tolerance of the scalar reference implementation.

/// Compute the ULP distance between two f32 values.
///
/// Returns the number of representable floats between `a` and `b`.
/// Special cases: if either value is NaN, returns `u32::MAX`.
/// If signs differ and neither is zero, returns `u32::MAX`.
#[must_use]
pub fn ulp_distance(a: f32, b: f32) -> u32 {
    if a.is_nan() || b.is_nan() {
        return u32::MAX;
    }
    if a == b {
        return 0;
    }
    let a_bits = a.to_bits() as i32;
    let b_bits = b.to_bits() as i32;
    // Handle sign mismatch (excluding ±0)
    if (a_bits < 0) != (b_bits < 0) {
        // Both ±0 case already handled by a == b above
        return u32::MAX;
    }
    a_bits.abs_diff(b_bits)
}

/// Assert that two f32 slices are equal within the given ULP tolerance.
///
/// # Panics
///
/// Panics if slices have different lengths or any element pair exceeds
/// the ULP tolerance.
pub fn assert_ulp_eq(a: &[f32], b: &[f32], max_ulp: u32) {
    assert_eq!(
        a.len(),
        b.len(),
        "slice length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        let dist = ulp_distance(va, vb);
        assert!(
            dist <= max_ulp,
            "ULP violation at index {i}: {va} vs {vb} (ULP distance {dist}, max {max_ulp})"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ulp_distance_identical() {
        assert_eq!(ulp_distance(1.0, 1.0), 0);
        assert_eq!(ulp_distance(0.0, 0.0), 0);
        assert_eq!(ulp_distance(-1.0, -1.0), 0);
    }

    #[test]
    fn test_ulp_distance_adjacent() {
        let a: f32 = 1.0;
        let b = f32::from_bits(a.to_bits() + 1);
        assert_eq!(ulp_distance(a, b), 1);
    }

    #[test]
    fn test_ulp_distance_nan() {
        assert_eq!(ulp_distance(f32::NAN, 1.0), u32::MAX);
        assert_eq!(ulp_distance(1.0, f32::NAN), u32::MAX);
        assert_eq!(ulp_distance(f32::NAN, f32::NAN), u32::MAX);
    }

    #[test]
    fn test_ulp_distance_sign_mismatch() {
        assert_eq!(ulp_distance(1.0, -1.0), u32::MAX);
    }

    #[test]
    fn test_ulp_distance_small_gap() {
        let a: f32 = 1.0;
        let b = f32::from_bits(a.to_bits() + 10);
        assert_eq!(ulp_distance(a, b), 10);
    }

    #[test]
    fn test_assert_ulp_eq_passes() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 2.0, 3.0];
        assert_ulp_eq(&a, &b, 0);
    }

    #[test]
    #[should_panic(expected = "ULP violation")]
    fn test_assert_ulp_eq_fails() {
        let a = [1.0f32];
        let b = [2.0f32];
        assert_ulp_eq(&a, &b, 0);
    }

    #[test]
    #[should_panic(expected = "slice length mismatch")]
    fn test_assert_ulp_eq_length_mismatch() {
        assert_ulp_eq(&[1.0], &[1.0, 2.0], 0);
    }

    #[test]
    fn test_ulp_distance_negative_zero() {
        assert_eq!(ulp_distance(0.0, -0.0), 0);
    }
}
