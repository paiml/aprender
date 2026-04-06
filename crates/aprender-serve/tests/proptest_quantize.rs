//! Property Tests for quantize/simd.rs
//!
//! Target: Increase simd.rs coverage from 6.5% to >40%
//! Strategy: Property-based testing of SIMD operations and scalar fallbacks

use proptest::prelude::*;
use realizar::quantize::{
    apply_rope_rotation_simd, f16_to_f32, fused_swiglu_simd, read_f16, softmax_simd,
};

// ============================================================================
// f16_to_f32 Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Property: f16_to_f32 preserves sign
    #[test]
    fn prop_f16_preserves_sign(bits in any::<u16>()) {
        let result = f16_to_f32(bits);

        // Check sign bit (bit 15)
        let sign_bit = (bits >> 15) & 1;
        if !result.is_nan() {
            if sign_bit == 1 {
                prop_assert!(result <= 0.0 || result.is_nan(), "negative f16 should give non-positive f32");
            } else if bits != 0 && (bits & 0x7FFF) != 0 {
                prop_assert!(result >= 0.0 || result.is_nan(), "positive f16 should give non-negative f32");
            }
        }
    }

    /// Property: f16_to_f32 zero produces zero
    #[test]
    fn prop_f16_zero_is_zero(sign in 0u16..=1) {
        let bits = sign << 15;  // Positive or negative zero
        let result = f16_to_f32(bits);
        prop_assert_eq!(result.abs(), 0.0);
    }

    /// Property: f16 special values (inf, nan) handled correctly
    #[test]
    fn prop_f16_special_values(mantissa in 0u16..=0x3FF, sign in 0u16..=1) {
        // Exponent all 1s (0x1F) = special value
        let bits = (sign << 15) | (0x1F << 10) | mantissa;
        let result = f16_to_f32(bits);

        if mantissa == 0 {
            // Infinity
            prop_assert!(result.is_infinite());
        } else {
            // NaN
            prop_assert!(result.is_nan());
        }
    }
}

// ============================================================================
// read_f16 Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Property: read_f16 correctly interprets little-endian bytes
    #[test]
    fn prop_read_f16_little_endian(low_byte in any::<u8>(), high_byte in any::<u8>()) {
        let bytes = [low_byte, high_byte];
        let result = read_f16(&bytes);

        // Should match manual conversion
        let bits = u16::from_le_bytes(bytes);
        let expected = f16_to_f32(bits);

        // Handle NaN specially (NaN != NaN)
        if expected.is_nan() {
            prop_assert!(result.is_nan());
        } else {
            prop_assert_eq!(result, expected);
        }
    }
}

// ============================================================================
// softmax_simd Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Property: softmax output sums to 1.0
    #[test]
    fn prop_softmax_sums_to_one(
        values in prop::collection::vec(-10.0f32..10.0, 1..=100)
    ) {
        let mut x = values;
        softmax_simd(&mut x);

        let sum: f32 = x.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5, "softmax sum {} should be ~1.0", sum);
    }

    /// Property: softmax output is non-negative
    #[test]
    fn prop_softmax_non_negative(
        values in prop::collection::vec(-100.0f32..100.0, 1..=100)
    ) {
        let mut x = values;
        softmax_simd(&mut x);

        for &v in &x {
            prop_assert!(v >= 0.0, "softmax value {} should be >= 0", v);
        }
    }

    /// Property: softmax preserves relative ordering (larger input -> larger probability)
    #[test]
    fn prop_softmax_preserves_order(
        values in prop::collection::vec(-10.0f32..10.0, 2..=50)
    ) {
        let original = values.clone();
        let mut x = values;
        softmax_simd(&mut x);

        // Find indices of max and min in original
        let max_idx = original.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let min_idx = original.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        if max_idx != min_idx {
            prop_assert!(x[max_idx] >= x[min_idx],
                "softmax should preserve max > min: {} vs {}", x[max_idx], x[min_idx]);
        }
    }

    /// Property: softmax is numerically stable (no inf/nan)
    #[test]
    fn prop_softmax_stable(
        values in prop::collection::vec(-500.0f32..500.0, 1..=100)
    ) {
        let mut x = values;
        softmax_simd(&mut x);

        for &v in &x {
            prop_assert!(!v.is_nan(), "softmax should not produce NaN");
            prop_assert!(!v.is_infinite(), "softmax should not produce Inf");
        }
    }

    /// Property: softmax with single element is 1.0
    #[test]
    fn prop_softmax_single_element(value in -100.0f32..100.0) {
        let mut x = vec![value];
        softmax_simd(&mut x);
        prop_assert!((x[0] - 1.0).abs() < 1e-6, "single element softmax should be 1.0, got {}", x[0]);
    }
}

// ============================================================================
// fused_swiglu_simd Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Property: SwiGLU output has same length as inputs
    #[test]
    fn prop_swiglu_preserves_length(
        len in 1usize..=256
    ) {
        let mut gate = vec![0.5f32; len];
        let up = vec![1.0f32; len];

        fused_swiglu_simd(&mut gate, &up);

        prop_assert_eq!(gate.len(), len);
    }

    /// Property: SwiGLU with zero gate produces zeros
    #[test]
    fn prop_swiglu_zero_gate(
        up in prop::collection::vec(-10.0f32..10.0, 1..=100)
    ) {
        let mut gate = vec![0.0f32; up.len()];
        fused_swiglu_simd(&mut gate, &up);

        // sigmoid(0) = 0.5, so gate[i] = 0 * 0.5 * up[i] = 0
        for &v in &gate {
            prop_assert!((v).abs() < 1e-5, "zero gate should produce zero output, got {}", v);
        }
    }

    /// Property: SwiGLU produces finite outputs for finite inputs
    #[test]
    fn prop_swiglu_finite(
        gate_vals in prop::collection::vec(-10.0f32..10.0, 1..=100),
        up_vals in prop::collection::vec(-10.0f32..10.0, 1..=100)
    ) {
        let len = gate_vals.len().min(up_vals.len());
        let mut gate = gate_vals[..len].to_vec();
        let up = &up_vals[..len];

        fused_swiglu_simd(&mut gate, up);

        for &v in &gate {
            prop_assert!(v.is_finite(), "SwiGLU output should be finite, got {}", v);
        }
    }

    /// Property: SwiGLU is bounded when inputs are bounded
    #[test]
    fn prop_swiglu_bounded(
        gate_vals in prop::collection::vec(-5.0f32..5.0, 8..=64),
        up_vals in prop::collection::vec(-5.0f32..5.0, 8..=64)
    ) {
        let len = gate_vals.len().min(up_vals.len());
        let mut gate = gate_vals[..len].to_vec();
        let up = &up_vals[..len];

        fused_swiglu_simd(&mut gate, up);

        // SwiGLU = x * sigmoid(x) * up
        // |sigmoid(x)| <= 1, so |SwiGLU| <= |x| * |up|
        let max_input = 5.0f32;
        let max_expected = max_input * max_input;  // x * up

        for &v in &gate {
            prop_assert!(v.abs() <= max_expected + 1.0,
                "SwiGLU output {} should be bounded", v);
        }
    }
}

// ============================================================================
// apply_rope_rotation_simd Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property: RoPE preserves combined vector norm (rotation preserves length)
    #[test]
    fn prop_rope_preserves_norm(
        len in 4usize..=64,
        values in prop::collection::vec(-5.0f32..5.0, 8..=128)
    ) {
        // Need at least 2*len values for x1 and x2
        if values.len() < len * 2 {
            return Ok(());
        }

        let mut x1 = values[..len].to_vec();
        let mut x2 = values[len..len*2].to_vec();
        let original_norm_sq: f32 = x1.iter().chain(x2.iter()).map(|v| v * v).sum();

        // Create valid cos/sin (cos²θ + sin²θ = 1)
        let cos_vals: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.1).cos()).collect();
        let sin_vals: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.1).sin()).collect();

        apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

        let new_norm_sq: f32 = x1.iter().chain(x2.iter()).map(|v| v * v).sum();

        // Norm should be preserved (within floating point error)
        let relative_error = (new_norm_sq - original_norm_sq).abs() / (original_norm_sq + 1e-8);
        prop_assert!(relative_error < 0.01,
            "RoPE should preserve norm: original² {} vs new² {}", original_norm_sq, new_norm_sq);
    }

    /// Property: RoPE with identity rotation (cos=1, sin=0) preserves vectors
    #[test]
    fn prop_rope_identity_rotation(
        len in 4usize..=64,
        values in prop::collection::vec(-10.0f32..10.0, 8..=128)
    ) {
        if values.len() < len * 2 {
            return Ok(());
        }

        let mut x1 = values[..len].to_vec();
        let mut x2 = values[len..len*2].to_vec();
        let original_x1 = x1.clone();
        let original_x2 = x2.clone();

        // Identity rotation: cos=1, sin=0
        let cos_vals = vec![1.0f32; len];
        let sin_vals = vec![0.0f32; len];

        apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

        for (i, (&orig, &rotated)) in original_x1.iter().zip(x1.iter()).enumerate() {
            prop_assert!((orig - rotated).abs() < 1e-5,
                "Identity rotation should preserve x1 at {}: {} vs {}", i, orig, rotated);
        }
        for (i, (&orig, &rotated)) in original_x2.iter().zip(x2.iter()).enumerate() {
            prop_assert!((orig - rotated).abs() < 1e-5,
                "Identity rotation should preserve x2 at {}: {} vs {}", i, orig, rotated);
        }
    }

    /// Property: RoPE output is finite for finite inputs
    #[test]
    fn prop_rope_finite_output(
        len in 4usize..=32,
        values in prop::collection::vec(-100.0f32..100.0, 8..=64)
    ) {
        if values.len() < len * 2 {
            return Ok(());
        }

        let mut x1 = values[..len].to_vec();
        let mut x2 = values[len..len*2].to_vec();
        let cos_vals: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.1).cos()).collect();
        let sin_vals: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.1).sin()).collect();

        apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

        for &v in x1.iter().chain(x2.iter()) {
            prop_assert!(v.is_finite(), "RoPE output should be finite");
        }
    }
}

// ============================================================================
// Additional SIMD Function Tests
// ============================================================================

#[test]
fn test_f16_known_values() {
    // Test known f16 values
    // 1.0 in f16: 0x3C00
    assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);

    // -1.0 in f16: 0xBC00
    assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 1e-6);

    // 0.5 in f16: 0x3800
    assert!((f16_to_f32(0x3800) - 0.5).abs() < 1e-6);

    // 2.0 in f16: 0x4000
    assert!((f16_to_f32(0x4000) - 2.0).abs() < 1e-6);

    // 0.0 in f16: 0x0000
    assert_eq!(f16_to_f32(0x0000), 0.0);

    // -0.0 in f16: 0x8000
    assert_eq!(f16_to_f32(0x8000), -0.0);
}

// ============================================================================
// Integration Tests: SIMD vs Scalar Equivalence
// ============================================================================

#[test]
fn test_softmax_simd_vs_reference() {
    // Reference implementation
    fn softmax_reference(x: &mut [f32]) {
        let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in x.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in x.iter_mut() {
            *v /= sum;
        }
    }

    let test_cases = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![-1.0, 0.0, 1.0],
        vec![0.5; 100],
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ];

    for case in test_cases {
        let mut simd_result = case.clone();
        let mut ref_result = case.clone();

        softmax_simd(&mut simd_result);
        softmax_reference(&mut ref_result);

        for (i, (&s, &r)) in simd_result.iter().zip(ref_result.iter()).enumerate() {
            assert!(
                (s - r).abs() < 1e-5,
                "softmax mismatch at {}: SIMD {} vs ref {}",
                i,
                s,
                r
            );
        }
    }
}

#[test]
fn test_swiglu_simd_deterministic() {
    // Test that SwiGLU is deterministic (same input -> same output)
    let gate1 = vec![1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0];
    let up = vec![1.0f32; 8];

    let mut result1 = gate1.clone();
    let mut result2 = gate1;

    fused_swiglu_simd(&mut result1, &up);
    fused_swiglu_simd(&mut result2, &up);

    for (i, (&r1, &r2)) in result1.iter().zip(result2.iter()).enumerate() {
        assert_eq!(r1, r2, "SwiGLU should be deterministic at {}", i);
    }
}

#[test]
fn test_swiglu_simd_zero_up() {
    // SwiGLU with zero up should give zero
    let mut gate = vec![1.0f32, 2.0, 3.0, 4.0];
    let up = vec![0.0f32; 4];

    fused_swiglu_simd(&mut gate, &up);

    for (i, &v) in gate.iter().enumerate() {
        assert_eq!(v, 0.0, "SwiGLU with zero up should be zero at {}", i);
    }
}
