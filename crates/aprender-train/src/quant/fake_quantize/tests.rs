//! Tests for fake quantization.

use super::*;
use crate::Tensor;
use approx::assert_abs_diff_eq;
use proptest::prelude::*;

// ========================================================================
// PROPERTY TESTS - Fake quantization correctness
// ========================================================================

proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(200))]

    /// STE backward should always pass gradients unchanged
    #[test]
    fn prop_ste_backward_identity(
        grad in prop::collection::vec(-10.0f32..10.0, 1..32),
    ) {
        let grad_tensor = Tensor::from_vec(grad.clone(), true);
        let fq = FakeQuantize::q8();

        let backward = fq.backward(&grad_tensor);

        // STE should pass through unchanged
        prop_assert_eq!(backward.len(), grad.len());
        for (i, &g) in grad.iter().enumerate() {
            prop_assert!(
                (backward.data()[i] - g).abs() < 1e-6,
                "STE should preserve gradient at index {}", i
            );
        }
    }

    /// Fake quantize should produce values that are multiples of scale
    #[test]
    fn prop_fake_quantize_produces_quantized_values(
        values in prop::collection::vec(-5.0f32..5.0, 4..32),
        bits in 4usize..9,
    ) {
        let input = Tensor::from_vec(values.clone(), false);
        let config = FakeQuantConfig::symmetric(bits);
        let mut fq = FakeQuantize::new(config);
        fq.calibrate(&values);

        let output = fq.forward(&input);

        // Each output value should be a valid quantized level
        let scale = fq.scale();
        for &val in output.data() {
            // Value should be approximately q * scale for some integer q
            let q = (val / scale).round();
            let reconstructed = q * scale;
            prop_assert!(
                (val - reconstructed).abs() < 1e-5,
                "Value {} should be quantized (q={}, scale={})",
                val, q, scale
            );
        }
    }

    /// Fake quantize output should be bounded by qmin*scale and qmax*scale
    #[test]
    fn prop_fake_quantize_bounded_output(
        values in prop::collection::vec(-100.0f32..100.0, 4..32),
        bits in 4usize..9,
    ) {
        let input = Tensor::from_vec(values.clone(), false);
        let config = FakeQuantConfig::symmetric(bits);
        let mut fq = FakeQuantize::new(config);
        fq.calibrate(&values);

        let output = fq.forward(&input);

        let qmin_float = fq.config.qmin as f32 * fq.scale();
        let qmax_float = fq.config.qmax as f32 * fq.scale();

        for &val in output.data() {
            prop_assert!(
                val >= qmin_float - 1e-5 && val <= qmax_float + 1e-5,
                "Output {} should be in [{}, {}]",
                val, qmin_float, qmax_float
            );
        }
    }

    /// Calibration should set scale based on data range
    #[test]
    fn prop_calibration_sets_appropriate_scale(
        values in prop::collection::vec(-10.0f32..10.0, 4..32),
        bits in 4usize..9,
    ) {
        let config = FakeQuantConfig::symmetric(bits);
        let mut fq = FakeQuantize::new(config);

        fq.calibrate(&values);

        prop_assert!(fq.is_initialized());
        prop_assert!(fq.scale() > 0.0);

        // For symmetric, scale should be max_abs / qmax
        let max_abs = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let expected_scale = max_abs / fq.config.qmax as f32;

        // Allow small tolerance for numerical precision
        if max_abs > 1e-8 {
            prop_assert!(
                (fq.scale() - expected_scale).abs() < 1e-5,
                "Scale {} should be {} (max_abs={}, qmax={})",
                fq.scale(), expected_scale, max_abs, fq.config.qmax
            );
        }
    }

    /// Number of quantization levels should be correct
    #[test]
    fn prop_num_levels_correct(bits in 2usize..10) {
        let config = FakeQuantConfig::symmetric(bits);
        let fq = FakeQuantize::new(config);

        // Symmetric: qmin = -(2^(bits-1)-1), qmax = 2^(bits-1)-1
        // Levels = qmax - qmin + 1 = 2 * (2^(bits-1)-1) + 1 = 2^bits - 1
        let expected = (1 << bits) - 1;
        prop_assert_eq!(fq.num_levels(), expected);
    }
}

// ========================================================================
// UNIT TESTS
// ========================================================================

#[test]
fn test_fake_quantize_config_symmetric() {
    let config = FakeQuantConfig::symmetric(4);
    assert_eq!(config.bits, 4);
    assert!(config.symmetric);
    assert_eq!(config.qmin, -7);
    assert_eq!(config.qmax, 7);

    let config8 = FakeQuantConfig::symmetric(8);
    assert_eq!(config8.qmin, -127);
    assert_eq!(config8.qmax, 127);
}

#[test]
fn test_fake_quantize_config_asymmetric() {
    let config = FakeQuantConfig::asymmetric(4);
    assert_eq!(config.bits, 4);
    assert!(!config.symmetric);
    assert_eq!(config.qmin, 0);
    assert_eq!(config.qmax, 15);

    let config8 = FakeQuantConfig::asymmetric(8);
    assert_eq!(config8.qmin, 0);
    assert_eq!(config8.qmax, 255);
}

#[test]
fn test_fake_quantize_forward() {
    let input = Tensor::from_vec(vec![0.0, 1.0, -1.0, 0.5, -0.5], false);
    let mut fq = FakeQuantize::q8();
    fq.calibrate(input.data().as_slice().expect("operation should succeed"));

    let output = fq.forward(&input);

    assert_eq!(output.len(), 5);
    // 0 should stay 0
    assert_abs_diff_eq!(output.data()[0], 0.0, epsilon = 1e-5);
}

#[test]
fn test_fake_quantize_forward_with_calibration() {
    let input = Tensor::from_vec(vec![0.0, 1.0, -1.0, 0.5, -0.5], false);
    let mut fq = FakeQuantize::q8();

    assert!(!fq.is_initialized());

    let output = fq.forward_with_calibration(&input);

    assert!(fq.is_initialized());
    assert_eq!(output.len(), 5);
}

#[test]
fn test_ste_backward() {
    let grad = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);
    let fq = FakeQuantize::q8();

    let backward = fq.backward(&grad);

    // STE: gradient should pass through unchanged
    assert_eq!(backward.len(), 4);
    for i in 0..4 {
        assert_abs_diff_eq!(backward.data()[i], grad.data()[i], epsilon = 1e-6);
    }
}

#[test]
fn test_clamped_ste_backward() {
    let grad = Tensor::from_vec(vec![1.0, 1.0, 1.0], true);
    let input = Tensor::from_vec(vec![0.5, 10.0, -10.0], false); // 10, -10 outside range

    let mut fq = FakeQuantize::q4();
    fq.scale = 1.0; // Set scale so range is [-7, 7]
    fq.initialized = true;

    let backward = fq.backward_clamped(&grad, &input);

    // 0.5 is in range: gradient passes
    assert_abs_diff_eq!(backward.data()[0], 1.0, epsilon = 1e-6);
    // 10.0 is outside range: gradient clipped to 0
    assert_abs_diff_eq!(backward.data()[1], 0.0, epsilon = 1e-6);
    // -10.0 is outside range: gradient clipped to 0
    assert_abs_diff_eq!(backward.data()[2], 0.0, epsilon = 1e-6);
}

#[test]
fn test_calibration_symmetric() {
    let mut fq = FakeQuantize::q8();
    let data = vec![0.0, 1.0, -2.0, 1.5, -1.5];

    fq.calibrate(&data);

    // max_abs = 2.0, qmax = 127
    // scale = 2.0 / 127
    let expected_scale = 2.0 / 127.0;
    assert_abs_diff_eq!(fq.scale(), expected_scale, epsilon = 1e-6);
    assert_eq!(fq.zero_point(), 0);
}

#[test]
fn test_fake_quantize_convenience_function() {
    let input = Tensor::from_vec(vec![0.0, 1.0, -1.0], false);

    let output = fake_quantize(&input, 8, true);

    assert_eq!(output.len(), 3);
}

#[test]
fn test_ste_backward_convenience_function() {
    let grad = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);

    let backward = ste_backward(&grad);

    for i in 0..3 {
        assert_abs_diff_eq!(backward.data()[i], grad.data()[i], epsilon = 1e-6);
    }
}

#[test]
fn test_num_levels() {
    let fq4 = FakeQuantize::q4();
    assert_eq!(fq4.num_levels(), 15); // -7 to 7 = 15 levels

    let fq8 = FakeQuantize::q8();
    assert_eq!(fq8.num_levels(), 255); // -127 to 127 = 255 levels
}

#[test]
fn test_quantize_dequantize_round_trip() {
    let input = Tensor::from_vec(vec![0.0, 0.5, 1.0, -0.5, -1.0], false);
    let mut fq = FakeQuantize::q8();
    fq.calibrate(input.data().as_slice().expect("operation should succeed"));

    let output = fq.forward(&input);

    // Output should be close to input (with quantization noise)
    for (i, (&orig, &out)) in input.data().iter().zip(output.data().iter()).enumerate() {
        let error = (orig - out).abs();
        assert!(error < 0.1, "Error {error} at index {i} too large: {orig} vs {out}");
    }
}

#[test]
fn test_fake_quant_config_default() {
    let config = FakeQuantConfig::default();
    // Default should be q8_symmetric
    assert_eq!(config.bits, 8);
    assert!(config.symmetric);
    assert_eq!(config.qmin, -127);
    assert_eq!(config.qmax, 127);
}

#[test]
fn test_calibrate_empty_data() {
    let mut fq = FakeQuantize::q8();
    fq.calibrate(&[]);
    // Should not crash, and scale should remain at default
    assert!(!fq.is_initialized());
    assert_eq!(fq.scale(), 1.0);
}

#[test]
fn test_calibration_asymmetric() {
    let config = FakeQuantConfig::asymmetric(8);
    let mut fq = FakeQuantize::new(config);

    // All positive data
    let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    fq.calibrate(&data);

    assert!(fq.is_initialized());
    assert!(fq.scale() > 0.0);
    // For asymmetric, zero_point should be computed
    // scale = (4-0) / (255-0) ≈ 0.0157
    // zero_point = round(0 - 0/scale) = 0
    assert!(fq.zero_point() >= 0);
}

#[test]
fn test_calibration_asymmetric_negative_offset() {
    let config = FakeQuantConfig::asymmetric(8);
    let mut fq = FakeQuantize::new(config);

    // Shifted data
    let data = vec![10.0, 11.0, 12.0, 13.0, 14.0];
    fq.calibrate(&data);

    assert!(fq.is_initialized());
    assert!(fq.scale() > 0.0);
}

#[test]
fn test_asymmetric_forward() {
    let config = FakeQuantConfig::asymmetric(8);
    let mut fq = FakeQuantize::new(config);

    let data = vec![0.0, 0.5, 1.0, 1.5, 2.0];
    fq.calibrate(&data);

    let input = Tensor::from_vec(data.clone(), false);
    let output = fq.forward(&input);

    assert_eq!(output.len(), 5);
    // Output should be quantized values
    for &val in output.data() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_fake_quantize_asymmetric_convenience() {
    let input = Tensor::from_vec(vec![0.0, 1.0, 2.0], false);
    let output = fake_quantize(&input, 8, false);
    assert_eq!(output.len(), 3);
}

#[test]
fn test_fake_quant_config_clone() {
    let config = FakeQuantConfig::symmetric(4);
    let cloned = config.clone();
    assert_eq!(config.bits, cloned.bits);
    assert_eq!(config.symmetric, cloned.symmetric);
    assert_eq!(config.qmin, cloned.qmin);
    assert_eq!(config.qmax, cloned.qmax);
}

#[test]
fn test_fake_quantize_clone() {
    let mut fq = FakeQuantize::q8();
    fq.scale = 0.5;
    fq.zero_point = 10;
    fq.initialized = true;

    let cloned = fq.clone();
    assert_eq!(fq.scale, cloned.scale);
    assert_eq!(fq.zero_point, cloned.zero_point);
    assert_eq!(fq.initialized, cloned.initialized);
}

#[test]
fn test_calibration_near_zero_scale() {
    let mut fq = FakeQuantize::q8();
    // All zeros - would result in zero scale without protection
    let data = vec![0.0, 0.0, 0.0];
    fq.calibrate(&data);

    // Scale should be clamped to minimum
    assert!(fq.scale() >= 1e-10);
    assert!(fq.is_initialized());
}
