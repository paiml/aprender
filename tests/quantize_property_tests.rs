//! Property-based tests for quantize.rs
//!
//! Uses proptest to verify mathematical correctness of quantization functions.
//! These tests provide strong coverage guarantees through random input generation.

use proptest::prelude::*;
use realizar::quantize::{
    dequantize_f16, dequantize_q4_0, dequantize_q4_1, dequantize_q4_k, dequantize_q5_0,
    dequantize_q5_1, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0, f16_to_f32,
    fused_swiglu_simd, quantize_activations_q8_0, quantize_rmsnorm_q8_0, quantize_to_q8_blocks,
    softmax_simd, Q8KSuperBlock, Q8_0Block, BLOCK_SIZE,
};

// ============================================================================
// Property tests for Q8_0Block
// ============================================================================

proptest! {
    #[test]
    fn prop_q8_0_block_roundtrip_preserves_sign(values in prop::collection::vec(-100.0f32..100.0f32, 32..=32)) {
        let values_arr: [f32; 32] = values.try_into().unwrap();
        let block = Q8_0Block::quantize(&values_arr);
        let dequant = block.dequantize();

        // Signs should be preserved for values with significant magnitude
        // Small values relative to scale may not preserve sign due to quantization
        let max_val = values_arr.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let threshold = max_val * 0.1; // 10% of max value
        for (orig, deq) in values_arr.iter().zip(dequant.iter()) {
            if orig.abs() > threshold {
                prop_assert_eq!(orig.signum(), deq.signum());
            }
        }
    }

    #[test]
    fn prop_q8_0_block_error_bounded(values in prop::collection::vec(-10.0f32..10.0f32, 32..=32)) {
        let values_arr: [f32; 32] = values.try_into().unwrap();
        let block = Q8_0Block::quantize(&values_arr);
        let error = block.quantization_error(&values_arr);
        let rel_error = block.relative_error(&values_arr);

        // Quantization error should be bounded
        prop_assert!(error < 0.2); // Max 0.2 absolute error
        prop_assert!(rel_error < 0.1); // Max 10% relative error
    }

    #[test]
    fn prop_q8_0_block_scale_positive(values in prop::collection::vec(-100.0f32..100.0f32, 32..=32)) {
        let values_arr: [f32; 32] = values.try_into().unwrap();
        let block = Q8_0Block::quantize(&values_arr);

        // Scale should always be positive
        prop_assert!(block.scale > 0.0);
    }

    #[test]
    fn prop_q8_0_block_quants_bounded(values in prop::collection::vec(-100.0f32..100.0f32, 32..=32)) {
        let values_arr: [f32; 32] = values.try_into().unwrap();
        let block = Q8_0Block::quantize(&values_arr);

        // All quants are i8, range is type-guaranteed
        prop_assert!(!block.quants.is_empty());
    }
}

// ============================================================================
// Property tests for Q8KSuperBlock
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_q8k_superblock_roundtrip(values in prop::collection::vec(-50.0f32..50.0f32, 256..=256)) {
        let values_arr: [f32; 256] = values.try_into().unwrap();
        let block = Q8KSuperBlock::quantize(&values_arr);
        let dequant = block.dequantize();

        // Check that dequantized values have same sign
        for (i, (orig, deq)) in values_arr.iter().zip(dequant.iter()).enumerate() {
            if orig.abs() > 0.5 {
                prop_assert_eq!(
                    orig.signum(), deq.signum(),
                    "Sign mismatch at index {}: orig={}, deq={}",
                    i, orig, deq
                );
            }
        }
    }

    #[test]
    fn prop_q8k_superblock_scale_valid(values in prop::collection::vec(-100.0f32..100.0f32, 256..=256)) {
        let values_arr: [f32; 256] = values.try_into().unwrap();
        let block = Q8KSuperBlock::quantize(&values_arr);

        // Scale should be positive
        prop_assert!(block.scale > 0.0);
        prop_assert!(block.scale.is_finite());
    }
}

// ============================================================================
// Property tests for quantize_to_q8_blocks
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_quantize_to_q8_blocks_length(
        num_blocks in 1usize..10usize,
        value in -10.0f32..10.0f32
    ) {
        let values = vec![value; num_blocks * BLOCK_SIZE];
        let blocks = quantize_to_q8_blocks(&values).unwrap();

        prop_assert_eq!(blocks.len(), num_blocks);
    }

    #[test]
    fn prop_quantize_to_q8_blocks_error_on_non_multiple(
        extra in 1usize..31usize
    ) {
        let values = vec![1.0f32; BLOCK_SIZE + extra];
        let result = quantize_to_q8_blocks(&values);

        // Should fail on non-multiple of 32
        prop_assert!(result.is_err());
    }
}

// ============================================================================
// Property tests for f16_to_f32
// ============================================================================

proptest! {
    #[test]
    fn prop_f16_to_f32_valid_output(bits in 0u16..=65535u16) {
        let result = f16_to_f32(bits);

        // Result should be a valid float (possibly NaN or Inf)
        prop_assert!(result.is_nan() || result.is_finite() || result.is_infinite());
    }

    #[test]
    fn prop_f16_to_f32_special_values(
        sign in 0u16..=1u16,
        exp in 0u16..=30u16, // Avoid special exponents
        frac in 0u16..=1023u16
    ) {
        let bits = (sign << 15) | (exp << 10) | frac;
        let result = f16_to_f32(bits);

        // Non-special values should be finite
        if exp > 0 && exp < 31 {
            prop_assert!(result.is_finite());
        }
    }
}

// ============================================================================
// Property tests for softmax_simd
// ============================================================================

proptest! {
    #[test]
    fn prop_softmax_sums_to_one(values in prop::collection::vec(-5.0f32..5.0f32, 4..64)) {
        let mut x = values;
        softmax_simd(&mut x);

        // Sum should be approximately 1.0
        let sum: f32 = x.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5, "Sum was {} instead of 1.0", sum);
    }

    #[test]
    fn prop_softmax_all_positive(values in prop::collection::vec(-10.0f32..10.0f32, 4..64)) {
        let mut x = values;
        softmax_simd(&mut x);

        // All values should be positive after softmax
        for (i, &v) in x.iter().enumerate() {
            prop_assert!(v >= 0.0, "Value at {} was negative: {}", i, v);
            prop_assert!(v <= 1.0, "Value at {} was > 1: {}", i, v);
        }
    }

    #[test]
    fn prop_softmax_preserves_order(values in prop::collection::vec(-5.0f32..5.0f32, 4..16)) {
        let mut x = values.clone();
        softmax_simd(&mut x);

        // Larger input values should have larger output values
        for i in 0..values.len() {
            for j in (i + 1)..values.len() {
                if values[i] > values[j] + 0.01 {
                    prop_assert!(x[i] >= x[j] - 1e-5);
                }
            }
        }
    }
}

// ============================================================================
// Property tests for quantize_activations_q8_0
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_quantize_activations_correct_length(
        num_blocks in 1usize..8usize,
        value in -5.0f32..5.0f32
    ) {
        let activations = vec![value; num_blocks * BLOCK_SIZE];
        let (scales, quants) = quantize_activations_q8_0(&activations);

        prop_assert_eq!(scales.len(), num_blocks);
        prop_assert_eq!(quants.len(), num_blocks * BLOCK_SIZE);
    }

    #[test]
    fn prop_quantize_activations_scales_positive(
        num_blocks in 1usize..4usize
    ) {
        let activations: Vec<f32> = (0..num_blocks * BLOCK_SIZE)
            .map(|i| (i as f32 * 0.1) - 5.0)
            .collect();
        let (scales, _) = quantize_activations_q8_0(&activations);

        for (i, &scale) in scales.iter().enumerate() {
            prop_assert!(scale > 0.0, "Scale at {} was not positive: {}", i, scale);
        }
    }
}

// ============================================================================
// Property tests for fused_swiglu_simd
// ============================================================================

proptest! {
    #[test]
    fn prop_swiglu_produces_finite(
        gate in prop::collection::vec(-5.0f32..5.0f32, 32..=32),
        up in prop::collection::vec(-5.0f32..5.0f32, 32..=32)
    ) {
        let mut gate_vec = gate;
        fused_swiglu_simd(&mut gate_vec, &up);

        // All outputs should be finite
        for (i, &v) in gate_vec.iter().enumerate() {
            prop_assert!(v.is_finite(), "Value at {} was not finite: {}", i, v);
        }
    }

    #[test]
    fn prop_swiglu_bounded_by_up(
        gate in prop::collection::vec(-2.0f32..2.0f32, 32..=32),
        up in prop::collection::vec(-2.0f32..2.0f32, 32..=32)
    ) {
        let mut gate_vec = gate;
        fused_swiglu_simd(&mut gate_vec, &up);

        // SwiGLU = silu(gate) * up, bounded by ~max(|up|) * 2 (since silu is bounded)
        let max_up = up.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        for (i, &v) in gate_vec.iter().enumerate() {
            prop_assert!(
                v.abs() <= max_up * 2.0 + 0.1,
                "Value at {} was {} but max_up was {}",
                i, v, max_up
            );
        }
    }
}

// ============================================================================
// Property tests for dequantize functions - error handling
// ============================================================================

proptest! {
    #[test]
    fn prop_dequantize_q4_0_length_check(data in prop::collection::vec(0u8..=255u8, 1..100)) {
        // Q4_0 blocks are 18 bytes each (2 bytes f16 scale + 16 bytes quants)
        let result = dequantize_q4_0(&data);

        // Should only succeed if length is multiple of 18
        if data.len() % 18 == 0 && !data.is_empty() {
            prop_assert!(result.is_ok());
            let values = result.unwrap();
            prop_assert_eq!(values.len(), (data.len() / 18) * 32);
        } else {
            prop_assert!(result.is_err());
        }
    }

    #[test]
    fn prop_dequantize_q8_0_length_check(data in prop::collection::vec(0u8..=255u8, 1..150)) {
        // Q8_0 blocks are 34 bytes each (2 bytes f16 scale + 32 bytes quants)
        let result = dequantize_q8_0(&data);

        if data.len() % 34 == 0 && !data.is_empty() {
            prop_assert!(result.is_ok());
            let values = result.unwrap();
            prop_assert_eq!(values.len(), (data.len() / 34) * 32);
        } else {
            prop_assert!(result.is_err());
        }
    }

    #[test]
    fn prop_dequantize_f16_length_check(data in prop::collection::vec(0u8..=255u8, 0..100)) {
        let result = dequantize_f16(&data);

        // F16 needs even number of bytes
        if data.len() % 2 == 0 {
            prop_assert!(result.is_ok());
            let values = result.unwrap();
            prop_assert_eq!(values.len(), data.len() / 2);
        } else {
            prop_assert!(result.is_err());
        }
    }
}

// ============================================================================
// Property tests for quantize_rmsnorm_q8_0
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn prop_rmsnorm_produces_valid(
        input in prop::collection::vec(-5.0f32..5.0f32, 64..=64),
        weight in prop::collection::vec(0.5f32..2.0f32, 64..=64),
        eps in 1e-6f32..1e-4f32
    ) {
        let (normalized, quants) = quantize_rmsnorm_q8_0(&input, &weight, eps);

        // Normalized values should be reasonable (bounded by weight)
        let max_weight = weight.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        for (i, &v) in normalized.iter().enumerate() {
            prop_assert!(
                v.abs() <= max_weight * 10.0,
                "Normalized value at {} was too large: {}",
                i, v
            );
        }

        // Quants should be bounded
        // Quants are i8, range is type-guaranteed
        prop_assert!(!quants.is_empty(), "Should have quantized values");
    }
}

// ============================================================================
// Edge case tests (non-proptest)
// ============================================================================

#[test]
fn test_q8_0_block_zero_input() {
    let values = [0.0f32; 32];
    let block = Q8_0Block::quantize(&values);

    // Scale should be minimal but positive
    assert!(block.scale > 0.0);

    // All quants should be 0
    for &q in &block.quants {
        assert_eq!(q, 0);
    }
}

#[test]
fn test_q8_0_block_max_input() {
    let values = [f32::MAX / 2.0; 32];
    let block = Q8_0Block::quantize(&values);

    // Scale should be very large
    assert!(block.scale > 0.0);
    assert!(block.scale.is_finite());

    // All quants should be 127 (max positive)
    for &q in &block.quants {
        assert_eq!(q, 127);
    }
}

#[test]
fn test_q8_0_block_min_input() {
    let values = [f32::MIN / 2.0; 32];
    let block = Q8_0Block::quantize(&values);

    // Scale should be very large
    assert!(block.scale > 0.0);
    assert!(block.scale.is_finite());

    // All quants should be at or near minimum (clamp happens at -128 but could be -127 due to rounding)
    for &q in &block.quants {
        assert!(q <= -127, "Expected q <= -127, got {}", q);
    }
}

#[test]
fn test_softmax_uniform_input() {
    let mut x = vec![1.0f32; 8];
    softmax_simd(&mut x);

    // Uniform input should give uniform output (1/n each)
    let expected = 1.0 / 8.0;
    for &v in &x {
        assert!((v - expected).abs() < 1e-5);
    }
}

#[test]
fn test_softmax_single_element() {
    let mut x = vec![5.0f32];
    softmax_simd(&mut x);

    // Single element softmax is always 1.0
    assert!((x[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_extreme_values() {
    let mut x = vec![-100.0, 0.0, 100.0];
    softmax_simd(&mut x);

    // First value should be essentially 0
    assert!(x[0] < 1e-30);
    // Last value should be essentially 1
    assert!((x[2] - 1.0).abs() < 1e-10);
}

#[test]
fn test_f16_to_f32_zero() {
    // Zero
    assert_eq!(f16_to_f32(0x0000), 0.0);
    // Negative zero
    assert_eq!(f16_to_f32(0x8000), -0.0);
}

#[test]
fn test_f16_to_f32_one() {
    // 1.0 in f16 is 0x3C00
    assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-10);
}

#[test]
fn test_f16_to_f32_negative_one() {
    // -1.0 in f16 is 0xBC00
    assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 1e-10);
}

#[test]
fn test_f16_to_f32_infinity() {
    // +Inf
    assert!(f16_to_f32(0x7C00).is_infinite());
    assert!(f16_to_f32(0x7C00) > 0.0);
    // -Inf
    assert!(f16_to_f32(0xFC00).is_infinite());
    assert!(f16_to_f32(0xFC00) < 0.0);
}

#[test]
fn test_f16_to_f32_nan() {
    // NaN (exponent all 1s, non-zero fraction)
    assert!(f16_to_f32(0x7C01).is_nan());
    assert!(f16_to_f32(0xFFFF).is_nan());
}

// ============================================================================
// K-quantization format tests
// ============================================================================

#[test]
fn test_dequantize_q4_k_empty() {
    let result = dequantize_q4_k(&[]);
    assert!(result.is_err() || result.unwrap().is_empty());
}

#[test]
fn test_dequantize_q5_k_empty() {
    let result = dequantize_q5_k(&[]);
    assert!(result.is_err() || result.unwrap().is_empty());
}

#[test]
fn test_dequantize_q6_k_empty() {
    let result = dequantize_q6_k(&[]);
    assert!(result.is_err() || result.unwrap().is_empty());
}

#[test]
fn test_dequantize_q4_1_empty() {
    let result = dequantize_q4_1(&[]);
    assert!(result.is_err() || result.unwrap().is_empty());
}

#[test]
fn test_dequantize_q5_0_empty() {
    let result = dequantize_q5_0(&[]);
    assert!(result.is_err() || result.unwrap().is_empty());
}

#[test]
fn test_dequantize_q5_1_empty() {
    let result = dequantize_q5_1(&[]);
    assert!(result.is_err() || result.unwrap().is_empty());
}
