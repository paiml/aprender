//! Tests for GGUF quantization formats

use super::*;
use approx::assert_abs_diff_eq;
use proptest::prelude::*;

// ========================================================================
// PROPERTY TESTS - Bit packing correctness
// ========================================================================

proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(200))]

    /// Q4_0 round-trip should preserve values within quantization error
    #[test]
    fn prop_q4_0_round_trip(
        values in prop::collection::vec(-10.0f32..10.0, 32..128),
    ) {
        let quantized = Q4_0::quantize(&values);
        let dequantized = quantized.dequantize();

        prop_assert_eq!(dequantized.len(), values.len());

        // Quantization error should be bounded
        for (i, (&orig, &deq)) in values.iter().zip(dequantized.iter()).enumerate() {
            let error = (orig - deq).abs();
            let max_error = quantized.scales[i / GGUF_BLOCK_SIZE] * 1.5;
            prop_assert!(
                error <= max_error,
                "Q4_0 error {} > {} at index {}",
                error, max_error, i
            );
        }
    }

    /// Q8_0 round-trip should preserve values within quantization error
    #[test]
    fn prop_q8_0_round_trip(
        values in prop::collection::vec(-10.0f32..10.0, 32..128),
    ) {
        let quantized = Q8_0::quantize(&values);
        let dequantized = quantized.dequantize();

        prop_assert_eq!(dequantized.len(), values.len());

        // Q8_0 should have smaller error than Q4_0
        for (i, (&orig, &deq)) in values.iter().zip(dequantized.iter()).enumerate() {
            let error = (orig - deq).abs();
            let max_error = quantized.scales[i / GGUF_BLOCK_SIZE] * 1.1;
            prop_assert!(
                error <= max_error,
                "Q8_0 error {} > {} at index {}",
                error, max_error, i
            );
        }
    }

    /// Q4_0 should use correct number of blocks
    #[test]
    fn prop_q4_0_block_count(len in 1usize..256) {
        let values = vec![1.0f32; len];
        let quantized = Q4_0::quantize(&values);

        let expected_blocks = len.div_ceil(GGUF_BLOCK_SIZE);
        prop_assert_eq!(quantized.num_blocks(), expected_blocks);
        prop_assert_eq!(quantized.scales.len(), expected_blocks);
        prop_assert_eq!(quantized.data.len(), expected_blocks * 16);
    }

    /// Q8_0 should use correct number of blocks
    #[test]
    fn prop_q8_0_block_count(len in 1usize..256) {
        let values = vec![1.0f32; len];
        let quantized = Q8_0::quantize(&values);

        let expected_blocks = len.div_ceil(GGUF_BLOCK_SIZE);
        prop_assert_eq!(quantized.num_blocks(), expected_blocks);
        prop_assert_eq!(quantized.scales.len(), expected_blocks);
    }

    /// Q4_0 compression ratio should be close to 8x for large tensors
    #[test]
    fn prop_q4_0_compression(len in 256usize..1024) {
        let values = vec![1.0f32; len];
        let quantized = Q4_0::quantize(&values);

        let ratio = quantized.compression_ratio();
        // Q4_0: 4 bits/value + scale overhead → ~7x compression
        prop_assert!(ratio > 5.0, "Q4_0 compression {} should be > 5x", ratio);
        prop_assert!(ratio < 10.0, "Q4_0 compression {} should be < 10x", ratio);
    }

    /// Q8_0 compression ratio should be close to 4x for large tensors
    #[test]
    fn prop_q8_0_compression(len in 256usize..1024) {
        let values = vec![1.0f32; len];
        let quantized = Q8_0::quantize(&values);

        let ratio = quantized.compression_ratio();
        // Q8_0: 8 bits/value + scale overhead → ~3.5x compression
        prop_assert!(ratio > 3.0, "Q8_0 compression {} should be > 3x", ratio);
        prop_assert!(ratio < 5.0, "Q8_0 compression {} should be < 5x", ratio);
    }
}

// ========================================================================
// UNIT TESTS
// ========================================================================

#[test]
fn test_q4_0_basic() {
    let values = vec![0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1.5];
    let quantized = Q4_0::quantize(&values);
    let dequantized = quantized.dequantize();

    assert_eq!(dequantized.len(), values.len());

    // Check approximate reconstruction
    for (orig, deq) in values.iter().zip(dequantized.iter()) {
        let error = (orig - deq).abs();
        assert!(error < 1.0, "Error {error} too large");
    }
}

#[test]
fn test_q8_0_basic() {
    let values = vec![0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1.5];
    let quantized = Q8_0::quantize(&values);
    let dequantized = quantized.dequantize();

    assert_eq!(dequantized.len(), values.len());

    // Q8_0 should have better precision than Q4_0
    for (orig, deq) in values.iter().zip(dequantized.iter()) {
        let error = (orig - deq).abs();
        assert!(error < 0.1, "Error {error} too large for Q8_0");
    }
}

#[test]
fn test_q4_0_block_size() {
    // Exactly one block
    let values = vec![1.0; GGUF_BLOCK_SIZE];
    let quantized = Q4_0::quantize(&values);
    assert_eq!(quantized.num_blocks(), 1);
    assert_eq!(quantized.data.len(), 16);

    // Two blocks
    let values = vec![1.0; GGUF_BLOCK_SIZE + 1];
    let quantized = Q4_0::quantize(&values);
    assert_eq!(quantized.num_blocks(), 2);
    assert_eq!(quantized.data.len(), 32);
}

#[test]
fn test_q8_0_block_size() {
    // Exactly one block
    let values = vec![1.0; GGUF_BLOCK_SIZE];
    let quantized = Q8_0::quantize(&values);
    assert_eq!(quantized.num_blocks(), 1);

    // Two blocks
    let values = vec![1.0; GGUF_BLOCK_SIZE + 1];
    let quantized = Q8_0::quantize(&values);
    assert_eq!(quantized.num_blocks(), 2);
}

#[test]
fn test_q4_0_zeros() {
    let values = vec![0.0; 64];
    let quantized = Q4_0::quantize(&values);
    let dequantized = quantized.dequantize();

    for val in dequantized {
        assert_abs_diff_eq!(val, 0.0, epsilon = 1e-5);
    }
}

#[test]
fn test_q8_0_zeros() {
    let values = vec![0.0; 64];
    let quantized = Q8_0::quantize(&values);
    let dequantized = quantized.dequantize();

    for val in dequantized {
        assert_abs_diff_eq!(val, 0.0, epsilon = 1e-5);
    }
}

#[test]
fn test_gguf_quant_type() {
    assert_eq!(GGUFQuantType::Q4_0.bits(), 4);
    assert_eq!(GGUFQuantType::Q8_0.bits(), 8);

    assert_eq!(GGUFQuantType::Q4_0.bytes_per_block(), 18);
    assert_eq!(GGUFQuantType::Q8_0.bytes_per_block(), 34);

    assert_abs_diff_eq!(GGUFQuantType::Q4_0.theoretical_compression(), 8.0, epsilon = 0.1);
    assert_abs_diff_eq!(GGUFQuantType::Q8_0.theoretical_compression(), 4.0, epsilon = 0.1);
}

#[test]
fn test_q4_0_memory_bytes() {
    let values = vec![1.0; 1024];
    let quantized = Q4_0::quantize(&values);

    // 1024 values = 32 blocks
    // GGUF bytes: 32 * 18 = 576 bytes
    assert_eq!(quantized.num_blocks(), 32);
    assert_eq!(quantized.gguf_bytes(), 32 * 18);

    let ratio = quantized.compression_ratio();
    assert!(ratio > 7.0, "Compression ratio {ratio} should be > 7x");
}

#[test]
fn test_q8_0_memory_bytes() {
    let values = vec![1.0; 1024];
    let quantized = Q8_0::quantize(&values);

    // 1024 values = 32 blocks
    // GGUF bytes: 32 * 34 = 1088 bytes
    assert_eq!(quantized.num_blocks(), 32);
    assert_eq!(quantized.gguf_bytes(), 32 * 34);

    let ratio = quantized.compression_ratio();
    assert!(ratio > 3.5, "Compression ratio {ratio} should be > 3.5x");
}

#[test]
fn test_q8_0_better_than_q4_0() {
    // Q8_0 should have lower error than Q4_0
    let values: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();

    let q4 = Q4_0::quantize(&values);
    let q8 = Q8_0::quantize(&values);

    let deq4 = q4.dequantize();
    let deq8 = q8.dequantize();

    let error4: f32 = values.iter().zip(deq4.iter()).map(|(a, b)| (a - b).abs()).sum();
    let error8: f32 = values.iter().zip(deq8.iter()).map(|(a, b)| (a - b).abs()).sum();

    assert!(error8 < error4, "Q8_0 error {error8} should be < Q4_0 error {error4}");
}

#[test]
fn test_q4_0_negative_values() {
    let values = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
    let quantized = Q4_0::quantize(&values);
    let dequantized = quantized.dequantize();

    for (&orig, &deq) in values.iter().zip(dequantized.iter()) {
        // Both should be negative
        assert!(deq < 0.0, "Expected negative, got {deq}");
        // Error should be reasonable
        let error = (orig - deq).abs();
        assert!(error < 2.0, "Error {error} too large");
    }
}

#[test]
fn test_q4_0_partial_block() {
    // Test with non-multiple of 32 elements
    let values: Vec<f32> = (0..50).map(|i| i as f32 * 0.1).collect();
    let quantized = Q4_0::quantize(&values);
    let dequantized = quantized.dequantize();

    assert_eq!(dequantized.len(), values.len());
}

#[test]
fn test_q8_0_partial_block() {
    // Test with non-multiple of 32 elements
    let values: Vec<f32> = (0..50).map(|i| i as f32 * 0.1).collect();
    let quantized = Q8_0::quantize(&values);
    let dequantized = quantized.dequantize();

    assert_eq!(dequantized.len(), values.len());
}

#[test]
fn test_q4_0_clone() {
    let values = vec![1.0, 2.0, 3.0, 4.0];
    let quantized = Q4_0::quantize(&values);
    let cloned = quantized.clone();
    assert_eq!(quantized.len, cloned.len);
    assert_eq!(quantized.scales, cloned.scales);
}

#[test]
fn test_q8_0_clone() {
    let values = vec![1.0, 2.0, 3.0, 4.0];
    let quantized = Q8_0::quantize(&values);
    let cloned = quantized.clone();
    assert_eq!(quantized.len, cloned.len);
    assert_eq!(quantized.scales, cloned.scales);
}

#[test]
fn test_q8_0_negative_values() {
    let values = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
    let quantized = Q8_0::quantize(&values);
    let dequantized = quantized.dequantize();

    for (&orig, &deq) in values.iter().zip(dequantized.iter()) {
        assert!(deq < 0.0, "Expected negative, got {deq}");
        let error = (orig - deq).abs();
        assert!(error < 0.5, "Error {error} too large for Q8_0");
    }
}

#[test]
fn test_gguf_quant_type_clone() {
    let qt = GGUFQuantType::Q4_0;
    let cloned = qt;
    assert_eq!(qt.bits(), cloned.bits());
}
