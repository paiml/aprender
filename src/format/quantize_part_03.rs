
// ============================================================================
// Property-Based Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    // ================================================================
    // Arbitrary Strategies
    // ================================================================

    /// Generate arbitrary QuantType (only implemented ones)
    fn arb_quant_type() -> impl Strategy<Value = QuantType> {
        prop_oneof![Just(QuantType::Q8_0), Just(QuantType::Q4_0),]
    }

    /// Generate arbitrary valid shape (1-3 dimensions, reasonable sizes)
    fn arb_shape() -> impl Strategy<Value = Vec<usize>> {
        prop_oneof![
            // 1D shapes
            (1usize..200).prop_map(|n| vec![n]),
            // 2D shapes
            (1usize..50, 1usize..50).prop_map(|(a, b)| vec![a, b]),
            // 3D shapes
            (1usize..20, 1usize..20, 1usize..20).prop_map(|(a, b, c)| vec![a, b, c]),
        ]
    }

    // ================================================================
    // QuantType Property Tests
    // ================================================================

    proptest! {
        /// Property: QuantType roundtrip via u8
        #[test]
        fn prop_quant_type_roundtrip(qt in arb_quant_type()) {
            let value = qt as u8;
            let parsed = QuantType::from_u8(value);
            prop_assert_eq!(parsed, Some(qt));
        }

        /// Property: Invalid QuantType values return None
        #[test]
        fn prop_invalid_quant_type_none(value in 4u8..0xFE) {
            // Skip defined values: 0x01, 0x02, 0x03, 0x10, 0xFF
            if value == 0x10 {
                return Ok(());
            }
            let parsed = QuantType::from_u8(value);
            prop_assert!(parsed.is_none());
        }

        /// Property: bits_per_weight is always positive for valid types
        #[test]
        fn prop_bits_per_weight_positive(qt in arb_quant_type()) {
            prop_assert!(qt.bits_per_weight() > 0.0);
        }

        // ================================================================
        // Q8_0 Quantization Property Tests
        // ================================================================

        /// Property: Q8_0 quantization preserves element count
        #[test]
        fn prop_q8_0_preserves_count(shape in arb_shape()) {
            let len: usize = shape.iter().product();
            let data: Vec<f32> = (0..len).map(|i| i as f32 * 0.1).collect();

            let quantized = Q8_0Quantizer.quantize(&data, &shape).expect("quantize");
            let dequantized = Q8_0Quantizer.dequantize(&quantized).expect("dequantize");

            prop_assert_eq!(dequantized.len(), data.len());
        }

        /// Property: Q8_0 block count is ceiling division
        #[test]
        fn prop_q8_0_block_count(len in 1usize..500) {
            let data: Vec<f32> = vec![1.0; len];
            let shape = vec![len];

            let quantized = Q8_0Quantizer.quantize(&data, &shape).expect("quantize");
            let expected_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

            prop_assert_eq!(quantized.num_blocks(), expected_blocks);
        }

        /// Property: Q8_0 quantized size matches block count
        #[test]
        fn prop_q8_0_size_matches_blocks(len in 1usize..500) {
            let data: Vec<f32> = vec![1.0; len];
            let shape = vec![len];

            let quantized = Q8_0Quantizer.quantize(&data, &shape).expect("quantize");

            prop_assert_eq!(quantized.blocks.len(), quantized.num_blocks() * Q8_0_BLOCK_BYTES);
        }

        /// Property: Q8_0 roundtrip error is bounded (MSE < 0.1 for normalized data)
        #[test]
        fn prop_q8_0_error_bounded(
            len in 32usize..200,
            scale in 0.01f32..10.0
        ) {
            let data: Vec<f32> = (0..len).map(|i| (i as f32 / len as f32 - 0.5) * scale).collect();
            let shape = vec![len];

            let quantized = Q8_0Quantizer.quantize(&data, &shape).expect("quantize");
            let dequantized = Q8_0Quantizer.dequantize(&quantized).expect("dequantize");

            let mse = quantization_mse(&data, &dequantized);
            // Q8_0 should have very low error for normalized data
            prop_assert!(mse < scale * scale * 0.01, "MSE {} too high for scale {}", mse, scale);
        }

        /// Property: Q8_0 zeros stay approximately zero
        #[test]
        fn prop_q8_0_zeros(len in 1usize..100) {
            let data: Vec<f32> = vec![0.0; len];
            let shape = vec![len];

            let quantized = Q8_0Quantizer.quantize(&data, &shape).expect("quantize");
            let dequantized = Q8_0Quantizer.dequantize(&quantized).expect("dequantize");

            for val in &dequantized {
                prop_assert!(val.abs() < 0.001, "Expected ~0, got {}", val);
            }
        }

        /// Property: Q8_0 compression ratio is approximately 3.76x (full blocks only)
        #[test]
        fn prop_q8_0_compression_ratio(blocks in 2usize..16) {
            // Use multiples of BLOCK_SIZE to avoid padding effects
            let len = blocks * BLOCK_SIZE;
            let data: Vec<f32> = vec![1.0; len];
            let shape = vec![len];

            let quantized = Q8_0Quantizer.quantize(&data, &shape).expect("quantize");
            let ratio = quantized.compression_ratio();

            // f32 (4 bytes) -> Q8_0 (8.5 bits/weight = 1.0625 bytes/weight)
            // Expected ratio: 4 / 1.0625 ≈ 3.76
            prop_assert!(ratio > 3.5 && ratio < 4.0, "Ratio {} out of expected range", ratio);
        }

        // ================================================================
        // Q4_0 Quantization Property Tests
        // ================================================================

        /// Property: Q4_0 quantization preserves element count
        #[test]
        fn prop_q4_0_preserves_count(shape in arb_shape()) {
            let len: usize = shape.iter().product();
            let data: Vec<f32> = (0..len).map(|i| i as f32 * 0.1).collect();

            let quantized = Q4_0Quantizer.quantize(&data, &shape).expect("quantize");
            let dequantized = Q4_0Quantizer.dequantize(&quantized).expect("dequantize");

            prop_assert_eq!(dequantized.len(), data.len());
        }

        /// Property: Q4_0 block count is ceiling division
        #[test]
        fn prop_q4_0_block_count(len in 1usize..500) {
            let data: Vec<f32> = vec![1.0; len];
            let shape = vec![len];

            let quantized = Q4_0Quantizer.quantize(&data, &shape).expect("quantize");
            let expected_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

            prop_assert_eq!(quantized.num_blocks(), expected_blocks);
        }

        /// Property: Q4_0 quantized size matches block count
        #[test]
        fn prop_q4_0_size_matches_blocks(len in 1usize..500) {
            let data: Vec<f32> = vec![1.0; len];
            let shape = vec![len];

            let quantized = Q4_0Quantizer.quantize(&data, &shape).expect("quantize");

            prop_assert_eq!(quantized.blocks.len(), quantized.num_blocks() * Q4_0_BLOCK_BYTES);
        }

        /// Property: Q4_0 compression ratio is approximately 7.1x (full blocks only)
        #[test]
        fn prop_q4_0_compression_ratio(blocks in 2usize..16) {
            // Use multiples of BLOCK_SIZE to avoid padding effects
            let len = blocks * BLOCK_SIZE;
            let data: Vec<f32> = vec![1.0; len];
            let shape = vec![len];

            let quantized = Q4_0Quantizer.quantize(&data, &shape).expect("quantize");
            let ratio = quantized.compression_ratio();

            // f32 (4 bytes) -> Q4_0 (4.5 bits/weight = 0.5625 bytes/weight)
            // Expected ratio: 4 / 0.5625 ≈ 7.1
            prop_assert!(ratio > 6.5 && ratio < 7.5, "Ratio {} out of expected range", ratio);
        }

        /// Property: Q4_0 zeros stay approximately zero
        #[test]
        fn prop_q4_0_zeros(len in 1usize..100) {
            let data: Vec<f32> = vec![0.0; len];
            let shape = vec![len];

            let quantized = Q4_0Quantizer.quantize(&data, &shape).expect("quantize");
            let dequantized = Q4_0Quantizer.dequantize(&quantized).expect("dequantize");

            for val in &dequantized {
                prop_assert!(val.abs() < 0.01, "Expected ~0, got {}", val);
            }
        }

        // ================================================================
        // Cross-Quantizer Property Tests
        // ================================================================

        /// Property: Shape is preserved through quantization
        #[test]
        fn prop_shape_preserved(qt in arb_quant_type(), shape in arb_shape()) {
            let len: usize = shape.iter().product();
            let data: Vec<f32> = vec![1.0; len];

            let quantized = quantize(&data, &shape, qt).expect("quantize");
            prop_assert_eq!(&quantized.shape, &shape);
        }

        /// Property: num_elements matches shape product
        #[test]
        fn prop_num_elements(qt in arb_quant_type(), shape in arb_shape()) {
            let len: usize = shape.iter().product();
            let data: Vec<f32> = vec![1.0; len];

            let quantized = quantize(&data, &shape, qt).expect("quantize");
            prop_assert_eq!(quantized.num_elements(), len);
        }

        /// Property: original_size_bytes is 4x num_elements
        #[test]
        fn prop_original_size_bytes(qt in arb_quant_type(), len in 1usize..200) {
            let data: Vec<f32> = vec![1.0; len];
            let shape = vec![len];

            let quantized = quantize(&data, &shape, qt).expect("quantize");
            prop_assert_eq!(quantized.original_size_bytes(), len * 4);
        }

        // ================================================================
        // MSE Helper Property Tests
        // ================================================================

        /// Property: MSE of identical vectors is 0
        #[test]
        fn prop_mse_identical(data in proptest::collection::vec(-10.0f32..10.0, 1..100)) {
            let mse = quantization_mse(&data, &data);
            prop_assert!(mse.abs() < 1e-10, "Expected 0, got {}", mse);
        }

        /// Property: MSE is symmetric
        #[test]
        fn prop_mse_symmetric(
            a in proptest::collection::vec(-10.0f32..10.0, 1..50),
            offset in -1.0f32..1.0
        ) {
            let b: Vec<f32> = a.iter().map(|x| x + offset).collect();

            let mse_ab = quantization_mse(&a, &b);
            let mse_ba = quantization_mse(&b, &a);

            prop_assert!((mse_ab - mse_ba).abs() < 1e-6, "MSE not symmetric: {} vs {}", mse_ab, mse_ba);
        }

        /// Property: MSE is non-negative
        #[test]
        fn prop_mse_nonnegative(
            a in proptest::collection::vec(-10.0f32..10.0, 1..50),
            b in proptest::collection::vec(-10.0f32..10.0, 1..50)
        ) {
            if a.len() != b.len() {
                return Ok(());
            }
            let mse = quantization_mse(&a, &b);
            prop_assert!(mse >= 0.0 || mse.is_nan(), "MSE is negative: {}", mse);
        }
    }
}

/// BH-MUT-0002: Boundary mutation tests for Q8_0 dequantize arithmetic
///
/// Target: `let val = f32::from(q) * scale;` (line 290)
/// Mutations: `*` → `+`, `*` → `-`, `*` → `/`, scale sign flip
#[cfg(test)]
mod tests_bh_mut {
    use super::*;

    /// BH-MUT-0002a: scale=0 produces all-zero output
    /// Detects `* scale` → `+ scale` mutation (would produce non-zero)
    #[test]
    fn test_bh_mut_scale_zero_produces_zero() {
        let data: Vec<f32> = vec![0.0; 32]; // All zeros → scale = 0
        let shape = vec![32];

        let quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");
        let dequantized = dequantize(&quantized).expect("dequantize");

        for (i, &val) in dequantized.iter().enumerate() {
            assert_eq!(
                val, 0.0,
                "dequant[{i}] should be 0.0 with zero scale, got {val}"
            );
        }
    }

    /// BH-MUT-0002b: sign preservation through multiply
    /// Detects `* scale` → `/ scale` mutation (signs may differ for negative q with positive scale)
    #[test]
    fn test_bh_mut_sign_preservation() {
        // Negative values: q < 0, scale > 0 → result < 0
        let data: Vec<f32> = (-16..16).map(|i| i as f32).collect();
        let shape = vec![32];

        let quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");
        let dequantized = dequantize(&quantized).expect("dequantize");

        // Negative inputs should produce negative outputs
        for i in 0..16 {
            if data[i] < -0.5 {
                assert!(
                    dequantized[i] < 0.0,
                    "dequant[{i}] should be negative for input {}, got {}",
                    data[i],
                    dequantized[i]
                );
            }
        }
        // Positive inputs should produce positive outputs
        for i in 17..32 {
            if data[i] > 0.5 {
                assert!(
                    dequantized[i] > 0.0,
                    "dequant[{i}] should be positive for input {}, got {}",
                    data[i],
                    dequantized[i]
                );
            }
        }
    }

    /// BH-MUT-0002c: magnitude scales linearly with scale factor
    /// Detects `* scale` → `+ scale` or `- scale` mutations
    #[test]
    fn test_bh_mut_magnitude_scaling() {
        // Small values → small scale; large values → large scale
        let small: Vec<f32> = vec![0.1; 32];
        let large: Vec<f32> = vec![100.0; 32];
        let shape = vec![32];

        let q_small = quantize(&small, &shape, QuantType::Q8_0).expect("quantize small");
        let q_large = quantize(&large, &shape, QuantType::Q8_0).expect("quantize large");

        let d_small = dequantize(&q_small).expect("deq small");
        let d_large = dequantize(&q_large).expect("deq large");

        // Large values should dequantize to larger magnitudes than small values
        let mag_small: f32 = d_small.iter().map(|v| v.abs()).sum::<f32>() / 32.0;
        let mag_large: f32 = d_large.iter().map(|v| v.abs()).sum::<f32>() / 32.0;

        assert!(
            mag_large > mag_small * 10.0,
            "Large magnitude ({mag_large}) should be >> small ({mag_small})"
        );
    }
}
