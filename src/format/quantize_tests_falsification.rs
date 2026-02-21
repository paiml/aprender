
/// Falsification tests per spec v3.0.0 Section BB (Quantization)
#[cfg(test)]
mod tests_falsification_bb {
    use super::*;

    /// BB1: Q4_0 round-trip reconstruction error must be <5%
    /// Falsification: If error >5%, quantization is lossy beyond acceptable threshold
    #[test]
    fn test_bb1_q4_0_roundtrip_error_under_5_percent() {
        // Generate realistic weight distribution (normal-ish around 0)
        let data: Vec<f32> = (0..1024)
            .map(|i| {
                let x = (i as f32 - 512.0) / 512.0; // Range [-1, 1]
                x * 0.1 // Small weights typical in neural nets
            })
            .collect();
        let shape = vec![1024];

        let quantized = quantize(&data, &shape, QuantType::Q4_0).expect("quantize failed");
        let dequantized = dequantize(&quantized).expect("dequantize failed");

        // Calculate relative error
        let mut total_sq_error = 0.0_f64;
        let mut total_sq_orig = 0.0_f64;
        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            total_sq_error += ((*orig - *deq) as f64).powi(2);
            total_sq_orig += (*orig as f64).powi(2);
        }

        let relative_error = if total_sq_orig > 0.0 {
            (total_sq_error / total_sq_orig).sqrt()
        } else {
            0.0
        };

        assert!(
            relative_error < 0.05,
            "BB1 FALSIFIED: Q4_0 relative error {:.2}% exceeds 5% threshold",
            relative_error * 100.0
        );
    }

    /// BB3: Quantization must be deterministic
    /// Falsification: Same input produces different output
    #[test]
    fn test_bb3_quantization_deterministic() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
        let shape = vec![128];

        // Run quantization 10 times
        let mut results: Vec<Vec<u8>> = Vec::new();
        for _ in 0..10 {
            let quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");
            results.push(quantized.blocks.clone());
        }

        // All results must be identical
        let first = &results[0];
        for (i, result) in results.iter().enumerate().skip(1) {
            assert_eq!(
                first, result,
                "BB3 FALSIFIED: Quantization run {} differs from run 0",
                i
            );
        }
    }

    /// BB3b: Q4_0 quantization must also be deterministic
    #[test]
    fn test_bb3_q4_0_deterministic() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
        let shape = vec![128];

        let q1 = quantize(&data, &shape, QuantType::Q4_0).expect("quantize 1");
        let q2 = quantize(&data, &shape, QuantType::Q4_0).expect("quantize 2");

        assert_eq!(
            q1.blocks, q2.blocks,
            "BB3 FALSIFIED: Q4_0 quantization is non-deterministic"
        );
    }

    /// BB4: Block size must be 32 elements (GGUF compatibility)
    /// Falsification: Non-32 block size is accepted
    #[test]
    fn test_bb4_block_size_is_32() {
        assert_eq!(
            BLOCK_SIZE, 32,
            "BB4 FALSIFIED: Block size is {} instead of 32",
            BLOCK_SIZE
        );
    }

    /// BB4b: Verify quantized blocks use correct size
    #[test]
    fn test_bb4_quantized_block_size_correct() {
        let data: Vec<f32> = vec![1.0; 64]; // 2 blocks
        let shape = vec![64];

        let quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");

        assert_eq!(
            quantized.block_size, 32,
            "BB4 FALSIFIED: QuantizedBlock has wrong block_size: {}",
            quantized.block_size
        );
    }

    /// BB5: Scale factors must be stored and applied correctly
    /// Falsification: dequant(quant(x)) != x / scale (approximately)
    #[test]
    fn test_bb5_scale_factors_correct() {
        // Use known values to verify scale calculation
        let data: Vec<f32> = vec![127.0; 32]; // Max value for Q8_0
        let shape = vec![32];

        let quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");

        // Extract scale from block (first 2 bytes as f16)
        let scale_bytes = [quantized.blocks[0], quantized.blocks[1]];
        let scale = f16::from_le_bytes(scale_bytes).to_f32();

        // Scale should be max_abs / 127 = 127 / 127 = 1.0
        assert!(
            (scale - 1.0).abs() < 0.01,
            "BB5 FALSIFIED: Scale {:.4} != expected 1.0",
            scale
        );

        // Dequantized values should match original
        let dequantized = dequantize(&quantized).expect("dequantize");
        for (i, (orig, deq)) in data.iter().zip(dequantized.iter()).enumerate() {
            assert!(
                (orig - deq).abs() < 0.5,
                "BB5 FALSIFIED: Element {} differs: {} vs {}",
                i,
                orig,
                deq
            );
        }
    }

    /// BB6: Verify mixed quantization doesn't corrupt data
    /// (Test that Q8 and Q4 can coexist in same workflow)
    #[test]
    fn test_bb6_mixed_quantization_no_corruption() {
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let shape = vec![64];

        // Quantize same data with both methods
        let q8 = quantize(&data, &shape, QuantType::Q8_0).expect("Q8_0");
        let q4 = quantize(&data, &shape, QuantType::Q4_0).expect("Q4_0");

        // Both should dequantize without error
        let d8 = dequantize(&q8).expect("dequantize Q8_0");
        let d4 = dequantize(&q4).expect("dequantize Q4_0");

        assert_eq!(d8.len(), data.len(), "BB6 FALSIFIED: Q8_0 length mismatch");
        assert_eq!(d4.len(), data.len(), "BB6 FALSIFIED: Q4_0 length mismatch");

        // Q8_0 should be more accurate than Q4_0
        let mse8 = quantization_mse(&data, &d8);
        let mse4 = quantization_mse(&data, &d4);

        assert!(
            mse8 < mse4,
            "BB6 FALSIFIED: Q8_0 MSE ({}) should be less than Q4_0 MSE ({})",
            mse8,
            mse4
        );
    }
}

// ============================================================================
// Property-Based Falsification Tests (per spec v3.0.0 Section 2.7)
// Uses proptest to automatically generate falsifying inputs
// ============================================================================
#[cfg(test)]
mod tests_proptest_bb {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// BB3-PROP: Quantization is deterministic for ANY valid input
        /// FALSIFICATION: Any input produces non-deterministic output
        #[test]
        fn prop_bb3_quantization_deterministic(
            weights in prop::collection::vec(-1.0f32..1.0, 32..256),
        ) {
            let shape = vec![weights.len()];
            let q1 = quantize(&weights, &shape, QuantType::Q8_0).expect("quantize 1");
            let q2 = quantize(&weights, &shape, QuantType::Q8_0).expect("quantize 2");

            prop_assert_eq!(
                q1.blocks, q2.blocks,
                "BB3-PROP FALSIFIED: Non-deterministic quantization"
            );
        }

        /// BB1-PROP: Q8_0 round-trip error is bounded for ANY input
        /// FALSIFICATION: Any input produces >1% relative error for Q8_0
        #[test]
        fn prop_bb1_q8_roundtrip_bounded(
            weights in prop::collection::vec(-10.0f32..10.0, 32..256),
        ) {
            let shape = vec![weights.len()];
            let quantized = quantize(&weights, &shape, QuantType::Q8_0).expect("quantize");
            let dequantized = dequantize(&quantized).expect("dequantize");

            // Calculate MSE
            let mse: f64 = weights.iter()
                .zip(dequantized.iter())
                .map(|(o, d)| ((*o - *d) as f64).powi(2))
                .sum::<f64>() / weights.len() as f64;

            // Q8_0 should have very low error (< 0.1 for this range)
            prop_assert!(
                mse < 0.1,
                "BB1-PROP FALSIFIED: Q8_0 MSE {} exceeds 0.1",
                mse
            );
        }

        /// BB5-PROP: Scale factors are always positive for non-zero input
        /// FALSIFICATION: Scale factor is zero or negative for non-zero input
        #[test]
        fn prop_bb5_scale_positive(
            weights in prop::collection::vec(0.1f32..1.0, 32..64),
        ) {
            let shape = vec![weights.len()];
            let quantized = quantize(&weights, &shape, QuantType::Q8_0).expect("quantize");

            // Extract scale from first block
            if quantized.blocks.len() >= 2 {
                let scale_bytes = [quantized.blocks[0], quantized.blocks[1]];
                let scale = f16::from_le_bytes(scale_bytes).to_f32();

                prop_assert!(
                    scale > 0.0,
                    "BB5-PROP FALSIFIED: Scale {} is not positive",
                    scale
                );
            }
        }

        /// BB6-PROP: Q8_0 always more accurate than Q4_0
        /// FALSIFICATION: Q4_0 has lower MSE than Q8_0
        #[test]
        fn prop_bb6_q8_more_accurate_than_q4(
            weights in prop::collection::vec(-1.0f32..1.0, 64..128),
        ) {
            let shape = vec![weights.len()];

            let q8 = quantize(&weights, &shape, QuantType::Q8_0).expect("Q8_0");
            let q4 = quantize(&weights, &shape, QuantType::Q4_0).expect("Q4_0");

            let d8 = dequantize(&q8).expect("dequantize Q8");
            let d4 = dequantize(&q4).expect("dequantize Q4");

            let mse8: f64 = weights.iter()
                .zip(d8.iter())
                .map(|(o, d)| ((*o - *d) as f64).powi(2))
                .sum::<f64>() / weights.len() as f64;

            let mse4: f64 = weights.iter()
                .zip(d4.iter())
                .map(|(o, d)| ((*o - *d) as f64).powi(2))
                .sum::<f64>() / weights.len() as f64;

            prop_assert!(
                mse8 <= mse4,
                "BB6-PROP FALSIFIED: Q8_0 MSE {} > Q4_0 MSE {}",
                mse8, mse4
            );
        }

        /// BB4-PROP: All quantized blocks use correct block size
        /// FALSIFICATION: Block size differs from 32
        #[test]
        fn prop_bb4_block_size_always_32(
            len in 32usize..512,
        ) {
            let weights: Vec<f32> = (0..len).map(|i| i as f32 * 0.01).collect();
            let shape = vec![weights.len()];

            let quantized = quantize(&weights, &shape, QuantType::Q8_0).expect("quantize");

            prop_assert_eq!(
                quantized.block_size, 32,
                "BB4-PROP FALSIFIED: block_size is {} instead of 32",
                quantized.block_size
            );
        }
    }
}
