//! Tests for quantization granularity module

#![allow(clippy::module_inception)]
#[cfg(test)]
mod tests {
    use crate::quant::granularity::{
        calibrate_per_channel, calibrate_per_group, calibrate_per_tensor, compare_granularities,
        dequantize_tensor, dequantize_with_params, quantization_mse, quantize_tensor,
        quantize_with_params, QuantGranularity, QuantMode, QuantParams,
    };
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    #[test]
    fn test_per_tensor_symmetric_8bit() {
        let values = vec![1.0, -2.0, 3.0, -4.0, 5.0, -5.0];
        let params = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);

        assert_eq!(params.scales.len(), 1);
        assert!(params.zero_points.is_empty());
        assert_eq!(params.granularity, QuantGranularity::PerTensor);

        let quantized = quantize_with_params(&values, &params);
        let dequantized = dequantize_with_params(&quantized, &params);

        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            assert_abs_diff_eq!(orig, deq, epsilon = 0.1);
        }
    }

    #[test]
    fn test_per_tensor_asymmetric_8bit() {
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]; // All positive
        let params = calibrate_per_tensor(&values, 8, QuantMode::Asymmetric);

        assert_eq!(params.scales.len(), 1);
        assert_eq!(params.zero_points.len(), 1);
        assert_eq!(params.mode, QuantMode::Asymmetric);

        let quantized = quantize_with_params(&values, &params);
        let dequantized = dequantize_with_params(&quantized, &params);

        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            assert_abs_diff_eq!(orig, deq, epsilon = 0.1);
        }
    }

    #[test]
    fn test_per_channel_symmetric_8bit() {
        // 2 channels, 4 features each
        // Channel 0: small values, Channel 1: large values
        let values = vec![
            0.1, 0.2, -0.1, -0.2, // Channel 0
            10.0, 20.0, -10.0, -20.0, // Channel 1
        ];
        let params = calibrate_per_channel(&values, 2, 8, QuantMode::Symmetric);

        assert_eq!(params.scales.len(), 2);
        assert!(params.zero_points.is_empty());

        // Different scales for different channels
        assert!(params.scales[0] < params.scales[1]);

        let quantized = quantize_with_params(&values, &params);
        let dequantized = dequantize_with_params(&quantized, &params);

        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            let rel_error = (orig - deq).abs() / orig.abs().max(0.01);
            assert!(rel_error < 0.1, "Error too large: {orig} vs {deq}");
        }
    }

    #[test]
    fn test_per_channel_better_than_per_tensor() {
        // Values with very different scales per channel
        let values = vec![
            0.01, 0.02, -0.01, -0.02, // Channel 0: tiny
            100.0, 200.0, -100.0, -200.0, // Channel 1: huge
        ];

        let (pt_mse, pc_mse) = compare_granularities(&values, 2, 8);

        // Per-channel should have lower error
        assert!(
            pc_mse <= pt_mse,
            "Per-channel MSE ({pc_mse}) should be <= per-tensor MSE ({pt_mse})"
        );
    }

    #[test]
    fn test_per_group_quantization() {
        let values: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let params = calibrate_per_group(&values, 10, 8, QuantMode::Symmetric);

        assert_eq!(params.scales.len(), 10); // 100 values / 10 per group

        let quantized = quantize_with_params(&values, &params);
        let dequantized = dequantize_with_params(&quantized, &params);

        let mse = quantization_mse(&values, &dequantized);
        assert!(mse < 0.01, "MSE {mse} too large");
    }

    #[test]
    fn test_4bit_quantization() {
        let values = vec![1.0, -2.0, 3.0, -4.0, 5.0, -5.0, 6.0, -7.0];
        let params = calibrate_per_tensor(&values, 4, QuantMode::Symmetric);

        // 4-bit symmetric: qmax = 7
        assert!(params.scales[0] == 7.0 / 7.0); // max_abs / 7

        let quantized = quantize_with_params(&values, &params);
        let dequantized = dequantize_with_params(&quantized, &params);

        // 4-bit has lower precision
        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            assert_abs_diff_eq!(orig, deq, epsilon = 1.5);
        }
    }

    #[test]
    fn test_quantized_tensor_struct() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];

        let quantized =
            quantize_tensor(&values, &shape, QuantGranularity::PerChannel, QuantMode::Symmetric, 8);

        assert_eq!(quantized.shape, vec![2, 3]);
        assert_eq!(quantized.params.scales.len(), 2);
        assert_eq!(quantized.data.len(), 6);

        let dequantized = dequantize_tensor(&quantized);
        assert_eq!(dequantized.len(), 6);
    }

    #[test]
    fn test_memory_bytes() {
        let values = vec![1.0; 100];
        let shape = vec![100];

        let quantized =
            quantize_tensor(&values, &shape, QuantGranularity::PerTensor, QuantMode::Symmetric, 8);

        // 100 bytes data + 4 bytes scale = 104 bytes
        assert_eq!(quantized.memory_bytes(), 104);
    }

    #[test]
    fn test_empty_values() {
        let values: Vec<f32> = vec![];
        let params = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);
        assert_eq!(params.scales[0], 1e-8 / 127.0);
    }

    #[test]
    fn test_zeros() {
        let values = vec![0.0; 10];
        let params = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);

        let quantized = quantize_with_params(&values, &params);
        let dequantized = dequantize_with_params(&quantized, &params);

        for val in dequantized {
            assert_abs_diff_eq!(val, 0.0, epsilon = 1e-6);
        }
    }

    // Property tests

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_per_tensor_round_trip(values in proptest::collection::vec(-100.0f32..100.0, 1..100)) {
            let params = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);
            let quantized = quantize_with_params(&values, &params);
            let dequantized = dequantize_with_params(&quantized, &params);

            prop_assert_eq!(dequantized.len(), values.len());

            let mse = quantization_mse(&values, &dequantized);
            // 8-bit should have low error
            prop_assert!(mse < 10.0, "MSE {} too large", mse);
        }

        #[test]
        fn prop_per_channel_scales_count(
            num_channels in 1usize..10,
            features_per_channel in 1usize..20
        ) {
            let values: Vec<f32> = (0..num_channels * features_per_channel)
                .map(|i| i as f32 * 0.1)
                .collect();

            let params = calibrate_per_channel(&values, num_channels, 8, QuantMode::Symmetric);

            prop_assert_eq!(params.scales.len(), num_channels);
        }

        #[test]
        fn prop_per_group_scales_count(
            total_values in 10usize..200,
            group_size in 1usize..20
        ) {
            let values: Vec<f32> = (0..total_values)
                .map(|i| i as f32 * 0.1)
                .collect();

            let params = calibrate_per_group(&values, group_size, 8, QuantMode::Symmetric);

            let expected_groups = total_values.div_ceil(group_size);
            prop_assert_eq!(params.scales.len(), expected_groups);
        }

        #[test]
        fn prop_per_channel_better_or_equal(
            num_channels in 2usize..5,
            features_per_channel in 5usize..20,
            scale_factor in 1.0f32..100.0
        ) {
            // Generate values where channels have different scales
            let values: Vec<f32> = (0..num_channels)
                .flat_map(|ch| {
                    let ch_scale = (ch as f32 + 1.0) * scale_factor;
                    (0..features_per_channel).map(move |i| (i as f32 * 0.1 - 0.5) * ch_scale)
                })
                .collect();

            let (pt_mse, pc_mse) = compare_granularities(&values, num_channels, 8);

            // Per-channel should be at least as good as per-tensor
            prop_assert!(
                pc_mse <= pt_mse * 1.01, // Small tolerance for floating point
                "Per-channel MSE ({}) should be <= per-tensor MSE ({})",
                pc_mse,
                pt_mse
            );
        }

        #[test]
        fn prop_symmetric_zero_mean(values in proptest::collection::vec(-100.0f32..100.0, 10..100)) {
            // For symmetric quantization, zero should map to zero
            let params = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);

            let zero_quantized = quantize_with_params(&[0.0], &params);
            let zero_dequantized = dequantize_with_params(&zero_quantized, &params);

            prop_assert!(zero_dequantized[0].abs() < 0.01, "Zero should map to ~zero");
        }

        #[test]
        fn prop_4bit_vs_8bit_accuracy(values in proptest::collection::vec(-100.0f32..100.0, 10..100)) {
            let params_8bit = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);
            let params_4bit = calibrate_per_tensor(&values, 4, QuantMode::Symmetric);

            let q8 = quantize_with_params(&values, &params_8bit);
            let q4 = quantize_with_params(&values, &params_4bit);

            let d8 = dequantize_with_params(&q8, &params_8bit);
            let d4 = dequantize_with_params(&q4, &params_4bit);

            let mse_8bit = quantization_mse(&values, &d8);
            let mse_4bit = quantization_mse(&values, &d4);

            // 8-bit should generally be better than 4-bit
            prop_assert!(
                mse_8bit <= mse_4bit * 1.01,
                "8-bit MSE ({}) should be <= 4-bit MSE ({})",
                mse_8bit,
                mse_4bit
            );
        }
    }

    // Additional edge case tests

    #[test]
    fn test_per_channel_asymmetric() {
        let values = vec![
            0.0, 1.0, 2.0, 3.0, // Channel 0
            10.0, 11.0, 12.0, 13.0, // Channel 1
        ];
        let params = calibrate_per_channel(&values, 2, 8, QuantMode::Asymmetric);

        assert_eq!(params.scales.len(), 2);
        assert_eq!(params.zero_points.len(), 2);
        assert!(params.is_asymmetric());

        // Just verify quantization produces valid output
        let quantized = quantize_with_params(&values, &params);
        assert_eq!(quantized.len(), values.len());
    }

    #[test]
    fn test_per_group_asymmetric() {
        let values: Vec<f32> = (0..40).map(|i| i as f32).collect();
        let params = calibrate_per_group(&values, 10, 8, QuantMode::Asymmetric);

        assert_eq!(params.scales.len(), 4); // 40 values / 10 per group
        assert_eq!(params.zero_points.len(), 4);
        assert_eq!(params.granularity, QuantGranularity::PerGroup(10));

        // Just verify quantization produces valid output
        let quantized = quantize_with_params(&values, &params);
        assert_eq!(quantized.len(), values.len());
    }

    #[test]
    fn test_per_channel_empty_values() {
        let values: Vec<f32> = vec![];
        let params = calibrate_per_channel(&values, 0, 8, QuantMode::Symmetric);

        assert_eq!(params.scales.len(), 1);
        assert_eq!(params.scales[0], 1.0);
    }

    #[test]
    fn test_per_group_single_group() {
        let values = vec![1.0, 2.0, 3.0];
        let params = calibrate_per_group(&values, 10, 8, QuantMode::Symmetric);

        // Group size > values count: should be 1 group
        assert_eq!(params.num_groups(), 1);
    }

    #[test]
    fn test_quant_params_num_groups() {
        let params = QuantParams {
            scales: vec![1.0, 2.0, 3.0],
            zero_points: vec![],
            granularity: QuantGranularity::PerChannel,
            mode: QuantMode::Symmetric,
            bits: 8,
        };
        assert_eq!(params.num_groups(), 3);
    }

    #[test]
    fn test_quant_params_is_asymmetric() {
        let symmetric = QuantParams {
            scales: vec![1.0],
            zero_points: vec![],
            granularity: QuantGranularity::PerTensor,
            mode: QuantMode::Symmetric,
            bits: 8,
        };
        assert!(!symmetric.is_asymmetric());

        let asymmetric = QuantParams {
            scales: vec![1.0],
            zero_points: vec![128],
            granularity: QuantGranularity::PerTensor,
            mode: QuantMode::Asymmetric,
            bits: 8,
        };
        assert!(asymmetric.is_asymmetric());
    }

    #[test]
    fn test_quantization_mse_mismatched_lengths() {
        let original = vec![1.0, 2.0, 3.0];
        let dequantized = vec![1.0, 2.0];
        let mse = quantization_mse(&original, &dequantized);
        assert_eq!(mse, f32::MAX);
    }

    #[test]
    fn test_quantization_mse_empty() {
        let original: Vec<f32> = vec![];
        let dequantized: Vec<f32> = vec![];
        let mse = quantization_mse(&original, &dequantized);
        assert_eq!(mse, f32::MAX);
    }

    #[test]
    fn test_quantize_tensor_per_group() {
        let values: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let shape = vec![100];

        let quantized = quantize_tensor(
            &values,
            &shape,
            QuantGranularity::PerGroup(10),
            QuantMode::Symmetric,
            8,
        );

        assert_eq!(quantized.shape, vec![100]);
        assert_eq!(quantized.params.scales.len(), 10);
        assert_eq!(quantized.data.len(), 100);
    }

    #[test]
    fn test_dequantize_tensor() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];

        let quantized =
            quantize_tensor(&values, &shape, QuantGranularity::PerChannel, QuantMode::Symmetric, 8);

        let dequantized = dequantize_tensor(&quantized);
        assert_eq!(dequantized.len(), 6);

        let mse = quantization_mse(&values, &dequantized);
        assert!(mse < 0.1, "MSE {mse} too large");
    }

    #[test]
    fn test_4bit_per_channel() {
        let values = vec![
            0.1, 0.2, 0.3, 0.4, // Channel 0: small
            1.0, 2.0, 3.0, 4.0, // Channel 1: larger
        ];
        let params = calibrate_per_channel(&values, 2, 4, QuantMode::Symmetric);

        assert_eq!(params.bits, 4);
        assert_eq!(params.scales.len(), 2);

        let quantized = quantize_with_params(&values, &params);
        let dequantized = dequantize_with_params(&quantized, &params);

        // 4-bit has coarser quantization
        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            assert_abs_diff_eq!(orig, deq, epsilon = 1.0);
        }
    }

    #[test]
    fn test_quantized_tensor_memory_with_zero_points() {
        let values = vec![1.0; 100];
        let shape = vec![100];

        let quantized =
            quantize_tensor(&values, &shape, QuantGranularity::PerTensor, QuantMode::Asymmetric, 8);

        // 100 bytes data + 4 bytes scale + 4 bytes zero_point = 108 bytes
        assert_eq!(quantized.memory_bytes(), 108);
    }

    #[test]
    fn test_per_channel_single_channel() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let params = calibrate_per_channel(&values, 1, 8, QuantMode::Symmetric);

        assert_eq!(params.scales.len(), 1);
    }

    #[test]
    fn test_negative_values_asymmetric() {
        let values = vec![-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0];
        let params = calibrate_per_tensor(&values, 8, QuantMode::Asymmetric);

        let quantized = quantize_with_params(&values, &params);
        let dequantized = dequantize_with_params(&quantized, &params);

        let mse = quantization_mse(&values, &dequantized);
        assert!(mse < 0.1, "MSE {mse} too large for negative values");
    }
}
