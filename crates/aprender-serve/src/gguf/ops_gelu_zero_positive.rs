
// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_zero() {
        let mut input = vec![0.0];
        gelu(&mut input);
        assert!((input[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_positive() {
        let mut input = vec![1.0];
        gelu(&mut input);
        // GELU(1) ≈ 0.8413
        assert!((input[0] - 0.8413).abs() < 0.01);
    }

    #[test]
    fn test_silu_zero() {
        let mut input = vec![0.0];
        silu(&mut input);
        assert!((input[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_silu_positive() {
        let mut input = vec![1.0];
        silu(&mut input);
        // SiLU(1) = 1 * sigmoid(1) ≈ 0.7311
        assert!((input[0] - 0.7311).abs() < 0.01);
    }

    #[test]
    fn test_argmax() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(argmax(&logits), 3);
    }

    #[test]
    fn test_argmax_negative() {
        let logits = vec![-1.0, -0.5, -2.0];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn test_softmax_uniform() {
        let mut logits = vec![0.0, 0.0, 0.0];
        softmax(&mut logits);
        for &p in &logits {
            assert!((p - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mut logits = vec![1.0, 2.0, 3.0];
        softmax(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_bias() {
        let mut output = vec![1.0, 2.0, 3.0, 4.0];
        let bias = vec![0.1, 0.2];
        add_bias(&mut output, &bias);
        assert!((output[0] - 1.1).abs() < 1e-6);
        assert!((output[1] - 2.2).abs() < 1e-6);
        assert!((output[2] - 3.1).abs() < 1e-6);
        assert!((output[3] - 4.2).abs() < 1e-6);
    }

    #[test]
    fn test_rms_norm_unit_weight() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let output = rms_norm(&input, &weight, 1e-5);

        // Check output is normalized
        let sum_sq: f32 = output.iter().map(|x| x * x).sum();
        let rms = (sum_sq / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 0.1); // Approximately unit RMS
    }

    #[test]
    fn test_layer_norm_no_bias() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let output = layer_norm(&input, &weight, None, 1e-5);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_layer_norm_with_bias() {
        let input = vec![1.0, 2.0];
        let weight = vec![1.0, 1.0];
        let bias = vec![0.5, 0.5];
        let output = layer_norm(&input, &weight, Some(&bias), 1e-5);
        assert_eq!(output.len(), 2);
        // GH-278: True LayerNorm subtracts mean (1.5), so normalized = [-1.0, 1.0]
        // After bias: [-0.5, 1.5]
        assert!((output[0] - (-0.5)).abs() < 0.01, "output[0]={}", output[0]);
        assert!((output[1] - 1.5).abs() < 0.01, "output[1]={}", output[1]);
    }

    // =========================================================================
    // Additional tests for rms_norm_into (zero-allocation path)
    // =========================================================================

    #[test]
    fn test_rms_norm_into_unit_weight() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 4];
        rms_norm_into(&input, &weight, 1e-5, &mut output);

        // Check output is normalized
        let sum_sq: f32 = output.iter().map(|x| x * x).sum();
        let rms = (sum_sq / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_rms_norm_into_zeros() {
        let input = vec![0.0, 0.0, 0.0, 0.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![999.0; 4];
        rms_norm_into(&input, &weight, 1e-5, &mut output);

        // Output should be near zero (only eps prevents division by zero)
        for &val in &output {
            assert!(val.abs() < 1e-3);
        }
    }

    #[test]
    fn test_rms_norm_into_with_weight_scaling() {
        let input = vec![1.0, 1.0, 1.0, 1.0];
        let weight = vec![2.0, 0.5, 1.0, 3.0];
        let mut output = vec![0.0; 4];
        rms_norm_into(&input, &weight, 1e-5, &mut output);

        // Each output should be scaled by weight
        // Input RMS = 1.0, so output ≈ weight
        assert!((output[0] - 2.0).abs() < 0.1);
        assert!((output[1] - 0.5).abs() < 0.1);
        assert!((output[2] - 1.0).abs() < 0.1);
        assert!((output[3] - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_rms_norm_into_negative_values() {
        let input = vec![-1.0, -2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 4];
        rms_norm_into(&input, &weight, 1e-5, &mut output);

        // Signs should be preserved
        assert!(output[0] < 0.0);
        assert!(output[1] < 0.0);
        assert!(output[2] > 0.0);
        assert!(output[3] > 0.0);
    }

    // =========================================================================
    // Additional tests for layer_norm_into (zero-allocation path)
    // =========================================================================

    #[test]
    fn test_layer_norm_into_no_bias() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 4];
        layer_norm_into(&input, &weight, None, 1e-5, &mut output);

        assert_eq!(output.len(), 4);
        // Should be normalized
        for &val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_layer_norm_into_with_bias() {
        let input = vec![1.0, 2.0];
        let weight = vec![1.0, 1.0];
        let bias = vec![10.0, 20.0];
        let mut output = vec![0.0; 2];
        layer_norm_into(&input, &weight, Some(&bias), 1e-5, &mut output);

        // Bias should be added
        assert!(output[0] > 9.0);
        assert!(output[1] > 19.0);
    }

    #[test]
    fn test_layer_norm_into_zeros() {
        let input = vec![0.0, 0.0, 0.0, 0.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![999.0; 4];
        layer_norm_into(&input, &weight, None, 1e-5, &mut output);

        // Should handle zeros gracefully
        for &val in &output {
            assert!(val.is_finite());
            assert!(val.abs() < 1e-2);
        }
    }

    // =========================================================================
    // Additional tests for gelu activation
    // =========================================================================

    #[test]
    fn test_gelu_negative() {
        let mut input = vec![-1.0];
        gelu(&mut input);
        // GELU(-1) ≈ -0.1587
        assert!((input[0] - (-0.1587)).abs() < 0.01);
    }

    #[test]
    fn test_gelu_large_positive() {
        let mut input = vec![10.0];
        gelu(&mut input);
        // GELU(x) → x for large positive x
        assert!((input[0] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_gelu_large_negative() {
        let mut input = vec![-10.0];
        gelu(&mut input);
        // GELU(x) → 0 for large negative x
        assert!(input[0].abs() < 0.01);
    }

    #[test]
    fn test_gelu_batch() {
        let mut input = vec![0.0, 1.0, -1.0, 2.0];
        gelu(&mut input);
        assert!((input[0] - 0.0).abs() < 0.01);
        assert!((input[1] - 0.8413).abs() < 0.01);
        assert!((input[2] - (-0.1587)).abs() < 0.01);
        assert!(input[3] > 1.9); // GELU(2) ≈ 1.96
    }

    // =========================================================================
    // Additional tests for silu activation
    // =========================================================================

    #[test]
    fn test_silu_negative() {
        let mut input = vec![-1.0];
        silu(&mut input);
        // SiLU(-1) = -1 * sigmoid(-1) ≈ -0.2689
        assert!((input[0] - (-0.2689)).abs() < 0.01);
    }

    #[test]
    fn test_silu_large_positive() {
        let mut input = vec![10.0];
        silu(&mut input);
        // SiLU(x) → x for large positive x (sigmoid → 1)
        assert!((input[0] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_silu_large_negative() {
        let mut input = vec![-10.0];
        silu(&mut input);
        // SiLU(x) → 0 for large negative x (sigmoid → 0)
        assert!(input[0].abs() < 0.001);
    }

    #[test]
    fn test_silu_batch() {
        let mut input = vec![0.0, 1.0, -1.0, 2.0];
        silu(&mut input);
        assert!((input[0] - 0.0).abs() < 0.01);
        assert!((input[1] - 0.7311).abs() < 0.01);
        assert!((input[2] - (-0.2689)).abs() < 0.01);
        assert!(input[3] > 1.7); // SiLU(2) ≈ 1.76
    }

    // =========================================================================
    // Additional tests for argmax
    // =========================================================================

    #[test]
    fn test_argmax_single_element() {
        let logits = vec![42.0];
        assert_eq!(argmax(&logits), 0);
    }

    #[test]
    fn test_argmax_all_same() {
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        // First occurrence wins
        assert_eq!(argmax(&logits), 0);
    }

    #[test]
    fn test_argmax_infinity() {
        let logits = vec![1.0, f32::INFINITY, 3.0];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn test_argmax_with_nan() {
        // NaN should not be selected as max
        let logits = vec![f32::NAN, 1.0, 2.0];
        // NaN comparisons are false, so 2.0 at index 2 should win
        assert_eq!(argmax(&logits), 2);
    }

    // =========================================================================
    // Additional tests for softmax
    // =========================================================================

    #[test]
    fn test_softmax_single_element() {
        let mut logits = vec![5.0];
        softmax(&mut logits);
        assert!((logits[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_large_values() {
        // Should not overflow due to numerical stability (subtract max)
        let mut logits = vec![1000.0, 1001.0, 1002.0];
        softmax(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Largest should have highest probability
        assert!(logits[2] > logits[1]);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_softmax_negative_values() {
        let mut logits = vec![-1.0, -2.0, -3.0];
        softmax(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Least negative should have highest probability
        assert!(logits[0] > logits[1]);
        assert!(logits[1] > logits[2]);
    }

    #[test]
    fn test_softmax_dominance() {
        // One very large value should dominate
        let mut logits = vec![0.0, 0.0, 100.0];
        softmax(&mut logits);
        assert!(logits[2] > 0.99);
        assert!(logits[0] < 0.01);
        assert!(logits[1] < 0.01);
    }

    // =========================================================================
    // Additional tests for add_bias
    // =========================================================================

    #[test]
    fn test_add_bias_single_element() {
        let mut output = vec![1.0];
        let bias = vec![0.5];
        add_bias(&mut output, &bias);
        assert!((output[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_add_bias_multiple_sequences() {
        let mut output = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 sequences of dim 2
        let bias = vec![0.1, 0.2];
        add_bias(&mut output, &bias);
        assert!((output[0] - 1.1).abs() < 1e-6);
        assert!((output[1] - 2.2).abs() < 1e-6);
        assert!((output[2] - 3.1).abs() < 1e-6);
        assert!((output[3] - 4.2).abs() < 1e-6);
        assert!((output[4] - 5.1).abs() < 1e-6);
        assert!((output[5] - 6.2).abs() < 1e-6);
    }

    #[test]
    fn test_add_bias_negative() {
        let mut output = vec![1.0, 2.0];
        let bias = vec![-0.5, -1.0];
        add_bias(&mut output, &bias);
        assert!((output[0] - 0.5).abs() < 1e-6);
        assert!((output[1] - 1.0).abs() < 1e-6);
    }

    // =========================================================================
    // Additional tests for rms_norm
    // =========================================================================

    #[test]
    fn test_rms_norm_multiple_sequences() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 sequences of dim 4
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let output = rms_norm(&input, &weight, 1e-5);

        assert_eq!(output.len(), 8);
        // Each sequence should be independently normalized
        let seq1_sum_sq: f32 = output[0..4].iter().map(|x| x * x).sum();
        let seq2_sum_sq: f32 = output[4..8].iter().map(|x| x * x).sum();
        let rms1 = (seq1_sum_sq / 4.0).sqrt();
        let rms2 = (seq2_sum_sq / 4.0).sqrt();
        assert!((rms1 - 1.0).abs() < 0.1);
        assert!((rms2 - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_rms_norm_zeros() {
        let input = vec![0.0, 0.0, 0.0, 0.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let output = rms_norm(&input, &weight, 1e-5);

        // Output should be near zero
        for &val in &output {
            assert!(val.abs() < 1e-3);
        }
    }

    #[test]
    fn test_rms_norm_preserves_sign() {
        let input = vec![-3.0, 2.0, -1.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let output = rms_norm(&input, &weight, 1e-5);

        assert!(output[0] < 0.0);
        assert!(output[1] > 0.0);
        assert!(output[2] < 0.0);
        assert!(output[3] > 0.0);
    }

    // =========================================================================
    // Additional tests for layer_norm
    // =========================================================================

    #[test]
    fn test_layer_norm_multiple_sequences() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 sequences of dim 3
        let weight = vec![1.0, 1.0, 1.0];
        let output = layer_norm(&input, &weight, None, 1e-5);

        assert_eq!(output.len(), 6);
        // Each sequence should be independently normalized
        for &val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_layer_norm_preserves_sign() {
        let input = vec![-3.0, 2.0, -1.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let output = layer_norm(&input, &weight, None, 1e-5);

        assert!(output[0] < 0.0);
        assert!(output[1] > 0.0);
        assert!(output[2] < 0.0);
        assert!(output[3] > 0.0);
    }

    #[test]
    fn test_layer_norm_weight_scaling() {
        // GH-278: True LayerNorm - verify weight scaling of normalized values
        // Use unit weight first, then scaled weight, check ratio
        let input = vec![1.0, 3.0, 2.0, 4.0];
        let unit_weight = vec![1.0, 1.0, 1.0, 1.0];
        let scaled_weight = vec![2.0, 0.5, 1.0, 3.0];
        let unit_out = layer_norm(&input, &unit_weight, None, 1e-5);
        let scaled_out = layer_norm(&input, &scaled_weight, None, 1e-5);

        // scaled_out[i] should equal unit_out[i] * scaled_weight[i]
        for i in 0..4 {
            let expected = unit_out[i] * scaled_weight[i];
            assert!(
                (scaled_out[i] - expected).abs() < 1e-5,
                "index {}: expected {}, got {}", i, expected, scaled_out[i]
            );
        }
    }
}
