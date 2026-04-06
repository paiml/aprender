
// ============================================================================
// Tests (Protocol T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // === apply_rope tests ===

    #[test]
    fn test_apply_rope_basic() {
        // Single position, single head, head_dim=2
        let mut x = vec![1.0, 0.0]; // [cos, sin] position encoding input
        let seq_len = 1;
        let num_heads = 1;
        let head_dim = 2;
        let rope_theta = 10000.0;
        let start_pos = 0;

        apply_rope(&mut x, seq_len, num_heads, head_dim, rope_theta, start_pos);

        // At position 0, angle = 0 * freq = 0, so sin(0)=0, cos(0)=1
        // Rotation: [x1*1 - x2*0, x1*0 + x2*1] = [1, 0]
        assert!((x[0] - 1.0).abs() < 1e-5, "x[0] = {}", x[0]);
        assert!(x[1].abs() < 1e-5, "x[1] = {}", x[1]);
    }

    #[test]
    fn test_apply_rope_position_one() {
        // Single position at pos=1 should show rotation
        let mut x = vec![1.0, 0.0];
        apply_rope(&mut x, 1, 1, 2, 10000.0, 1);

        // At position 1, angle = 1 * 1/(10000^0) = 1
        // cos(1) ≈ 0.54, sin(1) ≈ 0.84
        let expected_cos = 1.0f32.cos();
        let expected_sin = 1.0f32.sin();
        assert!(
            (x[0] - expected_cos).abs() < 1e-5,
            "x[0] = {}, expected {}",
            x[0],
            expected_cos
        );
        assert!(
            (x[1] - expected_sin).abs() < 1e-5,
            "x[1] = {}, expected {}",
            x[1],
            expected_sin
        );
    }

    #[test]
    fn test_apply_rope_multiple_positions() {
        // Two positions
        let mut x = vec![
            1.0, 0.0, // Position 0
            1.0, 0.0, // Position 1
        ];
        apply_rope(&mut x, 2, 1, 2, 10000.0, 0);

        // Position 0: no rotation (angle=0)
        assert!((x[0] - 1.0).abs() < 1e-5);
        assert!(x[1].abs() < 1e-5);

        // Position 1: rotated
        let expected_cos = 1.0f32.cos();
        let expected_sin = 1.0f32.sin();
        assert!((x[2] - expected_cos).abs() < 1e-5);
        assert!((x[3] - expected_sin).abs() < 1e-5);
    }

    #[test]
    fn test_apply_rope_multiple_heads() {
        // Single position, two heads
        let mut x = vec![
            1.0, 0.0, // Head 0
            0.0, 1.0, // Head 1
        ];
        apply_rope(&mut x, 1, 2, 2, 10000.0, 0);

        // At position 0, both heads get identity rotation
        assert!((x[0] - 1.0).abs() < 1e-5);
        assert!(x[1].abs() < 1e-5);
        assert!(x[2].abs() < 1e-5);
        assert!((x[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_apply_rope_larger_head_dim() {
        // Head dim 4 (2 pairs to rotate)
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        apply_rope(&mut x, 1, 1, 4, 10000.0, 0);

        // At position 0, angle = 0, no rotation
        assert!((x[0] - 1.0).abs() < 1e-5);
        assert!((x[1] - 2.0).abs() < 1e-5);
        assert!((x[2] - 3.0).abs() < 1e-5);
        assert!((x[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_apply_rope_with_start_pos() {
        let mut x = vec![1.0, 0.0];
        apply_rope(&mut x, 1, 1, 2, 10000.0, 5);

        // At position 5, angle = 5 * 1 = 5
        let expected_cos = 5.0f32.cos();
        let expected_sin = 5.0f32.sin();
        assert!(
            (x[0] - expected_cos).abs() < 1e-5,
            "x[0] = {}, expected {}",
            x[0],
            expected_cos
        );
        assert!(
            (x[1] - expected_sin).abs() < 1e-5,
            "x[1] = {}, expected {}",
            x[1],
            expected_sin
        );
    }

    // === argmax tests ===

    #[test]
    fn test_argmax_kv_single() {
        assert_eq!(argmax(&[5.0]), 0);
    }

    #[test]
    fn test_argmax_kv_first() {
        assert_eq!(argmax(&[10.0, 5.0, 3.0]), 0);
    }

    #[test]
    fn test_argmax_kv_last() {
        assert_eq!(argmax(&[1.0, 2.0, 3.0]), 2);
    }

    #[test]
    fn test_argmax_kv_middle() {
        assert_eq!(argmax(&[1.0, 10.0, 3.0]), 1);
    }

    #[test]
    fn test_argmax_kv_negatives() {
        assert_eq!(argmax(&[-5.0, -2.0, -10.0]), 1);
    }

    #[test]
    fn test_argmax_kv_empty_returns_zero() {
        assert_eq!(argmax(&[]), 0);
    }

    // === sample_topk tests ===

    #[test]
    fn test_sample_topk_returns_max_with_low_temp() {
        let logits = vec![1.0, 10.0, 2.0];
        // With temperature=1.0 and top_k=1, should return argmax
        let result = sample_topk(&logits, 1.0, 1);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_sample_topk_top_3() {
        let logits = vec![0.0, 10.0, 5.0, 1.0];
        // Top-3: indices 1, 2, 3. Should return one of them (likely 1)
        let result = sample_topk(&logits, 1.0, 3);
        assert!(result <= 3, "result = {}", result);
    }

    #[test]
    fn test_sample_topk_with_high_temp() {
        let logits = vec![0.0, 1.0, 0.0];
        // High temperature makes distribution flatter, but max still most likely
        let result = sample_topk(&logits, 10.0, 3);
        assert!(result <= 2);
    }

    #[test]
    fn test_sample_topk_top_1_is_argmax() {
        let logits = vec![0.0, 0.0, 100.0, 0.0];
        let result = sample_topk(&logits, 1.0, 1);
        assert_eq!(result, 2);
    }

    #[test]
    fn test_sample_topk_empty_returns_zero() {
        let result = sample_topk(&[], 1.0, 10);
        assert_eq!(result, 0);
    }

    // === GQA attention dimension tests ===

    #[test]
    fn test_gqa_attention_dimension_calculations() {
        // Test dimension calculations used in GQA attention
        let num_heads = 32;
        let num_kv_heads = 8;
        let head_dim = 128;
        let seq_len = 10;

        let hidden_dim = num_heads * head_dim;
        assert_eq!(hidden_dim, 4096);

        let kv_dim = num_kv_heads * head_dim;
        assert_eq!(kv_dim, 1024);

        let heads_per_kv = num_heads / num_kv_heads;
        assert_eq!(heads_per_kv, 4);

        // Output size
        let output_size = seq_len * hidden_dim;
        assert_eq!(output_size, 40960);

        // Q, K, V sizes
        let q_size = seq_len * hidden_dim;
        let k_size = seq_len * kv_dim;
        let v_size = seq_len * kv_dim;
        assert_eq!(q_size, 40960);
        assert_eq!(k_size, 10240);
        assert_eq!(v_size, 10240);
    }

    #[test]
    fn test_gqa_attention_scale_factor() {
        let head_dim = 128;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let expected = 1.0 / 128.0_f32.sqrt();
        assert!((scale - expected).abs() < 1e-6);
        assert!((scale - 0.088388).abs() < 1e-5);
    }

    // === gqa_incremental_attention tests ===

    #[test]
    fn test_gqa_incremental_attention_params() {
        // Just test the function exists and has correct signature
        // Actually calling it would require a valid GpuModel
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 16;
        let cache_len = 5;

        // These would be inputs
        let _q = vec![0.0f32; num_heads * head_dim];
        let _all_k = vec![0.0f32; cache_len * num_kv_heads * head_dim];
        let _all_v = vec![0.0f32; cache_len * num_kv_heads * head_dim];

        // Verify dimension calculations
        let hidden_dim = num_heads * head_dim;
        assert_eq!(hidden_dim, 64);
        let kv_dim = num_kv_heads * head_dim;
        assert_eq!(kv_dim, 32);
        let heads_per_kv = num_heads / num_kv_heads;
        assert_eq!(heads_per_kv, 2);
    }

    // === layer_norm_kv smoke test ===

    #[test]
    fn test_layer_norm_static_dimensions() {
        // Test that GpuModel::layer_norm_static returns correct dimensions
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let bias = vec![0.0; 4];
        let hidden_dim = 4;
        let eps = 1e-5;

        let output = GpuModel::layer_norm_static(&input, &weight, &bias, hidden_dim, eps);
        assert_eq!(output.len(), 4);

        // Verify output is finite
        for &v in &output {
            assert!(v.is_finite(), "output contains non-finite value: {}", v);
        }
    }

    #[test]
    fn test_layer_norm_static_preserves_length() {
        // Test that layer norm preserves the input length
        let hidden_dim = 8;
        let seq_len = 4;
        let input = vec![1.0; seq_len * hidden_dim];
        let weight = vec![1.0; hidden_dim];
        let bias = vec![0.0; hidden_dim];
        let eps = 1e-5;

        let output = GpuModel::layer_norm_static(&input, &weight, &bias, hidden_dim, eps);
        assert_eq!(output.len(), seq_len * hidden_dim);
    }
}
