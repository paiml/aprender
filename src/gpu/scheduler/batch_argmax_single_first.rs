
// ============================================================================
// Tests (Protocol T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // === argmax tests ===

    #[test]
    fn test_argmax_single_element() {
        assert_eq!(argmax(&[5.0]), 0);
    }

    #[test]
    fn test_argmax_first_is_max() {
        assert_eq!(argmax(&[10.0, 5.0, 3.0, 1.0]), 0);
    }

    #[test]
    fn test_argmax_last_is_max() {
        assert_eq!(argmax(&[1.0, 2.0, 3.0, 4.0]), 3);
    }

    #[test]
    fn test_argmax_middle_is_max() {
        assert_eq!(argmax(&[1.0, 5.0, 3.0]), 1);
    }

    #[test]
    fn test_argmax_with_negatives() {
        assert_eq!(argmax(&[-5.0, -1.0, -10.0]), 1);
    }

    #[test]
    fn test_argmax_with_equal_values() {
        // First max wins
        let result = argmax(&[3.0, 3.0, 3.0]);
        assert!(result <= 2);
    }

    #[test]
    fn test_argmax_large_vocab_uses_chunked_path() {
        // Create logits larger than 1024 to trigger chunked path
        let mut logits = vec![0.0f32; 2000];
        logits[1500] = 10.0;
        assert_eq!(argmax(&logits), 1500);
    }

    #[test]
    fn test_argmax_large_vocab_first_chunk() {
        let mut logits = vec![0.0f32; 5000];
        logits[100] = 10.0;
        assert_eq!(argmax(&logits), 100);
    }

    #[test]
    fn test_argmax_large_vocab_last_chunk() {
        let mut logits = vec![0.0f32; 5000];
        logits[4999] = 10.0;
        assert_eq!(argmax(&logits), 4999);
    }

    #[test]
    fn test_argmax_empty_returns_zero() {
        assert_eq!(argmax(&[]), 0);
    }

    // === optimized_lm_head_argmax_transposed tests ===

    #[test]
    fn test_lm_head_argmax_basic() {
        // 4 vocab, 2 hidden
        let hidden = vec![1.0, 2.0];
        // Transposed weights: each row is weights for one vocab token
        // Token 0: [0.0, 0.0] -> dot = 0
        // Token 1: [1.0, 0.0] -> dot = 1
        // Token 2: [0.0, 1.0] -> dot = 2
        // Token 3: [1.0, 1.0] -> dot = 3
        let weight_t = vec![
            0.0, 0.0, // Token 0
            1.0, 0.0, // Token 1
            0.0, 1.0, // Token 2
            1.0, 1.0, // Token 3
        ];
        let bias = vec![0.0; 4];

        let result = optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, 2, 4);
        assert_eq!(result, 3); // Token 3 has highest dot product
    }

    #[test]
    fn test_lm_head_argmax_with_bias() {
        let hidden = vec![1.0, 1.0];
        let weight_t = vec![
            1.0, 1.0, // Token 0: dot = 2
            0.0, 0.0, // Token 1: dot = 0
        ];
        let bias = vec![0.0, 10.0]; // Token 1 gets big bias

        let result = optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, 2, 2);
        assert_eq!(result, 1); // Bias makes token 1 win
    }

    #[test]
    fn test_lm_head_argmax_negative_weights() {
        let hidden = vec![1.0, 1.0];
        let weight_t = vec![
            -1.0, -1.0, // Token 0: dot = -2
            1.0, 1.0, // Token 1: dot = 2
        ];
        let bias = vec![0.0, 0.0];

        let result = optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, 2, 2);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_lm_head_argmax_large_vocab() {
        // Test with larger vocabulary to exercise parallel chunks
        let hidden_dim = 64;
        let vocab_size = 10000;
        let hidden = vec![1.0; hidden_dim];

        // Set up weights so token 5000 wins
        let mut weight_t = vec![0.0; vocab_size * hidden_dim];
        for i in 0..hidden_dim {
            weight_t[5000 * hidden_dim + i] = 1.0;
        }
        let bias = vec![0.0; vocab_size];

        let result =
            optimized_lm_head_argmax_transposed(&hidden, &weight_t, &bias, hidden_dim, vocab_size);
        assert_eq!(result, 5000);
    }

    // === simplified_attention tests ===

    #[test]
    fn test_simplified_attention_single_position() {
        let config = GpuModelConfig {
            hidden_dim: 4,
            num_heads: 2,
            num_kv_heads: 2,
            intermediate_dim: 8,
            vocab_size: 100,
            num_layers: 1,
            eps: 1e-5,
            rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
        };

        // Single position: QKV for seq_len=1
        // Q: [1, 0, 1, 0], K: [1, 0, 1, 0], V: [0.5, 0.5, 0.5, 0.5]
        let qkv = vec![
            1.0, 0.0, 1.0, 0.0, // Q
            1.0, 0.0, 1.0, 0.0, // K
            0.5, 0.5, 0.5, 0.5, // V
        ];

        let output = simplified_attention(&config, &qkv, 1).expect("output");
        assert_eq!(output.len(), 4);
        // Single position: attention = softmax([score]) * V = 1.0 * V = V
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - 0.5).abs() < 1e-5,
                "output[{}] = {}, expected 0.5",
                i,
                v
            );
        }
    }

    #[test]
    fn test_simplified_attention_two_positions_causal() {
        let config = GpuModelConfig {
            hidden_dim: 2,
            num_heads: 1,
            num_kv_heads: 1,
            intermediate_dim: 4,
            vocab_size: 100,
            num_layers: 1,
            eps: 1e-5,
            rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
        };

        // Two positions: position 1 attends to both 0 and 1
        // Q: [[1, 0], [1, 0]]
        // K: [[1, 0], [0, 1]]
        // V: [[1, 0], [0, 1]]
        let qkv = vec![
            // Q (seq_len * hidden_dim)
            1.0, 0.0, 1.0, 0.0, // K (seq_len * hidden_dim)
            1.0, 0.0, 0.0, 1.0, // V (seq_len * hidden_dim)
            1.0, 0.0, 0.0, 1.0,
        ];

        let output = simplified_attention(&config, &qkv, 2).expect("output");
        assert_eq!(output.len(), 4);

        // Position 0: only attends to itself, output = V[0] = [1, 0]
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!(output[1].abs() < 1e-5);
    }

    #[test]
    fn test_simplified_attention_multi_head() {
        let config = GpuModelConfig {
            hidden_dim: 4,
            num_heads: 2,
            num_kv_heads: 2,
            intermediate_dim: 8,
            vocab_size: 100,
            num_layers: 1,
            eps: 1e-5,
            rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
        };

        // Single position with 2 heads (head_dim = 2)
        let qkv = vec![
            1.0, 1.0, 1.0, 1.0, // Q
            1.0, 1.0, 1.0, 1.0, // K
            0.5, 0.5, 0.5, 0.5, // V
        ];

        let output = simplified_attention(&config, &qkv, 1).expect("output");
        assert_eq!(output.len(), 4);
    }

    // === GpuModelConfig tests ===

    #[test]
    fn test_gpu_model_config_kv_dim() {
        let config = GpuModelConfig {
            hidden_dim: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            intermediate_dim: 11008,
            vocab_size: 32000,
            num_layers: 32,
            eps: 1e-5,
            rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
        };

        // kv_dim = head_dim * num_kv_heads = (4096/32) * 8 = 128 * 8 = 1024
        assert_eq!(config.kv_dim(), 1024);
    }

    #[test]
    fn test_gpu_model_config_qkv_dim() {
        let config = GpuModelConfig {
            hidden_dim: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            intermediate_dim: 11008,
            vocab_size: 32000,
            num_layers: 32,
            eps: 1e-5,
            rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
        };

        // qkv_dim = (num_heads + 2*num_kv_heads) * head_dim = (32+16)*128 = 6144
        assert_eq!(config.qkv_dim(), 6144);
    }

    #[test]
    fn test_gpu_model_config_head_dim() {
        let config = GpuModelConfig {
            hidden_dim: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            intermediate_dim: 11008,
            vocab_size: 32000,
            num_layers: 32,
            eps: 1e-5,
            rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
        };

        // head_dim = hidden_dim / num_heads = 4096 / 32 = 128
        assert_eq!(config.head_dim(), 128);
    }

    #[test]
    fn test_gpu_model_config_gqa_ratio() {
        let config = GpuModelConfig {
            hidden_dim: 2048,
            num_heads: 32,
            num_kv_heads: 4,
            intermediate_dim: 5632,
            vocab_size: 32000,
            num_layers: 22,
            eps: 1e-5,
            rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
        };

        // heads_per_kv = num_heads / num_kv_heads = 32 / 4 = 8
        let heads_per_kv = config.num_heads / config.num_kv_heads;
        assert_eq!(heads_per_kv, 8);
    }

    #[test]
    fn test_gpu_model_config_mha_ratio() {
        // Standard MHA: num_kv_heads == num_heads
        let config = GpuModelConfig {
            hidden_dim: 768,
            num_heads: 12,
            num_kv_heads: 12,
            intermediate_dim: 3072,
            vocab_size: 30522,
            num_layers: 12,
            eps: 1e-12,
            rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
        };

        let heads_per_kv = config.num_heads / config.num_kv_heads;
        assert_eq!(heads_per_kv, 1);
        assert_eq!(config.kv_dim(), config.hidden_dim);
    }
}
