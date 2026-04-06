
/// PMAT-216: Implement TracedForward trait for GPU backend
///
/// This ensures GPU inference provides the same tracing capability as CPU,
/// enabling layer-by-layer parity testing to catch divergence bugs early.
impl crate::apr_transformer::TracedForward for GpuModel {
    fn forward_traced(&mut self, tokens: &[u32]) -> Result<ForwardTrace> {
        // Convert u32 tokens to usize for GPU forward
        let usize_tokens: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
        self.forward_traced_gpu(&usize_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // GpuModelConfig tests
    // ========================================================================

    #[test]
    fn test_gpu_model_config_head_dim() {
        let config = GpuModelConfig {
            vocab_size: 32000,
            hidden_dim: 256,
            num_heads: 8,
            num_kv_heads: 8,
            num_layers: 4,
            intermediate_dim: 512,
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
        assert_eq!(config.head_dim(), 32);
    }

    #[test]
    fn test_gpu_model_config_kv_dim() {
        let config = GpuModelConfig {
            vocab_size: 32000,
            hidden_dim: 256,
            num_heads: 8,
            num_kv_heads: 4, // GQA: fewer KV heads
            num_layers: 4,
            intermediate_dim: 512,
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
        // kv_dim = num_kv_heads * head_dim = 4 * 32 = 128
        assert_eq!(config.kv_dim(), 128);
    }

    #[test]
    fn test_gpu_model_config_qkv_dim_mha() {
        // Multi-head attention: num_kv_heads == num_heads
        let config = GpuModelConfig {
            vocab_size: 32000,
            hidden_dim: 256,
            num_heads: 8,
            num_kv_heads: 8,
            num_layers: 4,
            intermediate_dim: 512,
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
        // MHA: qkv_dim = hidden_dim + 2 * kv_dim = 256 + 2 * 256 = 768
        assert_eq!(config.qkv_dim(), 768);
    }

    #[test]
    fn test_gpu_model_config_qkv_dim_gqa() {
        // Grouped query attention: num_kv_heads < num_heads
        let config = GpuModelConfig {
            vocab_size: 32000,
            hidden_dim: 256,
            num_heads: 8,
            num_kv_heads: 2, // GQA: 4x fewer KV heads
            num_layers: 4,
            intermediate_dim: 512,
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
        // GQA: qkv_dim = hidden_dim + 2 * kv_dim = 256 + 2 * 64 = 384
        assert_eq!(config.qkv_dim(), 384);
    }

    #[test]
    fn test_gpu_model_config_is_gqa() {
        // MHA
        let mha_config = GpuModelConfig {
            vocab_size: 32000,
            hidden_dim: 256,
            num_heads: 8,
            num_kv_heads: 8,
            num_layers: 4,
            intermediate_dim: 512,
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
        assert!(!mha_config.is_gqa());

        // GQA
        let gqa_config = GpuModelConfig {
            vocab_size: 32000,
            hidden_dim: 256,
            num_heads: 8,
            num_kv_heads: 2,
            num_layers: 4,
            intermediate_dim: 512,
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
        assert!(gqa_config.is_gqa());
    }

    #[test]
    fn test_gpu_model_config_clone() {
        let config = GpuModelConfig {
            vocab_size: 32000,
            hidden_dim: 256,
            num_heads: 8,
            num_kv_heads: 4,
            num_layers: 4,
            intermediate_dim: 512,
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
        let cloned = config.clone();
        assert_eq!(cloned.vocab_size, 32000);
        assert_eq!(cloned.hidden_dim, 256);
    }

    #[test]
    fn test_gpu_model_config_debug() {
        let config = GpuModelConfig {
            vocab_size: 32000,
            hidden_dim: 256,
            num_heads: 8,
            num_kv_heads: 8,
            num_layers: 4,
            intermediate_dim: 512,
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
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("GpuModelConfig"));
        assert!(debug_str.contains("vocab_size: 32000"));
    }

    // ========================================================================
    // GpuGenerateConfig tests
    // ========================================================================

    #[test]
    fn test_gpu_generate_config_default() {
        let config = GpuGenerateConfig::default();
        assert_eq!(config.max_tokens, 64);
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.top_k, 1);
        assert!(config.stop_tokens.is_empty());
    }

    #[test]
    fn test_gpu_generate_config_deterministic() {
        let config = GpuGenerateConfig::deterministic(128);
        assert_eq!(config.max_tokens, 128);
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.top_k, 1);
    }

    #[test]
    fn test_gpu_generate_config_with_sampling() {
        let config = GpuGenerateConfig::with_sampling(256, 0.7, 40);
        assert_eq!(config.max_tokens, 256);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_k, 40);
    }

    #[test]
    fn test_gpu_generate_config_with_stop_tokens() {
        let config = GpuGenerateConfig::deterministic(64).with_stop_tokens(vec![0, 2]);
        assert_eq!(config.stop_tokens, vec![0, 2]);
    }

    #[test]
    fn test_gpu_generate_config_clone() {
        let config = GpuGenerateConfig::with_sampling(100, 0.5, 20);
        let cloned = config.clone();
        assert_eq!(cloned.max_tokens, 100);
        assert_eq!(cloned.temperature, 0.5);
    }

    #[test]
    fn test_gpu_generate_config_debug() {
        let config = GpuGenerateConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("GpuGenerateConfig"));
    }

    // ========================================================================
    // AttentionBuffers tests
    // ========================================================================

    #[test]
    fn test_attention_buffers_new() {
        let model_config = GpuModelConfig {
            vocab_size: 32000,
            hidden_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            num_layers: 2,
            intermediate_dim: 128,
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
        let buffers = AttentionBuffers::new(&model_config, 512);
        assert_eq!(buffers.q_buffer.len(), 64); // hidden_dim
        assert_eq!(buffers.scores_buffer.len(), 4 * 512); // num_heads * max_seq_len
        assert_eq!(buffers.output_buffer.len(), 64);
        assert_eq!(buffers.kv_proj_buffer.len(), 64);
        assert_eq!(buffers.ffn_buffer.len(), 128); // intermediate_dim
        assert_eq!(buffers.max_seq_len, 512);
    }

    #[test]
    fn test_attention_buffers_reset() {
        let model_config = GpuModelConfig {
            vocab_size: 32000,
            hidden_dim: 32,
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: 1,
            intermediate_dim: 64,
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
        let mut buffers = AttentionBuffers::new(&model_config, 128);

        // Modify buffers
        buffers.q_buffer[0] = 1.0;
        buffers.scores_buffer[0] = 2.0;
        buffers.output_buffer[0] = 3.0;
        buffers.kv_proj_buffer[0] = 4.0;
        buffers.ffn_buffer[0] = 5.0;

        // Reset
        buffers.reset();

        // All should be zero
        assert!(buffers.q_buffer.iter().all(|&x| x == 0.0));
        assert!(buffers.scores_buffer.iter().all(|&x| x == 0.0));
        assert!(buffers.output_buffer.iter().all(|&x| x == 0.0));
        assert!(buffers.kv_proj_buffer.iter().all(|&x| x == 0.0));
        assert!(buffers.ffn_buffer.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_attention_buffers_debug() {
        let model_config = GpuModelConfig {
            vocab_size: 32000,
            hidden_dim: 32,
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: 1,
            intermediate_dim: 64,
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
        let buffers = AttentionBuffers::new(&model_config, 128);
        let debug_str = format!("{:?}", buffers);
        assert!(debug_str.contains("AttentionBuffers"));
    }

    // ========================================================================
    // WeightType tests
    // ========================================================================

    #[test]
    fn test_weight_type_debug() {
        let wt = WeightType::Qkv;
        let debug_str = format!("{:?}", wt);
        assert!(debug_str.contains("Qkv"));
    }

    #[test]
    fn test_weight_type_clone() {
        let wt = WeightType::Output;
        let cloned = wt;
        assert!(matches!(cloned, WeightType::Output));
    }

    #[test]
    fn test_weight_type_copy() {
        let wt = WeightType::FfnFc1;
        let copied = wt;
        assert!(matches!(copied, WeightType::FfnFc1));
        assert!(matches!(wt, WeightType::FfnFc1)); // Original still valid (Copy)
    }

    #[test]
    fn test_weight_type_all_variants() {
        let variants = [
            WeightType::Qkv,
            WeightType::Output,
            WeightType::FfnFc1,
            WeightType::FfnFc2,
            WeightType::LmHead,
        ];
        assert_eq!(variants.len(), 5);
    }

    // ========================================================================
    // layer_norm_static tests
    // ========================================================================

    #[test]
    fn test_layer_norm_static_single_row() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let bias = vec![0.0, 0.0, 0.0, 0.0];
        let eps = 1e-5;

        let output = GpuModel::layer_norm_static(&input, &weight, &bias, 4, eps);
        assert_eq!(output.len(), 4);

        // Verify RMSNorm: each element is x / rms
        let sum_sq: f32 = input.iter().map(|x| x * x).sum();
        let rms = (sum_sq / 4.0 + eps).sqrt();
        for (i, &x) in input.iter().enumerate() {
            let expected = x / rms;
            assert!((output[i] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_layer_norm_static_with_weight_bias() {
        let input = vec![2.0, 2.0, 2.0, 2.0];
        let weight = vec![2.0, 2.0, 2.0, 2.0];
        let bias = vec![0.5, 0.5, 0.5, 0.5];
        let eps = 1e-5;

        let output = GpuModel::layer_norm_static(&input, &weight, &bias, 4, eps);

        // RMS of [2,2,2,2] = sqrt(16/4 + eps) = sqrt(4 + eps) ≈ 2.0
        // Normalized: 2.0 / 2.0 = 1.0
        // Scaled: 1.0 * 2.0 + 0.5 = 2.5
        for &val in &output {
            assert!((val - 2.5).abs() < 0.01);
        }
    }

    #[test]
    fn test_layer_norm_static_multi_row() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 rows of 4
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let bias = vec![0.0, 0.0, 0.0, 0.0];
        let eps = 1e-5;

        let output = GpuModel::layer_norm_static(&input, &weight, &bias, 4, eps);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_layer_norm_static_eps_effect() {
        // Near-zero input tests eps stabilization
        let input = vec![0.001, 0.001, 0.001, 0.001];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let bias = vec![0.0, 0.0, 0.0, 0.0];

        let output_small_eps = GpuModel::layer_norm_static(&input, &weight, &bias, 4, 1e-10);
        let output_large_eps = GpuModel::layer_norm_static(&input, &weight, &bias, 4, 1e-2);

        // Both should produce valid results
        assert!(output_small_eps.iter().all(|x| x.is_finite()));
        assert!(output_large_eps.iter().all(|x| x.is_finite()));
    }

    // ========================================================================
    // BlockWeights tests
    // ========================================================================

    #[test]
    fn test_block_weights_structure() {
        let block = BlockWeights {
            attn_norm_weight: vec![1.0; 64],
            attn_norm_bias: vec![0.0; 64],
            qkv_weight: vec![0.01; 64 * 192], // hidden_dim * 3*hidden_dim
            qkv_bias: vec![0.0; 192],
            out_weight: vec![0.01; 64 * 64],
            out_bias: vec![0.0; 64],
            ffn_norm_weight: vec![1.0; 64],
            ffn_norm_bias: vec![0.0; 64],
            ffn_fc1_weight: vec![0.01; 64 * 256],
            ffn_fc1_bias: vec![0.0; 256],
            ffn_fc2_weight: vec![0.01; 256 * 64],
            ffn_fc2_bias: vec![0.0; 64],
            ffn_gate_weight: None,
            linear_attn: None,
        moe_experts: None,
        };

        assert_eq!(block.attn_norm_weight.len(), 64);
        assert!(block.ffn_gate_weight.is_none());
    }

    #[test]
    fn test_block_weights_with_swiglu() {
        let block = BlockWeights {
            attn_norm_weight: vec![1.0; 64],
            attn_norm_bias: vec![0.0; 64],
            qkv_weight: vec![0.01; 64 * 192],
            qkv_bias: vec![0.0; 192],
            out_weight: vec![0.01; 64 * 64],
            out_bias: vec![0.0; 64],
            ffn_norm_weight: vec![1.0; 64],
            ffn_norm_bias: vec![0.0; 64],
            ffn_fc1_weight: vec![0.01; 64 * 256],
            ffn_fc1_bias: vec![0.0; 256],
            ffn_fc2_weight: vec![0.01; 256 * 64],
            ffn_fc2_bias: vec![0.0; 64],
            ffn_gate_weight: Some(vec![0.01; 64 * 256]), // SwiGLU gate
            linear_attn: None,
        moe_experts: None,
        };

        assert!(block.ffn_gate_weight.is_some());
        assert_eq!(block.ffn_gate_weight.as_ref().expect("as_ref").len(), 64 * 256);
    }
}
