
#[cfg(test)]
mod tests {
    use super::*;
    use crate::apr_transformer::AprTransformerConfig;

    #[test]
    fn test_config_to_gpu() {
        let apr_config = AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim: 512,
            num_layers: 4,
            num_heads: 8,
            num_kv_heads: 4,
            vocab_size: 32000,
            intermediate_dim: 1024,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        };

        let gpu_config = AprToGpuAdapter::config_to_gpu(&apr_config);

        assert_eq!(gpu_config.vocab_size, 32000);
        assert_eq!(gpu_config.hidden_dim, 512);
        assert_eq!(gpu_config.num_heads, 8);
        assert_eq!(gpu_config.num_kv_heads, 4);
        assert_eq!(gpu_config.num_layers, 4);
        assert_eq!(gpu_config.intermediate_dim, 1024);
        assert_eq!(gpu_config.eps, 1e-5);
    }

    #[test]
    fn test_transpose_matrix() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let transposed = transpose_matrix(&data, 2, 3); // 3x2

        assert_eq!(transposed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_identity() {
        let data = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let transposed = transpose_matrix(&data, 2, 2);

        assert_eq!(transposed, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_transpose_single_row() {
        let data = vec![1.0, 2.0, 3.0, 4.0]; // 1x4
        let transposed = transpose_matrix(&data, 1, 4); // 4x1

        assert_eq!(transposed, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_transpose_single_col() {
        let data = vec![1.0, 2.0, 3.0, 4.0]; // 4x1
        let transposed = transpose_matrix(&data, 4, 1); // 1x4

        assert_eq!(transposed, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_transpose_3x4() {
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect(); // 3x4
        let transposed = transpose_matrix(&data, 3, 4); // 4x3

        // Original: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
        // Transposed: [[1,5,9], [2,6,10], [3,7,11], [4,8,12]]
        assert_eq!(transposed[0], 1.0);
        assert_eq!(transposed[1], 5.0);
        assert_eq!(transposed[2], 9.0);
        assert_eq!(transposed[3], 2.0);
        assert_eq!(transposed[4], 6.0);
    }

    #[test]
    fn test_transpose_double_transpose_is_identity() {
        let data: Vec<f32> = (1..=20).map(|x| x as f32).collect();
        let rows = 4;
        let cols = 5;

        let transposed1 = transpose_matrix(&data, rows, cols);
        let transposed2 = transpose_matrix(&transposed1, cols, rows);

        assert_eq!(transposed2, data);
    }

    #[test]
    fn test_apr_gpu_error_dequant_display() {
        let err = AprGpuError::DequantError("test error".to_string());
        let display = format!("{}", err);
        assert!(display.contains("dequantize"));
        assert!(display.contains("test error"));
    }

    #[test]
    fn test_apr_gpu_error_dimension_mismatch_display() {
        let err = AprGpuError::DimensionMismatch {
            expected: 100,
            actual: 50,
        };
        let display = format!("{}", err);
        assert!(display.contains("dimension mismatch"));
        assert!(display.contains("100"));
        assert!(display.contains("50"));
    }

    #[test]
    fn test_apr_gpu_error_gpu_model_error_display() {
        let err = AprGpuError::GpuModelError("creation failed".to_string());
        let display = format!("{}", err);
        assert!(display.contains("GpuModel"));
        assert!(display.contains("creation failed"));
    }

    #[test]
    fn test_config_to_gpu_with_gqa() {
        // Test Grouped Query Attention config
        let apr_config = AprTransformerConfig {
            architecture: "llama".to_string(),
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8, // GQA: fewer KV heads
            vocab_size: 32000,
            intermediate_dim: 11008,
            context_length: 4096,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        };

        let gpu_config = AprToGpuAdapter::config_to_gpu(&apr_config);

        assert_eq!(gpu_config.num_heads, 32);
        assert_eq!(gpu_config.num_kv_heads, 8);
        // Verify GQA ratio
        assert_eq!(gpu_config.num_heads / gpu_config.num_kv_heads, 4);
    }

    #[test]
    fn test_config_to_gpu_with_mha() {
        // Test Multi-Head Attention config (num_heads == num_kv_heads)
        let apr_config = AprTransformerConfig {
            architecture: "phi2".to_string(),
            hidden_dim: 2560,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32, // MHA: same as num_heads
            vocab_size: 51200,
            intermediate_dim: 10240,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        };

        let gpu_config = AprToGpuAdapter::config_to_gpu(&apr_config);

        assert_eq!(gpu_config.num_heads, gpu_config.num_kv_heads);
    }

    #[test]
    fn test_apr_gpu_error_debug() {
        let err = AprGpuError::DequantError("debug test".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("DequantError"));
        assert!(debug.contains("debug test"));
    }

    #[test]
    fn test_transpose_preserves_sum() {
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let transposed = transpose_matrix(&data, 4, 6);

        let original_sum: f32 = data.iter().sum();
        let transposed_sum: f32 = transposed.iter().sum();
        assert!((original_sum - transposed_sum).abs() < 1e-6);
    }

    #[test]
    fn test_config_to_gpu_hybrid_fields_passthrough() {
        // GH-278: Verify hybrid attention fields propagate through config_to_gpu
        let apr_config = AprTransformerConfig {
            architecture: "qwen3_5".to_string(),
            hidden_dim: 5120,
            num_layers: 64,
            num_heads: 40,
            num_kv_heads: 8,
            vocab_size: 248064,
            intermediate_dim: 25600,
            context_length: 131072,
            rope_theta: 1_300_000.0,
            eps: 1e-6,
            eos_token_id: Some(248044),
            explicit_head_dim: Some(256),
            layer_types: Some(vec![
                "full_attention".to_string(),
                "linear_attention".to_string(),
            ]),
            linear_key_head_dim: Some(128),
            linear_value_head_dim: Some(128),
            linear_num_key_heads: Some(16),
            linear_num_value_heads: Some(48),
            linear_conv_kernel_dim: Some(4),
            num_experts: None,
            num_experts_per_tok: None,
            expert_intermediate_size: None,
        };

        let gpu_config = AprToGpuAdapter::config_to_gpu(&apr_config);

        assert_eq!(gpu_config.explicit_head_dim, Some(256));
        assert_eq!(gpu_config.head_dim(), 256);
        assert_eq!(gpu_config.linear_key_head_dim, Some(128));
        assert_eq!(gpu_config.linear_value_head_dim, Some(128));
        assert_eq!(gpu_config.linear_num_key_heads, Some(16));
        assert_eq!(gpu_config.linear_num_value_heads, Some(48));
        assert_eq!(gpu_config.linear_conv_kernel_dim, Some(4));
        // Verify layer_types propagated
        let lt = gpu_config.layer_types.as_ref().expect("layer_types");
        assert_eq!(lt.len(), 2);
        assert!(!gpu_config.is_linear_layer(0)); // full_attention
        assert!(gpu_config.is_linear_layer(1)); // linear_attention
        // Verify derived dimensions
        assert_eq!(gpu_config.linear_key_dim(), 16 * 128); // 2048
        assert_eq!(gpu_config.linear_value_dim(), 48 * 128); // 6144
        assert_eq!(gpu_config.linear_conv_dim(), 2 * 2048 + 6144); // 10240
    }

    #[test]
    fn test_config_to_gpu_no_hybrid_fields() {
        // Backwards compat: standard model has all hybrid fields as None
        let apr_config = AprTransformerConfig {
            architecture: "llama".to_string(),
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            vocab_size: 32000,
            intermediate_dim: 11008,
            context_length: 4096,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
            ..Default::default()
        };

        let gpu_config = AprToGpuAdapter::config_to_gpu(&apr_config);

        assert!(gpu_config.explicit_head_dim.is_none());
        assert!(gpu_config.layer_types.is_none());
        assert!(gpu_config.linear_key_head_dim.is_none());
        assert!(!gpu_config.is_linear_layer(0));
        assert_eq!(gpu_config.head_dim(), 128); // computed: 4096/32
        assert_eq!(gpu_config.linear_key_dim(), 0);
        assert_eq!(gpu_config.linear_value_dim(), 0);
    }
}
