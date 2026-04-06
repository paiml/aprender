
// ============================================================================
// Tests (F-BUILD-007)
// ============================================================================

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    #[test]
    fn test_config_extraction() {
        let json = SafetensorsConfig {
            hidden_size: Some(1536),
            num_hidden_layers: Some(28),
            num_attention_heads: Some(12),
            num_key_value_heads: Some(2),
            vocab_size: Some(151936),
            intermediate_size: Some(8960),
            max_position_embeddings: Some(32768),
            rope_theta: Some(1000000.0),
            rms_norm_eps: Some(1e-6),
            architectures: Some(vec!["Qwen2ForCausalLM".to_string()]),
            model_type: Some("qwen2".to_string()),
            bos_token_id: Some(151643),
            eos_token_id: Some(151645),
            tie_word_embeddings: Some(true), // F-GT-002: Qwen2 uses tied embeddings
            ..Default::default()
        };

        let config = SafeTensorsCudaModel::extract_config(&json).expect("config");
        assert_eq!(config.hidden_dim, 1536);
        assert_eq!(config.num_layers, 28);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.num_kv_heads, 2);
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.intermediate_dim, 8960);
        assert_eq!(config.context_length, 32768);
        assert!((config.rope_theta - 1_000_000.0).abs() < 1.0);
        assert!((config.eps - 1e-6).abs() < 1e-9);
    }

    #[test]
    fn test_config_extraction_defaults() {
        // Test with minimal config (uses defaults for optional fields)
        let json = SafetensorsConfig {
            hidden_size: Some(768),
            num_hidden_layers: Some(12),
            num_attention_heads: Some(12),
            num_key_value_heads: None, // Will default to num_attention_heads
            vocab_size: Some(50257),
            intermediate_size: None,       // Will default to 4 * hidden_dim
            max_position_embeddings: None, // Will default to 2048
            rope_theta: None,              // Will default to 10000.0
            rms_norm_eps: None,            // Will default to 1e-6
            architectures: None,
            model_type: None,
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: None, // Will default to false
            ..Default::default()
        };

        let config = SafeTensorsCudaModel::extract_config(&json).expect("config");
        assert_eq!(config.hidden_dim, 768);
        assert_eq!(config.intermediate_dim, 768 * 4); // Default
        assert_eq!(config.context_length, 2048); // Default
        assert!((config.rope_theta - 10000.0).abs() < 0.1); // Default
        assert!((config.eps - 1e-6).abs() < 1e-9); // Default
    }

    #[test]
    fn test_config_extraction_missing_hidden_size() {
        let json = SafetensorsConfig {
            hidden_size: None, // Required, should fail
            num_hidden_layers: Some(12),
            num_attention_heads: Some(12),
            num_key_value_heads: None,
            vocab_size: Some(50257),
            intermediate_size: None,
            max_position_embeddings: None,
            rope_theta: None,
            rms_norm_eps: None,
            architectures: None,
            model_type: None,
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: None,
            ..Default::default()
        };

        let result = SafeTensorsCudaModel::extract_config(&json);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_extraction_missing_layers() {
        let json = SafetensorsConfig {
            hidden_size: Some(768),
            num_hidden_layers: None, // Required
            num_attention_heads: Some(12),
            num_key_value_heads: None,
            vocab_size: Some(50257),
            intermediate_size: None,
            max_position_embeddings: None,
            rope_theta: None,
            rms_norm_eps: None,
            architectures: None,
            model_type: None,
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: None,
            ..Default::default()
        };

        let result = SafeTensorsCudaModel::extract_config(&json);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_extraction_missing_attention_heads() {
        let json = SafetensorsConfig {
            hidden_size: Some(768),
            num_hidden_layers: Some(12),
            num_attention_heads: None, // Required
            num_key_value_heads: None,
            vocab_size: Some(50257),
            intermediate_size: None,
            max_position_embeddings: None,
            rope_theta: None,
            rms_norm_eps: None,
            architectures: None,
            model_type: None,
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: None,
            ..Default::default()
        };

        let result = SafeTensorsCudaModel::extract_config(&json);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_extraction_missing_vocab_size() {
        let json = SafetensorsConfig {
            hidden_size: Some(768),
            num_hidden_layers: Some(12),
            num_attention_heads: Some(12),
            num_key_value_heads: None,
            vocab_size: None, // Required
            intermediate_size: None,
            max_position_embeddings: None,
            rope_theta: None,
            rms_norm_eps: None,
            architectures: None,
            model_type: None,
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: None,
            ..Default::default()
        };

        let result = SafeTensorsCudaModel::extract_config(&json);
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose_for_gemm_identity() {
        // 2x2 matrix transpose
        let weight = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2],[3,4]] row-major
        let transposed = SafeTensorsCudaModel::transpose_for_gemm(&weight, 2, 2);
        // Expected: [[1,3],[2,4]] row-major = [1,3,2,4]
        assert_eq!(transposed, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_transpose_for_gemm_rectangular() {
        // 2x3 matrix (n=2 rows, k=3 cols) -> 3x2 (k=3 rows, n=2 cols)
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3],[4,5,6]]
        let transposed = SafeTensorsCudaModel::transpose_for_gemm(&weight, 2, 3);
        // Expected: [[1,4],[2,5],[3,6]] row-major = [1,4,2,5,3,6]
        assert_eq!(transposed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_for_gemm_single_row() {
        // 1xk matrix (single row)
        let weight = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2,3,4]]
        let transposed = SafeTensorsCudaModel::transpose_for_gemm(&weight, 1, 4);
        // Expected: [[1],[2],[3],[4]] row-major = [1,2,3,4] (same as input)
        assert_eq!(transposed, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_transpose_for_gemm_single_col() {
        // nx1 matrix (single column)
        let weight = vec![1.0, 2.0, 3.0, 4.0]; // [[1],[2],[3],[4]]
        let transposed = SafeTensorsCudaModel::transpose_for_gemm(&weight, 4, 1);
        // Expected: [[1,2,3,4]] row-major = [1,2,3,4] (same as input)
        assert_eq!(transposed, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_transpose_for_gemm_4x4() {
        // 4x4 matrix
        let weight = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let transposed = SafeTensorsCudaModel::transpose_for_gemm(&weight, 4, 4);
        // Diagonal should be unchanged
        assert_eq!(transposed[0], 1.0);
        assert_eq!(transposed[5], 6.0);
        assert_eq!(transposed[10], 11.0);
        assert_eq!(transposed[15], 16.0);
        // Off-diagonal should swap
        assert_eq!(transposed[1], 5.0); // [0,1] gets [1,0]
        assert_eq!(transposed[4], 2.0); // [1,0] gets [0,1]
    }

    #[test]
    fn test_concat_qkv_transposed_simple() {
        // Simplest case: 2 heads, head_dim=2, kv_heads=1
        // hidden_dim = 4, kv_dim = 2
        let q = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]; // 4x4
        let k = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2x4
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2x4

        let qkv = SafeTensorsCudaModel::concat_qkv_transposed(&q, &k, &v, 4, 2);

        // Output should be [hidden_dim, hidden_dim + kv_dim + kv_dim] = [4, 8]
        assert_eq!(qkv.len(), 4 * 8);
    }

    #[test]
    fn test_concat_qkv_transposed_dimensions() {
        // Real-world-like dimensions
        let hidden_dim = 64;
        let kv_dim = 16;

        let q = vec![0.1f32; hidden_dim * hidden_dim];
        let k = vec![0.2f32; kv_dim * hidden_dim];
        let v = vec![0.3f32; kv_dim * hidden_dim];

        let qkv = SafeTensorsCudaModel::concat_qkv_transposed(&q, &k, &v, hidden_dim, kv_dim);

        let expected_len = hidden_dim * (hidden_dim + 2 * kv_dim);
        assert_eq!(qkv.len(), expected_len);
    }

    #[test]
    fn test_safetensors_cuda_config_debug() {
        let config = SafeTensorsCudaConfig {
            architecture: "Qwen2".to_string(),
            hidden_dim: 768,
            num_layers: 12,
            num_heads: 12,
            num_kv_heads: 4,
            vocab_size: 50257,
            intermediate_dim: 3072,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-6,
            tie_word_embeddings: true,
            has_qk_norm: false,
            has_bias: true,
            eos_token_id: None,
        };

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("Qwen2"));
        assert!(debug_str.contains("768"));
        assert!(debug_str.contains("12"));
    }

    #[test]
    fn test_safetensors_cuda_config_clone() {
        let config = SafeTensorsCudaConfig {
            architecture: "LLaMA".to_string(),
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            vocab_size: 32000,
            intermediate_dim: 11008,
            context_length: 4096,
            rope_theta: 10000.0,
            eps: 1e-5,
            tie_word_embeddings: false,
            has_qk_norm: false,
            has_bias: false,
            eos_token_id: None,
        };

        let cloned = config.clone();
        assert_eq!(cloned.architecture, config.architecture);
        assert_eq!(cloned.hidden_dim, config.hidden_dim);
        assert_eq!(cloned.num_layers, config.num_layers);
        assert_eq!(cloned.num_heads, config.num_heads);
        assert_eq!(cloned.num_kv_heads, config.num_kv_heads);
        assert_eq!(cloned.vocab_size, config.vocab_size);
        assert_eq!(cloned.intermediate_dim, config.intermediate_dim);
        assert_eq!(cloned.context_length, config.context_length);
        assert!((cloned.rope_theta - config.rope_theta).abs() < 0.001);
        assert!((cloned.eps - config.eps).abs() < 1e-10);
    }

    #[test]
    fn test_transpose_preserves_values() {
        // All values should be preserved, just reordered
        let weight: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let transposed = SafeTensorsCudaModel::transpose_for_gemm(&weight, 3, 4);

        let original_sum: f32 = weight.iter().sum();
        let transposed_sum: f32 = transposed.iter().sum();
        assert!((original_sum - transposed_sum).abs() < 1e-6);
    }

    #[test]
    fn test_transpose_double_transpose_is_identity() {
        // Transpose twice should give original
        let weight: Vec<f32> = (1..=20).map(|x| x as f32).collect();
        let n = 4;
        let k = 5;

        let transposed1 = SafeTensorsCudaModel::transpose_for_gemm(&weight, n, k);
        let transposed2 = SafeTensorsCudaModel::transpose_for_gemm(&transposed1, k, n);

        for (orig, back) in weight.iter().zip(transposed2.iter()) {
            assert!((orig - back).abs() < 1e-6);
        }
    }

    #[test]
    fn test_estimate_vram_bytes_qwen2_1_5b() {
        // GH-201: Test VRAM estimation for Qwen2.5-Coder-1.5B
        // This is the model that triggered the OOM issue
        let config = SafeTensorsCudaConfig {
            architecture: "Qwen2".to_string(),
            hidden_dim: 1536,
            num_layers: 28,
            num_heads: 12,
            num_kv_heads: 2,
            vocab_size: 151936,
            intermediate_dim: 8960,
            context_length: 32768,
            rope_theta: 1000000.0,
            eps: 1e-6,
            tie_word_embeddings: true,
            has_qk_norm: false,
            has_bias: true,
            eos_token_id: None,
        };

        let vram = SafeTensorsCudaModel::estimate_vram_bytes(&config, 2048);
        let vram_mb = vram / (1024 * 1024);

        // Qwen2.5-Coder-1.5B (1.5B params) requires ~6GB in F32:
        // - LM head: 1536 * 151936 * 4 = ~887 MB
        // - 28 layers × (QKV + O + gate + up + down + norms)
        //   - QKV: 1536 * (1536 + 2*256) * 4 = ~12.5 MB
        //   - O: 1536 * 1536 * 4 = ~9.4 MB
        //   - Gate: 1536 * 8960 * 4 = ~55 MB
        //   - Up: 1536 * 8960 * 4 = ~55 MB
        //   - Down: 8960 * 1536 * 4 = ~55 MB
        //   Total per layer: ~187 MB × 28 = ~5.2 GB
        // - KV cache: 2 * 28 * 2048 * 256 * 4 = ~57 MB
        // Total: ~6GB
        assert!(
            vram_mb > 5500 && vram_mb < 7000,
            "Expected 5.5-7 GB for Qwen2.5-Coder-1.5B F32, got {} MB",
            vram_mb
        );
    }

    #[test]
    fn test_estimate_vram_bytes_scales_with_layers() {
        // More layers should require more VRAM
        let config_12 = SafeTensorsCudaConfig {
            architecture: "Test".to_string(),
            hidden_dim: 768,
            num_layers: 12,
            num_heads: 12,
            num_kv_heads: 12,
            vocab_size: 50257,
            intermediate_dim: 3072,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-6,
            tie_word_embeddings: false,
            has_qk_norm: false,
            has_bias: false,
            eos_token_id: None,
        };

        let config_24 = SafeTensorsCudaConfig {
            num_layers: 24,
            ..config_12.clone()
        };

        let vram_12 = SafeTensorsCudaModel::estimate_vram_bytes(&config_12, 1024);
        let vram_24 = SafeTensorsCudaModel::estimate_vram_bytes(&config_24, 1024);

        // 24 layers should use roughly 2x the layer weight memory
        assert!(
            vram_24 > vram_12,
            "24 layers ({}) should use more VRAM than 12 layers ({})",
            vram_24,
            vram_12
        );
    }

    #[test]
    fn test_estimate_vram_bytes_scales_with_seq_len() {
        // Longer sequences need more KV cache
        let config = SafeTensorsCudaConfig {
            architecture: "Test".to_string(),
            hidden_dim: 768,
            num_layers: 12,
            num_heads: 12,
            num_kv_heads: 12,
            vocab_size: 50257,
            intermediate_dim: 3072,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-6,
            tie_word_embeddings: false,
            has_qk_norm: false,
            has_bias: false,
            eos_token_id: None,
        };

        let vram_1k = SafeTensorsCudaModel::estimate_vram_bytes(&config, 1024);
        let vram_4k = SafeTensorsCudaModel::estimate_vram_bytes(&config, 4096);

        // 4k context should use more VRAM due to KV cache
        assert!(
            vram_4k > vram_1k,
            "4k context ({}) should use more VRAM than 1k context ({})",
            vram_4k,
            vram_1k
        );
    }

    // ========================================================================
    // transpose_for_gemm: undersized weight guard (PMAT-805 branch coverage)
    // ========================================================================

    #[test]
    fn test_transpose_for_gemm_undersized_weight() {
        // Weight has fewer elements than n*k — should zero-pad and transpose
        let weight = vec![1.0, 2.0, 3.0]; // 3 elements, but n=2, k=3 expects 6
        let transposed = SafeTensorsCudaModel::transpose_for_gemm(&weight, 2, 3);

        // Output should be k×n = 3×2 = 6 elements
        assert_eq!(transposed.len(), 6);

        // First row of input [1,2,3] transposes to column 0: transposed[0]=1, transposed[2]=2, transposed[4]=3
        assert_eq!(transposed[0], 1.0); // [0,0] -> [0,0]
        assert_eq!(transposed[2], 2.0); // [0,1] -> [1,0]
        assert_eq!(transposed[4], 3.0); // [0,2] -> [2,0]

        // Second row is all zeros (weight was too short)
        assert_eq!(transposed[1], 0.0); // [0,1]
        assert_eq!(transposed[3], 0.0); // [1,1]
        assert_eq!(transposed[5], 0.0); // [2,1]
    }

    #[test]
    fn test_transpose_for_gemm_undersized_weight_partial_row() {
        // Weight has partial second row
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // 5 of expected 8 (n=2, k=4)
        let transposed = SafeTensorsCudaModel::transpose_for_gemm(&weight, 2, 4);

        assert_eq!(transposed.len(), 8);
        // First row [1,2,3,4] fully present
        assert_eq!(transposed[0], 1.0);
        assert_eq!(transposed[2], 2.0);
        assert_eq!(transposed[4], 3.0);
        assert_eq!(transposed[6], 4.0);
        // Second row [5,_,_,_] — only first element present
        assert_eq!(transposed[1], 5.0);
        assert_eq!(transposed[3], 0.0); // zero-padded
        assert_eq!(transposed[5], 0.0);
        assert_eq!(transposed[7], 0.0);
    }

    #[test]
    fn test_transpose_for_gemm_empty_weight() {
        // Empty weight array — should produce all zeros
        let weight: Vec<f32> = vec![];
        let transposed = SafeTensorsCudaModel::transpose_for_gemm(&weight, 3, 2);

        assert_eq!(transposed.len(), 6);
        assert!(transposed.iter().all(|&x| x == 0.0));
    }

    // ========================================================================
    // concat_qkv_transposed: verify correctness of concatenation order
    // ========================================================================

    #[test]
    fn test_concat_qkv_transposed_content_correctness() {
        // 2x2 Q, 1x2 K, 1x2 V (hidden_dim=2, kv_dim=1)
        let q = vec![1.0, 2.0, 3.0, 4.0]; // 2×2 Q weight
        let k = vec![5.0, 6.0];            // 1×2 K weight
        let v = vec![7.0, 8.0];            // 1×2 V weight

        let qkv = SafeTensorsCudaModel::concat_qkv_transposed(&q, &k, &v, 2, 1);

        // Output: [hidden_dim, hidden_dim + 2*kv_dim] = [2, 4]
        assert_eq!(qkv.len(), 8);

        // After transpose: q_t=[1,3,2,4], k_t=[5,6], v_t=[7,8]
        // Row 0: [q_t[0..2], k_t[0..1], v_t[0..1]] = [1, 3, 5, 7]
        // Row 1: [q_t[2..4], k_t[1..2], v_t[1..2]] = [2, 4, 6, 8]
        assert_eq!(qkv[0], 1.0);
        assert_eq!(qkv[1], 3.0);
        assert_eq!(qkv[2], 5.0);
        assert_eq!(qkv[3], 7.0);
        assert_eq!(qkv[4], 2.0);
        assert_eq!(qkv[5], 4.0);
        assert_eq!(qkv[6], 6.0);
        assert_eq!(qkv[7], 8.0);
    }

    #[test]
    fn test_concat_qkv_transposed_equal_kv() {
        // Test where kv_dim == hidden_dim (MHA, no GQA)
        let hidden_dim = 4;
        let kv_dim = 4;
        let q = vec![1.0f32; hidden_dim * hidden_dim];
        let k = vec![2.0f32; kv_dim * hidden_dim];
        let v = vec![3.0f32; kv_dim * hidden_dim];

        let qkv = SafeTensorsCudaModel::concat_qkv_transposed(&q, &k, &v, hidden_dim, kv_dim);
        let expected_len = hidden_dim * (hidden_dim + 2 * kv_dim);
        assert_eq!(qkv.len(), expected_len);
    }
}
