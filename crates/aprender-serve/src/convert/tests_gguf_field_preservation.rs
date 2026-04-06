
    // ==========================================================================
    // Converter Tests
    // ==========================================================================

    #[test]
    fn test_from_gguf_transformer_config_preserved() {
        // Create a mock GGUF transformer
        let gguf = create_mock_gguf_transformer(4, 1, 10, 8);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

        assert_eq!(apr.config.architecture, gguf.config.architecture);
        assert_eq!(apr.config.hidden_dim, gguf.config.hidden_dim);
        assert_eq!(apr.config.num_layers, gguf.config.num_layers);
        assert_eq!(apr.config.vocab_size, gguf.config.vocab_size);
    }

    #[test]
    fn test_from_gguf_transformer_weights_preserved() {
        let gguf = create_mock_gguf_transformer(4, 1, 10, 8);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

        assert_eq!(apr.token_embedding, gguf.token_embedding);
        assert_eq!(apr.output_norm_weight, gguf.output_norm_weight);
        assert_eq!(apr.lm_head_weight, gguf.lm_head_weight);
    }

    #[test]
    fn test_from_gguf_transformer_layers_preserved() {
        let gguf = create_mock_gguf_transformer(4, 2, 10, 8);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

        assert_eq!(apr.layers.len(), gguf.layers.len());
        for (apr_layer, gguf_layer) in apr.layers.iter().zip(gguf.layers.iter()) {
            assert_eq!(apr_layer.attn_norm_weight, gguf_layer.attn_norm_weight);
            assert_eq!(apr_layer.qkv_weight, gguf_layer.qkv_weight);
            assert_eq!(apr_layer.ffn_up_weight, gguf_layer.ffn_up_weight);
            assert_eq!(apr_layer.ffn_down_weight, gguf_layer.ffn_down_weight);
        }
    }

    // ==========================================================================
    // APR Serialization Tests
    // ==========================================================================

    #[test]
    fn test_to_apr_bytes_header_valid() {
        let apr = create_test_apr_transformer(4, 1, 10, 8);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        // Check header (APR v2 format)
        assert_eq!(&bytes[0..4], &MAGIC); // APR2 magic
        assert_eq!(bytes[4], 2); // version major (v2)
        assert_eq!(bytes[5], 0); // version minor

        // Check tensor count (at bytes 8-11 in v2)
        let tensor_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(tensor_count, 1); // We store weights as single tensor
    }

    #[test]
    fn test_apr_bytes_roundtrip() {
        let original = create_test_apr_transformer(4, 1, 10, 8);
        let bytes = GgufToAprConverter::to_apr_bytes(&original).expect("serialize");
        let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

        assert_eq!(original.config, loaded.config);
        assert_eq!(original.token_embedding, loaded.token_embedding);
        assert_eq!(original.layers.len(), loaded.layers.len());
    }

    #[test]
    fn test_from_apr_bytes_missing_weights() {
        // Create bytes with valid v2 header but no weights tensor
        let mut bytes = vec![0u8; 128];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2; // v2
        bytes[8..12].copy_from_slice(&0u32.to_le_bytes()); // 0 tensors
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata offset
        bytes[20..24].copy_from_slice(&2u32.to_le_bytes()); // metadata size
        bytes[24..32].copy_from_slice(&66u64.to_le_bytes()); // tensor index offset
        bytes[32..40].copy_from_slice(&66u64.to_le_bytes()); // data offset (same = empty index)
        bytes[64..66].copy_from_slice(b"{}"); // minimal JSON metadata

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err()); // Should fail: no weights tensor
    }

    // ==========================================================================
    // Stats Tests
    // ==========================================================================

    #[test]
    fn test_stats_basic() {
        let apr = create_test_apr_transformer(64, 2, 1000, 256);
        let stats = GgufToAprConverter::stats(&apr);

        assert_eq!(stats.num_layers, 2);
        assert_eq!(stats.hidden_dim, 64);
        assert_eq!(stats.vocab_size, 1000);
        assert!(stats.total_parameters > 0);
        assert!(stats.memory_bytes_f32 > 0);
    }

    #[test]
    fn test_stats_memory_conversions() {
        let apr = create_test_apr_transformer(64, 1, 100, 128);
        let stats = GgufToAprConverter::stats(&apr);

        // Memory should be params * 4 bytes
        assert_eq!(stats.memory_bytes_f32, stats.total_parameters * 4);

        // MB should be bytes / 1M
        let expected_mb = stats.memory_bytes_f32 as f64 / (1024.0 * 1024.0);
        assert!((stats.memory_mb() - expected_mb).abs() < 0.0001);
    }

    #[test]
    fn test_stats_parameter_conversions() {
        let apr = create_test_apr_transformer(64, 1, 100, 128);
        let stats = GgufToAprConverter::stats(&apr);

        let expected_m = stats.total_parameters as f64 / 1_000_000.0;
        assert!((stats.parameters_m() - expected_m).abs() < 0.0001);
    }

    // ==========================================================================
    // Inference Equivalence Tests
    // ==========================================================================

    #[test]
    fn test_inference_produces_output() {
        let apr = create_test_apr_transformer(4, 1, 10, 8);
        let tokens = vec![1, 2, 3];

        let result = apr.forward(&tokens);
        assert!(result.is_ok());

        let logits = result.expect("forward");
        assert_eq!(logits.len(), apr.config.vocab_size);
    }

    #[test]
    fn test_inference_deterministic() {
        let apr = create_test_apr_transformer(4, 1, 10, 8);
        let tokens = vec![1, 2, 3];

        let logits1 = apr.forward(&tokens).expect("forward 1");
        let logits2 = apr.forward(&tokens).expect("forward 2");

        assert_eq!(logits1, logits2, "Inference should be deterministic");
    }

    // ==========================================================================
    // Helper Functions
    // ==========================================================================

    fn create_mock_gguf_transformer(
        hidden_dim: usize,
        num_layers: usize,
        vocab_size: usize,
        intermediate_dim: usize,
    ) -> GGUFTransformer {
        use crate::gguf::{GGUFConfig, GGUFTransformerLayer};

        let config = GGUFConfig {
            architecture: "test_arch".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("test_arch"),
            hidden_dim,
            num_layers,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size,
            intermediate_dim,
            context_length: 512,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0, // NORM style (adjacent pairs)
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let layers: Vec<GGUFTransformerLayer> = (0..num_layers)
            .map(|_| GGUFTransformerLayer {
                attn_norm_weight: vec![1.0; hidden_dim],
                attn_norm_bias: None,
                qkv_weight: vec![0.01; hidden_dim * 3 * hidden_dim],
                qkv_bias: None,
                attn_output_weight: vec![0.01; hidden_dim * hidden_dim],
                attn_output_bias: None,
                ffn_gate_weight: None,
                ffn_gate_bias: None,
                ffn_up_weight: vec![0.01; hidden_dim * intermediate_dim],
                ffn_up_bias: None,
                ffn_down_weight: vec![0.01; intermediate_dim * hidden_dim],
                ffn_down_bias: None,
                ffn_norm_weight: None,
                ffn_norm_bias: None,
                attn_q_norm_weight: None,
                attn_k_norm_weight: None,
            })
            .collect();

        GGUFTransformer {
            config,
            token_embedding: vec![0.1; vocab_size * hidden_dim],
            position_embedding: None,
            layers,
            output_norm_weight: vec![1.0; hidden_dim],
            output_norm_bias: None,
            lm_head_weight: vec![0.01; hidden_dim * vocab_size],
            lm_head_bias: None,
        }
    }

    fn create_test_apr_transformer(
        hidden_dim: usize,
        num_layers: usize,
        vocab_size: usize,
        intermediate_dim: usize,
    ) -> AprTransformer {
        let config = AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size,
            intermediate_dim,
            context_length: 512,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        };

        let layers: Vec<AprTransformerLayer> = (0..num_layers)
            .map(|_| AprTransformerLayer {
                attn_norm_weight: vec![1.0; hidden_dim],
                attn_norm_bias: None,
                qkv_weight: vec![0.01; hidden_dim * 3 * hidden_dim],
                qkv_bias: None,
                attn_output_weight: vec![0.01; hidden_dim * hidden_dim],
                attn_output_bias: None,
                ffn_gate_weight: None,
                ffn_gate_bias: None,
                ffn_up_weight: vec![0.01; hidden_dim * intermediate_dim],
                ffn_up_bias: None,
                ffn_down_weight: vec![0.01; intermediate_dim * hidden_dim],
                ffn_down_bias: None,
                ffn_norm_weight: None,
                ffn_norm_bias: None,
                attn_q_norm_weight: None,
                attn_k_norm_weight: None,
                linear_attn_z_weight: None,
                linear_attn_b_weight: None,
                linear_attn_a_weight: None,
                linear_attn_conv1d_weight: None,
                linear_attn_a_log: None,
                linear_attn_dt_bias: None,
                linear_attn_norm_weight: None,
                moe_gate_weight: None,
                moe_expert_gate_up: None,
                moe_expert_down: None,
                moe_shared_gate: None,
                moe_shared_up: None,
                moe_shared_down: None,
                moe_shared_expert_gate_weight: None,
            })
            .collect();

        AprTransformer {
            config,
            token_embedding: vec![0.1; vocab_size * hidden_dim],
            layers,
            output_norm_weight: vec![1.0; hidden_dim],
            output_norm_bias: None,
            lm_head_weight: vec![0.01; hidden_dim * vocab_size],
            lm_head_bias: None,
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        }
    }

    // ==========================================================================
    // ConversionStats Coverage Tests
    // ==========================================================================

    #[test]
    fn test_stats_memory_gb() {
        let stats = ConversionStats {
            total_parameters: 1_000_000_000, // 1B params
            memory_bytes_f32: 4_000_000_000, // 4GB
            num_layers: 24,
            hidden_dim: 2048,
            vocab_size: 50000,
            architecture: "test".to_string(),
        };

        let expected_gb = 4.0 / 1.073741824; // 4GB / GiB conversion
        assert!((stats.memory_gb() - expected_gb).abs() < 0.1);
    }

    #[test]
    fn test_stats_parameters_b() {
        let stats = ConversionStats {
            total_parameters: 7_000_000_000,  // 7B params
            memory_bytes_f32: 28_000_000_000, // 28GB
            num_layers: 32,
            hidden_dim: 4096,
            vocab_size: 32000,
            architecture: "llama".to_string(),
        };

        assert!((stats.parameters_b() - 7.0).abs() < 0.001);
    }

    #[test]
    fn test_stats_debug() {
        let stats = ConversionStats {
            total_parameters: 1000,
            memory_bytes_f32: 4000,
            num_layers: 1,
            hidden_dim: 32,
            vocab_size: 100,
            architecture: "mini".to_string(),
        };

        // Test Debug trait
        let debug_str = format!("{stats:?}");
        assert!(debug_str.contains("mini"));
        assert!(debug_str.contains("1000"));
    }

    #[test]
    fn test_stats_clone() {
        let stats = ConversionStats {
            total_parameters: 500,
            memory_bytes_f32: 2000,
            num_layers: 2,
            hidden_dim: 16,
            vocab_size: 50,
            architecture: "tiny".to_string(),
        };

        let cloned = stats.clone();
        assert_eq!(cloned.total_parameters, stats.total_parameters);
        assert_eq!(cloned.architecture, stats.architecture);
    }

    // ==========================================================================
    // Error Path Coverage Tests
    // ==========================================================================

    #[test]
    fn test_from_apr_bytes_truncated_tensor_index() {
        // Create bytes with valid v2 header but truncated before tensor index
        let mut bytes = vec![0u8; 80]; // Just past header
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2; // v2
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes()); // 1 tensor
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata offset
        bytes[20..24].copy_from_slice(&2u32.to_le_bytes()); // metadata size
        bytes[24..32].copy_from_slice(&66u64.to_le_bytes()); // tensor index offset
        bytes[32..40].copy_from_slice(&200u64.to_le_bytes()); // data offset beyond end
        bytes[64..66].copy_from_slice(b"{}"); // minimal JSON metadata

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err()); // Should fail: truncated
    }

    #[test]
    fn test_from_apr_bytes_truncated_tensor_data() {
        // Create bytes with valid header and index but truncated data
        let mut bytes = vec![0u8; 128];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2;
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes());
        bytes[20..24].copy_from_slice(&2u32.to_le_bytes());
        bytes[24..32].copy_from_slice(&66u64.to_le_bytes());
        bytes[32..40].copy_from_slice(&110u64.to_le_bytes()); // data starts at 110
        bytes[64..66].copy_from_slice(b"{}");

        // Add a tensor index entry pointing to data beyond file end
        let index_json =
            r#"[{"name":"weights","dtype":"json","shape":[1000],"offset":0,"size":1000}]"#;
        let index_bytes = index_json.as_bytes();
        let index_end = 66 + index_bytes.len();
        bytes.resize(index_end + 10, 0); // Only add 10 bytes, not 1000
        bytes[66..index_end].copy_from_slice(index_bytes);

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err()); // Should fail: truncated tensor data
    }

    #[test]
    fn test_from_apr_bytes_invalid_json_tensor_index() {
        // Create bytes with valid header but invalid JSON in tensor index
        let mut bytes = vec![0u8; 100];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2;
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes());
        bytes[20..24].copy_from_slice(&2u32.to_le_bytes());
        bytes[24..32].copy_from_slice(&66u64.to_le_bytes()); // index at 66
        bytes[32..40].copy_from_slice(&90u64.to_le_bytes()); // data at 90
        bytes[64..66].copy_from_slice(b"{}");
        // Invalid JSON at tensor index position
        bytes[66..78].copy_from_slice(b"not valid js");

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err()); // Should fail: invalid JSON
    }

    // ==========================================================================
    // RawTensor Coverage Tests
    // ==========================================================================

    #[test]
    fn test_raw_tensor_debug() {
        let tensor = RawTensor {
            name: "test.weight".to_string(),
            data: vec![0u8; 100],
            shape: vec![10, 10],
            dtype: 0, // F32
        };

        let debug_str = format!("{tensor:?}");
        assert!(debug_str.contains("test.weight"));
        assert!(debug_str.contains("[10, 10]"));
    }

    #[test]
    fn test_raw_tensor_clone() {
        let tensor = RawTensor {
            name: "test.weight".to_string(),
            data: vec![1, 2, 3, 4],
            shape: vec![2, 2],
            dtype: 1, // F16
        };

        let cloned = tensor.clone();
        assert_eq!(cloned.name, tensor.name);
        assert_eq!(cloned.data, tensor.data);
        assert_eq!(cloned.shape, tensor.shape);
        assert_eq!(cloned.dtype, tensor.dtype);
    }
