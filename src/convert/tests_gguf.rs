
    // =========================================================================
    // Coverage Tests: GgufToAprConverter::from_gguf_transformer edge cases
    // =========================================================================

    #[test]
    fn test_from_gguf_transformer_large_model_cov() {
        let gguf = create_mock_gguf_transformer(128, 8, 32000, 256);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

        assert_eq!(apr.config.hidden_dim, 128);
        assert_eq!(apr.config.num_layers, 8);
        assert_eq!(apr.config.vocab_size, 32000);
        assert_eq!(apr.layers.len(), 8);
    }

    #[test]
    fn test_from_gguf_transformer_single_head_cov() {
        let mut gguf = create_mock_gguf_transformer(8, 1, 10, 1);
        gguf.config.num_heads = 1;
        gguf.config.num_kv_heads = 1;

        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
        assert_eq!(apr.config.num_heads, 1);
        assert_eq!(apr.config.num_kv_heads, 1);
    }

    #[test]
    fn test_from_gguf_transformer_gqa_cov() {
        // Grouped Query Attention: num_kv_heads < num_heads
        let mut gguf = create_mock_gguf_transformer(32, 2, 100, 8);
        gguf.config.num_heads = 8;
        gguf.config.num_kv_heads = 2; // GQA ratio of 4:1

        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
        assert_eq!(apr.config.num_heads, 8);
        assert_eq!(apr.config.num_kv_heads, 2);
    }

    // =========================================================================
    // Coverage Tests: Layer weight preservation
    // =========================================================================

    #[test]
    fn test_layer_bias_preservation_cov() {
        let gguf = create_mock_gguf_transformer(8, 2, 20, 4);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

        for (apr_layer, gguf_layer) in apr.layers.iter().zip(gguf.layers.iter()) {
            assert_eq!(apr_layer.attn_norm_bias, gguf_layer.attn_norm_bias);
            assert_eq!(apr_layer.qkv_bias, gguf_layer.qkv_bias);
            assert_eq!(apr_layer.attn_output_bias, gguf_layer.attn_output_bias);
            assert_eq!(apr_layer.ffn_gate_bias, gguf_layer.ffn_gate_bias);
            assert_eq!(apr_layer.ffn_up_bias, gguf_layer.ffn_up_bias);
            assert_eq!(apr_layer.ffn_down_bias, gguf_layer.ffn_down_bias);
            assert_eq!(apr_layer.ffn_norm_bias, gguf_layer.ffn_norm_bias);
        }
    }

    #[test]
    fn test_layer_ffn_weights_cov() {
        let gguf = create_mock_gguf_transformer(16, 3, 50, 8);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

        for (apr_layer, gguf_layer) in apr.layers.iter().zip(gguf.layers.iter()) {
            assert_eq!(apr_layer.ffn_gate_weight, gguf_layer.ffn_gate_weight);
            assert_eq!(apr_layer.ffn_up_weight, gguf_layer.ffn_up_weight);
            assert_eq!(apr_layer.ffn_down_weight, gguf_layer.ffn_down_weight);
            assert_eq!(apr_layer.ffn_norm_weight, gguf_layer.ffn_norm_weight);
        }
    }

    #[test]
    fn test_layer_attention_weights_cov() {
        let gguf = create_mock_gguf_transformer(32, 4, 100, 8);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

        for (apr_layer, gguf_layer) in apr.layers.iter().zip(gguf.layers.iter()) {
            assert_eq!(apr_layer.attn_norm_weight, gguf_layer.attn_norm_weight);
            assert_eq!(apr_layer.qkv_weight, gguf_layer.qkv_weight);
            assert_eq!(apr_layer.attn_output_weight, gguf_layer.attn_output_weight);
        }
    }

    // =========================================================================
    // Coverage Tests: Config field preservation
    // =========================================================================

    #[test]
    fn test_config_rope_theta_cov() {
        let mut gguf = create_mock_gguf_transformer(8, 1, 10, 4);
        gguf.config.rope_theta = 1000000.0; // Llama-3 style

        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
        assert!((apr.config.rope_theta - 1000000.0).abs() < 0.01);
    }

    #[test]
    fn test_config_eps_cov() {
        let mut gguf = create_mock_gguf_transformer(8, 1, 10, 4);
        gguf.config.eps = 1e-6; // Different epsilon

        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
        assert!((apr.config.eps - 1e-6).abs() < 1e-9);
    }

    #[test]
    fn test_config_context_length_cov() {
        let mut gguf = create_mock_gguf_transformer(8, 1, 10, 4);
        gguf.config.context_length = 131072; // Long context

        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
        assert_eq!(apr.config.context_length, 131072);
    }

    #[test]
    fn test_config_intermediate_dim_cov() {
        let mut gguf = create_mock_gguf_transformer(8, 1, 10, 4);
        gguf.config.intermediate_dim = 2048;

        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
        assert_eq!(apr.config.intermediate_dim, 2048);
    }

    // =========================================================================
    // Coverage Tests: get_f32 Float64 conversion
    // =========================================================================

    #[test]
    fn test_q4k_converter_get_f32_float64_large_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Float64(1e10));
        let result = GgufToAprQ4KConverter::get_f32(&metadata, "key");
        assert!(result.is_some());
    }

    #[test]
    fn test_q4k_converter_get_f32_float64_small_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Float64(1e-10));
        let result = GgufToAprQ4KConverter::get_f32(&metadata, "key");
        assert!(result.is_some());
        assert!(result.expect("operation failed") > 0.0);
    }

    // =========================================================================
    // Coverage Tests: get_u32 Int32 conversion
    // =========================================================================

    #[test]
    fn test_q4k_converter_get_u32_int32_positive_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Int32(12345));
        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert_eq!(result, Some(12345));
    }

    #[test]
    fn test_q4k_converter_get_u32_int32_zero_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Int32(0));
        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert_eq!(result, Some(0));
    }

    // =========================================================================
    // Extended Coverage Tests for ConversionStats
    // =========================================================================

    #[test]
    fn test_conversion_stats_memory_mb_cov() {
        let stats = ConversionStats {
            total_parameters: 1_000_000,
            memory_bytes_f32: 4 * 1024 * 1024, // 4 MB
            num_layers: 12,
            hidden_dim: 768,
            vocab_size: 50257,
            architecture: "gpt2".to_string(),
        };
        assert!((stats.memory_mb() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_conversion_stats_memory_gb_cov() {
        let stats = ConversionStats {
            total_parameters: 1_000_000_000,
            memory_bytes_f32: 4 * 1024 * 1024 * 1024, // 4 GB
            num_layers: 32,
            hidden_dim: 4096,
            vocab_size: 32000,
            architecture: "llama".to_string(),
        };
        assert!((stats.memory_gb() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_conversion_stats_parameters_m_cov() {
        let stats = ConversionStats {
            total_parameters: 125_000_000, // 125M
            memory_bytes_f32: 500_000_000,
            num_layers: 12,
            hidden_dim: 768,
            vocab_size: 50257,
            architecture: "gpt2".to_string(),
        };
        assert!((stats.parameters_m() - 125.0).abs() < 0.1);
    }

    #[test]
    fn test_conversion_stats_parameters_b_cov() {
        let stats = ConversionStats {
            total_parameters: 7_000_000_000, // 7B
            memory_bytes_f32: 28_000_000_000,
            num_layers: 32,
            hidden_dim: 4096,
            vocab_size: 32000,
            architecture: "llama".to_string(),
        };
        assert!((stats.parameters_b() - 7.0).abs() < 0.1);
    }

    #[test]
    fn test_conversion_stats_clone_cov() {
        let stats = ConversionStats {
            total_parameters: 1000,
            memory_bytes_f32: 4000,
            num_layers: 1,
            hidden_dim: 64,
            vocab_size: 100,
            architecture: "test".to_string(),
        };
        let cloned = stats.clone();
        assert_eq!(stats.total_parameters, cloned.total_parameters);
        assert_eq!(stats.architecture, cloned.architecture);
    }

    #[test]
    fn test_conversion_stats_debug_cov() {
        let stats = ConversionStats {
            total_parameters: 1000,
            memory_bytes_f32: 4000,
            num_layers: 1,
            hidden_dim: 64,
            vocab_size: 100,
            architecture: "debug_test".to_string(),
        };
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("total_parameters"));
        assert!(debug_str.contains("debug_test"));
    }

    // =========================================================================
    // Extended Coverage Tests for RawTensor
    // =========================================================================

    #[test]
    fn test_raw_tensor_new_cov() {
        let tensor = RawTensor {
            name: "test_tensor".to_string(),
            data: vec![0u8; 100],
            shape: vec![10, 10],
            dtype: 0, // F32
        };
        assert_eq!(tensor.name, "test_tensor");
        assert_eq!(tensor.data.len(), 100);
        assert_eq!(tensor.shape, vec![10, 10]);
        assert_eq!(tensor.dtype, 0);
    }

    // =========================================================================
    // Extended Coverage Tests for Q4KConversionStats
    // =========================================================================

    #[test]
    fn test_q4k_conversion_stats_new_ext_cov() {
        let stats = Q4KConversionStats {
            tensor_count: 10,
            q4k_tensor_count: 5,
            total_bytes: 1000,
            architecture: "llama".to_string(),
            num_layers: 12,
            hidden_size: 768,
        };
        assert_eq!(stats.tensor_count, 10);
        assert_eq!(stats.q4k_tensor_count, 5);
        assert_eq!(stats.total_bytes, 1000);
        assert_eq!(stats.architecture, "llama");
        assert_eq!(stats.num_layers, 12);
        assert_eq!(stats.hidden_size, 768);
    }

    #[test]
    fn test_q4k_conversion_stats_clone_ext_cov() {
        let stats = Q4KConversionStats {
            tensor_count: 5,
            q4k_tensor_count: 3,
            total_bytes: 500,
            architecture: "gpt2".to_string(),
            num_layers: 6,
            hidden_size: 512,
        };
        let cloned = stats.clone();
        assert_eq!(stats.tensor_count, cloned.tensor_count);
        assert_eq!(stats.q4k_tensor_count, cloned.q4k_tensor_count);
        assert_eq!(stats.architecture, cloned.architecture);
    }

    #[test]
    fn test_q4k_conversion_stats_debug_ext_cov() {
        let stats = Q4KConversionStats {
            tensor_count: 1,
            q4k_tensor_count: 1,
            total_bytes: 100,
            architecture: "tiny".to_string(),
            num_layers: 2,
            hidden_size: 256,
        };
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("tensor_count"));
        assert!(debug_str.contains("architecture"));
    }

    // =========================================================================
    // Extended Coverage Tests for GgufToAprQ4KConverter helpers
    // =========================================================================

    #[test]
    fn test_q4k_converter_get_string_ext_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::String("value".to_string()));
        let result = GgufToAprQ4KConverter::get_string(&metadata, "key");
        assert_eq!(result, Some("value".to_string()));
    }

    #[test]
    fn test_q4k_converter_get_string_wrong_type_ext_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt32(42));
        let result = GgufToAprQ4KConverter::get_string(&metadata, "key");
        assert_eq!(result, None);
    }

    #[test]
    fn test_q4k_converter_get_u32_uint32_ext_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt32(12345));
        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert_eq!(result, Some(12345));
    }

    #[test]
    fn test_q4k_converter_get_u32_uint64_ext_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt64(999));
        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert_eq!(result, Some(999));
    }

    #[test]
    fn test_q4k_converter_get_u32_wrong_type_ext_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "key".to_string(),
            GGUFValue::String("not a number".to_string()),
        );
        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert_eq!(result, None);
    }

    #[test]
    fn test_q4k_converter_get_f32_float32_ext_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Float32(3.14159));
        let result = GgufToAprQ4KConverter::get_f32(&metadata, "key");
        assert!(result.is_some());
        assert!((result.expect("operation failed") - 3.14159).abs() < 1e-5);
    }

    #[test]
    fn test_q4k_converter_get_f32_wrong_type_ext_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt32(42));
        let result = GgufToAprQ4KConverter::get_f32(&metadata, "key");
        assert_eq!(result, None);
    }

    // ==========================================================================
    // Deep Coverage Tests (prefix: _deep_ccov_)
    // ==========================================================================

    #[test]
    fn test_deep_ccov_from_apr_bytes_invalid_transformer_json() {
        // Create valid APR header with a weights tensor that has invalid JSON
        let mut bytes = vec![0u8; 512];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2; // v2
        bytes[5] = 0;
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes()); // 1 tensor
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata offset
        bytes[20..24].copy_from_slice(&2u32.to_le_bytes()); // metadata size
        bytes[24..32].copy_from_slice(&66u64.to_le_bytes()); // tensor index offset
        bytes[32..40].copy_from_slice(&200u64.to_le_bytes()); // data offset

        // Add minimal metadata at offset 64
        bytes[64..66].copy_from_slice(b"{}");

        // Add tensor index entry at offset 66 pointing to "weights"
        let tensor_index =
            r#"[{"name":"weights","dtype":"json","shape":[100],"offset":0,"size":100}]"#;
        let index_bytes = tensor_index.as_bytes();
        bytes[66..66 + index_bytes.len()].copy_from_slice(index_bytes);

        // Add invalid JSON for the weights payload at offset 200
        let invalid_weights = b"{ this is not valid json }}}";
        bytes[200..200 + invalid_weights.len()].copy_from_slice(invalid_weights);

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(
            result.is_err(),
            "Should fail on invalid transformer JSON: {:?}",
            result
        );
    }
