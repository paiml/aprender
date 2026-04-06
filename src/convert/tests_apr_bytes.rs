
    #[test]
    fn test_apr_bytes_large_model_cov() {
        // Larger model to test serialization
        let apr = create_test_apr_transformer(64, 4, 100, 32);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        // Should successfully serialize
        assert!(!bytes.is_empty());
        assert_eq!(&bytes[0..4], &MAGIC);
    }

    // =========================================================================
    // Coverage Tests: from_apr_bytes edge cases
    // =========================================================================

    #[test]
    fn test_from_apr_bytes_too_short_cov() {
        let bytes = vec![0u8; 10]; // Too short for header
        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_apr_bytes_bad_magic_cov() {
        let mut bytes = vec![0u8; 128];
        bytes[0..4].copy_from_slice(b"BADM"); // Wrong magic
        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_apr_bytes_wrong_version_cov() {
        let mut bytes = vec![0u8; 128];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 99; // Unsupported version
        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err());
    }

    // =========================================================================
    // Coverage Tests: Q4K converter helper functions
    // =========================================================================

    #[test]
    fn test_q4k_converter_get_string_missing_cov() {
        use std::collections::HashMap;
        let metadata: HashMap<String, crate::gguf::GGUFValue> = HashMap::new();
        let result = GgufToAprQ4KConverter::get_string(&metadata, "nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_q4k_converter_get_u32_missing_cov() {
        use std::collections::HashMap;
        let metadata: HashMap<String, crate::gguf::GGUFValue> = HashMap::new();
        let result = GgufToAprQ4KConverter::get_u32(&metadata, "nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_q4k_converter_get_f32_missing_cov() {
        use std::collections::HashMap;
        let metadata: HashMap<String, crate::gguf::GGUFValue> = HashMap::new();
        let result = GgufToAprQ4KConverter::get_f32(&metadata, "nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_q4k_converter_get_u32_from_uint32_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt32(256));
        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert_eq!(result, Some(256));
    }

    #[test]
    fn test_q4k_converter_get_u32_from_uint64_large_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt64(1024));
        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert_eq!(result, Some(1024));
    }

    #[test]
    fn test_q4k_converter_get_f32_from_float32_cov() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Float32(1.5));
        let result = GgufToAprQ4KConverter::get_f32(&metadata, "key");
        assert!(result.is_some());
        assert!((result.expect("operation failed") - 1.5).abs() < 0.0001);
    }

    // =========================================================================
    // PMAT-107: infer_rope_type Tests
    // =========================================================================

    #[test]
    fn test_pmat_107_infer_rope_type_qwen2_is_neox() {
        // Qwen2 architecture should use NEOX style (type 2)
        let metadata: std::collections::HashMap<String, crate::gguf::GGUFValue> =
            std::collections::HashMap::new();
        let result = GgufToAprQ4KConverter::infer_rope_type("qwen2", &metadata);
        assert_eq!(result, 2, "Qwen2 should use NEOX style (rope_type=2)");
    }

    #[test]
    fn test_pmat_107_infer_rope_type_llama_is_norm() {
        // LLaMA architecture should use NORM style (type 0)
        let metadata: std::collections::HashMap<String, crate::gguf::GGUFValue> =
            std::collections::HashMap::new();
        let result = GgufToAprQ4KConverter::infer_rope_type("llama", &metadata);
        assert_eq!(result, 0, "LLaMA should use NORM style (rope_type=0)");
    }

    #[test]
    fn test_pmat_107_infer_rope_type_phi3_is_neox() {
        // Phi3 architecture should use NEOX style (type 2)
        let metadata: std::collections::HashMap<String, crate::gguf::GGUFValue> =
            std::collections::HashMap::new();
        let result = GgufToAprQ4KConverter::infer_rope_type("phi3", &metadata);
        assert_eq!(result, 2, "Phi3 should use NEOX style (rope_type=2)");
    }

    #[test]
    fn test_pmat_107_infer_rope_type_gemma_is_neox() {
        // Gemma architecture should use NEOX style (type 2)
        let metadata: std::collections::HashMap<String, crate::gguf::GGUFValue> =
            std::collections::HashMap::new();
        let result = GgufToAprQ4KConverter::infer_rope_type("gemma", &metadata);
        assert_eq!(result, 2, "Gemma should use NEOX style (rope_type=2)");
    }

    #[test]
    fn test_pmat_107_infer_rope_type_scaling_yarn_overrides() {
        // rope.scaling.type=yarn should override architecture inference
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "llama.rope.scaling.type".to_string(),
            GGUFValue::String("yarn".to_string()),
        );
        let result = GgufToAprQ4KConverter::infer_rope_type("llama", &metadata);
        assert_eq!(result, 2, "yarn scaling type should override to NEOX style");
    }

    #[test]
    fn test_pmat_107_infer_rope_type_scaling_linear_is_norm() {
        // rope.scaling.type=linear should use NORM style
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "qwen2.rope.scaling.type".to_string(),
            GGUFValue::String("linear".to_string()),
        );
        let result = GgufToAprQ4KConverter::infer_rope_type("qwen2", &metadata);
        assert_eq!(result, 0, "linear scaling type should use NORM style");
    }

    #[test]
    fn test_pmat_107_infer_rope_type_unknown_defaults_to_norm() {
        // Unknown architecture should default to NORM style (type 0)
        let metadata: std::collections::HashMap<String, crate::gguf::GGUFValue> =
            std::collections::HashMap::new();
        let result = GgufToAprQ4KConverter::infer_rope_type("unknown_arch", &metadata);
        assert_eq!(result, 0, "Unknown arch should default to NORM style");
    }

    // =========================================================================
    // Coverage Tests: ConversionStats
    // =========================================================================

    #[test]
    fn test_conversion_stats_fields_cov() {
        let stats = ConversionStats {
            total_parameters: 1000000,
            memory_bytes_f32: 4000000,
            num_layers: 12,
            hidden_dim: 768,
            vocab_size: 32000,
            architecture: "llama".to_string(),
        };
        assert_eq!(stats.total_parameters, 1000000);
        assert_eq!(stats.memory_bytes_f32, 4000000);
        assert_eq!(stats.num_layers, 12);
        assert_eq!(stats.hidden_dim, 768);
        assert_eq!(stats.vocab_size, 32000);
        assert_eq!(stats.architecture, "llama");
    }

    #[test]
    fn test_conversion_stats_zero_values_cov() {
        let stats = ConversionStats {
            total_parameters: 0,
            memory_bytes_f32: 0,
            num_layers: 0,
            hidden_dim: 0,
            vocab_size: 0,
            architecture: String::new(),
        };
        assert_eq!(stats.total_parameters, 0);
        assert_eq!(stats.memory_bytes_f32, 0);
        assert_eq!(stats.num_layers, 0);
    }

    #[test]
    fn test_conversion_stats_debug_clone_cov() {
        let stats = ConversionStats {
            total_parameters: 500000,
            memory_bytes_f32: 2000000,
            num_layers: 6,
            hidden_dim: 512,
            vocab_size: 10000,
            architecture: "phi2".to_string(),
        };
        let debug = format!("{:?}", stats);
        assert!(debug.contains("ConversionStats"));

        let cloned = stats.clone();
        assert_eq!(cloned.total_parameters, stats.total_parameters);
    }

    // =========================================================================
    // Coverage Tests: AprTransformerConfig
    // =========================================================================

    #[test]
    fn test_apr_transformer_config_partial_eq_cov() {
        let config1 = create_test_apr_transformer(4, 1, 10, 8).config;
        let config2 = create_test_apr_transformer(4, 1, 10, 8).config;
        assert_eq!(config1, config2);
    }

    #[test]
    fn test_apr_transformer_config_not_equal_cov() {
        let config1 = create_test_apr_transformer(4, 1, 10, 8).config;
        let config2 = create_test_apr_transformer(8, 2, 20, 16).config;
        assert_ne!(config1, config2);
    }

    // =========================================================================
    // Coverage Tests: RawTensor struct
    // =========================================================================

    #[test]
    fn test_raw_tensor_debug_cov() {
        let tensor = RawTensor {
            name: "test_tensor".to_string(),
            data: vec![1, 2, 3, 4],
            shape: vec![2, 2],
            dtype: 0, // F32
        };
        let debug_str = format!("{:?}", tensor);
        assert!(debug_str.contains("test_tensor"));
        assert!(debug_str.contains("shape"));
    }

    #[test]
    fn test_raw_tensor_clone_cov() {
        let tensor = RawTensor {
            name: "cloneable".to_string(),
            data: vec![10, 20, 30],
            shape: vec![3],
            dtype: 1, // F16
        };
        let cloned = tensor.clone();
        assert_eq!(cloned.name, tensor.name);
        assert_eq!(cloned.data, tensor.data);
        assert_eq!(cloned.shape, tensor.shape);
        assert_eq!(cloned.dtype, tensor.dtype);
    }

    #[test]
    fn test_raw_tensor_empty_data_cov() {
        let tensor = RawTensor {
            name: "empty".to_string(),
            data: vec![],
            shape: vec![0],
            dtype: 0,
        };
        assert!(tensor.data.is_empty());
        assert_eq!(tensor.shape, vec![0]);
    }

    #[test]
    fn test_raw_tensor_large_dtype_cov() {
        let tensor = RawTensor {
            name: "q4k".to_string(),
            data: vec![0u8; 144], // Q4_K: 256 elements = 144 bytes
            shape: vec![256],
            dtype: 12, // Q4_K
        };
        assert_eq!(tensor.dtype, 12);
        assert_eq!(tensor.data.len(), 144);
    }

    #[test]
    fn test_raw_tensor_multidim_shape_cov() {
        let tensor = RawTensor {
            name: "3d_tensor".to_string(),
            data: vec![0u8; 24],
            shape: vec![2, 3, 4],
            dtype: 0,
        };
        assert_eq!(tensor.shape.len(), 3);
        let total: usize = tensor.shape.iter().product();
        assert_eq!(total, 24);
    }

    // =========================================================================
    // Coverage Tests: Q4KConversionStats struct
    // =========================================================================

    #[test]
    fn test_q4k_conversion_stats_all_fields_cov() {
        let stats = Q4KConversionStats {
            tensor_count: 200,
            q4k_tensor_count: 150,
            total_bytes: 5_000_000,
            architecture: "mistral".to_string(),
            num_layers: 24,
            hidden_size: 3072,
        };
        assert_eq!(stats.tensor_count, 200);
        assert_eq!(stats.q4k_tensor_count, 150);
        assert_eq!(stats.total_bytes, 5_000_000);
        assert_eq!(stats.architecture, "mistral");
        assert_eq!(stats.num_layers, 24);
        assert_eq!(stats.hidden_size, 3072);
    }

    #[test]
    fn test_q4k_conversion_stats_zero_values_cov() {
        let stats = Q4KConversionStats {
            tensor_count: 0,
            q4k_tensor_count: 0,
            total_bytes: 0,
            architecture: String::new(),
            num_layers: 0,
            hidden_size: 0,
        };
        assert_eq!(stats.tensor_count, 0);
        assert_eq!(stats.total_bytes, 0);
        assert!(stats.architecture.is_empty());
    }

    // =========================================================================
    // Coverage Tests: GgufToAprConverter::stats
    // =========================================================================

    #[test]
    fn test_stats_function_cov() {
        let apr = create_test_apr_transformer(16, 4, 1000, 64);
        let stats = GgufToAprConverter::stats(&apr);

        assert_eq!(stats.num_layers, 4);
        assert_eq!(stats.hidden_dim, 16);
        assert_eq!(stats.vocab_size, 1000);
        assert!(stats.total_parameters > 0);
        assert!(stats.memory_bytes_f32 > 0);
    }

    #[test]
    fn test_stats_memory_calculations_cov() {
        let apr = create_test_apr_transformer(32, 2, 500, 128);
        let stats = GgufToAprConverter::stats(&apr);

        // Memory should be related to parameters
        assert!(stats.memory_mb() > 0.0);
        assert!(stats.memory_gb() < stats.memory_mb());
        assert!(stats.parameters_m() > 0.0);
    }

    // =========================================================================
    // Coverage Tests: from_apr_bytes additional error paths
    // =========================================================================

    #[test]
    fn test_from_apr_bytes_truncated_data_cov() {
        // Create valid header but truncate before data section
        let apr = create_test_apr_transformer(4, 1, 10, 8);
        let mut bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        // Truncate to only keep header and partial metadata
        bytes.truncate(100);

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_apr_bytes_invalid_metadata_json_cov() {
        let mut bytes = vec![0u8; 256];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2; // v2
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes()); // 1 tensor
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata offset
        bytes[20..24].copy_from_slice(&50u32.to_le_bytes()); // metadata size = 50
        bytes[24..32].copy_from_slice(&120u64.to_le_bytes()); // tensor index offset
        bytes[32..40].copy_from_slice(&200u64.to_le_bytes()); // data offset

        // Invalid JSON metadata (just garbage bytes)
        for i in 64..114 {
            bytes[i] = b'X';
        }

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        // Should error due to invalid metadata JSON
        assert!(result.is_err());
    }

    // =========================================================================
    // Coverage Tests: to_apr_bytes edge cases
    // =========================================================================

    #[test]
    fn test_to_apr_bytes_empty_layers_cov() {
        let apr = create_test_apr_transformer(4, 0, 10, 8);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        // Should still serialize with valid header
        assert_eq!(&bytes[0..4], &MAGIC);
        assert!(bytes.len() > HEADER_SIZE);
    }

    #[test]
    fn test_to_apr_bytes_large_vocab_cov() {
        let apr = create_test_apr_transformer(8, 1, 50000, 32);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        assert!(!bytes.is_empty());
        assert_eq!(&bytes[0..4], &MAGIC);
    }

    #[test]
    fn test_to_apr_bytes_small_dims_cov() {
        // Minimum viable transformer
        let apr = create_test_apr_transformer(2, 1, 4, 2);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        assert!(!bytes.is_empty());
    }
