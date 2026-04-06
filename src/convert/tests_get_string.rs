
    // ==========================================================================
    // GgufToAprQ4KConverter Helper Tests
    // ==========================================================================

    #[test]
    fn test_get_string_helper() {
        use crate::gguf::GGUFValue;
        use std::collections::HashMap;

        let mut metadata = HashMap::new();
        metadata.insert(
            "name".to_string(),
            GGUFValue::String("test_model".to_string()),
        );
        metadata.insert("count".to_string(), GGUFValue::UInt32(42));

        let result = GgufToAprQ4KConverter::get_string(&metadata, "name");
        assert_eq!(result, Some("test_model".to_string()));

        let missing = GgufToAprQ4KConverter::get_string(&metadata, "nonexistent");
        assert_eq!(missing, None);

        // Test wrong type returns None
        let wrong_type = GgufToAprQ4KConverter::get_string(&metadata, "count");
        assert_eq!(wrong_type, None);
    }

    #[test]
    fn test_get_u32_helper() {
        use crate::gguf::GGUFValue;
        use std::collections::HashMap;

        let mut metadata = HashMap::new();
        metadata.insert("count".to_string(), GGUFValue::UInt32(42));
        metadata.insert("signed".to_string(), GGUFValue::Int32(100));
        metadata.insert("big".to_string(), GGUFValue::UInt64(200));
        metadata.insert("name".to_string(), GGUFValue::String("test".to_string()));

        let result = GgufToAprQ4KConverter::get_u32(&metadata, "count");
        assert_eq!(result, Some(42));

        let signed = GgufToAprQ4KConverter::get_u32(&metadata, "signed");
        assert_eq!(signed, Some(100));

        let big = GgufToAprQ4KConverter::get_u32(&metadata, "big");
        assert_eq!(big, Some(200));

        let missing = GgufToAprQ4KConverter::get_u32(&metadata, "nonexistent");
        assert_eq!(missing, None);

        let wrong_type = GgufToAprQ4KConverter::get_u32(&metadata, "name");
        assert_eq!(wrong_type, None);
    }

    #[test]
    fn test_get_f32_helper() {
        use crate::gguf::GGUFValue;
        use std::collections::HashMap;

        let mut metadata = HashMap::new();
        metadata.insert("scale".to_string(), GGUFValue::Float32(3.14));
        metadata.insert("big_scale".to_string(), GGUFValue::Float64(2.71828));
        metadata.insert("count".to_string(), GGUFValue::UInt32(42));

        let result = GgufToAprQ4KConverter::get_f32(&metadata, "scale");
        assert!(result.is_some());
        assert!((result.expect("operation failed") - 3.14).abs() < 0.001);

        let big = GgufToAprQ4KConverter::get_f32(&metadata, "big_scale");
        assert!(big.is_some());
        assert!((big.expect("operation failed") - 2.71828).abs() < 0.001);

        let missing = GgufToAprQ4KConverter::get_f32(&metadata, "nonexistent");
        assert_eq!(missing, None);

        let wrong_type = GgufToAprQ4KConverter::get_f32(&metadata, "count");
        assert_eq!(wrong_type, None);
    }

    // ==========================================================================
    // Q4KConversionStats Coverage Tests
    // ==========================================================================

    #[test]
    fn test_q4k_conversion_stats_debug() {
        let stats = Q4KConversionStats {
            tensor_count: 100,
            q4k_tensor_count: 80,
            total_bytes: 1_000_000,
            architecture: "llama".to_string(),
            num_layers: 32,
            hidden_size: 4096,
        };

        let debug_str = format!("{stats:?}");
        assert!(debug_str.contains("llama"));
        assert!(debug_str.contains("100"));
        assert!(debug_str.contains("32"));
    }

    #[test]
    fn test_q4k_conversion_stats_clone() {
        let stats = Q4KConversionStats {
            tensor_count: 50,
            q4k_tensor_count: 40,
            total_bytes: 500_000,
            architecture: "qwen".to_string(),
            num_layers: 16,
            hidden_size: 2048,
        };

        let cloned = stats.clone();
        assert_eq!(cloned.tensor_count, stats.tensor_count);
        assert_eq!(cloned.architecture, stats.architecture);
        assert_eq!(cloned.num_layers, stats.num_layers);
    }

    // ==========================================================================
    // Additional From APR Bytes Error Tests
    // ==========================================================================

    #[test]
    fn test_from_apr_bytes_v1_format() {
        // Create APR v1 format header (should be handled or error gracefully)
        let mut bytes = vec![0u8; 128];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 1; // v1 (not v2)
        bytes[5] = 0;

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        // May succeed with v1 fallback or fail, but shouldn't panic
        let _ = result;
    }

    #[test]
    fn test_from_apr_bytes_wrong_magic() {
        let mut bytes = vec![0u8; 128];
        bytes[0..4].copy_from_slice(b"XXXX"); // Wrong magic
        bytes[4] = 2;

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_apr_bytes_too_short() {
        // Only 4 bytes (magic only)
        let bytes = vec![0x41, 0x50, 0x52, 0x32]; // APR2

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err());
    }

    // ==========================================================================
    // Stats Edge Cases
    // ==========================================================================

    #[test]
    fn test_stats_zero_params() {
        let stats = ConversionStats {
            total_parameters: 0,
            memory_bytes_f32: 0,
            num_layers: 0,
            hidden_dim: 0,
            vocab_size: 0,
            architecture: "empty".to_string(),
        };

        assert_eq!(stats.memory_mb(), 0.0);
        assert_eq!(stats.memory_gb(), 0.0);
        assert_eq!(stats.parameters_m(), 0.0);
        assert_eq!(stats.parameters_b(), 0.0);
    }

    #[test]
    fn test_stats_small_model() {
        let stats = ConversionStats {
            total_parameters: 1000,
            memory_bytes_f32: 4000,
            num_layers: 1,
            hidden_dim: 16,
            vocab_size: 100,
            architecture: "tiny".to_string(),
        };

        assert!(stats.memory_mb() > 0.0);
        assert!(stats.parameters_m() > 0.0);
        assert!(stats.parameters_b() < 0.001);
    }

    // ==========================================================================
    // APR Bytes Serialization Additional Tests
    // ==========================================================================

    #[test]
    fn test_to_apr_bytes_multiple_layers() {
        let apr = create_test_apr_transformer(64, 4, 1000, 256);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        // Should have valid header
        assert_eq!(&bytes[0..4], &MAGIC);
        assert!(bytes.len() > HEADER_SIZE);
    }

    #[test]
    fn test_to_apr_bytes_single_layer() {
        let apr = create_test_apr_transformer(32, 1, 100, 64);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        assert_eq!(&bytes[0..4], &MAGIC);
    }

    #[test]
    fn test_apr_roundtrip_multiple_layers() {
        let original = create_test_apr_transformer(32, 3, 500, 128);
        let bytes = GgufToAprConverter::to_apr_bytes(&original).expect("serialize");
        let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

        assert_eq!(original.config.num_layers, loaded.config.num_layers);
        assert_eq!(original.layers.len(), loaded.layers.len());
    }

    // ==========================================================================
    // Coverage Tests: ConversionStats
    // ==========================================================================

    #[test]
    fn test_conversion_stats_debug() {
        let stats = ConversionStats {
            total_parameters: 1_000_000,
            memory_bytes_f32: 4_000_000,
            num_layers: 12,
            hidden_dim: 768,
            vocab_size: 50000,
            architecture: "bert".to_string(),
        };
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("ConversionStats"));
        assert!(debug_str.contains("1000000"));
        assert!(debug_str.contains("bert"));
    }

    #[test]
    fn test_conversion_stats_clone() {
        let stats = ConversionStats {
            total_parameters: 7_000_000_000,
            memory_bytes_f32: 28_000_000_000,
            num_layers: 32,
            hidden_dim: 4096,
            vocab_size: 32000,
            architecture: "llama".to_string(),
        };
        let cloned = stats.clone();
        assert_eq!(cloned.total_parameters, stats.total_parameters);
        assert_eq!(cloned.architecture, stats.architecture);
    }

    #[test]
    fn test_conversion_stats_large_model() {
        let stats = ConversionStats {
            total_parameters: 70_000_000_000, // 70B
            memory_bytes_f32: 280_000_000_000,
            num_layers: 80,
            hidden_dim: 8192,
            vocab_size: 128000,
            architecture: "llama3".to_string(),
        };
        assert!(stats.parameters_b() > 69.0 && stats.parameters_b() < 71.0);
        assert!(stats.memory_gb() > 250.0);
    }

    // ==========================================================================
    // Coverage Tests: from_apr_bytes error paths
    // ==========================================================================

    #[test]
    fn test_from_apr_bytes_invalid_tensor_index_json() {
        // Create header pointing to invalid JSON
        let mut bytes = vec![0u8; 200];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2; // v2
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes()); // 1 tensor
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata offset
        bytes[20..24].copy_from_slice(&2u32.to_le_bytes()); // metadata size
        bytes[24..32].copy_from_slice(&66u64.to_le_bytes()); // tensor index offset
        bytes[32..40].copy_from_slice(&100u64.to_le_bytes()); // data offset
        bytes[64..66].copy_from_slice(b"{}"); // metadata
        // Invalid JSON for tensor index (length must match exactly)
        let invalid_json = b"not valid json{{{";
        bytes[66..66 + invalid_json.len()].copy_from_slice(invalid_json);

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(result.is_err());
    }

    // ==========================================================================
    // Coverage Tests: GgufToAprQ4KConverter helpers
    // ==========================================================================

    #[test]
    fn test_q4k_converter_get_string_missing() {
        let metadata = std::collections::HashMap::new();
        let result = GgufToAprQ4KConverter::get_string(&metadata, "missing_key");
        assert!(result.is_none());
    }

    #[test]
    fn test_q4k_converter_get_u32_missing() {
        let metadata = std::collections::HashMap::new();
        let result = GgufToAprQ4KConverter::get_u32(&metadata, "missing_key");
        assert!(result.is_none());
    }

    #[test]
    fn test_q4k_converter_get_f32_missing() {
        let metadata = std::collections::HashMap::new();
        let result = GgufToAprQ4KConverter::get_f32(&metadata, "missing_key");
        assert!(result.is_none());
    }

    #[test]
    fn test_q4k_converter_get_string_present() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::String("value".to_string()));

        let result = GgufToAprQ4KConverter::get_string(&metadata, "key");
        assert_eq!(result, Some("value".to_string()));
    }

    #[test]
    fn test_q4k_converter_get_u32_from_int32() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Int32(42));

        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert_eq!(result, Some(42));
    }

    #[test]
    fn test_q4k_converter_get_u32_from_uint64() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt64(100));

        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert_eq!(result, Some(100));
    }

    #[test]
    fn test_q4k_converter_get_f32_from_float64() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Float64(3.14159));

        let result = GgufToAprQ4KConverter::get_f32(&metadata, "key");
        assert!(result.is_some());
        assert!((result.expect("operation failed") - 3.14159).abs() < 0.0001);
    }

    #[test]
    fn test_q4k_converter_get_string_wrong_type() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt32(42));

        let result = GgufToAprQ4KConverter::get_string(&metadata, "key");
        assert!(result.is_none());
    }

    #[test]
    fn test_q4k_converter_get_u32_wrong_type() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "key".to_string(),
            GGUFValue::String("not a number".to_string()),
        );

        let result = GgufToAprQ4KConverter::get_u32(&metadata, "key");
        assert!(result.is_none());
    }

    #[test]
    fn test_q4k_converter_get_f32_wrong_type() {
        use crate::gguf::GGUFValue;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "key".to_string(),
            GGUFValue::String("not a float".to_string()),
        );

        let result = GgufToAprQ4KConverter::get_f32(&metadata, "key");
        assert!(result.is_none());
    }

    // =========================================================================
    // Coverage Tests: GgufToAprConverter
    // =========================================================================

    #[test]
    fn test_gguf_to_apr_converter_zero_layers_cov() {
        let gguf = create_mock_gguf_transformer(4, 0, 10, 8);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
        assert!(apr.layers.is_empty());
    }

    #[test]
    fn test_gguf_to_apr_converter_multiple_layers_cov() {
        let gguf = create_mock_gguf_transformer(8, 4, 20, 16);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
        assert_eq!(apr.layers.len(), 4);
    }

    #[test]
    fn test_gguf_to_apr_converter_config_all_fields_cov() {
        let gguf = create_mock_gguf_transformer(16, 2, 32, 8);
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

        assert_eq!(apr.config.hidden_dim, gguf.config.hidden_dim);
        assert_eq!(apr.config.num_layers, gguf.config.num_layers);
        assert_eq!(apr.config.num_heads, gguf.config.num_heads);
        assert_eq!(apr.config.num_kv_heads, gguf.config.num_kv_heads);
        assert_eq!(apr.config.vocab_size, gguf.config.vocab_size);
        assert_eq!(apr.config.intermediate_dim, gguf.config.intermediate_dim);
        assert_eq!(apr.config.context_length, gguf.config.context_length);
        assert_eq!(apr.config.rope_theta, gguf.config.rope_theta);
        assert_eq!(apr.config.eps, gguf.config.eps);
    }

    // =========================================================================
    // Coverage Tests: APR bytes serialization
    // =========================================================================

    #[test]
    fn test_to_apr_bytes_alignment_cov() {
        let apr = create_test_apr_transformer(8, 2, 20, 8);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        // Check header is 64 bytes (HEADER_SIZE) and data follows
        assert!(bytes.len() >= HEADER_SIZE);
        // Metadata offset starts at HEADER_SIZE (64 bytes)
        let metadata_offset =
            u64::from_le_bytes(bytes[12..20].try_into().expect("index out of bounds")) as usize;
        assert_eq!(metadata_offset, HEADER_SIZE);
    }
