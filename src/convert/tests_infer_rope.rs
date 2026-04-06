
    // =========================================================================
    // GgufToAprQ4KConverter::infer_rope_type
    // =========================================================================

    #[test]
    fn test_infer_rope_type_llama_norm() {
        let metadata = HashMap::new();
        let rope = GgufToAprQ4KConverter::infer_rope_type("llama", &metadata);
        assert_eq!(rope, 0); // NORM style
    }

    #[test]
    fn test_infer_rope_type_qwen2_neox() {
        let metadata = HashMap::new();
        let rope = GgufToAprQ4KConverter::infer_rope_type("qwen2", &metadata);
        assert_eq!(rope, 2); // NEOX style
    }

    #[test]
    fn test_infer_rope_type_qwen3_neox() {
        let metadata = HashMap::new();
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("qwen3", &metadata),
            2
        );
    }

    #[test]
    fn test_infer_rope_type_phi2_neox() {
        let metadata = HashMap::new();
        assert_eq!(GgufToAprQ4KConverter::infer_rope_type("phi2", &metadata), 2);
    }

    #[test]
    fn test_infer_rope_type_phi3_neox() {
        let metadata = HashMap::new();
        assert_eq!(GgufToAprQ4KConverter::infer_rope_type("phi3", &metadata), 2);
    }

    #[test]
    fn test_infer_rope_type_gemma_neox() {
        let metadata = HashMap::new();
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("gemma", &metadata),
            2
        );
    }

    #[test]
    fn test_infer_rope_type_gemma2_neox() {
        let metadata = HashMap::new();
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("gemma2", &metadata),
            2
        );
    }

    #[test]
    fn test_infer_rope_type_falcon_neox() {
        let metadata = HashMap::new();
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("falcon", &metadata),
            2
        );
    }

    #[test]
    fn test_infer_rope_type_stablelm_neox() {
        let metadata = HashMap::new();
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("stablelm", &metadata),
            2
        );
    }

    #[test]
    fn test_infer_rope_type_starcoder2_neox() {
        let metadata = HashMap::new();
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("starcoder2", &metadata),
            2
        );
    }

    #[test]
    fn test_infer_rope_type_gptneox_neox() {
        let metadata = HashMap::new();
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("gptneox", &metadata),
            2
        );
    }

    #[test]
    fn test_infer_rope_type_bert_neox() {
        let metadata = HashMap::new();
        assert_eq!(GgufToAprQ4KConverter::infer_rope_type("bert", &metadata), 2);
    }

    #[test]
    fn test_infer_rope_type_deepseek2_neox() {
        let metadata = HashMap::new();
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("deepseek2", &metadata),
            2
        );
    }

    #[test]
    fn test_infer_rope_type_internlm2_neox() {
        let metadata = HashMap::new();
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("internlm2", &metadata),
            2
        );
    }

    #[test]
    fn test_infer_rope_type_unknown_defaults_norm() {
        let metadata = HashMap::new();
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("custom_arch", &metadata),
            0
        );
    }

    #[test]
    fn test_infer_rope_type_explicit_scaling_none() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert(
            "llama.rope.scaling.type".to_string(),
            GGUFValue::String("none".to_string()),
        );
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("llama", &metadata),
            0
        );
    }

    #[test]
    fn test_infer_rope_type_explicit_scaling_linear() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert(
            "llama.rope.scaling.type".to_string(),
            GGUFValue::String("linear".to_string()),
        );
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("llama", &metadata),
            0
        );
    }

    #[test]
    fn test_infer_rope_type_explicit_scaling_yarn() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert(
            "llama.rope.scaling.type".to_string(),
            GGUFValue::String("yarn".to_string()),
        );
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("llama", &metadata),
            2
        );
    }

    #[test]
    fn test_infer_rope_type_explicit_scaling_neox() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert(
            "llama.rope.scaling.type".to_string(),
            GGUFValue::String("neox".to_string()),
        );
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("llama", &metadata),
            2
        );
    }

    #[test]
    fn test_infer_rope_type_explicit_scaling_unknown_falls_through() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert(
            "llama.rope.scaling.type".to_string(),
            GGUFValue::String("custom_scale".to_string()),
        );
        // Unknown scaling type → falls through to architecture check → llama → 0
        assert_eq!(
            GgufToAprQ4KConverter::infer_rope_type("llama", &metadata),
            0
        );
    }

    // =========================================================================
    // GgufToAprQ4KConverter helper methods
    // =========================================================================

    #[test]
    fn test_get_string_found() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::String("value".to_string()));
        let result = GgufToAprQ4KConverter::get_string(&metadata, "key");
        assert_eq!(result, Some("value".to_string()));
    }

    #[test]
    fn test_get_string_missing() {
        let metadata = HashMap::new();
        assert_eq!(GgufToAprQ4KConverter::get_string(&metadata, "key"), None);
    }

    #[test]
    fn test_get_string_wrong_type() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt32(42));
        assert_eq!(GgufToAprQ4KConverter::get_string(&metadata, "key"), None);
    }

    #[test]
    fn test_get_u32_from_uint32() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt32(42));
        assert_eq!(GgufToAprQ4KConverter::get_u32(&metadata, "key"), Some(42));
    }

    #[test]
    fn test_get_u32_from_int32() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Int32(42));
        assert_eq!(GgufToAprQ4KConverter::get_u32(&metadata, "key"), Some(42));
    }

    #[test]
    fn test_get_u32_from_uint64() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::UInt64(100));
        assert_eq!(GgufToAprQ4KConverter::get_u32(&metadata, "key"), Some(100));
    }

    #[test]
    fn test_get_u32_missing() {
        let metadata = HashMap::new();
        assert_eq!(GgufToAprQ4KConverter::get_u32(&metadata, "key"), None);
    }

    #[test]
    fn test_get_u32_wrong_type() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert(
            "key".to_string(),
            GGUFValue::String("not_a_number".to_string()),
        );
        assert_eq!(GgufToAprQ4KConverter::get_u32(&metadata, "key"), None);
    }

    #[test]
    fn test_get_f32_from_float32() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Float32(3.14));
        let result = GgufToAprQ4KConverter::get_f32(&metadata, "key");
        assert!(result.is_some());
        assert!((result.unwrap() - 3.14).abs() < 1e-6);
    }

    #[test]
    fn test_get_f32_from_float64() {
        use crate::gguf::GGUFValue;
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), GGUFValue::Float64(2.718));
        let result = GgufToAprQ4KConverter::get_f32(&metadata, "key");
        assert!(result.is_some());
        assert!((result.unwrap() - 2.718).abs() < 1e-3);
    }

    #[test]
    fn test_get_f32_missing() {
        let metadata = HashMap::new();
        assert_eq!(GgufToAprQ4KConverter::get_f32(&metadata, "key"), None);
    }

    // =========================================================================
    // RawTensor struct
    // =========================================================================

    #[test]
    fn test_raw_tensor_construction() {
        let tensor = RawTensor {
            name: "layer.0.weight".to_string(),
            data: vec![0u8; 144],
            shape: vec![256],
            dtype: 12, // Q4_K
        };
        assert_eq!(tensor.name, "layer.0.weight");
        assert_eq!(tensor.data.len(), 144);
        assert_eq!(tensor.shape, vec![256]);
        assert_eq!(tensor.dtype, 12);
    }

    #[test]
    fn test_raw_tensor_debug() {
        let tensor = RawTensor {
            name: "test".to_string(),
            data: vec![1, 2, 3],
            shape: vec![3],
            dtype: 0,
        };
        let debug = format!("{:?}", tensor);
        assert!(debug.contains("test"));
    }

    #[test]
    fn test_raw_tensor_clone() {
        let tensor = RawTensor {
            name: "test".to_string(),
            data: vec![1, 2, 3],
            shape: vec![3],
            dtype: 0,
        };
        let cloned = tensor.clone();
        assert_eq!(cloned.name, tensor.name);
        assert_eq!(cloned.data, tensor.data);
    }

    // =========================================================================
    // Q4KConversionStats
    // =========================================================================

    #[test]
    fn test_q4k_conversion_stats_debug() {
        let stats = Q4KConversionStats {
            tensor_count: 100,
            q4k_tensor_count: 80,
            total_bytes: 1_000_000,
            architecture: "qwen2".to_string(),
            num_layers: 28,
            hidden_size: 1536,
        };
        let debug = format!("{:?}", stats);
        assert!(debug.contains("qwen2"));
        assert!(debug.contains("1536"));
    }

    #[test]
    fn test_q4k_conversion_stats_clone() {
        let stats = Q4KConversionStats {
            tensor_count: 10,
            q4k_tensor_count: 8,
            total_bytes: 500,
            architecture: "llama".to_string(),
            num_layers: 2,
            hidden_size: 64,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.tensor_count, stats.tensor_count);
        assert_eq!(cloned.architecture, stats.architecture);
    }

    // =========================================================================
    // ConversionStats edge cases
    // =========================================================================

    #[test]
    fn test_conversion_stats_zero_values() {
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
    fn test_conversion_stats_large_model() {
        let stats = ConversionStats {
            total_parameters: 7_000_000_000,
            memory_bytes_f32: 28_000_000_000,
            num_layers: 32,
            hidden_dim: 4096,
            vocab_size: 32000,
            architecture: "llama".to_string(),
        };
        assert!((stats.parameters_b() - 7.0).abs() < 0.01);
        assert!((stats.parameters_m() - 7000.0).abs() < 1.0);
        assert!(stats.memory_gb() > 25.0);
        assert!(stats.memory_mb() > 25000.0);
    }

    #[test]
    fn test_conversion_stats_debug() {
        let stats = ConversionStats {
            total_parameters: 100,
            memory_bytes_f32: 400,
            num_layers: 1,
            hidden_dim: 4,
            vocab_size: 8,
            architecture: "test".to_string(),
        };
        let debug = format!("{:?}", stats);
        assert!(debug.contains("test"));
    }

    // =========================================================================
    // from_apr_bytes additional error paths
    // =========================================================================

    #[test]
    fn test_from_apr_bytes_truncated_after_header() {
        use crate::apr::{HEADER_SIZE, MAGIC};

        // Build a header that claims tensor index is beyond data
        let mut header = vec![0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&MAGIC);
        header[4] = 2; // version major
        header[5] = 0; // version minor
                       // tensor_index_offset pointing beyond the data
        header[24..32].copy_from_slice(&1000u64.to_le_bytes());
        // data_offset
        header[32..40].copy_from_slice(&2000u64.to_le_bytes());

        let result = GgufToAprConverter::from_apr_bytes(&header);
        assert!(result.is_err());
    }
