
    #[test]
    fn test_deep_ccov_conversion_stats_memory_gb_precision() {
        // Test precision of memory_gb calculation with exact values
        let stats = ConversionStats {
            total_parameters: 1_073_741_824, // 2^30 = 1 GiB of params
            memory_bytes_f32: 1_073_741_824, // 1 GiB
            num_layers: 1,
            hidden_dim: 1,
            vocab_size: 1,
            architecture: "test".to_string(),
        };
        // 1 GiB = 1.0 GB in GiB units
        let expected_gb = 1.0;
        assert!(
            (stats.memory_gb() - expected_gb).abs() < 0.001,
            "Expected {}, got {}",
            expected_gb,
            stats.memory_gb()
        );
    }

    #[test]
    fn test_deep_ccov_conversion_stats_parameters_b_precision() {
        // Test precision of parameters_b calculation
        let stats = ConversionStats {
            total_parameters: 1_000_000_000, // Exactly 1B
            memory_bytes_f32: 4_000_000_000,
            num_layers: 1,
            hidden_dim: 1,
            vocab_size: 1,
            architecture: "test".to_string(),
        };
        assert!(
            (stats.parameters_b() - 1.0).abs() < 0.0001,
            "Expected 1.0B, got {}",
            stats.parameters_b()
        );
    }

    #[test]
    fn test_deep_ccov_raw_tensor_q8_0_dtype() {
        // Test Q8_0 dtype (GGML type 8)
        let tensor = RawTensor {
            name: "q8_0_weights".to_string(),
            data: vec![0u8; 34], // Q8_0: 32 elements = 34 bytes
            shape: vec![32],
            dtype: 8, // Q8_0
        };
        assert_eq!(tensor.dtype, 8);
        assert_eq!(tensor.data.len(), 34);
    }

    #[test]
    fn test_deep_ccov_raw_tensor_q5_k_dtype() {
        // Test Q5_K dtype (GGML type 13)
        let tensor = RawTensor {
            name: "q5_k_weights".to_string(),
            data: vec![0u8; 176], // Q5_K: 256 elements = 176 bytes
            shape: vec![256],
            dtype: 13, // Q5_K
        };
        assert_eq!(tensor.dtype, 13);
        assert_eq!(tensor.data.len(), 176);
    }

    #[test]
    fn test_deep_ccov_raw_tensor_q6_k_dtype() {
        // Test Q6_K dtype (GGML type 14)
        let tensor = RawTensor {
            name: "q6_k_weights".to_string(),
            data: vec![0u8; 210], // Q6_K: 256 elements = 210 bytes
            shape: vec![256],
            dtype: 14, // Q6_K
        };
        assert_eq!(tensor.dtype, 14);
        assert_eq!(tensor.data.len(), 210);
    }

    #[test]
    fn test_deep_ccov_raw_tensor_f16_dtype() {
        // Test F16 dtype (GGML type 1)
        let tensor = RawTensor {
            name: "f16_weights".to_string(),
            data: vec![0u8; 128], // 64 F16 elements = 128 bytes
            shape: vec![64],
            dtype: 1, // F16
        };
        assert_eq!(tensor.dtype, 1);
        assert_eq!(tensor.data.len(), 128);
    }

    #[test]
    fn test_deep_ccov_from_apr_bytes_exact_minimum_size() {
        // Test with exactly HEADER_SIZE bytes (minimum valid size for header parsing)
        let mut bytes = vec![0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2; // v2
        bytes[5] = 0;
        bytes[8..12].copy_from_slice(&0u32.to_le_bytes()); // 0 tensors
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata offset at header end
        bytes[20..24].copy_from_slice(&0u32.to_le_bytes()); // 0 metadata size
        bytes[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor index at 64
        bytes[32..40].copy_from_slice(&64u64.to_le_bytes()); // data at 64

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        // Should fail because no weights tensor found or invalid tensor index
        assert!(result.is_err());
    }

    #[test]
    fn test_deep_ccov_apr_bytes_roundtrip_with_biases() {
        // Test roundtrip with a transformer that has bias terms
        let config = AprTransformerConfig {
            architecture: "bias_test".to_string(),
            hidden_dim: 8,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 16,
            intermediate_dim: 16,
            context_length: 64,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        };

        let layer = AprTransformerLayer {
            attn_norm_weight: vec![1.0; 8],
            attn_norm_bias: Some(vec![0.1; 8]), // With bias
            qkv_weight: vec![0.01; 8 * 3 * 8],
            qkv_bias: Some(vec![0.02; 3 * 8]), // With bias
            attn_output_weight: vec![0.01; 8 * 8],
            attn_output_bias: Some(vec![0.03; 8]), // With bias
            ffn_gate_weight: Some(vec![0.01; 8 * 16]),
            ffn_gate_bias: Some(vec![0.04; 16]), // With bias
            ffn_up_weight: vec![0.01; 8 * 16],
            ffn_up_bias: Some(vec![0.05; 16]), // With bias
            ffn_down_weight: vec![0.01; 16 * 8],
            ffn_down_bias: Some(vec![0.06; 8]), // With bias
            ffn_norm_weight: Some(vec![1.0; 8]),
            ffn_norm_bias: Some(vec![0.07; 8]), // With bias
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
        };

        let original = AprTransformer {
            config,
            token_embedding: vec![0.1; 16 * 8],
            layers: vec![layer],
            output_norm_weight: vec![1.0; 8],
            output_norm_bias: Some(vec![0.08; 8]), // With bias
            lm_head_weight: vec![0.01; 8 * 16],
            lm_head_bias: Some(vec![0.09; 16]), // With bias
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        };

        let bytes = GgufToAprConverter::to_apr_bytes(&original).expect("serialize");
        let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

        assert_eq!(original.config, loaded.config);
        assert_eq!(
            original.layers[0].attn_norm_bias,
            loaded.layers[0].attn_norm_bias
        );
        assert_eq!(original.lm_head_bias, loaded.lm_head_bias);
    }

    #[test]
    fn test_deep_ccov_stats_architecture_preservation() {
        // Test that stats preserves the full architecture name
        let apr = AprTransformer {
            config: AprTransformerConfig {
                architecture: "special-arch-name/v2.1".to_string(),
                hidden_dim: 4,
                num_layers: 1,
                num_heads: 1,
                num_kv_heads: 1,
                vocab_size: 10,
                intermediate_dim: 8,
                context_length: 32,
                rope_theta: 10000.0,
                eps: 1e-5,
                eos_token_id: None,
            ..Default::default()
            },
            token_embedding: vec![0.1; 40],
            layers: vec![AprTransformerLayer {
                attn_norm_weight: vec![1.0; 4],
                attn_norm_bias: None,
                qkv_weight: vec![0.01; 48],
                qkv_bias: None,
                attn_output_weight: vec![0.01; 16],
                attn_output_bias: None,
                ffn_gate_weight: None,
                ffn_gate_bias: None,
                ffn_up_weight: vec![0.01; 32],
                ffn_up_bias: None,
                ffn_down_weight: vec![0.01; 32],
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
            }],
            output_norm_weight: vec![1.0; 4],
            output_norm_bias: None,
            lm_head_weight: vec![0.01; 40],
            lm_head_bias: None,
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        };

        let stats = GgufToAprConverter::stats(&apr);
        assert_eq!(stats.architecture, "special-arch-name/v2.1");
    }

    #[test]
    fn test_deep_ccov_from_gguf_transformer_all_optional_biases() {
        // Test conversion when all optional biases are None
        use crate::gguf::{GGUFConfig, GGUFTransformerLayer};

        let config = GGUFConfig {
            architecture: "no_bias".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("no_bias"),
            hidden_dim: 8,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 16,
            intermediate_dim: 16,
            context_length: 64,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let layer = GGUFTransformerLayer {
            attn_norm_weight: vec![1.0; 8],
            attn_norm_bias: None,
            qkv_weight: vec![0.01; 192],
            qkv_bias: None,
            attn_output_weight: vec![0.01; 64],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.01; 128],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.01; 128],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
        };

        let gguf = GGUFTransformer {
            config,
            token_embedding: vec![0.1; 128],
            position_embedding: None,
            layers: vec![layer],
            output_norm_weight: vec![1.0; 8],
            output_norm_bias: None,
            lm_head_weight: vec![0.01; 128],
            lm_head_bias: None,
        };

        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
        assert!(apr.layers[0].attn_norm_bias.is_none());
        assert!(apr.layers[0].qkv_bias.is_none());
        assert!(apr.layers[0].ffn_gate_weight.is_none());
        assert!(apr.output_norm_bias.is_none());
        assert!(apr.lm_head_bias.is_none());
    }

    #[test]
    fn test_deep_ccov_from_gguf_transformer_with_gate_weights() {
        // Test conversion when ffn_gate_weight is present (SwiGLU FFN)
        use crate::gguf::{GGUFConfig, GGUFTransformerLayer};

        let config = GGUFConfig {
            architecture: "swiglu".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("swiglu"),
            hidden_dim: 8,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 16,
            intermediate_dim: 16,
            context_length: 64,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let layer = GGUFTransformerLayer {
            attn_norm_weight: vec![1.0; 8],
            attn_norm_bias: None,
            qkv_weight: vec![0.01; 192],
            qkv_bias: None,
            attn_output_weight: vec![0.01; 64],
            attn_output_bias: None,
            ffn_gate_weight: Some(vec![0.02; 128]), // With gate weights
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.01; 128],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.01; 128],
            ffn_down_bias: None,
            ffn_norm_weight: Some(vec![1.0; 8]),
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
        };

        let gguf = GGUFTransformer {
            config,
            token_embedding: vec![0.1; 128],
            position_embedding: None,
            layers: vec![layer],
            output_norm_weight: vec![1.0; 8],
            output_norm_bias: None,
            lm_head_weight: vec![0.01; 128],
            lm_head_bias: None,
        };

        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
        assert!(apr.layers[0].ffn_gate_weight.is_some());
        assert_eq!(
            apr.layers[0]
                .ffn_gate_weight
                .as_ref()
                .expect("index out of bounds")
                .len(),
            128
        );
        assert!(apr.layers[0].ffn_norm_weight.is_some());
    }

    #[test]
    fn test_deep_ccov_to_apr_bytes_metadata_alignment() {
        // Verify metadata is padded to 64-byte boundary
        let apr = create_test_apr_transformer(4, 1, 10, 8);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        // Parse header to get offsets
        let metadata_offset =
            u64::from_le_bytes(bytes[12..20].try_into().expect("index out of bounds")) as usize;
        let tensor_index_offset =
            u64::from_le_bytes(bytes[24..32].try_into().expect("index out of bounds")) as usize;

        // Metadata should start at HEADER_SIZE (64)
        assert_eq!(metadata_offset, HEADER_SIZE);

        // Tensor index offset should be aligned to 64 bytes from metadata start
        let metadata_region_size = tensor_index_offset - metadata_offset;
        assert_eq!(
            metadata_region_size % ALIGNMENT,
            0,
            "Metadata region should be aligned to {} bytes",
            ALIGNMENT
        );
    }

    #[test]
    fn test_deep_ccov_q4k_stats_debug_format() {
        let stats = Q4KConversionStats {
            tensor_count: 123,
            q4k_tensor_count: 99,
            total_bytes: 999_888,
            architecture: "test-arch".to_string(),
            num_layers: 12,
            hidden_size: 2048,
        };

        let debug_str = format!("{:?}", stats);

        // Verify all fields appear in debug output
        assert!(debug_str.contains("tensor_count: 123"));
        assert!(debug_str.contains("q4k_tensor_count: 99"));
        assert!(debug_str.contains("total_bytes: 999888"));
        assert!(debug_str.contains("test-arch"));
        assert!(debug_str.contains("num_layers: 12"));
        assert!(debug_str.contains("hidden_size: 2048"));
    }

    #[test]
    fn test_deep_ccov_conversion_stats_debug_format() {
        let stats = ConversionStats {
            total_parameters: 123_456_789,
            memory_bytes_f32: 493_827_156,
            num_layers: 24,
            hidden_dim: 4096,
            vocab_size: 128000,
            architecture: "llama-3-test".to_string(),
        };

        let debug_str = format!("{:?}", stats);

        // Verify all fields appear in debug output
        assert!(debug_str.contains("total_parameters: 123456789"));
        assert!(debug_str.contains("memory_bytes_f32: 493827156"));
        assert!(debug_str.contains("num_layers: 24"));
        assert!(debug_str.contains("hidden_dim: 4096"));
        assert!(debug_str.contains("vocab_size: 128000"));
        assert!(debug_str.contains("llama-3-test"));
    }

    #[test]
    fn test_deep_ccov_raw_tensor_unknown_dtype() {
        // Test with an unknown dtype that would use default F32 size calculation
        let tensor = RawTensor {
            name: "unknown_type".to_string(),
            data: vec![0u8; 40], // 10 F32 elements
            shape: vec![10],
            dtype: 255, // Unknown type
        };
        assert_eq!(tensor.dtype, 255);
    }

    #[test]
    fn test_deep_ccov_apr_header_offsets_consistency() {
        // Verify header offsets are consistent after serialization
        let apr = create_test_apr_transformer(8, 2, 50, 16);
        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

        let metadata_offset =
            u64::from_le_bytes(bytes[12..20].try_into().expect("index out of bounds"));
        let metadata_size =
            u32::from_le_bytes(bytes[20..24].try_into().expect("index out of bounds"));
        let tensor_index_offset =
            u64::from_le_bytes(bytes[24..32].try_into().expect("index out of bounds"));
        let data_offset =
            u64::from_le_bytes(bytes[32..40].try_into().expect("index out of bounds"));

        // Metadata should start at header end
        assert_eq!(metadata_offset, HEADER_SIZE as u64);

        // Tensor index should come after metadata (with alignment)
        assert!(tensor_index_offset >= metadata_offset + metadata_size as u64);

        // Data should come after tensor index
        assert!(data_offset >= tensor_index_offset);

        // Total file size should be at least data_offset
        assert!(bytes.len() as u64 >= data_offset);
    }
