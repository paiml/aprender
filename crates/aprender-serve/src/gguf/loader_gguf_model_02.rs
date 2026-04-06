
    #[test]
    fn test_gguf_model_decode_basic() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .add_string_array("tokenizer.ggml.tokens", &["<unk>", "hello", " world"])
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        let decoded = model.decode(&[1, 2]);
        assert!(decoded.contains("hello"));
        assert!(decoded.contains("world"));
    }

    #[test]
    fn test_gguf_model_decode_no_vocab_fallback() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        // Without vocab, decode uses ASCII fallback
        let decoded = model.decode(&[72, 73]); // H, I
        assert_eq!(decoded, "HI");
    }

    #[test]
    fn test_gguf_model_encode_basic() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .add_string("tokenizer.ggml.model", "llama")
            .add_string_array(
                "tokenizer.ggml.tokens",
                &["<unk>", "hell", "o", "▁world", "▁"],
            )
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        let tokens = model.encode("hello world");
        assert!(tokens.is_some());
        let tokens = tokens.expect("tokens");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_gguf_model_encode_no_vocab() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        let tokens = model.encode("hello");
        assert!(tokens.is_none());
    }

    #[test]
    fn test_gguf_model_all_metadata_types() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .add_u8("test.u8", 42)
            .add_i8("test.i8", -10)
            .add_u16("test.u16", 1234)
            .add_i16("test.i16", -5678)
            .add_u32("test.u32", 100_000)
            .add_i32("test.i32", -200_000)
            .add_f32("test.f32", 3.14)
            .add_bool("test.bool", true)
            .add_string("test.string", "hello")
            .add_u64("test.u64", 999_999_999)
            .add_i64("test.i64", -888_888_888)
            .add_f64("test.f64", 2.71828)
            .build();

        let model = GGUFModel::from_bytes(&data);
        assert!(
            model.is_ok(),
            "All metadata types should parse: {:?}",
            model.err()
        );

        let model = model.expect("model");
        assert!(!model.metadata.is_empty());
    }

    #[test]
    fn test_gguf_model_get_tensor_f32_q4_k() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).expect("model");

        // Q4_K tensor (blk.0.attn_q.weight)
        let result = model.get_tensor_f32("blk.0.attn_q.weight", &data);
        assert!(
            result.is_ok(),
            "Q4_K tensor extraction failed: {:?}",
            result.err()
        );
        let values = result.expect("values");
        assert!(!values.is_empty());
    }

    #[test]
    fn test_gguf_model_get_tensor_f32_various_qtypes() {
        use crate::gguf::test_factory::*;

        // Test F16 tensor
        let f16_data = create_f16_data(64);
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .add_f16_tensor("test_f16", &[64], &f16_data)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        let result = model.get_tensor_f32("test_f16", &data);
        assert!(result.is_ok(), "F16 tensor failed: {:?}", result.err());

        // Test Q8_0 tensor
        let q8_data = create_q8_0_data(64);
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .add_q8_0_tensor("test_q8", &[64], &q8_data)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        let result = model.get_tensor_f32("test_q8", &data);
        assert!(result.is_ok(), "Q8_0 tensor failed: {:?}", result.err());

        // Test Q6_K tensor
        let q6k_data = create_q6_k_data(256);
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .add_q6_k_tensor("test_q6k", &[256], &q6k_data)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        let result = model.get_tensor_f32("test_q6k", &data);
        assert!(result.is_ok(), "Q6_K tensor failed: {:?}", result.err());

        // Test Q2_K tensor
        let q2k_data = create_q2_k_data(256);
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .add_q2_k_tensor("test_q2k", &[256], &q2k_data)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        let result = model.get_tensor_f32("test_q2k", &data);
        assert!(result.is_ok(), "Q2_K tensor failed: {:?}", result.err());
    }

    #[test]
    fn test_gguf_model_get_tensor_f32_q4_1() {
        use crate::gguf::test_factory::*;
        let q4_1_data = create_q4_1_data(64);
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .add_q4_1_tensor("test_q4_1", &[64], &q4_1_data)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        let result = model.get_tensor_f32("test_q4_1", &data);
        assert!(result.is_ok(), "Q4_1 tensor failed: {:?}", result.err());
    }

    #[test]
    fn test_gguf_model_get_tensor_f32_q5_0() {
        use crate::gguf::test_factory::*;
        let q5_0_data = create_q5_0_data(64);
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .add_q5_0_tensor("test_q5_0", &[64], &q5_0_data)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        let result = model.get_tensor_f32("test_q5_0", &data);
        assert!(result.is_ok(), "Q5_0 tensor failed: {:?}", result.err());
    }

    #[test]
    fn test_gguf_model_get_tensor_f32_q5_1() {
        use crate::gguf::test_factory::*;
        let q5_1_data = create_q5_1_data(64);
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .add_q5_1_tensor("test_q5_1", &[64], &q5_1_data)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        let result = model.get_tensor_f32("test_q5_1", &data);
        assert!(result.is_ok(), "Q5_1 tensor failed: {:?}", result.err());
    }

    #[test]
    fn test_gguf_model_get_tensor_f32_q5_k() {
        use crate::gguf::test_factory::*;
        let q5_k_data = create_q5_k_data(256);
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .add_q5_k_tensor("test_q5k", &[256], &q5_k_data)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        let result = model.get_tensor_f32("test_q5k", &data);
        assert!(result.is_ok(), "Q5_K tensor failed: {:?}", result.err());
    }

    #[test]
    fn test_gguf_builder_default() {
        use crate::gguf::test_factory::GGUFBuilder;
        let builder = GGUFBuilder::default();
        let data = builder.build();
        // Should parse (0 tensors, 0 metadata)
        let model = GGUFModel::from_bytes(&data);
        assert!(model.is_ok());
        let model = model.expect("model");
        assert!(model.tensors.is_empty());
        assert!(model.metadata.is_empty());
    }

    #[test]
    fn test_gguf_model_architecture_none() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new().build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        assert!(model.architecture().is_none());
    }

    #[test]
    fn test_gguf_model_embedding_dim_no_arch() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new().build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        assert!(model.embedding_dim().is_none());
    }

    #[test]
    fn test_gguf_model_num_layers_no_arch() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new().build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        assert!(model.num_layers().is_none());
    }

    #[test]
    fn test_gguf_model_num_heads_no_arch() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new().build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        assert!(model.num_heads().is_none());
    }

    #[test]
    fn test_gguf_model_rope_type_with_yarn_metadata() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .add_string("llama.rope.scaling.type", "yarn")
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        assert_eq!(model.rope_type(), Some(2)); // yarn -> NEOX
    }

    #[test]
    fn test_gguf_model_rope_type_with_none_metadata() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .add_string("llama.rope.scaling.type", "none")
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        assert_eq!(model.rope_type(), Some(0)); // none -> NORM
    }

    #[test]
    fn test_gguf_model_rope_type_with_linear_metadata() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .add_string("llama.rope.scaling.type", "linear")
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        assert_eq!(model.rope_type(), Some(0)); // linear -> NORM
    }

    #[test]
    fn test_gguf_model_decode_byte_tokens() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .add_string_array("tokenizer.ggml.tokens", &["<0x48>", "<0x69>"])
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        let decoded = model.decode(&[0, 1]);
        assert_eq!(decoded, "Hi");
    }

    // ============================================================================
    // T-COV-95 Phase 52: Additional GGUF loader coverage tests
    // Binary parsing, value type dispatch, error paths, metadata helpers
    // ============================================================================

    #[test]
    fn test_gguf_read_string_empty() {
        // A GGUF with a zero-length metadata key (empty string key)
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

        // Key: empty string (length 0)
        data.extend_from_slice(&0u64.to_le_bytes());
        // Value type: u32 (type 4)
        data.extend_from_slice(&4u32.to_le_bytes());
        // Value: 42u32
        data.extend_from_slice(&42u32.to_le_bytes());

        let model = GGUFModel::from_bytes(&data);
        assert!(
            model.is_ok(),
            "Empty string key should parse: {:?}",
            model.err()
        );
        let model = model.expect("model");
        assert!(model.metadata.contains_key(""));
    }

    #[test]
    fn test_gguf_read_string_truncated() {
        // Data too short to read string length
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

        // Only 4 bytes for the string length (need 8)
        data.extend_from_slice(&[0u8; 4]);

        let result = GGUFModel::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_read_string_length_too_large() {
        // String length says 1000 but data only has a few bytes
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

        // Key length = 1000, but only a few bytes follow
        data.extend_from_slice(&1000u64.to_le_bytes());
        data.extend_from_slice(b"short");

        let result = GGUFModel::from_bytes(&data);
        assert!(result.is_err());
    }
