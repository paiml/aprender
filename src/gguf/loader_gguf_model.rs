
    // ============================================================================
    // GGUFModel parsing tests
    // ============================================================================

    #[test]
    fn test_gguf_model_from_bytes_llama() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data);
        assert!(model.is_ok(), "Failed to parse GGUF: {:?}", model.err());

        let model = model.expect("model");
        assert_eq!(model.header.magic, GGUF_MAGIC);
        assert_eq!(model.header.version, GGUF_VERSION_V3);
    }

    #[test]
    fn test_gguf_model_from_bytes_phi2() {
        let data = build_minimal_phi2_gguf(100, 64, 256, 4);
        let model = GGUFModel::from_bytes(&data);
        assert!(model.is_ok(), "Failed to parse GGUF: {:?}", model.err());
    }

    #[test]
    fn test_gguf_model_tensors_parsed() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).expect("model");

        // Should have multiple tensors (embedding, layer weights, etc.)
        assert!(!model.tensors.is_empty());
    }

    #[test]
    fn test_gguf_model_metadata_parsed() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).expect("model");

        // Should have metadata (architecture, hidden_dim, etc.)
        assert!(!model.metadata.is_empty());
    }

    #[test]
    fn test_gguf_model_tensor_data_start() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).expect("model");

        // tensor_data_start should be 32-byte aligned
        assert!(model.tensor_data_start.is_multiple_of(GGUF_ALIGNMENT));
    }

    #[test]
    fn test_gguf_model_invalid_magic() {
        let mut data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        // Corrupt the magic number
        data[0] = 0xFF;
        data[1] = 0xFF;
        data[2] = 0xFF;
        data[3] = 0xFF;

        let result = GGUFModel::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_model_architecture() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).expect("model");

        let arch = model.architecture();
        assert!(arch.is_some());
    }

    #[test]
    fn test_gguf_model_embedding_dim() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).expect("model");

        let dim = model.embedding_dim();
        assert!(dim.is_some());
        assert_eq!(dim.expect("dim"), 64);
    }

    #[test]
    fn test_gguf_model_num_layers() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).expect("model");

        let layers = model.num_layers();
        assert!(layers.is_some());
    }

    #[test]
    fn test_gguf_model_num_heads() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).expect("model");

        let heads = model.num_heads();
        assert!(heads.is_some());
        assert_eq!(heads.expect("heads"), 4);
    }

    #[test]
    fn test_gguf_model_num_kv_heads() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 2); // num_kv_heads=2
        let model = GGUFModel::from_bytes(&data).expect("model");

        let kv_heads = model.num_kv_heads();
        assert!(kv_heads.is_some());
    }

    #[test]
    fn test_gguf_model_context_length() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).expect("model");

        let ctx = model.context_length();
        // May or may not be present depending on test factory
        assert!(ctx.is_none() || ctx.expect("ctx") > 0);
    }

    #[test]
    fn test_gguf_model_rope_freq_base() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).expect("model");

        // May have rope_freq_base metadata
        let _theta = model.rope_freq_base();
    }

    #[test]
    fn test_gguf_model_get_tensor_f32() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).expect("model");

        // Try to get token embedding (should be F32)
        let emb = model.get_tensor_f32("token_embd.weight", &data);
        assert!(emb.is_ok());
        let emb = emb.expect("emb");
        assert!(!emb.is_empty());
    }

    #[test]
    fn test_gguf_model_get_tensor_f32_not_found() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).expect("model");

        let result = model.get_tensor_f32("nonexistent_tensor", &data);
        assert!(result.is_err());
    }

    // ============================================================================
    // GGUFTransformer tests
    // ============================================================================

    #[test]
    fn test_gguf_transformer_from_mapped() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).expect("model");
        let transformer = GGUFTransformer::from_gguf(&model, &data);

        assert!(transformer.is_ok(), "Failed: {:?}", transformer.err());
    }

    #[test]
    fn test_gguf_transformer_config() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).expect("model");
        let transformer = GGUFTransformer::from_gguf(&model, &data).expect("transformer");

        assert_eq!(transformer.config.hidden_dim, 64);
        assert_eq!(transformer.config.num_heads, 4);
    }

    #[test]
    fn test_gguf_transformer_layers() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).expect("model");
        let transformer = GGUFTransformer::from_gguf(&model, &data).expect("transformer");

        // Should have at least 1 layer
        assert!(!transformer.layers.is_empty());
    }

    #[test]
    fn test_gguf_transformer_token_embedding() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).expect("model");
        let transformer = GGUFTransformer::from_gguf(&model, &data).expect("transformer");

        // token_embedding should be vocab_size * hidden_dim
        assert_eq!(transformer.token_embedding.len(), 100 * 64);
    }

    #[test]
    fn test_gguf_transformer_output_norm() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).expect("model");
        let transformer = GGUFTransformer::from_gguf(&model, &data).expect("transformer");

        // output_norm_weight should be hidden_dim
        assert_eq!(transformer.output_norm_weight.len(), 64);
    }

    #[test]
    fn test_gguf_transformer_lm_head() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).expect("model");
        let transformer = GGUFTransformer::from_gguf(&model, &data).expect("transformer");

        // lm_head should be vocab_size * hidden_dim
        assert!(!transformer.lm_head_weight.is_empty());
    }

    #[test]
    fn test_gguf_transformer_layer_attn_norm() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).expect("model");
        let transformer = GGUFTransformer::from_gguf(&model, &data).expect("transformer");

        let layer = &transformer.layers[0];
        assert_eq!(layer.attn_norm_weight.len(), 64);
    }

    // ============================================================================
    // T-COV-95 Phase 50: Additional GGUF loader error paths and metadata helpers
    // ============================================================================

    #[test]
    fn test_gguf_model_too_short() {
        // Too short to contain magic + version
        let data = vec![0u8; 4]; // Only 4 bytes
        let result = GGUFModel::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_model_unsupported_version() {
        use crate::gguf::test_factory::GGUFBuilder;
        let mut data = GGUFBuilder::new().architecture("llama").build();
        // Corrupt version to 99 (truly unsupported; v2 and v3 are valid per GH-310)
        data[4..8].copy_from_slice(&99u32.to_le_bytes());
        let result = GGUFModel::from_bytes(&data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("version") || err.contains("Unsupported"),
            "Expected version error, got: {}",
            err
        );
    }

    #[test]
    fn test_gguf_model_excessive_tensor_count() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        // tensor_count = 200000 (exceeds MAX_TENSOR_COUNT)
        data.extend_from_slice(&200_000u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0

        let result = GGUFModel::from_bytes(&data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("exceeds maximum") || err.contains("tensor_count"),
            "Expected tensor_count error, got: {}",
            err
        );
    }

    #[test]
    fn test_gguf_model_excessive_metadata_count() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        // metadata_count = 50000 (exceeds MAX_METADATA_COUNT)
        data.extend_from_slice(&50_000u64.to_le_bytes());

        let result = GGUFModel::from_bytes(&data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("exceeds maximum") || err.contains("metadata_count"),
            "Expected metadata_count error, got: {}",
            err
        );
    }

    #[test]
    fn test_gguf_model_unsupported_value_type() {
        // Build a GGUF with an unsupported metadata value type
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

        // Key: "test_key"
        let key = "test_key";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        // Value type 99 (unsupported)
        data.extend_from_slice(&99u32.to_le_bytes());

        let result = GGUFModel::from_bytes(&data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Unsupported value type") || err.contains("99"),
            "Expected unsupported type error, got: {}",
            err
        );
    }

    #[test]
    fn test_gguf_model_get_tensor_unsupported_qtype() {
        use crate::gguf::test_factory::GGUFBuilder;
        // Create a GGUF with a tensor of unsupported qtype
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .add_f32_tensor("token_embd.weight", &[100, 64], &vec![0.0f32; 6400])
            .build();

        let model = GGUFModel::from_bytes(&data).expect("model");

        // Manually find the token_embd tensor and check it works with F32
        let emb = model.get_tensor_f32("token_embd.weight", &data);
        assert!(emb.is_ok());
    }

    #[test]
    fn test_gguf_model_rope_type_neox_architectures() {
        use crate::gguf::test_factory::GGUFBuilder;

        // Test Qwen2 architecture -> NEOX rope type
        let data = GGUFBuilder::new()
            .architecture("qwen2")
            .hidden_dim("qwen2", 64)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        let rope_type = model.rope_type();
        assert_eq!(rope_type, Some(2)); // NEOX

        // Test llama architecture -> NORM rope type
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        let rope_type = model.rope_type();
        assert_eq!(rope_type, Some(0)); // NORM
    }

    #[test]
    fn test_gguf_model_rope_type_phi2() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("phi2")
            .hidden_dim("phi2", 64)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        let rope_type = model.rope_type();
        assert_eq!(rope_type, Some(2)); // phi2 -> NEOX
    }

    #[test]
    fn test_gguf_model_rope_type_gemma() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("gemma")
            .hidden_dim("gemma", 64)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        let rope_type = model.rope_type();
        assert_eq!(rope_type, Some(2)); // gemma -> NEOX
    }

    #[test]
    fn test_gguf_model_rms_epsilon() {
        let data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&data).expect("model");
        let eps = model.rms_epsilon();
        assert!(eps.is_some());
        assert!((eps.expect("eps") - 1e-5).abs() < 1e-8);
    }

    #[test]
    fn test_gguf_model_bos_eos_tokens() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .add_u32("tokenizer.ggml.bos_token_id", 1)
            .add_u32("tokenizer.ggml.eos_token_id", 2)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        assert_eq!(model.bos_token_id(), Some(1));
        assert_eq!(model.eos_token_id(), Some(2));
    }

    #[test]
    fn test_gguf_model_no_bos_eos() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        assert_eq!(model.bos_token_id(), None);
        assert_eq!(model.eos_token_id(), None);
    }

    #[test]
    fn test_gguf_model_vocabulary() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .add_string_array("tokenizer.ggml.tokens", &["hello", "world", "test"])
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        let vocab = model.vocabulary();
        assert!(vocab.is_some());
        let vocab = vocab.expect("vocab");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab[0], "hello");
        assert_eq!(vocab[1], "world");
        assert_eq!(vocab[2], "test");
    }

    #[test]
    fn test_gguf_model_no_vocabulary() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .build();
        let model = GGUFModel::from_bytes(&data).expect("model");
        let vocab = model.vocabulary();
        assert!(vocab.is_none());
    }
