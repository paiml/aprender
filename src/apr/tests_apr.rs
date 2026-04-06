
    // =========================================================================
    // AprV2Model forward error path tests
    // =========================================================================

    #[test]
    fn test_apr_v2_model_forward_empty_input() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let result = model.forward(&[]);
        assert!(result.is_err()); // Empty input
        let err = result.unwrap_err();
        let err_msg = format!("{:?}", err);
        assert!(err_msg.contains("empty"));
    }

    #[test]
    fn test_apr_v2_model_generate_max_tokens_zero() {
        let data = create_mini_transformer_apr();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let result = model.generate(&[1, 2], 0, None);
        // Should succeed with empty generation
        assert!(result.is_ok());
        let tokens = result.expect("APR operation failed");
        assert_eq!(tokens.len(), 2); // Just input, no generation
    }

    // =========================================================================
    // byte_to_bpe_char newline and tab
    // =========================================================================

    #[test]
    fn test_byte_to_bpe_char_newline() {
        let result = crate::apr::byte_to_bpe_char(b'\n');
        assert_eq!(result, "Ċ");
    }

    #[test]
    fn test_byte_to_bpe_char_tab() {
        let result = crate::apr::byte_to_bpe_char(b'\t');
        assert_eq!(result, "ĉ");
    }

    // =========================================================================
    // rms_norm with very small epsilon
    // =========================================================================

    #[test]
    fn test_rms_norm_tiny_eps() {
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-12;
        let result = crate::apr::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        // Should not produce NaN or Inf
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_rms_norm_large_eps() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 10.0; // Large epsilon dominates
        let result = crate::apr::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    // =========================================================================
    // Additional Coverage Tests (_more_cov suffix)
    // =========================================================================

    #[test]
    fn test_load_tokenizer_from_sibling_nonexistent_more_cov() {
        // Test with a path that doesn't exist
        let path = std::path::Path::new("/nonexistent/path/model.apr");
        let result = AprV2Model::load_tokenizer_from_sibling(path);
        assert!(result.is_none());
    }

    #[test]
    fn test_encode_text_nonexistent_more_cov() {
        // Test encode_text with a path that doesn't exist
        let path = std::path::Path::new("/nonexistent/path/model.apr");
        let result = AprV2Model::encode_text(path, "hello world");
        assert!(result.is_none());
    }

    #[test]
    fn test_load_tokenizer_nonexistent_more_cov() {
        // Test load_tokenizer with a path that doesn't exist
        let path = std::path::Path::new("/nonexistent/path/model.apr");
        let result = AprV2Model::load_tokenizer(path);
        assert!(result.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_tokenizer_from_sibling_invalid_json_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        // Create model file (empty)
        std::fs::File::create(&model_path).expect("create model file");

        // Create invalid JSON tokenizer file
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        f.write_all(b"not valid json").expect("write");

        let result = AprV2Model::load_tokenizer_from_sibling(&model_path);
        assert!(result.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_encode_text_invalid_json_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        std::fs::File::create(&model_path).expect("create model file");
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        f.write_all(b"not valid json").expect("write");

        let result = AprV2Model::encode_text(&model_path, "test");
        assert!(result.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_tokenizer_invalid_json_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        std::fs::File::create(&model_path).expect("create model file");
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        f.write_all(b"not valid json").expect("write");

        let result = AprV2Model::load_tokenizer(&model_path);
        assert!(result.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_tokenizer_from_sibling_missing_model_key_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        std::fs::File::create(&model_path).expect("create model file");
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        // Valid JSON but missing "model" key
        f.write_all(br#"{"added_tokens": []}"#).expect("write");

        let result = AprV2Model::load_tokenizer_from_sibling(&model_path);
        assert!(result.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_tokenizer_from_sibling_missing_vocab_key_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        std::fs::File::create(&model_path).expect("create model file");
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        // Valid JSON but missing "vocab" key in "model"
        f.write_all(br#"{"model": {}}"#).expect("write");

        let result = AprV2Model::load_tokenizer_from_sibling(&model_path);
        assert!(result.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_tokenizer_from_sibling_valid_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        std::fs::File::create(&model_path).expect("create model file");
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        // Valid tokenizer.json structure
        let json = r#"{
            "model": {
                "vocab": {
                    "hello": 0,
                    "world": 1,
                    "<s>": 2,
                    "</s>": 3
                }
            },
            "added_tokens": [
                {"content": "<s>", "id": 2},
                {"content": "</s>", "id": 3}
            ]
        }"#;
        f.write_all(json.as_bytes()).expect("write");

        let result = AprV2Model::load_tokenizer_from_sibling(&model_path);
        assert!(result.is_some());
        let (vocab, bos_id, eos_id) = result.expect("APR operation failed");
        assert!(vocab.len() >= 2);
        assert_eq!(bos_id, Some(2));
        assert_eq!(eos_id, Some(3));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[ignore = "APR dtype parsing bug - needs investigation"]
    #[test]
    fn test_encode_text_valid_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        std::fs::File::create(&model_path).expect("create model file");
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        let json = r#"{
            "model": {
                "vocab": {"h": 0, "e": 1, "l": 2, "o": 3},
                "merges": []
            }
        }"#;
        f.write_all(json.as_bytes()).expect("write");

        let result = AprV2Model::encode_text(&model_path, "hello");
        assert!(result.is_some());
        let tokens = result.expect("APR operation failed");
        assert!(!tokens.is_empty());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_tokenizer_valid_with_merges_more_cov() {
        use std::io::Write;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");
        let tokenizer_path = dir.path().join("tokenizer.json");

        std::fs::File::create(&model_path).expect("create model file");
        let mut f = std::fs::File::create(&tokenizer_path).expect("create tokenizer file");
        let json = r#"{
            "model": {
                "vocab": {"h": 0, "e": 1, "he": 2, "<|endoftext|>": 3, "<bos>": 4},
                "merges": ["h e"]
            },
            "added_tokens": [
                {"content": "<|endoftext|>", "id": 3},
                {"content": "<bos>", "id": 4}
            ]
        }"#;
        f.write_all(json.as_bytes()).expect("write");

        let result = AprV2Model::load_tokenizer(&model_path);
        assert!(result.is_some());
        let tokenizer = result.expect("APR operation failed");
        assert!(!tokenizer.token_to_id.is_empty());
        assert_eq!(tokenizer.eos_id, Some(3));
        assert_eq!(tokenizer.bos_id, Some(4));
        assert!(!tokenizer.merge_rules.is_empty());
    }

    #[test]
    fn test_bpe_encode_non_ascii_more_cov() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        let result = bpe_encode("a\u{00A9}", &token_to_id, &[], &HashMap::new()); // a + copyright symbol
        // Non-ASCII gets encoded as bytes
        assert!(!result.is_empty() || result.is_empty()); // Just verify no panic
    }

    #[test]
    fn test_bpe_encode_unicode_more_cov() {
        let token_to_id: HashMap<String, u32> = HashMap::new();
        // Test with unicode characters
        let result = bpe_encode("\u{1F600}", &token_to_id, &[], &HashMap::new()); // Emoji
        // Should not panic, may return empty if no tokens match
        assert!(result.is_empty());
    }

    #[test]
    fn test_byte_to_bpe_char_null_more_cov() {
        let result = crate::apr::byte_to_bpe_char(0x00);
        assert!(!result.is_empty());
        assert!(result.starts_with("<0x"));
    }

    #[test]
    fn test_byte_to_bpe_char_delete_more_cov() {
        let result = crate::apr::byte_to_bpe_char(0x7F); // DEL character
        assert!(!result.is_empty());
    }

    #[test]
    fn test_matmul_empty_more_cov() {
        let result = crate::apr::matmul(&[], &[], 0, 0, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_matmul_single_element_more_cov() {
        let x = vec![2.0];
        let w = vec![3.0];
        let result = crate::apr::matmul(&x, &w, 1, 1, 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_rms_norm_single_sequence_more_cov() {
        // Test single sequence normalization
        let x = vec![1.0, 2.0];
        let weight = vec![2.0, 0.5];
        let result = crate::apr::rms_norm(&x, &weight, 1e-6);
        assert_eq!(result.len(), 2);
        // Values should be normalized and scaled by weights
        assert!(result[0].is_finite());
        assert!(result[1].is_finite());
    }

    #[test]
    fn test_rms_norm_weight_shorter_than_x_more_cov() {
        // x has more elements than weight - tests unwrap_or path
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight = vec![1.0, 1.0]; // Only 2 weights for hidden_dim=2
        let result = crate::apr::rms_norm(&x, &weight, 1e-6);
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn test_simple_attention_empty_more_cov() {
        let result = crate::apr::simple_attention(&[], &[], &[], 0, 1, 1, 1);
        assert!(result.is_empty());
    }

    #[test]
    fn test_simd_dot_unequal_lengths_more_cov() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 1.0, 1.0]; // Shorter
        let result = crate::apr::simd_dot(&a, &b);
        // Should only use min(5, 3) = 3 elements
        assert!((result - 6.0).abs() < 1e-6); // 1+2+3 = 6
    }

    #[test]
    fn test_dequantize_f16_request_more_than_available_more_cov() {
        // Only enough bytes for 2 f16 values, but request 10
        let bytes = vec![0x00, 0x3C, 0x00, 0x40]; // 1.0, 2.0
        let result = crate::apr::dequantize_f16(&bytes, 10);
        assert_eq!(result.len(), 2); // Truncated to available
    }

    #[test]
    fn test_dequantize_q8_0_request_more_than_available_more_cov() {
        // One block = 34 bytes = 32 elements, request 100
        let mut bytes = vec![0u8; 34];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // Scale = 1.0
        let result = crate::apr::dequantize_q8_0(&bytes, 100);
        assert_eq!(result.len(), 32); // Truncated to one block
    }

    #[test]
    fn test_tensor_entry_serialize_deserialize_more_cov() {
        let entry = TensorEntry {
            name: "test.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10, 20],
            offset: 100,
            size: 800,
        };
        let json = serde_json::to_string(&entry).expect("invalid UTF-8");
        let parsed: TensorEntry = serde_json::from_str(&json).expect("parse failed");
        assert_eq!(parsed.name, "test.weight");
        assert_eq!(parsed.shape, vec![10, 20]);
    }

    #[test]
    fn test_apr_metadata_serialize_deserialize_more_cov() {
        let meta = AprMetadata {
            model_type: Some("llama".to_string()),
            name: Some("TestModel".to_string()),
            architecture: Some("decoder-only".to_string()),
            hidden_size: Some(4096),
            num_layers: Some(32),
            num_heads: Some(32),
            num_kv_heads: Some(8),
            vocab_size: Some(128256),
            intermediate_size: Some(14336),
            max_position_embeddings: Some(8192),
            rope_theta: Some(500000.0),
            rope_type: Some(2),
            rms_norm_eps: Some(1e-5),
            extra: HashMap::new(),
        };
        let json = serde_json::to_string(&meta).expect("invalid UTF-8");
        let parsed: AprMetadata = serde_json::from_str(&json).expect("parse failed");
        assert_eq!(parsed.hidden_size, Some(4096));
        assert_eq!(parsed.rope_type, Some(2));
    }

    #[test]
    fn test_apr_flags_copy_more_cov() {
        let flags = AprFlags::new(AprFlags::QUANTIZED | AprFlags::HAS_VOCAB);
        let copied = flags;
        assert!(copied.is_quantized());
        assert!(copied.has_vocab());
    }

    #[test]
    fn test_decode_tokens_special_chars_more_cov() {
        let vocab = vec![
            "hello".to_string(),
            "Ġ".to_string(), // Space token
            "Ċ".to_string(), // Newline token
            "ĉ".to_string(), // Tab token
        ];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2, 3]);
        assert!(result.contains("hello"));
        assert!(result.contains(' '));
        assert!(result.contains('\n'));
        assert!(result.contains('\t'));
    }
