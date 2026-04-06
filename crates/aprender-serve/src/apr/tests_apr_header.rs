
    // =========================================================================
    // AprHeader Tests
    // =========================================================================

    #[test]
    fn test_apr_header_from_bytes_valid() {
        let mut data = Vec::new();
        data.extend_from_slice(&MAGIC); // APR\0
        data.extend_from_slice(&[2, 0]); // version 2.0
        data.extend_from_slice(&[0, 0]); // flags
        data.extend_from_slice(&5u32.to_le_bytes()); // tensor_count = 5
        data.extend_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data.extend_from_slice(&100u32.to_le_bytes()); // metadata_size
        data.extend_from_slice(&164u64.to_le_bytes()); // tensor_index_offset
        data.extend_from_slice(&500u64.to_le_bytes()); // data_offset
        data.extend_from_slice(&0u32.to_le_bytes()); // checksum
        data.extend_from_slice(&[0u8; 20]); // reserved

        let result = AprHeader::from_bytes(&data);
        assert!(result.is_ok());
        let header = result.expect("APR operation failed");
        assert_eq!(header.version.0, 2);
        assert_eq!(header.version.1, 0);
        assert_eq!(header.tensor_count, 5);
    }

    #[test]
    fn test_apr_header_from_bytes_wrong_magic() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF"); // Wrong magic
        data.extend_from_slice(&[0u8; 60]); // padding

        let result = AprHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_header_from_bytes_too_short() {
        let data = vec![0u8; 10]; // Too short
        let result = AprHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    // =========================================================================
    // AprMetadata Tests
    // =========================================================================

    #[test]
    fn test_apr_metadata_is_transformer_true() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: Some(4),
            num_heads: Some(8),
            vocab_size: Some(32000),
            ..Default::default()
        };
        assert!(meta.is_transformer());
    }

    #[test]
    fn test_apr_metadata_is_transformer_missing_hidden() {
        let meta = AprMetadata {
            hidden_size: None,
            num_layers: Some(4),
            num_heads: Some(8),
            vocab_size: Some(32000),
            ..Default::default()
        };
        assert!(!meta.is_transformer());
    }

    #[test]
    fn test_apr_metadata_is_transformer_missing_layers() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: None,
            num_heads: Some(8),
            vocab_size: Some(32000),
            ..Default::default()
        };
        assert!(!meta.is_transformer());
    }

    #[test]
    fn test_apr_metadata_default() {
        let meta = AprMetadata::default();
        assert!(meta.hidden_size.is_none());
        assert!(meta.num_layers.is_none());
        assert!(!meta.is_transformer());
    }

    // =========================================================================
    // AprV2Model Tests - Basic Operations
    // =========================================================================

    #[test]
    fn test_apr_v2_model_from_bytes_minimal() {
        let data = create_test_apr_model();
        let result = AprV2Model::from_bytes(data);
        assert!(result.is_ok());
        let model = result.expect("APR operation failed");
        // Helper creates 1 tensor
        assert_eq!(model.tensor_count(), 1);
    }

    #[test]
    fn test_apr_v2_model_tensor_names() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let names = model.tensor_names();
        // Helper creates tensor "test.weight"
        assert_eq!(names.len(), 1);
        assert!(names.contains(&"test.weight"));
    }

    #[test]
    fn test_apr_v2_model_metadata_default() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let meta = model.metadata();
        assert!(!meta.is_transformer());
    }

    #[test]
    fn test_apr_v2_model_get_tensor_not_found() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        assert!(model.get_tensor("nonexistent").is_none());
    }

    #[test]
    fn test_apr_v2_model_get_tensor_bytes_not_found() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let result = model.get_tensor_bytes("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_v2_model_get_tensor_f32_not_found() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let result = model.get_tensor_f32("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_v2_model_estimated_parameters() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        // Helper creates 1 tensor with shape [4,4] = 16 elements
        assert_eq!(model.estimated_parameters(), 16);
    }

    #[test]
    fn test_apr_v2_model_is_mmap_false() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        assert!(!model.is_mmap());
    }

    // =========================================================================
    // AprV2Model Tests - predict
    // =========================================================================

    #[test]
    fn test_apr_v2_model_predict_no_tensors() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let features = vec![1.0, 2.0, 3.0];
        let result = model.predict(&features);
        assert!(result.is_ok());
        // With no tensors, returns sum of features
        let output = result.expect("APR operation failed");
        assert_eq!(output.len(), 1);
        assert!((output[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_apr_v2_model_predict_empty_features() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let features: Vec<f32> = vec![];
        let result = model.predict(&features);
        assert!(result.is_ok());
        let output = result.expect("APR operation failed");
        assert_eq!(output[0], 0.0);
    }

    // =========================================================================
    // AprV2Model Tests - forward
    // =========================================================================

    #[test]
    fn test_apr_v2_model_forward_empty_tokens() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let result = model.forward(&[]);
        assert!(result.is_err()); // Empty tokens should fail
    }

    #[test]
    fn test_apr_v2_model_forward_not_transformer() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let result = model.forward(&[1, 2, 3]);
        // Should fail because metadata doesn't indicate transformer
        assert!(result.is_err());
    }

    // =========================================================================
    // decode_tokens Tests
    // =========================================================================

    #[test]
    fn test_decode_tokens_basic() {
        let vocab = vec!["hello".to_string(), "world".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1]);
        assert!(result.contains("hello"));
        assert!(result.contains("world"));
    }

    #[test]
    fn test_decode_tokens_empty_input() {
        let vocab = vec!["hello".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_decode_tokens_out_of_bounds() {
        let vocab = vec!["hello".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 5, 10]);
        // Should contain "hello" and [id] for invalid tokens
        assert!(result.contains("hello"));
        assert!(result.contains("[5]"));
        assert!(result.contains("[10]"));
    }

    #[test]
    fn test_decode_tokens_sentencepiece_prefix() {
        let vocab = vec!["▁hello".to_string(), "▁world".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1]);
        // Sentencepiece prefix should be converted to space
        assert!(result.contains("hello"));
        assert!(result.contains("world"));
    }

    #[test]
    fn test_decode_tokens_empty_vocab() {
        let vocab: Vec<String> = vec![];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
        // All tokens out of bounds, formatted as [id]
        assert!(result.contains("[0]"));
        assert!(result.contains("[1]"));
        assert!(result.contains("[2]"));
    }

    // =========================================================================
    // bpe_encode Tests
    // =========================================================================

    #[test]
    fn test_bpe_encode_empty_text() {
        let token_to_id: HashMap<String, u32> = HashMap::new();
        let merge_rules: Vec<(String, String)> = vec![];
        let result = bpe_encode("", &token_to_id, &merge_rules, &HashMap::new());
        assert!(result.is_empty());
    }

    #[test]
    fn test_bpe_encode_single_char() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        let merge_rules: Vec<(String, String)> = vec![];
        let result = bpe_encode("a", &token_to_id, &merge_rules, &HashMap::new());
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_unknown_chars() {
        let token_to_id: HashMap<String, u32> = HashMap::new();
        let merge_rules: Vec<(String, String)> = vec![];
        let result = bpe_encode("xyz", &token_to_id, &merge_rules, &HashMap::new());
        // Unknown chars return empty or default behavior
        assert!(result.is_empty());
    }

    #[test]
    fn test_bpe_encode_with_merge() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("h".to_string(), 0);
        token_to_id.insert("e".to_string(), 1);
        token_to_id.insert("he".to_string(), 2);
        let merge_rules = vec![("h".to_string(), "e".to_string())];
        let result = bpe_encode("he", &token_to_id, &merge_rules, &HashMap::new());
        // Should merge h+e -> he
        assert!(!result.is_empty());
    }

    // =========================================================================
    // BpeTokenizer Extended Tests
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_encode_whitespace() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert(" ".to_string(), 0);
        token_to_id.insert("a".to_string(), 1);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec![" ".to_string(), "a".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let encoded = tokenizer.encode(" a ");
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_decode_sentencepiece() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("▁hello".to_string(), 0);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["▁hello".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let decoded = tokenizer.decode(&[0]);
        assert!(decoded.contains("hello"));
    }

    #[test]
    fn test_bpe_tokenizer_decode_unknown_id() {
        let tokenizer = BpeTokenizer {
            token_to_id: HashMap::new(),
            id_to_token: vec!["a".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let decoded = tokenizer.decode(&[0, 100, 200]);
        // Should handle out of bounds gracefully
        assert!(decoded.contains("a") || decoded.contains("<unk>"));
    }

    // =========================================================================
    // dequantize_q4_k Tests
    // =========================================================================

    #[test]
    fn test_dequantize_q4_k_empty() {
        let result = crate::apr::dequantize_q4_k(&[], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q6_k_empty() {
        let result = crate::apr::dequantize_q6_k(&[], 0);
        assert!(result.is_empty());
    }

    // =========================================================================
    // dtype_to_ggml_qtype Tests
    // =========================================================================

    #[test]
    fn test_dtype_to_ggml_qtype_f32() {
        // F32 is not a quantized type, returns None
        assert_eq!(crate::apr::dtype_to_ggml_qtype("F32"), None);
    }

    #[test]
    fn test_dtype_to_ggml_qtype_f16() {
        // F16 is not a quantized type, returns None
        assert_eq!(crate::apr::dtype_to_ggml_qtype("F16"), None);
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q4_0() {
        // Q4_0 is qtype 2 in GGML
        let result = crate::apr::dtype_to_ggml_qtype("Q4_0");
        assert!(result.is_some());
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q8_0() {
        // Q8_0 is qtype 8 in GGML
        let result = crate::apr::dtype_to_ggml_qtype("Q8_0");
        assert!(result.is_some());
    }

    #[test]
    fn test_dtype_to_ggml_qtype_unknown() {
        assert_eq!(crate::apr::dtype_to_ggml_qtype("UNKNOWN"), None);
    }

    // =========================================================================
    // is_quantized_dtype Tests
    // =========================================================================

    #[test]
    fn test_is_quantized_dtype_f32() {
        assert!(!crate::apr::is_quantized_dtype("F32"));
    }

    #[test]
    fn test_is_quantized_dtype_f16() {
        assert!(!crate::apr::is_quantized_dtype("F16"));
    }

    #[test]
    fn test_is_quantized_dtype_q4_0() {
        assert!(crate::apr::is_quantized_dtype("Q4_0"));
    }

    #[test]
    fn test_is_quantized_dtype_q8_0() {
        assert!(crate::apr::is_quantized_dtype("Q8_0"));
    }

    #[test]
    fn test_is_quantized_dtype_q4_k() {
        assert!(crate::apr::is_quantized_dtype("Q4_K"));
    }

    #[test]
    fn test_is_quantized_dtype_q6_k() {
        assert!(crate::apr::is_quantized_dtype("Q6_K"));
    }

    // =========================================================================
    // byte_to_bpe_char Tests
    // =========================================================================

    #[test]
    fn test_byte_to_bpe_char_ascii() {
        // ASCII printable range
        assert_eq!(crate::apr::byte_to_bpe_char(b'a'), "a");
        assert_eq!(crate::apr::byte_to_bpe_char(b'z'), "z");
        assert_eq!(crate::apr::byte_to_bpe_char(b'0'), "0");
    }
