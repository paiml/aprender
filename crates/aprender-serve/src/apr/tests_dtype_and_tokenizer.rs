
    // =========================================================================
    // Helper: Create binary tensor entry for tests
    // =========================================================================

    fn create_binary_tensor_entry(
        name: &str,
        dtype: u8,
        shape: &[u64],
        offset: u64,
        size: u64,
    ) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        data.push(dtype);
        data.push(shape.len() as u8);
        for &dim in shape {
            data.extend_from_slice(&dim.to_le_bytes());
        }
        data.extend_from_slice(&offset.to_le_bytes());
        data.extend_from_slice(&size.to_le_bytes());
        data
    }

    // =========================================================================
    // dtype_to_ggml_qtype tests
    // =========================================================================

    #[test]
    fn test_dtype_to_ggml_qtype_q4_k_lowercase() {
        assert_eq!(dtype_to_ggml_qtype("q4_k"), Some(12));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q5_k_uppercase() {
        assert_eq!(dtype_to_ggml_qtype("Q5_K"), Some(13));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q5_k_lowercase() {
        assert_eq!(dtype_to_ggml_qtype("q5_k"), Some(13));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q6_k_uppercase() {
        assert_eq!(dtype_to_ggml_qtype("Q6_K"), Some(14));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q6_k_lowercase() {
        assert_eq!(dtype_to_ggml_qtype("q6_k"), Some(14));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q8_0_uppercase() {
        assert_eq!(dtype_to_ggml_qtype("Q8_0"), Some(8));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q8_0_lowercase() {
        assert_eq!(dtype_to_ggml_qtype("q8_0"), Some(8));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q4_0_uppercase() {
        assert_eq!(dtype_to_ggml_qtype("Q4_0"), Some(2));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q4_0_lowercase() {
        assert_eq!(dtype_to_ggml_qtype("q4_0"), Some(2));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q4_1() {
        assert_eq!(dtype_to_ggml_qtype("Q4_1"), Some(3));
        assert_eq!(dtype_to_ggml_qtype("q4_1"), Some(3));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_q5_0() {
        assert_eq!(dtype_to_ggml_qtype("Q5_0"), Some(6));
        assert_eq!(dtype_to_ggml_qtype("q5_0"), Some(6));
    }

    #[test]
    fn test_dtype_to_ggml_qtype_f32_returns_none() {
        assert_eq!(dtype_to_ggml_qtype("F32"), None);
        assert_eq!(dtype_to_ggml_qtype("f32"), None);
    }

    #[test]
    fn test_dtype_to_ggml_qtype_f16_returns_none() {
        assert_eq!(dtype_to_ggml_qtype("F16"), None);
        assert_eq!(dtype_to_ggml_qtype("f16"), None);
    }

    #[test]
    fn test_dtype_to_ggml_qtype_unknown_returns_none() {
        assert_eq!(dtype_to_ggml_qtype("UNKNOWN"), None);
        assert_eq!(dtype_to_ggml_qtype("BF16"), None);
    }

    // =========================================================================
    // is_quantized_dtype tests
    // =========================================================================

    #[test]
    fn test_is_quantized_dtype_true_for_q4_k() {
        assert!(is_quantized_dtype("Q4_K"));
        assert!(is_quantized_dtype("q4_k"));
    }

    #[test]
    fn test_is_quantized_dtype_true_for_q8_0() {
        assert!(is_quantized_dtype("Q8_0"));
    }

    #[test]
    fn test_is_quantized_dtype_false_for_f32() {
        assert!(!is_quantized_dtype("F32"));
        assert!(!is_quantized_dtype("f32"));
    }

    #[test]
    fn test_is_quantized_dtype_false_for_bf16() {
        assert!(!is_quantized_dtype("BF16"));
    }

    // =========================================================================
    // TensorEntry from_binary - additional dtype coverage (using GGML dtype IDs)
    // GGML types: 0=F32, 1=F16, 2=Q4_0, 3=Q4_1, 6=Q5_0, 7=Q5_1,
    //   8=Q8_0, 9=Q8_1, 10=Q2_K, 11=Q3_K, 12=Q4_K, 13=Q5_K, 14=Q6_K, 30=BF16
    // =========================================================================

    #[test]
    fn test_tensor_entry_from_binary_dtype_q5_0() {
        let data = create_binary_tensor_entry("test.q50", 6, &[32], 0, 22);
        let (entry, _) = TensorEntry::from_binary(&data).expect("parse q5_0");
        assert_eq!(entry.dtype, "Q5_0");
    }

    #[test]
    fn test_tensor_entry_from_binary_dtype_q5_1() {
        let data = create_binary_tensor_entry("test.q51", 7, &[32], 0, 24);
        let (entry, _) = TensorEntry::from_binary(&data).expect("parse q5_1");
        assert_eq!(entry.dtype, "Q5_1");
    }

    #[test]
    fn test_tensor_entry_from_binary_dtype_q8_1() {
        // GH-438: byte 9 is now APR-native "q8" (was GGML "Q8_1" before GH-438)
        let data = create_binary_tensor_entry("test.q81", 9, &[32], 0, 36);
        let (entry, _) = TensorEntry::from_binary(&data).expect("parse q8_1");
        assert_eq!(entry.dtype, "q8");
    }

    #[test]
    fn test_tensor_entry_from_binary_dtype_q2_k() {
        let data = create_binary_tensor_entry("test.q2k", 10, &[256], 0, 84);
        let (entry, _) = TensorEntry::from_binary(&data).expect("parse q2_k");
        assert_eq!(entry.dtype, "Q2_K");
    }

    #[test]
    fn test_tensor_entry_from_binary_dtype_q4_k() {
        let data = create_binary_tensor_entry("test.q4k", 12, &[256], 0, 144);
        let (entry, _) = TensorEntry::from_binary(&data).expect("parse q4k");
        assert_eq!(entry.dtype, "Q4_K");
    }

    #[test]
    fn test_tensor_entry_from_binary_dtype_q6_k() {
        let data = create_binary_tensor_entry("test.q6k", 14, &[256], 0, 210);
        let (entry, _) = TensorEntry::from_binary(&data).expect("parse q6k");
        assert_eq!(entry.dtype, "Q6_K");
    }

    #[test]
    fn test_tensor_entry_from_binary_dtype_q8_0() {
        // GH-438: byte 8 is now APR-native "q4" (was GGML "Q8_0" before GH-438)
        let data = create_binary_tensor_entry("test.q80", 8, &[32], 0, 34);
        let (entry, _) = TensorEntry::from_binary(&data).expect("parse q8_0");
        assert_eq!(entry.dtype, "q4");
    }

    #[test]
    fn test_tensor_entry_from_binary_dtype_unknown_defaults_f32() {
        let data = create_binary_tensor_entry("test.unknown", 255, &[100], 0, 400);
        let (entry, _) = TensorEntry::from_binary(&data).expect("parse unknown");
        assert_eq!(entry.dtype, "F32");
    }

    // =========================================================================
    // AprMetadata embedded tokenizer tests
    // =========================================================================

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_present() {
        let mut extra = HashMap::new();
        extra.insert(
            "tokenizer.vocabulary".to_string(),
            serde_json::json!(["hello", "world", "test"]),
        );
        let meta = AprMetadata {
            extra,
            ..Default::default()
        };

        let vocab = meta.get_embedded_vocabulary();
        assert!(vocab.is_some());
        let v = vocab.unwrap();
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], "hello");
    }

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_missing() {
        let meta = AprMetadata::default();
        assert!(meta.get_embedded_vocabulary().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_empty_array() {
        let mut extra = HashMap::new();
        extra.insert("tokenizer.vocabulary".to_string(), serde_json::json!([]));
        let meta = AprMetadata {
            extra,
            ..Default::default()
        };
        assert!(meta.get_embedded_vocabulary().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_not_array() {
        let mut extra = HashMap::new();
        extra.insert(
            "tokenizer.vocabulary".to_string(),
            serde_json::json!("not an array"),
        );
        let meta = AprMetadata {
            extra,
            ..Default::default()
        };
        assert!(meta.get_embedded_vocabulary().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_bos_token_id_present() {
        let mut extra = HashMap::new();
        extra.insert("tokenizer.bos_token_id".to_string(), serde_json::json!(1));
        let meta = AprMetadata {
            extra,
            ..Default::default()
        };
        assert_eq!(meta.get_embedded_bos_token_id(), Some(1));
    }

    #[test]
    fn test_apr_metadata_get_embedded_bos_token_id_missing() {
        let meta = AprMetadata::default();
        assert!(meta.get_embedded_bos_token_id().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_eos_token_id_present() {
        let mut extra = HashMap::new();
        extra.insert("tokenizer.eos_token_id".to_string(), serde_json::json!(2));
        let meta = AprMetadata {
            extra,
            ..Default::default()
        };
        assert_eq!(meta.get_embedded_eos_token_id(), Some(2));
    }

    #[test]
    fn test_apr_metadata_get_embedded_eos_token_id_missing() {
        let meta = AprMetadata::default();
        assert!(meta.get_embedded_eos_token_id().is_none());
    }

    // =========================================================================
    // SimpleTokenizer tests
    // =========================================================================

    #[test]
    fn test_simple_tokenizer_new() {
        let vocab = vec!["<s>".to_string(), "hello".to_string(), "</s>".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, Some(0), Some(2));

        assert_eq!(tokenizer.vocab_size(), 3);
        assert_eq!(tokenizer.bos_token_id, Some(0));
        assert_eq!(tokenizer.eos_token_id, Some(2));
    }

    #[test]
    fn test_simple_tokenizer_vocab_size() {
        let tokenizer = SimpleTokenizer::new(vec!["a".to_string(), "b".to_string()], None, None);
        assert_eq!(tokenizer.vocab_size(), 2);
    }

    #[test]
    fn test_simple_tokenizer_is_eos_true() {
        let tokenizer = SimpleTokenizer::new(vec!["x".to_string()], None, Some(42));
        assert!(tokenizer.is_eos(42));
    }

    #[test]
    fn test_simple_tokenizer_is_eos_false() {
        let tokenizer = SimpleTokenizer::new(vec!["x".to_string()], None, Some(42));
        assert!(!tokenizer.is_eos(99));
    }

    #[test]
    fn test_simple_tokenizer_is_eos_none() {
        let tokenizer = SimpleTokenizer::new(vec!["x".to_string()], None, None);
        assert!(!tokenizer.is_eos(0));
    }

    #[test]
    fn test_simple_tokenizer_is_bos_true() {
        let tokenizer = SimpleTokenizer::new(vec!["x".to_string()], Some(10), None);
        assert!(tokenizer.is_bos(10));
    }

    #[test]
    fn test_simple_tokenizer_is_bos_false() {
        let tokenizer = SimpleTokenizer::new(vec!["x".to_string()], Some(10), None);
        assert!(!tokenizer.is_bos(99));
    }

    #[test]
    fn test_simple_tokenizer_is_bos_none() {
        let tokenizer = SimpleTokenizer::new(vec!["x".to_string()], None, None);
        assert!(!tokenizer.is_bos(0));
    }

    // =========================================================================
    // BpeTokenizer tests
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_encode_simple() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("h".to_string(), 0);
        token_to_id.insert("e".to_string(), 1);
        token_to_id.insert("l".to_string(), 2);
        token_to_id.insert("o".to_string(), 3);
        token_to_id.insert("he".to_string(), 4);
        token_to_id.insert("ll".to_string(), 5);

        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["h", "e", "l", "o", "he", "ll"]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            merge_rules: vec![
                ("h".to_string(), "e".to_string()),
                ("l".to_string(), "l".to_string()),
            ],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };

        let ids = tokenizer.encode("hello");
        // "hello" -> ['h', 'e', 'l', 'l', 'o'] -> ['he', 'l', 'l', 'o'] -> ['he', 'll', 'o']
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_decode() {
        let tokenizer = BpeTokenizer {
            token_to_id: HashMap::new(),
            id_to_token: vec!["hello".to_string(), "Ġworld".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };

        let text = tokenizer.decode(&[0, 1]);
        assert_eq!(text, "hello world");
    }

    // =========================================================================
    // byte_to_bpe_char tests
    // =========================================================================

    #[test]
    fn test_byte_to_bpe_char_space() {
        assert_eq!(byte_to_bpe_char(b' '), "Ġ");
    }

    #[test]
    fn test_byte_to_bpe_char_newline() {
        assert_eq!(byte_to_bpe_char(b'\n'), "Ċ");
    }

    #[test]
    fn test_byte_to_bpe_char_tab() {
        assert_eq!(byte_to_bpe_char(b'\t'), "ĉ");
    }

    #[test]
    fn test_byte_to_bpe_char_ascii_letter() {
        assert_eq!(byte_to_bpe_char(b'a'), "a");
        assert_eq!(byte_to_bpe_char(b'Z'), "Z");
    }

    #[test]
    fn test_byte_to_bpe_char_ascii_digit() {
        assert_eq!(byte_to_bpe_char(b'0'), "0");
        assert_eq!(byte_to_bpe_char(b'9'), "9");
    }

    #[test]
    fn test_byte_to_bpe_char_punctuation() {
        assert_eq!(byte_to_bpe_char(b'!'), "!");
        assert_eq!(byte_to_bpe_char(b'.'), ".");
    }

    #[test]
    fn test_byte_to_bpe_char_non_printable() {
        // Non-printable control character
        let result = byte_to_bpe_char(0x01);
        assert!(result.starts_with("<0x"));
        assert!(result.ends_with('>'));
    }

    #[test]
    fn test_byte_to_bpe_char_high_byte() {
        let result = byte_to_bpe_char(0xFF);
        assert_eq!(result, "<0xFF>");
    }

    // =========================================================================
    // AprV2Model::decode_tokens tests
    // =========================================================================

    #[test]
    fn test_decode_tokens_basic() {
        let vocab = vec!["hello".to_string(), "world".to_string()];
        let text = AprV2Model::decode_tokens(&vocab, &[0, 1]);
        assert_eq!(text, "helloworld");
    }
