
    #[test]
    fn test_f16_to_f32_half() {
        let half: u16 = 0x3800; // 0.5 in f16
        let result = crate::apr::f16_to_f32(half);
        assert!((result - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_two() {
        let two: u16 = 0x4000; // 2.0 in f16
        let result = crate::apr::f16_to_f32(two);
        assert!((result - 2.0).abs() < 1e-6);
    }

    // =========================================================================
    // dequantize_f16 Tests
    // =========================================================================

    #[test]
    fn test_dequantize_f16_empty() {
        let result = crate::apr::dequantize_f16(&[], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_f16_single() {
        // f16: 0x3C00 = 1.0 (little-endian: 0x00, 0x3C)
        let data: &[u8] = &[0x00, 0x3C];
        let result = crate::apr::dequantize_f16(data, 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_dequantize_f16_multiple() {
        // f16: 0x3C00 = 1.0, 0x4000 = 2.0
        let data: &[u8] = &[0x00, 0x3C, 0x00, 0x40];
        let result = crate::apr::dequantize_f16(data, 2);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 1e-4);
        assert!((result[1] - 2.0).abs() < 1e-4);
    }

    // =========================================================================
    // dequantize_q8_0 Tests
    // =========================================================================

    #[test]
    fn test_dequantize_q8_0_basic() {
        // Q8_0 block: 2-byte f16 scale + 32 int8 values
        // Create a simple block with scale = 1.0 (0x3C00) and values 0..32
        let mut data = Vec::new();
        data.push(0x00); // scale low byte
        data.push(0x3C); // scale high byte (1.0 in f16)
        for i in 0..32u8 {
            data.push(i);
        }
        let result = crate::apr::dequantize_q8_0(&data, 32);
        assert_eq!(result.len(), 32);
        // First value should be 0 * 1.0 = 0.0
        assert!((result[0] - 0.0).abs() < 1e-4);
        // Value at index 10 should be 10 * 1.0 = 10.0
        assert!((result[10] - 10.0).abs() < 1e-4);
    }

    // =========================================================================
    // BPE Tokenizer Tests
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_empty_vocab() {
        let tokenizer = BpeTokenizer {
            token_to_id: HashMap::new(),
            id_to_token: vec![],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let encoded = tokenizer.encode("");
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_single_char_vocab() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        token_to_id.insert("b".to_string(), 1);
        token_to_id.insert("c".to_string(), 2);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let encoded = tokenizer.encode("abc");
        assert_eq!(encoded.len(), 3);
    }

    #[test]
    fn test_bpe_tokenizer_unknown_char() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["a".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let encoded = tokenizer.encode("xyz");
        // Unknown chars should be handled gracefully
        assert!(encoded.is_empty() || encoded.iter().all(|&t| t == 0));
    }

    #[test]
    fn test_bpe_tokenizer_decode_empty() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["a".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let decoded = tokenizer.decode(&[]);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_decode_valid() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("hello".to_string(), 0);
        token_to_id.insert("world".to_string(), 1);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["hello".to_string(), "world".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let decoded = tokenizer.decode(&[0, 1]);
        assert!(decoded.contains("hello"));
        assert!(decoded.contains("world"));
    }

    #[test]
    fn test_bpe_tokenizer_with_merge_rules() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("h".to_string(), 0);
        token_to_id.insert("e".to_string(), 1);
        token_to_id.insert("he".to_string(), 2);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["h".to_string(), "e".to_string(), "he".to_string()],
            merge_rules: vec![("h".to_string(), "e".to_string())],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let encoded = tokenizer.encode("he");
        // After merging h+e -> he, should have 1 token
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_with_bos_eos() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("<s>".to_string(), 0);
        token_to_id.insert("</s>".to_string(), 1);
        token_to_id.insert("a".to_string(), 2);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["<s>".to_string(), "</s>".to_string(), "a".to_string()],
            merge_rules: vec![],
            bos_id: Some(0),
            eos_id: Some(1),
            special_tokens: HashMap::new(),
        };
        assert_eq!(tokenizer.bos_id, Some(0));
        assert_eq!(tokenizer.eos_id, Some(1));
    }

    // =========================================================================
    // GH-189: extract_special_tokens_from_vocab Tests
    // =========================================================================

    #[test]
    fn test_extract_special_tokens_from_vocab_qwen() {
        // GH-189: Test extraction of Qwen-style special tokens
        let mut vocab: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        vocab.insert("<|im_start|>".to_string(), 151644);
        vocab.insert("<|im_end|>".to_string(), 151645);
        vocab.insert("<|endoftext|>".to_string(), 151643);
        vocab.insert("hello".to_string(), 15339);
        vocab.insert("world".to_string(), 1917);

        let special = extract_special_tokens_from_vocab(&vocab);

        assert_eq!(special.len(), 3);
        assert_eq!(special.get("<|im_start|>"), Some(&151644));
        assert_eq!(special.get("<|im_end|>"), Some(&151645));
        assert_eq!(special.get("<|endoftext|>"), Some(&151643));
        // Regular tokens should NOT be in special_tokens
        assert!(!special.contains_key("hello"));
        assert!(!special.contains_key("world"));
    }

    #[test]
    fn test_extract_special_tokens_from_vocab_llama() {
        // GH-189: Test extraction of LLaMA-style special tokens
        let mut vocab: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        vocab.insert("<s>".to_string(), 1);
        vocab.insert("</s>".to_string(), 2);
        vocab.insert("<unk>".to_string(), 0);
        vocab.insert("<pad>".to_string(), 3);
        vocab.insert("the".to_string(), 1000);

        let special = extract_special_tokens_from_vocab(&vocab);

        assert!(special.contains_key("<s>"));
        assert!(special.contains_key("</s>"));
        assert!(special.contains_key("<unk>"));
        assert!(special.contains_key("<pad>"));
        assert!(!special.contains_key("the"));
    }

    #[test]
    fn test_extract_special_tokens_from_vocab_empty() {
        // GH-189: Empty vocab should return empty special tokens
        let vocab: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        let special = extract_special_tokens_from_vocab(&vocab);
        assert!(special.is_empty());
    }

    #[test]
    fn test_extract_special_tokens_from_vocab_no_special() {
        // GH-189: Vocab with no special tokens should return empty
        let mut vocab: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert("the".to_string(), 2);

        let special = extract_special_tokens_from_vocab(&vocab);
        assert!(special.is_empty());
    }

    #[test]
    fn test_extract_special_tokens_pattern_matching() {
        // GH-189: Test that any <|...|> pattern is captured
        let mut vocab: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        vocab.insert("<|custom_token|>".to_string(), 999);
        vocab.insert("<|another|>".to_string(), 998);
        vocab.insert("not_special".to_string(), 100);

        let special = extract_special_tokens_from_vocab(&vocab);

        assert!(special.contains_key("<|custom_token|>"));
        assert!(special.contains_key("<|another|>"));
        assert!(!special.contains_key("not_special"));
    }

    // =========================================================================
    // is_apr_file Tests
    // =========================================================================

    #[test]
    fn test_is_apr_file_nonexistent() {
        // Non-existent file returns false (can't read magic bytes)
        assert!(!is_apr_file("/nonexistent/path/model.apr"));
    }

    #[test]
    fn test_is_apr_file_wrong_extension() {
        // Non-existent files all return false
        assert!(!is_apr_file("/some/path/model.gguf"));
        assert!(!is_apr_file("/some/path/model.safetensors"));
        assert!(!is_apr_file("/some/path/model.bin"));
    }

    #[test]
    fn test_is_apr_file_no_extension() {
        assert!(!is_apr_file("/some/path/model"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_is_apr_file_with_valid_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        // Write APR magic bytes
        temp.write_all(&MAGIC).expect("write magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert!(is_apr_file(temp.path()));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_is_apr_file_with_wrong_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        // Write wrong magic bytes
        temp.write_all(b"GGUF").expect("write magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert!(!is_apr_file(temp.path()));
    }

    // =========================================================================
    // detect_format Tests
    // =========================================================================

    #[test]
    fn test_detect_format_apr() {
        assert_eq!(detect_format("/path/model.apr"), "apr");
    }

    #[test]
    fn test_detect_format_gguf() {
        assert_eq!(detect_format("/path/model.gguf"), "gguf");
    }

    #[test]
    fn test_detect_format_safetensors() {
        assert_eq!(detect_format("/path/model.safetensors"), "safetensors");
    }

    // =========================================================================
    // TensorEntry Tests
    // =========================================================================

    #[test]
    fn test_tensor_entry_dtype_sizes() {
        let entry = TensorEntry {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10, 20],
            offset: 0,
            size: 800, // 10*20*4 bytes
        };
        assert_eq!(entry.element_count(), 200);
    }

    #[test]
    fn test_tensor_entry_empty_shape() {
        let entry = TensorEntry {
            name: "scalar".to_string(),
            dtype: "F32".to_string(),
            shape: vec![],
            offset: 0,
            size: 4,
        };
        assert_eq!(entry.element_count(), 1);
    }

    #[test]
    fn test_tensor_entry_4d_shape() {
        let entry = TensorEntry {
            name: "4d".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 3, 4, 5],
            offset: 0,
            size: 480,
        };
        assert_eq!(entry.element_count(), 120);
    }

    #[test]
    fn test_tensor_entry_1d_shape() {
        let entry = TensorEntry {
            name: "1d".to_string(),
            dtype: "F32".to_string(),
            shape: vec![100],
            offset: 0,
            size: 400,
        };
        assert_eq!(entry.element_count(), 100);
    }

    // =========================================================================
    // ModelData Tests
    // =========================================================================

    #[test]
    fn test_model_data_len() {
        let data = ModelData::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(data.len(), 5);
    }

    #[test]
    fn test_model_data_is_empty() {
        let empty = ModelData::from_vec(vec![]);
        assert!(empty.is_empty());

        let non_empty = ModelData::from_vec(vec![1]);
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_model_data_as_slice() {
        let data = ModelData::from_vec(vec![10, 20, 30]);
        let slice = data.as_slice();
        assert_eq!(slice, &[10, 20, 30]);
    }

    #[test]
    fn test_model_data_large() {
        let large_data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let data = ModelData::from_vec(large_data);
        assert_eq!(data.len(), 1000);
        assert_eq!(data.as_slice()[0], 0);
        assert_eq!(data.as_slice()[255], 255);
        assert_eq!(data.as_slice()[256], 0);
    }
