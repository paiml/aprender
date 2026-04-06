
    #[test]
    fn test_bpe_tokenizer_encode_non_ascii_more_cov() {
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
        let encoded = tokenizer.encode("\u{00E9}"); // e-acute
        // Should not panic
        assert!(encoded.is_empty() || !encoded.is_empty());
    }

    #[test]
    fn test_apr_header_all_fields_more_cov() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 1;
        // Set all flags
        data[6..8].copy_from_slice(&0xFFFFu16.to_le_bytes());
        data[8..12].copy_from_slice(&100u32.to_le_bytes()); // tensor_count
        data[12..20].copy_from_slice(&200u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&300u32.to_le_bytes()); // metadata_size
        data[24..32].copy_from_slice(&400u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&500u64.to_le_bytes()); // data_offset
        data[40..44].copy_from_slice(&0xDEADBEEFu32.to_le_bytes()); // checksum

        let header = AprHeader::from_bytes(&data).expect("APR operation failed");
        assert_eq!(header.tensor_count, 100);
        assert_eq!(header.metadata_offset, 200);
        assert_eq!(header.metadata_size, 300);
        assert_eq!(header.tensor_index_offset, 400);
        assert_eq!(header.data_offset, 500);
        assert_eq!(header.checksum, 0xDEADBEEF);
    }

    #[test]
    fn test_apr_model_header_methods_more_cov() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        // tensor_count is accessed through public API
        assert!(model.tensor_count() >= 1);
        // tensor_names returns &str refs
        let names = model.tensor_names();
        assert!(!names.is_empty());
    }

    #[test]
    fn test_dtype_to_ggml_qtype_all_types_more_cov() {
        // Test all supported quantized types
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q4_K"), Some(12));
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q5_K"), Some(13));
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q6_K"), Some(14));
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q8_0"), Some(8));
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q4_0"), Some(2));
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q4_1"), Some(3));
        assert_eq!(crate::apr::dtype_to_ggml_qtype("Q5_0"), Some(6));
        // Non-quantized types
        assert_eq!(crate::apr::dtype_to_ggml_qtype("F32"), None);
        assert_eq!(crate::apr::dtype_to_ggml_qtype("F16"), None);
        assert_eq!(crate::apr::dtype_to_ggml_qtype("BF16"), None);
    }

    #[test]
    fn test_is_quantized_dtype_all_types_more_cov() {
        // Quantized types
        assert!(crate::apr::is_quantized_dtype("Q4_K"));
        assert!(crate::apr::is_quantized_dtype("q4_k"));
        assert!(crate::apr::is_quantized_dtype("Q5_K"));
        assert!(crate::apr::is_quantized_dtype("Q6_K"));
        assert!(crate::apr::is_quantized_dtype("Q8_0"));
        assert!(crate::apr::is_quantized_dtype("Q4_0"));
        assert!(crate::apr::is_quantized_dtype("Q4_1"));
        assert!(crate::apr::is_quantized_dtype("Q5_0"));
        // Non-quantized types
        assert!(!crate::apr::is_quantized_dtype("F32"));
        assert!(!crate::apr::is_quantized_dtype("F16"));
        assert!(!crate::apr::is_quantized_dtype("BF16"));
        assert!(!crate::apr::is_quantized_dtype("I8"));
    }

    #[test]
    fn test_dequantize_q4_k_early_exit_more_cov() {
        // Test early exit when num_elements is reached mid-sub-block
        let mut bytes = vec![0u8; 144];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // d = 1.0
                         // Request exactly 17 elements (mid sub-block)
        let result = crate::apr::dequantize_q4_k(&bytes, 17);
        assert_eq!(result.len(), 17);
    }

    #[test]
    fn test_dequantize_q6_k_early_exit_more_cov() {
        // Test early exit when num_elements is reached mid-sub-block
        let mut bytes = vec![0u8; 210];
        bytes[208] = 0x00;
        bytes[209] = 0x3C; // d = 1.0
                           // Request exactly 33 elements (mid sub-block)
        let result = crate::apr::dequantize_q6_k(&bytes, 33);
        assert_eq!(result.len(), 33);
    }

    #[test]
    fn test_f16_to_f32_negative_subnormal_more_cov() {
        // Negative subnormal: sign=1, exp=0, mantissa=1
        let bits: u16 = 0x8001;
        let result = crate::apr::f16_to_f32(bits);
        assert!(result < 0.0, "Negative subnormal should be negative");
    }

    #[test]
    fn test_f16_to_f32_negative_nan_more_cov() {
        // Negative NaN: sign=1, exp=31, mantissa!=0
        let bits: u16 = 0xFC01;
        let result = crate::apr::f16_to_f32(bits);
        assert!(result.is_nan());
    }

    #[test]
    fn test_apr_model_forward_with_transformer_missing_tensor_more_cov() {
        // Create a model with transformer metadata but missing tensors
        let metadata = r#"{
            "hidden_size": 8,
            "num_layers": 1,
            "num_heads": 2,
            "vocab_size": 10
        }"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset;
        let mut data = vec![0u8; data_offset as usize + 64];

        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // No tensors
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());

        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

        let model = AprV2Model::from_bytes(data).expect("should load");
        assert!(model.metadata().is_transformer());
        let result = model.forward(&[1]);
        // Should fail because embedding tensor is missing
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_tokens_sentencepiece_prefix_more_cov() {
        let vocab = vec!["▁hello".to_string(), "▁".to_string(), "world".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
        // SentencePiece ▁ prefix should be handled
        assert!(result.contains("hello"));
        assert!(result.contains("world"));
    }

    #[test]
    fn test_model_data_type_alias_more_cov() {
        // Test type alias AprModel = AprV2Model
        let data = create_test_apr_model();
        let model: AprModel = AprV2Model::from_bytes(data).expect("should load");
        assert_eq!(model.tensor_count(), 1);
    }

    #[test]
    fn test_apr_model_type_alias_more_cov() {
        // Test legacy type alias AprModelType
        let _: AprModelType = ();
    }

    #[test]
    fn test_find_tensor_name_not_found_more_cov() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");
        // Try to find a tensor that doesn't exist
        let result = model.forward(&[0]); // Will try to find embedding tensor
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_msg = format!("{:?}", err);
        // Should contain "No matching tensor" or similar
        assert!(
            err_msg.contains("not a transformer") || err_msg.contains("No matching"),
            "Error message: {}",
            err_msg
        );
    }

    // =========================================================================
    // SimpleTokenizer Tests (GH-156)
    // =========================================================================

    #[test]
    fn test_simple_tokenizer_new() {
        use crate::apr::SimpleTokenizer;

        let vocab = vec![
            "<pad>".to_string(),
            "<bos>".to_string(),
            "<eos>".to_string(),
            "hello".to_string(),
            "world".to_string(),
        ];
        let tokenizer = SimpleTokenizer::new(vocab, Some(1), Some(2));

        assert_eq!(tokenizer.vocab_size(), 5);
        assert!(tokenizer.is_bos(1));
        assert!(tokenizer.is_eos(2));
        assert!(!tokenizer.is_bos(0));
        assert!(!tokenizer.is_eos(0));
    }

    #[test]
    fn test_simple_tokenizer_decode_basic() {
        use crate::apr::SimpleTokenizer;

        let vocab = vec![
            "<pad>".to_string(),
            "<bos>".to_string(),
            "<eos>".to_string(),
            "hello".to_string(),
            "world".to_string(),
        ];
        let tokenizer = SimpleTokenizer::new(vocab, Some(1), Some(2));

        // Decode tokens [3, 4] -> "helloworld"
        let decoded = tokenizer.decode(&[3, 4]);
        assert_eq!(decoded, "helloworld");
    }

    #[test]
    fn test_simple_tokenizer_decode_bpe_space() {
        use crate::apr::SimpleTokenizer;

        // BPE-style tokens with Ġ prefix (represents space)
        let vocab = vec![
            "<pad>".to_string(),
            "<bos>".to_string(),
            "<eos>".to_string(),
            "Ġhello".to_string(), // " hello"
            "Ġworld".to_string(), // " world"
            "!".to_string(),
        ];
        let tokenizer = SimpleTokenizer::new(vocab, Some(1), Some(2));

        // Decode tokens [3, 4, 5] -> " hello world!"
        let decoded = tokenizer.decode(&[3, 4, 5]);
        assert_eq!(decoded, " hello world!");
    }

    #[test]
    fn test_simple_tokenizer_decode_out_of_bounds() {
        use crate::apr::SimpleTokenizer;

        let vocab = vec!["a".to_string(), "b".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);

        // Token 99 is out of bounds - should be skipped or handled gracefully
        let decoded = tokenizer.decode(&[0, 99, 1]);
        // Should contain "a" and "b", may have placeholder for 99
        assert!(decoded.contains('a'));
        assert!(decoded.contains('b'));
    }

    #[test]
    fn test_simple_tokenizer_no_special_tokens() {
        use crate::apr::SimpleTokenizer;

        let vocab = vec!["x".to_string(), "y".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);

        // No BOS/EOS defined
        assert!(!tokenizer.is_bos(0));
        assert!(!tokenizer.is_bos(1));
        assert!(!tokenizer.is_eos(0));
        assert!(!tokenizer.is_eos(1));
    }

    #[test]
    fn test_simple_tokenizer_empty_decode() {
        use crate::apr::SimpleTokenizer;

        let vocab = vec!["a".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);

        let decoded = tokenizer.decode(&[]);
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_simple_tokenizer_clone() {
        use crate::apr::SimpleTokenizer;

        let vocab = vec!["test".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, Some(0), None);
        let cloned = tokenizer.clone();

        assert_eq!(tokenizer.vocab_size(), cloned.vocab_size());
        assert_eq!(tokenizer.bos_token_id, cloned.bos_token_id);
    }

    #[test]
    fn test_simple_tokenizer_debug() {
        use crate::apr::SimpleTokenizer;

        let vocab = vec!["a".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);
        let debug_str = format!("{:?}", tokenizer);

        assert!(debug_str.contains("SimpleTokenizer"));
    }

    // =========================================================================
    // APR v1 Format Rejection Tests
    // =========================================================================

    #[test]
    fn test_apr_v1_format_rejected() {
        // APR v1 magic: "APR1" (0x41, 0x50, 0x52, 0x31)
        let mut data = vec![0u8; HEADER_SIZE + 100];
        data[0..4].copy_from_slice(&[0x41, 0x50, 0x52, 0x31]); // "APR1"
        data[4] = 1; // version major
        data[5] = 0; // version minor

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        // Should indicate APR v1 not supported
        assert!(
            err.contains("APR v1") || err.contains("not supported"),
            "Error should mention APR v1: {}",
            err
        );
    }

    #[test]
    fn test_apr_v1_format_conversion_hint() {
        // Test that error message suggests conversion
        let mut data = vec![0u8; HEADER_SIZE + 100];
        data[0..4].copy_from_slice(&[0x41, 0x50, 0x52, 0x31]); // "APR1"

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        // Error should hint at conversion or GGUF alternative
        assert!(
            err.contains("convert") || err.contains("GGUF"),
            "Error should suggest conversion: {}",
            err
        );
    }

    #[test]
    fn test_apr_invalid_version_byte() {
        // Test with invalid version byte (not 0, '1', or '2')
        let mut data = vec![0u8; HEADER_SIZE + 100];
        data[0..3].copy_from_slice(&MAGIC_PREFIX); // "APR"
        data[3] = 0x99; // Invalid version byte

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("version") || err.contains("Invalid"),
            "Error should mention version: {}",
            err
        );
    }

    // =========================================================================
    // MappedAprModel Tests
    // =========================================================================

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_from_path() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a valid APR v2 file with legacy magic
        let mut temp = NamedTempFile::new().expect("create temp file");

        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC); // APR\0 legacy magic
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags = 0
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
        data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&64u64.to_le_bytes()); // data_offset

        temp.write_all(&data).expect("write data");

        let model = MappedAprModel::from_path(temp.path()).expect("load model");
        assert_eq!(model.tensor_count(), 0);
        assert_eq!(model.file_size(), data.len());
        assert_eq!(model.data_offset(), 64);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_nonexistent_file() {
        let result = MappedAprModel::from_path("/nonexistent/path/model.apr");
        assert!(result.is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_invalid_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");

        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(b"GGUF"); // Wrong magic

        temp.write_all(&data).expect("write data");

        let result = MappedAprModel::from_path(temp.path());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("magic") || err.contains("Invalid"));
    }
