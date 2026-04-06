
    // =========================================================================
    // byte_to_bpe_char extended coverage
    // =========================================================================

    #[test]
    fn test_byte_to_bpe_char_all_printable_ascii() {
        // Test all printable ASCII characters
        for b in 33u8..=126 {
            let result = crate::apr::byte_to_bpe_char(b);
            assert!(!result.is_empty());
        }
    }

    #[test]
    fn test_byte_to_bpe_char_control_chars() {
        // Test control characters 0-31
        for b in 0u8..32 {
            let result = crate::apr::byte_to_bpe_char(b);
            assert!(!result.is_empty());
        }
    }

    #[test]
    fn test_byte_to_bpe_char_extended_ascii() {
        // Test extended ASCII 128-255
        for b in 128u8..=255 {
            let result = crate::apr::byte_to_bpe_char(b);
            assert!(!result.is_empty());
        }
    }

    // =========================================================================
    // BpeTokenizer extended coverage
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_decode_multiple() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("hello".to_string(), 0);
        token_to_id.insert(" ".to_string(), 1);
        token_to_id.insert("world".to_string(), 2);

        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["hello".to_string(), " ".to_string(), "world".to_string()],
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };

        let decoded = tokenizer.decode(&[0, 1, 2]);
        assert!(decoded.contains("hello"));
        assert!(decoded.contains("world"));
    }

    // =========================================================================
    // AprHeader flags coverage
    // =========================================================================

    #[test]
    fn test_apr_header_with_flags() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        // Set quantized flag
        data[6] = 0x20;
        data[7] = 0x00;

        let header = AprHeader::from_bytes(&data).expect("APR operation failed");
        assert!(header.flags.is_quantized());
    }

    // =========================================================================
    // AprV2Model predict with weight tensor
    // =========================================================================

    #[test]
    fn test_apr_v2_model_predict_single_feature() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let features = vec![1.0];
        let result = model.predict(&features);
        assert!(result.is_ok());
    }

    #[test]
    fn test_apr_v2_model_predict_many_features() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let features: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let result = model.predict(&features);
        assert!(result.is_ok());
    }

    // =========================================================================
    // TensorEntry element_count edge cases
    // =========================================================================

    #[test]
    fn test_tensor_entry_element_count_1d() {
        let entry = TensorEntry {
            name: "vec".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10],
            offset: 0,
            size: 40,
        };
        assert_eq!(entry.element_count(), 10);
    }

    #[test]
    fn test_tensor_entry_element_count_3d() {
        let entry = TensorEntry {
            name: "3d".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 3, 4],
            offset: 0,
            size: 96,
        };
        assert_eq!(entry.element_count(), 24);
    }

    // =========================================================================
    // dequantize_f16 edge cases
    // =========================================================================

    #[test]
    fn test_dequantize_f16_odd_bytes() {
        // Odd number of bytes - last byte ignored
        let bytes = vec![0x00, 0x3C, 0x00]; // 1.0 + extra byte
        let result = crate::apr::dequantize_f16(&bytes, 2);
        assert_eq!(result.len(), 1); // Only one complete f16
    }

    // =========================================================================
    // detect_format additional cases
    // =========================================================================

    #[test]
    fn test_detect_format_bin() {
        // .bin is not a recognized format, returns "unknown"
        assert_eq!(detect_format("/path/model.bin"), "unknown");
    }

    #[test]
    fn test_detect_format_pt() {
        // .pt is not a recognized format, returns "unknown"
        assert_eq!(detect_format("/path/model.pt"), "unknown");
    }

    #[test]
    fn test_detect_format_no_extension() {
        // No extension returns "unknown"
        assert_eq!(detect_format("/path/model"), "unknown");
    }

    #[test]
    fn test_detect_format_hidden_file() {
        // Hidden file with no extension returns "unknown"
        assert_eq!(detect_format("/path/.hidden"), "unknown");
    }

    // =========================================================================
    // dequantize Early Exit Tests - Hit break paths by requesting fewer elements
    // =========================================================================

    #[test]
    fn test_dequantize_q8_0_fewer_than_block() {
        // Q8_0 block: 2-byte scale + 32 i8 values = 34 bytes, 32 elements
        // Request only 10 elements to trigger early exit
        let mut bytes = vec![0u8; 34];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // Scale = 1.0 in f16
        for i in 0..32 {
            bytes[2 + i] = (i + 1) as u8; // Values 1-32
        }

        let result = crate::apr::dequantize_q8_0(&bytes, 10);
        assert_eq!(result.len(), 10);
        // First value: 1 * 1.0 = 1.0
        assert!((result[0] - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_dequantize_q8_0_multiple_blocks() {
        // Two Q8_0 blocks = 68 bytes, 64 elements
        let mut bytes = vec![0u8; 68];
        // Block 1
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // Scale = 1.0
                         // Block 2
        bytes[34] = 0x00;
        bytes[35] = 0x3C; // Scale = 1.0

        let result = crate::apr::dequantize_q8_0(&bytes, 64);
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_dequantize_q4_k_fewer_than_superblock() {
        // Q4_K super-block: 144 bytes = 256 elements
        // Request only 100 elements to trigger early exit
        let mut bytes = vec![0u8; 144];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // d = 1.0

        let result = crate::apr::dequantize_q4_k(&bytes, 100);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_dequantize_q4_k_multiple_superblocks() {
        // Two Q4_K super-blocks = 288 bytes = 512 elements
        let mut bytes = vec![0u8; 288];
        // First superblock
        bytes[0] = 0x00;
        bytes[1] = 0x3C;
        // Second superblock
        bytes[144] = 0x00;
        bytes[145] = 0x3C;

        let result = crate::apr::dequantize_q4_k(&bytes, 512);
        assert_eq!(result.len(), 512);
    }

    #[test]
    fn test_dequantize_q6_k_fewer_than_superblock() {
        // Q6_K super-block: 210 bytes = 256 elements
        // Request only 50 elements to trigger early exit
        let mut bytes = vec![0u8; 210];
        bytes[208] = 0x00;
        bytes[209] = 0x3C; // d = 1.0

        let result = crate::apr::dequantize_q6_k(&bytes, 50);
        assert_eq!(result.len(), 50);
    }

    #[test]
    fn test_dequantize_q6_k_multiple_superblocks() {
        // Two Q6_K super-blocks = 420 bytes = 512 elements
        let mut bytes = vec![0u8; 420];
        // First superblock d at 208-209
        bytes[208] = 0x00;
        bytes[209] = 0x3C;
        // Second superblock d at 418-419
        bytes[418] = 0x00;
        bytes[419] = 0x3C;

        let result = crate::apr::dequantize_q6_k(&bytes, 512);
        assert_eq!(result.len(), 512);
    }

    // =========================================================================
    // BpeTokenizer encode tests
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_encode_empty() {
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
    fn test_bpe_tokenizer_encode_with_merges() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        token_to_id.insert("b".to_string(), 1);
        token_to_id.insert("ab".to_string(), 2);

        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["a".to_string(), "b".to_string(), "ab".to_string()],
            merge_rules: vec![("a".to_string(), "b".to_string())],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };
        let encoded = tokenizer.encode("ab");
        // Should merge a+b -> ab
        assert!(!encoded.is_empty());
    }

    // =========================================================================
    // AprV2Model find_tensor_name tests
    // =========================================================================

    #[test]
    fn test_apr_v2_model_tensor_count() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        assert_eq!(model.tensor_count(), 1);
    }

    #[test]
    fn test_apr_v2_model_total_parameters() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        // Test model has 4x4 = 16 element tensor
        assert!(model.estimated_parameters() > 0);
    }

    // =========================================================================
    // AprMetadata serialization tests
    // =========================================================================

    #[test]
    fn test_apr_metadata_to_json() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: Some(4),
            num_heads: Some(8),
            vocab_size: Some(32000),
            ..Default::default()
        };
        let json = serde_json::to_string(&meta).expect("invalid UTF-8");
        assert!(json.contains("256"));
        assert!(json.contains("hidden_size"));
    }

    #[test]
    fn test_apr_metadata_roundtrip() {
        let meta = AprMetadata {
            hidden_size: Some(1024),
            num_layers: Some(12),
            num_heads: Some(16),
            vocab_size: Some(50000),
            model_type: Some("llama".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&meta).expect("invalid UTF-8");
        let parsed: AprMetadata = serde_json::from_str(&json).expect("parse failed");
        assert_eq!(parsed.hidden_size, Some(1024));
        assert_eq!(parsed.model_type, Some("llama".to_string()));
    }

    // =========================================================================
    // TensorEntry tests with various shapes
    // =========================================================================

    #[test]
    fn test_tensor_entry_5d_shape() {
        let entry = TensorEntry {
            name: "5d".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2, 3, 4, 5, 6],
            offset: 0,
            size: 2880,
        };
        assert_eq!(entry.element_count(), 720);
    }

    #[test]
    fn test_tensor_entry_single_element() {
        let entry = TensorEntry {
            name: "single".to_string(),
            dtype: "F32".to_string(),
            shape: vec![1],
            offset: 0,
            size: 4,
        };
        assert_eq!(entry.element_count(), 1);
    }

    // =========================================================================
    // is_apr_file tests (requires actual file with magic bytes)
    // =========================================================================

    #[test]
    fn test_is_apr_file_nonexistent_path() {
        // Non-existent file returns false
        assert!(!is_apr_file("/nonexistent/path/model.apr"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_is_apr_file_with_apr_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(&MAGIC).expect("write magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert!(is_apr_file(temp.path()));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_is_apr_file_without_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(b"GGUF").expect("write wrong magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert!(!is_apr_file(temp.path()));
    }

    // =========================================================================
    // More rms_norm tests
    // =========================================================================

    #[test]
    fn test_rms_norm_large() {
        let x: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let weight: Vec<f32> = vec![1.0; 64];
        let eps = 1e-6;
        let result = crate::apr::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_rms_norm_negative_values() {
        let x = vec![-1.0, -2.0, -3.0, -4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-6;
        let result = crate::apr::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        // Normalized negative values should remain negative
        assert!(result[0] < 0.0);
    }

    // =========================================================================
    // matmul extended tests
    // =========================================================================

    #[test]
    fn test_matmul_zeros() {
        let x = vec![0.0; 4];
        let w = vec![1.0; 4];
        let result = crate::apr::matmul(&x, &w, 2, 2, 2);
        assert_eq!(result.len(), 4);
        for v in &result {
            assert!(v.abs() < 1e-6);
        }
    }
