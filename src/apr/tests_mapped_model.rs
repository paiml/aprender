
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_truncated() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");

        // Write less than HEADER_SIZE bytes
        let data = vec![0u8; 32];
        temp.write_all(&data).expect("write data");

        let result = MappedAprModel::from_path(temp.path());
        assert!(result.is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_data_access() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");

        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes());
        data[8..12].copy_from_slice(&0u32.to_le_bytes());
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[20..24].copy_from_slice(&0u32.to_le_bytes());
        data[24..32].copy_from_slice(&64u64.to_le_bytes());
        data[32..40].copy_from_slice(&64u64.to_le_bytes());

        temp.write_all(&data).expect("write data");

        let model = MappedAprModel::from_path(temp.path()).expect("load model");

        // Verify data() returns the full mmap
        let model_data = model.data();
        assert_eq!(model_data.len(), 128);
        assert_eq!(&model_data[0..4], &MAGIC);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_find_tensor_not_found() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");

        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes());
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // 0 tensors
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[20..24].copy_from_slice(&0u32.to_le_bytes());
        data[24..32].copy_from_slice(&64u64.to_le_bytes());
        data[32..40].copy_from_slice(&64u64.to_le_bytes());

        temp.write_all(&data).expect("write data");

        let model = MappedAprModel::from_path(temp.path()).expect("load model");
        assert!(model.find_tensor("nonexistent").is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_get_tensor_data_not_found() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");

        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes());
        data[8..12].copy_from_slice(&0u32.to_le_bytes());
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[20..24].copy_from_slice(&0u32.to_le_bytes());
        data[24..32].copy_from_slice(&64u64.to_le_bytes());
        data[32..40].copy_from_slice(&64u64.to_le_bytes());

        temp.write_all(&data).expect("write data");

        let model = MappedAprModel::from_path(temp.path()).expect("load model");
        let result = model.get_tensor_data("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_mapped_apr_model_dtype_to_qtype() {
        // F32 -> 0
        assert_eq!(MappedAprModel::dtype_to_qtype("F32"), 0);
        // F16 -> 1
        assert_eq!(MappedAprModel::dtype_to_qtype("F16"), 1);
        // Q4_0 -> 2
        assert_eq!(MappedAprModel::dtype_to_qtype("Q4_0"), 2);
        // Q4_1 -> 3
        assert_eq!(MappedAprModel::dtype_to_qtype("Q4_1"), 3);
        // Q5_0 -> 6
        assert_eq!(MappedAprModel::dtype_to_qtype("Q5_0"), 6);
        // Q5_1 -> 7
        assert_eq!(MappedAprModel::dtype_to_qtype("Q5_1"), 7);
        // Q8_0 -> 8
        assert_eq!(MappedAprModel::dtype_to_qtype("Q8_0"), 8);
        // Q8_1 -> 9
        assert_eq!(MappedAprModel::dtype_to_qtype("Q8_1"), 9);
        // Q2_K -> 10
        assert_eq!(MappedAprModel::dtype_to_qtype("Q2_K"), 10);
        // Q3_K -> 11
        assert_eq!(MappedAprModel::dtype_to_qtype("Q3_K"), 11);
        // Q4_K -> 12
        assert_eq!(MappedAprModel::dtype_to_qtype("Q4_K"), 12);
        // Q5_K -> 13
        assert_eq!(MappedAprModel::dtype_to_qtype("Q5_K"), 13);
        // Q6_K -> 14
        assert_eq!(MappedAprModel::dtype_to_qtype("Q6_K"), 14);
        // IQ2_XXS -> 16
        assert_eq!(MappedAprModel::dtype_to_qtype("IQ2_XXS"), 16);
        // IQ2_XS -> 17
        assert_eq!(MappedAprModel::dtype_to_qtype("IQ2_XS"), 17);
        // BF16 -> 30
        assert_eq!(MappedAprModel::dtype_to_qtype("BF16"), 30);
        // Unknown -> 0 (default to F32)
        assert_eq!(MappedAprModel::dtype_to_qtype("UNKNOWN"), 0);
    }

    // =========================================================================
    // Embedded Tokenizer Tests (GH-156)
    // =========================================================================

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_none() {
        let meta = AprMetadata::default();
        assert!(meta.get_embedded_vocabulary().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_empty() {
        let mut meta = AprMetadata::default();
        meta.extra
            .insert("tokenizer.vocabulary".to_string(), serde_json::json!([]));
        // Empty array should return None
        assert!(meta.get_embedded_vocabulary().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_valid() {
        let mut meta = AprMetadata::default();
        meta.extra.insert(
            "tokenizer.vocabulary".to_string(),
            serde_json::json!(["<pad>", "<bos>", "<eos>", "hello", "world"]),
        );

        let vocab = meta.get_embedded_vocabulary().expect("should have vocab");
        assert_eq!(vocab.len(), 5);
        assert_eq!(vocab[0], "<pad>");
        assert_eq!(vocab[3], "hello");
    }

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_invalid_type() {
        let mut meta = AprMetadata::default();
        // Not an array
        meta.extra.insert(
            "tokenizer.vocabulary".to_string(),
            serde_json::json!("not an array"),
        );
        assert!(meta.get_embedded_vocabulary().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_vocabulary_mixed_types() {
        let mut meta = AprMetadata::default();
        // Array with mixed types - only strings should be kept
        meta.extra.insert(
            "tokenizer.vocabulary".to_string(),
            serde_json::json!(["valid", 123, "also_valid", null]),
        );

        let vocab = meta.get_embedded_vocabulary().expect("should have vocab");
        assert_eq!(vocab.len(), 2);
        assert_eq!(vocab[0], "valid");
        assert_eq!(vocab[1], "also_valid");
    }

    #[test]
    fn test_apr_metadata_get_embedded_bos_token_id_none() {
        let meta = AprMetadata::default();
        assert!(meta.get_embedded_bos_token_id().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_bos_token_id_valid() {
        let mut meta = AprMetadata::default();
        meta.extra
            .insert("tokenizer.bos_token_id".to_string(), serde_json::json!(1));

        assert_eq!(meta.get_embedded_bos_token_id(), Some(1));
    }

    #[test]
    fn test_apr_metadata_get_embedded_bos_token_id_invalid_type() {
        let mut meta = AprMetadata::default();
        meta.extra.insert(
            "tokenizer.bos_token_id".to_string(),
            serde_json::json!("not a number"),
        );
        assert!(meta.get_embedded_bos_token_id().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_eos_token_id_none() {
        let meta = AprMetadata::default();
        assert!(meta.get_embedded_eos_token_id().is_none());
    }

    #[test]
    fn test_apr_metadata_get_embedded_eos_token_id_valid() {
        let mut meta = AprMetadata::default();
        meta.extra
            .insert("tokenizer.eos_token_id".to_string(), serde_json::json!(2));

        assert_eq!(meta.get_embedded_eos_token_id(), Some(2));
    }

    #[test]
    fn test_apr_metadata_get_embedded_eos_token_id_invalid_type() {
        let mut meta = AprMetadata::default();
        meta.extra.insert(
            "tokenizer.eos_token_id".to_string(),
            serde_json::json!({"nested": "object"}),
        );
        assert!(meta.get_embedded_eos_token_id().is_none());
    }

    #[test]
    fn test_load_embedded_tokenizer_no_vocab() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        // Model has no embedded vocabulary
        assert!(model.load_embedded_tokenizer().is_none());
    }

    /// Helper to create APR model with embedded tokenizer
    fn create_apr_model_with_embedded_tokenizer() -> Vec<u8> {
        let metadata = r#"{
            "architecture": "test",
            "vocab_size": 5,
            "hidden_size": 64,
            "tokenizer.vocabulary": ["<pad>", "<bos>", "<eos>", "hello", "world"],
            "tokenizer.bos_token_id": 1,
            "tokenizer.eos_token_id": 2
        }"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset;

        let total_size = data_offset as usize + 64; // Some padding
        let mut data = vec![0u8; total_size];

        // Header
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags = 0
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());

        // Metadata
        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

        data
    }

    #[test]
    fn test_load_embedded_tokenizer_valid() {
        let data = create_apr_model_with_embedded_tokenizer();
        let model = AprV2Model::from_bytes(data).expect("should load");

        let tokenizer = model
            .load_embedded_tokenizer()
            .expect("should have embedded tokenizer");

        assert_eq!(tokenizer.vocab_size(), 5);
        assert_eq!(tokenizer.bos_token_id, Some(1));
        assert_eq!(tokenizer.eos_token_id, Some(2));
        assert!(tokenizer.is_bos(1));
        assert!(tokenizer.is_eos(2));
    }

    #[test]
    fn test_load_embedded_tokenizer_decode() {
        let data = create_apr_model_with_embedded_tokenizer();
        let model = AprV2Model::from_bytes(data).expect("should load");

        let tokenizer = model
            .load_embedded_tokenizer()
            .expect("should have embedded tokenizer");

        let decoded = tokenizer.decode(&[3, 4]);
        assert_eq!(decoded, "helloworld");
    }

    // =========================================================================
    // Additional RoPE Metadata Tests
    // =========================================================================

    #[test]
    fn test_apr_metadata_rope_theta() {
        let meta = AprMetadata {
            rope_theta: Some(10000.0),
            ..Default::default()
        };
        assert_eq!(meta.rope_theta, Some(10000.0));
    }

    #[test]
    fn test_apr_metadata_rope_type() {
        // rope_type 0 = NORM (adjacent pairs)
        // rope_type 2 = NEOX (split halves) - used by Qwen2.5
        let meta = AprMetadata {
            rope_type: Some(2),
            ..Default::default()
        };
        assert_eq!(meta.rope_type, Some(2));
    }

    #[test]
    fn test_apr_metadata_rms_norm_eps() {
        let meta = AprMetadata {
            rms_norm_eps: Some(1e-6),
            ..Default::default()
        };
        assert!((meta.rms_norm_eps.unwrap() - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_apr_metadata_num_kv_heads() {
        // GQA models have fewer KV heads than query heads
        let meta = AprMetadata {
            num_heads: Some(32),
            num_kv_heads: Some(4), // GQA ratio 8:1
            ..Default::default()
        };
        assert_eq!(meta.num_heads, Some(32));
        assert_eq!(meta.num_kv_heads, Some(4));
    }

    #[test]
    fn test_apr_metadata_max_position_embeddings() {
        let meta = AprMetadata {
            max_position_embeddings: Some(4096),
            ..Default::default()
        };
        assert_eq!(meta.max_position_embeddings, Some(4096));
    }

    #[test]
    fn test_apr_metadata_intermediate_size() {
        let meta = AprMetadata {
            hidden_size: Some(2048),
            intermediate_size: Some(5632), // SwiGLU FFN size
            ..Default::default()
        };
        assert_eq!(meta.intermediate_size, Some(5632));
    }

    // =========================================================================
    // Additional dtype_to_ggml_qtype Tests
    // =========================================================================

    #[test]
    fn test_dtype_to_ggml_qtype_case_insensitive() {
        // Lowercase variants
        assert_eq!(dtype_to_ggml_qtype("q4_k"), Some(12));
        assert_eq!(dtype_to_ggml_qtype("q5_k"), Some(13));
        assert_eq!(dtype_to_ggml_qtype("q6_k"), Some(14));
        assert_eq!(dtype_to_ggml_qtype("q8_0"), Some(8));
        assert_eq!(dtype_to_ggml_qtype("q4_0"), Some(2));
        assert_eq!(dtype_to_ggml_qtype("q4_1"), Some(3));
        assert_eq!(dtype_to_ggml_qtype("q5_0"), Some(6));
    }

    #[test]
    fn test_is_quantized_dtype_comprehensive() {
        // Quantized types
        assert!(is_quantized_dtype("Q4_K"));
        assert!(is_quantized_dtype("Q5_K"));
        assert!(is_quantized_dtype("Q6_K"));
        assert!(is_quantized_dtype("Q8_0"));
        assert!(is_quantized_dtype("Q4_0"));
        assert!(is_quantized_dtype("Q4_1"));
        assert!(is_quantized_dtype("Q5_0"));

        // Non-quantized types
        assert!(!is_quantized_dtype("F32"));
        assert!(!is_quantized_dtype("F16"));
        assert!(!is_quantized_dtype("BF16"));
        assert!(!is_quantized_dtype("unknown"));
    }

    // =========================================================================
    // Additional TensorEntry Tests
    // =========================================================================

    #[test]
    fn test_tensor_entry_from_binary_all_dtypes() {
        // GH-191 FIX / GH-438: dtype bytes use GGML type IDs,
        // except 8/9 which are APR-native types (not GGML Q8_0/Q8_1)
        let dtypes = [
            (0u8, "F32"),
            (1, "F16"),
            (2, "Q4_0"),  // GGML type 2
            (3, "Q4_1"),  // GGML type 3
            (6, "Q5_0"),  // GGML type 6
            (7, "Q5_1"),  // GGML type 7
            (8, "q4"),    // GH-438: APR-native Q4 (legacy ID, was GGML Q8_0)
            (9, "q8"),    // GH-438: APR-native Q8 (legacy ID, was GGML Q8_1)
            (10, "Q2_K"), // GGML type 10
            (11, "Q3_K"), // GGML type 11
            (12, "Q4_K"), // GGML type 12
            (13, "Q5_K"), // GGML type 13
            (14, "Q6_K"), // GGML type 14
            (30, "BF16"), // GGML type 30
        ];

        for (dtype_byte, expected_dtype) in dtypes {
            let data = create_binary_tensor_entry("test", dtype_byte, &[10], 0, 40);
            let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
            assert_eq!(
                entry.dtype, expected_dtype,
                "dtype_byte {} should map to {}",
                dtype_byte, expected_dtype
            );
        }
    }
